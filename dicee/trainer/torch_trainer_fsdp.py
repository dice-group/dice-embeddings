import os
from typing import Iterable

import torch
import torch.distributed as dist
from dicee.abstracts import AbstractTrainer
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def make_iterable_verbose(iterable_object, verbose, desc="Default", position=None, leave=True) -> Iterable:
    if verbose:
        return tqdm(iterable_object, desc=desc, position=position, leave=leave)
    else:
        return iterable_object


class TorchFSDPTrainer(AbstractTrainer):
    """A single-node multi-GPU trainer based on FSDP."""

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = None
        self.raw_model = None
        self.optimizer = None
        self.cpu_sparse_embedding = None
        self.cpu_sparse_optimizer = None
        self.loss_func = None
        self.train_dataset_loader = None
        self.loss_history = []
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}["float16"]
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args
        self.on_fit_start(self, model)

        base_loader = kwargs['train_dataloaders']
        self.train_dataset_loader = DataLoader(
            base_loader.dataset,
            batch_size=self.attributes.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.attributes.num_core,
            persistent_workers=False,
            collate_fn=base_loader.dataset.collate_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(base_loader.dataset),
        )

        self.raw_model = model
        self.raw_model.to(self.device)
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            self.model = self.raw_model
            self._sync_replicated_parameters()
        else:
            self.model = FSDP(
                self.raw_model,
                device_id=self.device,
                use_orig_params=True,
                sync_module_states=True,
                param_init_fn=self._param_init_fn,
            )
        self.loss_func = model.loss
        self.optimizer = model.configure_optimizers(parameters=self.model.parameters())
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            self._init_cpu_sparse_optimizer()

        num_of_batches = len(self.train_dataset_loader)
        for epoch in (tqdm_bar := make_iterable_verbose(
            range(self.attributes.num_epochs),
            verbose=self.local_rank == self.global_rank == 0,
            position=0,
            leave=True,
        )):
            self.train_dataset_loader.sampler.set_epoch(epoch)
            epoch_loss = 0.0
            for i, z in enumerate(self.train_dataset_loader):
                source, targets = self.extract_input_outputs(z)
                batch_loss = self._run_batch(source, targets)
                epoch_loss += batch_loss
                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(
                            f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, "
                            f"loss_epoch={epoch_loss / i:.5f}"
                        )
                    else:
                        tqdm_bar.set_postfix_str(
                            f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}"
                        )

            avg_epoch_loss = epoch_loss / num_of_batches
            self.loss_history.append(avg_epoch_loss)

            if self.local_rank == self.global_rank == 0:
                self.raw_model.loss_history = list(self.loss_history)
                for c in self.callbacks:
                    c.on_train_epoch_end(self, self.raw_model)

        dist.barrier()
        trained_model = self._materialize_full_state_on_rank_zero()
        self.on_fit_end(self, trained_model)
        return trained_model

    def _param_init_fn(self, module: torch.nn.Module) -> None:
        init_fn = getattr(self.raw_model, "initialize_meta_parameters", None)
        if callable(init_fn):
            init_fn(module, self.device)

    def _run_batch(self, source: torch.LongTensor, targets: torch.FloatTensor) -> float:
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            self.optimizer.zero_grad(set_to_none=True)
            self.cpu_sparse_optimizer.zero_grad(set_to_none=True)
            output = self.model(source)
            loss = self.loss_func(output, targets)
            batch_loss = loss.item()
            loss.backward()
            self._sync_replicated_gradients()
            self.optimizer.step()
            self._step_cpu_sparse_optimizer()
            self.raw_model.local_entity_embeddings.weight.grad = None
            return batch_loss

        with self.ctx:
            output = self.model(source)
            loss = self.loss_func(output, targets)
            batch_loss = loss.item()
        self.scaler.scale(loss).backward()
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            self._sync_replicated_gradients()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return batch_loss

    def _init_cpu_sparse_optimizer(self) -> None:
        local_weight = self.raw_model.local_entity_embeddings.weight.detach().cpu()
        self.cpu_sparse_embedding = torch.nn.Embedding(
            local_weight.shape[0],
            local_weight.shape[1],
            sparse=True,
            device="cpu",
        )
        self.cpu_sparse_embedding.weight.data.copy_(local_weight)
        self.cpu_sparse_optimizer = torch.optim.SparseAdam(
            [self.cpu_sparse_embedding.weight],
            lr=self.raw_model.learning_rate,
        )

    def _step_cpu_sparse_optimizer(self) -> None:
        sparse_grad = self.raw_model.local_entity_embeddings.weight.grad
        if sparse_grad is None:
            return

        sparse_grad = sparse_grad.coalesce()
        cpu_grad = torch.sparse_coo_tensor(
            sparse_grad.indices().cpu(),
            sparse_grad.values().cpu(),
            sparse_grad.size(),
            device="cpu",
        ).coalesce()
        self.cpu_sparse_embedding.weight.grad = cpu_grad
        self.cpu_sparse_optimizer.step()

        updated_rows = cpu_grad.indices()[0].unique(sorted=True)
        updated_values = self.cpu_sparse_embedding.weight.data.index_select(0, updated_rows)
        self.raw_model.local_entity_embeddings.weight.data.index_copy_(
            0,
            updated_rows.to(self.device),
            updated_values.to(self.device),
        )
        self.cpu_sparse_embedding.weight.grad = None

    def extract_input_outputs(self, z: list):
        if len(z) == 2:
            x_batch, y_batch = z
            x_batch = x_batch.pin_memory().to(self.local_rank, non_blocking=True)
            y_batch = y_batch.pin_memory().to(self.local_rank, non_blocking=True)
            return x_batch, y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch = z
            x_batch = x_batch.pin_memory().to(self.local_rank, non_blocking=True)
            y_batch = y_batch.pin_memory().to(self.local_rank, non_blocking=True)
            y_idx_batch = y_idx_batch.pin_memory().to(self.local_rank, non_blocking=True)
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError('Unexpected batch shape..')

    def _materialize_full_state_on_rank_zero(self) -> torch.nn.Module:
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            return self._materialize_sharded_distmult_on_rank_zero()
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.model.state_dict()
        if self.local_rank == self.global_rank == 0:
            self.raw_model.load_state_dict(state_dict, strict=True)
            self.raw_model.loss_history = list(self.loss_history)
        return self.raw_model

    def _sync_replicated_parameters(self) -> None:
        for name, param in self.raw_model.named_parameters():
            if name != "local_entity_embeddings.weight":
                dist.broadcast(param.data, src=0)

    def _sync_replicated_gradients(self) -> None:
        world_size = dist.get_world_size()
        for name, param in self.raw_model.named_parameters():
            if name != "local_entity_embeddings.weight" and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

    def _materialize_sharded_distmult_on_rank_zero(self) -> torch.nn.Module:
        local_weight = self.raw_model.local_entity_embeddings.weight.detach()
        local_rows = torch.tensor([local_weight.shape[0]], device=self.device, dtype=torch.long)
        row_sizes = [torch.zeros_like(local_rows) for _ in range(dist.get_world_size())]
        dist.all_gather(row_sizes, local_rows)
        max_rows = max(int(x.item()) for x in row_sizes)
        padded = torch.zeros(max_rows, local_weight.shape[1], device=self.device, dtype=local_weight.dtype)
        padded[: local_weight.shape[0]] = local_weight
        gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, padded)
        if self.local_rank == self.global_rank == 0:
            full_entity_weight = torch.cat(
                [tensor[: int(size.item())].cpu() for tensor, size in zip(gathered, row_sizes)],
                dim=0,
            )[: self.raw_model.num_entities]
            full_entity_embeddings = torch.nn.Embedding(self.raw_model.num_entities, self.raw_model.embedding_dim)
            full_entity_embeddings.weight.data.copy_(full_entity_weight)
            self.raw_model.entity_embeddings = full_entity_embeddings
            self.raw_model.local_entity_embeddings = None
            self.raw_model.manual_sharded_entity_training = False
            self.raw_model.loss_history = list(self.loss_history)
        return self.raw_model
