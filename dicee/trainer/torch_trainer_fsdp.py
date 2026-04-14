import os
from typing import Iterable

import torch
import torch.distributed as dist
from dicee.abstracts import AbstractTrainer
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def make_iterable_verbose(iterable_object, verbose, desc="Default", position=None, leave=True) -> Iterable:
    if verbose:
        return tqdm(iterable_object, desc=desc, position=position, leave=leave)
    else:
        return iterable_object


class TorchFSDPTrainer(AbstractTrainer):
    """
    Optimized single-node multi-GPU trainer based on FSDP.
    
    Optimizations include:
    - Persistent workers and prefetching for data loading
    - Async CUDA streams for CPU-GPU transfers
    - Mixed precision training on all code paths
    - Enhanced FSDP configuration with sharding strategies
    - Gradient accumulation limits to prevent OOM
    - Optional torch.compile support
    - Reduced profiling overhead
    """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = None
        self.raw_model = None
        self.optimizer = None
        self.gpu_sparse_optimizer = None
        self.cpu_sparse_embedding = None
        self.cpu_sparse_optimizer = None
        self.pending_cpu_sparse_grad = None
        self.loss_func = None
        self.train_dataset_loader = None
        self.loss_history = []
        
        # Sparse optimizer configuration
        self.sparse_step_interval = max(1, int(getattr(args, "fsdp_sparse_step_interval", 4)))
        self.use_cpu_sparse_optimizer = getattr(args, "fsdp_sparse_optimizer_device", "cpu") == "cpu"
        self.max_accumulated_sparse_grad_nnz = getattr(args, "max_accumulated_sparse_grad_nnz", 1_000_000)
        
        # Mixed precision configuration
        ptdtype_str = getattr(args, "precision", "bfloat16")
        ptdtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
        ptdtype = ptdtype_map.get(ptdtype_str, torch.bfloat16)
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(ptdtype == torch.float16))
        
        # FSDP configuration
        self.use_compile = getattr(args, "use_compile", False)
        self.sharding_strategy = getattr(args, "sharding_strategy", "FULL_SHARD")
        self.gradient_clip_val = getattr(args, "gradient_clip_val", None)
        
        # DataLoader configuration
        self.num_workers = getattr(args, "num_workers", self.attributes.num_core)
        self.prefetch_factor = getattr(args, "prefetch_factor", 4)
        
        # CUDA stream for async operations
        self.async_stream = torch.cuda.Stream()

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args
        self.on_fit_start(self, model)

        # Setup optimized DataLoader
        base_loader = kwargs['train_dataloaders']
        self.train_dataset_loader = DataLoader(
            base_loader.dataset,
            batch_size=self.attributes.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,  # Only if workers > 0
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=base_loader.dataset.collate_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(
                base_loader.dataset,
                shuffle=True,
                drop_last=False,
            ),
        )

        self.raw_model = model
        self.raw_model.to(self.device)
        
        # Setup model with FSDP or manual sharding
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            self.model = self.raw_model
            self._sync_replicated_parameters()
        else:
            self.model = self._wrap_model_with_fsdp()
        
        self.loss_func = model.loss
        self.optimizer = model.configure_optimizers(parameters=self.model.parameters())
        
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            self._init_sparse_optimizer()
        
        # Optional: Compile model for additional speedup (PyTorch 2.0+)
        if self.use_compile and hasattr(torch, 'compile'):
            if self.local_rank == 0:
                print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode='reduce-overhead')

        num_of_batches = len(self.train_dataset_loader)
        
        # Training loop
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
                batch_loss = self._run_batch(source, targets, batch_idx=i + 1)
                epoch_loss += batch_loss
                
                # Update progress bar
                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(
                            f"batch={i}/{num_of_batches}, loss_step={batch_loss:.5f}, "
                            f"loss_epoch={epoch_loss / i:.5f}"
                        )
                    else:
                        tqdm_bar.set_postfix_str(
                            f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}"
                        )

            # Flush any remaining sparse gradients at epoch end
            if getattr(self.raw_model, "manual_sharded_entity_training", False) and self.use_cpu_sparse_optimizer:
                self._flush_cpu_sparse_optimizer()

            avg_epoch_loss = epoch_loss / num_of_batches
            self.loss_history.append(avg_epoch_loss)

            # Callbacks on rank 0
            if self.local_rank == self.global_rank == 0:
                self.raw_model.loss_history = list(self.loss_history)
                for c in self.callbacks:
                    c.on_train_epoch_end(self, self.raw_model)

        dist.barrier()
        trained_model = self._materialize_full_state_on_rank_zero()
        self.on_fit_end(self, trained_model)
        return trained_model

    def _wrap_model_with_fsdp(self) -> FSDP:
        """Wrap model with FSDP using optimized configuration."""
        # Mixed precision policy
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        
        # Sharding strategy
        sharding_strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        strategy = sharding_strategy_map.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        return FSDP(
            self.raw_model,
            device_id=self.device,
            use_orig_params=True,
            sync_module_states=True,
            param_init_fn=self._param_init_fn,
            sharding_strategy=strategy,
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch for better performance
            limit_all_gathers=True,  # Reduce memory spikes
            forward_prefetch=True,  # Prefetch forward passes
        )

    def _param_init_fn(self, module: torch.nn.Module) -> None:
        """Initialize meta parameters if the model provides an initialization function."""
        init_fn = getattr(self.raw_model, "initialize_meta_parameters", None)
        if callable(init_fn):
            init_fn(module, self.device)

    def _run_batch(self, source: torch.LongTensor, targets: torch.FloatTensor, batch_idx: int) -> float:
        """Run a single training batch with optimized mixed precision and gradient handling."""
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            return self._run_batch_manual_sharding(source, targets, batch_idx)
        else:
            return self._run_batch_fsdp(source, targets)

    def _run_batch_manual_sharding(
        self, 
        source: torch.LongTensor, 
        targets: torch.FloatTensor, 
        batch_idx: int
    ) -> float:
        """Run batch with manual sharding and sparse optimizer."""
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.use_cpu_sparse_optimizer:
            if self.cpu_sparse_optimizer is not None:
                self.cpu_sparse_optimizer.zero_grad(set_to_none=True)
        elif self.gpu_sparse_optimizer is not None:
            self.gpu_sparse_optimizer.zero_grad(set_to_none=True)

        with self.ctx:
            output = self.model(source)
            loss = self.loss_func(output, targets)

        batch_loss = loss.item()

        self.scaler.scale(loss).backward()

        self._sync_replicated_gradients()

        self.scaler.unscale_(self.optimizer)

        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in self.raw_model.named_parameters() 
                 if n != "local_entity_embeddings.weight"],
                self.gradient_clip_val
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_cpu_sparse_optimizer:
            self._accumulate_cpu_sparse_grad()
            if batch_idx % self.sparse_step_interval == 0:
                self._flush_cpu_sparse_optimizer()
        else:
            self.gpu_sparse_optimizer.step()

        self.raw_model.local_entity_embeddings.weight.grad = None
        
        return batch_loss

    def _run_batch_fsdp(self, source: torch.LongTensor, targets: torch.FloatTensor) -> float:
        """Run batch with standard FSDP."""
        with self.ctx:
            output = self.model(source)
            loss = self.loss_func(output, targets)
            batch_loss = loss.item()
        
        self.scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)
        
        # Optional gradient clipping
        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        return batch_loss

    def _init_sparse_optimizer(self) -> None:
        """Initialize sparse optimizer on CPU or GPU."""
        if not self.use_cpu_sparse_optimizer:
            self.gpu_sparse_optimizer = torch.optim.SparseAdam(
                [self.raw_model.local_entity_embeddings.weight],
                lr=self.raw_model.learning_rate,
            )
            return

        # CPU sparse optimizer with pinned memory for faster transfers
        local_weight = self.raw_model.local_entity_embeddings.weight.detach().cpu()
        
        self.cpu_sparse_embedding = torch.nn.Embedding(
            local_weight.shape[0],
            local_weight.shape[1],
            sparse=True,
            device="cpu",
        )
        
        # Pin memory for faster CPU-GPU transfers
        self.cpu_sparse_embedding.weight.data = local_weight.pin_memory()
        
        self.cpu_sparse_optimizer = torch.optim.SparseAdam(
            [self.cpu_sparse_embedding.weight],
            lr=self.raw_model.learning_rate,
        )

    def _accumulate_cpu_sparse_grad(self) -> None:
        """Accumulate sparse gradients on CPU with overflow protection."""
        sparse_grad = self.raw_model.local_entity_embeddings.weight.grad
        if sparse_grad is None:
            return

        sparse_grad = sparse_grad.coalesce()
        if sparse_grad._nnz() == 0:
            return
        
        # Prevent unbounded accumulation - flush if too many non-zeros
        if self.pending_cpu_sparse_grad is not None and \
           self.pending_cpu_sparse_grad._nnz() > self.max_accumulated_sparse_grad_nnz:
            self._flush_cpu_sparse_optimizer()
        
        # Transfer to CPU
        cpu_grad = torch.sparse_coo_tensor(
            sparse_grad.indices().cpu(),
            sparse_grad.values().cpu(),
            sparse_grad.size(),
            device="cpu",
            check_invariants=False,
        ).coalesce()
        
        if self.pending_cpu_sparse_grad is None:
            self.pending_cpu_sparse_grad = cpu_grad
            return

        # Accumulate gradients
        self.pending_cpu_sparse_grad = torch.sparse_coo_tensor(
            torch.cat((self.pending_cpu_sparse_grad.indices(), cpu_grad.indices()), dim=1),
            torch.cat((self.pending_cpu_sparse_grad.values(), cpu_grad.values()), dim=0),
            cpu_grad.size(),
            device="cpu",
            check_invariants=False,
        ).coalesce()

    def _flush_cpu_sparse_optimizer(self) -> None:
        """Flush accumulated sparse gradients from CPU to GPU asynchronously."""
        if self.pending_cpu_sparse_grad is None:
            return

        self.cpu_sparse_optimizer.zero_grad(set_to_none=True)
        self.cpu_sparse_embedding.weight.grad = self.pending_cpu_sparse_grad
        self.cpu_sparse_optimizer.step()

        updated_rows = self.pending_cpu_sparse_grad.indices()[0].unique(sorted=True)
        updated_values = self.cpu_sparse_embedding.weight.data.index_select(0, updated_rows)
        
        # Use async stream for non-blocking transfer
        with torch.cuda.stream(self.async_stream):
            self.raw_model.local_entity_embeddings.weight.data.index_copy_(
                0,
                updated_rows.to(self.device, non_blocking=True),
                updated_values.to(self.device, non_blocking=True),
            )
        
        # Clear accumulated gradients
        self.cpu_sparse_embedding.weight.grad = None
        self.pending_cpu_sparse_grad = None

    def extract_input_outputs(self, z: list):
        """Extract inputs and outputs from batch, avoiding redundant pinning."""
        # DataLoader already pins memory, so we skip redundant pin_memory() calls
        if len(z) == 2:
            x_batch, y_batch = z
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            return x_batch, y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch = z
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            y_idx_batch = y_idx_batch.to(self.device, non_blocking=True)
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError('Unexpected batch shape..')

    def _materialize_full_state_on_rank_zero(self) -> torch.nn.Module:
        """Materialize full model state on rank 0 for checkpointing."""
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
        """Synchronize replicated parameters across all ranks."""
        for name, param in self.raw_model.named_parameters():
            if name != "local_entity_embeddings.weight":
                dist.broadcast(param.data, src=0)

    def _sync_replicated_gradients(self) -> None:
        """Synchronize and average replicated gradients across all ranks."""
        world_size = dist.get_world_size()
        for name, param in self.raw_model.named_parameters():
            if name != "local_entity_embeddings.weight" and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

    def _materialize_sharded_distmult_on_rank_zero(self) -> torch.nn.Module:
        """Gather sharded entity embeddings to rank 0 using CPU objects."""
        local_weight = self.raw_model.local_entity_embeddings.weight.detach().cpu()
        gathered = [None for _ in range(dist.get_world_size())] if self.local_rank == self.global_rank == 0 else None
        dist.gather_object(local_weight, object_gather_list=gathered, dst=self.global_rank)

        if self.local_rank == self.global_rank == 0:
            full_entity_weight = torch.cat(
                gathered,
                dim=0,
            )[: self.raw_model.num_entities]
            
            full_entity_embeddings = torch.nn.Embedding(
                self.raw_model.num_entities, 
                self.raw_model.embedding_dim
            )
            full_entity_embeddings.weight.data.copy_(full_entity_weight)
            
            self.raw_model.entity_embeddings = full_entity_embeddings
            self.raw_model.local_entity_embeddings = None
            self.raw_model.manual_sharded_entity_training = False
            self.raw_model.loss_history = list(self.loss_history)
        
        return self.raw_model
