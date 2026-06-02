import os
import weakref
from typing import Dict, Tuple, Type

import numpy as np
import torch
import torch.distributed as dist

from .base_model import BaseKGE

_FSDP_SHARDED_MODEL_CACHE: Dict[Type[BaseKGE], Type[BaseKGE]] = {}


class _CPUSparseRowAdam:
    """Adam for sparse embedding rows with optimizer state kept on CPU."""

    def __init__(self, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, pin_memory: bool = False):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.step_count = 0
        self.row_to_state: Dict[int, int] = {}
        self.exp_avg = None
        self.exp_avg_sq = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        return None

    def _empty_state(self, dim: int) -> torch.Tensor:
        tensor = torch.empty((0, dim), dtype=torch.float32, device="cpu")
        return tensor.pin_memory() if self.pin_memory else tensor

    def _zeros_state(self, rows: int, dim: int) -> torch.Tensor:
        tensor = torch.zeros((rows, dim), dtype=torch.float32, device="cpu")
        return tensor.pin_memory() if self.pin_memory else tensor

    def _state_positions(self, rows: torch.Tensor, dim: int) -> torch.Tensor:
        if self.exp_avg is None:
            self.exp_avg = self._empty_state(dim)
            self.exp_avg_sq = self._empty_state(dim)

        positions = torch.empty(rows.numel(), dtype=torch.long, device="cpu")
        num_new_rows = 0
        for i, row in enumerate(rows.tolist()):
            pos = self.row_to_state.get(row)
            if pos is None:
                pos = len(self.row_to_state)
                self.row_to_state[row] = pos
                num_new_rows += 1
            positions[i] = pos

        if num_new_rows:
            self.exp_avg = torch.cat((self.exp_avg, self._zeros_state(num_new_rows, dim)), dim=0)
            self.exp_avg_sq = torch.cat((self.exp_avg_sq, self._zeros_state(num_new_rows, dim)), dim=0)

        return positions

    def step_sparse_grad(
        self,
        weight: torch.Tensor,
        sparse_grad: torch.Tensor,
        device: torch.device,
        stream: torch.cuda.Stream = None,
    ) -> None:
        sparse_grad = sparse_grad.coalesce()
        if sparse_grad._nnz() == 0:
            return

        rows = sparse_grad.indices()[0].detach().cpu()
        grads = sparse_grad.values().detach().cpu().float()
        positions = self._state_positions(rows, grads.shape[1])
        self.step_count += 1

        exp_avg = self.exp_avg.index_select(0, positions)
        exp_avg_sq = self.exp_avg_sq.index_select(0, positions)
        exp_avg.mul_(self.beta1).add_(grads, alpha=1 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(grads, grads, value=1 - self.beta2)

        self.exp_avg.index_copy_(0, positions, exp_avg)
        self.exp_avg_sq.index_copy_(0, positions, exp_avg_sq)

        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count
        rows_on_device = rows.to(device, non_blocking=True)
        current_values = weight.data.index_select(0, rows_on_device).detach().cpu().float()
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
        updated_values = current_values.addcdiv(exp_avg, denom, value=-(self.lr / bias_correction1))
        updated_values = updated_values.to(device=device, dtype=weight.dtype, non_blocking=True)

        if stream is None:
            weight.data.index_copy_(0, rows_on_device, updated_values)
        else:
            with torch.cuda.stream(stream):
                weight.data.index_copy_(0, rows_on_device, updated_values)
            torch.cuda.current_stream().wait_stream(stream)


class _ShardedEntityEmbeddingProxy(torch.nn.Module):
    """Route embedding lookups through the owning model's distributed shard lookup."""

    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))

    def forward(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("The sharded entity embedding owner is no longer available.")
        return owner._entity_lookup(entity_ids)

    @property
    def weight(self):
        raise RuntimeError(
            "Manual FSDP entity sharding does not expose a full entity embedding weight. "
            "Use a sample-based scoring technique or materialize the model after training."
        )


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            ctx.world_size = dist.get_world_size()
        else:
            ctx.world_size = 1
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized():
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
            grad_output.div_(ctx.world_size)
        return grad_output


class FSDPShardedEntityModel(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.manual_sharded_entity_training = self.defer_large_embeddings
        self.local_entity_embeddings = None
        self.local_entity_start = 0
        self.local_entity_end = self.num_entities
        self.local_entity_count = self.num_entities
        self._batch_lookup_ids = None
        self._batch_lookup_embeddings = None
        self.fsdp_sharded_device = None
        self.fsdp_sharded_async_stream = None
        self.fsdp_use_cpu_sparse_optimizer = True
        self.fsdp_max_accumulated_sparse_grad_nnz = 1_000_000
        self.gpu_sparse_optimizer = None
        self.cpu_sparse_embedding = None
        self.cpu_sparse_optimizer = None
        self.pending_cpu_sparse_grad = None
        if self.manual_sharded_entity_training:
            rank, world_size = self._distributed_rank_world_size()
            shard_size = (self.num_entities + world_size - 1) // world_size
            self.local_entity_start = rank * shard_size
            self.local_entity_end = min(self.local_entity_start + shard_size, self.num_entities)
            self.local_entity_count = max(0, self.local_entity_end - self.local_entity_start)
            self.local_entity_embeddings = torch.nn.Embedding(
                self.local_entity_count,
                self.embedding_dim,
                sparse=True,
            )
            self.param_init(self.local_entity_embeddings.weight.data)
            self.entity_embeddings = _ShardedEntityEmbeddingProxy(self)

    @staticmethod
    def _distributed_rank_world_size() -> Tuple[int, int]:
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

    def configure_optimizers(self, parameters=None):
        if not self.manual_sharded_entity_training:
            return super().configure_optimizers(parameters=parameters)
        if parameters is not None:
            return super().configure_optimizers(parameters=parameters)
        dense_parameters = [
            param for name, param in self.named_parameters()
            if not self._is_sparse_training_parameter(name)
        ]
        return super().configure_optimizers(parameters=dense_parameters)

    def fsdp_ignored_modules(self):
        if not self.manual_sharded_entity_training:
            return []
        ignored_modules = [self.local_entity_embeddings]
        if self.cpu_sparse_embedding is not None:
            ignored_modules.append(self.cpu_sparse_embedding)
        return ignored_modules

    def fsdp_dense_optimizer_parameters(self, wrapped_model=None):
        module = wrapped_model if wrapped_model is not None else self
        return [
            param for name, param in module.named_parameters()
            if not self._is_sparse_training_parameter(name)
        ]

    @staticmethod
    def _is_sparse_training_parameter(name: str) -> bool:
        return (
            "local_entity_embeddings.weight" in name
            or "cpu_sparse_embedding.weight" in name
        )

    @staticmethod
    def fsdp_state_dict_excluded_prefixes() -> Tuple[str, ...]:
        return ("local_entity_embeddings.", "cpu_sparse_embedding.")

    def setup_fsdp_sharded_entity_training(
        self,
        device: torch.device,
        use_cpu_sparse_optimizer: bool = True,
        max_accumulated_sparse_grad_nnz: int = 1_000_000,
        async_stream: torch.cuda.Stream = None,
    ) -> None:
        """Prepare replicated parameters and sparse entity optimizer for FSDP training."""
        if not self.manual_sharded_entity_training:
            return

        self.fsdp_sharded_device = device
        self.fsdp_use_cpu_sparse_optimizer = use_cpu_sparse_optimizer
        self.fsdp_max_accumulated_sparse_grad_nnz = max_accumulated_sparse_grad_nnz
        self.fsdp_sharded_async_stream = async_stream
        self._init_sparse_optimizer()

    def zero_sparse_optimizer_grad(self) -> None:
        if not self.manual_sharded_entity_training:
            return
        if self.fsdp_use_cpu_sparse_optimizer:
            if self.cpu_sparse_optimizer is not None:
                self.cpu_sparse_optimizer.zero_grad(set_to_none=True)
        elif self.gpu_sparse_optimizer is not None:
            self.gpu_sparse_optimizer.zero_grad(set_to_none=True)

    def step_sparse_optimizer(self, batch_idx: int, sparse_step_interval: int) -> None:
        if not self.manual_sharded_entity_training:
            return

        if self.fsdp_use_cpu_sparse_optimizer:
            self._accumulate_cpu_sparse_grad()
            if batch_idx % sparse_step_interval == 0:
                self.flush_sparse_optimizer()
        elif self.gpu_sparse_optimizer is not None:
            self.gpu_sparse_optimizer.step()

        self.local_entity_embeddings.weight.grad = None

    def flush_sparse_optimizer(self) -> None:
        if not self.manual_sharded_entity_training or not self.fsdp_use_cpu_sparse_optimizer:
            return
        if self.pending_cpu_sparse_grad is None:
            return

        self.cpu_sparse_optimizer.step_sparse_grad(
            self.local_entity_embeddings.weight,
            self.pending_cpu_sparse_grad,
            self.fsdp_sharded_device,
            self.fsdp_sharded_async_stream,
        )
        self.pending_cpu_sparse_grad = None

    def materialize_sharded_entity_model_on_rank_zero(self, loss_history=None) -> torch.nn.Module:
        """Gather entity shards and rebuild the full embedding table on rank 0."""
        if self.fsdp_sharded_async_stream is not None:
            self.fsdp_sharded_async_stream.synchronize()

        local_weight = self.local_entity_embeddings.weight.detach().contiguous()
        is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        full_entity_embeddings = None
        copied_rows = 0
        if is_rank_zero:
            if local_weight.shape[1] != self.embedding_dim:
                raise RuntimeError("Local entity shard has an unexpected embedding dimension.")
            full_entity_embeddings = torch.nn.Embedding(
                self.num_entities,
                self.embedding_dim,
                device="cpu",
                dtype=local_weight.dtype,
            )

        if dist.is_initialized():
            copied_rows = self._materialize_entity_shards_with_all_gather(
                local_weight=local_weight,
                full_entity_embeddings=full_entity_embeddings,
            )
        else:
            local_weight = local_weight.cpu()
            shard_rows = min(local_weight.shape[0], self.num_entities)
            full_entity_embeddings.weight.data[:shard_rows].copy_(local_weight[:shard_rows])
            copied_rows = shard_rows

        if is_rank_zero:
            if copied_rows != self.num_entities:
                raise RuntimeError(
                    f"Materialized {copied_rows} entity rows, expected {self.num_entities}."
                )

            materialized_args = dict(self.args)
            materialized_args["fsdp_sharded_entity"] = False
            materialized_model = self.__class__(materialized_args)
            materialized_model.entity_embeddings = full_entity_embeddings
            materialized_model.manual_sharded_entity_training = False
            if loss_history is not None:
                materialized_model.loss_history = list(loss_history)
            return materialized_model

        return self

    def _materialize_entity_shards_with_all_gather(
        self,
        local_weight: torch.Tensor,
        full_entity_embeddings: torch.nn.Embedding,
        chunk_rows: int = 65_536,
    ) -> int:
        """Gather local entity shards with the existing process group.

        FSDP training initializes NCCL for CUDA tensors. Reusing that group avoids creating a
        separate Gloo group and keeps every rank in the same collective calls during finalization.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        shard_size = (self.num_entities + world_size - 1) // world_size
        copied_rows = 0

        local_start = rank * shard_size
        local_end = min(local_start + shard_size, self.num_entities)
        expected_local_rows = max(0, local_end - local_start)
        if local_weight.shape != (expected_local_rows, self.embedding_dim):
            raise RuntimeError(
                f"Local entity shard has shape {tuple(local_weight.shape)}, "
                f"expected {(expected_local_rows, self.embedding_dim)}."
            )

        chunk_rows = max(1, min(chunk_rows, shard_size))
        for chunk_start in range(0, shard_size, chunk_rows):
            current_chunk_rows = min(chunk_rows, shard_size - chunk_start)
            chunk = torch.zeros(
                current_chunk_rows,
                self.embedding_dim,
                dtype=local_weight.dtype,
                device=local_weight.device,
            )

            valid_local_end = min(chunk_start + current_chunk_rows, expected_local_rows)
            valid_local_rows = max(0, valid_local_end - chunk_start)
            if valid_local_rows > 0:
                chunk[:valid_local_rows].copy_(local_weight[chunk_start:valid_local_end])

            gathered_chunks = [torch.empty_like(chunk) for _ in range(world_size)]
            dist.all_gather(gathered_chunks, chunk)

            if rank != 0:
                continue

            for shard_rank, shard_chunk in enumerate(gathered_chunks):
                shard_global_start = shard_rank * shard_size
                shard_global_end = min(shard_global_start + shard_size, self.num_entities)
                shard_rows = max(0, shard_global_end - shard_global_start)
                valid_end = min(chunk_start + current_chunk_rows, shard_rows)
                valid_rows = max(0, valid_end - chunk_start)
                if valid_rows == 0:
                    continue
                destination_start = shard_global_start + chunk_start
                destination_end = destination_start + valid_rows
                full_entity_embeddings.weight.data[destination_start:destination_end].copy_(
                    shard_chunk[:valid_rows].cpu()
                )
                copied_rows += valid_rows

        return copied_rows

    def _init_sparse_optimizer(self) -> None:
        if not self.fsdp_use_cpu_sparse_optimizer:
            self.gpu_sparse_optimizer = torch.optim.SparseAdam(
                [self.local_entity_embeddings.weight],
                lr=self.learning_rate,
            )
            return

        self.cpu_sparse_optimizer = _CPUSparseRowAdam(
            lr=self.learning_rate,
            pin_memory=False,
        )

    def _accumulate_cpu_sparse_grad(self) -> None:
        sparse_grad = self.local_entity_embeddings.weight.grad
        if sparse_grad is None:
            return

        sparse_grad = sparse_grad.coalesce()
        if sparse_grad._nnz() == 0:
            return

        if (
            self.pending_cpu_sparse_grad is not None
            and self.pending_cpu_sparse_grad._nnz() > self.fsdp_max_accumulated_sparse_grad_nnz
        ):
            self.flush_sparse_optimizer()

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

        if self.pending_cpu_sparse_grad.size() != cpu_grad.size():
            raise RuntimeError("Cannot accumulate sparse gradients with different shapes.")
        self.pending_cpu_sparse_grad = torch.sparse_coo_tensor(
            torch.cat((self.pending_cpu_sparse_grad.indices(), cpu_grad.indices()), dim=1),
            torch.cat((self.pending_cpu_sparse_grad.values(), cpu_grad.values()), dim=0),
            cpu_grad.size(),
            device="cpu",
            check_invariants=False,
        ).coalesce()

    def _entity_lookup(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        if not self.manual_sharded_entity_training:
            return self.entity_embeddings(entity_ids)

        if self._batch_lookup_ids is not None and self._batch_lookup_embeddings is not None:
            original_shape = entity_ids.shape
            flat_entity_ids = entity_ids.contiguous().reshape(-1)
            positions = torch.searchsorted(self._batch_lookup_ids, flat_entity_ids)
            positions = positions.clamp(max=self._batch_lookup_ids.numel() - 1)
            if torch.equal(self._batch_lookup_ids.index_select(0, positions), flat_entity_ids):
                return self._batch_lookup_embeddings.index_select(0, positions).reshape(
                    *original_shape,
                    self.embedding_dim,
                )
            raise RuntimeError("Sharded entity lookup cache miss during a cached lookup.")

        return self._distributed_entity_lookup(entity_ids)

    def _distributed_entity_lookup(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        flat_entity_ids = entity_ids.contiguous().reshape(-1)
        if flat_entity_ids.numel() == 0:
            return torch.empty(
                *entity_ids.shape,
                self.embedding_dim,
                device=entity_ids.device,
                dtype=self.local_entity_embeddings.weight.dtype,
            )

        unique_entity_ids = self._global_unique_entity_ids(flat_entity_ids)
        outputs = torch.zeros(
            unique_entity_ids.shape[0],
            self.embedding_dim,
            device=unique_entity_ids.device,
            dtype=self.local_entity_embeddings.weight.dtype,
        )
        mask = (unique_entity_ids >= self.local_entity_start) & (unique_entity_ids < self.local_entity_end)
        if mask.any():
            local_ids = unique_entity_ids[mask] - self.local_entity_start
            outputs[mask] = self.local_entity_embeddings(local_ids)
        outputs = _AllReduceSum.apply(outputs)
        positions = torch.searchsorted(unique_entity_ids, flat_entity_ids)
        if not torch.equal(unique_entity_ids.index_select(0, positions), flat_entity_ids):
            raise RuntimeError("Distributed entity lookup failed to map all requested entity ids.")
        return outputs.index_select(0, positions).reshape(*entity_ids.shape, self.embedding_dim)

    @staticmethod
    def _global_unique_entity_ids(entity_ids: torch.LongTensor) -> torch.LongTensor:
        local_unique = torch.unique(entity_ids, sorted=True)
        if not dist.is_initialized():
            return local_unique

        local_count = torch.tensor([local_unique.numel()], device=entity_ids.device, dtype=torch.long)
        counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(counts, local_count)
        counts = torch.cat(counts)
        max_count = int(counts.max().item())
        if max_count == 0:
            return local_unique

        padded = torch.empty(max_count, device=entity_ids.device, dtype=torch.long)
        if local_unique.numel() > 0:
            padded[:local_unique.numel()] = local_unique
        if local_unique.numel() < max_count:
            padded[local_unique.numel():] = 0

        gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, padded)
        requested_ids = [
            rank_ids[:int(rank_count.item())]
            for rank_ids, rank_count in zip(gathered, counts)
            if int(rank_count.item()) > 0
        ]
        if not requested_ids:
            return local_unique
        return torch.unique(torch.cat(requested_ids), sorted=True)

    def _prime_batch_lookup_cache(self, entity_ids: torch.LongTensor) -> None:
        unique_entity_ids = torch.unique(entity_ids, sorted=True)
        self._batch_lookup_ids = unique_entity_ids
        self._batch_lookup_embeddings = self._distributed_entity_lookup(unique_entity_ids)

    def _clear_batch_lookup_cache(self) -> None:
        self._batch_lookup_ids = None
        self._batch_lookup_embeddings = None

    def forward(self, x, y_idx: torch.LongTensor = None) -> torch.FloatTensor:
        if not self.manual_sharded_entity_training or not isinstance(x, tuple):
            if y_idx is None:
                return super().forward(x)
            return super().forward(x, y_idx)

        source, target_entity_idx = x
        self._prime_batch_lookup_cache(torch.cat((source[:, 0], target_entity_idx.reshape(-1))))
        try:
            if y_idx is None:
                return super().forward((source, target_entity_idx))
            return super().forward((source, target_entity_idx), y_idx)
        finally:
            self._clear_batch_lookup_cache()

    def get_triple_representation(self, idx_hrt):
        if not self.manual_sharded_entity_training:
            return super().get_triple_representation(idx_hrt)
        idx_head_entity, idx_relation, idx_tail_entity = idx_hrt[:, 0], idx_hrt[:, 1], idx_hrt[:, 2]
        self._prime_batch_lookup_cache(torch.cat((idx_head_entity, idx_tail_entity)))
        try:
            head_ent_emb = self.normalize_head_entity_embeddings(
                self.input_dp_ent_real(self._entity_lookup(idx_head_entity))
            )
            rel_ent_emb = self.normalize_relation_embeddings(
                self.input_dp_rel_real(self.relation_embeddings(idx_relation))
            )
            tail_ent_emb = self.normalize_tail_entity_embeddings(self._entity_lookup(idx_tail_entity))
        finally:
            self._clear_batch_lookup_cache()
        return head_ent_emb, rel_ent_emb, tail_ent_emb

    def get_head_relation_representation(self, indexed_triple):
        if not self.manual_sharded_entity_training:
            return super().get_head_relation_representation(indexed_triple)
        idx_head_entity, idx_relation = indexed_triple[:, 0], indexed_triple[:, 1]
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self._entity_lookup(idx_head_entity))
        )
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return head_ent_emb, rel_ent_emb

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.manual_sharded_entity_training:
            return super().get_embeddings()
        raise RuntimeError("Full entity embeddings are materialized by the trainer on rank 0 after training.")


_FSDPShardedEntityModel = FSDPShardedEntityModel


def create_fsdp_sharded_model_class(model_class: Type[BaseKGE]) -> Type[BaseKGE]:
    """Create a manual entity-sharded FSDP variant for a Dice model class."""
    if issubclass(model_class, FSDPShardedEntityModel):
        return model_class
    if model_class not in _FSDP_SHARDED_MODEL_CACHE:
        _FSDP_SHARDED_MODEL_CACHE[model_class] = type(
            f"FSDPSharded{model_class.__name__}",
            (FSDPShardedEntityModel, model_class),
            {
                "__module__": model_class.__module__,
                "__doc__": f"Manual entity-sharded FSDP variant of {model_class.__name__}.",
            },
        )
    return _FSDP_SHARDED_MODEL_CACHE[model_class]
