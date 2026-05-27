import os
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from .base_model import BaseKGE


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
            if name != "local_entity_embeddings.weight"
        ]
        return super().configure_optimizers(parameters=dense_parameters)

    def fsdp_ignored_modules(self):
        if not self.manual_sharded_entity_training:
            return []
        return [self.local_entity_embeddings]

    def fsdp_dense_optimizer_parameters(self, wrapped_model=None):
        module = wrapped_model if wrapped_model is not None else self
        return [
            param for name, param in module.named_parameters()
            if "local_entity_embeddings.weight" not in name
        ]

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

        self.cpu_sparse_optimizer.zero_grad(set_to_none=True)
        self.cpu_sparse_embedding.weight.grad = self.pending_cpu_sparse_grad
        self.cpu_sparse_optimizer.step()

        updated_rows = self.pending_cpu_sparse_grad.indices()[0].unique(sorted=True)
        updated_values = self.cpu_sparse_embedding.weight.data.index_select(0, updated_rows)

        stream = self.fsdp_sharded_async_stream
        if stream is None:
            self.local_entity_embeddings.weight.data.index_copy_(
                0,
                updated_rows.to(self.fsdp_sharded_device, non_blocking=True),
                updated_values.to(self.fsdp_sharded_device, non_blocking=True),
            )
        else:
            with torch.cuda.stream(stream):
                self.local_entity_embeddings.weight.data.index_copy_(
                    0,
                    updated_rows.to(self.fsdp_sharded_device, non_blocking=True),
                    updated_values.to(self.fsdp_sharded_device, non_blocking=True),
                )
            torch.cuda.current_stream().wait_stream(stream)

        self.cpu_sparse_embedding.weight.grad = None
        self.pending_cpu_sparse_grad = None

    def materialize_sharded_entity_model_on_rank_zero(self, loss_history=None) -> torch.nn.Module:
        """Gather entity shards and rebuild the full embedding table on rank 0."""
        if self.fsdp_sharded_async_stream is not None:
            self.fsdp_sharded_async_stream.synchronize()

        local_weight = self.local_entity_embeddings.weight.detach().cpu()
        is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        gathered = [None for _ in range(world_size)] if is_rank_zero else None
        if dist.is_initialized():
            dist.gather_object(local_weight, object_gather_list=gathered, dst=0)
        else:
            gathered = [local_weight]

        if is_rank_zero:
            expected_dtype = gathered[0].dtype
            expected_dim = gathered[0].shape[1]
            if any(weight.dtype != expected_dtype or weight.shape[1] != expected_dim for weight in gathered):
                raise RuntimeError("Gathered entity shards have inconsistent dtype or embedding dimension.")

            full_entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
            offset = 0
            for shard_weight in gathered:
                shard_rows = min(shard_weight.shape[0], self.num_entities - offset)
                if shard_rows <= 0:
                    break
                full_entity_embeddings.weight.data[offset: offset + shard_rows].copy_(shard_weight[:shard_rows])
                offset += shard_rows

            self.entity_embeddings = full_entity_embeddings
            self.local_entity_embeddings = None
            self.manual_sharded_entity_training = False
            if loss_history is not None:
                self.loss_history = list(loss_history)

        return self

    def _init_sparse_optimizer(self) -> None:
        if not self.fsdp_use_cpu_sparse_optimizer:
            self.gpu_sparse_optimizer = torch.optim.SparseAdam(
                [self.local_entity_embeddings.weight],
                lr=self.learning_rate,
            )
            return

        local_weight = self.local_entity_embeddings.weight.detach().cpu()
        if torch.cuda.is_available():
            local_weight = local_weight.pin_memory()
        self.cpu_sparse_embedding = torch.nn.Embedding(
            local_weight.shape[0],
            local_weight.shape[1],
            sparse=True,
            device="cpu",
            _weight=local_weight,
        )
        self.cpu_sparse_optimizer = torch.optim.SparseAdam(
            [self.cpu_sparse_embedding.weight],
            lr=self.learning_rate,
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
            entity_ids = entity_ids.contiguous()
            positions = torch.searchsorted(self._batch_lookup_ids, entity_ids)
            positions = positions.clamp(max=self._batch_lookup_ids.numel() - 1)
            if torch.equal(self._batch_lookup_ids.index_select(0, positions), entity_ids):
                return self._batch_lookup_embeddings.index_select(0, positions)
            raise RuntimeError("Sharded entity lookup cache miss during a cached lookup.")

        return self._distributed_entity_lookup(entity_ids)

    def _distributed_entity_lookup(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        unique_entity_ids, inverse_indices = torch.unique(entity_ids, sorted=False, return_inverse=True)
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
        return outputs.index_select(0, inverse_indices)

    def _prime_batch_lookup_cache(self, entity_ids: torch.LongTensor) -> None:
        unique_entity_ids = torch.unique(entity_ids, sorted=True)
        self._batch_lookup_ids = unique_entity_ids
        self._batch_lookup_embeddings = self._distributed_entity_lookup(unique_entity_ids)

    def _clear_batch_lookup_cache(self) -> None:
        self._batch_lookup_ids = None
        self._batch_lookup_embeddings = None

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


class FSDPDistMult(FSDPShardedEntityModel):
    def __init__(self, args):
        super().__init__(args)
        self.name = "DistMult"

    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor):
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)), emb_E.transpose(1, 0))

    def forward_k_vs_all(self, x: torch.LongTensor):
        if self.manual_sharded_entity_training:
            raise NotImplementedError("Sharded DistMult currently supports only NegSample training.")
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel, emb_E=self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        if self.manual_sharded_entity_training:
            self._prime_batch_lookup_cache(torch.cat((x[:, 0], target_entity_idx.reshape(-1))))
            try:
                t = self._entity_lookup(target_entity_idx.reshape(-1)).reshape(target_entity_idx.shape[0], target_entity_idx.shape[1], -1)
                emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
            finally:
                self._clear_batch_lookup_cache()
            hr = torch.einsum("bd, bd -> bd", emb_head_real, emb_rel_real)
            return torch.einsum("bd, bkd -> bk", hr, t)
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        hr = torch.einsum("bd, bd -> bd", emb_head_real, emb_rel_real)
        t = self.entity_embeddings(target_entity_idx)
        return torch.einsum("bd, bkd -> bk", hr, t)

    def score(self, h, r, t):
        return (self.hidden_dropout(self.hidden_normalizer(h * r)) * t).sum(dim=1)


class FSDPComplEx(FSDPShardedEntityModel):
    def __init__(self, args):
        super().__init__(args)
        self.name = "ComplEx"

    @staticmethod
    def _split_complex(tensor: torch.FloatTensor):
        if tensor.shape[-1] % 2 != 0:
            raise ValueError("ComplEx requires an even embedding dimension.")
        return torch.chunk(tensor, 2, dim=-1)

    @staticmethod
    def score(head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor, tail_ent_emb: torch.FloatTensor):
        emb_head_real, emb_head_imag = FSDPComplEx._split_complex(head_ent_emb)
        emb_rel_real, emb_rel_imag = FSDPComplEx._split_complex(rel_ent_emb)
        emb_tail_real, emb_tail_imag = FSDPComplEx._split_complex(tail_ent_emb)
        real_real_real = (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    @staticmethod
    def k_vs_all_score(emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor):
        emb_head_real, emb_head_imag = FSDPComplEx._split_complex(emb_h)
        emb_rel_real, emb_rel_imag = FSDPComplEx._split_complex(emb_r)
        emb_tail_real, emb_tail_imag = FSDPComplEx._split_complex(emb_E)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        real_real_real = torch.mm(emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        if self.manual_sharded_entity_training:
            raise NotImplementedError("Sharded ComplEx currently supports only NegSample training.")
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        return self.k_vs_all_score(head_ent_emb, rel_ent_emb, self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        if self.manual_sharded_entity_training:
            self._prime_batch_lookup_cache(torch.cat((x[:, 0], target_entity_idx.reshape(-1))))
            try:
                emb_t = self._entity_lookup(target_entity_idx.reshape(-1)).reshape(target_entity_idx.shape[0], target_entity_idx.shape[1], -1)
                emb_h, emb_r = self.get_head_relation_representation(x)
            finally:
                self._clear_batch_lookup_cache()
        else:
            emb_t = self.entity_embeddings(target_entity_idx)
            emb_h, emb_r = self.get_head_relation_representation(x)
        emb_head_real, emb_head_imag = self._split_complex(emb_h)
        emb_rel_real, emb_rel_imag = self._split_complex(emb_r)
        emb_tail_real, emb_tail_imag = self._split_complex(emb_t)
        real_real_real = torch.einsum("bd, bkd -> bk", emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.einsum("bd, bkd -> bk", emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.einsum("bd, bkd -> bk", emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.einsum("bd, bkd -> bk", emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real


class FSDPTransE(FSDPShardedEntityModel):
    def __init__(self, args):
        super().__init__(args)
        self.name = "TransE"
        self._norm = 2
        self.margin = 4

    def score(self, head_ent_emb, rel_ent_emb, tail_ent_emb):
        return self.margin - torch.nn.functional.pairwise_distance(
            head_ent_emb + rel_ent_emb,
            tail_ent_emb,
            p=self._norm,
        )

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        if self.manual_sharded_entity_training:
            raise NotImplementedError("Sharded TransE currently supports only NegSample/FixedNegSample training.")
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        distance = torch.nn.functional.pairwise_distance(
            torch.unsqueeze(emb_head_real + emb_rel_real, 1),
            self.entity_embeddings.weight,
            p=self._norm,
        )
        return self.margin - distance
