import os
import time
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
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _FSDPShardedEntityModel(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.manual_sharded_entity_training = self.defer_large_embeddings
        self.local_entity_embeddings = None
        self.local_entity_start = 0
        self.local_entity_end = self.num_entities
        self.local_entity_count = self.num_entities
        self._lookup_profile_total = 0.0
        self._lookup_profile_calls = 0
        self._lookup_profile_unique = 0
        self._batch_lookup_ids = None
        self._batch_lookup_embeddings = None
        if self.manual_sharded_entity_training:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
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

    def configure_optimizers(self, parameters=None):
        if not self.manual_sharded_entity_training:
            return super().configure_optimizers(parameters=parameters)
        dense_parameters = [
            param for name, param in self.named_parameters()
            if name != "local_entity_embeddings.weight"
        ]
        return super().configure_optimizers(parameters=dense_parameters)

    def _entity_lookup(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        if not self.manual_sharded_entity_training:
            return self.entity_embeddings(entity_ids)

        if self._batch_lookup_ids is not None and self._batch_lookup_embeddings is not None:
            positions = torch.searchsorted(self._batch_lookup_ids, entity_ids)
            if torch.equal(self._batch_lookup_ids.index_select(0, positions), entity_ids):
                return self._batch_lookup_embeddings.index_select(0, positions)

        return self._distributed_entity_lookup(entity_ids)

    def _distributed_entity_lookup(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        start_time = time.perf_counter()
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
        self._lookup_profile_total += time.perf_counter() - start_time
        self._lookup_profile_calls += 1
        self._lookup_profile_unique += int(unique_entity_ids.numel())
        return outputs.index_select(0, inverse_indices)

    def _prime_batch_lookup_cache(self, entity_ids: torch.LongTensor) -> None:
        unique_entity_ids = torch.unique(entity_ids, sorted=True)
        self._batch_lookup_ids = unique_entity_ids
        self._batch_lookup_embeddings = self._distributed_entity_lookup(unique_entity_ids)

    def _clear_batch_lookup_cache(self) -> None:
        self._batch_lookup_ids = None
        self._batch_lookup_embeddings = None

    def consume_lookup_profile(self):
        profile = {
            "lookup_seconds": self._lookup_profile_total,
            "lookup_calls": self._lookup_profile_calls,
            "lookup_unique_rows": self._lookup_profile_unique,
        }
        self._lookup_profile_total = 0.0
        self._lookup_profile_calls = 0
        self._lookup_profile_unique = 0
        return profile

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


class FSDPDistMult(_FSDPShardedEntityModel):
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
            t = self._entity_lookup(target_entity_idx.reshape(-1)).reshape(target_entity_idx.shape[0], target_entity_idx.shape[1], -1)
            emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
            hr = torch.einsum("bd, bd -> bd", emb_head_real, emb_rel_real)
            return torch.einsum("bd, bkd -> bk", hr, t)
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        hr = torch.einsum("bd, bd -> bd", emb_head_real, emb_rel_real)
        t = self.entity_embeddings(target_entity_idx)
        return torch.einsum("bd, bkd -> bk", hr, t)

    def score(self, h, r, t):
        return (self.hidden_dropout(self.hidden_normalizer(h * r)) * t).sum(dim=1)


class FSDPComplEx(_FSDPShardedEntityModel):
    def __init__(self, args):
        super().__init__(args)
        self.name = "ComplEx"

    @staticmethod
    def score(head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor, tail_ent_emb: torch.FloatTensor):
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)
        real_real_real = (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    @staticmethod
    def k_vs_all_score(emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor):
        emb_head_real, emb_head_imag = torch.hsplit(emb_h, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(emb_r, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(emb_E, 2)
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
            emb_t = self._entity_lookup(target_entity_idx.reshape(-1)).reshape(target_entity_idx.shape[0], target_entity_idx.shape[1], -1)
            emb_h, emb_r = self.get_head_relation_representation(x)
        else:
            emb_t = self.entity_embeddings(target_entity_idx)
            emb_h, emb_r = self.get_head_relation_representation(x)
        emb_head_real, emb_head_imag = torch.hsplit(emb_h, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(emb_r, 2)
        emb_tail_real, emb_tail_imag = torch.split(emb_t, self.embedding_dim // 2, dim=-1)
        real_real_real = torch.einsum("bd, bkd -> bk", emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.einsum("bd, bkd -> bk", emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.einsum("bd, bkd -> bk", emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.einsum("bd, bkd -> bk", emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
