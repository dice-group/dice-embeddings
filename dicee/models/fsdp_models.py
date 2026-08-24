"""
Row-wise sharded entity embeddings for multi-GPU training.

Entity embeddings are partitioned row-wise across ranks:
  - dist.all_to_all_single routes index requests to the owning rank
  - A custom autograd Function propagates gradients back through the all_to_all
  - _LocalSparseAdam updates only the rows that received a non-zero gradient

Public API:
  model.setup_fsdp_training(device, lr)              — called by trainer before loop
  model.gather_entity_embeddings_on_rank_zero()       — called by trainer after loop
  model.save_local_shard_checkpoint(path)             — per-rank, called periodically during loop
  model.load_local_shard_checkpoint(path)             — per-rank, called by trainer on resume
  create_fsdp_sharded_model_class(ModelClass)         — factory used by static_funcs.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_model import BaseKGE

_SHARDED_MODEL_CACHE: Dict[Type[BaseKGE], Type[BaseKGE]] = {}


# ---------------------------------------------------------------------------
# Local sparse Adam for a single per-rank embedding shard
# ---------------------------------------------------------------------------

class _LocalSparseAdam:
    """Sparse Adam that only updates rows that received a non-zero gradient.

    State tensors live on the same device as the embedding weight and are
    stored in bfloat16 (same exponent range as float32, half the memory).
    The Adam arithmetic is done in float32 and written back to bfloat16.
    """

    def __init__(
        self,
        local_rows: int,
        embedding_dim: int,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        device=None,
        state_dtype: torch.dtype = torch.bfloat16,
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.state_dtype = state_dtype

        self.exp_avg = torch.zeros(local_rows, embedding_dim,
                                   device=device, dtype=state_dtype)
        self.exp_avg_sq = torch.zeros(local_rows, embedding_dim,
                                      device=device, dtype=state_dtype)

    @torch.no_grad()
    def step(self, weight: torch.Tensor, grad: torch.Tensor) -> None:
        """Apply Adam update to rows with non-zero gradient.

        Works with optimizer states on any device (GPU or CPU).
        GPU states: pure GPU kernels, zero PCIe traffic, < 1 ms per step.
        CPU states: GPU→CPU index+grad transfer, CPU float32 math, CPU→GPU write-back.
        """
        if grad.is_sparse:
            grad_c = grad.coalesce()
            active_rows = grad_c.indices()[0]   # [k] on grad's device (GPU)
            g            = grad_c.values()       # [k, D] on GPU
        else:
            active_rows = grad.norm(dim=1).nonzero(as_tuple=True)[0]
            g            = grad[active_rows]

        if active_rows.numel() == 0:
            return

        self.step_count += 1

        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count
        step_size = self.lr * math.sqrt(bc2) / bc1

        # Move index and grad to wherever the optimizer states live.
        # GPU→GPU: zero-cost; GPU→CPU: PCIe transfer (only when states are on CPU).
        state_device = self.exp_avg.device
        idx = active_rows.to(state_device)
        g_f = g.to(device=state_device, dtype=torch.float32)

        ea  = self.exp_avg[idx].float()
        eas = self.exp_avg_sq[idx].float()

        ea.mul_(self.beta1).add_(g_f, alpha=1.0 - self.beta1)
        eas.mul_(self.beta2).addcmul_(g_f, g_f, value=1.0 - self.beta2)

        self.exp_avg[idx]    = ea.to(self.state_dtype)
        self.exp_avg_sq[idx] = eas.to(self.state_dtype)

        update = ea / (eas.sqrt() + self.eps)
        # index_add_ writes back to weight in-place.
        # fancy-indexing (weight.data[active_rows].add_(...)) creates a copy and
        # discards it — the original tensor would never be modified.
        weight.data.index_add_(
            0, active_rows,
            update.to(device=weight.device, dtype=weight.dtype) * (-step_size),
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return this rank's optimizer state (moment buffers + step count)."""
        return {
            "exp_avg": self.exp_avg.detach().cpu(),
            "exp_avg_sq": self.exp_avg_sq.detach().cpu(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state saved by :meth:`state_dict`, in-place."""
        self.exp_avg.copy_(state["exp_avg"].to(device=self.exp_avg.device, dtype=self.exp_avg.dtype))
        self.exp_avg_sq.copy_(state["exp_avg_sq"].to(device=self.exp_avg_sq.device, dtype=self.exp_avg_sq.dtype))
        self.step_count = state["step_count"]


# ---------------------------------------------------------------------------
# Differentiable all-to-all embedding lookup
# ---------------------------------------------------------------------------

class _RowWiseEmbLookup(torch.autograd.Function):
    """Forward : route index requests → local lookup → route embeddings back.
    Backward : route grad back to owning ranks → scatter-add into weight.grad.

    Both directions use dist.all_to_all_single (variable-length splits).
    """

    @staticmethod
    def forward(
        ctx,
        weight,          # [local_rows, D]  nn.Parameter on this rank
        sorted_indices,  # [N]              indices sorted by owner rank (int64)
        send_counts,     # list[int]        #indices sent to each rank
        recv_counts,     # list[int]        #indices received from each rank
        unsort_idx,      # [N]              inverse permutation (int64)
        start_row,       # int              global row offset for this rank
        local_rows,      # int              number of rows on this rank
    ):
        total_recv = sum(recv_counts)
        D = weight.shape[1]
        device = weight.device

        # 1. Send index requests to owning ranks
        recv_indices = torch.empty(total_recv, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.all_to_all_single(
                recv_indices, sorted_indices,
                output_split_sizes=recv_counts,
                input_split_sizes=send_counts,
            )
        else:
            recv_indices.copy_(sorted_indices)

        # 2. Local lookup (localize to [0, local_rows))
        local_idx = (recv_indices - start_row).clamp(0, max(local_rows - 1, 0))
        local_embs = weight[local_idx]  # [total_recv, D]

        # 3. Send embeddings back to requesting ranks
        N = sorted_indices.shape[0]
        sorted_result = torch.empty(N, D, device=device, dtype=weight.dtype)
        if dist.is_initialized():
            dist.all_to_all_single(
                sorted_result, local_embs,
                output_split_sizes=send_counts,
                input_split_sizes=recv_counts,
            )
        else:
            sorted_result.copy_(local_embs)

        # 4. Unsort — restore original request order
        output = sorted_result[unsort_idx]

        ctx.save_for_backward(local_idx, unsort_idx)
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.local_rows  = local_rows
        ctx.D           = D
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """grad_output: [N, D] gradient for the embedding outputs (after unsort)."""
        local_idx, unsort_idx = ctx.saved_tensors
        D      = ctx.D
        device = grad_output.device

        # Invert the unsort to put grads back in sorted (send) order
        sort_idx    = torch.argsort(unsort_idx)
        grad_sorted = grad_output[sort_idx]           # [N, D]

        # Route grads back to owning ranks (reverse of step 3)
        total_recv = sum(ctx.recv_counts)
        grad_local = torch.empty(total_recv, D, device=device, dtype=grad_output.dtype)
        if dist.is_initialized():
            dist.all_to_all_single(
                grad_local, grad_sorted,
                output_split_sizes=ctx.recv_counts,
                input_split_sizes=ctx.send_counts,
            )
        else:
            grad_local.copy_(grad_sorted)

        # Return a SPARSE gradient instead of a dense [local_rows, D] tensor.
        # Dense would allocate 42+ GiB (= one full shard) every backward pass.
        # Sparse stores only the ~total_recv accessed rows (~153 MB).
        # Duplicate indices (same entity looked up multiple times per batch)
        # are merged by .coalesce() inside _LocalSparseAdam.step.
        sparse_grad = torch.sparse_coo_tensor(
            local_idx.unsqueeze(0),   # [1, total_recv] — row indices
            grad_local,               # [total_recv, D] — gradient values
            size=(ctx.local_rows, D),
            device=device,
        )
        # (weight, sorted_indices, send_counts, recv_counts, unsort_idx, start_row, local_rows)
        return sparse_grad, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Row-wise sharded embedding module
# ---------------------------------------------------------------------------

class RowWiseShardedEmbedding(nn.Module):
    """Entity embedding table sharded row-wise across ranks.

    Each rank owns rows [start_row, end_row).  forward() looks like nn.Embedding
    so every existing scoring function works unchanged.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int,
        world_size: int,
        device: torch.device,
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.rank           = rank
        self.world_size     = world_size
        self.device         = device

        self.shard_size = math.ceil(num_embeddings / world_size)
        self.start_row  = rank * self.shard_size
        self.end_row    = min(self.start_row + self.shard_size, num_embeddings)
        self.local_rows = self.end_row - self.start_row

        # bfloat16 halves the per-rank GPU footprint (42.8 GiB → 21.4 GiB for 180M entities).
        # bfloat16 has the same exponent range as float32 so no overflow risk.
        # The Adam optimizer states remain float32 on CPU for numerical stability.
        self.weight = nn.Parameter(
            torch.empty(self.local_rows, embedding_dim, device=device, dtype=weight_dtype)
        )
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(embedding_dim))

        self._local_adam: Optional[_LocalSparseAdam] = None

    def forward(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        original_shape = entity_ids.shape
        flat = entity_ids.reshape(-1)     # [N]

        # Which rank owns each requested index?
        owner_ranks = (flat // self.shard_size).clamp(0, self.world_size - 1)

        # Sort by owner so all_to_all sends contiguous chunks
        sort_idx    = torch.argsort(owner_ranks, stable=True)
        unsort_idx  = torch.argsort(sort_idx,    stable=True)
        sorted_flat = flat[sort_idx]

        send_counts = torch.bincount(owner_ranks, minlength=self.world_size).tolist()
        send_counts = [int(c) for c in send_counts]

        # Exchange counts with all other ranks
        if dist.is_initialized():
            send_t = torch.tensor(send_counts, dtype=torch.long, device=self.device)
            recv_t = torch.zeros_like(send_t)
            dist.all_to_all_single(recv_t, send_t)
            recv_counts = [int(c) for c in recv_t.tolist()]
        else:
            recv_counts = send_counts

        out_flat = _RowWiseEmbLookup.apply(
            self.weight, sorted_flat,
            send_counts, recv_counts,
            unsort_idx,
            self.start_row, self.local_rows,
        )

        if len(original_shape) == 1:
            return out_flat
        return out_flat.reshape(*original_shape, self.embedding_dim)


# ---------------------------------------------------------------------------
# Model mixin
# ---------------------------------------------------------------------------

class FSDPShardedEntityModel(BaseKGE):
    """Mixin that defers entity embedding allocation and wires up row-wise sharding.

    Lifecycle
    ---------
    1. __init__()                         — entity_embeddings is None
    2. setup_fsdp_training(dev, lr)   — creates RowWiseShardedEmbedding
    3. gather_entity_embeddings_on_rank_zero() — called by trainer after last epoch
    """

    def __init__(self, args):
        _args = dict(args)
        _args["fsdp_sharded_entity"] = True   # BaseKGE sets entity_embeddings=None
        super().__init__(_args)
        self._fsdp_dmp     = None   # kept so existing trainer isinstance checks pass
        self._fsdp_adapter: Optional[RowWiseShardedEmbedding] = None

    def setup_fsdp_training(
        self,
        device: torch.device,
        lr: float,
        optimizer_cls=None,       # unused — we always use _LocalSparseAdam
        optimizer_kwargs: dict = None,
        adam_device: torch.device = None,
    ) -> None:
        """Create the per-rank embedding shard and its local sparse Adam.

        adam_device — where exp_avg / exp_avg_sq are stored:
          GPU  (default) : pure GPU kernels, no PCIe, ~42.8 GiB extra GPU RAM per rank.
          CPU            : lower GPU footprint, PCIe transfer each step.
        Falls back to the embedding device (GPU) when not specified.
        """
        rank       = dist.get_rank()       if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        sharded_emb = RowWiseShardedEmbedding(
            num_embeddings=self.num_entities,
            embedding_dim=self.embedding_dim,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        # Respect the model's weight initializer (xavier_normal or identity)
        self.param_init(sharded_emb.weight.data)

        lr_eff = (optimizer_kwargs or {}).get("lr", lr)
        state_device = adam_device if adam_device is not None else device
        sharded_emb._local_adam = _LocalSparseAdam(
            local_rows=sharded_emb.local_rows,
            embedding_dim=self.embedding_dim,
            lr=lr_eff,
            device=state_device,
            state_dtype=torch.bfloat16,
        )

        self._fsdp_adapter = sharded_emb
        self.entity_embeddings  = sharded_emb

    def gather_entity_embeddings_on_rank_zero(self) -> Optional[nn.Embedding]:
        """Gather the full entity table on rank 0; return None on other ranks."""
        if self._fsdp_adapter is None:
            raise RuntimeError("setup_fsdp_training() must be called before gathering.")
        return _gather_shards(self._fsdp_adapter)

    def save_local_shard_checkpoint(self, path: str) -> None:
        """Save this rank's entity embedding shard + its local Adam state to *path*.

        Cheap and O(local_rows) — unlike gather_entity_embeddings_on_rank_zero(),
        no collective communication is involved, so this is safe to call
        periodically even when the full table is too large to gather.
        """
        adapter = self._fsdp_adapter
        if adapter is None or adapter._local_adam is None:
            raise RuntimeError("setup_fsdp_training() must be called before checkpointing.")
        torch.save({
            "weight": adapter.weight.data.detach().cpu(),
            "adam": adapter._local_adam.state_dict(),
            "start_row": adapter.start_row,
            "local_rows": adapter.local_rows,
            "rank": adapter.rank,
            "world_size": adapter.world_size,
        }, path)

    def load_local_shard_checkpoint(self, path: str) -> None:
        """Load this rank's entity embedding shard + local Adam state from *path*.

        The checkpoint must have been saved by a run with the same world_size:
        shard boundaries (start_row/local_rows) are derived from world_size and
        num_entities, so a mismatch means this rank does not own the same rows
        as when the checkpoint was written.
        """
        adapter = self._fsdp_adapter
        if adapter is None or adapter._local_adam is None:
            raise RuntimeError("setup_fsdp_training() must be called before checkpointing.")
        ckpt = torch.load(path, map_location="cpu")
        if ckpt["start_row"] != adapter.start_row or ckpt["local_rows"] != adapter.local_rows:
            raise RuntimeError(
                f"Shard checkpoint layout mismatch on rank {adapter.rank}: checkpoint covers "
                f"rows [{ckpt['start_row']}, {ckpt['start_row'] + ckpt['local_rows']}), but this "
                f"run's shard is [{adapter.start_row}, {adapter.start_row + adapter.local_rows}). "
                f"Resuming a sharded FSDP checkpoint requires launching with the same world_size "
                f"used to save it (checkpoint world_size={ckpt['world_size']}, "
                f"current world_size={adapter.world_size})."
            )
        adapter.weight.data.copy_(ckpt["weight"].to(device=adapter.weight.device, dtype=adapter.weight.dtype))
        adapter._local_adam.load_state_dict(ckpt["adam"])

    def get_embeddings(self):
        raise RuntimeError(
            "Full entity embeddings are not available during sharded training; "
            "they are materialised by the trainer on rank 0 after training."
        )

    def forward_k_vs_all(self, *args, **kwargs):
        raise RuntimeError(
            "forward_k_vs_all accesses entity_embeddings.weight directly and cannot "
            "run with row-wise sharding. Use --scoring_technique FSDP1vsSample instead."
        )


# ---------------------------------------------------------------------------
# Gather helper
# ---------------------------------------------------------------------------

def _gather_shards(sharded_emb: RowWiseShardedEmbedding) -> Optional[nn.Embedding]:
    """Collect all per-rank shards and build a full nn.Embedding on rank 0."""
    world_size   = dist.get_world_size() if dist.is_initialized() else 1
    rank         = dist.get_rank()       if dist.is_initialized() else 0
    local_w_cpu  = sharded_emb.weight.data.detach().cpu().float()
    local_rows   = local_w_cpu.shape[0]
    D            = local_w_cpu.shape[1]
    num_entities = sharded_emb.num_embeddings

    if world_size == 1:
        emb = nn.Embedding(num_entities, D)
        emb.weight.data.copy_(local_w_cpu)
        return emb

    # Exchange shard sizes so every rank knows the row offsets
    rows_t      = torch.tensor([local_rows], device=sharded_emb.device, dtype=torch.long)
    all_rows_t  = [torch.zeros_like(rows_t) for _ in range(world_size)]
    dist.all_gather(all_rows_t, rows_t)
    all_rows    = [int(t.item()) for t in all_rows_t]

    if sum(all_rows) != num_entities:
        raise RuntimeError(f"Gathered {sum(all_rows)} rows, expected {num_entities}.")

    dest_starts = [sum(all_rows[:i]) for i in range(world_size)]
    full_weight = torch.zeros(num_entities, D, dtype=torch.float32) if rank == 0 else None

    max_rows = max(all_rows)
    chunk    = min(65_536, max_rows)
    dev      = sharded_emb.device
    # gathered and buf must be on the same CUDA device as the input —
    # NCCL cannot write into CPU memory via all_gather (cudaErrorIllegalAddress).
    gathered = [torch.empty(chunk, D, dtype=torch.float32, device=dev) for _ in range(world_size)]

    for start in range(0, max_rows, chunk):
        cur   = min(chunk, max_rows - start)
        buf   = torch.zeros(chunk, D, dtype=torch.float32, device=dev)
        valid = min(cur, max(0, local_rows - start))
        if valid > 0:
            buf[:valid].copy_(local_w_cpu[start: start + valid])
        dist.all_gather(gathered, buf)
        if rank == 0:
            for r, shard_chunk in enumerate(gathered):
                v = min(cur, max(0, all_rows[r] - start))
                if v <= 0:
                    continue
                dst = dest_starts[r] + start
                full_weight[dst: dst + v] = shard_chunk[:v].cpu()

    if rank != 0:
        return None
    emb = nn.Embedding(num_entities, D)
    emb.weight.data.copy_(full_weight)
    return emb


# ---------------------------------------------------------------------------
# Factory — API unchanged so static_funcs.py needs no edits
# ---------------------------------------------------------------------------

def create_fsdp_sharded_model_class(model_class: Type[BaseKGE]) -> Type[BaseKGE]:
    """Return a row-wise sharded variant of *model_class*.

    The returned class inherits all scoring functions from *model_class* unchanged.
    Only entity_embeddings is replaced at training time.
    """
    if issubclass(model_class, FSDPShardedEntityModel):
        return model_class
    if model_class not in _SHARDED_MODEL_CACHE:
        _SHARDED_MODEL_CACHE[model_class] = type(
            f"FSDP{model_class.__name__}",
            (FSDPShardedEntityModel, model_class),
            {
                "__module__": model_class.__module__,
                "__doc__": f"Row-wise sharded variant of {model_class.__name__}.",
            },
        )
    return _SHARDED_MODEL_CACHE[model_class]
