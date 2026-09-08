"""
Multi-GPU trainer: row-wise sharded entity embeddings + FSDP dense parameters.

Entity embeddings are sharded row-wise across ranks via dist.all_to_all_single.
Each rank owns a contiguous slice of entity rows; forward and backward use
all_to_all to route index requests and gradient returns to the owning rank.
A per-rank _LocalSparseAdam updates only the rows that received a non-zero
gradient each batch.

Dense parameters (relation embeddings, Clifford coefficients, …) are wrapped
with FSDP and updated by the standard dense optimizer.
"""

import json
import logging
import math
import os
import threading
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader

from dicee.abstracts import AbstractTrainer
from dicee.static_funcs_training import make_iterable_verbose

logger = logging.getLogger(__name__)

try:
    from torch._dynamo.eval_frame import OptimizedModule
except ImportError:
    OptimizedModule = None


class _FastDistributedSampler(torch.utils.data.Sampler):
    """Drop-in replacement for DistributedSampler that avoids torch.randperm().tolist().

    PyTorch's DistributedSampler calls torch.randperm(N).tolist() at the start
    of every epoch. For large N (tens of millions), .tolist() allocates ~N*28 bytes
    of Python int objects and takes several seconds — causing a visible stall at
    each epoch boundary.

    This sampler uses numpy: it shuffles a numpy int32 array in-place and yields
    numpy scalars directly. numpy scalars support __index__ and work as dataset
    indices in __getitem__, so no .tolist() is ever needed.
    """

    def __init__(self, dataset, num_replicas: int, rank: int,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        self.n = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if drop_last and self.n % num_replicas != 0:
            self.num_samples = math.ceil((self.n - num_replicas) / num_replicas)
        else:
            self.num_samples = math.ceil(self.n / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        indices = np.arange(self.n, dtype=np.int32)
        if self.shuffle:
            # numpy default_rng is fast and produces the same sequence as
            # torch.randperm for the same seed (different algorithm, but
            # deterministic per (seed, epoch) pair — sufficient for shuffling).
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)  # in-place, no extra allocation

        if not self.drop_last:
            padding = self.total_size - self.n
            if padding > 0:
                indices = np.concatenate([indices, indices[:padding]])
        else:
            indices = indices[: self.total_size]

        per_rank = self.total_size // self.num_replicas
        my_slice = indices[self.rank * per_rank : (self.rank + 1) * per_rank]
        # yield numpy int32 scalars — they satisfy __index__ so DataLoader
        # workers can use them directly as dataset indices without any Python list.
        return iter(my_slice)


def _move_to_device(batch, device: torch.device):
    """Move a dataloader batch (2- or 3-element) to *device*."""
    def move(v):
        if isinstance(v, tuple):
            return tuple(move(i) for i in v)
        if isinstance(v, list):
            return [move(i) for i in v]
        return v.to(device, non_blocking=True)

    if len(batch) == 2:
        x, y = batch
        return move(x), move(y)
    if len(batch) == 3:
        x, y_idx, y = batch
        return (move(x), move(y_idx)), move(y)
    raise ValueError(f"Unexpected batch length: {len(batch)}")


class TorchFSDPTrainer(AbstractTrainer):
    """Single-node multi-GPU trainer: row-wise sharded entity embeddings + FSDP.

    Entity embeddings
    -----------------
    Sharded row-wise across GPUs.  all_to_all routes each index request to its
    owning rank; embeddings and gradients travel back the same way.
    _LocalSparseAdam updates only the rows accessed in each batch — O(active_rows)
    not O(shard_size).  Each entity receives gradient from every rank's batches.

    Dense parameters (relation embeddings, Clifford coefficients, …)
    ----------------------------------------------------------------
    Wrapped with FSDP; updated by the standard dense optimizer step().

    Usage
    -----
    Launched with torchrun:
        torchrun --nproc_per_node=4 -u dicee --trainer torchFSDP ...
    """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.is_global_zero = (self.global_rank == 0)
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.model = None        # FSDP-wrapped (dense part)
        self.raw_model = None    # original model instance
        self.optimizer = None    # dense optimizer
        self.loss_func = None
        self.train_dataset_loader = None
        self._adam_thread: threading.Thread | None = None
        self._entity_grad_scale: float = 1.0  # scale captured before scaler.update()

        fsdp_kwargs = getattr(args, "fsdp_trainer_kwargs", {}) or {}

        ptdtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        # bfloat16 by default: halves entity-table memory vs float32, same exponent
        # range so no overflow risk.
        self.ptdtype = ptdtype_map.get(fsdp_kwargs.get("precision", "bfloat16"), torch.bfloat16)
        self.ctx = torch.amp.autocast(device_type="cuda", dtype=self.ptdtype)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.ptdtype == torch.float16))
        # One dtype for both the entity table and the dense params.
        self.entity_weight_dtype = self.ptdtype

        self.sharding_strategy = fsdp_kwargs.get("sharding_strategy", "FULL_SHARD")
        self.gradient_clip_val = fsdp_kwargs.get("gradient_clip_val", None)
        self.use_compile = fsdp_kwargs.get("use_compile", False)

        # DataLoader config
        self.num_workers = fsdp_kwargs.get("num_workers", self.attributes.num_core)
        self.prefetch_factor = fsdp_kwargs.get("prefetch_factor", 4)

        _optim_dev = fsdp_kwargs.get("fsdp_optim_device", "cpu")
        self.entity_optim_device = (
            torch.device(f"cuda:{self.local_rank}") if _optim_dev == "gpu"
            else torch.device("cpu")
        )

        # Mid-run checkpointing: save every N epochs so a killed/preempted run
        # (e.g. OOM, node failure, SLURM preemption) can be resumed instead of
        # restarting from scratch. None (default) disables periodic saving.
        self.checkpoint_every_n_epochs = fsdp_kwargs.get("checkpoint_every_n_epochs", None)
        self._checkpoint_dir = os.path.join(self.attributes.full_storage_path, "fsdp_shard_checkpoint")
        # Set while a resume load is in flight, between setup_fsdp_training() and
        # the dense optimizer being constructed — see fit().
        self._pending_resume_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        (model,) = args
        self.on_fit_start(self, model)

        base_loader = kwargs["train_dataloaders"]
        self.train_dataset_loader = DataLoader(
            base_loader.dataset,
            batch_size=self.attributes.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=base_loader.dataset.collate_fn,
            sampler=_FastDistributedSampler(
                base_loader.dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=getattr(self.attributes, "random_seed", 0),
                drop_last=False,
            ),
        )

        self.raw_model = model
        self.raw_model.to(self.device)

        # Resolve a checkpoint to resume from (if any) BEFORE dense params are
        # wrapped: rank 0 loads the saved dense weights directly into raw_model,
        # and FSDP's sync_module_states=True (below) broadcasts them to every
        # other rank as part of the normal wrap — no extra collective needed.
        resume_dir = self._resolve_resume_checkpoint_dir()
        start_epoch = 0
        if resume_dir is not None:
            logger.info(f"[Rank {self.global_rank}] Resuming FSDP training from {resume_dir}")
            self._load_dense_checkpoint(resume_dir)

        # 1. FSDP-wrap dense parameters FIRST, while entity_embeddings is still None.
        #    FSDP init (sync_module_states, device_id broadcast) never sees the large
        #    entity shard — avoids the double-allocation that caused OOM.
        self.model = self._wrap_dense_with_fsdp()

        # 2. Create the per-rank entity embedding shard AFTER FSDP is initialised.
        #    FSDP does not manage these parameters; each rank holds its own shard
        #    independently.
        self.raw_model.setup_fsdp_training(
            device=self.device,
            lr=self.attributes.learning_rate,
            adam_device=self.entity_optim_device,
            weight_dtype=self.entity_weight_dtype,
        )

        if resume_dir is not None:
            self.raw_model.load_local_shard_checkpoint(self._entity_shard_path(resume_dir))
            meta = self._load_meta(resume_dir)
            start_epoch = meta["epoch"]
            self.raw_model.loss_history = list(meta["loss_history"])

        self.loss_func = model.loss

        # 3. Dense optimizer covers relation embeddings + model-specific params.
        #    Entity embedding weight is excluded (handled by _LocalSparseAdam).
        dense_params = self._collect_dense_params()
        self.optimizer = model.configure_optimizers(parameters=dense_params)

        if resume_dir is not None:
            self._load_optimizer_and_scaler_checkpoint(resume_dir)

        if self.use_compile and hasattr(torch, "compile"):
            if self.local_rank == 0:
                logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        num_of_batches = len(self.train_dataset_loader)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Training loop
        for epoch in (tqdm_bar := make_iterable_verbose(
            range(start_epoch, self.attributes.num_epochs),
            verbose=self.is_global_zero,
            position=0,
            leave=True,
        )):
            self.train_dataset_loader.sampler.set_epoch(epoch)
            if self.is_global_zero:
                self.on_train_epoch_start(self, self.raw_model)
            epoch_loss = 0.0

            for i, z in enumerate(self.train_dataset_loader):
                source, targets = self.extract_input_outputs(z)
                batch_loss = self._run_batch(source, targets)
                epoch_loss += batch_loss

                if hasattr(tqdm_bar, "set_description_str"):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    tqdm_bar.set_postfix_str(
                        f"batch={i + 1}/{num_of_batches}, "
                        f"loss_step={batch_loss:.5f}, "
                        f"loss_epoch={epoch_loss / (i + 1):.5f}"
                    )

            loss_t = torch.tensor(epoch_loss / num_of_batches, device=self.device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            avg_epoch_loss = loss_t.item()
            self.raw_model.loss_history.append(avg_epoch_loss)

            # Collective — every rank must participate, so this runs unconditionally
            # (unlike the is_global_zero-gated callbacks below).
            if self.checkpoint_every_n_epochs and (epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self._save_sharded_checkpoint(epoch + 1)

            if self.is_global_zero:
                self.on_train_epoch_end(self, self.raw_model)

        # Flush the last batch's async Adam before the barrier.
        if self._adam_thread is not None:
            self._adam_thread.join()
            self._adam_thread = None

        # Materialize: gather all entity shards + dense FSDP state on rank 0.
        dist.barrier()
        trained_model = self._materialize_model()

        if self.is_global_zero:
            self.on_fit_end(self, trained_model)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        return trained_model

    # ------------------------------------------------------------------
    # Batch step
    # ------------------------------------------------------------------

    def _run_batch(
        self, source: torch.LongTensor, targets: torch.FloatTensor
    ) -> float:
        """One gradient step.

        Async Adam thread timing
        ------------------------
        The thread is launched at the END of batch N (after scaler.update) and
        joined at the START of batch N+1 (before model(source) / FSDP all_gather).

        Why this window is safe:
          • After scaler.update() — FSDP reduce_scatter and GradScaler are fully
            done; neither will re-enter CUDA Graph capture until the next backward.
          • Before model(source) — FSDP all_gather begins here.  Joining before
            this call ensures the thread has no live CUDA-stream access before FSDP
            touches the default stream again.

        What the overlap buys:
          DataLoader.__next__() + zero_grad() run on the main thread while the
          background thread runs PCIe transfer + CPU Adam (CPU states) or a few
          GPU kernels (GPU states), both of which are outside the CUDA Graph
          capture window.
        """
        # ── join previous batch's Adam BEFORE FSDP all_gather ──────────────────
        if self._adam_thread is not None:
            self._adam_thread.join()
            self._adam_thread = None

        self.optimizer.zero_grad(set_to_none=True)

        with self.ctx:
            output = self.model(source)          # ← FSDP all_gather starts here
            loss = self.loss_func(output, targets)

        batch_loss = loss.item()

        self.scaler.scale(loss).backward()       # ← FSDP reduce_scatter + capture

        # Dense optimizer (all FSDP/scaler GPU operations finish inside update())
        self.scaler.unscale_(self.optimizer)
        if self.gradient_clip_val is not None:
            self._clip_dense_grad_norm()
        self.scaler.step(self.optimizer)
        # Capture the scale used in this backward BEFORE update() changes it.
        # get_scale() after update() returns the next batch's scale, which would
        # produce incorrect unscaling of the entity embedding gradient.
        self._entity_grad_scale = self.scaler.get_scale()
        self.scaler.update()

        # ── launch entity Adam AFTER all capture activity is done ───────────────
        self._adam_thread = self._launch_entity_adam_async()

        return batch_loss

    def _launch_entity_adam_async(self) -> "threading.Thread | None":
        """Snapshot and unscale the entity grad on the main thread, then run
        _LocalSparseAdam in a daemon thread.

        Unscaling must happen on the main thread immediately after scaler.update()
        because scaler.get_scale() is not thread-safe and the scale value is only
        valid until the next scaler.update() call.
        """
        adapter = getattr(self.raw_model, "_fsdp_adapter", None)
        if adapter is None or not hasattr(adapter, "_local_adam"):
            return None
        weight = adapter.weight
        if weight.grad is None:
            return None

        grad = weight.grad   # sparse COO from _RowWiseEmbLookup.backward

        # Manual unscale for fp16 GradScaler; bfloat16 has scaler disabled (no-op path).
        # Use _entity_grad_scale (captured before scaler.update()) — not get_scale(),
        # which returns the next batch's scale after update() has already run.
        if self.scaler.is_enabled():
            scale = self._entity_grad_scale
            if scale == 0.0 or not math.isfinite(scale):
                weight.grad = None
                return None
            if grad.is_sparse:
                grad_c = grad.coalesce()
                grad = torch.sparse_coo_tensor(
                    grad_c.indices(),
                    grad_c.values() / scale,
                    grad_c.size(),
                    device=grad_c.device,
                )
            else:
                grad = grad.detach().div_(scale)

        # Clear weight.grad on the main thread; the background thread holds the snapshot.
        weight.grad = None

        def _run():
            adapter._local_adam.step(weight, grad)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    # Input extraction
    # ------------------------------------------------------------------

    def extract_input_outputs(self, z):
        return _move_to_device(z, self.device)

    # ------------------------------------------------------------------
    # FSDP for dense parameters
    # ------------------------------------------------------------------

    def _wrap_dense_with_fsdp(self) -> FSDP:
        """Wrap dense parameters with FSDP, excluding the sharded entity embedding.

        RowWiseShardedEmbedding is passed as ignored_modules so FSDP never
        touches it — each rank legitimately holds a different shard.
        """
        mp_policy = MixedPrecision(
            param_dtype=self.ptdtype,
            reduce_dtype=self.ptdtype,
            buffer_dtype=self.ptdtype,
        )
        strategy = getattr(
            ShardingStrategy, self.sharding_strategy, ShardingStrategy.FULL_SHARD
        )

        return FSDP(
            self.raw_model,
            device_id=self.device,
            use_orig_params=True,
            sync_module_states=True,
            sharding_strategy=strategy,
            mixed_precision=mp_policy,
            ignored_modules=[],
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            forward_prefetch=True,
        )

    def _collect_dense_params(self):
        """Return parameters for the dense optimizer (excludes the entity embedding shard)."""
        entity_param_ids = set()
        adapter = getattr(self.raw_model, "_fsdp_adapter", None)
        if adapter is not None:
            for p in adapter.parameters():
                entity_param_ids.add(id(p))

        return [p for p in self.model.parameters() if id(p) not in entity_param_ids]

    # ------------------------------------------------------------------
    # Mid-run checkpoint (crash / preemption recovery)
    # ------------------------------------------------------------------
    #
    # Unlike _materialize_model() (which gathers the full entity table onto
    # rank 0's CPU — fine once, at the very end, but O(num_entities) memory
    # on a single host), these checkpoints are O(local shard) per rank:
    # each rank writes only its own slice of the entity table + its own
    # local Adam state + its own (already-sharded) dense-optimizer state.
    # Only the dense *parameters* are gathered (small — relation embeddings
    # etc., not the entity table) so they can be reloaded before the model
    # is wrapped with FSDP and broadcast via sync_module_states.
    #
    # Resuming requires the same world_size used to save: shard boundaries
    # are a deterministic function of (num_entities, world_size), so a
    # different world_size means a different, incompatible partition.

    def _rank_state_path(self, checkpoint_dir: str) -> str:
        return os.path.join(
            checkpoint_dir, f"rank_state_rank{self.global_rank}_of_{dist.get_world_size()}.pt"
        )

    def _entity_shard_path(self, checkpoint_dir: str) -> str:
        return os.path.join(
            checkpoint_dir, f"entity_shard_rank{self.global_rank}_of_{dist.get_world_size()}.pt"
        )

    @staticmethod
    def _load_meta(checkpoint_dir: str) -> dict:
        with open(os.path.join(checkpoint_dir, "meta.json")) as f:
            return json.load(f)

    def _resolve_resume_checkpoint_dir(self) -> Optional[str]:
        """Return a checkpoint directory to resume from, or None for a fresh run.

        Checked, in order:
          1. This run's own experiment folder — auto-resume after a
             crash/preemption that restarted into the same output dir.
             Nothing found here is not an error: it just means a fresh run.
          2. --continual_learning's folder — an explicit resume request. If the
             user pointed at this path, silently falling back to random init
             would be a much worse failure mode than a loud error, so a missing
             checkpoint here *is* an error (see the message below for why a
             fully-materialized model.pt alone isn't enough).

        meta.json is written last during save (see _save_sharded_checkpoint),
        so its presence means a complete checkpoint.
        """
        world_size = dist.get_world_size()

        if os.path.isfile(os.path.join(self._checkpoint_dir, "meta.json")):
            return self._validate_checkpoint_world_size(self._checkpoint_dir, world_size)

        cl = getattr(self.attributes, "continual_learning", None)
        if cl:
            cl_dir = os.path.join(cl, "fsdp_shard_checkpoint")
            if not os.path.isfile(os.path.join(cl_dir, "meta.json")):
                raise RuntimeError(
                    f"--continual_learning was set to '{cl}' but no sharded FSDP checkpoint "
                    f"was found at '{cl_dir}'. Resuming torchFSDP training is only supported "
                    f"from a checkpoint a previous torchFSDP run wrote itself (enable it with "
                    f"--fsdp_trainer_kwargs '{{\"checkpoint_every_n_epochs\": N}}'), not from a "
                    f"fully-materialized model.pt."
                )
            return self._validate_checkpoint_world_size(cl_dir, world_size)

        return None

    @staticmethod
    def _validate_checkpoint_world_size(checkpoint_dir: str, world_size: int) -> str:
        meta = TorchFSDPTrainer._load_meta(checkpoint_dir)
        if meta["world_size"] != world_size:
            raise RuntimeError(
                f"Cannot resume FSDP checkpoint at {checkpoint_dir}: it was saved with "
                f"world_size={meta['world_size']} but this run was launched with "
                f"world_size={world_size}. Resuming a sharded FSDP checkpoint requires "
                f"launching with the same number of ranks used to save it."
            )
        return checkpoint_dir

    def _load_dense_checkpoint(self, checkpoint_dir: str) -> None:
        """Rank 0 loads dense params into raw_model, pre-FSDP-wrap.

        sync_module_states=True (in _wrap_dense_with_fsdp) broadcasts these
        values to every other rank as part of the FSDP() constructor call —
        no separate collective load is needed here.
        """
        if self.global_rank != 0:
            return
        dense_state = torch.load(os.path.join(checkpoint_dir, "dense_state.pt"), map_location="cpu")
        self.raw_model.load_state_dict(dense_state, strict=True)

    def _load_optimizer_and_scaler_checkpoint(self, checkpoint_dir: str) -> None:
        """Every rank restores its own (already-sharded) dense optimizer + GradScaler state."""
        rank_state = torch.load(self._rank_state_path(checkpoint_dir), map_location="cpu")
        self.optimizer.load_state_dict(rank_state["dense_optimizer"])
        self.scaler.load_state_dict(rank_state["scaler"])

    def _save_sharded_checkpoint(self, epoch: int) -> None:
        """Collective: every rank must call this together.

        Write order matters for crash-safety: every per-rank/temp file is
        written and renamed into place first, then meta.json is written last
        as the commit marker. _resolve_resume_checkpoint_dir() only looks at
        meta.json, so a crash mid-save leaves no resumable (but truncated)
        checkpoint behind — the next attempt just overwrites the tmp files.
        """
        # The async Adam thread from the last batch of this epoch may still be
        # mutating entity weight/optimizer state — must finish before saving.
        if self._adam_thread is not None:
            self._adam_thread.join()
            self._adam_thread = None

        os.makedirs(self._checkpoint_dir, exist_ok=True)
        dist.barrier()

        rank_state_path = self._rank_state_path(self._checkpoint_dir)
        tmp = rank_state_path + ".tmp"
        torch.save({
            "dense_optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, tmp)
        os.replace(tmp, rank_state_path)

        entity_shard_path = self._entity_shard_path(self._checkpoint_dir)
        entity_shard_tmp = entity_shard_path + ".tmp"
        self.raw_model.save_local_shard_checkpoint(entity_shard_tmp)
        os.replace(entity_shard_tmp, entity_shard_path)

        # _gather_full_state_dict() is a collective — every rank must call it, even
        # though only rank 0's return value is populated (rank0_only=True below it).
        dense_state = self._strip_entity_keys(self._gather_full_state_dict())
        if self.is_global_zero:
            dense_tmp = os.path.join(self._checkpoint_dir, "dense_state.pt.tmp")
            dense_path = os.path.join(self._checkpoint_dir, "dense_state.pt")
            torch.save(dense_state, dense_tmp)
            os.replace(dense_tmp, dense_path)

        dist.barrier()

        if self.is_global_zero:
            meta_tmp = os.path.join(self._checkpoint_dir, "meta.json.tmp")
            meta_path = os.path.join(self._checkpoint_dir, "meta.json")
            with open(meta_tmp, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "world_size": dist.get_world_size(),
                    "num_entities": self.raw_model.num_entities,
                    "embedding_dim": self.raw_model.embedding_dim,
                    "loss_history": list(self.raw_model.loss_history),
                }, f)
            os.replace(meta_tmp, meta_path)
            logger.info(f"Saved FSDP shard checkpoint at epoch {epoch} → {self._checkpoint_dir}")

        dist.barrier()

    # ------------------------------------------------------------------
    # Materialization after training
    # ------------------------------------------------------------------

    def _materialize_model(self) -> torch.nn.Module:
        """Gather entity shards + dense FSDP state; rebuild a plain CPU model on rank 0.

          1. (collective) gather entity shards → nn.Embedding on rank 0, None elsewhere.
             Backed by a file-mapped tensor, not anonymous RAM, so an arbitrarily large
             table doesn't need to fit in rank 0's free RAM. The backing file is
             unlinked as soon as it is mapped, so nothing is left in the run directory.
          2. (collective) gather FSDP full state dict, offloaded to CPU, rank-0-only
          3. rank 0 only: construct a plain (non-sharded) model, wire in entity embeddings,
             strip adapter keys, load dense state

        The materialised model is always float32, regardless of the training `precision`.
        CPU evaluation, CSV export, and KGE inference all require float32.
        """
        entity_mmap_path = os.path.join(self.attributes.full_storage_path, "entity_embeddings.mmap")
        full_entity_emb = self.raw_model.gather_entity_embeddings_on_rank_zero(
            mmap_path=entity_mmap_path, out_dtype=torch.float32
        )
        state_dict = self._gather_full_state_dict()

        if self.global_rank != 0:
            return None

        concrete_cls = self._concrete_model_class()
        plain_args = dict(self.raw_model.args)
        # Construct with entity allocation deferred, then attach the gathered table below.
        plain_args["fsdp_sharded_entity"] = True
        trained_model = concrete_cls(plain_args)
        plain_args["fsdp_sharded_entity"] = False
        trained_model.defer_large_embeddings = False
        # Dense params stay float32: FSDP's MixedPrecision keeps float32 master weights,
        # so load_state_dict below restores full precision.
        trained_model.entity_embeddings = full_entity_emb
        trained_model.loss_history = list(self.raw_model.loss_history)

        dense_state_dict = self._strip_entity_keys(state_dict)
        self._load_dense_state_dict_for_materialized_model(trained_model, dense_state_dict)
        return trained_model

    def _gather_full_state_dict(self) -> dict:
        """Collect the full (un-sharded) FSDP state dict on rank 0, offloaded to CPU."""
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        fsdp_model = self._unwrap_compiled(self.model)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            return fsdp_model.state_dict()

    @staticmethod
    def _strip_entity_keys(state_dict: dict) -> dict:
        """Drop entity-shard keys that ride along in state_dict() once entity_embeddings
        is attached to raw_model (it's added *after* FSDP wraps raw_model, so FSDP never
        manages it — but a plain nn.Module.state_dict() walk still finds it sitting there).

        Used both for the final materialized model (entity table is wired in separately
        via gather_entity_embeddings_on_rank_zero) and for the periodic dense checkpoint
        (which must stay entity-free — the whole point of sharded checkpointing is to
        never require a full-table gather except once, at the very end of training).
        """
        excluded = ("entity_embeddings.", "_fsdp_adapter.")
        return {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in excluded)}

    @staticmethod
    def _load_dense_state_dict_for_materialized_model(
        model: torch.nn.Module, state_dict: dict
    ) -> None:
        """Load *state_dict* into *model* (strict=False); raise on unexpected mismatches.

        entity_embeddings.weight is expected to be absent (already assigned directly).
        """
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        expected_missing = set()
        if getattr(model, "entity_embeddings", None) is not None:
            expected_missing.add("entity_embeddings.weight")
        unexpected_missing = set(missing) - expected_missing
        if unexpected_missing or unexpected:
            raise RuntimeError(
                f"Unexpected dense state load result. "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    def _concrete_model_class(self):
        """Return the concrete scoring model class (e.g. Keci) from the MRO.

        create_fsdp_sharded_model_class inserts FSDPShardedEntityModel
        before the base scoring class in the MRO.  We want the first BaseKGE
        subclass that is not FSDPShardedEntityModel.
        """
        from dicee.models.base_model import BaseKGE, BaseKGELightning
        from dicee.models.fsdp_models import FSDPShardedEntityModel

        _skip = {FSDPShardedEntityModel, BaseKGE, BaseKGELightning}
        for cls in type(self.raw_model).__mro__:
            if (
                cls not in _skip
                and isinstance(cls, type)
                and issubclass(cls, BaseKGE)
                and not issubclass(cls, FSDPShardedEntityModel)
            ):
                return cls
        raise RuntimeError(
            f"Could not find a concrete model class in MRO of "
            f"{type(self.raw_model).__name__}."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clip_dense_grad_norm(self) -> None:
        fsdp_model = self._unwrap_compiled(self.model)
        if isinstance(fsdp_model, FSDP):
            fsdp_model.clip_grad_norm_(self.gradient_clip_val)
        else:
            dense_params = self._collect_dense_params()
            torch.nn.utils.clip_grad_norm_(dense_params, self.gradient_clip_val)

    @staticmethod
    def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
        if OptimizedModule is not None and isinstance(model, OptimizedModule):
            return model._orig_mod
        return model
