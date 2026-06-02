import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.utils.data import DataLoader

from dicee.abstracts import AbstractTrainer
from dicee.static_funcs_training import make_iterable_verbose

try:
    from torch._dynamo.eval_frame import OptimizedModule
except ImportError:
    OptimizedModule = None

torch.set_float32_matmul_precision('high')


def move_batch_to_device(batch: list, device: torch.device, pin_memory: bool = False):
    """Move a Dice dataloader batch to a device."""

    def move(value):
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        if pin_memory:
            value = value.pin_memory()
        return value.to(device, non_blocking=True)

    if len(batch) == 2:
        x_batch, y_batch = batch
        return move(x_batch), move(y_batch)

    if len(batch) == 3:
        x_batch, y_idx_batch, y_batch = batch
        return (move(x_batch), move(y_idx_batch)), move(y_batch)

    raise ValueError("Unexpected batch shape..")


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
        torch.cuda.set_device(self.device)
        self.model = None
        self.raw_model = None
        self.optimizer = None
        self.optimizer_parameters = None
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

        scoring_technique = getattr(args, "scoring_technique", None)
        self.use_gpu_1vs_sample = (
            scoring_technique == "FSDP1vsSample"
            and not getattr(args, "byte_pair_encoding", False)
        )

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
            self.raw_model.setup_fsdp_sharded_entity_training(
                device=self.device,
                use_cpu_sparse_optimizer=self.use_cpu_sparse_optimizer,
                max_accumulated_sparse_grad_nnz=self.max_accumulated_sparse_grad_nnz,
                async_stream=self.async_stream,
            )
            # Keep the local sparse entity shard outside FSDP; dense parameters are still FSDP-managed.
            self.model = self._wrap_model_with_fsdp(
                ignored_modules=self.raw_model.fsdp_ignored_modules(),
            )
            optimizer_parameters = self.raw_model.fsdp_dense_optimizer_parameters(self.model)
        else:
            self.model = self._wrap_model_with_fsdp()
            optimizer_parameters = self.model.parameters()

        self.loss_func = model.loss
        self.optimizer = model.configure_optimizers(parameters=optimizer_parameters)
        self.optimizer_parameters = self._optimizer_parameters()

        # Optional: Compile model for additional speedup (PyTorch 2.0+)
        if self.use_compile and hasattr(torch, 'compile'):
            if self.local_rank == 0:
                print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode='reduce-overhead')

        num_of_batches = len(self.train_dataset_loader)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

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
            if getattr(self.raw_model, "manual_sharded_entity_training", False):
                self.raw_model.flush_sparse_optimizer()

            avg_epoch_loss = epoch_loss / num_of_batches
            self.loss_history.append(avg_epoch_loss)

            # Epoch callbacks commonly save or inspect parameters. In manual sharded mode the full
            # entity table exists only after final materialization, so only expose loss history here.
            if self.local_rank == self.global_rank == 0:
                self.raw_model.loss_history = list(self.loss_history)
                if not getattr(self.raw_model, "manual_sharded_entity_training", False):
                    for c in self.callbacks:
                        c.on_train_epoch_end(self, self.raw_model)

        # Full-state materialization and final callbacks are rank-0 only; keep other ranks alive until done.
        dist.barrier()
        trained_model = self._materialize_full_state_on_rank_zero()
        if self.global_rank == 0:
            self.on_fit_end(self, trained_model)
        dist.barrier()
        return trained_model

    def _wrap_model_with_fsdp(self, ignored_modules=None) -> FSDP:
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
            ignored_modules=ignored_modules,
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
        self.raw_model.zero_sparse_optimizer_grad()

        with self.ctx:
            output = self.model(source)
            loss = self.loss_func(output, targets)

        batch_loss = loss.item()

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)

        if self.gradient_clip_val is not None:
            # Manual sharded mode clips dense FSDP-managed parameters only. The sparse entity shard
            # is ignored by FSDP and updated through its sparse optimizer path.
            self._clip_grad_norm(self.optimizer_parameters)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.raw_model.step_sparse_optimizer(
            batch_idx=batch_idx,
            sparse_step_interval=self.sparse_step_interval,
        )

        return batch_loss

    def _run_batch_fsdp(self, source: torch.LongTensor, targets: torch.FloatTensor) -> float:
        """Run batch with standard FSDP."""
        # Zero before forward so a failed batch cannot leak stale gradients into the next batch.
        self.optimizer.zero_grad(set_to_none=True)

        with self.ctx:
            output = self.model(source)
            loss = self.loss_func(output, targets)

        batch_loss = loss.item()

        self.scaler.scale(loss).backward()

        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)

        # Optional gradient clipping
        if self.gradient_clip_val is not None:
            self._clip_grad_norm()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return batch_loss

    def extract_input_outputs(self, z: list):
        """Extract inputs and outputs from batch, avoiding redundant pinning."""
        if self.use_gpu_1vs_sample:
            return self._create_gpu_1vs_sample_batch(z)
        return move_batch_to_device(z, self.device, pin_memory=False)

    def _create_gpu_1vs_sample_batch(self, positive_triples: torch.Tensor):
        positive_triples = positive_triples.to(self.device, non_blocking=True)
        source = positive_triples[:, :2]
        positive_tail_idx = positive_triples[:, 2:3]
        size_of_batch = positive_triples.shape[0]
        neg_ratio = int(getattr(self.attributes, "neg_ratio", 1))
        label_smoothing_rate = float(getattr(self.attributes, "label_smoothing_rate", 0.0))
        num_entities = int(self.attributes.num_entities)

        if num_entities <= 1:
            raise ValueError("FSDP1vsSample requires at least two entities for negative sampling.")
        negative_tail_idx = torch.randint(
            1,
            num_entities,
            size=(size_of_batch, neg_ratio),
            device=self.device,
            dtype=torch.long,
        )
        negative_tail_idx = (negative_tail_idx + positive_tail_idx) % num_entities
        target_entity_idx = torch.cat((positive_tail_idx, negative_tail_idx), dim=1)

        positive_labels = torch.ones(
            (size_of_batch, 1),
            device=self.device,
            dtype=torch.float32,
        ) - label_smoothing_rate
        negative_labels = torch.zeros(
            (size_of_batch, neg_ratio),
            device=self.device,
            dtype=torch.float32,
        ) + label_smoothing_rate
        labels = torch.cat((positive_labels, negative_labels), dim=1)
        return (source, target_entity_idx), labels

    def _materialize_full_state_on_rank_zero(self) -> torch.nn.Module:
        """Materialize full model state on rank 0 for checkpointing."""
        if getattr(self.raw_model, "manual_sharded_entity_training", False):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            fsdp_model = self._unwrap_optimized_model(self.model)
            with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = fsdp_model.state_dict()

            trained_model = self.raw_model.materialize_sharded_entity_model_on_rank_zero(
                loss_history=self.loss_history,
            )

            if self.local_rank == self.global_rank == 0:
                excluded_prefixes = self.raw_model.fsdp_state_dict_excluded_prefixes()
                dense_state_dict = {
                    key: value for key, value in state_dict.items()
                    if not key.startswith(excluded_prefixes)
                }
                self._load_dense_state_dict_for_materialized_model(trained_model, dense_state_dict)

            # Nonzero ranks participate in collectives but intentionally do not return a usable full model.
            return trained_model if self.global_rank == 0 else None

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        fsdp_model = self._unwrap_optimized_model(self.model)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = fsdp_model.state_dict()

        trained_model = None
        if self.global_rank == 0:
            trained_model = self.raw_model.__class__(dict(self.raw_model.args))
            trained_model.load_state_dict(state_dict, strict=True)
            trained_model.loss_history = list(self.loss_history)

        # Nonzero ranks participate in collectives but intentionally do not return a usable full model.
        return trained_model

    @staticmethod
    def _unwrap_optimized_model(model: torch.nn.Module) -> torch.nn.Module:
        """Return the original module when torch.compile wraps the model."""
        if OptimizedModule is not None and isinstance(model, OptimizedModule):
            return model._orig_mod
        return model

    def _clip_grad_norm(self, parameters=None) -> None:
        fsdp_model = self._unwrap_optimized_model(self.model)
        if parameters is None and isinstance(fsdp_model, FSDP):
            fsdp_model.clip_grad_norm_(self.gradient_clip_val)
            return
        # Hybrid sparse mode clips only the dense optimizer params; ignored sparse embeddings are handled separately.
        if parameters is None:
            parameters = self.model.parameters()
        torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_val)

    def _optimizer_parameters(self):
        return [
            param
            for param_group in self.optimizer.param_groups
            for param in param_group["params"]
        ]

    @staticmethod
    def _load_dense_state_dict_for_materialized_model(model: torch.nn.Module, state_dict: dict) -> None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        expected_missing = set()
        if getattr(model, "entity_embeddings", None) is not None:
            expected_missing.add("entity_embeddings.weight")
        unexpected_missing = set(missing) - expected_missing
        if unexpected_missing or unexpected:
            raise RuntimeError(
                f"Unexpected dense state load result. missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
