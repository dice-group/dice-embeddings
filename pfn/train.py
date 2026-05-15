"""Meta-train a GraphPFN model on episodic samples from real KGs.

Usage
-----
    # Basic training on a single KG:
    python pfn_train.py --kg-dir KGs/Countries-S1/ --epochs 10000 --save model.pt

    # Train across all KGs under KGs/:
    python pfn_train.py --kg-dir KGs/ --epochs 10000 --save model.pt

    # Resume from an existing checkpoint:
    python pfn_train.py --kg-dir KGs/ --epochs 20000 --save model.pt

    # Use entity-centric sampling or disable support permutation (defaults: random + permute):
    python pfn_train.py --kg-dir KGs/ --epochs 10000 --save model.pt \\
        --support-sampler entity-centric --no-permute-support

    # Larger model (embed-dim 1024, 12 layers, 16 heads):
    python pfn_train.py --kg-dir KGs/ --epochs 10000 --save model.pt \\
        --embed-dim 1024 --num-layers 12 --num-heads 16

    # Stop early once epoch BCE loss drops to 0.02:
    python pfn_train.py --kg-dir KGs/Countries-S1/ --epochs 50000 --save model.pt \\
        --early-stop-loss 0.02

    # Training followed by link-prediction evaluation:
    python pfn_train.py --kg-dir KGs/UMLS/ --epochs 5000 --save model.pt \\
        --eval-train KGs/UMLS/train.txt --eval-test KGs/UMLS/test.txt


    # Multi-GPU DDP training with torchrun (1 visible GPU):
    torchrun --standalone --nproc_per_node=1 pfn_train.py  --ddp --kg-dir KGs --epochs 10 --save model.pt
"""

import argparse
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from pfn.dataset import PFNDataset, RandomSupportPrior, RichSubgraphPrior, _load_real_triples, build_dataset
from pfn.inference import evaluate
from pfn.model import TriplePFN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _estimate_train_step_flops(
    model: TriplePFN,
    loss_fn: nn.Module,
    support: torch.Tensor,
    query: torch.Tensor,
    label: torch.Tensor,
    device: torch.device,
) -> Optional[int]:
    """Estimate total FLOPs for one train step (forward + backward).

    Returns None if profiler FLOP accounting is unavailable on this platform.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=activities, with_flops=True, acc_events=True) as prof:
        logits = model(support, query)
        loss = loss_fn(logits, label)
        loss.backward()
    model.zero_grad(set_to_none=True)

    total_flops = 0
    for evt in prof.key_averages():
        total_flops += int(getattr(evt, "flops", 0) or 0)
    return total_flops if total_flops > 0 else None


def _format_flops(flops: float) -> str:
    """Human-readable FLOPs units."""
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    value = float(flops)
    unit_idx = 0
    while value >= 1000.0 and unit_idx < len(units) - 1:
        value /= 1000.0
        unit_idx += 1
    return f"{value:.3f} {units[unit_idx]}"


def _get_distributed_context(use_ddp: bool) -> tuple[bool, int, int, int]:
    """Resolve torchrun-style distributed rank information from the environment."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = use_ddp or world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return enabled, world_size, rank, local_rank


def _is_rank_zero(rank: int) -> bool:
    """Return True when the current process is the primary worker."""
    return rank == 0


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module for plain or DDP-wrapped models."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def _distributed_mean(value: float, device: torch.device, enabled: bool) -> float:
    """Average a scalar across DDP workers, or return it unchanged."""
    if not enabled:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _broadcast_stop_flag(stop_training: bool, device: torch.device, enabled: bool) -> bool:
    """Broadcast the early-stop decision from rank 0 to all ranks."""
    if not enabled:
        return stop_training
    flag = torch.tensor(1 if stop_training else 0, device=device, dtype=torch.int64)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _cleanup_distributed(enabled: bool) -> None:
    """Destroy the process group when DDP was initialised."""
    if enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    num_epochs: int = 1000,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    kg_dir: str = "KGs/",
    context_size: int = 32,
    num_episodes: int = 100_000,
    negative_ratio: int = 1,
    dataset_cache_dir: str = ".pfn_cache",
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    embed_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    seed: int = 42,
    warmup_ratio: float = 0.05,
    eval_train_file: Optional[str] = None,
    eval_test_file: Optional[str] = None,
    support_sampler: str = "entity-centric",
    permute_support: bool = False,
    early_stop_loss: Optional[float] = None,
    report_flops: bool = True,
    use_ddp: bool = False,
) -> TriplePFN:
    """Meta-train a GraphPFN model on episodic samples from real KGs.

    Parameters
    ----------
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for DataLoader.
    learning_rate : float
        Peak learning rate for AdamW after warmup.
    kg_dir : str
        Root directory containing KGs (each with a ``train.txt`` file).
    context_size : int
        Number of support triples per episode.
    num_episodes : int
        Total number of episodes to pre-generate.
    negative_ratio : int
        Number of negative examples generated per positive example.
    dataset_cache_dir : str
        Directory for pre-computed episodes; regenerated on each run.
    save_path : str or None
        If given, save the final model here; if it already exists, resume.
    device : torch.device or None
        Training device. Defaults to CUDA if available, else CPU.
    embed_dim : int
        Model embedding dimension.
    num_heads : int
        Number of transformer attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate.
    seed : int
        Random seed.
    warmup_ratio : float
        Fraction of total training steps used for linear LR warmup.
    eval_train_file : str or None
        Path to train.txt for post-training evaluation.
    eval_test_file : str or None
        Path to test.txt for post-training evaluation.
    support_sampler : str
        ``"entity-centric"`` or ``"random"``.
    permute_support : bool
        Apply random permutation to support order per episode.
    early_stop_loss : float or None
        Stop once epoch-average BCE loss <= this value.
    use_ddp : bool
        If True, enable torchrun-style DistributedDataParallel across all
        visible GPUs. Also auto-enables when launched with ``WORLD_SIZE>1``.

    Returns
    -------
    TriplePFN
        The trained model (on CPU).
    """
    _set_seed(seed)

    if negative_ratio < 0:
        raise ValueError(f"negative_ratio must be >= 0, got {negative_ratio}.")
    if early_stop_loss is not None and early_stop_loss < 0:
        raise ValueError(f"early_stop_loss must be >= 0, got {early_stop_loss}.")
    ddp_enabled, world_size, rank, local_rank = _get_distributed_context(use_ddp)
    if ddp_enabled and not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs.")

    if ddp_enabled and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        if device is not None and device.type != "cuda":
            raise ValueError("DDP training requires CUDA devices; do not pass a CPU device.")
        device = torch.device("cuda", local_rank)
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = _is_rank_zero(rank)

    try:
        if is_main_process:
            print(f"Loading KGs from {kg_dir!r}...")
        kg_pools = _load_real_triples(kg_dir)
        if not kg_pools:
            raise ValueError(f"No KGs found in {kg_dir!r}")
        if support_sampler not in {"entity-centric", "random"}:
            raise ValueError(
                f"Unknown support_sampler={support_sampler!r}. Expected 'entity-centric' or 'random'."
            )

        if support_sampler == "entity-centric":
            if is_main_process:
                print("Initialising RichSubgraphPrior (entity-centric support sampling)...")
            prior = RichSubgraphPrior(kg_pools=kg_pools)
            if permute_support and is_main_process:
                print("  Note: --permute-support currently applies to random sampler only; ignoring it.")
        else:
            if is_main_process:
                print(
                    "Initialising RandomSupportPrior "
                    f"(random support sampling, permute_support={permute_support})..."
                )
            prior = RandomSupportPrior(kg_pools=kg_pools, permute_support=permute_support)

        if is_main_process:
            print(f"Pre-generating {num_episodes:,} episodes to {dataset_cache_dir!r}...")
            # Use time-based seed for episode generation to ensure different episodes across runs
            episode_seed = seed + int(time.time() * 1000) % 1000000
            _set_seed(episode_seed)
            build_dataset(prior, num_episodes, context_size, dataset_cache_dir, negative_ratio=negative_ratio)
            # Restore original seed for reproducible model training
            _set_seed(seed)
        if ddp_enabled:
            dist.barrier()

        dataset = PFNDataset(dataset_cache_dir, expected_context_size=context_size)
        sampler = DistributedSampler(dataset, shuffle=True) if ddp_enabled else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        num_samples = len(dataset)
        batches_per_epoch = len(dataloader)
        if is_main_process:
            print(
                f"Dataset: {num_samples:,} episodes  |  "
                f"Batch size per process: {batch_size}  |  "
                f"Processes: {world_size}  |  "
                f"Batches per epoch per process: {batches_per_epoch:,}  |  "
                f"Total optimizer updates: {batches_per_epoch * num_epochs:,}"
            )

        model = TriplePFN(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main_process:
            print(f"Model initialized with {num_params:,} trainable parameters.")
        if save_path and os.path.isfile(save_path):
            if is_main_process:
                print(f"Resuming from checkpoint {save_path!r}...")
            ckpt = torch.load(save_path, map_location=device)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            model.load_state_dict(state)
        elif is_main_process:
            print("Starting training from scratch.")

        if ddp_enabled:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        total_steps = num_epochs * len(dataloader)
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        loss_fn = nn.BCEWithLogitsLoss()

        if is_main_process:
            mode_str = "DDP" if ddp_enabled else "single-process"
            print(
                f"Training on {device} for {num_epochs} epochs [{mode_str}]  "
                f"(seed={seed}, lr={learning_rate}, warmup={warmup_steps} steps, total={total_steps} steps)..."
            )

        flops_reported = False
        for epoch in range(num_epochs):
            model.train()
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_batches = 0

            progress_bar = None
            batch_iter = dataloader
            if is_main_process:
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=False)
                batch_iter = progress_bar

            for _batch_idx, (support, query, label) in enumerate(batch_iter, start=1):
                support = support.to(device, non_blocking=device.type == "cuda")
                query = query.to(device, non_blocking=device.type == "cuda")
                label = label.to(device, non_blocking=device.type == "cuda")

                if report_flops and not flops_reported and is_main_process:
                    try:
                        total_flops = _estimate_train_step_flops(
                            model=_unwrap_model(model),
                            loss_fn=loss_fn,
                            support=support,
                            query=query,
                            label=label,
                            device=device,
                        )
                        if total_flops is None:
                            print("FLOPs report: unavailable on this platform/backend.")
                        else:
                            step_human = _format_flops(total_flops)
                            run_total_flops = total_flops * total_steps * world_size
                            run_total_human = _format_flops(run_total_flops)
                            print(
                                "FLOPs report (first mini-batch on rank 0): "
                                f"{total_flops:,} FLOPs per train step (forward+backward) = {step_human}."
                            )
                            print(
                                "Estimated total compute for planned training across all processes: "
                                f"{run_total_flops:,} FLOPs over {total_steps * world_size:,} rank-steps = {run_total_human}."
                            )
                    except Exception as exc:
                        print(f"FLOPs report skipped: profiler error: {exc}")
                    flops_reported = True

                logits = model(support, query)
                loss = loss_fn(logits, label)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                if progress_bar is not None:
                    current_lr = optimizer.param_groups[0]["lr"]
                    running_avg = epoch_loss / num_batches
                    progress_bar.set_postfix(bce=f"{batch_loss:.4f}", avg=f"{running_avg:.4f}", lr=f"{current_lr:.2e}")

            avg_loss_local = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_loss = _distributed_mean(avg_loss_local, device, ddp_enabled)
            if is_main_process:
                print(f"  Epoch {epoch + 1:>5d} / {num_epochs}  |  Avg Loss: {avg_loss:.6f}")

            should_stop = False
            if is_main_process and early_stop_loss is not None and avg_loss <= early_stop_loss:
                print(
                    "Early stopping triggered: "
                    f"avg BCE loss {avg_loss:.6f} <= threshold {early_stop_loss:.6f} "
                    f"at epoch {epoch + 1}."
                )
                should_stop = True
            if _broadcast_stop_flag(should_stop, device, ddp_enabled):
                break

        base_model = _unwrap_model(model)
        if save_path and is_main_process:
            base_model.cpu()
            torch.save(
                {
                    "hparams": {
                        "embed_dim": embed_dim,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "dropout": dropout,
                    },
                    "run_config": {
                        "context_size": context_size,
                        "support_sampler": support_sampler,
                        "permute_support": permute_support,
                        "batch_size": batch_size,
                        "num_episodes": num_episodes,
                        "negative_ratio": negative_ratio,
                        "use_ddp": ddp_enabled,
                        "world_size": world_size,
                    },
                    "state_dict": base_model.state_dict(),
                },
                save_path,
            )
            print(f"Model saved to {save_path!r}")

        if ddp_enabled:
            dist.barrier()

        if eval_train_file and eval_test_file and is_main_process:
            print("\n" + "=" * 60)
            print("Link-Prediction Evaluation")
            print("=" * 60)
            base_model.to(device)
            base_model.eval()
            metrics = evaluate(base_model, train_file=eval_train_file, test_file=eval_test_file, device=device, context_size=context_size)
            base_model.cpu()
            print(f"  MRR:     {metrics['MRR']:.4f}")
            print(f"  MR:      {metrics['MR']:.1f}")
            print(f"  Hits@1:  {metrics['Hits@1']:.4f}")
            print(f"  Hits@3:  {metrics['Hits@3']:.4f}")
            print(f"  Hits@10: {metrics['Hits@10']:.4f}")

        return base_model.cpu()
    finally:
        _cleanup_distributed(ddp_enabled)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Meta-train a GraphPFN model on episodic KG samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of full passes over the episode dataset.")
    parser.add_argument("--batch-size", type=int, default=256, help="Episodes per mini-batch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate (AdamW).")
    parser.add_argument("--kg-dir", type=str, default="KGs", help="Root directory containing KGs.")
    parser.add_argument(
        "--context-size", type=int, default=32,
        help="Number of support triples per episode.",
    )
    parser.add_argument("--num-episodes", type=int, default=10_000, help="Total episodes to pre-generate and cache.")
    parser.add_argument(
        "--negative-ratio", type=int, default=1,
        help="Negative episodes generated per positive episode.",
    )
    parser.add_argument(
        "--support-sampler", type=str, choices=("entity-centric", "random"), default="random",
        help="Support sampling strategy per episode.",
    )
    parser.add_argument(
        "--permute-support", action="store_true", default=True,
        help="Randomly permute support order per episode."
    )
    parser.add_argument(
        "--no-permute-support", dest="permute_support", action="store_false",
        help="Disable support permutation."
    )
    parser.add_argument("--embed-dim", type=int, default=512, help="Model embedding dimension.")
    parser.add_argument("--num-heads", type=int, default=8, help="Transformer attention heads (must divide --embed-dim).")
    parser.add_argument("--num-layers", type=int, default=6, help="Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of steps used for linear LR warmup.")
    parser.add_argument("--save", type=str, default=None, help="Path to save the trained model (.pt).")
    parser.add_argument("--eval-train", type=str, default=None, metavar="TRAIN_FILE", help="train.txt for post-training eval.")
    parser.add_argument("--eval-test", type=str, default=None, metavar="TEST_FILE", help="test.txt for post-training eval.")
    parser.add_argument(
        "--early-stop-loss", type=float, default=None,
        help="Stop early once epoch-average BCE loss <= this value.",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable torchrun-style DistributedDataParallel across all visible GPUs.",
    )

    args = parser.parse_args()
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kg_dir=args.kg_dir,
        context_size=args.context_size,
        num_episodes=args.num_episodes,
        negative_ratio=args.negative_ratio,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        save_path=args.save,
        eval_train_file=args.eval_train,
        eval_test_file=args.eval_test,
        support_sampler=args.support_sampler,
        permute_support=args.permute_support,
        early_stop_loss=args.early_stop_loss,
        use_ddp=args.ddp,
    )


if __name__ == "__main__":
    main()
