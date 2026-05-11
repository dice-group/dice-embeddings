import time
from typing import Callable, Optional, Tuple

import torch


def find_good_batch_size(
    train_loader: torch.utils.data.DataLoader,
    training_step_fn: Callable,
    device,
) -> Tuple[int, Optional[float]]:
    """Find a batch size that uses GPU memory efficiently.

    Progressively increases batch size (with tunable delta) until GPU memory
    exceeds 90% or a CUDA OOM error is raised, then backs off to the last safe
    batch size.  Only supported on CUDA devices; returns the initial batch size
    unchanged on CPU.

    Args:
        train_loader: DataLoader wrapping the training dataset.
        training_step_fn: Callable[[batch], float] — runs one forward+backward
            pass and returns the scalar loss.  Each trainer passes its own
            implementation so this function stays trainer-agnostic.
        device: torch.device (or anything accepted by torch.device()) used to
            monitor GPU memory.

    Returns:
        (batch_size, runtime) where runtime is the wall-clock seconds of the
        last successful batch, or None when batch finding was skipped.
    """
    initial_batch_size = train_loader.batch_size
    training_dataset_size = len(train_loader.dataset)

    if initial_batch_size >= training_dataset_size:
        return training_dataset_size, None

    # Normalise device
    if not isinstance(device, torch.device):
        device = torch.device(device)

    if device.type != "cuda":
        print("Auto batch finding requires a CUDA device — skipping.")
        return initial_batch_size, None

    print(f"Auto batch finding — training data points: {training_dataset_size}")

    def _try_increasing(batch_size: int, delta: int):
        """Increase batch_size until OOM or >90 % GPU memory, return history."""
        history = []
        try:
            while True:
                start = time.time()
                loader = torch.utils.data.DataLoader(
                    train_loader.dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    collate_fn=train_loader.dataset.collate_fn,
                    pin_memory=False,
                    drop_last=False,
                    persistent_workers=False,
                )
                batch_loss = None
                for batch in loader:
                    batch_loss = training_step_fn(batch)
                    break

                free, total = torch.cuda.mem_get_info(device=device)
                pct_used = (total - free) / total
                rt = time.time() - start

                print(
                    f"Batch Loss: {batch_loss:.4f}\t"
                    f"GPU Usage: {pct_used:.3f}\t"
                    f"Runtime: {rt:.3f}s\t"
                    f"Batch Size: {batch_size}"
                )
                history.append((batch_size, rt))

                # Stay below 90 % to avoid the illegal-memory-access bug:
                # https://github.com/pytorch/pytorch/issues/21819
                if pct_used > 0.9:
                    return history, False
                if batch_size < training_dataset_size:
                    batch_size += int(batch_size / delta)
                else:
                    return history, True

        except torch.OutOfMemoryError:
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(
                f"CUDA OOM at batch_size={batch_size} "
                f"({gpu_mem:.2f} GB total, {allocated:.2f} GB allocated)"
            )
            return history, False

    full_history = []
    batch_size = initial_batch_size
    batch_rt = None

    for delta in range(1, 5):
        result, completed = _try_increasing(batch_size, delta=delta)
        full_history.extend(result)

        if completed:
            batch_size, batch_rt = full_history[-1]
        else:
            assert len(full_history) > 2, "GPU memory error on the very first batch"
            batch_size, batch_rt = full_history[-2]
            break

        if batch_size >= training_dataset_size:
            batch_size = training_dataset_size
            break

    return batch_size, batch_rt
