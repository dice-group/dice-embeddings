"""Deterministic execution for pretraining and evaluation, as the research runs used it.

CUDA results of the graph models differ between two calls unless deterministic algorithms are on, and deterministic
cuBLAS needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (or ``:16:8``) in the environment before CUDA starts. The command lines
set it with ``cublas_workspace``; ``require_cublas_workspace`` checks it. CPU results depend on the torch thread count:
the research runs used ``THREADS``.
"""
import os
from contextlib import contextmanager
from typing import Iterator

import torch

CUBLAS_WORKSPACES = (':4096:8', ':16:8')

THREADS = 2
"""The torch thread count of the research runs on the CPU, in training and in evaluation."""


@contextmanager
def torch_threads(threads: int) -> Iterator[None]:
    """Run the block with ``threads`` torch threads; the previous count is restored.

    Args:
        threads: The torch thread count inside the block.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def cublas_workspace() -> Iterator[None]:
    """Run the block with ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` when the variable is unset, and unset it afterwards.

    A set value stays as it is. Call it before CUDA starts.
    """
    unset = 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
    if unset:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = CUBLAS_WORKSPACES[0]
    try:
        yield
    finally:
        if unset:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)


def require_cublas_workspace() -> None:
    """Refuse deterministic CUDA work without a deterministic cuBLAS workspace setting.

    Raises:
        RuntimeError: When ``CUBLAS_WORKSPACE_CONFIG`` is not one of ``CUBLAS_WORKSPACES``.
    """
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_WORKSPACES:
        raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts for deterministic CUDA execution')


@contextmanager
def deterministic_execution() -> Iterator[None]:
    """Deterministic algorithms, float32 matmul precision ``'highest'`` and no cuDNN TF32 inside the block.

    The previous settings, warn-only mode included, are restored on exit.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision, tf32 = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = tf32
