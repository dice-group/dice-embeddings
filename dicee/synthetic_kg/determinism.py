"""Deterministic execution for pretraining and evaluation, as the research runs used it.

CUDA results of the graph models differ between two calls unless deterministic algorithms are on, and deterministic
cuBLAS needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (or ``:16:8``) in the environment before CUDA starts. The command lines
set it with ``cublas_workspace``; ``require_cublas_workspace`` checks it. CPU results depend on the torch thread count:
the research runs used ``THREADS``.
"""
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

import torch

from ..models._inference import float32_precision_backends

CUBLAS_WORKSPACES = (':4096:8', ':16:8')
CUBLAS_HINT = 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts for deterministic CUDA execution'

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
        raise RuntimeError(CUBLAS_HINT)


def device_problem(device: str) -> Optional[str]:
    """Why a device string names no device of this machine for deterministic work, or ``None`` when it names one.

    Args:
        device: ``'cpu'``, ``'cuda'`` or ``'cuda:<index>'``.

    Returns:
        ``None`` for the CPU and for an existing CUDA device with a deterministic cuBLAS workspace setting; else the
        reason.
    """
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError) as error:
        return f'device {device!r}: {error}'
    if parsed.type == 'cpu':
        return None
    if parsed.type != 'cuda':
        return f'device {device!r}: expected cpu or cuda'
    if not torch.cuda.is_available() or (parsed.index or 0) >= torch.cuda.device_count():
        return f'device {device!r}: no such CUDA device on this machine'
    return CUBLAS_HINT if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_WORKSPACES else None


@contextmanager
def deterministic_execution() -> Iterator[None]:
    """Deterministic algorithms, float32 matmul precision ``'highest'`` and no cuDNN TF32 inside the block.

    A torch with per-backend precision settings gets ``fp32_precision = 'ieee'`` on every backend instead, and the
    block never reads the legacy flags: torch refuses them once cuDNN's convolution and RNN settings differ. Older torch
    (the recorded 2.5.1) has only the legacy flags. The previous settings, warn-only mode included, are restored on exit.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    backends = float32_precision_backends()
    if backends:
        previous: list[Any] = [backend.fp32_precision for backend in backends]
    else:
        previous = [torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32]
    torch.use_deterministic_algorithms(True)
    _set_precision(backends, ['ieee'] * len(backends) if backends else ['highest', False])
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        _set_precision(backends, previous)


def _set_precision(backends: Sequence[Any], values: Sequence[Any]) -> None:
    """Set one ``fp32_precision`` per backend, or without backends the matmul precision and cuDNN's TF32 flag."""
    if backends:
        for backend, value in reversed(tuple(zip(backends, values, strict=True))):
            backend.fp32_precision = value
    else:
        torch.set_float32_matmul_precision(values[0])
        torch.backends.cudnn.allow_tf32 = values[1]
