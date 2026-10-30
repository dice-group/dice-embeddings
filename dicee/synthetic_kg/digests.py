"""Digests that pin worlds, weights and files, by the rules of the ProofPFN golden references.

``tensor_digest`` is the research ``weights_digest``: for each key in sorted order, the key (UTF-8), then
``str((tuple(shape), dtype))`` (UTF-8), then the raw bytes of the contiguous CPU tensor. All digests are hex SHA-256.
"""
import hashlib
from pathlib import Path
from typing import Mapping, Union

import torch


def tensor_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 of a mapping of named tensors (the research ``weights_digest``).

    Args:
        tensors: Named tensors, for example a state dictionary.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    """SHA-256 of a file's bytes.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
