"""Digests that pin worlds, weights and files, by the rules of the ProofPFN golden references.

``tensor_digest`` is the research ``weights_digest``: for each key in sorted order, the key (UTF-8), then
``str((tuple(shape), dtype))`` (UTF-8), then the raw bytes of the contiguous CPU tensor. ``json_sha256`` is the JSON
digest. All digests are hex SHA-256.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Union

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


def json_sha256(value: Any) -> str:
    """SHA-256 of ``json.dumps(value, sort_keys=True, separators=(',', ':'))``, floats as ``repr``.

    Args:
        value: A JSON-safe value.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    """SHA-256 of a file's bytes.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
