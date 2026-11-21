"""The ProofPFN checkpoint format and its loader.

Format ``proofpfn-checkpoint-v1`` is a dictionary that ``torch.load(path, weights_only=True)`` reads::

    {'format': 'proofpfn-checkpoint-v1', 'model_name': 'ULTRA', 'model': <state dict>,
     'model_config': {'ultra_dim': 64, 'ultra_num_layers': 6, 'ultra_relation_weighting': ..., 'ultra_pair_channels': ...},
     'variant': 'PCUltra-S', 'seed': 42, 'updates': 24000, 'weights': 'ema', 'real_training_updates': 0,
     'order_sha256': <digest of the run's world order>, 'provenance': {...}}

The state dictionary has ULTRA's 82 keys; with pair channels the relation tables have 6 rows. ``model_config`` is the
variant's (``Variant.model_config``), so ``ULTRA(model_config).load_pretrained(path)`` loads a checkpoint;
``model_from_checkpoint`` (a loaded dictionary) and ``load_model`` (a file) do it with the checks of this module.
``provenance`` is a free dictionary that says where the weights come from.
"""
import io
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Union

import torch

from ..models.ultra import ULTRA
from .variants import ULTRA_DIM, ULTRA_LAYERS, Variant, variant_of

FORMAT = 'proofpfn-checkpoint-v1'
MODEL_NAME = 'ULTRA'
WEIGHTS = ('raw', 'ema')
KEYS = ('format', 'model_name', 'model', 'model_config', 'variant', 'seed', 'updates', 'weights', 'real_training_updates',
        'order_sha256', 'provenance')


class CheckpointError(ValueError):
    """A checkpoint that fails the format or the checks of its variant."""


@lru_cache(maxsize=None)
def _shapes(relation_weighting: str, pair_channels: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Keys and shapes of the state dictionary of ULTRA with these options, in state-dictionary order."""
    with torch.random.fork_rng(devices=[]):
        model = ULTRA(dict(ultra_dim=ULTRA_DIM, ultra_num_layers=ULTRA_LAYERS, ultra_relation_weighting=relation_weighting,
                           ultra_pair_channels=pair_channels))
    return tuple((key, tuple(value.shape)) for key, value in model.state_dict().items())


def _state_shapes(variant: Union[str, Variant]) -> dict[str, tuple[int, ...]]:
    """The keys and shapes of a variant's state dictionary.

    Args:
        variant: The variant or its name.

    Returns:
        Key to shape, in state-dictionary order.
    """
    preset = variant_of(variant)
    return dict(_shapes(preset.relation_weighting, preset.pair_channels))


def check_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Refuse a dictionary that is not a valid ``proofpfn-checkpoint-v1`` checkpoint of its variant.

    Args:
        checkpoint: The checkpoint dictionary.

    Raises:
        CheckpointError: With every violated rule.
    """
    if not isinstance(checkpoint, Mapping):
        raise CheckpointError('a checkpoint is a dictionary')
    problems = []
    if set(checkpoint) != set(KEYS):
        problems.append(f'keys {sorted(checkpoint)} instead of {sorted(KEYS)}')
    if checkpoint.get('format') != FORMAT or checkpoint.get('model_name') != MODEL_NAME:
        problems.append(f"format {checkpoint.get('format')!r} of model {checkpoint.get('model_name')!r}")
    try:
        preset = variant_of(checkpoint.get('variant', ''))
    except KeyError as error:
        raise CheckpointError('; '.join([*problems, str(error)])) from None
    if checkpoint.get('model_config') != preset.model_config():
        problems.append(f"model_config {checkpoint.get('model_config')} is not {preset.name}'s {preset.model_config()}")
    if checkpoint.get('weights') not in WEIGHTS:
        problems.append(f"weights {checkpoint.get('weights')!r}")
    for key in ('seed', 'updates', 'real_training_updates'):
        value = checkpoint.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            problems.append(f'{key} {value!r} is not a nonnegative integer')
    digest = checkpoint.get('order_sha256')
    pinned = preset.pinned_order(checkpoint['seed']) if isinstance(checkpoint.get('seed'), int) else None
    if not isinstance(digest, str) or len(digest) != 64 or (pinned is not None and digest != pinned):
        problems.append(f'order_sha256 {digest!r} is not the order of {preset.name}, seed {checkpoint.get("seed")}')
    if not isinstance(checkpoint.get('provenance'), Mapping):
        problems.append('provenance is not a dictionary')
    state = checkpoint.get('model')
    shapes = _state_shapes(preset)
    if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        problems.append('model is not a state dictionary')
    elif {k: tuple(v.shape) for k, v in state.items()} != shapes:
        problems.append(f"the state dictionary does not have the keys and shapes of {preset.name}")
    elif not all(bool(torch.isfinite(v).all()) for v in state.values()):
        problems.append('the state dictionary holds a value that is not finite')
    if problems:
        raise CheckpointError('; '.join(problems))


def make_checkpoint(state: Mapping[str, torch.Tensor], *, variant: Union[str, Variant], seed: int, updates: int, weights: str,
                    order_sha256: str, provenance: Mapping[str, Any], real_training_updates: int = 0) -> dict[str, Any]:
    """A checked checkpoint dictionary of a variant's model state.

    Args:
        state: The model state, ULTRA's keys with the variant's shapes; tensors are copied to the CPU.
        variant: The variant or its name.
        seed: The run seed.
        updates: The number of training updates behind the weights.
        weights: ``'raw'`` or ``'ema'``.
        order_sha256: The digest of the run's world order.
        provenance: JSON-like record of where the weights come from.
        real_training_updates: Updates on real graphs outside the prior (zero for every variant).

    Returns:
        The checkpoint dictionary.

    Raises:
        CheckpointError: When the result is not a valid checkpoint.
    """
    preset = variant_of(variant)
    checkpoint = dict(format=FORMAT, model_name=MODEL_NAME, model={k: v.detach().cpu().clone() for k, v in state.items()},
                      model_config=preset.model_config(), variant=preset.name, seed=seed, updates=updates, weights=weights,
                      real_training_updates=real_training_updates, order_sha256=order_sha256, provenance=dict(provenance))
    check_checkpoint(checkpoint)
    return checkpoint


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> Path:
    """Write bytes atomically: into a hidden temporary file next to the target, then rename it.

    A reader sees the old file or the new one, never a part. The temporary name carries the process id, so two processes
    never write into the same temporary file. A failed write removes it.

    Args:
        path: The target file; its folder is created.
        data: The bytes.

    Returns:
        The path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def atomic_save(value: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """``torch.save`` a dictionary atomically (``atomic_write_bytes``).

    Args:
        value: The dictionary.
        path: The target file; its folder is created.

    Returns:
        The path.
    """
    buffer = io.BytesIO()
    torch.save(dict(value), buffer)
    return atomic_write_bytes(path, buffer.getvalue())


def save_checkpoint(checkpoint: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Write a checked checkpoint atomically (``atomic_save``).

    Args:
        checkpoint: The checkpoint dictionary.
        path: The target file.

    Returns:
        The path.
    """
    check_checkpoint(checkpoint)
    return atomic_save(checkpoint, path)


def _load(path: Union[str, Path]) -> Any:
    """``torch.load`` of a checkpoint on the CPU with ``weights_only=True``.

    Raises:
        CheckpointError: When the file is missing or unreadable, empty, not a pickle or a broken archive.
    """
    if not Path(path).is_file():
        raise CheckpointError(f'{Path(path).name} does not exist')
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, OSError) as error:
        first = str(error).split('\n', 1)[0]
        raise CheckpointError(f'{Path(path).name} does not load with weights_only=True ({type(error).__name__}: {first})') from None


def read_checkpoint(path: Union[str, Path]) -> dict[str, Any]:
    """Read and check a checkpoint (``weights_only=True``).

    Args:
        path: The checkpoint file.

    Returns:
        The checkpoint dictionary.

    Raises:
        CheckpointError: When the file does not load or is not a valid checkpoint.
    """
    checkpoint: dict[str, Any] = _load(path)
    check_checkpoint(checkpoint)
    return checkpoint


def model_from_checkpoint(checkpoint: Mapping[str, Any], device: Union[str, torch.device] = 'cpu') -> ULTRA:
    """ULTRA with a checkpoint's options and weights (``ULTRA.load_checkpoint``), frozen for inference.

    Args:
        checkpoint: The checkpoint dictionary.
        device: The device of the model.

    Returns:
        The model in evaluation mode without gradients, without a graph; the global random state is unchanged.

    Raises:
        CheckpointError: When the dictionary is not a valid checkpoint.
    """
    check_checkpoint(checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = ULTRA(dict(checkpoint['model_config']))
    model.load_checkpoint(checkpoint)
    model.to(device).eval().requires_grad_(False)
    return model


def load_model(path: Union[str, Path], device: Union[str, torch.device] = 'cpu') -> ULTRA:
    """ULTRA with the options and weights of a checkpoint file, frozen for inference (``model_from_checkpoint``).

    Args:
        path: The checkpoint file, read once with ``weights_only=True``.
        device: The device of the model.

    Returns:
        The model in evaluation mode without gradients, without a graph; the global random state is unchanged.

    Raises:
        CheckpointError: When the file does not load or is not a valid checkpoint.
    """
    return model_from_checkpoint(_load(path), device)
