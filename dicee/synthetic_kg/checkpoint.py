"""The ProofPFN checkpoint format, its loader, and the converter of research exports.

Format ``proofpfn-checkpoint-v1`` is a dictionary that ``torch.load(path, weights_only=True)`` reads::

    {'format': 'proofpfn-checkpoint-v1', 'model_name': 'ULTRA', 'model': <state dict>,
     'model_config': {'ultra_dim': 64, 'ultra_num_layers': 6, 'ultra_relation_weighting': ..., 'ultra_pair_channels': ...},
     'arm': 'S', 'seed': 42, 'updates': 24000, 'weights': 'ema', 'real_training_updates': 0,
     'order_sha256': <digest of the run's world order>, 'provenance': {...}}

The state dictionary has ULTRA's 82 keys; with pair channels the relation tables have 6 rows. ``model_config`` is the
arm's (``ArmPreset.model_config``), so ``ULTRA(model_config).load_pretrained(path)`` loads a checkpoint, and
``load_model`` does it with the checks of this module.

``from_research_export`` converts an ``export-<N>.pt`` of the research trainer. Its model holds ULTRA's tensors plus a
context adapter and a role adapter that stayed frozen with zero output layers, and three marker buffers; CountULTRA
exports also keep relation-table rows 4 and 5 at zero. The converter checks that these parts are inert, drops them, and
records the export's SHA-256. Tensors are compared, not file bytes: research exports carry their run record.
"""
import io
import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Union

import torch

from ..models.ultra import ULTRA
from .arms import ULTRA_DIM, ULTRA_LAYERS, ArmPreset, arm_preset, research_record
from .digests import file_sha256
from .recipe import CONSTANT_UPDATES, DECAY_UPDATES, EMA_DECAY, LEARNING_RATE, MICROBATCH, WEIGHT_DECAY

logger = logging.getLogger(__name__)

FORMAT = 'proofpfn-checkpoint-v1'
MODEL_NAME = 'ULTRA'
WEIGHTS = ('raw', 'ema')
KEYS = ('format', 'model_name', 'model', 'model_config', 'arm', 'seed', 'updates', 'weights', 'real_training_updates',
        'order_sha256', 'provenance')

RESEARCH_FORMAT = 'prior-fix-v1'
RESEARCH_ARM = 'decay'
RESEARCH_FIELDS = ('seed', 'updates', 'real_training_updates', 'model_config', 'model_kind', 'arms', 'run')
# Run-record entries that the research trainer wrote only for options other than its defaults; an export of an arm
# has exactly the arm's (arms.research_record).
RESEARCH_OPTIONS = ('schedule', 'target', 'backbone', 'context_adapter', 'pair_channels', 'warm_restart', 'screen', 'update', 'start_state')
RESEARCH_ADAPTERS = ('context_adapter.0.weight', 'context_adapter.0.bias', 'context_adapter.2.weight',
                     'role_adapter.0.weight', 'role_adapter.0.bias', 'role_adapter.2.weight')
ADAPTER_OUTPUTS = ('context_adapter.2.weight', 'role_adapter.2.weight')
CONTEXT_MARKER = [1, 0, 2]   # version 1, context mode 'joint', injection at entity layer 2
RELATION_TABLE_ROWS = 4      # ULTRA's relation-graph edge types; rows 4 and 5 are the pair channels


class CheckpointError(ValueError):
    """A checkpoint, or a research export, that fails the format or its arm's checks."""


@lru_cache(maxsize=None)
def _shapes(relation_weighting: str, pair_channels: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Keys and shapes of the state dictionary of ULTRA with these options, in state-dictionary order."""
    with torch.random.fork_rng(devices=[]):
        model = ULTRA(dict(ultra_dim=ULTRA_DIM, ultra_num_layers=ULTRA_LAYERS, ultra_relation_weighting=relation_weighting,
                           ultra_pair_channels=pair_channels))
    return tuple((key, tuple(value.shape)) for key, value in model.state_dict().items())


def state_shapes(arm: Union[str, ArmPreset]) -> dict[str, tuple[int, ...]]:
    """The keys and shapes of an arm's state dictionary.

    Args:
        arm: The arm or its name.

    Returns:
        Key to shape, in state-dictionary order.
    """
    preset = arm_preset(arm)
    return dict(_shapes(preset.relation_weighting, preset.pair_channels))


def check_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Refuse a dictionary that is not a valid ``proofpfn-checkpoint-v1`` checkpoint of its arm.

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
        preset = arm_preset(checkpoint.get('arm', ''))
    except KeyError as error:
        raise CheckpointError('; '.join([*problems, str(error)])) from None
    if checkpoint.get('model_config') != preset.model_config():
        problems.append(f"model_config {checkpoint.get('model_config')} is not arm {preset.name}'s {preset.model_config()}")
    if checkpoint.get('weights') not in WEIGHTS:
        problems.append(f"weights {checkpoint.get('weights')!r}")
    for key in ('seed', 'updates', 'real_training_updates'):
        value = checkpoint.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            problems.append(f'{key} {value!r} is not a nonnegative integer')
    digest = checkpoint.get('order_sha256')
    pinned = preset.pinned_order(checkpoint['seed']) if isinstance(checkpoint.get('seed'), int) else None
    if not isinstance(digest, str) or len(digest) != 64 or (pinned is not None and digest != pinned):
        problems.append(f'order_sha256 {digest!r} is not the order of arm {preset.name}, seed {checkpoint.get("seed")}')
    if not isinstance(checkpoint.get('provenance'), Mapping):
        problems.append('provenance is not a dictionary')
    state = checkpoint.get('model')
    shapes = state_shapes(preset)
    if not isinstance(state, Mapping) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        problems.append('model is not a state dictionary')
    elif {k: tuple(v.shape) for k, v in state.items()} != shapes:
        problems.append(f"the state dictionary does not have arm {preset.name}'s keys and shapes")
    if problems:
        raise CheckpointError('; '.join(problems))


def make_checkpoint(state: Mapping[str, torch.Tensor], *, arm: Union[str, ArmPreset], seed: int, updates: int, weights: str,
                    order_sha256: str, provenance: Mapping[str, Any], real_training_updates: int = 0) -> dict[str, Any]:
    """A checked checkpoint dictionary of an arm's model state.

    Args:
        state: The model state, ULTRA's keys with the arm's shapes; tensors are copied to the CPU.
        arm: The arm or its name.
        seed: The run seed.
        updates: The number of training updates behind the weights.
        weights: ``'raw'`` or ``'ema'``.
        order_sha256: The digest of the run's world order.
        provenance: JSON-like record of where the weights come from.
        real_training_updates: Updates on real graphs outside the prior (zero for every ported arm).

    Returns:
        The checkpoint dictionary.

    Raises:
        CheckpointError: When the result is not a valid checkpoint.
    """
    preset = arm_preset(arm)
    checkpoint = dict(format=FORMAT, model_name=MODEL_NAME, model={k: v.detach().cpu().clone() for k, v in state.items()},
                      model_config=preset.model_config(), arm=preset.name, seed=seed, updates=updates, weights=weights,
                      real_training_updates=real_training_updates, order_sha256=order_sha256, provenance=dict(provenance))
    check_checkpoint(checkpoint)
    return checkpoint


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> Path:
    """Write bytes atomically: into a hidden temporary file next to the target, then rename it.

    A reader sees the old file or the new one, never a part. The temporary name carries the process id, so two processes
    never write into the same temporary file.

    Args:
        path: The target file; its folder is created.
        data: The bytes.

    Returns:
        The path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)
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


def read_checkpoint(path: Union[str, Path]) -> dict[str, Any]:
    """Read and check a checkpoint (``weights_only=True``).

    Args:
        path: The checkpoint file.

    Returns:
        The checkpoint dictionary.

    Raises:
        CheckpointError: When the file is not a valid checkpoint.
    """
    checkpoint: dict[str, Any] = torch.load(path, map_location='cpu', weights_only=True)
    check_checkpoint(checkpoint)
    return checkpoint


def load_model(path: Union[str, Path], device: Union[str, torch.device] = 'cpu') -> ULTRA:
    """ULTRA with a checkpoint's options and weights, through ``ULTRA.load_pretrained``, frozen for inference.

    Args:
        path: The checkpoint file.
        device: The device of the model.

    Returns:
        The model in evaluation mode without gradients, without a graph; the global random state is unchanged.
    """
    checkpoint = read_checkpoint(path)
    with torch.random.fork_rng(devices=[]):
        model = ULTRA(dict(checkpoint['model_config']))
    model.load_pretrained(path)
    model.to(device).eval().requires_grad_(False)
    return model


def _research_problems(saved: Any, preset: ArmPreset) -> list[str]:
    """Why a loaded research export is not an inert export of the arm; empty when it is."""
    if not isinstance(saved, dict):
        return ['the export is not a dictionary']
    missing = [key for key in RESEARCH_FIELDS if key not in saved]
    if missing:
        return [f'fields {missing} are missing']
    problems = []
    kind = 'ULTRA' if preset.relation_weighting == 'binary' else 'RoleFreqModel'
    if saved['model_kind'] != kind:
        problems.append(f"model_kind {saved['model_kind']!r} is not {kind!r}")
    if saved['real_training_updates'] != 0 or saved.get('parent_step', 0) != 0:
        problems.append('the weights did not train from the random start on the prior only')
    run = saved['run'] if isinstance(saved['run'], dict) else {}
    expected = dict(research_record(preset), format=RESEARCH_FORMAT, seed=saved['seed'], start='random', n1=CONSTANT_UPDATES,
                    n2=DECAY_UPDATES, loss='hard-ce-full' if preset.target == 'hard' else 'rbw-ce-full', lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY, microbatch=MICROBATCH, nondeterministic=False, model_kind=kind)
    for key in dict.fromkeys([*expected, *RESEARCH_OPTIONS]):
        if run.get(key) != expected.get(key):
            problems.append(f'run record {key} is {run.get(key)!r}, arm {preset.name} has {expected.get(key)!r}')
    pinned = preset.pinned_order(saved['seed'])
    if pinned is not None and run.get('order_sha256') != pinned:
        problems.append(f"order digest {run.get('order_sha256')!r} is not arm {preset.name}'s order of seed {saved['seed']}")
    config = saved['model_config'] if isinstance(saved['model_config'], dict) else {}
    options: dict[str, Any] = dict(ultra_dim=ULTRA_DIM, ultra_num_layers=ULTRA_LAYERS)
    if kind == 'RoleFreqModel':
        options.update(ultra_relation_weighting=preset.relation_weighting, ultra_pair_channels=preset.pair_channels,
                       context_mode='joint', context_injection_layer=2)
    problems += [f'model_config {key} is {config.get(key)!r}, not {value!r}' for key, value in options.items() if config.get(key) != value]
    if kind == 'ULTRA' and {'ultra_relation_weighting', 'ultra_pair_channels'} & set(config):
        problems.append('a plain ULTRA export with relation-graph options')
    arms = saved['arms']
    if not isinstance(arms, dict) or set(arms) != {RESEARCH_ARM} or set(arms[RESEARCH_ARM]) != set(WEIGHTS):
        return [*problems, f"arms {list(arms) if isinstance(arms, dict) else arms!r} is not {{'{RESEARCH_ARM}': raw, ema}}"]
    shapes = state_shapes(preset)
    extra = (*RESEARCH_ADAPTERS, '_count_model_mode', '_pair_model_mode', '_context_mode') if kind == 'RoleFreqModel' else ()
    tables = [k for k in shapes if k.startswith('relation_model.layers.') and k.endswith('.relation.weight')]
    markers = {'_count_model_mode': [1, 1], '_pair_model_mode': [1, int(preset.pair_channels == 'on')], '_context_mode': CONTEXT_MARKER}
    for weights in WEIGHTS:
        state = arms[RESEARCH_ARM][weights]
        if not isinstance(state, dict) or set(state) != set(shapes) | set(extra):
            problems.append(f'{weights}: the keys are not the research keys of arm {preset.name}')
            continue
        widened = {k: (RELATION_TABLE_ROWS + 2, shapes[k][1]) if kind == 'RoleFreqModel' and k in tables else shapes[k] for k in shapes}
        problems += [f'{weights}: {k} has shape {tuple(state[k].shape)}' for k in shapes if tuple(state[k].shape) != widened[k]]
        if kind != 'RoleFreqModel':
            continue
        problems += [f'{weights}: marker {k} is {state[k].tolist()}, not {v}' for k, v in markers.items() if state[k].tolist() != v]
        problems += [f'{weights}: the adapter output {k} is not zero' for k in ADAPTER_OUTPUTS if bool(state[k].any())]
        if preset.pair_channels == 'off':
            problems += [f'{weights}: the pair rows of {k} are not zero' for k in tables if bool(state[k][RELATION_TABLE_ROWS:].any())]
    if not problems and kind == 'RoleFreqModel':
        raw, ema = arms[RESEARCH_ARM]['raw'], arms[RESEARCH_ARM]['ema']
        problems += [f'{k} differs between raw and EMA weights: it was trained' for k in extra if not torch.equal(raw[k], ema[k])]
    return problems


def from_research_export(path: Union[str, Path], arm: Union[str, ArmPreset], weights: str = 'ema') -> dict[str, Any]:
    """Convert a research ``export-<N>.pt`` of an arm into a checkpoint, after checking that the dropped parts are inert.

    The export must be the arm's: the same run options (mixture, order, families, target, model kind, ULTRA options),
    the pinned order of its seed, adapters whose output layers are zero and whose raw and EMA tensors are equal (never
    trained), the research marker values and, without pair channels, zero pair rows. The checkpoint keeps ULTRA's
    tensors; the relation tables of CountULTRA lose their zero rows 4 and 5.

    Args:
        path: The export file, read with ``weights_only=True``.
        arm: The arm or its name.
        weights: ``'ema'`` (the evaluated weights) or ``'raw'``.

    Returns:
        The checkpoint dictionary; ``provenance`` records the export's name, SHA-256, research arm and run record.

    Raises:
        CheckpointError: When a check fails.
    """
    preset = arm_preset(arm)
    if weights not in WEIGHTS:
        raise CheckpointError(f'weights {weights!r}; expected one of {WEIGHTS}')
    path = Path(path)
    try:
        saved = torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise CheckpointError(f'{path.name} does not load with weights_only=True: {error}') from None
    problems = _research_problems(saved, preset)
    if problems:
        raise CheckpointError(f'{path.name} is not an inert research export of arm {preset.name}: ' + '; '.join(problems))
    state = saved['arms'][RESEARCH_ARM][weights]
    shapes = state_shapes(preset)
    clean = {k: state[k][:shape[0]] if tuple(state[k].shape) != shape else state[k] for k, shape in shapes.items()}
    run = saved['run']
    provenance = dict(source='research-export', research_export=path.name, research_export_sha256=file_sha256(path),
                      research_arm=run['arm'], research_model_kind=saved['model_kind'], research_run=run,
                      dropped=sorted(set(state) - set(shapes)) + sorted(f'{k}[{shape[0]}:]' for k, shape in shapes.items() if tuple(state[k].shape) != shape))
    checkpoint = make_checkpoint(clean, arm=preset, seed=saved['seed'], updates=saved['updates'], weights=weights,
                                 order_sha256=run['order_sha256'], provenance=provenance, real_training_updates=saved['real_training_updates'])
    logger.info('converted %s: arm %s, seed %d, update %d, %s weights', path.name, preset.name, saved['seed'], saved['updates'], weights)
    return checkpoint
