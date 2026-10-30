"""Access to the ProofPFN golden fixtures of the tests in ``tests/`` and ``benchmarks/proofpfn/tests/``, and the builders
of the test checkpoints and research exports that both use.

Small fixtures are copies in ``tests/fixtures/proofpfn`` (see its README). Large golden files stay outside the
repository: a test that needs one calls ``golden_large`` (or ``golden_root`` for other files of the folder) and skips
unless the environment variable ``PROOFPFN_GOLDEN`` names the proofpfn-golden folder (or its ``golden`` subfolder).

The bitwise training goldens hold in their recorded environment only: the CPU training (G4) for torch 2.5.1 with its
AVX-512 CPU kernels and 2 threads, the CUDA goldens for torch 2.5.1 with CUDA 11.8 on ``GOLDEN_GPU``. Other kernels and
CUDA libraries change the last bits of the backward pass and of AdamW, so ``require_cpu_training_golden_environment``
and ``require_gpu_golden_environment`` skip these tests elsewhere.
"""
import gzip
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from dicee.synthetic_kg.arms import ARMS, initial_model, research_record
from dicee.synthetic_kg.checkpoint import (
    ADAPTER_OUTPUTS,
    CONTEXT_MARKER,
    RESEARCH_ADAPTERS,
    RESEARCH_ARM,
    RESEARCH_FORMAT,
    WEIGHTS,
    make_checkpoint,
    save_checkpoint,
)
from dicee.synthetic_kg.determinism import CUBLAS_WORKSPACES
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.priors import FamilySpec, PriorConfig
from dicee.synthetic_kg.recipe import CONSTANT_UPDATES, DECAY_UPDATES, EMA_DECAY, LEARNING_RATE, MICROBATCH, WEIGHT_DECAY

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'proofpfn'
GOLDEN_GPU = 'NVIDIA GeForce RTX 4070 Ti SUPER'
"""The GPU of every CUDA golden."""
# Research hook modules of the record and real families (module:function 'sample'); s5_null:<generator> is the null.
RESEARCH_GENERATORS = {'pf_prior2f': 'categorical2', 'pf_rec1': 'rec1', 'pf_attr2': 'attr2', 'pf_list1': 'list1', 'pt_real': 'real'}


def research_prior(entry: dict) -> PriorConfig:
    """The prior of a research arm, from the --mix string, order name and hook records of a G1 arm entry."""
    specs = []
    for name, hook in sorted(entry['hooks'].items()):
        params = dict(hook['params'])
        budget = params.pop('swap_budget', None)
        if hook['module'] == 's5_null':
            generator = hook['function']
            assert budget is not None
        else:
            generator = RESEARCH_GENERATORS[hook['module']]
            assert hook['function'] == 'sample' and budget is None
        pairs = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in params.items())
        specs.append(FamilySpec(name, generator, hook['first'], pairs, budget))
    return PriorConfig(entry['mix_string'], entry['order_name'], tuple(specs))


def fixture(name: str) -> Any:
    """A JSON fixture of ``tests/fixtures/proofpfn``, gzip-compressed when the name ends with ``.gz``."""
    data = (FIXTURES / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)


def require_cpu_training_golden_environment() -> None:
    """Skip a bitwise CPU training golden (G4) unless torch is 2.5.1 and runs its AVX-512 CPU kernels."""
    version, capability = torch.__version__.split('+')[0], torch.backends.cpu.get_cpu_capability()
    if (version, capability) != ('2.5.1', 'AVX512'):
        pytest.skip(f'the CPU training goldens hold for torch 2.5.1 with AVX512 CPU kernels, not torch {version} with {capability}')


def require_gpu_golden_environment() -> None:
    """Skip a CUDA golden without CUDA, without a deterministic cuBLAS setting, on another GPU than ``GOLDEN_GPU``, or with
    another torch than 2.5.1 with CUDA 11.8."""
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in CUBLAS_WORKSPACES:
        pytest.skip('deterministic cuBLAS needs CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts')
    if torch.cuda.get_device_name(0) != GOLDEN_GPU:
        pytest.skip(f'the CUDA goldens were recorded on {GOLDEN_GPU}')
    if (torch.__version__.split('+')[0], torch.version.cuda) != ('2.5.1', '11.8'):
        pytest.skip(f'the CUDA goldens hold for torch 2.5.1 with CUDA 11.8, not torch {torch.__version__} with CUDA {torch.version.cuda}')


def golden_root() -> Path:
    """The proofpfn-golden folder of ``PROOFPFN_GOLDEN``, which may name its ``golden`` subfolder; skips without it."""
    value = os.environ.get('PROOFPFN_GOLDEN')
    if not value or not Path(value).is_dir():
        pytest.skip('set PROOFPFN_GOLDEN to the proofpfn-golden folder')
    root = Path(value)
    return root.parent if root.name == 'golden' else root


def golden_large(name: str) -> Path:
    """A file below ``golden/large`` of the proofpfn-golden folder; skips the test when it is not available."""
    path = golden_root() / 'golden' / 'large' / name
    if not path.exists():
        pytest.skip(f'golden file large/{name} not available (set PROOFPFN_GOLDEN to the proofpfn-golden folder)')
    return path


def per_key_digests(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    """The tensor digest of each entry alone, in sorted key order; it names the tensor behind a differing digest."""
    return {key: tensor_digest({key: value}) for key, value in sorted(tensors.items())}


def typed(value: Any) -> Any:
    """JSON with the Python type of every leaf, in the layout of the research dump of the recipe (tuples, ints and floats
    kept apart)."""
    if isinstance(value, dict):
        return {str(k): typed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {'type': type(value).__name__, 'items': [typed(v) for v in value]}
    return {'type': type(value).__name__, 'value': value}


def built_export(arm: str, seed: int = 42) -> dict:
    """A research-format export of an arm's initial weights (``pt_train.export`` layout), with the inert research parts:
    zero adapter output layers, the three markers and, for CountULTRA, zero pair rows of the relation tables."""
    preset = ARMS[arm]
    state = initial_model(arm, seed).state_dict()
    state = {k: torch.cat((v, v.new_zeros(2, v.shape[1]))) if k.endswith('.relation.weight') and len(v) == 4 else v for k, v in state.items()}
    adapters = {k: torch.zeros(3, 2) if k in ADAPTER_OUTPUTS else torch.ones(3, 2) for k in RESEARCH_ADAPTERS}
    markers = {'_count_model_mode': torch.tensor([1, 1]), '_pair_model_mode': torch.tensor([1, int(preset.pair_channels == 'on')]),
               '_context_mode': torch.tensor(CONTEXT_MARKER)}
    weights = {name: {k: v.clone() for k, v in {**state, **adapters, **markers}.items()} for name in WEIGHTS}
    run = dict(research_record(preset), format=RESEARCH_FORMAT, seed=seed, start='random', n1=CONSTANT_UPDATES, n2=DECAY_UPDATES,
               loss='rbw-ce-full', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, ema_decay=EMA_DECAY, microbatch=MICROBATCH,
               nondeterministic=False, model_kind='RoleFreqModel', order_sha256=preset.pinned_order(seed), arm=preset.research_arm)
    config = dict(preset.model_config(), context_mode='joint', context_injection_layer=2)
    return dict(seed=seed, updates=0, real_training_updates=0, model_config=config, model_kind='RoleFreqModel',
                arms={RESEARCH_ARM: weights}, run=run)


def clean_checkpoints(folder: Path, arm: str) -> str:
    """Checkpoints of an arm's initial weights for the three seeds; returns the template."""
    for seed in (42, 43, 44):
        order = ARMS[arm].pinned_order(seed)
        assert order is not None
        checkpoint = make_checkpoint(initial_model(arm, seed).state_dict(), arm=arm, seed=seed, updates=0, weights='raw',
                                     order_sha256=order, provenance={'source': 'test'})
        save_checkpoint(checkpoint, folder / f'{arm}-{seed}.pt')
    return str(folder / f'{arm}-{{seed}}.pt')
