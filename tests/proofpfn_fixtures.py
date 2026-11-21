"""Fixtures of the ProofPFN tests in ``tests/`` and ``benchmarks/proofpfn/tests/``: the regression pins and test checkpoints.

``tests/fixtures/proofpfn/pins.json`` holds values that the package computes itself (its README lists them). ``pins()``
reads it. ``python -m tests.proofpfn_fixtures --write-pins`` computes the worlds, orders, mixtures and initial weights
again and writes the file; the other sections (first training updates, the statistics of the evaluation on worlds of
the prior, the gate graphs) are kept as they are. Write the pins again only for an intended change of the worlds, and
say so in the commit: the pins are what keeps the worlds of the thesis runs fixed.

The pinned training updates are bitwise in their recorded environment only: torch 2.5.1 with AVX-512 CPU kernels and 2
threads. ``require_training_pin_environment`` skips these tests elsewhere.
"""
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from dicee.synthetic_kg.batch import World, WorldBatch
from dicee.synthetic_kg.checkpoint import make_checkpoint, save_checkpoint
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.prepare import PreparedWorld, prepare_world
from dicee.synthetic_kg.priors import PriorConfig, Slot, build_order, mixture_bounds, order_sha256, real, world
from dicee.synthetic_kg.variants import SEEDS, VARIANTS, initial_model

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'proofpfn'
PINS = FIXTURES / 'pins.json'
PRIORS: Mapping[str, PriorConfig] = {'synthetic': VARIANTS['PCUltra-S'].prior, 'null': VARIANTS['PCUltra-N'].prior,
                                     'real': VARIANTS['PCUltra-R'].prior, 'mixed': VARIANTS['PCUltra-M'].prior}
"""The priors by the label of the pins: the world keys are ``'<seed>/<label>/<family>/<position>'``."""
MODELS = {'PCUltra': 'PCUltra-S', 'CountULTRA': 'CountULTRA-S', 'ULTRA': 'ULTRA-S'}
"""A variant of each model: the initial weights depend on the seed and the two model options only."""
TASK_FIELDS = ('sampling_seed', 'requested_positives')


@lru_cache(maxsize=None)
def pins() -> dict[str, Any]:
    """The regression pins (``tests/fixtures/proofpfn/pins.json``)."""
    value: dict[str, Any] = json.loads(PINS.read_text())
    return value


def json_digest(value: Any) -> str:
    """SHA-256 of the canonical JSON text of a value (sorted keys, no spaces; numpy scalars as Python numbers)."""
    def plain(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=plain).encode()).hexdigest()


def int_digest(values: Any) -> str:
    """SHA-256 of int64 little-endian values."""
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def content_tensors(batch: Mapping[str, WorldBatch]) -> dict[str, torch.Tensor]:
    """The training content of a world under the keys ``'<task>/<name>'``: per task ``groups`` (int64), ``head_prediction``
    (bool), ``context`` and ``truth`` as sorted int64 ``[n, 3]``, and ``num_entities`` and ``num_relations`` as int64
    scalars."""
    out = {}
    for task, b in batch.items():
        out.update({f'{task}/groups': b.groups.to(torch.long), f'{task}/head_prediction': b.head_prediction.to(torch.bool),
                    f'{task}/context': torch.tensor(sorted(b.context), dtype=torch.long).reshape(-1, 3),
                    f'{task}/truth': torch.tensor(sorted(b.truth), dtype=torch.long).reshape(-1, 3),
                    f'{task}/num_entities': torch.tensor(int(b.num_entities), dtype=torch.long),
                    f'{task}/num_relations': torch.tensor(int(b.num_relations), dtype=torch.long)})
    return out


def content_sha256(batch: Mapping[str, WorldBatch]) -> str:
    """The content digest of a world: the tensor digest of its ``content_tensors``, which does not depend on the task
    order."""
    return tensor_digest(content_tensors(batch))


def record_sha256(generated: World) -> str:
    """The digest of a world's record: its task order, its metadata and the template provenance of its tasks."""
    meta = {k: v for k, v in generated.meta.items() if k != 'recipe_batch'}
    provenance = {task: {k: generated.meta['recipe_batch'][task][k] for k in TASK_FIELDS} for task in generated.batch}
    assert generated.meta['recipe_batch'] == provenance
    return json_digest(dict(tasks=list(generated.batch), meta=meta, recipe_batch=provenance))


def prepared_tensors(prepared: PreparedWorld) -> dict[str, torch.Tensor]:
    """The prepared tensors by name: ``'<task>/graph/<name>'``, ``'<task>/soft'`` and ``'<task>/context_mask'``.

    The graph names are ``graph_triples``, ``relation_id_map``, ``rel_edge_index``, ``rel_edge_type``, ``edge_weight``
    (only with count weighting) and the int64 scalars ``num_entities``, ``num_relations`` and ``num_direct``.
    """
    out = {}
    for name, task in prepared.tasks.items():
        graph = task.graph
        values = dict(graph_triples=graph.graph_triples, relation_id_map=graph.relation_id_map, rel_edge_index=graph.relation_graph.edge_index,
                      rel_edge_type=graph.relation_graph.edge_type, edge_weight=graph.relation_graph.edge_weight,
                      num_entities=torch.tensor(graph.num_entities), num_relations=torch.tensor(graph.num_relations),
                      num_direct=torch.tensor(graph.num_direct))
        out.update({f'{name}/graph/{key}': value for key, value in values.items() if value is not None})
        out.update({f'{name}/soft': task.soft, f'{name}/context_mask': task.context_mask})
    return out


def pinned_world(name: str) -> tuple[Slot, World]:
    """The slot and the generated world of a pinned world name."""
    seed, label = name.split('/')[:2]
    slot = Slot(*pins()['worlds'][name]['slot'])
    return slot, world(int(seed), slot, PRIORS[label])


def world_pin(slot: Slot, generated: World) -> dict[str, Any]:
    """The pin of a world: its slot and the digests of its content, its record and its prepared tensors for PCUltra."""
    options = VARIANTS['PCUltra-S'].model_config()
    prepared = prepare_world(generated, slot, options['ultra_relation_weighting'], options['ultra_pair_channels'])
    return dict(slot=list(slot), content=content_sha256(generated.batch), record=record_sha256(generated),
                prepared=tensor_digest(prepared_tensors(prepared)))


def require_training_pin_environment() -> None:
    """Skip a bitwise training test unless torch is 2.5.1 and runs its AVX-512 CPU kernels."""
    version, capability = torch.__version__.split('+')[0], torch.backends.cpu.get_cpu_capability()
    if (version, capability) != ('2.5.1', 'AVX512'):
        pytest.skip(f'the pinned updates hold for torch 2.5.1 with AVX512 CPU kernels, not torch {version} with {capability}')


def require_real_sources() -> None:
    """Skip a test of real worlds without their source graphs."""
    if not os.environ.get(real.SOURCES_ENV):
        pytest.skip(f'real worlds need their source graphs (set {real.SOURCES_ENV})')


def clean_checkpoints(folder: Path, variant: str) -> str:
    """Checkpoints of a variant's initial weights for the three seeds; returns the template."""
    for seed in SEEDS:
        order = VARIANTS[variant].pinned_order(seed)
        assert order is not None
        checkpoint = make_checkpoint(initial_model(variant, seed).state_dict(), variant=variant, seed=seed, updates=0, weights='raw',
                                     order_sha256=order, provenance={'source': 'test'})
        save_checkpoint(checkpoint, folder / f'{variant}-{seed}.pt')
    return str(folder / f'{variant}-{{seed}}.pt')


# ------------------------------------------------------------------------------------------- writing the pins

def computed_pins() -> dict[str, Any]:
    """The sections of the pins that the package computes: ``worlds``, ``orders``, ``mixtures`` and ``initial_weights``.

    The worlds are those of the present file (their names and slots); the real worlds need their source graphs.
    """
    worlds = {}
    for name in pins()['worlds']:
        slot, generated = pinned_world(name)
        worlds[name] = world_pin(slot, generated)
    orders: dict[str, dict[str, str]] = {}
    mixtures = {}
    for label in ('synthetic', 'real', 'mixed'):
        prior = PRIORS[label]
        orders[label] = {str(seed): order_sha256(build_order(seed, prior)[:400]) for seed in SEEDS}
        mixtures[label] = dict(shares={k: repr(v) for k, v in prior.shares.items()},
                               bounds=[[name, repr(bound)] for name, bound in mixture_bounds(prior)])
    initial = {model: {str(seed): tensor_digest(initial_model(variant, seed).state_dict()) for seed in SEEDS}
               for model, variant in MODELS.items()}
    return dict(worlds=worlds, orders=orders, mixtures=mixtures, initial_weights=initial)


def dumps_pins(value: Mapping[str, Any]) -> str:
    """The text of the pins file: one line per world, the other sections indented."""
    parts = []
    for key, section in value.items():
        if key == 'worlds':
            rows = ',\n'.join(f'  {json.dumps(name)}: {json.dumps(entry)}' for name, entry in section.items())
            parts.append(f' "worlds": {{\n{rows}\n }}')
        else:
            body = json.dumps(section, indent=1).replace('\n', '\n ')
            parts.append(f' {json.dumps(key)}: {body}')
    return '{\n' + ',\n'.join(parts) + '\n}\n'


def write_pins() -> list[str]:
    """Compute the computed sections again and write the pins file.

    Returns:
        The names of the sections that changed.
    """
    from dicee.synthetic_kg.determinism import THREADS, torch_threads
    before = pins()
    with torch_threads(THREADS):
        after = {**before, **computed_pins()}
    PINS.write_text(dumps_pins(after))
    pins.cache_clear()
    return [key for key in after if after[key] != before[key]]


if __name__ == '__main__':
    if sys.argv[1:] != ['--write-pins']:
        sys.exit('usage: python -m tests.proofpfn_fixtures --write-pins   (with PROOFPFN_REAL_SOURCES for the real worlds)')
    changed = write_pins()
    print(f'wrote {PINS}; changed sections: {changed or "none"}')
