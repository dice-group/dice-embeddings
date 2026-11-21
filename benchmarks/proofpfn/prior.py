"""Evaluation on fresh worlds of the synthetic prior: how well pretrained models answer the queries of worlds that no
training order reaches.

The thesis uses it for the structural null: the model trained on the intact prior (PCUltra-S) against the model
trained with four families rewired (PCUltra-N), on intact worlds of each family.

- Worlds: intact worlds of S's prior, seed 42, at held-out indices: the hook indices from rec1 8,810,000 (S's
  parameters), attr2 7,810,000, list1 11,810,000 and categorical2 7,700,000; plain: the stage-2 indices from 120,000
  that are plain worlds. At most 100 worlds per family.
- Scores: every model scores every query of a world (tasks sorted by name, tail then head queries, in chunks of 2)
  over all entities through the training-mode path of pretraining (``GraphKGE.forward_query_facts`` in training
  mode on the prepared graph, deterministic execution).
- Metric: per query the reciprocal of the rank of the answer among all entities with the query's other true answers in
  the world's truth and the anchor removed, ties at their mean (``data.rank_of``); per world the mean.
- Per group of models (the seeds of one name) and family: MRR = the mean over the seeds of the mean over the family's
  worlds of the per-world MRR.
"""
import platform
from collections import defaultdict
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch

from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.batch import World
from dicee.synthetic_kg.determinism import THREADS, deterministic_execution, require_cublas_workspace, torch_threads
from dicee.synthetic_kg.prepare import PreparedWorld, answers_by_anchor, prepare_world
from dicee.synthetic_kg.priors import Slot, world
from dicee.synthetic_kg.streams import stage2_family
from dicee.synthetic_kg.variants import VARIANTS

from .data import rank_of
from .systems import Checkpoint

FAMILIES = ('rec1', 'attr2', 'list1', 'categorical2', 'plain')
HELDOUT_FIRST = {'rec1': 8_810_000, 'attr2': 7_810_000, 'list1': 11_810_000, 'categorical2': 7_700_000}
"""The first held-out hook index of each record family."""
PLAIN_FIRST, PLAIN_END = 120_000, 130_000
"""The stage-2 range of the plain held-out worlds."""
WORLD_SEED = 42
PER_FAMILY = 100
CHUNK = 2


def heldout_slots(family: str, count: int = PER_FAMILY) -> list[Slot]:
    """The first ``count`` held-out worlds of a family.

    Args:
        family: One of ``FAMILIES``.
        count: At most ``PER_FAMILY``.

    Returns:
        Their slots: hook indices from ``HELDOUT_FIRST``, or the plain stage-2 indices from ``PLAIN_FIRST``.
    """
    if family not in FAMILIES or not 1 <= count <= PER_FAMILY:
        raise ValueError(f'there are up to {PER_FAMILY} held-out worlds of each of {FAMILIES}')
    if family != 'plain':
        return [Slot(family, 'hook', HELDOUT_FIRST[family] + k) for k in range(count)]
    out = []
    for index in range(PLAIN_FIRST, PLAIN_END):
        if stage2_family(WORLD_SEED, index) == 'plain':
            out.append(Slot('plain', 's2', index))
            if len(out) == count:
                return out
    raise ValueError(f'fewer than {count} plain worlds in the held-out range')


def reciprocal_ranks(model: ULTRA, generated: World, prepared: PreparedWorld) -> np.ndarray:
    """The per-query reciprocal ranks of a model on a world.

    Args:
        model: ULTRA in training mode, with the options of the preparation.
        generated: The world (its first task's truth is the world's truth).
        prepared: The world prepared for the model.

    Returns:
        One reciprocal rank per query.
    """
    tails, heads = answers_by_anchor(next(iter(generated.batch.values())).truth)
    out = []
    for name in sorted(prepared.tasks):
        task = prepared.tasks[name]
        task.graph.attach(model)
        queries = task.queries.to(model.device)
        for head in (False, True):
            rows_of_direction = (task.head_prediction == head).nonzero().flatten().tolist()
            for start in range(0, len(rows_of_direction), CHUNK):
                rows = rows_of_direction[start:start + CHUNK]
                scores = model.forward_query_facts(queries[rows], head).detach().double().cpu().numpy()
                for j, q in enumerate(rows):
                    h, r, t = task.queries[q].tolist()
                    anchor, target = (t, h) if head else (h, t)
                    banned = ((heads[r, t] if head else tails[h, r]) - {target}) | {anchor}
                    out.append(1.0 / rank_of(scores[j], target, banned))
    return np.array(out)


def score(models: Mapping[str, Mapping[str, ULTRA]], families: Sequence[str] = FAMILIES, per_family: int = PER_FAMILY) -> dict[str, Any]:
    """The per-world MRR of every model on the held-out worlds.

    Args:
        models: Group name to label to ULTRA in training mode, seeds in order.
        families: The families.
        per_family: Worlds per family.

    Returns:
        ``mrr`` (group to label to family to the per-world MRRs) and ``queries`` (family to the queries of each world).
    """
    mrr: dict[str, dict[str, dict[str, list[float]]]] = {group: {label: defaultdict(list) for label in members} for group, members in models.items()}
    queries: dict[str, list[int]] = defaultdict(list)
    options = sorted({(m.relation_weighting, m.pair_channels) for members in models.values() for m in members.values()})
    with torch_threads(THREADS), deterministic_execution():
        for family in families:
            for slot in heldout_slots(family, per_family):
                generated = world(WORLD_SEED, slot, VARIANTS['PCUltra-S'].prior)
                prepared = {option: prepare_world(generated, slot, *option) for option in options}
                for group, members in models.items():
                    for label, model in members.items():
                        ranks = reciprocal_ranks(model, generated, prepared[model.relation_weighting, model.pair_channels])
                        mrr[group][label][family].append(float(ranks.mean()))
                queries[family].append(len(ranks))
    return dict(mrr={group: {label: dict(values) for label, values in members.items()} for group, members in mrr.items()}, queries=dict(queries))


def evaluate(groups: Mapping[str, Sequence[Checkpoint]], families: Sequence[str] = FAMILIES, per_family: int = PER_FAMILY,
             device: Union[str, torch.device] = 'cpu') -> dict[str, Any]:
    """Score groups of checkpoints on the held-out worlds of the prior.

    Args:
        groups: Name to the checkpoints of the name (its seeds, or one checkpoint), in order.
        families: The families.
        per_family: Held-out worlds per family.
        device: The scoring device.

    Returns:
        The record: ``models`` (name to label to the SHA-256 of the file), the settings, ``results`` (name to family to
        ``worlds``, ``MRR`` and ``seeds``, the MRR of each model), ``queries``, ``per_world`` (``score``'s MRRs) and
        ``provenance``.

    Raises:
        ValueError: Without a group, or for an unknown family or world count.
        systems.SystemCheckError: When a checkpoint file is missing or invalid.
        RuntimeError: For CUDA without a deterministic cuBLAS workspace setting.
    """
    if not groups or any(not members for members in groups.values()):
        raise ValueError('give at least one model')
    if not set(families) <= set(FAMILIES) or not 1 <= per_family <= PER_FAMILY:
        raise ValueError(f'there are 1 to {PER_FAMILY} held-out worlds of each of the families {FAMILIES}')
    if torch.device(device).type == 'cuda':
        require_cublas_workspace()
    models: dict[str, dict[str, ULTRA]] = {}
    for name, members in groups.items():
        models[name] = {}
        for system in members:
            model = system.build_model(0, 0)
            assert isinstance(model, ULTRA)
            models[name][system.label] = model.to(device).train()
    scored = score(models, families, per_family)
    results = {name: {family: dict(worlds=len(next(iter(members.values()))[family]),
                                   MRR=float(np.mean([np.mean(values[family]) for values in members.values()])),
                                   seeds=[float(np.mean(values[family])) for values in members.values()]) for family in families}
               for name, members in scored['mrr'].items()}
    record: dict[str, Any] = dict(models={name: {s.label: s.sha256() for s in members} for name, members in groups.items()}, device=str(device),
                                  world_seed=WORLD_SEED, per_family=per_family, families=list(families), results=results, queries=scored['queries'],
                                  per_world=scored['mrr'])
    record['provenance'] = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=str(device),
                                deterministic=True, threads=THREADS)
    return record
