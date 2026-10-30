"""The N validity gate of amendment A1: N against S on intact held-out worlds of the prior (registration section 8.4).

The structural null N was trained with the four record families nulled. The gate asks whether N then scores below S
on intact worlds of every nulled family, and close to S on plain worlds. This module ports
``icml-roadmap-20261025/step5/s5_manip.py`` as ``pcu-training-20261026/manipulation.py`` ran it, on the clean worlds,
preparation and scores:

- Worlds: intact (S's prior), seed 42, held-out indices that no training order reaches: the hook indices from rec1
  8,810,000 (S's parameters), attr2 7,810,000, list1 11,810,000 and categorical2 7,700,000; plain: the stage-2 indices
  from 120,000 that are plain worlds (the held-out-fit worlds of ``prior-fix-v1/attr-v1``, whose ``heldout/index.json``
  lists the same slots). At most 100 worlds per family.
- Scores: every model scores every query of a world (tasks sorted by name, tail then head queries, in chunks of 2)
  over all entities through the training-mode path of pretraining (``GraphKGE.forward_query_facts`` in training
  mode on the prepared graph, deterministic execution).
- Metric: per query the reciprocal of the rank of the answer among all entities with the query's other true answers in
  the world's truth and the anchor removed, ties at their mean (``cells.rank_of``); per world the mean.
- Comparison per family (``compare``): MRR(model) = the mean over the family's worlds of the per-world MRR; D = the mean
  over A's seeds minus the mean over B's seeds; SE = sqrt(SE_world^2 + SE_seed^2), with SE_world = the SD over worlds
  of the per-world seed-mean difference / sqrt(worlds) and SE_seed = sqrt(var(A's seed MRRs) / n_A + var(B's) / n_B).
- Reading (``reading``, A = S, B = N): ``n_validity`` is ``'pass'`` when D > 2 SE on every nulled family and |D| <= 2 SE
  on plain.
"""
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch

from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.arms import ARMS, SEEDS
from dicee.synthetic_kg.batch import World
from dicee.synthetic_kg.determinism import deterministic_execution, require_cublas_workspace
from dicee.synthetic_kg.prepare import PreparedWorld, answers_by_anchor, prepare_world
from dicee.synthetic_kg.priors import Slot, world
from dicee.synthetic_kg.streams import stage2_family

from .cells import rank_of
from .systems import ProofPFNFamily, ultra_from_checkpoint

FAMILIES = ('rec1', 'attr2', 'list1', 'categorical2', 'plain')
NULLED = ('rec1', 'attr2', 'list1', 'categorical2')
HELDOUT_FIRST = {'rec1': 8_810_000, 'attr2': 7_810_000, 'list1': 11_810_000, 'categorical2': 7_700_000}
"""The first held-out hook index of each nulled family."""
PLAIN_FIRST, PLAIN_END = 120_000, 130_000
"""The stage-2 range of the plain held-out worlds."""
WORLD_SEED = 42
PER_FAMILY = 100
CHUNK = 2
THREADS = 2


def heldout_slots(family: str, count: int = PER_FAMILY) -> list[Slot]:
    """The first ``count`` held-out worlds of a family.

    Args:
        family: One of ``FAMILIES``.
        count: At most ``PER_FAMILY``.

    Returns:
        Their slots: hook indices from ``HELDOUT_FIRST``, or the plain stage-2 indices from ``PLAIN_FIRST``.
    """
    if family not in FAMILIES or not 1 <= count <= PER_FAMILY:
        raise ValueError(f'the gate has up to {PER_FAMILY} held-out worlds of each of {FAMILIES}')
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
    """The per-query reciprocal ranks of a model on a world, in the gate's query order.

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
        models: Arm (``'a'``, ``'b'``) to model name to ULTRA in training mode, seeds in order.
        families: The families.
        per_family: Worlds per family.

    Returns:
        ``mrr`` (arm to model name to family to the per-world MRRs) and ``queries`` (family to the queries of each world).
    """
    mrr: dict[str, dict[str, dict[str, list[float]]]] = {arm: {name: defaultdict(list) for name in group} for arm, group in models.items()}
    queries: dict[str, list[int]] = defaultdict(list)
    options = sorted({(m.relation_weighting, m.pair_channels) for group in models.values() for m in group.values()})
    threads = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    try:
        with deterministic_execution():
            for family in families:
                for slot in heldout_slots(family, per_family):
                    generated = world(WORLD_SEED, slot, ARMS['S'].prior)
                    prepared = {option: prepare_world(generated, slot, *option) for option in options}
                    for arm, group in models.items():
                        for name, model in group.items():
                            ranks = reciprocal_ranks(model, generated, prepared[model.relation_weighting, model.pair_channels])
                            mrr[arm][name][family].append(float(ranks.mean()))
                    queries[family].append(len(ranks))
    finally:
        torch.set_num_threads(threads)
    return dict(mrr={arm: {name: dict(values) for name, values in group.items()} for arm, group in mrr.items()}, queries=dict(queries))


def compare(mrr: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]], families: Sequence[str] = FAMILIES) -> dict[str, dict[str, Any]]:
    """D = seed mean of A minus seed mean of B per family, with its SE (module docstring).

    Args:
        mrr: ``score``'s ``mrr``: arms ``'a'`` and ``'b'``, models in seed order.
        families: The families.

    Returns:
        Per family: worlds, both MRRs, the seed MRRs, D, SE_world, SE_seed, SE, z and the reading of D against 2 SE.
    """
    out = {}
    for family in families:
        a = np.array([values[family] for values in mrr['a'].values()])   # seeds x worlds
        b = np.array([values[family] for values in mrr['b'].values()])
        seeds_a, seeds_b = a.mean(1), b.mean(1)
        per_world = a.mean(0) - b.mean(0)
        worlds = len(per_world)
        se_world = float(per_world.std(ddof=1) / math.sqrt(worlds)) if worlds > 1 else 0.0
        se_seed = math.sqrt((seeds_a.var(ddof=1) / len(seeds_a) if len(seeds_a) > 1 else 0.0)
                            + (seeds_b.var(ddof=1) / len(seeds_b) if len(seeds_b) > 1 else 0.0))
        se = math.sqrt(se_world ** 2 + se_seed ** 2)
        d = float(seeds_a.mean() - seeds_b.mean())
        text = 'A above B by more than 2 SE' if d > 2 * se else 'within 2 SE' if abs(d) <= 2 * se else 'A below B by more than 2 SE'
        out[family] = dict(worlds=worlds, mrr_a=float(seeds_a.mean()), mrr_b=float(seeds_b.mean()), seeds_a=seeds_a.tolist(),
                           seeds_b=seeds_b.tolist(), D=d, se_world=se_world, se_seed=se_seed, se=se, z=d / se if se > 0 else None, reading=text)
    return out


def reading(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The gate rule of A1 with A = S and B = N.

    Args:
        results: ``compare``'s output with every family of ``FAMILIES``.

    Returns:
        ``n_validity`` (``'pass'`` or ``'fail'``), the condition of each nulled family (D > 2 SE) and of plain (|D| <= 2 SE).
    """
    missing = [family for family in FAMILIES if family not in results]
    if missing:
        raise ValueError(f'the gate needs every family; missing {missing}')
    nulled = {family: bool(results[family]['D'] > 2 * results[family]['se']) for family in NULLED}
    plain = bool(abs(results['plain']['D']) <= 2 * results['plain']['se'])
    return dict(n_validity='pass' if all(nulled.values()) and plain else 'fail', nulled_below=nulled, plain_within=plain)


def gate(s: ProofPFNFamily, n: ProofPFNFamily, families: Sequence[str] = FAMILIES, per_family: int = PER_FAMILY,
         device: Union[str, torch.device] = 'cpu') -> dict[str, Any]:
    """Run the gate: arm A = the family of S, arm B = the family of N.

    Args:
        s: S's family (three seeds).
        n: N's family.
        families: The families (all five for the reading).
        per_family: Held-out worlds per family.
        device: The scoring device.

    Returns:
        The record: the systems with their model hashes, the settings, ``results`` (``compare``), ``queries``,
        ``per_world`` (``score``'s MRRs) and, with every family, the ``reading``.

    Raises:
        ValueError: For families of other arms.
        RuntimeError: For CUDA without a deterministic cuBLAS workspace setting.
    """
    if (s.arm, n.arm) != ('S', 'N'):
        raise ValueError(f'the gate compares S with N, not {s.arm} with {n.arm}')
    if torch.device(device).type == 'cuda':
        require_cublas_workspace()
    models: dict[str, dict[str, ULTRA]] = {}
    for arm, system in (('a', s), ('b', n)):
        system.check()
        models[arm] = {f'{system.key}.seed{seed}': ultra_from_checkpoint(system.members[seed].checkpoint).to(device).train() for seed in SEEDS}
    scored = score(models, families, per_family)
    results = compare(scored['mrr'], families)
    record = dict(systems={arm: dict(key=system.key, arm=system.arm, model_sha256=system.model_sha256()) for arm, system in (('a', s), ('b', n))},
                  device=str(device), world_seed=WORLD_SEED, per_family=per_family, families=list(families), results=results,
                  queries=scored['queries'], per_world=scored['mrr'])
    if set(FAMILIES) <= set(families):
        record['reading'] = reading(results)
    return record
