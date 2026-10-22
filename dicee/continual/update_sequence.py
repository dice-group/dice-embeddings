"""Reproducible continual-learning update-sequence generation.

Builds a sequence of cumulative training snapshots

    T_0 subset T_1 subset ... subset T_S

from the *original training split only* of a dataset, such that

    T_i = T_{i-1} union DeltaT_i

and the final snapshot ``T_S`` equals the full original training set
(each training triple used exactly once across ``T_0`` and the
``DeltaT_i``). Validation/test triples are never touched by this module;
callers must pass only ``kg.train_set`` (see
``dicee.continual.README`` for the full data-leakage-prevention protocol).

Three scenarios are supported (Section 6 of the research spec):

``existing_entities``
    Every ``DeltaT_i`` only involves entities already present in
    ``T_{i-1}``. Implemented by first growing ``T_0`` (via a greedy
    entity-covering pass) until it already touches *every* entity that
    appears anywhere in the training set, then randomly partitioning the
    remainder into ``num_steps`` update batches. This is a documented
    fallback whenever ``initial_fraction`` alone would not reach full
    entity coverage: ``T_0`` is grown past the requested fraction, never
    shrunk below it, so the "no new entities in updates" invariant always
    holds.

``new_entities``
    ``T_0`` is a *connected* induced sub-graph covering only a target
    subset of entities (grown via BFS over incident triples from a seed
    entity, using the dataset's real connectivity -- never a random
    disconnected triple sample). All entities outside that core are
    introduced over the ``num_steps`` updates; every newly introduced
    entity is guaranteed at least one connecting triple to an
    already-known entity (or, failing that -- a documented, logged
    edge case -- is deferred to a later step). Leftover old-old triples
    (edges between two already-known entities that were not needed to
    reach the per-step entity quota) are appended to the *final* step so
    that the full training set is still partitioned exactly once.

``mixed``
    Identical construction to ``new_entities``, except leftover old-old
    triples are spread round-robin across *all* steps instead of being
    dumped into the last one, so every update batch contains both
    old-old facts and new-entity-introducing facts.

All randomness is derived from a single ``seed`` argument so sequences
are exactly reproducible, and :func:`generate_update_sequence` can
serialize the produced :class:`UpdateStep` metadata to JSON via
:func:`save_update_sequence` / reload it via :func:`load_update_sequence`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .graph_changes import GraphDiff, GraphSnapshot, diff_snapshots

logger = logging.getLogger(__name__)

VALID_SCENARIOS = ("existing_entities", "new_entities", "mixed")


@dataclass
class UpdateStep:
    """Metadata + payload for a single continual-learning update step."""

    step: int
    added_triples: np.ndarray  # (n_i, 3) int64, DeltaT_i
    cumulative_triples: np.ndarray  # (N_i, 3) int64, T_i
    num_added_triples: int
    num_active_entities: int
    num_new_entities: int
    num_active_relations: int
    num_new_relations: int
    density: float
    new_entity_ids: Tuple[int, ...] = field(default_factory=tuple)
    new_relation_ids: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def num_removed_entities(self) -> int:
        """Always 0 under the cumulative protocol (T_{i-1} subset T_i).

        Deletion/unlearning is explicitly out of scope for this milestone
        (Section 15); the field exists so reports can state this
        invariant explicitly rather than silently omitting it.
        """
        return 0

    @property
    def num_removed_relations(self) -> int:
        """Always 0 under the cumulative protocol; see ``num_removed_entities``."""
        return 0

    def to_metadata_dict(self) -> Dict:
        """JSON-serializable metadata (excludes the large triple arrays)."""
        return {
            "step": self.step,
            "num_added_triples": self.num_added_triples,
            "num_cumulative_triples": int(len(self.cumulative_triples)),
            "num_active_entities": self.num_active_entities,
            "num_new_entities": self.num_new_entities,
            "num_removed_entities": self.num_removed_entities,
            "num_active_relations": self.num_active_relations,
            "num_new_relations": self.num_new_relations,
            "num_removed_relations": self.num_removed_relations,
            "density": self.density,
            "new_entity_ids": list(self.new_entity_ids),
            "new_relation_ids": list(self.new_relation_ids),
        }


def _density(num_triples: int, num_entities: int) -> float:
    if num_entities <= 1:
        return 0.0
    # Undirected simple-graph density proxy: edges / (n choose 2).
    return float(num_triples) / float(num_entities * (num_entities - 1) / 2.0)


def _entity_adjacency(triples: np.ndarray) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for idx, (h, _r, t) in enumerate(triples):
        adj.setdefault(int(h), []).append(idx)
        adj.setdefault(int(t), []).append(idx)
    return adj


def _greedy_entity_cover(triples: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Return indices (subset of ``order``) forming a minimal entity-covering T0.

    Iterates triples in ``order`` and keeps adding triples that cover at
    least one previously-unseen entity, until every entity touched by
    ``triples`` has at least one incident triple in the returned set.
    """
    all_entities = set()
    for h, _r, t in triples:
        all_entities.add(int(h))
        all_entities.add(int(t))

    covered: set = set()
    chosen: List[int] = []
    for idx in order:
        h, _r, t = triples[idx]
        h, t = int(h), int(t)
        if h not in covered or t not in covered:
            chosen.append(int(idx))
            covered.add(h)
            covered.add(t)
        if len(covered) >= len(all_entities):
            break
    if len(covered) < len(all_entities):
        # Fallback: some entities only ever co-occur with already-covered
        # ones in triples we skipped; just scan remaining order once more.
        for idx in order:
            if len(covered) >= len(all_entities):
                break
            h, _r, t = triples[idx]
            h, t = int(h), int(t)
            if idx in chosen:
                continue
            if h not in covered or t not in covered:
                chosen.append(int(idx))
                covered.add(h)
                covered.add(t)
    return np.array(sorted(chosen), dtype=np.int64)


def _connected_core(
    triples: np.ndarray,
    target_num_entities: int,
    rng: np.random.Generator,
) -> Tuple[set, np.ndarray]:
    """Grow a connected induced sub-graph of (approximately) ``target_num_entities``.

    Returns the covered entity-id set and the indices of triples whose
    both endpoints fall inside that set.
    """
    adj = _entity_adjacency(triples)
    all_entities = list(adj.keys())
    if not all_entities:
        return set(), np.zeros((0,), dtype=np.int64)

    target_num_entities = max(1, min(target_num_entities, len(all_entities)))
    covered: set = set()
    remaining_seeds = list(all_entities)
    rng.shuffle(remaining_seeds)

    while len(covered) < target_num_entities and remaining_seeds:
        seed = remaining_seeds.pop()
        if seed in covered:
            continue
        frontier = [seed]
        covered.add(seed)
        while frontier and len(covered) < target_num_entities:
            e = frontier.pop()
            for tidx in adj.get(e, []):
                h, _r, t = triples[tidx]
                h, t = int(h), int(t)
                other = t if h == e else h
                if other not in covered:
                    covered.add(other)
                    frontier.append(other)
                    if len(covered) >= target_num_entities:
                        break
        # loop continues with a new seed (new component) if target not reached

    core_triple_idx = [
        idx for idx, (h, _r, t) in enumerate(triples)
        if int(h) in covered and int(t) in covered
    ]
    return covered, np.array(sorted(core_triple_idx), dtype=np.int64)


def _split_even(items: np.ndarray, num_steps: int, rng: np.random.Generator) -> List[np.ndarray]:
    perm = items.copy()
    rng.shuffle(perm)
    return [np.array(chunk, dtype=np.int64) for chunk in np.array_split(perm, num_steps)]


def _entity_batches(entities: List[int], num_steps: int, rng: np.random.Generator) -> List[List[int]]:
    ents = list(entities)
    rng.shuffle(ents)
    return [list(chunk) for chunk in np.array_split(np.array(ents, dtype=np.int64), num_steps)]


def generate_update_sequence(
    train_triples: np.ndarray,
    *,
    scenario: str = "existing_entities",
    seed: int = 0,
    initial_fraction: float = 0.5,
    num_steps: int = 3,
    update_batch_size: Optional[int] = None,
    min_incident_facts: int = 1,
) -> Tuple[GraphSnapshot, List[UpdateStep]]:
    """Build a reproducible continual-learning update sequence.

    Parameters
    ----------
    train_triples : np.ndarray
        ``(N, 3)`` int array of ``(head_idx, relation_idx, tail_idx)``
        from the *original training split only*.
    scenario : str
        One of ``existing_entities``, ``new_entities``, ``mixed``.
    seed : int
        Random seed; the whole procedure is deterministic given this seed.
    initial_fraction : float
        Target fraction of training triples (``existing_entities``) or
        entities (``new_entities`` / ``mixed``) placed in ``T_0``.
    num_steps : int
        Number of update steps after the initial graph (``>= 1``).
    update_batch_size : Optional[int]
        If given, used as a fixed batch size (in triples) when splitting
        the ``existing_entities`` remainder instead of an even split into
        ``num_steps`` chunks. Ignored for ``new_entities``/``mixed`` where
        batch composition is driven by entity introduction order.
    min_incident_facts : int
        Minimum number of connecting triples required before a new entity
        is considered to have "sufficient incident training facts".

    Returns
    -------
    (initial_snapshot, steps) : Tuple[GraphSnapshot, List[UpdateStep]]
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"scenario must be one of {VALID_SCENARIOS}, got {scenario!r}")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if not (0.0 < initial_fraction < 1.0):
        raise ValueError("initial_fraction must be in (0, 1)")

    train_triples = np.asarray(train_triples, dtype=np.int64)
    if train_triples.ndim != 2 or train_triples.shape[1] != 3:
        raise ValueError("train_triples must have shape (N, 3)")
    unique_rows = {tuple(row) for row in train_triples.tolist()}
    if len(unique_rows) != len(train_triples):
        raise ValueError(
            "train_triples contains exact duplicate rows; the cumulative "
            "snapshot protocol represents each snapshot as a *set* of "
            "triples, so duplicates would silently disappear from the "
            "partition-exactly-once accounting. Deduplicate train_triples "
            "before calling generate_update_sequence()."
        )

    rng = np.random.default_rng(seed)
    n = len(train_triples)
    all_idx = np.arange(n)

    if scenario == "existing_entities":
        order = all_idx.copy()
        rng.shuffle(order)
        t0_idx = _greedy_entity_cover(train_triples, order)
        if len(t0_idx) < int(round(initial_fraction * n)):
            # Grow T0 further (randomly) to respect the requested fraction,
            # without ever removing the entity-covering guarantee.
            target = int(round(initial_fraction * n))
            remaining = np.setdiff1d(order, t0_idx, assume_unique=False)
            extra_needed = max(0, target - len(t0_idx))
            t0_idx = np.concatenate([t0_idx, remaining[:extra_needed]])
        remaining_idx = np.setdiff1d(all_idx, t0_idx, assume_unique=False)

        if update_batch_size:
            chunks = [
                remaining_idx[i:i + update_batch_size]
                for i in range(0, len(remaining_idx), update_batch_size)
            ]
            # Collapse/pad to exactly num_steps batches (merge overflow into last).
            if len(chunks) > num_steps:
                head, tail = chunks[:num_steps - 1], chunks[num_steps - 1:]
                chunks = head + [np.concatenate(tail)]
            while len(chunks) < num_steps:
                chunks.append(np.zeros((0,), dtype=np.int64))
        else:
            chunks = _split_even(remaining_idx, num_steps, rng)

        return _materialize_steps(train_triples, t0_idx, chunks)

    # ---- new_entities / mixed ----
    all_entities = set()
    for h, _r, t in train_triples:
        all_entities.add(int(h))
        all_entities.add(int(t))
    target_core = max(1, int(round(initial_fraction * len(all_entities))))
    core_entities, t0_idx = _connected_core(train_triples, target_core, rng)

    remaining_idx = np.setdiff1d(all_idx, t0_idx, assume_unique=False)
    future_entities = sorted(all_entities - core_entities)
    entity_batches = _entity_batches(future_entities, num_steps, rng)

    known_entities = set(core_entities)
    remaining_set = set(remaining_idx.tolist())
    step_old_old: List[List[int]] = [[] for _ in range(num_steps)]
    step_new: List[List[int]] = [[] for _ in range(num_steps)]

    adj = _entity_adjacency(train_triples)

    for step_i, new_ents in enumerate(entity_batches):
        for ent in new_ents:
            incident = [tidx for tidx in adj.get(ent, []) if tidx in remaining_set]
            connecting = [
                tidx for tidx in incident
                if (int(train_triples[tidx][0]) in known_entities or int(train_triples[tidx][0]) == ent)
                and (int(train_triples[tidx][2]) in known_entities or int(train_triples[tidx][2]) == ent)
            ]
            if len(connecting) < min_incident_facts:
                # Documented fallback: take whatever is incident, even if it
                # (temporarily) connects to an entity scheduled for a later
                # step; this keeps every new entity minimally informative.
                connecting = incident
            if len(connecting) < min_incident_facts:
                logger.warning(
                    "Entity %s has fewer than %s incident training facts available "
                    "(found %s); proceeding with isolated/under-connected entity.",
                    ent, min_incident_facts, len(connecting),
                )
            for tidx in connecting:
                if tidx in remaining_set:
                    step_new[step_i].append(tidx)
                    remaining_set.discard(tidx)
            known_entities.add(ent)

    # Leftover old-old triples (both endpoints already known, not consumed above).
    leftover = sorted(remaining_set)
    if scenario == "new_entities":
        step_old_old[-1].extend(leftover)
    else:  # mixed: spread round-robin
        for i, tidx in enumerate(leftover):
            step_old_old[i % num_steps].append(tidx)

    chunks = [
        np.array(sorted(step_new[i] + step_old_old[i]), dtype=np.int64)
        for i in range(num_steps)
    ]
    return _materialize_steps(train_triples, t0_idx, chunks)


def _materialize_steps(
    train_triples: np.ndarray,
    t0_idx: np.ndarray,
    chunks: List[np.ndarray],
) -> Tuple[GraphSnapshot, List[UpdateStep]]:
    t0 = GraphSnapshot.from_array(train_triples[t0_idx])
    steps: List[UpdateStep] = []
    prev = t0
    seen = set(t0_idx.tolist())
    for i, chunk in enumerate(chunks, start=1):
        overlap = seen.intersection(chunk.tolist())
        if overlap:
            raise RuntimeError(f"Internal error: triple(s) assigned twice at step {i}")
        seen.update(chunk.tolist())
        cur_array = np.concatenate([prev.to_array(), train_triples[chunk]], axis=0) if len(chunk) else prev.to_array()
        current = GraphSnapshot.from_array(cur_array)
        diff: GraphDiff = diff_snapshots(prev, current)
        steps.append(UpdateStep(
            step=i,
            added_triples=diff.added_array(),
            cumulative_triples=current.to_array(),
            num_added_triples=len(diff.added_triples),
            num_active_entities=len(diff.active_entities),
            num_new_entities=len(diff.new_entities),
            num_active_relations=len(diff.active_relations),
            num_new_relations=len(diff.new_relations),
            density=_density(len(current.triples), len(diff.active_entities)),
            new_entity_ids=tuple(sorted(diff.new_entities)),
            new_relation_ids=tuple(sorted(diff.new_relations)),
        ))
        prev = current

    total_used = len(t0_idx) + sum(len(c) for c in chunks)
    if total_used != len(train_triples):
        raise RuntimeError(
            f"Partition does not account for all training triples exactly once: "
            f"used {total_used} of {len(train_triples)}."
        )
    return t0, steps


def save_update_sequence(path: str, initial: GraphSnapshot, steps: List[UpdateStep],
                          config: Optional[Dict] = None) -> None:
    """Persist the update sequence (metadata + triple arrays) for reproducibility."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "t0_triples.npy", initial.to_array())
    metadata = {
        "config": config or {},
        "num_t0_triples": int(len(initial)),
        "steps": [s.to_metadata_dict() for s in steps],
    }
    with open(out_dir / "sequence_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    for s in steps:
        np.save(out_dir / f"step_{s.step}_added.npy", s.added_triples)
        np.save(out_dir / f"step_{s.step}_cumulative.npy", s.cumulative_triples)


def load_update_sequence(path: str) -> Tuple[GraphSnapshot, List[UpdateStep], Dict]:
    """Reload a sequence previously written by :func:`save_update_sequence`."""
    in_dir = Path(path)
    with open(in_dir / "sequence_metadata.json") as f:
        metadata = json.load(f)
    t0 = GraphSnapshot.from_array(np.load(in_dir / "t0_triples.npy"))
    steps = []
    for meta in metadata["steps"]:
        i = meta["step"]
        added = np.load(in_dir / f"step_{i}_added.npy")
        cumulative = np.load(in_dir / f"step_{i}_cumulative.npy")
        steps.append(UpdateStep(
            step=i,
            added_triples=added,
            cumulative_triples=cumulative,
            num_added_triples=meta["num_added_triples"],
            num_active_entities=meta["num_active_entities"],
            num_new_entities=meta["num_new_entities"],
            num_active_relations=meta["num_active_relations"],
            num_new_relations=meta["num_new_relations"],
            density=meta["density"],
            new_entity_ids=tuple(meta["new_entity_ids"]),
            new_relation_ids=tuple(meta["new_relation_ids"]),
        ))
    return t0, steps, metadata["config"]


# ---------------------------------------------------------------------------
# Cached / pre-generated sequences
#
# Regenerating update sequences on every experiment run makes two separate
# runs with "the same" scenario/snapshot-count non-comparable, since the
# random partition into DeltaT_i differs run to run unless the seed, dataset,
# scenario, num_steps, initial_fraction, update_batch_size, and
# min_incident_facts are ALL identical. To remove this randomness as a
# confound, sequences are cached on disk the first time they are built for a
# given (dataset, scenario, num_steps, ...) combination and reused from then
# on, via `dicee/continual/experiments/prepare_dataset.py` (ahead-of-time)
# or transparently via `get_or_create_update_sequence` (lazily, on first use).
# ---------------------------------------------------------------------------

def sequence_cache_dir(
    cache_root: str,
    dataset_dir: str,
    scenario: str,
    num_steps: int,
    initial_fraction: float,
    seed: int,
    update_batch_size: Optional[int] = None,
    min_incident_facts: int = 1,
) -> Path:
    """Deterministic cache path for a given sequence-generation configuration.

    Layout: ``cache_root/<dataset_name>/<scenario>/K<num_steps>_frac<f>_seed<s>[_ubs<b>][_mif<m>]/``
    """
    dataset_name = Path(dataset_dir).resolve().name
    sub = f"K{num_steps}_frac{initial_fraction}_seed{seed}"
    if update_batch_size is not None:
        sub += f"_ubs{update_batch_size}"
    if min_incident_facts != 1:
        sub += f"_mif{min_incident_facts}"
    return Path(cache_root) / dataset_name / scenario / sub


def get_or_create_update_sequence(
    dataset_dir: str,
    train_triples: np.ndarray,
    scenario: str,
    num_steps: int,
    initial_fraction: float,
    seed: int,
    cache_root: str = "Experiments/continual_datasets",
    update_batch_size: Optional[int] = None,
    min_incident_facts: int = 1,
    force_regenerate: bool = False,
) -> Tuple[GraphSnapshot, List[UpdateStep], Path, bool]:
    """Load a cached update sequence if one exists, else generate + cache it.

    Returns ``(t0, steps, cache_dir, was_cached)``. Pass ``force_regenerate=True``
    to rebuild and overwrite an existing cache entry (e.g. after changing the
    partitioning algorithm itself, not just its parameters -- different
    parameters already map to different cache paths).
    """
    cache_dir = sequence_cache_dir(
        cache_root, dataset_dir, scenario, num_steps, initial_fraction, seed,
        update_batch_size, min_incident_facts,
    )
    meta_path = cache_dir / "sequence_metadata.json"

    if meta_path.exists() and not force_regenerate:
        t0, steps, _ = load_update_sequence(str(cache_dir))
        logger.info("Loaded cached update sequence from %s (%s triples in T0, %s steps)",
                    cache_dir, len(t0), len(steps))
        return t0, steps, cache_dir, True

    logger.info("No cached update sequence at %s (or force_regenerate=True); generating...", cache_dir)
    t0, steps = generate_update_sequence(
        train_triples, scenario=scenario, seed=seed, initial_fraction=initial_fraction,
        num_steps=num_steps, update_batch_size=update_batch_size, min_incident_facts=min_incident_facts,
    )
    save_update_sequence(str(cache_dir), t0, steps, config={
        "dataset_dir": dataset_dir, "scenario": scenario, "num_steps": num_steps,
        "initial_fraction": initial_fraction, "seed": seed,
        "update_batch_size": update_batch_size, "min_incident_facts": min_incident_facts,
    })
    return t0, steps, cache_dir, False

