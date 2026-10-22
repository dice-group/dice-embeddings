"""Graph differencing utilities for continual KGE experiments.

Given two *cumulative* training-triple snapshots ``T_{i-1}`` and ``T_i``
(with ``T_{i-1} subset T_i``), this module computes:

* the set of newly added triples ``DeltaT_i = T_i - T_{i-1}``;
* the set of entities / relations active at each snapshot;
* the set of *newly introduced* entities / relations at step ``i``.

All entity/relation identifiers used here are the integer indices produced
by ``dicee.knowledge_graph.KG`` (``entity_to_idx`` / ``relation_to_idx``),
so that they remain stable across snapshots (DICE never reassigns existing
indices -- new entities simply get new, larger indices appended to the
vocabulary, see :mod:`dicee.continual.initialization`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Set, Tuple

import numpy as np

Triple = Tuple[int, int, int]


@dataclass(frozen=True)
class GraphSnapshot:
    """A cumulative set of (head_idx, relation_idx, tail_idx) triples."""

    triples: FrozenSet[Triple]

    @staticmethod
    def from_array(arr: np.ndarray) -> "GraphSnapshot":
        if len(arr) == 0:
            return GraphSnapshot(triples=frozenset())
        return GraphSnapshot(triples=frozenset(tuple(int(x) for x in row) for row in arr))

    def to_array(self) -> np.ndarray:
        if not self.triples:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(sorted(self.triples), dtype=np.int64)

    @property
    def entities(self) -> Set[int]:
        ents: Set[int] = set()
        for h, _r, t in self.triples:
            ents.add(h)
            ents.add(t)
        return ents

    @property
    def relations(self) -> Set[int]:
        return {r for _h, r, _t in self.triples}

    def __len__(self) -> int:
        return len(self.triples)


@dataclass(frozen=True)
class GraphDiff:
    """Result of diffing two cumulative snapshots."""

    added_triples: FrozenSet[Triple]
    active_entities: FrozenSet[int]
    active_relations: FrozenSet[int]
    new_entities: FrozenSet[int]
    new_relations: FrozenSet[int]

    def added_array(self) -> np.ndarray:
        if not self.added_triples:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(sorted(self.added_triples), dtype=np.int64)


def diff_snapshots(previous: GraphSnapshot, current: GraphSnapshot) -> GraphDiff:
    """Compute the added triples and entity/relation novelty between snapshots.

    Parameters
    ----------
    previous : GraphSnapshot
        Snapshot ``T_{i-1}``.
    current : GraphSnapshot
        Snapshot ``T_i``. Must be a superset of ``previous`` (cumulative
        protocol); this is checked and raises ``ValueError`` otherwise.

    Returns
    -------
    GraphDiff
    """
    if not previous.triples.issubset(current.triples):
        missing = previous.triples - current.triples
        raise ValueError(
            "`current` snapshot must be a cumulative superset of `previous` "
            f"(protocol violation: {len(missing)} previous triples missing from current)."
        )

    added = current.triples - previous.triples
    active_entities = current.entities
    active_relations = current.relations
    new_entities = current.entities - previous.entities
    new_relations = current.relations - previous.relations

    return GraphDiff(
        added_triples=frozenset(added),
        active_entities=frozenset(active_entities),
        active_relations=frozenset(active_relations),
        new_entities=frozenset(new_entities),
        new_relations=frozenset(new_relations),
    )
