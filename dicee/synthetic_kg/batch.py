"""The world batch that a pretraining update reads, and its checks.

A world is a mapping from task names to ``WorldBatch`` objects. The research worlds have two tasks, ``'rule'`` and
``'base'``, in this insertion order; an update visits the tasks in mapping order, so generators must keep it. Over
all tasks a world has ``QUERIES`` query groups, ``HEAD_QUERIES`` of them head predictions.
"""
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import torch

from .recipe import HEAD_QUERIES, QUERIES

Triple = tuple[int, int, int]


@dataclass(frozen=True, eq=False)
class WorldBatch:
    """One task of a world: its query groups and its graph (the fields of the research ``RecipeBatch`` that an update reads).

    Attributes:
        groups: Int64 ``[Q, 1 + negatives, 3]``; row 0 of a group is the true query fact, the others are its negatives
            (head negatives for a head query, tail negatives otherwise; all false in ``truth``).
        head_prediction: Bool ``[Q]``; ``True`` where the group asks for the head.
        context: The observed facts, the graph of the task; no query fact is in it.
        truth: Every true fact of the world; it contains ``context`` and every query fact.
        num_entities: Entity count of the world.
        num_relations: Direct relation count of the world.
    """

    groups: torch.Tensor
    head_prediction: torch.Tensor
    context: frozenset[Triple]
    truth: frozenset[Triple]
    num_entities: int
    num_relations: int


class World(NamedTuple):
    """A generated world: its tasks in research order and the generator's metadata.

    Attributes:
        batch: Task name to ``WorldBatch``, in the research task order.
        meta: JSON-safe metadata: ``family`` (the mixture family), ``program_slot`` and the generator's own fields,
            plus ``world_sizes(batch)``.
    """

    batch: dict[str, WorldBatch]
    meta: dict[str, Any]


def check_batch(batch: Mapping[str, WorldBatch]) -> None:
    """Refuse a world that an update must not train on (the research ``check_batches``, plus layout checks).

    Every query fact is true and outside the context, the context is true, no negative is true; over all tasks there
    are ``QUERIES`` groups and ``HEAD_QUERIES`` head predictions.

    Args:
        batch: Task name to ``WorldBatch``.

    Raises:
        ValueError: With the first violated rule.
    """
    if not batch:
        raise ValueError('a world needs at least one task')
    for task, b in batch.items():
        groups, heads = b.groups, b.head_prediction
        if groups.dtype != torch.long or groups.ndim != 3 or groups.shape[2] != 3 or groups.shape[1] < 2:
            raise ValueError(f'{task}: groups must be int64 [queries, 1 + negatives, 3]')
        if heads.dtype != torch.bool or heads.shape != groups.shape[:1]:
            raise ValueError(f'{task}: head_prediction must be bool [queries]')
        if groups.numel() and (groups.min() < 0 or groups[..., (0, 2)].max() >= b.num_entities or groups[..., 1].max() >= b.num_relations):
            raise ValueError(f'{task}: group ids outside the world')
        positives = set(map(tuple, groups[:, 0].tolist()))
        if not positives <= b.truth or positives & b.context:
            raise ValueError(f'{task}: a query fact is false or observed')
        if not b.context <= b.truth:
            raise ValueError(f'{task}: the context holds a false fact')
        if any(tuple(row) in b.truth for row in groups[:, 1:].reshape(-1, 3).tolist()):
            raise ValueError(f'{task}: a negative is a true fact')
    if sum(len(b.groups) for b in batch.values()) != QUERIES:
        raise ValueError(f'a world has {QUERIES} query groups')
    if sum(int(b.head_prediction.sum()) for b in batch.values()) != HEAD_QUERIES:
        raise ValueError(f'a world has {HEAD_QUERIES} head queries')


def world_sizes(batch: Mapping[str, WorldBatch]) -> dict[str, int]:
    """The largest entity count and context size over the tasks (the research ``op_worlds.sizes``).

    Args:
        batch: Task name to ``WorldBatch``.

    Returns:
        ``{'n_entities': ..., 'n_context': ...}``.
    """
    return dict(n_entities=max(b.num_entities for b in batch.values()), n_context=max(len(b.context) for b in batch.values()))
