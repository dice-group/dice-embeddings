"""Worker preparation: a world becomes the tensors of one pretraining update.

``prepare_world`` derives, on the CPU and for every task of a world (in the world's task order), what an update reads
from the task's graph and queries alone, so that a worker process can do it while the model trains:

- the graph: ``GraphKGE.internal_graph`` of the task's context (sorted unique triples with technical inverses, the
  relation id map; every relation of a world is direct);
- the relation graph of the arm's ULTRA options (``ultra.relation_graph``): edge types 0 to 3, then 4 and 5 with pair
  channels, with float64 weights for count weighting;
- the query facts (row 0 of each query group) and the head/tail pattern;
- ``soft``: per query the RB-W target, uniform over the anchor's true answers in the query direction that are not in
  the context (``rbw_answers``);
- ``context_mask``: per query the answers in the context (``context_answers``), which the loss masks out.

Tasks with the same context and sizes share one graph. This is the research worker's preparation
(``statworlds-scale-v1-runtime/fast_pipeline.prepare_world`` with mode ``'rbce'``, targets from
``proofpfn-runtime/pp_mix``) without the observed-path features and role tables, which fed only adapters that stayed
at zero output. ``PreparedWorld.tensors`` names the tensors as the golden references do.
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union

import torch

from ..models.graph_model import GraphKGE
from ..models.ultra import ULTRA, RelationGraph, relation_graph
from .batch import Triple, World, WorldBatch
from .priors import PriorConfig, Slot, world


@dataclass(frozen=True, eq=False)
class PreparedGraph:
    """The graph of a task with its relation graph, as ``ULTRA.set_graph`` would build it (CPU tensors).

    Attributes:
        graph_triples: Sorted unique internal triples with technical inverses, int64 ``[E, 3]``.
        relation_id_map: Internal id of every relation, int64 ``[num_relations]`` (the identity).
        num_entities: Entity count of the world.
        num_relations: Relation count of the world.
        num_direct: Direct relations (``num_relations``; the inverse of ``r`` is ``r + num_direct``).
        relation_graph: The relation graph for the model options of the preparation.
    """

    graph_triples: torch.Tensor
    relation_id_map: torch.Tensor
    num_entities: int
    num_relations: int
    num_direct: int
    relation_graph: RelationGraph

    def attach(self, model: ULTRA) -> None:
        """Attach the graph to a model with the options of the preparation (``GraphKGE.attach_graph``)."""
        model.attach_graph(self.graph_triples, self.relation_id_map, self.num_entities, self.num_relations, self.num_direct,
                           self.relation_graph)


@dataclass(frozen=True, eq=False)
class PreparedTask:
    """The queries of one task with their targets, and the task's graph.

    Attributes:
        graph: The task's graph.
        queries: The true query facts (row 0 of each group), int64 ``[Q, 3]``.
        head_prediction: Bool ``[Q]``; ``True`` where the query asks for the head.
        soft: Float32 ``[Q, num_entities]``; each row is uniform over the query's true answers outside the context.
        context_mask: Bool ``[Q, num_entities]``; the query's answers in the context.
    """

    graph: PreparedGraph
    queries: torch.Tensor
    head_prediction: torch.Tensor
    soft: torch.Tensor
    context_mask: torch.Tensor


@dataclass(frozen=True, eq=False)
class PreparedWorld:
    """The tensors of one update: the prepared tasks of a world in the world's task order.

    Attributes:
        slot: The order slot of the world.
        tasks: Task name to prepared task.
        meta: The world's metadata (``World.meta``).
        relation_weighting: The ULTRA option of the relation graphs.
        pair_channels: The ULTRA option of the relation graphs.
    """

    slot: Slot
    tasks: dict[str, PreparedTask]
    meta: dict[str, Any]
    relation_weighting: str
    pair_channels: str

    @property
    def num_queries(self) -> int:
        """The query groups of the world over all tasks."""
        return sum(len(task.queries) for task in self.tasks.values())

    def tensors(self) -> dict[str, torch.Tensor]:
        """The prepared tensors by golden name, sorted by task: ``'<task>/graph/<name>'``, ``'<task>/soft'``, ``'<task>/context_mask'``.

        The graph names are ``graph_triples``, ``relation_id_map``, ``rel_edge_index``, ``rel_edge_type``,
        ``edge_weight`` (only with count weighting) and the int64 scalars ``num_entities``, ``num_relations`` and
        ``num_direct``.
        """
        out = {}
        for name in sorted(self.tasks):
            task = self.tasks[name]
            graph = task.graph
            edge_index, edge_type, edge_weight = graph.relation_graph
            values = dict(graph_triples=graph.graph_triples, relation_id_map=graph.relation_id_map, rel_edge_index=edge_index,
                          rel_edge_type=edge_type, edge_weight=edge_weight)
            out.update({f'{name}/graph/{key}': value for key, value in values.items() if value is not None})
            for key in ('num_entities', 'num_relations', 'num_direct'):
                out[f'{name}/graph/{key}'] = torch.tensor(int(getattr(graph, key)), dtype=torch.long)
            out[f'{name}/soft'] = task.soft
            out[f'{name}/context_mask'] = task.context_mask
        return out


def answers_by_anchor(facts: frozenset[Triple]) -> tuple[defaultdict[tuple[int, int], set[int]], defaultdict[tuple[int, int], set[int]]]:
    """The answers of every anchor among the facts.

    Args:
        facts: ``(h, r, t)`` facts.

    Returns:
        ``(tails, heads)``: the tails of every ``(h, r)`` and the heads of every ``(r, t)``.
    """
    tails: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    heads: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in facts:
        tails[h, r].add(t)
        heads[r, t].add(h)
    return tails, heads


def rbw_answers(batch: WorldBatch) -> list[list[int]]:
    """Per query group: the true answers of the anchor in the query direction that are not in the context, sorted.

    Args:
        batch: A task of a world.

    Returns:
        One sorted answer list per group; it holds the group's own answer.

    Raises:
        ValueError: When a group's own answer is not among them (a query fact that is false or observed).
    """
    tails, heads = answers_by_anchor(batch.truth)
    context = batch.context
    out = []
    for (h, r, t), head in zip(batch.groups[:, 0].tolist(), batch.head_prediction.tolist()):
        if head:
            answers = sorted(e for e in heads[r, t] if (e, r, t) not in context)
        else:
            answers = sorted(e for e in tails[h, r] if (h, r, e) not in context)
        if (h if head else t) not in answers:
            raise ValueError(f'the answer of query {(h, r, t)} is not a hidden true answer')
        out.append(answers)
    return out


def context_answers(batch: WorldBatch) -> list[list[int]]:
    """Per query group: the answers of the anchor in the query direction that are in the context, sorted.

    Args:
        batch: A task of a world.

    Returns:
        One sorted answer list per group.
    """
    tails, heads = answers_by_anchor(batch.context)
    return [sorted(heads[r, t] if head else tails[h, r]) for (h, r, t), head in zip(batch.groups[:, 0].tolist(), batch.head_prediction.tolist())]


def prepare_graph(context: frozenset[Triple], num_entities: int, num_relations: int, relation_weighting: str = 'binary',
                  pair_channels: str = 'off') -> PreparedGraph:
    """The graph and relation graph of a context, as ``ULTRA(options).set_graph(context)`` builds them.

    Args:
        context: The observed facts.
        num_entities: Entity count.
        num_relations: Relation count (all direct).
        relation_weighting: ``ultra_relation_weighting``.
        pair_channels: ``ultra_pair_channels``.

    Returns:
        The prepared graph (CPU tensors).
    """
    graph, mapping, num_direct = GraphKGE.internal_graph(torch.tensor(sorted(context), dtype=torch.long), num_entities, num_relations)
    return PreparedGraph(graph, mapping, int(num_entities), int(num_relations), num_direct,
                         relation_graph(graph, num_entities, num_direct, relation_weighting, pair_channels))


def prepare_task(batch: WorldBatch, graph: PreparedGraph) -> PreparedTask:
    """The queries of a task with their RB-W targets and context masks.

    Args:
        batch: A task of a world.
        graph: The task's prepared graph.

    Returns:
        The prepared task.
    """
    n = batch.num_entities
    soft = torch.zeros(len(batch.groups), n)
    mask = torch.zeros(len(batch.groups), n, dtype=torch.bool)
    for k, (answers, seen) in enumerate(zip(rbw_answers(batch), context_answers(batch))):
        soft[k, torch.as_tensor(answers)] = 1.0 / len(answers)
        if seen:
            mask[k, torch.as_tensor(seen)] = True
    return PreparedTask(graph, batch.groups[:, 0].clone(), batch.head_prediction.clone(), soft, mask)


def prepare_world(generated: World, slot: Slot, relation_weighting: str = 'binary', pair_channels: str = 'off') -> PreparedWorld:
    """The tensors of an update on a world, for a model with these ULTRA options.

    Args:
        generated: The world (``priors.world``).
        slot: Its order slot.
        relation_weighting: ``ultra_relation_weighting`` of the model.
        pair_channels: ``ultra_pair_channels`` of the model.

    Returns:
        The prepared world, tasks in the world's order.
    """
    graphs: list[tuple[WorldBatch, PreparedGraph]] = []
    tasks = {}
    for name, batch in generated.batch.items():
        graph = next((g for b, g in graphs if (b.num_entities, b.num_relations) == (batch.num_entities, batch.num_relations)
                      and b.context == batch.context), None)
        if graph is None:
            graph = prepare_graph(batch.context, batch.num_entities, batch.num_relations, relation_weighting, pair_channels)
            graphs.append((batch, graph))
        tasks[name] = prepare_task(batch, graph)
    return PreparedWorld(slot, tasks, dict(generated.meta), relation_weighting, pair_channels)


def prepare_slot(seed: int, slot: Slot, prior: PriorConfig, options: Mapping[str, str], real_sources: Union[str, Path, None] = None) -> PreparedWorld:
    """The worker job of pretraining: generate the world of an order slot and prepare it.

    Args:
        seed: The run seed.
        slot: The order slot.
        prior: The prior of the order.
        options: ``ultra_relation_weighting`` and ``ultra_pair_channels`` of the model.
        real_sources: The folder of the real source graphs (``priors.world``).

    Returns:
        The prepared world.
    """
    return prepare_world(world(seed, slot, prior, real_sources), slot, options['ultra_relation_weighting'], options['ultra_pair_channels'])
