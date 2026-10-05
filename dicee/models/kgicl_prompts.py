"""KG-ICL prompt graphs: deterministic example sampling and extraction.

Semantics follow nju-websoft/KG-ICL at UPSTREAM_COMMIT (``utils_all.py`` and
``data_loader.py``), with the prompt construction of the released datasets:
an example fact ``(u, q, v)``, every entity ``x`` within ``hops`` of both ``u``
and ``v`` with ``dist(x,u) + dist(x,v) <= hops``, up to ``open_nodes`` further
entities within ``hops`` of both, and every fact among the selected entities.
Upstream draws examples and the additional entities with unseeded NumPy
calls. Here every draw comes from a generator seeded by the prompt seed and
the public ID of the direct relation, so a graph and relation always yield
the same prompt graphs, independently of query batches and call order.
docs/kgicl.md lists the upstream defects that are corrected here.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional

import torch

UNREACHED = 1 << 30


def prompt_generator(*values: int) -> torch.Generator:
    """A CPU generator seeded from integer values, stable across processes and platforms."""
    text = 'kgicl-prompt:' + ':'.join(str(int(v)) for v in values)
    return torch.Generator().manual_seed(int(hashlib.sha256(text.encode()).hexdigest()[:15], 16))


@dataclass(frozen=True, eq=False)
class PromptGraph:
    """One prompt graph in encoder layout, specialized for a query relation.

    ``edge_index`` holds local (head, tail) pairs of every prompt fact and its
    inverse. ``edge_type`` uses internal relation IDs: direct relations in
    ``[0, R)`` and their inverses in ``[R, 2R)``. ``labels`` are the distance
    tokens ``(dist to u, dist to v)`` of the example fact ``(u, q, v)``;
    ``head`` and ``tail`` are the local nodes that receive the head and tail
    tokens. For an inverse query relation they are swapped, as upstream does.
    """

    edge_index: torch.Tensor
    edge_type: torch.Tensor
    labels: torch.Tensor
    head: int
    tail: int

    @property
    def num_nodes(self) -> int:
        return len(self.labels)


@dataclass(frozen=True, eq=False)
class PromptExample:
    """An example fact's extracted prompt graph, before specialization to a query relation.

    ``entities`` maps local nodes to graph entities: node 0 is the example head
    and, unless the example is a self-loop, node 1 its tail. ``triples`` are the
    deduplicated direct facts among these entities in local IDs.
    """
    entities: torch.Tensor  # [n]
    triples: torch.Tensor   # [E, 3]
    labels: torch.Tensor    # [n, 2]
    self_loop: bool


def _csr(keys, values, size):
    order = keys.argsort(stable=True)
    counts = torch.bincount(keys, minlength=size)
    return torch.cat((counts.new_zeros(1), counts.cumsum(0))), values[order]


def _gather(offsets, values, nodes):
    """Concatenated CSR rows of ``nodes``, in node order."""
    starts, counts = offsets[nodes], offsets[nodes + 1] - offsets[nodes]
    total = int(counts.sum())
    if not total:
        return values[:0]
    rows = torch.arange(len(nodes)).repeat_interleave(counts)
    index = starts[rows] + torch.arange(total) - (counts.cumsum(0) - counts)[rows]
    return values[index]


class PromptSampler:
    """Deterministic prompt graphs of an attached graph, kept on the host.

    Args:
        triples: ``[N, 3]`` internal (head, relation, tail) facts including
            inverse facts, as attached by ``GraphKGE.set_graph``.
        num_entities: Entity vocabulary size.
        num_direct: Number of direct relations ``R``; inverse IDs are ``r + R``.
        public_relations: Public ID of each internal direct relation, used for seeding.
        shots: Prompt graphs per query relation (upstream ``shot``).
        hops: Distance bound ``k`` (upstream ``hop``/``path_hop``).
        open_nodes: Additional entities beyond ``dist(x,u) + dist(x,v) <= k``
            (upstream ``enclosing=False``: 50).
        seed: Prompt seed.
    """

    def __init__(self, triples: torch.Tensor, num_entities: int, num_direct: int, public_relations: list,
                 shots: int = 5, hops: int = 3, open_nodes: int = 50, seed: int = 0):
        triples = triples.detach().cpu().long()
        direct = triples[triples[:, 1] < num_direct].unique(dim=0)
        self.num_entities, self.num_direct = num_entities, num_direct
        self.public_relations = list(public_relations)
        self.shots, self.hops, self.open_nodes, self.seed = shots, hops, open_nodes, seed
        # Facts of each relation in canonical (head, tail) order.
        relation_order = (direct[:, 1] * num_entities + direct[:, 0]) * num_entities + direct[:, 2]
        direct = direct[relation_order.argsort()]
        self.relation_offsets = torch.cat((torch.zeros(1, dtype=torch.long),
                                           torch.bincount(direct[:, 1], minlength=num_direct).cumsum(0)))
        self.relation_facts = direct[:, [0, 2]]
        # Facts by head, for induced subgraphs, and an undirected simple adjacency for distances.
        self.head_offsets, self.head_facts = _csr(direct[:, 0], direct, num_entities)
        pairs = torch.cat((direct[:, [0, 2]], direct[:, [2, 0]]))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]].unique(dim=0)
        self.adjacency_offsets, self.adjacency = _csr(pairs[:, 0], pairs[:, 1], num_entities)
        self._examples = {}

    def distances(self, source: int) -> torch.Tensor:
        """Undirected hop distances from ``source`` up to ``hops``; farther entities are ``UNREACHED``."""
        distance = torch.full((self.num_entities,), UNREACHED, dtype=torch.long)
        distance[source] = 0
        frontier = torch.tensor([source])
        for step in range(1, self.hops + 1):
            neighbors = _gather(self.adjacency_offsets, self.adjacency, frontier).unique()
            frontier = neighbors[distance[neighbors] == UNREACHED]
            if not len(frontier):
                break
            distance[frontier] = step
        return distance

    def extract(self, head: int, tail: int, generator: Optional[torch.Generator] = None) -> PromptExample:
        """Prompt graph of the example fact ``(head, ?, tail)``, before query specialization."""
        to_head, to_tail = self.distances(head), self.distances(tail)
        within = (to_head <= self.hops) & (to_tail <= self.hops)
        within[[head, tail]] = False
        candidates = within.nonzero().flatten()
        total = to_head[candidates] + to_tail[candidates]
        close, far = candidates[total <= self.hops], candidates[total > self.hops]
        if len(far) > self.open_nodes:
            far = far[torch.randperm(len(far), generator=generator)[:self.open_nodes]].sort().values
        rest = torch.cat((close, far)).sort().values
        nodes = torch.cat((torch.tensor([head] if head == tail else [head, tail]), rest))
        local = torch.full((self.num_entities,), -1, dtype=torch.long)
        local[nodes] = torch.arange(len(nodes))
        facts = _gather(self.head_offsets, self.head_facts, nodes)
        facts = facts[local[facts[:, 2]] >= 0]
        triples = torch.stack((local[facts[:, 0]], facts[:, 1], local[facts[:, 2]]), 1)
        labels = torch.stack((to_head[nodes], to_tail[nodes]), 1)
        if head == tail:
            labels[0] = torch.tensor([0, 0])
        else:
            labels[0], labels[1] = torch.tensor([0, 1]), torch.tensor([1, 0])
        return PromptExample(nodes, triples, labels, head == tail)

    def examples(self, relation: int) -> list:
        """Up to ``shots`` extracted example facts of a direct relation, in sampling order."""
        if relation not in self._examples:
            public = self.public_relations[relation]
            start, end = int(self.relation_offsets[relation]), int(self.relation_offsets[relation + 1])
            facts = self.relation_facts[start:end]
            order = torch.randperm(len(facts), generator=prompt_generator(self.seed, public))[:self.shots]
            self._examples[relation] = [self.extract(int(facts[i, 0]), int(facts[i, 1]),
                                                     prompt_generator(self.seed, public, slot))
                                        for slot, i in enumerate(order.tolist())]
        return self._examples[relation]

    def prompts(self, relation: int, slots: Optional[list] = None) -> list:
        """Prompt graphs of an internal query relation, one per shot.

        Slot ``j`` uses example ``j mod n`` when only ``n < shots`` examples exist.
        ``slots`` overrides that choice with explicit example indices (training draws).
        """
        direct = relation % self.num_direct
        examples = self.examples(direct)
        if not examples:
            # Upstream's fallback for a relation without facts: two nodes joined by
            # one query-relation edge, without its inverse.
            head, tail = (0, 1) if relation < self.num_direct else (1, 0)
            dummy = PromptGraph(torch.tensor([[0], [1]]), torch.tensor([relation]),
                                torch.tensor([[0, 1], [1, 0]]), head, tail)
            return [dummy] * self.shots
        if slots is None:
            slots = [j % len(examples) for j in range(self.shots)]
        return [specialize(examples[j], relation, self.num_direct) for j in slots]


def specialize(example: PromptExample, relation: int, num_direct: int) -> PromptGraph:
    """Add inverse facts and place the head/tail tokens for a direct or inverse query relation."""
    triples = example.triples
    inverse = torch.stack((triples[:, 2], triples[:, 1] + num_direct, triples[:, 0]), 1)
    both = torch.cat((triples, inverse))
    if example.self_loop:
        head = tail = 0
    else:
        head, tail = (0, 1) if relation < num_direct else (1, 0)
    return PromptGraph(both[:, [0, 2]].T.contiguous(), both[:, 1].contiguous(), example.labels, head, tail)


def answer_distance_rates(facts: torch.Tensor, background: torch.Tensor, num_entities: int,
                          max_distance: int = 9) -> list:
    """Fractions of reachable ``facts`` whose endpoints lie at each undirected distance in ``background``.

    Upstream (``utils.get_distance``) computes these rates from validation facts
    on the validation background graph and suppresses entities first reached at
    a distance whose rate is below 0.02. Unreachable facts are not counted.
    """
    facts, background = torch.as_tensor(facts).long().cpu(), torch.as_tensor(background).long().cpu()
    pairs = torch.cat((background[:, [0, 2]], background[:, [2, 0]]))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]].unique(dim=0)
    offsets, adjacency = _csr(pairs[:, 0], pairs[:, 1], num_entities)
    counts = [0] * (max_distance + 1)
    reachable = 0
    for source in facts[:, 0].unique().tolist():
        targets = facts[facts[:, 0] == source, 2]
        distance = torch.full((num_entities,), UNREACHED, dtype=torch.long)
        distance[source] = 0
        frontier = torch.tensor([source])
        remaining = targets.unique()
        step = 0
        while len(frontier) and (distance[remaining] == UNREACHED).any():
            step += 1
            neighbors = _gather(offsets, adjacency, frontier).unique()
            frontier = neighbors[distance[neighbors] == UNREACHED]
            distance[frontier] = step
        found = distance[targets]
        found = found[found != UNREACHED]
        reachable += len(found)
        for value in found.tolist():
            if value <= max_distance:
                counts[value] += 1
    return [count / reachable if reachable else float('nan') for count in counts]
