"""PCUltra: ULTRA with count-weighted relation-graph edges and two pair edge types, the model of ProofPFN.

PCUltra is ``ULTRA`` with ``ultra_relation_weighting='normalized'`` and ``ultra_pair_channels='on'``. It has the same
parameters and the same state dictionary: ULTRA's tensors, with six rows in each relation table of the relation model.
A checkpoint of ``dicee.synthetic_kg`` with these two options loads into it (``--model PCUltra --ultra_checkpoint FILE``).
It changes only ULTRA's relation graph:

1. Count. Each edge of ULTRA's relation graph (types hh, tt, ht and th) carries the weight ``c / sqrt(d_a * d_b)``. Here
   ``c`` counts the entities that have role a for one relation and role b for the other relation, and ``d_a`` and
   ``d_b`` count the entities in these roles.
2. Pair. Two more edge types link two relations whose facts share an entity pair, in the same orientation (type 4) or
   in the reverse orientation (type 5). The weight is ``c / sqrt(n_i * n_j)``, where ``c`` counts the shared pairs and
   ``n`` counts the facts of a relation.

PCUltra differs from the option form of ``ULTRA`` in how it computes. It is the form for inference and for large graphs:

- The relation graph and its weights come from sparse products (``sparse_relation_graph``), so the memory grows with the
  number of relation-graph edges. ``ultra.relation_graph`` fills dense tables of entities times relations. The edges,
  their order and their float64 weights are equal.
- At inference, each layer of the relation model computes its weighted message sum as one sparse matrix product,
  ``A @ stack_t(h * r_t)`` with ``A = [A_0 | ... | A_5]`` of shape N x 6N. No message is stored for each edge, where
  ULTRA's weighted layer stores them all (7.1 million edges times the query batch on AristoV4, for example). The result
  equals that layer's up to float32 rounding. Training, and inference with ``graph_inference_backend='torch'``, use
  ULTRA's weighted layer.
"""
from typing import Any, Optional, cast

import torch
from torch import nn

from ._fused_message import fused_norm_relu
from ._inference import inference_only
from ._relation_weights import INCIDENCE_PAIRS, PAIR_EDGE_TYPES
from .ultra import ULTRA, RelationalConv, RelationGraph, RelNBFNet, build_relation_graph

EDGE_TYPES = len(INCIDENCE_PAIRS) + len(PAIR_EDGE_TYPES)


def _with_inverses(direct: torch.Tensor, num_relations: int) -> torch.Tensor:
    """Direct triples followed by their inverse triples (relation + num_relations)."""
    inverse = direct[:, [2, 1, 0]].clone()
    inverse[:, 1] += num_relations
    return torch.cat((direct, inverse))


def count_weights(direct: torch.Tensor, num_entities: int, num_relations: int, edge_index: torch.Tensor,
                  edge_type: torch.Tensor) -> torch.Tensor:
    """Float64 count weights of ULTRA's relation-graph edges (types 0 to 3).

    The arithmetic per edge is that of ``_relation_weights.relation_count_weights``: ``sqrt(c^2 / (d_a d_b))`` after the
    reduction by the greatest common divisor. The counts come from sparse products, so the memory grows with the number
    of edges and not with entities times relations.

    Args:
        direct: Unique direct triples ``[E, 3]`` on the CPU.
        num_entities: Number of entities.
        num_relations: Number of direct relations.
        edge_index: Relation-graph edges ``[2, M]`` on the CPU (row 0: target relation, row 1: source relation).
        edge_type: Edge types ``[M]`` in 0 to 3.

    Returns:
        The weight of each edge, float64 ``[M]``.

    Raises:
        ValueError: For an edge whose two relations share no entity in the roles of its type.
    """
    nodes = 2 * num_relations
    triples = _with_inverses(direct, num_relations)
    incidence = []
    for column in (0, 2):
        pairs = torch.stack((triples[:, column], triples[:, 1])).unique(dim=1)
        ones = torch.ones(pairs.shape[1], dtype=torch.float64)
        incidence.append(torch.sparse_coo_tensor(pairs, ones, (num_entities, nodes)).coalesce())
    degrees = [torch.bincount(matrix.indices()[1], minlength=nodes) for matrix in incidence]
    keys = edge_index[0] * nodes + edge_index[1]
    counts = torch.zeros(len(edge_type), dtype=torch.long)
    denominator = torch.ones(len(edge_type), dtype=torch.long)
    for kind, (a, b) in enumerate(INCIDENCE_PAIRS):
        # Float64 sums of unit entries are exact: every count is an integer of at most num_entities < 2^53.
        product = torch.sparse.mm(incidence[a].transpose(0, 1), incidence[b]).coalesce()
        found = product.indices()[0] * nodes + product.indices()[1]   # sorted, as the product is coalesced
        selected = (edge_type == kind).nonzero().flatten()
        position = torch.searchsorted(found, keys[selected]).clamp_max(max(len(found) - 1, 0))
        if len(selected) and not torch.equal(found[position], keys[selected]):
            raise ValueError(f'Relation-graph edges of type {kind} without a shared entity')
        counts[selected] = product.values()[position].long()
        rows, cols = edge_index[0, selected], edge_index[1, selected]
        denominator[selected] = (degrees[a][rows] * degrees[b][cols]).clamp_min(1)
    numerator = counts.square()
    common = torch.gcd(numerator, denominator)
    weights: torch.Tensor = ((numerator // common).double() / (denominator // common).double()).sqrt()
    return weights


def pair_edges(direct: torch.Tensor, num_entities: int, num_relations: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pair edges (types 4 and 5) with their float64 weights, in the order of ``ultra.relation_graph``.

    Type 4 links relations i and j when facts (h, i, t) and (h, j, t) exist, type 5 when (h, i, t) and (t, j, h)
    exist. Matches of a relation with itself (type 4) and with its own inverse (type 5) are left out. The weight is
    ``c / sqrt(n_i n_j)`` as in ``_relation_weights.pair_channel_weights``.

    Args:
        direct: Unique direct triples ``[E, 3]`` on the CPU.
        num_entities: Number of entities.
        num_relations: Number of direct relations.

    Returns:
        Edges ``[2, P]`` (row 0: target relation, row 1: source relation), edge types ``[P]`` and float64 weights ``[P]``.
    """
    nodes = 2 * num_relations
    triples = _with_inverses(direct, num_relations)
    if not len(triples):
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float64)
    degree = torch.bincount(triples[:, 1], minlength=nodes)
    keys, ids = torch.unique(triples[:, 0] * num_entities + triples[:, 2], sorted=True, return_inverse=True)
    # Every reversed pair exists, because each fact has its inverse fact.
    reverse = torch.searchsorted(keys, triples[:, 2] * num_entities + triples[:, 0])
    ones = torch.ones(len(triples), dtype=torch.float64)
    same, flipped = (torch.sparse_coo_tensor(torch.stack((row, triples[:, 1])), ones, (len(keys), nodes)).coalesce()
                     for row in (ids, reverse))
    relation = torch.arange(nodes)
    edges, kinds, weights = [], [], []
    for kind, (right, excluded) in zip(PAIR_EDGE_TYPES, ((same, relation), (flipped, (relation + num_relations) % nodes))):
        product = torch.sparse.mm(same.transpose(0, 1), right).coalesce()
        (rows, cols), counts = product.indices(), product.values()
        keep = (counts != 0) & (cols != excluded[rows])
        rows, cols, counts = rows[keep], cols[keep], counts[keep]
        edges.append(torch.stack((rows, cols)))
        kinds.append(torch.full((len(rows),), kind, dtype=torch.long))
        weights.append(counts / (degree[rows] * degree[cols]).clamp_min(1).double().sqrt())
    return torch.cat(edges, 1), torch.cat(kinds), torch.cat(weights)


def sparse_relation_graph(graph_triples: torch.Tensor, num_entities: int, num_direct: int) -> RelationGraph:
    """The relation graph of PCUltra from sparse products.

    It equals ``ultra.relation_graph(graph_triples, num_entities, num_direct, 'normalized', 'on')``: the same edges in
    the same order, the same edge types and the same float64 weights.

    Args:
        graph_triples: Internal graph triples with technical inverses (``GraphKGE.internal_graph``).
        num_entities: Number of entities.
        num_direct: Number of direct relations.

    Returns:
        The relation graph on the device of ``graph_triples``, with the weights in float64.
    """
    edge_index, edge_type = build_relation_graph(graph_triples[:, [0, 2]].T, graph_triples[:, 1], num_entities, 2 * num_direct)
    device = edge_index.device
    direct = graph_triples[graph_triples[:, 1] < num_direct].detach().cpu()
    weight = count_weights(direct, num_entities, num_direct, edge_index.cpu(), edge_type.cpu())
    pair_index, pair_type, pair_weight = pair_edges(direct, num_entities, num_direct)
    return RelationGraph(torch.cat((edge_index, pair_index.to(device)), 1), torch.cat((edge_type, pair_type.to(device))),
                         torch.cat((weight, pair_weight)).to(device), 'on')


def weighted_message_sum(matrix: torch.Tensor, hidden: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
    """The sum of ``w_e * hidden[source] * r_type`` over the edges of each target relation, as one sparse product.

    Args:
        matrix: Sparse CSR matrix ``[N, T * N]``; column ``t * N + j`` holds the weight of the edge from ``j`` of type ``t``.
        hidden: Relation states ``[B, N, D]``.
        relations: Edge-type vectors ``[T, D]``.

    Returns:
        The message sums ``[B, N, D]``.
    """
    batch, nodes, dim = hidden.shape
    stacked = (hidden.unsqueeze(0) * relations[:, None, None, :]).transpose(1, 2).reshape(-1, batch * dim)
    result: torch.Tensor = (matrix @ stacked).view(nodes, batch, dim).transpose(0, 1)
    return result


class PCUltraRelationModel(RelNBFNet):
    """ULTRA's relation model on the weighted relation graph, with the sparse product at inference.

    It keeps the layers of the model that it wraps, so the parameters and their names do not change.
    """

    matrix_rows: Optional[torch.Tensor]
    matrix_columns: Optional[torch.Tensor]
    matrix_values: Optional[torch.Tensor]

    def __init__(self, original: RelNBFNet):
        nn.Module.__init__(self)
        self.layers, self.dim = original.layers, original.dim
        for name in ('matrix_rows', 'matrix_columns', 'matrix_values'):
            self.register_buffer(name, None, persistent=False)

    def attach(self, edge_index: torch.Tensor, edge_type: torch.Tensor, weight: torch.Tensor, num_nodes: int) -> None:
        """Store the matrix ``A = [A_0 | ... | A_5]`` of a relation graph for inference (rows: target relations).

        Args:
            edge_index: The relation-graph edges ``[2, M]``; row 0 receives the message of row 1.
            edge_type: The edge types ``[M]`` in 0 to 5.
            weight: The weight of each edge ``[M]``.
            num_nodes: The number of relations, technical inverses included.

        Raises:
            ValueError: When the relation graph repeats an edge.
        """
        columns = edge_type * num_nodes + edge_index[1]
        matrix = torch.sparse_coo_tensor(torch.stack((edge_index[0], columns)), weight, (num_nodes, EDGE_TYPES * num_nodes)).coalesce()
        if matrix._nnz() != len(weight):
            raise ValueError('The relation graph has repeated edges')
        csr = matrix.to_sparse_csr()
        self.matrix_rows, self.matrix_columns, self.matrix_values = csr.crow_indices(), csr.col_indices(), csr.values()

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor, num_relations: int, query: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Relation representations of the query relations.

        Args:
            edge_index: The relation-graph edges ``[2, M]``.
            edge_type: The edge types ``[M]``.
            num_relations: The number of relations, technical inverses included.
            query: The query relations ``[B]``.
            edge_weight: The weight of each edge ``[M]``.

        Returns:
            The relation states ``[B, num_relations, dim]``.

        Raises:
            RuntimeError: Without the weights or the attached matrix of this relation graph.
        """
        rows, columns, values = self.matrix_rows, self.matrix_columns, self.matrix_values
        if edge_weight is None or rows is None or columns is None or values is None or values.numel() != edge_type.numel():
            raise RuntimeError('Attach the weighted relation graph (set_graph) before scoring')
        first = cast(RelationalConv, self.layers[0])
        if not inference_only(self) or first.inference_backend == 'torch':
            # Training, and the 'torch' backend: ULTRA's weighted layer, one stored message per edge.
            plain: torch.Tensor = super().forward(edge_index, edge_type, num_relations, query, edge_weight)
            return plain
        boundary = first.linear.weight.new_zeros(len(query), num_relations, self.dim)
        boundary[torch.arange(len(query), device=query.device), query] = 1
        matrix = torch.sparse_csr_tensor(rows, columns, values.to(boundary.dtype), (num_relations, EDGE_TYPES * num_relations))
        hidden = boundary
        for layer in self.layers:
            conv = cast(RelationalConv, layer)
            update = weighted_message_sum(matrix, hidden, conv.relation.weight) + boundary
            value = conv.linear(torch.cat((hidden, update), dim=-1))
            fused = fused_norm_relu(value, hidden, conv.layer_norm, True, conv.inference_backend)
            hidden = fused if fused is not None else conv.layer_norm(value).relu() + hidden
        return hidden


class PCUltra(ULTRA):
    """ULTRA with count-weighted relation-graph edges and two pair edge types (see the module docstring).

    The two relation-graph options of ``ULTRA`` are fixed: ``ultra_relation_weighting='normalized'`` and
    ``ultra_pair_channels='on'``. Other values of these options in ``args`` are replaced.
    """

    name = 'PCUltra'
    graph_filename = 'pcultra_graph.pt'
    checkpoint_hint = ("PCUltra requires ULTRA's tensors with six-row relation tables in the relation model; "
                       'the thesis checkpoints have ultra_dim=64 and ultra_num_layers=6')

    def __init__(self, args: dict[str, Any]):
        super().__init__({**args, 'ultra_relation_weighting': 'normalized', 'ultra_pair_channels': 'on'})
        self.relation_model = PCUltraRelationModel(self.relation_model)

    def _build_relation_graph(self) -> None:
        num_entities, _ = self._require_graph()
        self._attach_relation_graph(sparse_relation_graph(self.graph_triples, num_entities, self.num_direct_relations))

    def _attach_relation_graph(self, graph: RelationGraph) -> None:
        super()._attach_relation_graph(graph)   # refuses a graph without weights or without the pair channels
        weight = self.rel_edge_weight
        assert weight is not None
        cast(PCUltraRelationModel, self.relation_model).attach(self.rel_edge_index, self.rel_edge_type, weight, 2 * self.num_direct_relations)
