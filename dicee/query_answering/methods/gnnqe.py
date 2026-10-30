"""GNN-QE's NBFNet inference without TorchDrug runtime dependencies."""

import torch
from torch import nn

from ._common import FuzzyGNN
from ._ordered_aggregation import aggregate, cached_layout


def degree_statistics(degree, pna):
    """Prepare graph-only values on their original device without reassociation."""
    count = degree[:, None, None] + 1
    scales = None
    if pna:
        scale = count.log()
        scale = scale / scale.mean()
        scales = torch.cat((torch.ones_like(scale), scale, 1 / scale.clamp_min(1e-2)), -1)
    return count, scales


class GNNQELayer(nn.Module):
    def __init__(self, dim, num_relations, *, aggregate='pna', layer_norm=True, dependent=True):
        super().__init__()
        if aggregate not in ('pna', 'sum', 'mean', 'max'):
            raise ValueError('Unsupported GNN-QE aggregation')
        self.dim, self.num_relations, self.aggregate = dim, num_relations, aggregate
        self.linear = nn.Linear(dim * (13 if aggregate == 'pna' else 2), dim)
        self.layer_norm = nn.LayerNorm(dim) if layer_norm else None
        if dependent:
            self.relation_linear = nn.Linear(dim, num_relations * dim)
        else:
            self.relation = nn.Embedding(num_relations, dim)

    def forward(self, hidden, boundary, query, edges, degree, statistics=None):
        relations = (self.relation_linear(query).reshape(len(query), self.num_relations, self.dim).transpose(0, 1)
                     if hasattr(self, 'relation_linear') else self.relation.weight[:, None].expand(-1, len(query), -1))
        if self.aggregate == 'pna':
            total, squares, high, low = aggregate(hidden, relations, edges, 'pna')
        else:
            total = aggregate(hidden, relations, edges) if self.aggregate != 'max' else None
            high = aggregate(hidden, relations, edges, 'max') if self.aggregate == 'max' else None

        if self.aggregate in ('mean', 'pna'):
            count, scales = statistics if statistics is not None else degree_statistics(degree, self.aggregate == 'pna')
        if self.aggregate == 'sum':
            update = total + boundary
        elif self.aggregate == 'mean':
            update = (total + boundary) / count
        elif self.aggregate == 'max':
            update = torch.maximum(high, boundary)
        else:
            mean = (total + boundary) / count
            variance = (squares + boundary.square()) / count - mean.square()
            features = torch.stack((mean, torch.maximum(high, boundary), torch.minimum(low, boundary),
                                    variance.clamp_min(1e-6).sqrt()), -1).flatten(-2)
            update = (features[..., None] * scales[:, :, None]).flatten(-2)
        value = self.linear(torch.cat((hidden, update), -1))
        if self.layer_norm is not None:
            value = self.layer_norm(value)
        return value.relu()


class GNNQEMLP(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim if index + 1 < num_layers else 1) for index in range(num_layers)])

    def forward(self, value):
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 < len(self.layers):
                value = value.relu()
        return value


class GNNQE(FuzzyGNN):
    """Full-graph inference with ordered sparse reductions.

    edge_batch_size and candidate_batch_size describe manifest metadata;
    they do not chunk execution or bound memory. Query batch size controls the
    dense state size.
    """
    def __init__(self, context, *, dim=32, num_layers=4, aggregate='pna', short_cut=True,
                 layer_norm=True, dependent=True, concat_hidden=False, mlp_layers=2, logic='product',
                 edge_batch_size=65536, candidate_batch_size=4096):
        super().__init__(context, logic)
        if min(dim, num_layers, mlp_layers, edge_batch_size, candidate_batch_size) < 1:
            raise ValueError('GNN-QE dimensions and batch sizes must be positive')
        self.short_cut, self.concat_hidden = short_cut, concat_hidden
        self.edge_batch_size, self.candidate_batch_size = edge_batch_size, candidate_batch_size
        self.query = nn.Embedding(context.num_relations, dim)
        self.layers = nn.ModuleList([GNNQELayer(dim, context.num_relations, aggregate=aggregate,
                                               layer_norm=layer_norm, dependent=dependent) for _ in range(num_layers)])
        self.mlp = GNNQEMLP(dim * (num_layers + 1 if concat_hidden else 2), mlp_layers)
        triples = torch.tensor(context.triples, dtype=torch.long).reshape(-1, 3)
        self.register_buffer('edge_index', triples[:, [2, 0]].T, persistent=False)
        self.register_buffer('edge_type', triples[:, 1], persistent=False)
        self.register_buffer('degree', torch.bincount(triples[:, 2], minlength=context.num_entities).float(), persistent=False)
        self._layouts = {}

    def project(self, membership, relation):
        return self.project_batch(membership[None], torch.tensor([relation], device=self.device))[0]

    def project_batch(self, membership, relation, *, relation_ids=None):
        edges = cached_layout(self._layouts, 'entity', self.edge_index, self.edge_type,
                               self.context.num_entities, relation_order=True)
        query = self.query(relation)
        boundary = torch.einsum('bn,bd->nbd', membership, query)
        hidden, hiddens = boundary, []
        statistics = (degree_statistics(self.degree, any(layer.aggregate == 'pna' for layer in self.layers))
                      if any(layer.aggregate in ('mean', 'pna') for layer in self.layers) else None)
        for layer in self.layers:
            update = layer(hidden, boundary, query, edges, self.degree, statistics)
            hidden = update + hidden if self.short_cut else update
            if self.concat_hidden:
                hiddens.append(hidden)
        values = hiddens if self.concat_hidden else [hidden]
        features = torch.cat([*values, query.expand(self.context.num_entities, -1, -1)], -1)
        return self.mlp(features).squeeze(-1).sigmoid().t()
