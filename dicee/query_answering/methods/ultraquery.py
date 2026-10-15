"""Native UltraQuery using DICE's checkpoint-compatible ULTRA layers."""

import torch

from ...models.ultra import EntityNBFNet, RelNBFNet, build_relation_graph
from ._common import FuzzyGNN, TensorCache
from ._ordered_aggregation import aggregate, cached_layout


class UltraQuery(FuzzyGNN):
    def __init__(self, context, *, dim=64, num_layers=6, threshold=0., logic='product', cache_bytes=64 * 2**20,
                 observed_traversal=False, calibration=None):
        super().__init__(context, logic, observed_traversal, calibration)
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between zero and one')
        if calibration and threshold:
            raise ValueError('A calibrated projection takes no membership threshold')
        self.threshold = threshold
        self.relation_model = RelNBFNet(dim, num_layers)
        self.entity_model = EntityNBFNet(dim, num_layers)
        triples = torch.tensor(context.triples, dtype=torch.long).reshape(-1, 3)
        edges, types = triples[:, [0, 2]].T, triples[:, 1]
        redges, rtypes = build_relation_graph(edges, types, context.num_entities, context.num_relations)
        for name, value in (('edge_index', edges), ('edge_type', types), ('rel_edge_index', redges), ('rel_edge_type', rtypes)):
            self.register_buffer(name, value, persistent=False)
        self.cache = TensorCache(cache_bytes)
        self._layouts = {}

    def project(self, membership, relation):
        return self.project_batch(membership[None], torch.tensor([relation], device=self.device), relation_ids=(relation,))[0]

    @staticmethod
    def _layer(layer, hidden, boundary, relations, edges):
        update = aggregate(hidden.transpose(0, 1), relations.transpose(0, 1), edges).transpose(0, 1) + boundary
        value = layer.linear(torch.cat((hidden, update), -1))
        return layer.layer_norm(value).relu() + hidden

    def project_batch(self, membership, relation, *, relation_ids=None, logits=False):
        rel_edges = cached_layout(self._layouts, 'relation', self.rel_edge_index, self.rel_edge_type,
                                   self.context.num_relations)
        ent_edges = cached_layout(self._layouts, 'entity', self.edge_index, self.edge_type, self.context.num_entities)
        self.cache.refresh(self)
        key = tuple(relation.tolist()) if relation_ids is None else tuple(relation_ids)
        relations = self.cache.get(key)
        if relations is None:
            boundary = membership.new_zeros(len(relation), self.context.num_relations, self.relation_model.dim)
            boundary[torch.arange(len(relation), device=self.device), relation] = 1
            relations = boundary
            for layer in self.relation_model.layers:
                rel = layer.relation.weight.expand(len(relation), -1, -1)
                relations = self._layer(layer, relations, boundary, rel, rel_edges)
            self.cache.put(key, relations)
        query = relations[torch.arange(len(relation), device=self.device), relation]
        if self.threshold > 0:
            membership = membership.masked_fill(membership <= self.threshold, 0)
        boundary = torch.einsum('bn,bd->bnd', membership, query)
        hidden = boundary
        for index, layer in enumerate(self.entity_model.layers):
            projection_key = ('projection', key, index)
            projected = self.cache.get(projection_key)
            if projected is None:
                projected = layer.relation_projection(relations)
                # Optional projections use spare capacity; reasoner outputs have priority.
                if key in self.cache.values:
                    self.cache.put(projection_key, projected, optional=True)
            hidden = self._layer(layer, hidden, boundary, projected, ent_edges)
        # Preserve the reference MLP arithmetic, including saturated probability ties.
        features = torch.cat((hidden, query[:, None].expand_as(hidden)), -1)
        output = self.entity_model.mlp(features).squeeze(-1)
        return output if logits else output.sigmoid()
