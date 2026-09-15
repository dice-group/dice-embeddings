"""Pure PyTorch ULTRA, compatible with DeepGraphLearning/ULTRA link predictors.

Architecture and graph construction adapted from
https://github.com/DeepGraphLearning/ULTRA at commit
427966ad8ed60420eef034063d44f3153addff90. Reference fixtures and the grouped
sampling and adversarial loss implementations also follow this upstream work.
Graphs are runtime context, excluded from the transferable state dictionary.

MIT License

Copyright (c) 2023 MilaGraph

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import OrderedDict
from typing import cast

import torch
from torch import nn

from ._fused_message import fused_distmult_sum, fused_norm_relu
from ._inference import candidate_features, candidate_slice, compiled_conv_update, conditioned_score, inference_only
from .graph_model import GraphKGE

UPSTREAM_COMMIT = "427966ad8ed60420eef034063d44f3153addff90"


class RelationalConv(nn.Module):
    """ULTRA's DistMult/sum convolution, including the boundary message."""

    def __init__(self, dim, project_relations=False):
        super().__init__()
        self.inference_backend = 'auto'
        self.inference_compile = False
        self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim * 2, dim)
        if project_relations:
            self.relation_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        else:
            self.relation = nn.Embedding(4, dim)

    def forward(self, states, boundary, edge_index, edge_type, relations=None, *, projected=None, residual=False,
                constant_relations=None):
        if projected is not None:
            relations = projected
        elif relations is None:
            relations = self.relation.weight.unsqueeze(0).expand(states.shape[0], -1, -1)
        else:
            relations = self.relation_projection(relations)
        if constant_relations is not None:
            relations = relations.expand(-1, constant_relations, -1)
        # The official fused rspmm path treats edge_index[0] as the output
        # row and edge_index[1] as the input column (opposite PyG's default
        # source-to-target propagation). Preserve this checkpoint convention.
        update = None if self.training else fused_distmult_sum(
            states, boundary, edge_index, edge_type, relations, self.inference_backend)
        if update is None:
            messages = states[:, edge_index[1]] * relations[:, edge_type]
            update = boundary.index_add(1, edge_index[0], messages)
        differentiable = torch.is_grad_enabled() and (states.requires_grad or update.requires_grad)
        if self.inference_compile and states.is_cuda and not differentiable and inference_only(self):
            # The region is reused across layers and models. Own its output so
            # a later CUDA-graph replay cannot overwrite a live hidden state.
            return compiled_conv_update()(states, update, self.linear.weight, self.linear.bias,
                                          self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps, residual).clone()
        value = self.linear(torch.cat((states, update), dim=-1))
        if not differentiable and inference_only(self):
            fused = fused_norm_relu(value, states, self.layer_norm, residual, self.inference_backend)
            if fused is not None:
                return fused
        value = self.layer_norm(value).relu()
        return value + states if residual else value


class RelNBFNet(nn.Module):
    def __init__(self, dim=64, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([RelationalConv(dim) for _ in range(num_layers)])
        self.dim = dim

    def forward(self, edge_index, edge_type, num_relations, query):
        first_layer = cast(RelationalConv, self.layers[0])
        boundary = first_layer.linear.weight.new_zeros(len(query), num_relations, self.dim)
        boundary[torch.arange(len(query), device=query.device), query] = 1
        hidden = boundary
        for layer in self.layers:
            hidden = layer(hidden, boundary, edge_index, edge_type, residual=True)
        return hidden


class EntityNBFNet(nn.Module):
    def __init__(self, dim=64, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([RelationalConv(dim, project_relations=True) for _ in range(num_layers)])
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.ReLU(), nn.Linear(dim * 2, 1))

    def forward(self, edge_index, edge_type, num_entities, relations, heads, rels, tails, projections=None):
        query = relations[torch.arange(len(heads), device=heads.device), rels]
        boundary = query.new_zeros(len(heads), num_entities, query.shape[-1])
        boundary[torch.arange(len(heads), device=heads.device), heads] = query
        hidden = boundary
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, boundary, edge_index, edge_type, relations,
                           projected=None if projections is None else projections[i], residual=True)
        # Project only requested candidates, avoiding the full-node scoring MLP.
        features = candidate_features(hidden, tails)
        if inference_only(self):
            return conditioned_score(self.mlp, features, query).squeeze(-1)
        return self.mlp(torch.cat((features, query[:, None].expand_as(features)), -1)).squeeze(-1)


def build_relation_graph(edge_index, edge_type, num_entities, num_relations):
    """Binary support of the hh, tt, ht, th incidence products, in upstream order."""
    incidences = []
    for endpoints in edge_index:
        pairs = torch.stack((endpoints, edge_type)).unique(dim=1)
        incidences.append(torch.sparse_coo_tensor(
            pairs, torch.ones(pairs.shape[1], device=pairs.device), (num_entities, num_relations)
        ).coalesce())
    heads, tails = incidences
    edges, types = [], []
    for kind, (left, right) in enumerate(((heads, heads), (tails, tails), (heads, tails), (tails, heads))):
        # Upstream incidence normalization changes positive weights, not support;
        # the relation reasoner uses only this support with unit edge weights.
        adjacency = torch.sparse.mm(left.transpose(0, 1), right).coalesce()
        edges.append(adjacency.indices())
        types.append(torch.full((adjacency._nnz(),), kind, dtype=torch.long, device=edge_type.device))
    return torch.cat(edges, 1), torch.cat(types)


class ULTRA(GraphKGE):
    """Pure PyTorch ULTRA with the official state dictionary and DICE triple order."""

    name = 'ULTRA'
    config_prefix = 'ultra'
    graph_filename = 'ultra_graph.pt'
    checkpoint_hint = 'official ULTRA requires ultra_dim=64, ultra_num_layers=6; all keys and shapes must match'
    deterministic_inference = True

    def __init__(self, args):
        super().__init__(args)
        self.dim = args.get('ultra_dim', 64)
        self.num_layers = args.get('ultra_num_layers', 6)
        if min(self.dim, self.num_layers) < 1:
            raise ValueError('ULTRA dimensions and layers must be positive')
        self.relation_model = RelNBFNet(self.dim, self.num_layers)
        self.entity_model = EntityNBFNet(self.dim, self.num_layers)
        for name in ('rel_edge_index', 'rel_edge_type'):
            self.register_buffer(name, None, persistent=False)
        self.relation_cache_mb = args.get('graph_relation_cache_mb', 64)
        self.projection_cache_mb = args.get('graph_projection_cache_mb', 64)
        if min(self.relation_cache_mb, self.projection_cache_mb) < 0:
            raise ValueError('Graph cache limits must be nonnegative')
        self.set_inference_backend(args.get('graph_inference_backend', 'auto'))
        for layer in (*self.relation_model.layers, *self.entity_model.layers):
            layer.inference_compile = args.get('graph_inference_compile', False)

    def clear_inference_cache(self):
        self._relation_cache = OrderedDict()
        self._relation_cache_token = None
        self._projection_cache = OrderedDict()
        self._projection_cache_token = None

    def _build_relation_graph(self):
        self.clear_inference_cache()
        self.rel_edge_index, self.rel_edge_type = build_relation_graph(
            self.edge_index, self.edge_type, self.num_entities, 2 * self.num_direct_relations)

    def _cached_relations(self, query_relations):
        # Autocast can change the cached representation's precision independently
        # of parameter dtype. Keep that path uncached, as well as all autograd.
        if (self.training or torch.is_autocast_enabled(self.device.type)
                or (torch.is_grad_enabled() and any(p.requires_grad for p in self.relation_model.parameters()))):
            return None
        capacity = int(self.relation_cache_mb * 2**20) // (2 * self.num_direct_relations * self.dim * next(self.parameters()).element_size())
        if not capacity:
            self.clear_inference_cache()
            return None
        token = self.inference_token()
        if token is None:
            return None
        if token != self._relation_cache_token:
            self.clear_inference_cache()
            self._relation_cache_token = token
        ids = query_relations.tolist()
        missing = list(dict.fromkeys(q for q in ids if q not in self._relation_cache))
        # Keep current-call values alive even when they exceed the LRU capacity.
        values = {q: self._relation_cache[q] for q in set(ids) if q in self._relation_cache}
        for start in range(0, len(missing), self.query_batch_size):
            keys = missing[start:start + self.query_batch_size]
            rels = self.relation_model(self.rel_edge_index, self.rel_edge_type, 2 * self.num_direct_relations,
                                       query_relations.new_tensor(keys))
            for key, value in zip(keys, rels):
                values[key] = value.clone()
        for key in dict.fromkeys(ids):
            self._relation_cache[key] = values[key]
            self._relation_cache.move_to_end(key)
            while len(self._relation_cache) > capacity:
                self._relation_cache.popitem(last=False)
        return values, ids

    def _cached_projections(self, rels, ids, token):
        capacity = int(self.projection_cache_mb * 2**20) // (self.num_layers * rels[0].numel() * rels.element_size())
        if not capacity:
            self._projection_cache.clear()
            return None
        if token is None:
            return None
        if token != self._projection_cache_token:
            self._projection_cache.clear()
            self._projection_cache_token = token
        missing = list(dict.fromkeys(q for q in ids if q not in self._projection_cache))
        values = {q: self._projection_cache[q] for q in set(ids) if q in self._projection_cache}
        if missing:
            indices = rels.new_tensor([ids.index(q) for q in missing], dtype=torch.long)
            source = rels.index_select(0, indices)
            projected = [layer.relation_projection(source) for layer in self.entity_model.layers]
            for i, key in enumerate(missing):
                values[key] = tuple(value[i].clone() for value in projected)
        for key in dict.fromkeys(ids):
            self._projection_cache[key] = values[key]
            self._projection_cache.move_to_end(key)
        while len(self._projection_cache) > capacity:
            self._projection_cache.popitem(last=False)
        return tuple(torch.stack([values[q][i] for q in ids]) for i in range(self.num_layers))

    def _score(self, heads, relations, candidates, query_relations, edges):
        output = []
        cached = self._cached_relations(query_relations) if len(heads) else None
        projection_token = self._relation_cache_token if cached is not None and inference_only(self) else None
        for start in range(0, len(heads), self.query_batch_size):
            sl = slice(start, start + self.query_batch_size)
            rels = (torch.stack([cached[0][q] for q in cached[1][sl]]) if cached is not None else
                    self.relation_model(self.rel_edge_index, self.rel_edge_type,
                                        2 * self.num_direct_relations, query_relations[sl]))
            projections = self._cached_projections(rels, cached[1][sl], projection_token) if projection_token is not None else None
            output.append(self.entity_model(*edges, self.num_entities, rels, heads[sl], relations[sl],
                                            candidate_slice(candidates, sl), projections))
        return torch.cat(output) if output else next(self.parameters()).new_empty((0, candidates.shape[1]))
