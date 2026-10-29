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
from typing import NamedTuple, Optional, cast

import torch
from torch import nn

from ._fused_message import fused_distmult_sum, fused_norm_relu
from ._inference import cached_relation_batch, candidate_features, candidate_slice, compiled_conv_update, conditioned_score, host_ids, inference_only, to_device
from ._relation_weights import INCIDENCE_PAIRS, PAIR_EDGE_TYPES, pair_channel_weights, relation_count_weights
from .graph_model import GraphKGE

UPSTREAM_COMMIT = "427966ad8ed60420eef034063d44f3153addff90"
RELATION_WEIGHTINGS = ('binary', 'normalized')
PAIR_CHANNELS = ('off', 'on')


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

    def weighted(self, states: torch.Tensor, boundary: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                 edge_weight: torch.Tensor) -> torch.Tensor:
        """Residual layer whose DistMult messages are scaled by per-edge weights, in plain PyTorch.

        Training and inference use the same operations, without the fused inference kernels: gather, multiply by the
        relation row, multiply by the weight, index_add onto the boundary, linear, layer norm, ReLU, residual.
        """
        relations = self.relation.weight.unsqueeze(0).expand(states.shape[0], -1, -1)
        messages = states[:, edge_index[1]] * relations[:, edge_type]
        messages = messages * edge_weight.to(messages.dtype)[None, :, None]
        update = boundary.index_add(1, edge_index[0], messages)
        value = self.linear(torch.cat((states, update), dim=-1))
        normalized: torch.Tensor = self.layer_norm(value)
        return normalized.relu() + states


class RelNBFNet(nn.Module):
    def __init__(self, dim=64, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([RelationalConv(dim) for _ in range(num_layers)])
        self.dim = dim

    def forward(self, edge_index, edge_type, num_relations, query, edge_weight: Optional[torch.Tensor] = None):
        """Relation representations of the query relations; ``edge_weight`` scales each relation-graph message."""
        if edge_weight is not None and edge_weight.shape != edge_type.shape:
            raise ValueError('Relation-graph edge weights must match the relation-graph edges')
        first_layer = cast(RelationalConv, self.layers[0])
        boundary = first_layer.linear.weight.new_zeros(len(query), num_relations, self.dim)
        boundary[torch.arange(len(query), device=query.device), query] = 1
        hidden = boundary
        for layer in self.layers:
            if edge_weight is None:
                hidden = layer(hidden, boundary, edge_index, edge_type, residual=True)
            else:
                hidden = cast(RelationalConv, layer).weighted(hidden, boundary, edge_index, edge_type, edge_weight)
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


def weighted_relation_graph(graph_triples: torch.Tensor, num_entities: int, num_direct: int,
                            pair_channels: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The relation graph with the normalized incidence count of each edge, optionally with the entity-pair channels.

    The edges of types 0 to 3 are ``build_relation_graph``'s, in its order; with ``pair_channels`` the pair edges of
    types 4 and 5 follow them (nonzero pair weights in row-major order). The weights are computed on the CPU in float64
    from the direct triples of the graph (``_relation_weights``).

    Args:
        graph_triples: Internal graph triples with technical inverses, as ``GraphKGE.set_graph`` stores them.
        num_entities: Number of entities.
        num_direct: Number of direct relations (relation ids ``num_direct`` and above are technical inverses).
        pair_channels: Append the two entity-pair channels.

    Returns:
        ``(edge_index [2, M], edge_type [M], edge_weight [M] float64)`` on the device of ``graph_triples``.
    """
    edge_index, edge_type = graph_triples[:, [0, 2]].T, graph_triples[:, 1]
    rel_edge_index, rel_edge_type = build_relation_graph(edge_index, edge_type, num_entities, 2 * num_direct)
    direct = graph_triples[graph_triples[:, 1] < num_direct].detach().cpu()
    counts = relation_count_weights(direct, num_entities, num_direct)
    weight = counts[rel_edge_type.cpu(), rel_edge_index[0].cpu(), rel_edge_index[1].cpu()]
    if pair_channels:
        pairs = pair_channel_weights(direct, num_entities, num_direct)
        kind, rows, cols = pairs.nonzero(as_tuple=True)
        device = rel_edge_index.device
        rel_edge_index = torch.cat((rel_edge_index, torch.stack((rows, cols)).to(device)), dim=1)
        rel_edge_type = torch.cat((rel_edge_type, (kind + PAIR_EDGE_TYPES[0]).to(device)))
        weight = torch.cat((weight, pairs[kind, rows, cols]))
    return rel_edge_index, rel_edge_type, weight.to(rel_edge_index.device)


class RelationGraph(NamedTuple):
    """ULTRA's relation graph: the edges between relations, their types and, with count weighting, their float64 weights."""

    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: Optional[torch.Tensor]


def relation_graph(graph_triples: torch.Tensor, num_entities: int, num_direct: int, relation_weighting: str = 'binary',
                   pair_channels: str = 'off') -> RelationGraph:
    """The relation graph of ULTRA with these options, as ``ULTRA.set_graph`` builds it.

    ``'binary'`` gives ``build_relation_graph``'s support without weights; ``'normalized'`` gives
    ``weighted_relation_graph`` with the pair channels when ``pair_channels`` is ``'on'``. The result can be computed in
    another process and attached with ``ULTRA.attach_graph``.

    Args:
        graph_triples: Internal graph triples with technical inverses (``GraphKGE.internal_graph``).
        num_entities: Number of entities.
        num_direct: Number of direct relations.
        relation_weighting: ``ultra_relation_weighting``.
        pair_channels: ``ultra_pair_channels``.

    Returns:
        The relation graph on the device of ``graph_triples``.
    """
    if relation_weighting not in RELATION_WEIGHTINGS or pair_channels not in PAIR_CHANNELS:
        raise ValueError(f'unknown ULTRA options {relation_weighting!r}, {pair_channels!r}')
    if pair_channels == 'on' and relation_weighting != 'normalized':
        raise ValueError("ultra_pair_channels='on' requires ultra_relation_weighting='normalized'")
    if relation_weighting == 'binary':
        edge_index, edge_type = build_relation_graph(graph_triples[:, [0, 2]].T, graph_triples[:, 1], num_entities, 2 * num_direct)
        return RelationGraph(edge_index, edge_type, None)
    return RelationGraph(*weighted_relation_graph(graph_triples, num_entities, num_direct, pair_channels == 'on'))


class ULTRA(GraphKGE):
    """Pure PyTorch ULTRA with the official state dictionary and DICE triple order.

    Two options extend the relation graph; their defaults give official ULTRA:

    - ``ultra_relation_weighting``: ``'binary'`` (default) uses the binary support of the relation graph.
      ``'normalized'`` multiplies every relation-graph message by the normalized incidence count of its edge
      (``weighted_relation_graph``); the relation reasoner then runs the weighted layers in plain PyTorch.
    - ``ultra_pair_channels``: ``'on'`` adds the two entity-pair edge types 4 and 5 with their weights, so the relation
      tables of the relation reasoner get 6 rows. Requires ``'normalized'``. The six tables are drawn after all of
      ULTRA's tensors, and their rows 0 to 3 are ULTRA's rows, so the other tensors equal official ULTRA's under the
      same random seed.

    The options are not stored in the state dictionary. A checkpoint that records them in ``model_config`` must
    agree with the model (``load_pretrained``).
    """

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
        self.relation_weighting = args.get('ultra_relation_weighting', 'binary')
        self.pair_channels = args.get('ultra_pair_channels', 'off')
        if self.relation_weighting not in RELATION_WEIGHTINGS:
            raise ValueError('ultra_relation_weighting must be binary or normalized')
        if self.pair_channels not in PAIR_CHANNELS:
            raise ValueError('ultra_pair_channels must be off or on')
        if self.pair_channels == 'on' and self.relation_weighting != 'normalized':
            raise ValueError("ultra_pair_channels='on' requires ultra_relation_weighting='normalized'")
        self.relation_model = RelNBFNet(self.dim, self.num_layers)
        self.entity_model = EntityNBFNet(self.dim, self.num_layers)
        if self.pair_channels == 'on':
            rows = len(INCIDENCE_PAIRS) + len(PAIR_EDGE_TYPES)
            for layer in self.relation_model.layers:
                conv = cast(RelationalConv, layer)
                table = nn.Embedding(rows, self.dim)
                with torch.no_grad():
                    table.weight[:len(INCIDENCE_PAIRS)].copy_(conv.relation.weight)
                conv.relation = table
        for name in ('rel_edge_index', 'rel_edge_type', 'rel_edge_weight'):
            self.register_buffer(name, None, persistent=False)
        self.relation_cache_mb = args.get('graph_relation_cache_mb', 64)
        self.projection_cache_mb = args.get('graph_projection_cache_mb', 64)
        if min(self.relation_cache_mb, self.projection_cache_mb) < 0:
            raise ValueError('Graph cache limits must be nonnegative')
        self.set_inference_backend(args.get('graph_inference_backend', 'auto'))
        for layer in (*self.relation_model.layers, *self.entity_model.layers):
            layer.inference_compile = args.get('graph_inference_compile', False)
        # 'query' conditions an inverse tail query on the inverse relation, as UltraQuery's projection does, for weights
        # trained that way; the default 'direct' is ULTRA link prediction.
        self.relation_conditioning = args.get('ultra_relation_conditioning', 'direct')
        if self.relation_conditioning not in ('direct', 'query'):
            raise ValueError('ultra_relation_conditioning must be direct or query')

    def clear_inference_cache(self):
        self._relation_cache = OrderedDict()
        self._relation_cache_token = None
        self._projection_cache = OrderedDict()
        self._projection_cache_token = None

    def _build_relation_graph(self):
        num_entities, _ = self._require_graph()
        self._attach_relation_graph(relation_graph(self.graph_triples, num_entities, self.num_direct_relations,
                                                   self.relation_weighting, self.pair_channels))

    def _attach_relation_graph(self, graph: RelationGraph) -> None:
        """Store a relation graph of ``relation_graph`` for this model's options; the weights take the parameter dtype."""
        edge_index, edge_type, edge_weight = graph
        if (edge_weight is None) != (self.relation_weighting == 'binary'):
            raise ValueError(f'the relation graph does not fit ultra_relation_weighting={self.relation_weighting!r}')
        if tuple(edge_index.shape) != (2, len(edge_type)) or (edge_weight is not None and edge_weight.shape != edge_type.shape):
            raise ValueError('Relation-graph edge indices, types and weights must match')
        self.clear_inference_cache()
        self.rel_edge_index = to_device(edge_index, self.device)
        self.rel_edge_type = to_device(edge_type, self.device)
        self.rel_edge_weight = None if edge_weight is None else to_device(edge_weight.to(dtype=next(self.parameters()).dtype), self.device)

    def load_pretrained(self, path):
        """Load weights; a checkpoint whose ``model_config`` records the relation-graph options must match them."""
        stored = torch.load(path, map_location='cpu', weights_only=True)
        config = stored.get('model_config') if isinstance(stored, dict) else None
        if isinstance(config, dict):
            options = {'ultra_relation_weighting': self.relation_weighting, 'ultra_pair_channels': self.pair_channels}
            defaults = {'ultra_relation_weighting': 'binary', 'ultra_pair_channels': 'off'}
            mismatch = {key: config.get(key, defaults[key]) for key, value in options.items() if config.get(key, defaults[key]) != value}
            if mismatch:
                raise ValueError(f'{self.name} checkpoint was saved with {mismatch}; build the model with the same options')
        return super().load_pretrained(path)

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
        ids = host_ids(query_relations)
        missing = list(dict.fromkeys(q for q in ids if q not in self._relation_cache))
        # Keep current-call values alive even when they exceed the LRU capacity.
        values = {q: self._relation_cache[q] for q in set(ids) if q in self._relation_cache}
        for start in range(0, len(missing), self.query_batch_size):
            keys = missing[start:start + self.query_batch_size]
            rels = self.relation_model(self.rel_edge_index, self.rel_edge_type, 2 * self.num_direct_relations,
                                       to_device(torch.tensor(keys), query_relations.device), edge_weight=self.rel_edge_weight)
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
            indices = to_device(torch.tensor([ids.index(q) for q in missing], dtype=torch.long), rels.device)
            source = rels.index_select(0, indices)
            projected = [layer.relation_projection(source) for layer in self.entity_model.layers]
            for i, key in enumerate(missing):
                values[key] = tuple(value[i].clone() for value in projected)
        for key in dict.fromkeys(ids):
            self._projection_cache[key] = values[key]
            self._projection_cache.move_to_end(key)
        while len(self._projection_cache) > capacity:
            self._projection_cache.popitem(last=False)
        return tuple(cached_relation_batch({q: value[i] for q, value in values.items()}, ids) for i in range(self.num_layers))

    def _score(self, heads, relations, candidates, query_relations, edges):
        output = []
        cached = self._cached_relations(query_relations) if len(heads) else None
        projection_token = self._relation_cache_token if cached is not None and inference_only(self) else None
        for start in range(0, len(heads), self.query_batch_size):
            sl = slice(start, start + self.query_batch_size)
            rels = (torch.stack([cached[0][q] for q in cached[1][sl]]) if cached is not None else
                    self.relation_model(self.rel_edge_index, self.rel_edge_type,
                                        2 * self.num_direct_relations, query_relations[sl], edge_weight=self.rel_edge_weight))
            projections = self._cached_projections(rels, cached[1][sl], projection_token) if projection_token is not None else None
            output.append(self.entity_model(*edges, self.num_entities, rels, heads[sl], relations[sl],
                                            candidate_slice(candidates, sl), projections))
        return torch.cat(output) if output else next(self.parameters()).new_empty((0, candidates.shape[1]))
