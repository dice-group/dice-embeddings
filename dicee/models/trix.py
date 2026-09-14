"""Pure PyTorch TRIX entity and relation prediction.

Reimplementation of Zhang et al., https://arxiv.org/abs/2502.19512, verified
against https://github.com/yuchengz99/TRIX at UPSTREAM_COMMIT. The module names,
update schedule, binary entity-labelled relation edges, and fused convolution
direction follow the released code/checkpoints (see docs/trix.md).
"""
from collections import OrderedDict, defaultdict
from typing import cast

import torch
from torch import nn

from ._inference import candidate_features, candidate_slice, conditioned_linear, conditioned_score, inference_only
from .graph_model import GraphKGE, RelationGraphKGE
from .ultra import RelationalConv

UPSTREAM_COMMIT = "7596e14eefefe89e61396205a0550172cadeddb0"
INTERACTIONS = ("hh", "ht", "th", "tt")


def build_relation_graph(edge_index, edge_type, num_entities, num_relations):
    """Sparse (relation, relation, shared entity) edges in official role order.

    Repeated incidences have binary support. hh/tt exclude equal relations;
    ht/th include them. Distinct shared entities remain distinct edges.
    No dense entity-by-relation-by-relation tensor is materialized.
    """
    heads, tails = defaultdict(set), defaultdict(set)
    for (h, t), r in zip(edge_index.T.cpu().tolist(), edge_type.cpu().tolist()):
        heads[h].add(r)
        tails[t].add(r)
    result = {}
    for role, left, right in (("hh", heads, heads), ("ht", heads, tails),
                              ("th", tails, heads), ("tt", tails, tails)):
        edges = [(r1, r2, entity) for entity in sorted(left.keys() & right.keys())
                 for r1 in sorted(left[entity]) for r2 in sorted(right[entity])
                 if role in ("ht", "th") or r1 != r2]
        triples = torch.tensor(edges, dtype=torch.long, device=edge_index.device).reshape(-1, 3)
        result[role] = (triples[:, :2].T.contiguous(), triples[:, 2].contiguous())
    return result


class EntityReasoner(nn.Module):
    def __init__(self, dim, num_layers, output_dim=1):
        super().__init__()
        self.layers = nn.ModuleList([RelationalConv(dim, project_relations=True) for _ in range(num_layers)])
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, output_dim))

    def features(self, edges, num_entities, relations, heads, rels, states=None, tails=None, candidates=None, split=False):
        if tails is None:
            query = relations[torch.arange(len(heads), device=heads.device), rels]
        else:
            query = relations.new_ones(len(heads), relations.shape[-1])
        boundary = query.new_zeros(len(heads), num_entities, query.shape[-1])
        batch = torch.arange(len(heads), device=heads.device)
        boundary[batch, heads] = query
        if tails is not None:
            # Addition is essential for h == t, where the two labels cancel.
            boundary[batch, tails] -= query
        hidden = boundary if states is None else states
        for layer in self.layers:
            hidden = layer(hidden, boundary, *edges, relations, residual=True)
        if candidates is not None:
            hidden = candidate_features(hidden, candidates)
        if split:
            return hidden, query
        return torch.cat((hidden, query[:, None].expand_as(hidden)), -1)


class RelationReasoner(nn.Module):
    def __init__(self, dim, num_layers, entity_feedback=False):
        super().__init__()
        if entity_feedback:
            self.node_mlp = nn.Linear(2 * dim, dim)
        for role in INTERACTIONS:
            setattr(self, "layers_" + role, nn.ModuleList(
                [RelationalConv(dim, project_relations=True) for _ in range(num_layers)]))

    def step(self, index, states, boundary, graph, entities, constant_relations=None):
        messages = [getattr(self, "layers_" + role)[index](states, boundary, *graph[role], entities,
                                                        constant_relations=constant_relations)
                    for role in INTERACTIONS]
        return messages[0] + messages[1] + messages[2] + messages[3] + states


class TRIXBase(GraphKGE):
    config_prefix = "trix"
    graph_filename = "trix_graph.pt"
    checkpoint_hint = "use the matching entity/relation checkpoint with trix_dim=32; all keys and shapes must match"

    def __init__(self, args):
        super().__init__(args)
        self.dim = args.get("trix_dim", 32)
        if self.dim < 1:
            raise ValueError("TRIX dimension must be positive")
        for role in INTERACTIONS:
            self.register_buffer("rel_edge_index_" + role, None, persistent=False)
            self.register_buffer("rel_edge_type_" + role, None, persistent=False)

    @property
    def relation_graph(self):
        return {role: (getattr(self, "rel_edge_index_" + role), getattr(self, "rel_edge_type_" + role))
                for role in INTERACTIONS}

    def _build_relation_graph(self):
        self.clear_inference_cache()
        graph = build_relation_graph(self.edge_index, self.edge_type, self.num_entities, 2 * self.num_direct_relations)
        for role, (index, types) in graph.items():
            setattr(self, "rel_edge_index_" + role, index)
            setattr(self, "rel_edge_type_" + role, types)


class TRIX(TRIXBase):
    """Entity predictor compatible with official ``entity_prediction.pth``.

    DICE order is (head, relation, tail). Head corruption retains the original
    relation seed, then uses the inverse relation for both entity reasoners.
    Explicit reciprocal DICE queries use that same convention.
    """
    name = "TRIX"
    deterministic_inference = True

    def __init__(self, args):
        super().__init__(args)
        self.relation_model = RelationReasoner(self.dim, 3, entity_feedback=True)
        self.entity_model_1 = EntityReasoner(self.dim, 2)
        self.entity_model_2 = EntityReasoner(self.dim, 4)
        self.relation_cache_mb = args.get('graph_relation_cache_mb', 64)
        if self.relation_cache_mb < 0:
            raise ValueError('graph_relation_cache_mb must be nonnegative')
        self.set_inference_backend(args.get('graph_inference_backend', 'auto'))
        for module in self.modules():
            if isinstance(module, RelationalConv):
                module.inference_compile = args.get('graph_inference_compile', False)

    def clear_inference_cache(self):
        self._initial_cache = OrderedDict()
        self._initial_cache_token = None

    def _first_relation_step(self, query_relations):
        weight = next(self.parameters())
        boundary = weight.new_zeros(len(query_relations), 2 * self.num_direct_relations, self.dim)
        boundary[torch.arange(len(query_relations), device=query_relations.device), query_relations] = 1
        # All entities initially have the same label. Project a single label
        # per query and broadcast it *after* the MLP, retaining the graph layout.
        entities = weight.new_ones(len(query_relations), 1, self.dim)
        return self.relation_model.step(0, boundary, boundary, self.relation_graph, entities, self.num_entities)

    def _cached_initial_relations(self, query_relations):
        if not len(query_relations) or not inference_only(self):
            return None
        capacity = int(self.relation_cache_mb * 2**20) // (2 * self.num_direct_relations * self.dim * next(self.parameters()).element_size())
        if not capacity:
            self._initial_cache.clear()
            return None
        token = self.inference_token()
        if token is None:
            return None
        if token != self._initial_cache_token:
            self._initial_cache.clear()
            self._initial_cache_token = token
        ids = query_relations.tolist()
        missing = list(dict.fromkeys(q for q in ids if q not in self._initial_cache))
        values = {q: self._initial_cache[q] for q in set(ids) if q in self._initial_cache}
        for start in range(0, len(missing), self.query_batch_size):
            keys = missing[start:start + self.query_batch_size]
            hidden = self._first_relation_step(query_relations.new_tensor(keys))
            values.update((key, value.clone()) for key, value in zip(keys, hidden))
        for key in dict.fromkeys(ids):
            self._initial_cache[key] = values[key]
            self._initial_cache.move_to_end(key)
        while len(self._initial_cache) > capacity:
            self._initial_cache.popitem(last=False)
        return values, ids

    def _reason(self, heads, relations, query_relations, edges, initial=None):
        num_entities, _ = self._require_graph()
        weight = next(self.parameters())
        if initial is not None or inference_only(self):
            hidden = self._first_relation_step(query_relations) if initial is None else initial
            features, query = self.entity_model_1.features(edges, num_entities, hidden, heads, relations, split=True)
            entities = conditioned_linear(self.relation_model.node_mlp, features, query)
            boundary = weight.new_zeros(len(heads), 2 * self.num_direct_relations, self.dim)
            boundary[torch.arange(len(heads), device=heads.device), query_relations] = 1
            for i in (1, 2):
                hidden = self.relation_model.step(i, hidden, boundary, self.relation_graph, entities)
            return hidden
        entities = weight.new_ones(len(heads), num_entities, self.dim)
        boundary = weight.new_zeros(len(heads), 2 * self.num_direct_relations, self.dim)
        boundary[torch.arange(len(heads), device=heads.device), query_relations] = 1
        hidden = boundary
        for i in range(3):
            hidden = self.relation_model.step(i, hidden, boundary, self.relation_graph, entities)
            if i == 0:
                features = self.entity_model_1.features(edges, self.num_entities, hidden, heads, relations)
                entities = self.relation_model.node_mlp(features)
        return hidden

    def _score(self, heads, relations, candidates, query_relations, edges):
        output = []
        cached = self._cached_initial_relations(query_relations)
        split = inference_only(self)
        for start in range(0, len(heads), self.query_batch_size):
            sl = slice(start, start + self.query_batch_size)
            initial = torch.stack([cached[0][q] for q in cached[1][sl]]) if cached is not None else None
            rels = self._reason(heads[sl], relations[sl], query_relations[sl], edges, initial)
            features = self.entity_model_2.features(edges, self.num_entities, rels, heads[sl], relations[sl],
                                                    candidates=candidate_slice(candidates, sl), split=split)
            output.append((conditioned_score(self.entity_model_2.mlp, *features) if split else
                           self.entity_model_2.mlp(features)).squeeze(-1))
        return torch.cat(output) if output else next(self.parameters()).new_empty((0, candidates.shape[1]))


class TRIXRelation(TRIXBase, RelationGraphKGE):
    """Relation predictor compatible with official ``relation_prediction.pth``.

    Pairs are (head, tail); triples and grouped relation corruptions use DICE's
    (head, relation, tail) order. One reasoning pass scores every relation for a
    given pair, including external inverse IDs when present in the vocabulary.
    """
    name = "TRIXRelation"

    def __init__(self, args):
        super().__init__(args)
        self.relation_model = nn.ModuleList([RelationReasoner(self.dim, 2) for _ in range(3)])
        self.entity_model = nn.ModuleList([EntityReasoner(self.dim, 2, self.dim) for _ in range(3)])
        self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, 1))
        self.set_inference_backend(args.get('graph_inference_backend', 'auto'))
        for module in self.modules():
            if isinstance(module, RelationalConv):
                module.inference_compile = args.get('graph_inference_compile', False)

    def _relation_score(self, pairs, candidates, edges):
        num_entities, _ = self._require_graph()
        output = []
        split = inference_only(self)
        for start in range(0, len(pairs), self.query_batch_size):
            sl = slice(start, start + self.query_batch_size)
            heads, tails = pairs[sl].unbind(-1)
            rels = next(self.parameters()).new_ones(len(heads), 2 * self.num_direct_relations, self.dim)
            boundary = torch.ones_like(rels)
            entities = rels.new_zeros(len(heads), num_entities, self.dim)
            batch = torch.arange(len(heads), device=heads.device)
            entities[batch, heads] += 1
            entities[batch, tails] -= 1
            for entity_layer, relation_layer in zip(self.entity_model, self.relation_model):
                entity_model = cast(EntityReasoner, entity_layer)
                relation_model = cast(RelationReasoner, relation_layer)
                features = entity_model.features(edges, self.num_entities, rels, heads, None, entities, tails, split=split)
                entities = conditioned_score(entity_model.mlp, *features) if split else entity_model.mlp(features)
                for i in range(2):
                    rels = relation_model.step(i, rels, boundary, self.relation_graph, entities)
            features = rels.gather(1, candidates[sl, :, None].expand(-1, -1, self.dim))
            output.append(self.mlp(features).squeeze(-1))
        return torch.cat(output) if output else next(self.parameters()).new_empty((0, candidates.shape[1]))
