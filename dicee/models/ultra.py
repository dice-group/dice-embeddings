"""Pure PyTorch ULTRA, compatible with DeepGraphLearning/ULTRA link predictors.

Architecture adapted from ULTRA (MIT); see THIRD_PARTY_NOTICES.md. Graphs are
runtime context, deliberately excluded from the transferable state dictionary.
"""
from pathlib import Path

import torch
from torch import nn

from .base_model import BaseKGE

UPSTREAM_COMMIT = "427966ad8ed60420eef034063d44f3153addff90"


class RelationalConv(nn.Module):
    """ULTRA's DistMult/sum convolution, including the boundary message."""

    def __init__(self, dim, project_relations=False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim * 2, dim)
        if project_relations:
            self.relation_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        else:
            self.relation = nn.Embedding(4, dim)

    def forward(self, states, boundary, edge_index, edge_type, relations=None):
        if relations is None:
            relations = self.relation.weight.unsqueeze(0).expand(states.shape[0], -1, -1)
        else:
            relations = self.relation_projection(relations)
        # The official fused rspmm path treats edge_index[0] as the output
        # row and edge_index[1] as the input column (opposite PyG's default
        # source-to-target propagation). Preserve this checkpoint convention.
        messages = states[:, edge_index[1]] * relations[:, edge_type]
        update = boundary.index_add(1, edge_index[0], messages)
        return self.layer_norm(self.linear(torch.cat((states, update), dim=-1))).relu()


class RelNBFNet(nn.Module):
    def __init__(self, dim=64, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([RelationalConv(dim) for _ in range(num_layers)])
        self.dim = dim

    def forward(self, edge_index, edge_type, num_relations, query):
        boundary = self.layers[0].linear.weight.new_zeros(len(query), num_relations, self.dim)
        boundary[torch.arange(len(query), device=query.device), query] = 1
        hidden = boundary
        for layer in self.layers:
            hidden = hidden + layer(hidden, boundary, edge_index, edge_type)
        return hidden


class EntityNBFNet(nn.Module):
    def __init__(self, dim=64, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([RelationalConv(dim, project_relations=True) for _ in range(num_layers)])
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.ReLU(), nn.Linear(dim * 2, 1))

    def forward(self, edge_index, edge_type, num_entities, relations, heads, rels, tails):
        query = relations[torch.arange(len(heads), device=heads.device), rels]
        boundary = query.new_zeros(len(heads), num_entities, query.shape[-1])
        boundary[torch.arange(len(heads), device=heads.device), heads] = query
        hidden = boundary
        for layer in self.layers:
            hidden = hidden + layer(hidden, boundary, edge_index, edge_type, relations)
        # Project only requested candidates, avoiding the full-node scoring MLP.
        features = hidden.gather(1, tails.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
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


class ULTRA(BaseKGE):
    """Graph-conditioned scores with DICE triple order (head, relation, tail).

    ``forward_grouped`` retains upstream head-corruption query conditioning.
    Flat triples always mean independent tail-oriented scores; callers predicting
    missing heads should use ``forward_k_vs_all_heads`` explicitly.
    """

    def __init__(self, args):
        args = dict(args)
        if args.get('byte_pair_encoding'):
            raise ValueError('ULTRA does not support BPE')
        if args.get('normalization') not in (None, 'None'):
            raise ValueError('ULTRA uses its own layer normalization; set normalization=None')
        args['normalization'] = None
        super().__init__(args)
        self.name = 'ULTRA'
        self.dim = args.get('ultra_dim', 64)
        self.num_layers = args.get('ultra_num_layers', 6)
        self.query_batch_size = args.get('ultra_query_batch_size', 8)
        if min(self.dim, self.num_layers, self.query_batch_size) < 1:
            raise ValueError('ULTRA dimensions, layers, and query batch size must be positive')
        self.relation_model = RelNBFNet(self.dim, self.num_layers)
        self.entity_model = EntityNBFNet(self.dim, self.num_layers)
        for name in ('graph_triples', 'relation_id_map', 'edge_index', 'edge_type', 'rel_edge_index', 'rel_edge_type'):
            self.register_buffer(name, None, persistent=False)
        self.num_direct_relations = 0

    def init_entity_embeddings(self, embedding_dim=None):
        pass

    def init_relation_embeddings(self, embedding_dim=None):
        pass

    def get_embeddings(self):
        raise NotImplementedError('ULTRA has query-conditioned representations, not static embedding tables')

    def set_graph(self, triples, num_entities=None, num_relations=None, inverse_relations=None):
        """Attach training facts; inverse_relations maps direct DICE IDs to inverse IDs.

        Without a mapping every supplied relation is treated as a direct relation.
        Repeated facts and explicitly supplied inverse facts are deduplicated.
        """
        device = self.device
        triples = torch.as_tensor(triples, dtype=torch.long, device=device).clone()
        ne = self.num_entities if num_entities is None else num_entities
        nr = self.num_relations if num_relations is None else num_relations
        if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples):
            raise ValueError('ULTRA requires a nonempty [N, 3] training graph')
        if ne is None or nr is None or ne < 1 or nr < 1:
            raise ValueError('Positive vocabulary sizes are required')
        if triples.min() < 0 or triples[:, [0, 2]].max() >= ne or triples[:, 1].max() >= nr:
            raise ValueError('Graph IDs are outside the supplied vocabulary')
        pairs = {int(k): int(v) for k, v in (inverse_relations or {}).items()}
        if (len(set(pairs.values())) != len(pairs) or set(pairs) & set(pairs.values())
                or any(min(k, v) < 0 or max(k, v) >= nr for k, v in pairs.items())):
            raise ValueError('Inverse relation mapping must contain disjoint, valid direct/inverse pairs')
        direct = sorted(set(range(nr)) - set(pairs.values()))
        num_direct = len(direct)
        mapping = torch.empty(nr, dtype=torch.long, device=device)
        for internal, external in enumerate(direct):
            mapping[external] = internal
            if external in pairs:
                mapping[pairs[external]] = internal + num_direct
        converted = triples.clone()
        converted[:, 1] = mapping[converted[:, 1]]
        inverse = converted[:, [2, 1, 0]].clone()
        inverse[:, 1] = (inverse[:, 1] + num_direct) % (2 * num_direct)
        graph = torch.cat((converted, inverse)).unique(dim=0)
        rel_edges, rel_types = build_relation_graph(graph[:, [0, 2]].T, graph[:, 1], ne, 2 * num_direct)
        self.num_entities, self.num_relations = int(ne), int(nr)
        self.num_direct_relations = num_direct
        self.graph_triples = graph
        self.relation_id_map = mapping
        self.edge_index, self.edge_type = graph[:, [0, 2]].T, graph[:, 1]
        self.rel_edge_index, self.rel_edge_type = rel_edges, rel_types
        return self

    def save_graph(self, path):
        self._require_graph()
        torch.save(dict(version=1, num_entities=self.num_entities, num_relations=self.num_relations,
                        num_direct_relations=self.num_direct_relations,
                        graph_triples=self.graph_triples.cpu(), relation_id_map=self.relation_id_map.cpu()), path)

    def load_graph(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f'ULTRA requires its graph artifact: {path}')
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('version') != 1:
            raise ValueError('Unsupported ULTRA graph artifact version')
        if (data['num_entities'], data['num_relations']) != (self.num_entities, self.num_relations):
            raise ValueError('ULTRA graph vocabulary does not match the experiment')
        self.num_direct_relations = data['num_direct_relations']
        self.graph_triples = data['graph_triples'].to(self.device)
        self.relation_id_map = data['relation_id_map'].to(self.device)
        self.edge_index, self.edge_type = self.graph_triples[:, [0, 2]].T, self.graph_triples[:, 1]
        self.rel_edge_index, self.rel_edge_type = build_relation_graph(
            self.edge_index, self.edge_type, self.num_entities, 2 * self.num_direct_relations)
        return self

    def load_pretrained(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = state.get('model', state)
        expected = self.state_dict()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected if k in state):
            raise ValueError('Checkpoint architecture mismatch: official ULTRA requires ultra_dim=64, ultra_num_layers=6; all keys and shapes must match')
        self.load_state_dict(state, strict=True)
        return self

    def _require_graph(self):
        if self.graph_triples is None:
            raise RuntimeError('Attach a training graph with set_graph() before scoring ULTRA')

    def _convert(self, triples):
        self._require_graph()
        triples = triples.to(device=self.device, dtype=torch.long).clone()
        if triples.numel() and (triples.min() < 0 or triples[..., [0, 2]].max() >= self.num_entities
                              or triples[..., 1].max() >= self.num_relations):
            raise ValueError('Query IDs are outside the graph vocabulary')
        triples[..., 1] = self.relation_id_map[triples[..., 1]]
        return triples

    def _training_edges(self, targets=None, queries=None):
        if not self.training:
            return self.edge_index, self.edge_type
        graph = self.graph_triples
        if queries is not None:
            # Hide all observed positive tails for each query, even for sampled
            # objectives. Candidate lists may include every entity, so they
            # must not be interpreted as a list of targets to remove.
            keys = graph[:, 0] * (2 * self.num_direct_relations) + graph[:, 1]
            qkeys = queries[:, 0] * (2 * self.num_direct_relations) + queries[:, 1]
            targets = graph[torch.isin(keys, qkeys)]
        targets = targets.reshape(-1, 3)
        inverse = targets[:, [2, 1, 0]].clone()
        inverse[:, 1] = (inverse[:, 1] + self.num_direct_relations) % (2 * self.num_direct_relations)
        targets = torch.cat((targets, inverse))
        def keys(t):
            return (t[:, 0] * (2 * self.num_direct_relations) + t[:, 1]) * self.num_entities + t[:, 2]
        keep = ~torch.isin(keys(graph), keys(targets))
        return self.edge_index[:, keep], self.edge_type[keep]

    def _score(self, heads, relations, candidates, query_relations, edges):
        output = []
        for start in range(0, len(heads), self.query_batch_size):
            sl = slice(start, start + self.query_batch_size)
            rels = self.relation_model(self.rel_edge_index, self.rel_edge_type,
                                       2 * self.num_direct_relations, query_relations[sl])
            output.append(self.entity_model(*edges, self.num_entities, rels, heads[sl], relations[sl], candidates[sl]))
        return torch.cat(output) if output else self.relation_model.layers[0].linear.weight.new_empty((0, candidates.shape[1]))

    def forward_grouped(self, triples, head_prediction=None):
        triples = self._convert(triples)
        h, r, t = triples.unbind(-1)
        if head_prediction is None:
            head_prediction = ~(h == h[:, :1]).all(dim=1)
        elif isinstance(head_prediction, bool):
            head_prediction = torch.full((len(h),), head_prediction, device=h.device)
        if not ((r == r[:, :1]).all() and torch.where(head_prediction, (t == t[:, :1]).all(1), (h == h[:, :1]).all(1)).all()):
            raise ValueError('Each negative group must share a relation and an uncorrupted endpoint')
        edges = self._training_edges(targets=triples)
        heads = torch.where(head_prediction, t[:, 0], h[:, 0])
        relations = (r[:, 0] + head_prediction.long() * self.num_direct_relations) % (2 * self.num_direct_relations)
        candidates = torch.where(head_prediction[:, None], h, t)
        # DICE reciprocal queries share their original direct relation seed.
        # Upstream conditions relation reasoning on the original relation before
        # transforming head corruption to inverse-relation tail prediction.
        return self._score(heads, relations, candidates, r[:, 0] % self.num_direct_relations, edges)

    def forward_triples(self, x):
        triples = self._convert(x)
        edges = self._training_edges(targets=triples)
        # Reuse message passing for identical (h,r), without changing query order.
        queries, inverse = triples[:, :2].unique(dim=0, return_inverse=True)
        result = self.relation_model.layers[0].linear.weight.new_empty(len(x))
        for start in range(0, len(queries), self.query_batch_size):
            q = queries[start:start + self.query_batch_size]
            indices = [(inverse == i).nonzero().flatten() for i in range(start, start + len(q))]
            padded = nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=-1)
            candidates = triples[padded.clamp_min(0), 2]
            scores = self._score(q[:, 0], q[:, 1], candidates, q[:, 1] % self.num_direct_relations, edges)
            valid = padded >= 0
            result = result.index_copy(0, padded[valid], scores[valid])
        return result

    def forward_k_vs_sample(self, x, target_entity_idx):
        self._require_graph()
        x = x.to(device=self.device, dtype=torch.long)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError('Tail queries must have shape [B, 2] in (head, relation) order')
        if x.numel() and (x.min() < 0 or x[:, 0].max() >= self.num_entities or x[:, 1].max() >= self.num_relations):
            raise ValueError('Query IDs are outside the graph vocabulary')
        queries = torch.stack((x[:, 0], self.relation_id_map[x[:, 1]]), dim=1)
        candidates = target_entity_idx.to(device=self.device, dtype=torch.long)
        if candidates.ndim == 1:
            candidates = candidates.expand(len(x), -1)
        if candidates.ndim != 2 or candidates.shape[0] != len(x):
            raise ValueError('Candidate IDs must have shape [K] or [B, K]')
        if candidates.numel() and (candidates.min() < 0 or candidates.max() >= self.num_entities):
            raise ValueError('Candidate IDs are outside the graph vocabulary')
        return self._score(queries[:, 0], queries[:, 1], candidates, queries[:, 1] % self.num_direct_relations, self._training_edges(queries=queries))

    def forward_k_vs_all(self, x):
        self._require_graph()
        return self.forward_k_vs_sample(x, torch.arange(self.num_entities, device=self.device).expand(len(x), -1))

    def forward_k_vs_all_heads(self, x, target_entity_idx=None):
        """Score head candidates for DICE pairs (relation, tail)."""
        self._require_graph()
        x = x.to(device=self.device, dtype=torch.long)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError('Head queries must have shape [B, 2] in (relation, tail) order')
        candidates = torch.arange(self.num_entities, device=self.device) if target_entity_idx is None else target_entity_idx.to(device=self.device, dtype=torch.long)
        if candidates.ndim == 1:
            candidates = candidates.expand(len(x), -1)
        triples = torch.stack((candidates, x[:, 0, None].expand_as(candidates), x[:, 1, None].expand_as(candidates)), -1)
        return self.forward_grouped(triples, head_prediction=True)

    def forward(self, x, y_idx=None):
        if isinstance(x, tuple):
            return self.forward_k_vs_sample(*x)
        if y_idx is not None:
            return self.forward_k_vs_sample(x, y_idx)
        if x.ndim == 3:
            return self.forward_grouped(x)
        if x.ndim == 2 and x.shape[1] == 2:
            return self.forward_k_vs_all(x)
        if x.ndim == 2 and x.shape[1] == 3:
            return self.forward_triples(x)
        raise ValueError('Expected [B,3] triples, [B,2] queries, or [B,K,3] negative groups')
