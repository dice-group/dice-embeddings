"""Pure PyTorch Flock, compatible with both official jw9730/flock checkpoints.

Architecture adapted from https://github.com/jw9730/flock at UPSTREAM_COMMIT.
See docs/flock.md for random-walk and numerical parity details.

MIT License

Copyright (c) 2025 Jinwoo Kim

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
import torch
from torch import nn

from .flock_walks import WalkGraph
from .graph_model import GraphKGE, RelationGraphKGE

UPSTREAM_COMMIT = "f35103d25a78bdf4075de5c673a51de4979aa4d7"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = dim * ((int(8 * dim / 3) + dim - 1) // dim)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class BidirectionalGRU(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.gru_norm = RMSNorm(dim)
        self.gru = nn.GRU(dim, dim, num_layers=layers, batch_first=True, bidirectional=True)
        self.gru_out = nn.Linear(2 * dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = FeedForward(dim)

    def forward(self, x):
        x = x + self.gru_out(self.gru(self.gru_norm(x))[0])
        return x + self.feed_forward(self.ffn_norm(x))


def consensus(x, logits, ids, num_ids):
    """Stable multihead softmax pooling over all occurrences of each graph ID."""
    batch, samples, length = ids.shape
    dim, heads = x.shape[-1], logits.shape[-1]
    index = (ids + torch.arange(batch, device=ids.device)[:, None, None] * num_ids).reshape(-1, 1)
    logits = logits.reshape(-1, heads)
    maxima = x.new_zeros(batch * num_ids, heads)
    maxima.scatter_reduce_(0, index.expand(-1, heads), logits, reduce="amax", include_self=False)
    weights = (logits - maxima.gather(0, index.expand(-1, heads))).exp()
    weighted = (x.reshape(-1, heads, dim // heads) * weights[..., None]).flatten(1)
    sums, totals = x.new_zeros(batch * num_ids, dim), x.new_zeros(batch * num_ids, heads)
    sums.scatter_reduce_(0, index.expand(-1, dim), weighted, reduce="sum", include_self=False)
    totals.scatter_reduce_(0, index.expand(-1, heads), weights, reduce="sum", include_self=False)
    return (sums.view(batch * num_ids, heads, dim // heads) / (totals[..., None] + 1e-8)).view(batch, num_ids, dim)


class FlockBase(GraphKGE):
    config_prefix = "flock"
    graph_filename = "flock_graph.pt"
    relation_prediction = False
    checkpoint_hint = "official Flock requires dim=64, walk_len=128, refinements=6, num_layers=1, attention_heads=4 and the matching task"

    # Registered by name below to preserve the official checkpoint keys.
    emb_anon_node: nn.ModuleList
    emb_anon_type: nn.ModuleList
    emb_restart: nn.ModuleList
    emb_neighbor: nn.ModuleList
    emb_direction: nn.ModuleList
    emb_head_is_query: nn.ModuleList
    emb_tail_is_query: nn.ModuleList
    emb_node_is_query: nn.ModuleList
    emb_type_is_query: nn.ModuleList
    from_node: nn.ModuleList
    from_type: nn.ModuleList
    to_node: nn.ModuleList
    to_type: nn.ModuleList
    node_logit: nn.ModuleList
    type_logit: nn.ModuleList

    def __init__(self, args):
        args = dict(args)
        args.setdefault("flock_query_batch_size", 1)
        super().__init__(args)
        self.dim = args.get("flock_dim", 64)
        self.walk_num = args.get("flock_walk_num", 128)
        self.walk_len = args.get("flock_walk_len", 128)
        self.refinements = args.get("flock_refinements", 6)
        layers = args.get("flock_num_layers", 1)
        heads = args.get("flock_attention_heads", 4)
        self.test_samples = args.get("flock_test_samples", 1)
        self.seed = args.get("flock_seed")
        if min(self.dim, self.walk_num, self.refinements, layers, heads, self.test_samples) < 1 or self.walk_len < 2:
            raise ValueError("Flock dimensions/counts must be positive and walk_len must be at least 2")
        if self.dim % heads:
            raise ValueError("flock_dim must be divisible by flock_attention_heads")
        if self.seed is not None and (not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 63):
            raise ValueError("flock_seed must be a nonnegative integer less than 2**63")
        self.node_init = nn.Parameter(torch.randn(self.dim))
        self.type_init = nn.Parameter(torch.randn(self.dim))
        query_names = ("emb_head_is_query", "emb_tail_is_query") if self.relation_prediction else ("emb_node_is_query", "emb_type_is_query")
        embeddings = dict(emb_anon_node=self.walk_len, emb_anon_type=self.walk_len + 1,
                          emb_restart=2, emb_neighbor=2, emb_direction=4, **dict.fromkeys(query_names, 2))
        for name, size in embeddings.items():
            setattr(self, name, nn.ModuleList([nn.Embedding(size, self.dim) for _ in range(self.refinements)]))
        self.net = nn.ModuleList([BidirectionalGRU(self.dim, layers) for _ in range(self.refinements)])
        for name in ("from_node", "from_type", "to_node", "to_type", "node_logit", "type_logit"):
            output_dim = heads if name.endswith("logit") else self.dim
            setattr(self, name, nn.ModuleList([nn.Linear(self.dim, output_dim) for _ in range(self.refinements)]))
        if self.relation_prediction:
            self.head_to_query = nn.Linear(self.dim, self.dim)
            self.tail_to_query = nn.Linear(self.dim, self.dim)
        self.head = nn.Sequential(nn.Linear(self.dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self._walk_graph: WalkGraph | None = None

    def _build_relation_graph(self):
        # Flock uses KG walks, without constructing a separate relation graph.
        num_entities, _ = self._require_graph()
        self._walk_graph = WalkGraph(self.edge_index, self.edge_type, num_entities, 2 * self.num_direct_relations)

    def _generator(self):
        return None if self.seed is None else torch.Generator().manual_seed(self.seed)

    def _draw_walks(self, graph, heads, tails, generator):
        return graph.sample(heads, tails, self.walk_num, self.walk_len, self.refinements, generator)

    def sample_walks(self, heads, tails=None, edges=None, generator=None):
        """Sample official-format records; IDs use the attached graph vocabulary.

        Returns (nodes, anonymous nodes, restarts, neighbors, relations,
        anonymous relations, directions), each shaped [T,B,S,L]. Anonymous
        names start at 1; no-relation markers are 2*R and L+1 respectively.
        """
        num_entities, _ = self._require_graph()
        if generator is None:
            generator = self._generator()
        graph = self._walk_graph if edges is None else WalkGraph(edges[0], edges[1], num_entities, 2 * self.num_direct_relations)
        return tuple(record.to(self.device) for record in self._draw_walks(graph, heads, tails, generator))

    def score_walks(self, heads, query, candidates, records):
        """Score fixed walks for parity/replay; query/candidates use internal IDs.

        Entity mode: query=relations, candidates=tails. Relation mode:
        query=tails, candidates=relations. This scores the supplied records;
        callers constructing training records must first remove target edges.
        """
        num_entities, _ = self._require_graph()
        records = tuple(record.to(device=self.device, dtype=torch.long) for record in records)
        if len(records) != 7 or any(record.ndim != 4 or record.shape != records[0].shape for record in records):
            raise ValueError("Expected seven matching [T,B,S,L] walk records")
        steps, batch, samples, length = records[0].shape
        if (steps, batch, length) != (self.refinements, len(heads), self.walk_len) or samples < 1:
            raise ValueError("Walk record dimensions do not match the model and query batch")
        heads, query, candidates = (x.to(self.device, dtype=torch.long) for x in (heads, query, candidates))
        walks, named_walks, restarts, neighbors, types, named_types, directions = records
        nr = 2 * self.num_direct_relations
        h_node = self.node_init[None, None].expand(batch, num_entities, -1)
        h_type = self.type_init[None, None].expand(batch, nr + 1, -1)
        for step in range(steps):
            node_ids, type_ids = walks[step].flatten(1), types[step].flatten(1)
            is_head = (walks[step] == heads[:, None, None]).long()
            if self.relation_prediction:
                is_tail = (walks[step] == query[:, None, None]).long()
                markers = (self.emb_head_is_query[step](is_head), self.emb_tail_is_query[step](is_tail))
            else:
                is_type = (types[step] == query[:, None, None]).long()
                markers = (self.emb_node_is_query[step](is_head), self.emb_type_is_query[step](is_type))
            x = (self.emb_anon_node[step](named_walks[step] - 1)
                 + self.emb_anon_type[step](named_types[step] - 1)
                 + self.emb_restart[step](restarts[step])
                 + self.emb_neighbor[step](neighbors[step])
                 + self.emb_direction[step](directions[step]) + markers[0] + markers[1])
            previous_nodes = h_node.gather(1, node_ids[..., None].expand(-1, -1, self.dim)).view(batch, samples, length, self.dim)
            previous_types = h_type.gather(1, type_ids[..., None].expand(-1, -1, self.dim)).view(batch, samples, length, self.dim)
            x = x + self.from_node[step](previous_nodes) + self.from_type[step](previous_types)
            x = self.net[step](x.view(batch * samples, length, self.dim))
            h_node = h_node + consensus(self.to_node[step](x), self.node_logit[step](x), walks[step], num_entities)
            h_type = h_type + consensus(self.to_type[step](x), self.type_logit[step](x), types[step], nr + 1)
        if self.relation_prediction:
            head = self.head_to_query(h_node.gather(1, heads[:, None, None].expand(-1, 1, self.dim)))
            tail = self.tail_to_query(h_node.gather(1, query[:, None, None].expand(-1, 1, self.dim)))
            features = head + tail + h_type.gather(1, candidates[..., None].expand(-1, -1, self.dim))
        else:
            features = (h_node.gather(1, candidates[..., None].expand(-1, -1, self.dim))
                        + h_type.gather(1, query[:, None, None].expand(-1, 1, self.dim)))
        return self.head(features).squeeze(-1).float()

    def _run_queries(self, heads, query, candidates, edges):
        num_entities, _ = self._require_graph()
        if not len(heads) or not candidates.shape[1]:
            return self.node_init.new_empty((len(heads), candidates.shape[1]))
        repeats = 1 if self.training else self.test_samples
        heads, query, candidates = (x.repeat_interleave(repeats, dim=0) for x in (heads, query, candidates))
        graph = self._walk_graph if not self.training else WalkGraph(edges[0], edges[1], num_entities, 2 * self.num_direct_relations)
        generator = self._generator()
        output = []
        for start in range(0, len(heads), self.query_batch_size):
            sl = slice(start, start + self.query_batch_size)
            tails = query[sl] if self.relation_prediction else None
            records = self._draw_walks(graph, heads[sl], tails, generator)
            output.append(self.score_walks(heads[sl], query[sl], candidates[sl], records))
        return torch.cat(output).view(-1, repeats, candidates.shape[1]).mean(1)


class Flock(FlockBase):
    """Entity-prediction Flock with official ``flock_entity.pth`` parameters."""
    name = "Flock"

    def _score(self, heads, relations, candidates, query_relations, edges):
        # Unlike ULTRA/TRIX, query conditioning occurs after head-to-tail
        # conversion, using the actual (possibly inverse) relation ID.
        return self._run_queries(heads, relations, candidates, edges)


class FlockRelation(FlockBase, RelationGraphKGE):
    """Relation-prediction Flock with official ``flock_relation.pth`` parameters."""
    name = "FlockRelation"
    relation_prediction = True

    def _relation_score(self, pairs, candidates, edges):
        return self._run_queries(pairs[:, 0], pairs[:, 1], candidates, edges)
