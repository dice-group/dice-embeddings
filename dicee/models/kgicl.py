"""Pure PyTorch KG-ICL, compatible with the official nju-websoft/KG-ICL checkpoints.

Reimplementation of Cui et al., "A Prompt-Based Knowledge Graph Foundation
Model for Universal In-Context Reasoning" (NeurIPS 2024,
https://arxiv.org/abs/2410.12288), verified against
https://github.com/nju-websoft/KG-ICL at UPSTREAM_COMMIT. Module names follow
the released ``model_best.tar`` state dictionaries. Prompt graphs are runtime
context derived from the attached graph (see kgicl_prompts.py), never part of
the transferable state. docs/kgicl.md documents the corrected upstream defects.
"""
import hashlib
from collections import OrderedDict
from typing import Optional, Sequence, cast

import torch
from torch import nn
from torch.nn import functional as F

from ._fused_kgicl import ATTENTION, fused_reasoning_supported, fused_scores, layer_tables
from ._inference import candidate_features, candidate_slice, host_ids, inference_only
from .graph_model import GraphKGE
from .kgicl_prompts import PromptGraph, PromptSampler

UPSTREAM_COMMIT = "6a3166e347ae468acdfb30a70a2cf3608b66b8f1"
# nn.RReLU's default bounds; in evaluation its negative slope is their mean.
RRELU_LOWER, RRELU_UPPER = 1. / 8, 1. / 3


def _split_linear(linear, *parts: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """``linear(cat([table[index] for table, index in parts], -1))`` without the concatenation.

    Each table is projected by its block of the weight before the per-edge gather.
    """
    dim = parts[0][0].shape[-1]
    value = F.linear(parts[0][0], linear.weight[:, :dim])[parts[0][1]]
    for k, (table, index) in enumerate(parts[1:], 1):
        value = value.add_(F.linear(table, linear.weight[:, k * dim:(k + 1) * dim])[index])
    return value.add_(linear.bias)


class PromptEncoder(nn.Module):
    """Upstream ``PromptEncoder`` with the released configuration.

    Settings: unified tokens, concatenated entity messages, sigmoid attention,
    max aggregation for entities and relations. Relation tokens start at zero
    except the query relation. ``ent_norm`` multiplies messages twice, as in
    the released model, so the checkpoints keep their trained behavior.
    """

    def __init__(self, dim=32, num_layers=3, hops=3):
        super().__init__()
        self.dim, self.num_layers, self.hops = dim, num_layers, hops
        self.start_relation_embeddings = nn.Embedding(1, dim)
        self.position_embedding = nn.Embedding((hops + 1) ** 2, dim)
        self.self_loop_embedding = nn.Embedding(1, dim)
        for embedding in (self.start_relation_embeddings, self.position_embedding, self.self_loop_embedding):
            nn.init.xavier_normal_(embedding.weight.data)
        self.act = nn.RReLU()
        self.W_ht2r = nn.ModuleList([nn.Linear(3 * dim, dim) for _ in range(num_layers)])
        self.W_message = nn.ModuleList([nn.Linear(3 * dim, dim) for _ in range(num_layers)])
        self.alpha = nn.ModuleList([nn.Linear(2 * dim, 1) for _ in range(num_layers)])
        self.beta = nn.ModuleList([nn.Linear(2 * dim, 1) for _ in range(num_layers)])
        self.loop_transfer = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(num_layers)])
        self.ent_transfer = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.rel_transfer = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.final_to_rel_embeddings = nn.Linear(dim * num_layers, dim)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm_rels = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers + 1)])
        self.layer_norm_ents = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers + 1)])
        self.layer_norm_loop = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers + 1)])

    @staticmethod
    def _max(values, index, size):
        # torch_scatter.scatter_max semantics: empty groups are zero.
        out = values.new_zeros(size, values.shape[-1])
        return out.scatter_reduce(0, index[:, None].expand_as(values), values, 'amax', include_self=False)

    def forward(self, graphs: Sequence[PromptGraph], relation: int, num_relations: int) -> torch.Tensor:
        """Mean prompt representation ``[num_relations + 1, dim]``; the last row is the self-loop relation."""
        device = self.position_embedding.weight.device
        shots = len(graphs)
        sizes = torch.tensor([g.num_nodes for g in graphs])
        node_offsets = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
        num_nodes = int(sizes.sum())
        edge_index = torch.cat([g.edge_index + offset for g, offset in zip(graphs, node_offsets.tolist())], 1).to(device)
        slots = torch.arange(shots) * num_relations
        edge_type = torch.cat([g.edge_type + slot for g, slot in zip(graphs, slots.tolist())]).to(device)
        edge_query = torch.cat([torch.full((g.edge_type.numel(),), relation + slot) for g, slot in zip(graphs, slots.tolist())]).to(device)
        labels = torch.cat([g.labels for g in graphs]).to(device)
        heads = (node_offsets + torch.tensor([g.head for g in graphs])).to(device)
        tails = (node_offsets + torch.tensor([g.tail for g in graphs])).to(device)
        query_slots = (relation + slots).to(device)

        node = self.position_embedding.weight[labels[:, 0] * (self.hops + 1) + labels[:, 1]]
        node = node.index_put((heads,), self.position_embedding.weight[0].expand(shots, -1))
        node = node.index_put((tails,), self.position_embedding.weight[1].expand(shots, -1))
        rel = node.new_zeros(num_relations * shots, self.dim)
        rel = rel.index_put((query_slots,), self.start_relation_embeddings.weight[0].expand(shots, -1))
        loop = self.self_loop_embedding.weight.expand(shots, -1)
        source, target = edge_index
        degree = torch.zeros(num_nodes, device=device).index_add_(0, target, torch.ones_like(target, dtype=node.dtype))
        inverse_sqrt = degree.pow(-0.5)
        inverse_sqrt = inverse_sqrt.masked_fill(inverse_sqrt == float('inf'), 0)
        norm = (inverse_sqrt[target] * inverse_sqrt[source])[:, None]
        layers = []
        for i in range(self.num_layers):
            if self.training:
                # RReLU samples a slope per element, so training keeps upstream's per-edge inputs.
                r, q = rel[edge_type], rel[edge_query]
                message = self.act(self.W_message[i](torch.cat((node[source], r, q), -1)))
                alpha = torch.sigmoid(self.alpha[i](self.act(torch.cat((r, q), -1))))
                message = message * alpha * norm * norm
            else:
                # Project entity and relation tables before gathering them per edge; this
                # avoids [edges, 3 * dim] inputs for the hub-sized prompt graphs of large KGs.
                message = self.act(_split_linear(self.W_message[i], (node, source), (rel, edge_type), (rel, edge_query)))
                active = self.act(rel)
                alpha = torch.sigmoid(_split_linear(self.alpha[i], (active, edge_type), (active, edge_query)))
                message = message.mul_(alpha).mul_(norm).mul_(norm)
            node = self.act(self.ent_transfer[i](self._max(message, target, num_nodes)))
            node = self.layer_norm_ents[i](node)
            if self.training:
                message = self.act(self.W_ht2r[i](torch.cat((node[source], node[target], q), -1)))
                message = message * torch.sigmoid(self.beta[i](torch.cat((r, q), -1)))
            else:
                message = self.act(_split_linear(self.W_ht2r[i], (node, source), (node, target), (rel, edge_query)))
                message = message.mul_(torch.sigmoid(_split_linear(self.beta[i], (rel, edge_type), (rel, edge_query))))
            rel = self.act(self.rel_transfer[i](self._max(message, edge_type, len(rel)))) + rel
            rel = self.layer_norm_rels[i](rel)
            layers.append(rel)
            loop = loop + self.act(self.loop_transfer[i](torch.cat((loop, rel[query_slots]), -1)))
            loop = self.layer_norm_loop[i](loop)
        final = self.layer_norm_rels[-1](self.act(self.final_to_rel_embeddings(torch.cat(layers, -1))))
        final = self.dropout(final.view(shots, num_relations, self.dim).mean(0))
        return torch.cat((final, loop.mean(0, keepdim=True)))


class _UnusedConv(nn.Module):
    """Parameters of the official checkpoints' unused NBFNet layer, kept for strict loading."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(2 * dim, dim)
        self.relation_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))


class GNNLayer(nn.Module):
    """Query-aware attention with TransE-style ``h + r`` messages and sum aggregation."""

    def __init__(self, dim=32, attn_dim=5):
        super().__init__()
        self.inference_backend = 'auto'
        self.Ws_attn = nn.Linear(dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.conv = _UnusedConv(dim)
        self.act = nn.RReLU()

    def forward(self, hidden, source, edge_batch, edge_relation, target, num_targets, relations, query):
        hs = hidden[source]
        hr = relations[edge_batch, edge_relation]
        hq = relations[edge_batch, query[edge_batch]]
        # Upstream constructs a fresh nn.RReLU per call, which always samples
        # random slopes. Evaluation uses the module's deterministic mean slope.
        attention = F.rrelu(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(hq), RRELU_LOWER, RRELU_UPPER, self.training)
        alpha = torch.sigmoid(self.w_alpha(attention))
        message = alpha * (hs + hr)
        aggregate = hidden.new_zeros(num_targets, hidden.shape[-1]).index_add_(0, target, message)
        return self.act(self.W_h(aggregate))


class KGICL(GraphKGE):
    """KG-ICL entity prediction with the official state dictionary and DICE triple order.

    Prompt graphs for a query relation are drawn deterministically from the
    attached graph; their encodings are cached per relation. The reasoner
    expands one hop per layer from the query head, as upstream does: entities
    outside the reached set have no state and score zero.
    """

    name = 'KGICL'
    config_prefix = 'kgicl'
    graph_filename = 'kgicl_graph.pt'
    checkpoint_hint = ('official KG-ICL-6L requires kgicl_dim=32, kgicl_attn_dim=5, kgicl_num_layers=6, '
                       'kgicl_prompt_layers=3, kgicl_prompt_hops=3; all keys and shapes must match')
    deterministic_inference = True

    def __init__(self, args):
        super().__init__(args)
        self.dim = args.get('kgicl_dim', 32)
        self.attn_dim = args.get('kgicl_attn_dim', 5)
        self.num_layers = args.get('kgicl_num_layers', 6)
        self.prompt_layers = args.get('kgicl_prompt_layers', 3)
        self.prompt_hops = args.get('kgicl_prompt_hops', 3)
        self.shots = args.get('kgicl_shots', 5)
        self.open_nodes = args.get('kgicl_prompt_open_nodes', 50)
        self.prompt_seed = args.get('kgicl_prompt_seed', 0)
        masked = args.get('kgicl_masked_distances')
        self.masked_distances = tuple(sorted(set(int(d) for d in masked))) if masked else ()
        if min(self.dim, self.attn_dim, self.num_layers, self.prompt_layers, self.prompt_hops, self.shots) < 1:
            raise ValueError('KG-ICL dimensions, layers, hops and shots must be positive')
        if self.open_nodes < 0 or any(d < 0 for d in self.masked_distances):
            raise ValueError('KG-ICL open nodes and masked distances must be nonnegative')
        if not isinstance(self.prompt_seed, int) or not 0 <= self.prompt_seed < 2 ** 63:
            raise ValueError('kgicl_prompt_seed must be a nonnegative integer less than 2**63')
        self.relation_encoder = PromptEncoder(self.dim, self.prompt_layers, self.prompt_hops)
        self.gnn_layers = nn.ModuleList([GNNLayer(self.dim, self.attn_dim) for _ in range(self.num_layers)])
        self.rel_transfer = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(self.num_layers)])
        self.layer_norms_rel = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(self.num_layers)])
        # Unused by the released model's forward pass; retained for strict checkpoint loading.
        self.layer_norms_query = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(self.num_layers)])
        self.query_transfer = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.num_layers)])
        self.W_score = nn.ModuleList([nn.Linear(2 * self.dim, 1, bias=False) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(args.get('kgicl_dropout', 0.0))
        self.W_final = nn.Linear(self.dim, 1, bias=False)
        self.gate = nn.GRU(self.dim, self.dim)
        # Prompt encodings and the fused path's per-layer relation tables, like ULTRA's caches.
        self.relation_cache_mb = args.get('graph_relation_cache_mb', 64)
        self.projection_cache_mb = args.get('graph_projection_cache_mb', 64)
        if min(self.relation_cache_mb, self.projection_cache_mb) < 0:
            raise ValueError('Graph cache limits must be nonnegative')
        self._prompt_overrides: dict[int, list[PromptGraph]] = {}
        self._sampler: Optional[PromptSampler] = None
        self.set_inference_backend(args.get('graph_inference_backend', 'auto'))

    # ----- checkpoint, graph and cache lifecycle -----

    def load_pretrained(self, path):
        """Load an official ``model_best.tar`` (``state_dict`` key), a ``model`` wrapper or a bare state."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        state = state.get('model', state)
        expected = self.state_dict()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError(f'Checkpoint architecture mismatch for {self.name}: {self.checkpoint_hint}')
        self.load_state_dict(state, strict=True)
        return self

    def clear_inference_cache(self):
        self._prompt_cache = OrderedDict()
        self._prompt_cache_token = None
        self._table_cache = OrderedDict()
        self._table_cache_token = None
        self._layout = None
        self._destination_layout = None

    def _build_relation_graph(self):
        # KG-ICL samples prompt graphs lazily; no relation graph is constructed.
        self.clear_inference_cache()
        self._sampler = None
        self._prompt_overrides = {}

    def prompt_settings(self):
        """Settings that determine prompt graphs and score masks, for cache identities."""
        return (self.shots, self.prompt_hops, self.open_nodes, self.prompt_seed, self.masked_distances,
                tuple(sorted((q, id(g)) for q, g in self._prompt_overrides.items())))

    def inference_token(self):
        token = super().inference_token()
        return None if token is None else (*token, self.prompt_settings())

    def prompt_identity(self) -> dict:
        """Stable description of prompt sampling for persistent score caches.

        Replayed prompts are identified by a digest of their tensors.
        """
        digest = None
        if self._prompt_overrides:
            content = hashlib.sha256()
            for relation in sorted(self._prompt_overrides):
                for graph in self._prompt_overrides[relation]:
                    content.update(repr((relation, graph.head, graph.tail)).encode())
                    for tensor in (graph.edge_index, graph.edge_type, graph.labels):
                        content.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            digest = content.hexdigest()
        return dict(sampler='kgicl-prompt-v1', shots=self.shots, hops=self.prompt_hops, open_nodes=self.open_nodes,
                    seed=self.prompt_seed, masked_distances=list(self.masked_distances), replayed=digest)

    @property
    def _entities(self) -> int:
        """Entity count of the attached graph; scoring and sampling run only after ``set_graph``."""
        return cast(int, self.num_entities)

    @property
    def sampler(self) -> PromptSampler:
        """Prompt sampler of the attached graph, created on first use."""
        self._require_graph()
        if self._sampler is None:
            mapping = self.relation_id_map.detach().cpu().tolist()
            public = {internal: external for external, internal in enumerate(mapping)}
            self._sampler = PromptSampler(self.graph_triples, self._entities, self.num_direct_relations,
                                          [public[r] for r in range(self.num_direct_relations)], shots=self.shots,
                                          hops=self.prompt_hops, open_nodes=self.open_nodes, seed=self.prompt_seed)
        return self._sampler

    def use_prompts(self, prompts: Optional[dict] = None):
        """Replay fixed prompt graphs, keyed by internal query relation; ``None`` restores sampling.

        Each value lists one ``PromptGraph`` per shot. Attaching a graph clears the replay.
        """
        prompts = dict(prompts or {})
        if any(len(graphs) < 1 for graphs in prompts.values()):
            raise ValueError('Replayed prompts need at least one graph per relation')
        self._prompt_overrides = {int(q): list(graphs) for q, graphs in prompts.items()}
        self.clear_inference_cache()
        return self

    def prompt_graphs(self, relation: int) -> list:
        """Prompt graphs of an internal query relation; training draws examples with replacement."""
        if relation in self._prompt_overrides:
            return self._prompt_overrides[relation]
        sampler = self.sampler
        slots = None
        if self.training:
            # Upstream training picks one of the loaded examples at random for each shot.
            count = len(sampler.examples(relation % self.num_direct_relations))
            if count:
                slots = torch.randint(count, (self.shots,)).tolist()
        return sampler.prompts(relation, slots)

    def encode_prompts(self, relation: int, graphs: Optional[Sequence[PromptGraph]] = None) -> torch.Tensor:
        """Prompt representation ``[2R + 1, dim]`` of an internal query relation."""
        self._require_graph()
        graphs = self.prompt_graphs(relation) if graphs is None else graphs
        encoded: torch.Tensor = self.relation_encoder(graphs, relation, 2 * self.num_direct_relations)
        return encoded

    def _prompts(self, relations):
        """Per-query prompt representations, cached by relation during inference."""
        ids = host_ids(relations)
        token = self.inference_token() if inference_only(self) and self.relation_cache_mb else None
        if token is None:
            values = {q: self.encode_prompts(q) for q in dict.fromkeys(ids)}
            return values, ids
        if token != self._prompt_cache_token:
            self._prompt_cache.clear()
            self._prompt_cache_token = token
        capacity = int(self.relation_cache_mb * 2**20) // ((2 * self.num_direct_relations + 1) * self.dim
                                                         * next(self.parameters()).element_size())
        values = {}
        for q in dict.fromkeys(ids):
            if q not in self._prompt_cache:
                self._prompt_cache[q] = self.encode_prompts(q).detach()
            self._prompt_cache.move_to_end(q)
            values[q] = self._prompt_cache[q]
        # Keep current-call values alive even when they exceed the LRU capacity.
        while len(self._prompt_cache) > max(capacity, 1):
            self._prompt_cache.popitem(last=False)
        return values, ids

    def _layer_tables(self, relations, prompts):
        """Stacked per-layer relation tables of the fused path, cached per relation like prompts."""
        token = self._prompt_cache_token
        size = (self.num_layers * (2 * self.num_direct_relations + 1) * (self.dim + ATTENTION)
                * next(self.parameters()).element_size())
        capacity = int(self.projection_cache_mb * 2**20) // size
        if token is None or not capacity:
            values = {q: layer_tables(self, q, prompts[q]) for q in relations}
        else:
            if token != self._table_cache_token:
                self._table_cache.clear()
                self._table_cache_token = token
            values = {}
            for q in relations:
                if q not in self._table_cache:
                    self._table_cache[q] = layer_tables(self, q, prompts[q])
                self._table_cache.move_to_end(q)
                values[q] = self._table_cache[q]
            while len(self._table_cache) > max(capacity, 1):
                self._table_cache.popitem(last=False)
        return [tuple(torch.stack([values[q][i][k] for q in relations]) for k in range(3))
                for i in range(self.num_layers)]

    # ----- reasoning -----

    def _out_edges(self, edges):
        """Edges grouped by source entity: offsets [N + 1], targets, types."""
        cached = self._layout
        if cached is not None and cached[0] is edges[0] and cached[1] is edges[1]:
            return cached[2]
        edge_index, edge_type = edges
        order = edge_index[0].argsort(stable=True)
        counts = torch.bincount(edge_index[0], minlength=self._entities)
        layout = (torch.cat((counts.new_zeros(1), counts.cumsum(0))), edge_index[1, order], edge_type[order])
        if not self.training:
            self._layout = (edge_index, edge_type, layout)
        return layout

    def _expand_scores(self, heads, relations, prompts, edges):
        """Upstream's hop-by-hop reasoning for a batch; returns ``[B, N]`` scores and first-reach depths."""
        n, dim = self._entities, self.dim
        offsets, targets, types = self._out_edges(edges)
        self_loop = 2 * self.num_direct_relations
        batch = torch.arange(len(heads), device=heads.device)
        node_batch, node_entity = batch, heads
        hidden = prompts[batch, relations]
        rel, state = prompts, None
        depth = heads.new_zeros(len(heads))
        for i, layer in enumerate(self.gnn_layers):
            counts = offsets[node_entity + 1] - offsets[node_entity]
            source = torch.arange(len(node_entity), device=heads.device).repeat_interleave(counts)
            starts = offsets[node_entity].repeat_interleave(counts)
            position = starts + torch.arange(len(source), device=heads.device) - (counts.cumsum(0) - counts).repeat_interleave(counts)
            loops = torch.arange(len(node_entity), device=heads.device)
            source = torch.cat((source, loops))
            edge_relation = torch.cat((types[position], torch.full_like(loops, self_loop)))
            edge_target = torch.cat((targets[position], node_entity))
            edge_batch = node_batch[source]
            keys, target = (edge_batch * n + edge_target).unique(sorted=True, return_inverse=True)
            previous = torch.searchsorted(keys, node_batch * n + node_entity)
            rel = self.layer_norms_rel[i](rel + F.relu(self.rel_transfer[i](rel)))
            hidden = layer(hidden, source, edge_batch, edge_relation, target, len(keys), rel, relations)
            initial = hidden.new_zeros(len(keys), dim)
            if state is not None:
                initial = initial.index_copy(0, previous, state)
            hidden = self.layer_norms[i](self.dropout(hidden))
            hidden, state = self.gate(hidden.unsqueeze(0), initial.unsqueeze(0))
            hidden, state = hidden.squeeze(0), state.squeeze(0)
            rel = self.layer_norms_rel[i](rel)
            depth = torch.full_like(keys, i + 1).index_copy(0, previous, depth)
            node_batch, node_entity = keys // n, keys % n
        scores = hidden.new_zeros(len(heads), n).index_put((node_batch, node_entity), self.W_final(hidden).squeeze(-1))
        reached = torch.full((len(heads), n), -1, dtype=torch.long, device=heads.device)
        reached = reached.index_put((node_batch, node_entity), depth)
        return scores, reached

    def _mask_distances(self, scores, depth, evaluation):
        """Zero entities first reached at masked distances; upstream masks heads also in training."""
        masked = [d for d in self.masked_distances if evaluation or d == 0]
        if masked and depth is not None:
            scores = scores.masked_fill(torch.isin(depth, torch.tensor(masked, device=depth.device)), 0)
        return scores

    def _all_scores(self, heads, relations, edges, fused):
        values, ids = self._prompts(relations)
        if fused:
            scores, depth = fused_scores(self, heads, relations, ids, values, edges)
        else:
            prompts = torch.stack([values[q] for q in ids])
            scores, depth = self._expand_scores(heads, relations, prompts, edges)
        return self._mask_distances(scores, depth, not self.training)

    def _score(self, heads, relations, candidates, query_relations, edges):
        # Query conditioning uses the actual (possibly inverse) relation, as upstream does.
        self._require_graph()
        output = []
        inference = inference_only(self)
        fused = inference and fused_reasoning_supported(self, edges)
        # Batched GPU reasoning is row-wise; the PyTorch path scores each query
        # alone during inference, so a row never depends on its batch.
        chunk = self.query_batch_size if fused or not inference else 1
        ids = host_ids(relations)
        for start in range(0, len(heads), chunk):
            sl = slice(start, start + chunk)
            part = relations[sl]
            part._dicee_ids = ids[sl]
            scores = self._all_scores(heads[sl], part, edges, fused)
            output.append(candidate_features(scores[..., None], candidate_slice(candidates, sl)).squeeze(-1))
        return torch.cat(output) if output else next(self.parameters()).new_empty((0, candidates.shape[1]))
