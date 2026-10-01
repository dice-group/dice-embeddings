"""CLMPT inference with released transformer and ComplEx checkpoint layouts."""

from collections import defaultdict

import torch
from torch import nn

from .._query import compile_query, validate_tree
from ..context import evaluation_mode
from ._bounded_attention import BoundedAttention
from ._common import QueryMethod, require_complex, union_branches


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.size = dim // heads
        for name in ('linear_q', 'linear_k', 'linear_v', 'output_layer'):
            setattr(self, name, nn.Linear(dim, dim, bias=False))

    def forward(self, value):
        batch, length, dim = value.shape
        q, k, v = [getattr(self, name)(value).view(batch, length, self.heads, self.size).transpose(1, 2)
                   for name in ('linear_q', 'linear_k', 'linear_v')]
        weights = ((q * self.size ** -.5) @ k.transpose(2, 3)).softmax(-1)
        result = (weights @ v).transpose(1, 2).contiguous().view(batch, length, dim)
        return self.output_layer(result)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.layer1, self.layer2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, value):
        return self.layer2(self.layer1(value).relu())


class EncoderLayer(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.self_attention_norm, self.ffn_norm = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.self_attention, self.ffn = Attention(dim), FeedForward(dim, hidden)

    def forward(self, value):
        value = value + self.self_attention(self.self_attention_norm(value))
        return value + self.ffn(self.ffn_norm(value))


class Transformer(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        self.scale = dim ** .5
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([EncoderLayer(dim, hidden) for _ in range(layers)])
        self.encoder.last_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, value):
        value = value * self.scale
        for layer in self.encoder.layers:
            value = layer(value)
        return self.encoder.last_norm(value)


def query_graph(tree):
    """Compile a DNF branch into ordered signed relation atoms."""
    atoms, anchors, variables = [], {}, ['f']

    def visit(node, target):
        op = node[0]
        if op == 'and':
            for child in node[1:]:
                visit(child, target)
            return
        sign = 1
        if op == 'not':
            node, sign = node[1], -1
        if node[0] != 'project':
            raise ValueError('CLMPT requires signed relation atoms')
        child = node[2]
        if child[0] == 'anchor':
            source = f's{len(anchors) + 1}'
            anchors[source] = child[1]
        else:
            source = f'e{len(variables)}'
            variables.append(source)
            visit(child, source)
        atoms.append((source, node[1], target, sign))

    visit(tree, 'f')
    return atoms, anchors, variables


class CLMPT(QueryMethod):
    def __init__(self, model, context, *, hidden_dim=8192, layers=2, pre_norm=True, aggregation='mean',
                 depth_shift=0, candidate_batch_size=4096):
        super().__init__(context)
        require_complex(model, context)
        dim = model.embedding_dim
        if dim % 8 or min(hidden_dim, layers, candidate_batch_size) < 1:
            raise ValueError('CLMPT needs dimensions divisible by eight and positive batch/layer sizes')
        if aggregation not in ('mean', 'sum', 'max'):
            raise ValueError('Unknown CLMPT aggregation')
        self.model, self.pre_norm, self.aggregation = model, pre_norm, aggregation
        self.depth_shift, self.candidate_batch_size = depth_shift, candidate_batch_size
        if pre_norm:
            self.transformer = Transformer(dim, hidden_dim, layers)
        else:
            layer = nn.TransformerEncoderLayer(dim, 8, hidden_dim, .1, 'relu')
            self.transformer = nn.TransformerEncoder(layer, layers, nn.LayerNorm(dim), enable_nested_tensor=False)
        for name in ('existential_embedding', 'universal_embedding', 'free_embedding'):
            self.register_parameter(name, nn.Parameter(torch.rand(1, dim)))

    @staticmethod
    def message(entity, relation, inverse=False):
        real, imag = entity.chunk(2, -1)
        rr, ri = relation.chunk(2, -1)
        if inverse:
            ri = -ri
        return torch.cat((real * rr - imag * ri, real * ri + imag * rr), -1)

    def embed_batch(self, trees):
        graphs = [query_graph(tree) for tree in trees]
        atoms, anchors, variables = graphs[0]
        def signature(graph):
            return [(a, c, d) for a, _, c, d in graph[0]], list(graph[1]), graph[2]
        if any(signature(graph) != signature(graphs[0]) for graph in graphs[1:]):
            raise ValueError('CLMPT batches must share a query structure')
        terms = {name: self.model.entity_embeddings(torch.tensor([graph[1][name] for graph in graphs], device=self.device))
                 for name in anchors}
        terms.update({name: self.free_embedding if name == 'f' else self.existential_embedding for name in variables})
        relations = [self.model.relation_embeddings(torch.tensor([graph[0][index][1] for graph in graphs], device=self.device))
                     for index in range(len(atoms))]
        for _ in range(max(1, len(variables) + self.depth_shift)):
            messages = defaultdict(list)
            for index, (source, _, target, sign) in enumerate(atoms):
                embedding = relations[index]
                messages[target].append(sign * self.message(terms[source], embedding))
                if source in variables:
                    messages[source].append(sign * self.message(terms[target], embedding, inverse=True))
            updated = dict(terms)
            for name in variables:
                # Keep the release's tensor layout, including post-norm batch coupling.
                with BoundedAttention():
                    value = self.transformer(torch.stack([*messages[name], terms[name].expand(len(trees), -1)], 1))
                updated[name] = (value.max(1).values if self.aggregation == 'max' else
                                 value.sum(1) if self.aggregation == 'sum' else value.mean(1))
            terms = updated
        return terms['f']

    def embed(self, tree):
        return self.embed_batch([tree])

    def encode_trees(self, trees):
        expanded = [union_branches(tree) for tree in trees]
        if len({len(branches) for branches in expanded}) != 1:
            raise ValueError('CLMPT batches must share a query structure')
        return [self.embed_batch(trees) for trees in zip(*expanded)]

    def score_embeddings(self, branches):
        return torch.cat([torch.stack([torch.cosine_similarity(branch[:, None], entities, dim=-1)
                                       for branch in branches]).max(0).values
                          for entities in self.model.entity_embeddings.weight.split(self.candidate_batch_size)], dim=-1)

    def score_trees(self, trees):
        return self.score_embeddings(self.encode_trees(trees))

    @torch.no_grad()
    def predict_batch(self, queries):
        trees = [compile_query(query) for query in queries]
        if not trees:
            return self.model.entity_embeddings.weight.new_empty((0, self.context.num_entities))
        for tree in trees:
            validate_tree(tree, self.context.num_entities, self.context.num_relations)
        with evaluation_mode(self):
            result = self.score_trees(trees)
        if not torch.isfinite(result).all():
            raise ValueError('CLMPT returned invalid scores')
        return result

    def score_tree(self, tree):
        return self.score_trees([tree])[0]
