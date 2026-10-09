"""ConE inference. Architecture: MIRALab-USTC/QE-ConE (see NOTICE.md)."""

import math

import torch
from torch import nn

from ._common import QueryMethod, TensorCache, union_branches


class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim=1600, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.layer0 = nn.Linear(hidden_dim, 2 * dim)
        self.layer1 = nn.Linear(2 * dim, hidden_dim)
        for index in range(2, num_layers + 1):
            setattr(self, f'layer{index}', nn.Linear(hidden_dim, hidden_dim))

    def forward(self, axis, aperture, relation_axis, relation_aperture):
        value = torch.cat((axis + relation_axis, aperture + relation_aperture), -1)
        for index in range(1, self.num_layers + 1):
            value = getattr(self, f'layer{index}')(value).relu()
        axis, aperture = self.layer0(value).chunk(2, -1)
        return axis.tanh() * math.pi, (2 * aperture).tanh() * math.pi / 2 + math.pi / 2


class ConeIntersection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_axis1, self.layer_axis2 = nn.Linear(2 * dim, dim), nn.Linear(dim, dim)
        self.layer_arg1, self.layer_arg2 = nn.Linear(2 * dim, dim), nn.Linear(dim, dim)

    def forward(self, axes, apertures):
        endpoints = torch.cat((axes - apertures, axes + apertures), -1)
        attention = self.layer_axis2(self.layer_axis1(endpoints).relu()).softmax(0)
        x, y = (attention * axes.cos()).sum(0), (attention * axes.sin()).sum(0)
        x = x.masked_fill(x.abs() < 1e-3, 1e-3)
        axis = (y / x).atan()
        axis = torch.where((x < 0) & (y >= 0), axis + math.pi, axis)
        axis = torch.where((x < 0) & (y < 0), axis - math.pi, axis)
        gate = self.layer_arg2(self.layer_arg1(endpoints).relu().mean(0)).sigmoid()
        return axis, apertures.min(0).values * gate


class ConE(QueryMethod):
    def __init__(self, context, *, dim, gamma, center_reg, projection_dim=1600, projection_layers=2, candidate_batch_size=4096, cache_bytes=16 * 2**20):
        super().__init__(context)
        if dim < 1 or candidate_batch_size < 1 or center_reg < 0:
            raise ValueError('Invalid ConE dimensions or center regularization')
        self.angle_cache = TensorCache(cache_bytes)
        self._range_token = None
        self.cen, self.candidate_batch_size = center_reg, candidate_batch_size
        self.gamma = nn.Parameter(torch.tensor([float(gamma)]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.tensor([(gamma + 2) / dim]), requires_grad=False)
        self.modulus = nn.Parameter(self.embedding_range.detach().clone() / 2)
        self.entity_embedding = nn.Parameter(torch.empty(context.num_entities, dim))
        self.axis_embedding = nn.Parameter(torch.empty(context.num_relations, dim))
        self.arg_embedding = nn.Parameter(torch.empty(context.num_relations, dim))
        self.cone_proj = ConeProjection(dim, projection_dim, projection_layers)
        self.cone_intersection = ConeIntersection(dim)
        for parameter in (self.entity_embedding, self.axis_embedding, self.arg_embedding):
            nn.init.uniform_(parameter, -self.embedding_range.item(), self.embedding_range.item())

    def angle(self, value):
        parameter = self.embedding_range
        token = None if parameter.is_inference() else (id(parameter), parameter._version, parameter.device, parameter.dtype)
        if token is None or token != self._range_token:
            self._range_value = parameter.item()
            self._range_token = token
        return (value / self._range_value).tanh() * math.pi

    def embed(self, tree):
        axis, aperture = self.embed_batch([tree])
        return axis[0], aperture[0]

    def embed_batch(self, trees):
        op = trees[0][0]
        if any(tree[0] != op or len(tree) != len(trees[0]) for tree in trees):
            raise ValueError('ConE batches must share a query structure')
        if op == 'anchor':
            axis = self.angle(self.entity_embedding[[tree[1] for tree in trees]])
            return axis, torch.zeros_like(axis)
        if op == 'project':
            relations = [tree[1] for tree in trees]
            return self.cone_proj(*self.embed_batch([tree[2] for tree in trees]), self.angle(self.axis_embedding[relations]),
                                  self.angle(self.arg_embedding[relations]))
        if op == 'not':
            axis, aperture = self.embed_batch([tree[1] for tree in trees])
            return torch.where(axis >= 0, axis - math.pi, axis + math.pi), math.pi - aperture
        if op == 'and':
            axes, apertures = zip(*(self.embed_batch([tree[index] for tree in trees]) for index in range(1, len(trees[0]))))
            return self.cone_intersection(torch.stack(axes), torch.stack(apertures))
        raise ValueError('ConE unions must be executed as DNF branches')

    def encode_trees(self, trees):
        branches = [union_branches(tree) for tree in trees]
        if len({len(group) for group in branches}) != 1:
            raise ValueError('ConE batches must share a query structure')
        axes, apertures = self.embed_batch([branch for group in branches for branch in group])
        return [values.reshape(len(trees), len(branches[0]), -1) for values in (axes, apertures)]

    def score_embeddings(self, embeddings):
        return torch.stack([self._score_branches(list(zip(axes, apertures))) for axes, apertures in zip(*embeddings)])

    def _score_branches(self, branches):
        chunks = []
        geometry = [(axis, (aperture / 2).sin().abs(), axis - aperture, axis + aperture)
                    for axis, aperture in branches]
        self.angle_cache.refresh(self)
        for index, raw in enumerate(self.entity_embedding.split(self.candidate_batch_size)):
            key = (self.candidate_batch_size, index)
            entities = self.angle_cache.get(key) if not torch.is_grad_enabled() else None
            if entities is None:
                entities = self.angle(raw)
                if not torch.is_grad_enabled():
                    self.angle_cache.put(key, entities, optional=True)
            scores = []
            for axis, radius, lower, upper in geometry:
                center = ((entities - axis) / 2).sin().abs()
                outside = torch.minimum(((entities - lower) / 2).sin().abs(),
                                        ((entities - upper) / 2).sin().abs())
                outside = outside.masked_fill(center < radius, 0)
                distance = outside.norm(p=1, dim=-1) + self.cen * torch.minimum(center, radius).norm(p=1, dim=-1)
                scores.append(self.gamma - distance * self.modulus)
            chunks.append(torch.stack(scores).max(0).values)
        return torch.cat(chunks)

    def score_tree(self, tree):
        return self.score_embeddings(self.encode_trees([tree]))[0]
