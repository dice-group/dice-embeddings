"""Flock's non-backtracking walk and recording protocol in PyTorch.

Semantics follow jw9730/flock at f35103d25a78bdf4075de5c673a51de4979aa4d7.
The RNG is PyTorch's CPU generator, not the reference C++ per-walk MT19937.
Walk records can be supplied explicitly to compare neural computations exactly.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch


def _walk_nodes(prefix: torch.Tensor, length: int, keys: torch.Tensor, counts: torch.Tensor,
                offsets: torch.Tensor, num_nodes: int, generator: Optional[torch.Generator] = None):
    walks = torch.empty((len(prefix), length), dtype=torch.long)
    walks[:, :prefix.shape[1]] = prefix
    for step in range(prefix.shape[1], length):
        current = walks[:, step - 1]
        degree = counts[current]
        if not len(keys):
            walks[:, step] = current
            continue
        start = offsets[current]
        excluded = torch.zeros_like(current, dtype=torch.bool)
        previous_index = torch.zeros_like(current)
        if step >= 2:
            previous_key = current * num_nodes + walks[:, step - 2]
            previous_index = torch.searchsorted(keys, previous_key).clamp_max(len(keys) - 1)
            excluded = (keys[previous_index] == previous_key) & (degree > 1)
        choice = (torch.rand([len(current)], generator=generator) * (degree - excluded.long())).long()
        choice += (excluded & (choice >= previous_index - start)).long()
        following = keys[(start + choice).clamp_max(len(keys) - 1)] % num_nodes
        walks[:, step] = torch.where(degree > 0, following, current)
    return walks


@lru_cache(maxsize=1)
def compiled_walk_nodes():
    # Script only the CPU transition loop. Torch's CPU Generator and the exact
    # order/shapes of random draws remain unchanged.
    return torch.jit.script(_walk_nodes)


def anonymize(values, missing_id=None, missing_name=None):
    """Assign 1-based names in order of discovery, independently in each row."""
    rows, length = values.shape
    if rows == 0:
        return values.clone()
    width = int(values.max()) + 1
    keys = (values + torch.arange(rows)[:, None] * width).flatten()
    unique, inverse = keys.unique(return_inverse=True)
    positions = torch.arange(length).expand(rows, -1)
    first = torch.full_like(unique, length)
    first.scatter_reduce_(0, inverse, positions.flatten(), reduce="amin")
    first = first[inverse].view_as(values)
    discoveries = first == positions
    if missing_id is not None:
        discoveries &= values != missing_id
    names = discoveries.long().cumsum(1).gather(1, first)
    if missing_id is not None:
        names = names.masked_fill(values == missing_id, missing_name)
    return names


@dataclass
class WalkGraph:
    """CPU CSR context, kept outside a model's transferable parameters."""
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    num_nodes: int
    num_types: int
    compile_sampler: bool = True

    def __post_init__(self):
        self.edge_index = self.edge_index.detach().cpu()
        self.edge_type = self.edge_type.detach().cpu()
        # Stable grouping preserves the original per-relation member order and
        # therefore every RNG draw, while avoiding R full graph scans per query.
        order = self.edge_type.argsort(stable=True)
        relation_counts = torch.bincount(self.edge_type, minlength=self.num_types).tolist()
        self.relation_members = order.split(relation_counts)
        h, t = self.edge_index
        keys = torch.cat((h * self.num_nodes + t, t * self.num_nodes + h))
        order = keys.argsort()
        self.type_keys, self.type_counts = keys[order].unique_consecutive(return_counts=True)
        self.type_offsets = self.type_counts.cumsum(0) - self.type_counts
        self.types = self.edge_type.repeat(2)[order]
        # The released C++ parser stores directions in queue<bool>, so its
        # intended loop marker 2 is actually emitted as 1. Preserve that value.
        self.directions = torch.cat(((h == t).long(), torch.ones_like(t)))[order]
        self.adjacency = {}
        for remove_loops in (False, True):
            adjacency_keys = self.type_keys
            if remove_loops:
                adjacency_keys = adjacency_keys[adjacency_keys // self.num_nodes != adjacency_keys % self.num_nodes]
            counts = torch.bincount(adjacency_keys // self.num_nodes, minlength=self.num_nodes)
            offsets = counts.cumsum(0) - counts
            self.adjacency[remove_loops] = (adjacency_keys, counts, offsets)

    def walk(self, prefix, length, remove_loops, generator=None, prefix_types=None):
        """Sample uniform neighbors, then a uniform typed edge in either direction."""
        prefix = prefix.cpu().reshape(len(prefix), -1)
        keys, counts, offsets = self.adjacency[remove_loops]
        sample = compiled_walk_nodes() if self.compile_sampler else _walk_nodes
        walks = sample(prefix, length, keys, counts, offsets, self.num_nodes, generator)
        types, directions = self.parse_types(walks, generator)
        if prefix_types is not None:
            types[:, 1] = prefix_types
            directions[:, 1] = 0
        names = anonymize(walks)
        type_names = anonymize(types, self.num_types, length + 1)
        zeros = torch.zeros_like(walks)
        return walks, names, zeros, zeros.clone(), types, type_names, directions

    def parse_types(self, walks, generator=None):
        types = torch.full_like(walks, self.num_types)
        directions = torch.full_like(walks, 3)
        if not len(self.type_keys):
            return types, directions
        keys = walks[:, :-1] * self.num_nodes + walks[:, 1:]
        groups = torch.searchsorted(self.type_keys, keys).clamp_max(len(self.type_keys) - 1)
        valid = self.type_keys[groups] == keys
        choices = (torch.rand(keys.shape, generator=generator) * self.type_counts[groups]).long()
        indices = self.type_offsets[groups] + choices
        types[:, 1:] = torch.where(valid, self.types[indices], self.num_types)
        directions[:, 1:] = torch.where(valid, self.directions[indices], 3)
        return types, directions

    def sample(self, heads, tails, walk_num, length, refinements, generator=None):
        """Return seven official-format records of shape [T,B,3N or 4N,L]."""
        heads = heads.cpu()
        batch = len(heads)
        groups = 2 if tails is None else 3
        starts = [heads.repeat(walk_num * refinements)]
        if tails is not None:
            starts.append(tails.cpu().repeat(walk_num * refinements))
        starts.append(torch.randint(self.num_nodes, (walk_num * refinements * batch,), generator=generator))
        node_records = self.walk(torch.cat(starts), length, True, generator)

        # Balanced per-relation edge prefixes, sampled without replacement in
        # repeated random permutations, then shuffled/subsampled per query.
        per_type = (walk_num // self.num_types + 1) * refinements * batch
        indices = []
        for members in self.relation_members:
            if len(members):
                repeats = (per_type + len(members) - 1) // len(members)
                selected = torch.cat([torch.randperm(len(members), generator=generator) for _ in range(repeats)])[:per_type]
                indices.append(members[selected])
        if indices:
            selected = torch.cat(indices)
            edge_records = self.walk(self.edge_index[:, selected].T, length, False, generator, self.edge_type[selected])
            actual = len(selected) // (refinements * batch)
        else:
            # A masked training graph can be empty. There are no typed prefixes;
            # use isolated-node records with the no-type marker in this case.
            edge_records = tuple(record[:walk_num * refinements * batch] for record in node_records)
            actual = walk_num
        permutations = torch.rand(refinements * batch, actual, generator=generator).argsort(dim=1)
        permutations = permutations.view(refinements, batch, actual).permute(2, 0, 1)
        permutations = permutations[..., None].expand(-1, -1, -1, length)
        output = []
        for nodes, edges in zip(node_records, edge_records):
            nodes = nodes.view(groups * walk_num, refinements, batch, length)
            edges = edges.view(actual, refinements, batch, length).gather(0, permutations)
            if actual < walk_num:
                edges = edges.repeat(walk_num // actual + 1, 1, 1, 1)
            output.append(torch.cat((nodes, edges[:walk_num])).permute(1, 2, 0, 3).contiguous())
        return tuple(output)
