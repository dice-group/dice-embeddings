"""Lazy reference adjacency with upstream QTO's canonical 100-head calibration."""

from collections import OrderedDict, defaultdict

import torch


class ReferenceRelations:
    def __init__(self, source, model, triples, n, device, threshold, fraction=100, cache_bytes=256 * 2**20):
        self.source, self.model = source, model
        self.n, self.device, self.threshold, self.fraction = n, device, threshold, fraction
        self.cache, self.cache_bytes, self.bytes = OrderedDict(), cache_bytes, 0
        self.observed = defaultdict(lambda: defaultdict(list))
        for head, relation, tail in triples:
            self.observed[relation][head].append(tail)

    def block(self, relation, start):
        key = relation, start
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        end = min(start + 100, self.n)
        heads = torch.arange(start, end, device=self.device)
        rel = torch.tensor([relation], device=self.device)
        scores = self.source.kge_forward(self.model, heads, rel, self.device, self.n)
        degree = scores.new_tensor([max(1, len(self.observed[relation][head])) for head in range(start, end)])
        value = torch.softmax(scores, dim=1) * degree[:, None]
        value = ((value >= self.threshold).float() * value).cpu()
        value = (value >= 1).float() * .9999 + (value < 1).float() * value
        for head in range(start, end):
            value[head - start, self.observed[relation][head]] = 1
        size = value.numel() * value.element_size()
        if size <= self.cache_bytes:
            while self.bytes + size > self.cache_bytes:
                _, previous = self.cache.popitem(last=False)
                self.bytes -= previous.numel() * previous.element_size()
            self.cache[key] = value
            self.bytes += size
        return value

    def __getitem__(self, relation):
        return _Relation(self, int(relation))


class _Relation:
    def __init__(self, owner, relation):
        self.owner, self.relation = owner, relation

    def __getitem__(self, part):
        return _Rows(self.owner, self.relation, part * (self.owner.n // self.owner.fraction))


class _Rows:
    def __init__(self, owner, relation, offset):
        self.owner, self.relation, self.offset = owner, relation, offset

    def to_dense(self):
        # Upstream immediately selects active rows; materialize only that selection.
        return self

    def __getitem__(self, key):
        indices, columns = key
        if columns != slice(None):
            raise ValueError('Expected the complete candidate domain')
        heads = indices.cpu().tolist()
        rows = []
        for index in heads:
            head = index + self.offset
            rows.append(self.owner.block(self.relation, head // 100 * 100)[head % 100])
        return torch.stack(rows).to(self.owner.device)
