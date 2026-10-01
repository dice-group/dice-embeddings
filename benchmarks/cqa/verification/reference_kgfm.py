"""Independent adapter and fixed-k algebra; atomic backbone logits are shared."""

import math
from collections import Counter, defaultdict

import torch


class ReferenceKGFM:
    def __init__(self, triples, n, nr, adapter, *, beam_size, device, restore_observed=True):
        if (adapter['feature_mode'] not in ('context_scores', 'context_scores_v1', 'global') or adapter['hidden_dim']
                or adapter['normalization'] != 'none' or adapter['observed_mix'] not in (0, 1)):
            raise ValueError('This reference covers the frozen linear observed-fact recipe')
        self.n, self.nr, self.adapter = n, nr, adapter
        self.k, self.device = beam_size, device
        self.restore_observed = restore_observed
        self.edges = defaultdict(set)
        triples = set(map(tuple, triples))
        self.degree = Counter(h for h, _, _ in triples)
        self.frequency = Counter(r for _, r, _ in triples)
        for h, r, t in triples:
            self.edges[h, r].add(t)
        self.rows = {}
        self.raw_rows = 0

    def capture(self, conditions, raw):
        """Calibrate fresh logits without calling the production feature/adapter code."""
        raw = raw.double()
        observed = torch.zeros_like(raw, dtype=torch.bool)
        features = []
        for i, (h, r) in enumerate(conditions):
            tails = self.edges[h, r]
            observed[i, sorted(tails)] = True
            features.append([1., math.log1p(len(tails)) / math.log1p(self.n),
                             math.log1p(self.degree[h]) / math.log1p(self.n * self.nr),
                             math.log1p(self.frequency[r]) / math.log1p(self.n**2)])
        mean = raw.mean(1)
        centered = raw - raw.max(1, keepdim=True).values
        mass = centered.exp()
        total = mass.sum(1)
        entropy = (total.log() - (mass * centered).sum(1) / total) / math.log(self.n)
        top = raw.topk(2, dim=1).values
        gap = ((top[:, 0] - top[:, 1]) / 4).tanh()
        count = observed.sum(1)
        contrast = torch.where(count > 0, (((raw * observed).sum(1) / count.clamp_min(1) - mean) / 4).tanh(), 0.)
        features = torch.cat((raw.new_tensor(features),
                              torch.stack(((mean / 4).tanh(), entropy, gap, contrast), dim=1)), 1)
        if self.adapter['feature_mode'] == 'global':
            features = raw.new_ones((len(raw), 1))
        u, v = (features @ raw.new_tensor(self.adapter['weights']).T).unbind(1)
        bound = math.log(self.adapter['scale_bound'])
        alpha = (bound * (u * (math.log(2) / bound)).tanh()).exp()
        logits = alpha[:, None] * raw + self.adapter['bias_bound'] * v.tanh()[:, None]
        calibrated = torch.nn.functional.logsigmoid(logits)
        if self.adapter['observed_mix'] == 1:
            calibrated = calibrated.masked_fill(observed, 0.)
        self.rows.update((tuple(condition), row.cpu()) for condition, row in zip(conditions, calibrated))
        self.raw_rows += len(conditions)

    @staticmethod
    def complement(value):
        result = torch.empty_like(value)
        small = value < -math.log(2)
        result[small] = torch.log1p(-value[small].exp())
        result[~small] = torch.log(-torch.expm1(value[~small]))
        return result

    def predict(self, query, operator='product'):
        """Interpret nested tuples directly, with no production parser or executor."""
        def visit(value):
            if isinstance(value, int):
                scores = torch.full((self.n,), -torch.inf, dtype=torch.float64, device=self.device)
                scores[value] = 0.
                return scores, {value}, True
            if (len(value) == 2 and value[1] and value[1] != (-1,)
                    and all(isinstance(r, int) for r in value[1])):
                scores, answers, positive = visit(value[0])
                for position, relation in enumerate(value[1]):
                    if relation == -2:
                        scores, positive = self.complement(scores), False
                        answers = set(range(self.n)) - answers
                        continue
                    if isinstance(value[0], int) and position == 0:
                        scores = self.rows[value[0], relation].to(self.device)
                    else:
                        heads = scores.argsort(descending=True, stable=True)[:self.k].tolist()
                        reduced = torch.full_like(scores, -torch.inf)
                        for head in heads:
                            if torch.isneginf(scores[head]).item():
                                continue
                            row = self.rows[head, relation].to(self.device)
                            joined = scores[head] + row if operator == 'product' else torch.minimum(scores[head], row)
                            reduced = torch.maximum(reduced, joined)
                        scores = reduced
                    answers = set().union(*(self.edges[h, relation] for h in answers))
                return scores, answers, positive
            union = value[-1] == (-1,)
            branches = [visit(branch) for branch in (value[:-1] if union else value)]
            rows = torch.stack([row for row, _, _ in branches])
            if operator == 'product':
                scores = self.complement(self.complement(rows).sum(0)) if union else rows.sum(0)
            else:
                scores = rows.max(0).values if union else rows.min(0).values
            sets = [answers for _, answers, _ in branches]
            answers = set.union(*sets) if union else set.intersection(*sets)
            return scores, answers, all(positive for _, _, positive in branches)

        scores, answers, positive = visit(query)
        if self.restore_observed and self.adapter['observed_mix'] == 1 and positive:
            scores = scores.clone()
            scores[sorted(answers)] = 0.
        return scores
