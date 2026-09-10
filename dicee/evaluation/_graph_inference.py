"""Reuse deterministic graph queries while retaining only compact rank data."""
import torch


class GraphRankPlan:
    """Score each directional query once for all its evaluation targets.

    Scores are discarded immediately after ranking all targets of a query. The
    retained data is O(test triples), not O(queries * entities). A runner may
    share this plan across progress batches; tie RNG is consumed only when each
    batch's ranks are requested, in the original tail/head order.
    """

    def __init__(self, triples):
        self.targets: dict[tuple[bool, int, int], dict[int, None]] = {}
        for h, r, t in triples:
            self.targets.setdefault((False, int(h), int(r)), {})[int(t)] = None
            self.targets.setdefault((True, int(r), int(t)), {})[int(h)] = None
        self.cache = {}
        self._token = None
        self.scored_queries = 0

    def rank(self, model, triples, ranker, er_vocab, re_vocab, batch_size):
        token = (id(model), model.inference_token(), ranker.tie_policy, id(er_vocab), id(re_vocab))
        if token != self._token or token[1] is None:
            self.cache.clear()
            self._token = token
        wanted = []
        for h, r, t in triples:
            wanted.extend([((False, int(h), int(r)), int(t)), ((True, int(r), int(t)), int(h))])
        for key, target in wanted:
            if target not in self.targets.setdefault(key, {}):
                self.targets[key][target] = None
                self.cache.pop(key, None)
        missing = list(dict.fromkeys(key for key, _ in wanted if key not in self.cache))
        for head_prediction in (False, True):
            keys = [key for key in missing if key[0] == head_prediction]
            for start in range(0, len(keys), batch_size):
                batch = keys[start:start + batch_size]
                queries = torch.tensor([key[1:] for key in batch], dtype=torch.long)
                scores = (model.forward_k_vs_all_heads(queries) if head_prediction else model.forward_k_vs_all(queries))
                self.scored_queries += len(batch)
                vocab = re_vocab if head_prediction else er_vocab
                # Sort ties retain the legacy per-vector CPU sort order. Other
                # policies need neither a sort nor a full score copy to CPU.
                if ranker.tie_policy == 'sort':
                    scores = scores.cpu()
                pending = [(i, target, key) for i, key in enumerate(batch) for target in self.targets[key]]
                for key in batch:
                    self.cache[key] = {}
                for offset in range(0, len(pending), batch_size):
                    chunk = pending[offset:offset + batch_size]
                    targets = [target for _, target, _ in chunk]
                    filters = [vocab[key[1:]] for _, _, key in chunk]
                    if ranker.tie_policy == 'sort':
                        values = [(ranker.rank(scores[i], target, filt), 0)
                                  for (i, target, _), filt in zip(chunk, filters)]
                    else:
                        values = ranker.bounds_batch(scores, targets, filters, row_indices=[i for i, _, _ in chunk])
                    for (_, target, key), value in zip(chunk, values):
                        self.cache[key][target] = value
        bounds = [self.cache[key][target] for key, target in wanted]
        return ranker.ranks_from_bounds(bounds)
