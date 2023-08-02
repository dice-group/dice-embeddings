import torch
from typing import Dict,Tuple,List
import numpy as np
def evaluate_lp(model, triple_idx, num_entities: int, er_vocab:Dict[Tuple,List], re_vocab:Dict[Tuple,List], info='Eval Starts'):
    # @TODO Document this code
    model.eval()
    print(f'Num of triples {len(triple_idx)}')
    print('** Evaluation without batching')
    hits = dict()
    reciprocal_ranks = []
    # Iterate over test triples
    all_entities = torch.arange(0, num_entities).long()
    all_entities = all_entities.reshape(len(all_entities), )
    # Iterating one by one is not good when you are using batch norm
    for h, r, t in triple_idx:
        # (1) Get a triple (head entity, relation, tail entity
        # (2) Predict missing heads and tails
        x = torch.stack((torch.tensor(h).repeat(num_entities, ),
                         torch.tensor(r).repeat(num_entities, ), all_entities), dim=1)

        predictions_tails = model.forward_triples(x)
        x = torch.stack((all_entities,
                         torch.tensor(r).repeat(num_entities, ), torch.tensor(t).repeat(num_entities)), dim=1)
        predictions_heads = model.forward_triples(x)
        del x

        if er_vocab:
            # 3. Computed filtered ranks for missing tail entities.
            # 3.1. Compute filtered tail entity rankings
            filt_tails = er_vocab[(h, r)]
            # 3.2 Get the predicted target's score
            target_value = predictions_tails[t].item()
            # 3.3 Filter scores of all triples containing filtered tail entities
            predictions_tails[filt_tails] = -np.Inf
            predictions_tails[t] = target_value

        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == t)[0][0]

        if re_vocab:
            filt_heads = re_vocab[(r, t)]
            # 4.2 Get the predicted target's score
            target_value = predictions_heads[h].item()
            # 4.3 Filter scores of all triples containing filtered head entities.
            predictions_heads[filt_heads] = -np.Inf
            predictions_heads[h] = target_value

        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_head_entity_rank = np.where(sort_idxs == h)[0][0]

        # 4. Add 1 to ranks as numpy array first item has the index of 0.
        filt_head_entity_rank += 1
        filt_tail_entity_rank += 1

        rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
        # 5. Store reciprocal ranks.
        reciprocal_ranks.append(rr)

        # 4. Compute Hit@N
        for hits_level in range(1, 11):
            res = 1 if filt_head_entity_rank <= hits_level else 0
            res += 1 if filt_tail_entity_rank <= hits_level else 0
            if res > 0:
                hits.setdefault(hits_level, []).append(res)

    mean_reciprocal_rank = sum(reciprocal_ranks) / (float(len(triple_idx) * 2))

    if 1 in hits:
        hit_1 = sum(hits[1]) / (float(len(triple_idx) * 2))
    else:
        hit_1 = 0

    if 3 in hits:
        hit_3 = sum(hits[3]) / (float(len(triple_idx) * 2))
    else:
        hit_3 = 0

    if 10 in hits:
        hit_10 = sum(hits[10]) / (float(len(triple_idx) * 2))
    else:
        hit_10 = 0

    results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
               'MRR': mean_reciprocal_rank}
    print(results)
    return results


def efficient_zero_grad(model):
    # Use this instead of
    # self.optimizer.zero_grad()
    #
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
    for param in model.parameters():
        param.grad = None

