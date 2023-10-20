import torch
from typing import Dict, Tuple, List
import numpy as np


def evaluate_lp(model, triple_idx, num_entities, er_vocab: Dict[Tuple, List], re_vocab: Dict[Tuple, List],
                info='Eval Starts'):
    """
    Evaluate model in a standard link prediction task

    for each triple
    the rank is computed by taking the mean of the filtered missing head entity rank and
    the filtered missing tail entity rank
    :param model:
    :param triple_idx:
    :param info:
    :return:
    """
    model.eval()
    print(info)
    print(f'Num of triples {len(triple_idx)}')
    print('** Evaluation without batching')
    hits = dict()
    reciprocal_ranks = []
    # Iterate over test triples
    all_entities = torch.arange(0, num_entities).long()
    all_entities = all_entities.reshape(len(all_entities), )
    # Iterating one by one is not good when you are using batch norm
    for i in range(0, len(triple_idx)):
        # (1) Get a triple (head entity, relation, tail entity
        data_point = triple_idx[i]
        h, r, t = data_point[0], data_point[1], data_point[2]

        # (2) Predict missing heads and tails
        x = torch.stack((torch.tensor(h).repeat(num_entities, ),
                         torch.tensor(r).repeat(num_entities, ),
                         all_entities), dim=1)

        predictions_tails = model.forward_triples(x)
        x = torch.stack((all_entities,
                         torch.tensor(r).repeat(num_entities, ),
                         torch.tensor(t).repeat(num_entities)
                         ), dim=1)

        predictions_heads = model.forward_triples(x)
        del x

        # 3. Computed filtered ranks for missing tail entities.
        # 3.1. Compute filtered tail entity rankings
        filt_tails = er_vocab[(h, r)]
        # 3.2 Get the predicted target's score
        target_value = predictions_tails[t].item()
        # 3.3 Filter scores of all triples containing filtered tail entities
        predictions_tails[filt_tails] = -np.Inf
        # 3.4 Reset the target's score
        predictions_tails[t] = target_value
        # 3.5. Sort the score
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == t)[0][0]

        # 4. Computed filtered ranks for missing head entities.
        # 4.1. Retrieve head entities to be filtered
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
        # print(f'{i}.th triple: mean reciprical rank:{rr}')

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


@torch.no_grad
def evaluate_bpe_lp(model, triple_idx: List[Tuple[Tuple[int], Tuple[int], Tuple[int]]],
                    ordered_bpe_entities_relations: List[Tuple[int]],
                    index_of_relations_on_ordered_bpe_entities_relations,
                    er_vocab: Dict[Tuple, List],
                    re_vocab: Dict[Tuple, List],
                    info='Eval Starts'):
    """

    Parameters
    ----------
    model
    triple_idx
    ordered_bpe_entities_relations
    index_of_relations_on_ordered_bpe_entities_relations
    er_vocab
    re_vocab
    info

    Returns
    -------

    """

    model.eval()
    print(info)
    print(f'Num of BPE triples {len(triple_idx)}')
    print('** Evaluation without batching')
    hits = dict()
    reciprocal_ranks = []
    # Iterate over test triples
    # all_entities = torch.arange(0, num_entities).long()
    # all_entities = all_entities.reshape(len(all_entities), )
    # Iterating one by one is not good when you are using batch norm
    num_tokens = len(ordered_bpe_entities_relations)
    ordered_all_tokens = torch.LongTensor(ordered_bpe_entities_relations)
    index_of_relations = index_of_relations_on_ordered_bpe_entities_relations
    for i in range(0, len(triple_idx)):
        # (1) Get a triple (head entity, relation, tail entity
        bpe_encoded_h, bpe_encoded_r, bpe_encoded_t = triple_idx[i]
        torch_bpe_encoded_h = torch.LongTensor(bpe_encoded_h).unsqueeze(0)
        torch_bpe_encoded_r = torch.LongTensor(bpe_encoded_r).unsqueeze(0)
        torch_bpe_encoded_t = torch.LongTensor(bpe_encoded_t).unsqueeze(0)

        x = torch.stack((ordered_all_tokens,
                         torch.repeat_interleave(input=torch_bpe_encoded_r, repeats=num_tokens, dim=0),
                         torch.repeat_interleave(input=torch_bpe_encoded_t, repeats=num_tokens, dim=0)), dim=1)
        predictions_heads = model.forward(x)
        predictions_heads[index_of_relations] = -np.Inf

        del x

        x = torch.stack((torch.repeat_interleave(input=torch_bpe_encoded_h, repeats=num_tokens, dim=0),
                         torch.repeat_interleave(input=torch_bpe_encoded_r, repeats=num_tokens, dim=0),
                         ordered_all_tokens), dim=1)
        predictions_tails = model.forward(x)
        predictions_tails[index_of_relations] = -np.Inf

        del x

        # 3. Computed filtered ranks for missing tail entities.
        # 3.1. Compute filtered tail entity rankings
        filt_bpe_encoded_list_of_tails = er_vocab[(bpe_encoded_h, bpe_encoded_r)]
        filt_tails = [ordered_bpe_entities_relations.index(i) for i in filt_bpe_encoded_list_of_tails]

        index_of_t_in_all_tokens: int
        index_of_t_in_all_tokens = ordered_bpe_entities_relations.index(bpe_encoded_t)
        # 3.2 Get the predicted target's score
        target_value = predictions_tails[index_of_t_in_all_tokens].item()
        # 3.3 Filter scores of all triples containing filtered tail entities
        predictions_tails[filt_tails] = -np.Inf
        # 3.4 Reset the target's score
        predictions_tails[index_of_t_in_all_tokens] = target_value
        # 3.5. Sort the score
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == index_of_t_in_all_tokens)[0][0]

        # 4. Computed filtered ranks for missing head entities.
        # 4.1. Retrieve head entities to be filtered
        filt_bpe_encoded_list_of_heads = re_vocab[(bpe_encoded_r, bpe_encoded_t)]

        index_of_h_in_all_tokens: int
        index_of_h_in_all_tokens = ordered_bpe_entities_relations.index(bpe_encoded_h)

        # 4.2 Get the predicted target's score
        target_value = predictions_heads[index_of_h_in_all_tokens].item()

        filt_heads = [ordered_bpe_entities_relations.index(i) for i in filt_bpe_encoded_list_of_heads]

        # 4.3 Filter scores of all triples containing filtered head entities.
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[index_of_h_in_all_tokens] = target_value
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_head_entity_rank = np.where(sort_idxs == index_of_h_in_all_tokens)[0][0]

        # 4. Add 1 to ranks as numpy array first item has the index of 0.
        filt_head_entity_rank += 1
        filt_tail_entity_rank += 1

        rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
        # 5. Store reciprocal ranks.
        reciprocal_ranks.append(rr)
        # print(f'{i}.th triple: mean reciprical rank:{rr}')

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
