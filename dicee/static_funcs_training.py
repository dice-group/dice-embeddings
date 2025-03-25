import torch
from typing import Dict, Tuple, List, Iterable
import numpy as np
from tqdm import tqdm

def make_iterable_verbose(iterable_object, verbose, desc="Default", position=None, leave=True) -> Iterable:
    if verbose:
        return tqdm(iterable_object, desc=desc, position=position, leave=leave)
    else:
        return iterable_object

@torch.no_grad()
def evaluate_lp(model=None, triple_idx=None, num_entities=None, er_vocab: Dict[Tuple, List]=None,
                re_vocab: Dict[Tuple, List]=None,
                info='Eval Starts', batch_size=128, chunk_size=1000):
    assert model is not None, "Model must be provided"
    assert triple_idx is not None, "triple_idx must be provided"
    assert num_entities is not None, "num_entities must be provided"
    assert er_vocab is not None, "er_vocab must be provided"
    assert re_vocab is not None, "re_vocab must be provided"

    model.eval()
    print(info)
    print(f'Num of triples {len(triple_idx)}')
    hits = dict()
    reciprocal_ranks = []
    # Iterate over test triples
    all_entities = torch.arange(0, num_entities).long()
    all_entities = all_entities.reshape(len(all_entities), )
    
    # Evaluation without Batching
    # for i in tqdm(range(0, len(triple_idx))):
    #     # (1) Get a triple (head entity, relation, tail entity
    #     data_point = triple_idx[i]
    #     h, r, t = data_point[0], data_point[1], data_point[2]

    #     # (2) Predict missing heads and tails
    #     x = torch.stack((torch.tensor(h).repeat(num_entities, ),
    #                      torch.tensor(r).repeat(num_entities, ),
    #                      all_entities), dim=1)

    #     predictions_tails = model(x)
    #     x = torch.stack((all_entities,
    #                      torch.tensor(r).repeat(num_entities, ),
    #                      torch.tensor(t).repeat(num_entities)
    #                      ), dim=1)

    #     predictions_heads = model(x)
    #     del x

    #     # 3. Computed filtered ranks for missing tail entities.
    #     # 3.1. Compute filtered tail entity rankings
    #     filt_tails = er_vocab[(h, r)]
    #     # 3.2 Get the predicted target's score
    #     target_value = predictions_tails[t].item()
    #     # 3.3 Filter scores of all triples containing filtered tail entities
    #     predictions_tails[filt_tails] = -np.Inf
    #     # 3.4 Reset the target's score
    #     predictions_tails[t] = target_value
    #     # 3.5. Sort the score
    #     _, sort_idxs = torch.sort(predictions_tails, descending=True)
    #     sort_idxs = sort_idxs.detach()
    #     filt_tail_entity_rank = np.where(sort_idxs == t)[0][0]

    #     # 4. Computed filtered ranks for missing head entities.
    #     # 4.1. Retrieve head entities to be filtered
    #     filt_heads = re_vocab[(r, t)]
    #     # 4.2 Get the predicted target's score
    #     target_value = predictions_heads[h].item()
    #     # 4.3 Filter scores of all triples containing filtered head entities.
    #     predictions_heads[filt_heads] = -np.Inf
    #     predictions_heads[h] = target_value
    #     _, sort_idxs = torch.sort(predictions_heads, descending=True)
    #     sort_idxs = sort_idxs.detach()
    #     filt_head_entity_rank = np.where(sort_idxs == h)[0][0]

    #     # 4. Add 1 to ranks as numpy array first item has the index of 0.
    #     filt_head_entity_rank += 1
    #     filt_tail_entity_rank += 1

    #     rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
    #     # 5. Store reciprocal ranks.
    #     reciprocal_ranks.append(rr)
    #     # print(f'{i}.th triple: mean reciprical rank:{rr}')

    #     # 4. Compute Hit@N
    #     for hits_level in range(1, 11):
    #         res = 1 if filt_head_entity_rank <= hits_level else 0
    #         res += 1 if filt_tail_entity_rank <= hits_level else 0
    #         if res > 0:
    #             hits.setdefault(hits_level, []).append(res)
    
    # Evaluation with Batching
    for batch_start in tqdm(range(0, len(triple_idx), batch_size), desc="Evaluating Batches"):
        batch_end = min(batch_start + batch_size, len(triple_idx))
        batch_triples = triple_idx[batch_start:batch_end]
        batch_size_current = len(batch_triples)
        
        # (1) Extract heads, relations, and tails for the batch
        h_batch = torch.tensor([data_point[0] for data_point in batch_triples])
        r_batch = torch.tensor([data_point[1] for data_point in batch_triples])
        t_batch = torch.tensor([data_point[2] for data_point in batch_triples])

        # Initialize score tensors
        predictions_tails = torch.zeros(batch_size_current, num_entities)
        predictions_heads = torch.zeros(batch_size_current, num_entities)

        # Process entities in chunks to manage memory usage
        for chunk_start in range(0, num_entities, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_entities)
            entities_chunk = all_entities[chunk_start:chunk_end]
            chunk_size_current = entities_chunk.size(0)
            
            # (2) Predict missing heads and tails
            # Prepare input tensors for tail prediction
            x_tails = torch.stack((
                h_batch.repeat_interleave(chunk_size_current),
                r_batch.repeat_interleave(chunk_size_current),
                entities_chunk.repeat(batch_size_current)
            ), dim=1)

            # Predict scores for missing tails
            preds_tails = model(x_tails)
            preds_tails = preds_tails.view(batch_size_current, chunk_size_current)
            predictions_tails[:, chunk_start:chunk_end] = preds_tails
            del x_tails

            # Prepare input tensors for head prediction
            x_heads = torch.stack((
                entities_chunk.repeat(batch_size_current),
                r_batch.repeat_interleave(chunk_size_current),
                t_batch.repeat_interleave(chunk_size_current)
            ), dim=1)

            # Predict scores for missing heads
            preds_heads = model(x_heads)
            preds_heads = preds_heads.view(batch_size_current, chunk_size_current)
            predictions_heads[:, chunk_start:chunk_end] = preds_heads
            del x_heads
    
        # Iterating one by one is not good when you are using batch norm
        for i in range(0, batch_size_current):
        # (1) Get a triple (head entity, relation, tail entity)
            h = h_batch[i].item()
            r = r_batch[i].item()
            t = t_batch[i].item()

            # 3. Computed filtered ranks for missing tail entities.
            # 3.1. Compute filtered tail entity rankings
            filt_tails = er_vocab[(h, r)]
            filt_tails_set = set(filt_tails) - {t}
            filt_tails_indices = list(filt_tails_set)
            # 3.2 Get the predicted target's score
            target_value = predictions_tails[i, t].item()
            # 3.3 Filter scores of all triples containing filtered tail entities
            predictions_tails[i, filt_tails_indices] = -np.Inf
            # 3.4 Reset the target's score
            predictions_tails[i, t] = target_value
            # 3.5. Sort the score
            _, sort_idxs = torch.sort(predictions_tails[i], descending=True)
            sort_idxs = sort_idxs.detach()
            filt_tail_entity_rank = np.where(sort_idxs == t)[0][0]

            # 4. Computed filtered ranks for missing head entities.
            # 4.1. Retrieve head entities to be filtered
            filt_heads = re_vocab[(r, t)]
            filt_heads_set = set(filt_heads) - {h}
            filt_heads_indices = list(filt_heads_set)
            # 4.2 Get the predicted target's score
            target_value = predictions_heads[i, h].item()
            # 4.3 Filter scores of all triples containing filtered head entities.
            predictions_heads[i, filt_heads_indices] = -np.Inf
            predictions_heads[i, h] = target_value
            _, sort_idxs = torch.sort(predictions_heads[i], descending=True)
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

@torch.no_grad()
def evaluate_lp_k_vs_all(model, triple_idx, er_vocab=None,info=None, batch_size:int =1) -> Dict:
    """
    Filtered link prediction evaluation.
    :param model:
    :param er_vocab:
    :param batch_size:
    :param triple_idx: test triples
    :param info:
    :param form_of_labelling:
    :return:
    """
    assert er_vocab is not None
    # (1) set model to eval model
    model.eval()
    num_triples = len(triple_idx)
    ranks = []
    # Hit range
    hits_range = [i for i in range(1, 11)]
    hits = {i: [] for i in hits_range}
    if info :
        print(info + ':', end=' ')
    # Iterate over integer indexed triples in mini batch fashion
    for i in tqdm(range(0, num_triples, batch_size)):
            # (1) Get a batch of data.
            data_batch = triple_idx[i:i + batch_size]
            # (2) Extract entities and relations.
            e1_idx_r_idx, e2_idx = torch.LongTensor(data_batch[:, [0, 1]]), torch.tensor(data_batch[:, 2])
            # (3) Predict missing entities, i.e., assign probs to all entities.
            with torch.no_grad():
                predictions = model(e1_idx_r_idx)
            # (4) Filter entities except the target entity
            for j in range(data_batch.shape[0]):
                # (4.1) Get the ids of the head entity, the relation and the target tail entity in the j.th triple.
                id_e, id_r, id_e_target = data_batch[j]
                # (4.2) Get all ids of all entities occurring with the head entity and relation extracted in 4.1.
                filt = er_vocab[(id_e, id_r)]
                # (4.3) Store the assigned score of the target tail entity extracted in 4.1.
                target_value = predictions[j, id_e_target].item()
                # (4.4.1) Filter all assigned scores for entities.
                predictions[j, filt] = -np.Inf
                # (4.5) Insert 4.3. after filtering.
                predictions[j, id_e_target] = target_value
            # (5) Sort predictions.
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            # (6) Compute the filtered ranks.
            for j in range(data_batch.shape[0]):
                # index between 0 and \inf
                rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
                ranks.append(rank)
                for hits_level in hits_range:
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
    # (7) Sanity checking: a rank for a triple
    assert len(triple_idx) == len(ranks) == num_triples
    hit_1 = sum(hits[1]) / num_triples
    hit_3 = sum(hits[3]) / num_triples
    hit_10 = sum(hits[10]) / num_triples
    mean_reciprocal_rank = np.mean(1. / np.array(ranks))
    results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
    return results


@torch.no_grad()
def evaluate_bpe_lp(model, triple_idx: List[Tuple], all_bpe_shaped_entities,
                    er_vocab: Dict[Tuple, List], re_vocab: Dict[Tuple, List],
                    info='Eval Starts'):

    assert isinstance(triple_idx, list)
    assert isinstance(triple_idx[0], tuple)
    assert len(triple_idx[0]) == 3
    model.eval()
    print(info)
    print(f'Num of triples {len(triple_idx)}')
    hits = dict()
    reciprocal_ranks = []
    # Iterate over test triples
    num_entities = len(all_bpe_shaped_entities)
    bpe_entity_to_idx = dict()
    all_bpe_entities = []
    for idx, (str_entity, bpe_entity, shaped_bpe_entity) in tqdm(enumerate(all_bpe_shaped_entities)):
        bpe_entity_to_idx[shaped_bpe_entity] = idx
        all_bpe_entities.append(shaped_bpe_entity)
    all_bpe_entities = torch.LongTensor(all_bpe_entities)
    for (bpe_h, bpe_r, bpe_t) in tqdm(triple_idx):
        # (1) Indices of head and tail entities in all entities
        idx_bpe_h= bpe_entity_to_idx[bpe_h]
        idx_bpe_t= bpe_entity_to_idx[bpe_t]

        # (2) Tensor representation of sequence of sub-word representation of entities and relations
        torch_bpe_h = torch.LongTensor(bpe_h).unsqueeze(0)
        torch_bpe_r = torch.LongTensor(bpe_r).unsqueeze(0)
        torch_bpe_t = torch.LongTensor(bpe_t).unsqueeze(0)

        # (3) Missing head and tail predictions
        x = torch.stack((torch.repeat_interleave(input=torch_bpe_h, repeats=num_entities, dim=0),
                         torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
                         all_bpe_entities), dim=1)
        predictions_tails = model(x)
        x = torch.stack((all_bpe_entities,
                         torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
                         torch.repeat_interleave(input=torch_bpe_t, repeats=num_entities, dim=0)), dim=1)
        predictions_heads = model(x)
        # 3. Computed filtered ranks for missing tail entities.
        # 3.1. Compute filtered tail entity rankings
        filt_tails = [bpe_entity_to_idx[i] for i in er_vocab[(bpe_h, bpe_r)]]
        # 3.2 Get the predicted target's score
        target_value = predictions_tails[idx_bpe_t].item()
        # 3.3 Filter scores of all triples containing filtered tail entities
        predictions_tails[filt_tails] = -np.Inf
        # 3.4 Reset the target's score
        predictions_tails[idx_bpe_t] = target_value
        # 3.5. Sort the score
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == idx_bpe_t)[0][0]

        # 4. Computed filtered ranks for missing head entities.
        # 4.1. Retrieve head entities to be filtered
        filt_heads = [bpe_entity_to_idx[i] for i in re_vocab[(bpe_r, bpe_t)]]
        # 4.2 Get the predicted target's score
        target_value = predictions_heads[idx_bpe_h].item()
        # 4.3 Filter scores of all triples containing filtered head entities.
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[idx_bpe_h] = target_value
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_head_entity_rank = np.where(sort_idxs == idx_bpe_h)[0][0]

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
