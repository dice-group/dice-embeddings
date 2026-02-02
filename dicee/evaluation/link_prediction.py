"""Link prediction evaluation functions.

This module provides various functions for evaluating link prediction
performance of knowledge graph embedding models.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .utils import (
    compute_metrics_from_ranks,
    compute_metrics_from_ranks_simple,
    update_hits,
    create_hits_dict,
    ALL_HITS_RANGE,
)


@torch.no_grad()
def evaluate_link_prediction_performance(
    model,
    triples,
    er_vocab: Dict[Tuple, List],
    re_vocab: Dict[Tuple, List]
) -> Dict[str, float]:
    """Evaluate link prediction performance with head and tail prediction.

    Performs filtered evaluation where known correct answers are filtered
    out before computing ranks.

    Args:
        model: KGE model wrapper with entity/relation mappings.
        triples: Test triples as list of (head, relation, tail) strings.
        er_vocab: Mapping (entity, relation) -> list of valid tail entities.
        re_vocab: Mapping (relation, entity) -> list of valid head entities.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.
    """
    model.model.eval()
    hits = {}
    reciprocal_ranks = []
    num_entities = model.num_entities

    all_entities = torch.arange(0, num_entities).long()
    all_entities = all_entities.reshape(len(all_entities),)

    for i in tqdm(range(0, len(triples))):
        data_point = triples[i]
        str_h, str_r, str_t = data_point[0], data_point[1], data_point[2]

        h = model.get_entity_index(str_h)
        r = model.get_relation_index(str_r)
        t = model.get_entity_index(str_t)

        # Predict missing tails
        x = torch.stack((
            torch.tensor(h).repeat(num_entities,),
            torch.tensor(r).repeat(num_entities,),
            all_entities
        ), dim=1)
        predictions_tails = model.model.forward_triples(x)

        # Predict missing heads
        x = torch.stack((
            all_entities,
            torch.tensor(r).repeat(num_entities,),
            torch.tensor(t).repeat(num_entities)
        ), dim=1)
        predictions_heads = model.model.forward_triples(x)
        del x

        # Filtered tail ranking
        filt_tails = [model.entity_to_idx[i] for i in er_vocab[(str_h, str_r)]]
        target_value = predictions_tails[t].item()
        predictions_tails[filt_tails] = -np.Inf
        predictions_tails[t] = target_value
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == t)[0][0]

        # Filtered head ranking
        filt_heads = [model.entity_to_idx[i] for i in re_vocab[(str_r, str_t)]]
        target_value = predictions_heads[h].item()
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[h] = target_value
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_head_entity_rank = np.where(sort_idxs == h)[0][0]

        # Add 1 as numpy arrays are 0-indexed
        filt_head_entity_rank += 1
        filt_tail_entity_rank += 1

        rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
        reciprocal_ranks.append(rr)

        # Compute Hit@N
        for hits_level in range(1, 11):
            res = 1 if filt_head_entity_rank <= hits_level else 0
            res += 1 if filt_tail_entity_rank <= hits_level else 0
            if res > 0:
                hits.setdefault(hits_level, []).append(res)

    return compute_metrics_from_ranks(
        ranks=[],  # Not used directly
        num_triples=len(triples),
        hits_dict=hits,
        scale_factor=2
    ) | {'MRR': sum(reciprocal_ranks) / (float(len(triples) * 2))}


@torch.no_grad()
def evaluate_link_prediction_performance_with_reciprocals(
    model,
    triples,
    er_vocab: Dict[Tuple, List]
) -> Dict[str, float]:
    """Evaluate link prediction with reciprocal relations.

    Optimized for models trained with reciprocal triples where only
    tail prediction is needed.

    Args:
        model: KGE model wrapper.
        triples: Test triples as list of (head, relation, tail) strings.
        er_vocab: Mapping (entity, relation) -> list of valid tail entities.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.
    """
    model.model.eval()
    entity_to_idx = model.entity_to_idx
    relation_to_idx = model.relation_to_idx
    batch_size = model.model.args["batch_size"]
    num_triples = len(triples)
    ranks: List[int] = []
    hits_range = ALL_HITS_RANGE
    hits = create_hits_dict(hits_range)

    for i in range(0, num_triples, batch_size):
        str_data_batch = triples[i:i + batch_size]
        data_batch = np.array([
            [entity_to_idx[str_triple[0]],
             relation_to_idx[str_triple[1]],
             entity_to_idx[str_triple[2]]]
            for str_triple in str_data_batch
        ])

        e1_idx_r_idx = torch.LongTensor(data_batch[:, [0, 1]])
        e2_idx = torch.tensor(data_batch[:, 2])
        predictions = model.model(e1_idx_r_idx)

        for j in range(data_batch.shape[0]):
            str_h, str_r, str_t = str_data_batch[j]
            id_e, id_r, id_e_target = data_batch[j]

            filt = [entity_to_idx[_] for _ in er_vocab[(str_h, str_r)]]
            target_value = predictions[j, id_e_target].item()
            predictions[j, filt] = -np.Inf
            predictions[j, id_e_target] = target_value

        _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

        for j in range(data_batch.shape[0]):
            rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
            ranks.append(rank)
            update_hits(hits, rank, hits_range)

    assert len(triples) == len(ranks) == num_triples
    return compute_metrics_from_ranks_simple(ranks, num_triples, hits)


def evaluate_link_prediction_performance_with_bpe_reciprocals(
    model,
    within_entities: List[str],
    triples: List[List[str]],
    er_vocab: Dict[Tuple, List]
) -> Dict[str, float]:
    """Evaluate link prediction with BPE encoding and reciprocals.

    Args:
        model: KGE model wrapper with BPE support.
        within_entities: List of entities to evaluate within.
        triples: Test triples as list of [head, relation, tail] strings.
        er_vocab: Mapping (entity, relation) -> list of valid tail entities.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.
    """
    triples = np.array(triples)
    model.model.eval()
    entity_to_idx = {ent: id_ for id_, ent in enumerate(within_entities)}
    padded_bpe_within_entities = model.get_bpe_token_representation(within_entities)
    padded_bpe_within_entities = torch.LongTensor(padded_bpe_within_entities)

    batch_size = model.model.args["batch_size"]
    num_triples = len(triples)
    ranks: List[int] = []
    hits_range = ALL_HITS_RANGE
    hits = create_hits_dict(hits_range)

    model.model.ordered_bpe_entities = padded_bpe_within_entities

    for i in range(0, num_triples, batch_size):
        str_data_batch = triples[i:i + batch_size]
        str_heads, str_rels = str_data_batch[:, 0].tolist(), str_data_batch[:, 1].tolist()

        padded_bpe_heads = torch.LongTensor(
            model.get_bpe_token_representation(str_heads)
        ).unsqueeze(1)
        padded_bpe_rels = torch.LongTensor(
            model.get_bpe_token_representation(str_rels)
        ).unsqueeze(1)

        e1_idx_r_idx = torch.cat((padded_bpe_heads, padded_bpe_rels), dim=1)
        predictions = model.model(e1_idx_r_idx)

        for j, (str_h, str_r, str_t) in enumerate(str_data_batch):
            id_e_target = entity_to_idx[str_t]
            filt = [entity_to_idx[_] for _ in er_vocab[(str_h, str_r)]]
            target_value = predictions[j, id_e_target].item()
            predictions[j, filt] = -np.Inf
            predictions[j, id_e_target] = target_value

        _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

        for j, (_, __, str_t) in enumerate(str_data_batch):
            rank = torch.where(sort_idxs[j] == entity_to_idx[str_t])[0].item() + 1
            ranks.append(rank)
            update_hits(hits, rank, hits_range)

    assert len(triples) == len(ranks) == num_triples
    return compute_metrics_from_ranks_simple(ranks, num_triples, hits)


def evaluate_link_prediction_performance_with_bpe(
    model,
    within_entities: List[str],
    triples: List[Tuple[str]],
    er_vocab: Dict[Tuple, List],
    re_vocab: Dict[Tuple, List]
) -> Dict[str, float]:
    """Evaluate link prediction with BPE encoding (head and tail).

    Args:
        model: KGE model wrapper with BPE support.
        within_entities: List of entities to evaluate within.
        triples: Test triples as list of (head, relation, tail) tuples.
        er_vocab: Mapping (entity, relation) -> list of valid tail entities.
        re_vocab: Mapping (relation, entity) -> list of valid head entities.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.
    """
    assert isinstance(triples, list)
    assert len(triples[0]) == 3
    model.model.eval()
    hits = {}
    reciprocal_ranks = []

    num_entities = len(within_entities)
    bpe_entity_to_idx = {}
    all_bpe_entities = []

    for idx, str_entity in tqdm(enumerate(within_entities)):
        shaped_bpe_entity = model.get_bpe_token_representation(str_entity)
        bpe_entity_to_idx[shaped_bpe_entity] = idx
        all_bpe_entities.append(shaped_bpe_entity)
    all_bpe_entities = torch.LongTensor(all_bpe_entities)

    for str_h, str_r, str_t in tqdm(triples):
        idx_bpe_h = bpe_entity_to_idx[model.get_bpe_token_representation(str_h)]
        idx_bpe_t = bpe_entity_to_idx[model.get_bpe_token_representation(str_t)]

        torch_bpe_h = torch.LongTensor(
            model.get_bpe_token_representation(str_h)
        ).unsqueeze(0)
        torch_bpe_r = torch.LongTensor(
            model.get_bpe_token_representation(str_r)
        ).unsqueeze(0)
        torch_bpe_t = torch.LongTensor(
            model.get_bpe_token_representation(str_t)
        ).unsqueeze(0)

        # Missing tail predictions
        x = torch.stack((
            torch.repeat_interleave(input=torch_bpe_h, repeats=num_entities, dim=0),
            torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
            all_bpe_entities
        ), dim=1)
        with torch.no_grad():
            predictions_tails = model.model(x)

        # Missing head predictions
        x = torch.stack((
            all_bpe_entities,
            torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
            torch.repeat_interleave(input=torch_bpe_t, repeats=num_entities, dim=0)
        ), dim=1)
        with torch.no_grad():
            predictions_heads = model.model(x)

        # Filter tails
        filt_tails = [
            bpe_entity_to_idx[model.get_bpe_token_representation(i)]
            for i in er_vocab[(str_h, str_r)]
        ]
        target_value = predictions_tails[idx_bpe_t].item()
        predictions_tails[filt_tails] = -np.Inf
        predictions_tails[idx_bpe_t] = target_value
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == idx_bpe_t)[0][0]

        # Filter heads
        filt_heads = [
            bpe_entity_to_idx[model.get_bpe_token_representation(i)]
            for i in re_vocab[(str_r, str_t)]
        ]
        target_value = predictions_heads[idx_bpe_h].item()
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[idx_bpe_h] = target_value
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_head_entity_rank = np.where(sort_idxs == idx_bpe_h)[0][0]

        filt_head_entity_rank += 1
        filt_tail_entity_rank += 1

        rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
        reciprocal_ranks.append(rr)

        for hits_level in range(1, 11):
            res = 1 if filt_head_entity_rank <= hits_level else 0
            res += 1 if filt_tail_entity_rank <= hits_level else 0
            if res > 0:
                hits.setdefault(hits_level, []).append(res)

    return compute_metrics_from_ranks(
        ranks=[],
        num_triples=len(triples),
        hits_dict=hits,
        scale_factor=2
    ) | {'MRR': sum(reciprocal_ranks) / (float(len(triples) * 2))}


@torch.no_grad()
def evaluate_lp(
    model,
    triple_idx,
    num_entities: int,
    er_vocab: Dict[Tuple, List],
    re_vocab: Dict[Tuple, List],
    info: str = 'Eval Starts',
    batch_size: int = 128,
    chunk_size: int = 1000
) -> Dict[str, float]:
    """Evaluate link prediction with batched processing.

    Memory-efficient evaluation using chunked entity scoring.

    Args:
        model: The KGE model to evaluate.
        triple_idx: Integer-indexed triples as numpy array.
        num_entities: Total number of entities.
        er_vocab: Mapping (head_idx, rel_idx) -> list of tail indices.
        re_vocab: Mapping (rel_idx, tail_idx) -> list of head indices.
        info: Description to print.
        batch_size: Batch size for triple processing.
        chunk_size: Chunk size for entity scoring.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.
    """
    assert model is not None, "Model must be provided"
    assert triple_idx is not None, "triple_idx must be provided"
    assert num_entities is not None, "num_entities must be provided"
    assert er_vocab is not None, "er_vocab must be provided"
    assert re_vocab is not None, "re_vocab must be provided"

    model.eval()
    print(info)
    print(f'Num of triples {len(triple_idx)}')

    hits = {}
    reciprocal_ranks = []
    all_entities = torch.arange(0, num_entities).long()

    for batch_start in tqdm(range(0, len(triple_idx), batch_size), desc="Evaluating Batches"):
        batch_end = min(batch_start + batch_size, len(triple_idx))
        batch_triples = triple_idx[batch_start:batch_end]
        batch_size_current = len(batch_triples)

        h_batch = torch.tensor([dp[0] for dp in batch_triples])
        r_batch = torch.tensor([dp[1] for dp in batch_triples])
        t_batch = torch.tensor([dp[2] for dp in batch_triples])

        predictions_tails = torch.zeros(batch_size_current, num_entities)
        predictions_heads = torch.zeros(batch_size_current, num_entities)

        # Process entities in chunks
        for chunk_start in range(0, num_entities, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_entities)
            entities_chunk = all_entities[chunk_start:chunk_end]
            chunk_size_current = entities_chunk.size(0)

            # Tail prediction
            x_tails = torch.stack((
                h_batch.repeat_interleave(chunk_size_current),
                r_batch.repeat_interleave(chunk_size_current),
                entities_chunk.repeat(batch_size_current)
            ), dim=1)
            preds_tails = model(x_tails).view(batch_size_current, chunk_size_current)
            predictions_tails[:, chunk_start:chunk_end] = preds_tails
            del x_tails

            # Head prediction
            x_heads = torch.stack((
                entities_chunk.repeat(batch_size_current),
                r_batch.repeat_interleave(chunk_size_current),
                t_batch.repeat_interleave(chunk_size_current)
            ), dim=1)
            preds_heads = model(x_heads).view(batch_size_current, chunk_size_current)
            predictions_heads[:, chunk_start:chunk_end] = preds_heads
            del x_heads

        # Compute filtered ranks
        for i in range(batch_size_current):
            h = h_batch[i].item()
            r = r_batch[i].item()
            t = t_batch[i].item()

            # Tail filtering
            filt_tails = set(er_vocab[(h, r)]) - {t}
            target_value = predictions_tails[i, t].item()
            predictions_tails[i, list(filt_tails)] = -np.Inf
            predictions_tails[i, t] = target_value
            _, sort_idxs = torch.sort(predictions_tails[i], descending=True)
            filt_tail_entity_rank = np.where(sort_idxs.detach() == t)[0][0]

            # Head filtering
            filt_heads = set(re_vocab[(r, t)]) - {h}
            target_value = predictions_heads[i, h].item()
            predictions_heads[i, list(filt_heads)] = -np.Inf
            predictions_heads[i, h] = target_value
            _, sort_idxs = torch.sort(predictions_heads[i], descending=True)
            filt_head_entity_rank = np.where(sort_idxs.detach() == h)[0][0]

            filt_head_entity_rank += 1
            filt_tail_entity_rank += 1

            rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
            reciprocal_ranks.append(rr)

            for hits_level in range(1, 11):
                res = 1 if filt_head_entity_rank <= hits_level else 0
                res += 1 if filt_tail_entity_rank <= hits_level else 0
                if res > 0:
                    hits.setdefault(hits_level, []).append(res)

    results = compute_metrics_from_ranks(
        ranks=[],
        num_triples=len(triple_idx),
        hits_dict=hits,
        scale_factor=2
    ) | {'MRR': sum(reciprocal_ranks) / (float(len(triple_idx) * 2))}

    print(results)
    return results


@torch.no_grad()
def evaluate_bpe_lp(
    model,
    triple_idx: List[Tuple],
    all_bpe_shaped_entities,
    er_vocab: Dict[Tuple, List],
    re_vocab: Dict[Tuple, List],
    info: str = 'Eval Starts'
) -> Dict[str, float]:
    """Evaluate link prediction with BPE-encoded entities.

    Args:
        model: The KGE model to evaluate.
        triple_idx: List of BPE-encoded triple tuples.
        all_bpe_shaped_entities: All entities with BPE representations.
        er_vocab: Mapping for tail filtering.
        re_vocab: Mapping for head filtering.
        info: Description to print.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.
    """
    assert isinstance(triple_idx, list)
    assert isinstance(triple_idx[0], tuple)
    assert len(triple_idx[0]) == 3

    model.eval()
    print(info)
    print(f'Num of triples {len(triple_idx)}')

    hits = {}
    reciprocal_ranks = []
    num_entities = len(all_bpe_shaped_entities)

    bpe_entity_to_idx = {}
    all_bpe_entities = []

    for idx, (str_entity, bpe_entity, shaped_bpe_entity) in tqdm(enumerate(all_bpe_shaped_entities)):
        bpe_entity_to_idx[shaped_bpe_entity] = idx
        all_bpe_entities.append(shaped_bpe_entity)
    all_bpe_entities = torch.LongTensor(all_bpe_entities)

    for (bpe_h, bpe_r, bpe_t) in tqdm(triple_idx):
        idx_bpe_h = bpe_entity_to_idx[bpe_h]
        idx_bpe_t = bpe_entity_to_idx[bpe_t]

        torch_bpe_h = torch.LongTensor(bpe_h).unsqueeze(0)
        torch_bpe_r = torch.LongTensor(bpe_r).unsqueeze(0)
        torch_bpe_t = torch.LongTensor(bpe_t).unsqueeze(0)

        # Tail predictions
        x = torch.stack((
            torch.repeat_interleave(input=torch_bpe_h, repeats=num_entities, dim=0),
            torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
            all_bpe_entities
        ), dim=1)
        predictions_tails = model(x)

        # Head predictions
        x = torch.stack((
            all_bpe_entities,
            torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
            torch.repeat_interleave(input=torch_bpe_t, repeats=num_entities, dim=0)
        ), dim=1)
        predictions_heads = model(x)

        # Filter tails
        filt_tails = [bpe_entity_to_idx[i] for i in er_vocab[(bpe_h, bpe_r)]]
        target_value = predictions_tails[idx_bpe_t].item()
        predictions_tails[filt_tails] = -np.Inf
        predictions_tails[idx_bpe_t] = target_value
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        filt_tail_entity_rank = np.where(sort_idxs.detach() == idx_bpe_t)[0][0]

        # Filter heads
        filt_heads = [bpe_entity_to_idx[i] for i in re_vocab[(bpe_r, bpe_t)]]
        target_value = predictions_heads[idx_bpe_h].item()
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[idx_bpe_h] = target_value
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        filt_head_entity_rank = np.where(sort_idxs.detach() == idx_bpe_h)[0][0]

        filt_head_entity_rank += 1
        filt_tail_entity_rank += 1

        rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
        reciprocal_ranks.append(rr)

        for hits_level in range(1, 11):
            res = 1 if filt_head_entity_rank <= hits_level else 0
            res += 1 if filt_tail_entity_rank <= hits_level else 0
            if res > 0:
                hits.setdefault(hits_level, []).append(res)

    results = compute_metrics_from_ranks(
        ranks=[],
        num_triples=len(triple_idx),
        hits_dict=hits,
        scale_factor=2
    ) | {'MRR': sum(reciprocal_ranks) / (float(len(triple_idx) * 2))}

    print(results)
    return results


@torch.no_grad()
def evaluate_lp_bpe_k_vs_all(
    model,
    triples: List[List[str]],
    er_vocab: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    func_triple_to_bpe_representation: Optional[Callable] = None,
    str_to_bpe_entity_to_idx: Optional[Dict] = None
) -> Dict[str, float]:
    """Evaluate BPE link prediction with KvsAll scoring.

    Args:
        model: The KGE model wrapper.
        triples: List of string triples.
        er_vocab: Entity-relation vocabulary for filtering.
        batch_size: Batch size for processing.
        func_triple_to_bpe_representation: Function to convert triples to BPE.
        str_to_bpe_entity_to_idx: Mapping from string entities to BPE indices.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.

    Raises:
        ValueError: If batch_size is not provided.
    """
    if batch_size is None:
        raise ValueError("batch_size must be provided")

    model.model.eval()
    num_triples = len(triples)
    ranks: List[int] = []
    hits_range = ALL_HITS_RANGE
    hits = create_hits_dict(hits_range)

    for i in range(0, num_triples, batch_size):
        str_data_batch = triples[i:i + batch_size]
        torch_batch_bpe_triple = torch.LongTensor([
            func_triple_to_bpe_representation(t) for t in str_data_batch
        ])

        bpe_hr = torch_batch_bpe_triple[:, [0, 1], :]
        predictions = model(bpe_hr)

        for j in range(len(predictions)):
            h, r, t = str_data_batch[j]
            id_e_target = str_to_bpe_entity_to_idx[t]
            filt_idx_entities = [str_to_bpe_entity_to_idx[_] for _ in er_vocab[(h, r)]]

            target_value = predictions[j, id_e_target].item()
            predictions[j, filt_idx_entities] = -np.Inf
            predictions[j, id_e_target] = target_value

        _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

        for j in range(len(predictions)):
            t = str_data_batch[j][2]
            rank = torch.where(sort_idxs[j] == str_to_bpe_entity_to_idx[t])[0].item() + 1
            ranks.append(rank)
            update_hits(hits, rank, hits_range)

    assert len(triples) == len(ranks) == num_triples
    return compute_metrics_from_ranks_simple(ranks, num_triples, hits)
