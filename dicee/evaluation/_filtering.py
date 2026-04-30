"""Filtering and ranking utilities for link prediction evaluation.

This module provides low-level helper functions for filtered ranking,
extracting common patterns shared across multiple evaluation functions.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch


def compute_filtered_rank(
    predictions: torch.Tensor,
    target_idx: int,
    filter_indices: List[int],
    exclude_target: bool = True
) -> int:
    """Compute filtered rank for a single prediction vector.

    Applies filtered setting by setting known correct entities to -Inf
    before ranking, then finds the rank of the target entity.

    Args:
        predictions: 1D tensor of prediction scores for all entities.
        target_idx: Index of the target entity to rank.
        filter_indices: Indices of entities to filter out (set to -Inf).
        exclude_target: If True, exclude target from filter_indices.

    Returns:
        1-indexed rank of the target entity (1 = best rank).

    Example:
        >>> predictions = torch.tensor([0.1, 0.9, 0.5, 0.3])
        >>> compute_filtered_rank(predictions, target_idx=2, filter_indices=[1, 2])
        1  # After filtering out index 1, target at index 2 ranks first
    """
    # Clone to avoid modifying input
    filtered_preds = predictions.clone()
    
    # Apply filtering
    if exclude_target:
        filter_set = set(filter_indices) - {target_idx}
    else:
        filter_set = set(filter_indices)
    
    if filter_set:
        filtered_preds[list(filter_set)] = -np.Inf
    
    # Restore target value only if it wasn't intentionally filtered
    if exclude_target or target_idx not in filter_indices:
        target_value = predictions[target_idx].item()
        filtered_preds[target_idx] = target_value
    
    # Sort and find rank
    _, sort_idxs = torch.sort(filtered_preds, descending=True)
    rank = np.where(sort_idxs.detach().cpu().numpy() == target_idx)[0][0]
    
    return rank + 1  # 1-indexed


def compute_filtered_rank_batch(
    predictions: torch.Tensor,
    target_indices: torch.Tensor,
    filter_indices_list: List[List[int]]
) -> List[int]:
    """Compute filtered ranks for a batch of predictions.

    Args:
        predictions: (batch_size, num_entities) tensor of scores.
        target_indices: (batch_size,) tensor of target entity indices.
        filter_indices_list: List of filter index lists, one per batch item.

    Returns:
        List of 1-indexed ranks, one per batch item.

    Example:
        >>> predictions = torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
        >>> targets = torch.tensor([2, 0])
        >>> filters = [[1, 2], [0, 2]]
        >>> compute_filtered_rank_batch(predictions, targets, filters)
        [1, 2]
    """
    batch_size = predictions.shape[0]
    ranks = []
    
    for i in range(batch_size):
        rank = compute_filtered_rank(
            predictions[i],
            target_indices[i].item(),
            filter_indices_list[i],
            exclude_target=True
        )
        ranks.append(rank)
    
    return ranks


def accumulate_bidirectional_hits(
    hits_dict: Dict[int, List],
    head_rank: int,
    tail_rank: int,
    hits_range: List[int] = None
) -> None:
    """Accumulate hits@k for bidirectional prediction (head + tail).

    Updates the hits dictionary in-place for both head and tail ranks.

    Args:
        hits_dict: Dictionary mapping k -> list of hit counts.
        head_rank: Rank of the head entity prediction (1-indexed).
        tail_rank: Rank of the tail entity prediction (1-indexed).
        hits_range: List of k values to track (default: 1-10).

    Example:
        >>> hits = {}
        >>> accumulate_bidirectional_hits(hits, head_rank=1, tail_rank=3)
        >>> hits[1]  # Both head (rank 1) and tail (rank 3) contribute
        [1]
        >>> hits[3]
        [2]  # Both hit@3
    """
    if hits_range is None:
        hits_range = list(range(1, 11))
    
    for k in hits_range:
        count = 0
        if head_rank <= k:
            count += 1
        if tail_rank <= k:
            count += 1
        if count > 0:
            hits_dict.setdefault(k, []).append(count)


def build_bpe_entity_index(
    bpe_entities,
    shaped_key_fn=None
) -> Tuple[Dict, torch.LongTensor]:
    """Build index mapping for BPE-encoded entities.

    Args:
        bpe_entities: Iterable of BPE entity representations.
            Can be tuples of (str_entity, bpe_entity, shaped_bpe_entity)
            or just shaped_bpe_entity values.
        shaped_key_fn: Optional function to extract shaped key from item.
            If None, assumes items are shaped BPE entities directly or
            tuples where shaped entity is at index 2.

    Returns:
        Tuple of:
            - Dictionary mapping shaped BPE entity -> integer index
            - LongTensor of all shaped BPE entities (num_entities, seq_len)

    Example:
        >>> entities = [(ent1, bpe1, shaped1), (ent2, bpe2, shaped2)]
        >>> idx_map, tensor = build_bpe_entity_index(entities)
        >>> idx_map[shaped1]
        0
        >>> tensor.shape
        torch.Size([2, seq_len])
    """
    bpe_entity_to_idx = {}
    all_bpe_entities = []
    
    for idx, item in enumerate(bpe_entities):
        if shaped_key_fn is not None:
            shaped_entity = shaped_key_fn(item)
        elif isinstance(item, tuple) and len(item) == 3:
            # (str_entity, bpe_entity, shaped_bpe_entity)
            shaped_entity = item[2]
        else:
            # Assume item is shaped entity directly
            shaped_entity = item
        
        bpe_entity_to_idx[shaped_entity] = idx
        all_bpe_entities.append(shaped_entity)
    
    return bpe_entity_to_idx, torch.LongTensor(all_bpe_entities)
