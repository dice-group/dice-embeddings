"""Filtering and ranking utilities for link prediction evaluation.

This module provides low-level helper functions for filtered ranking,
extracting common patterns shared across multiple evaluation functions.
"""

from collections.abc import Iterable, Mapping
from typing import Dict, List, Optional, SupportsInt, Tuple

import numpy as np
import torch

TIE_POLICIES = ("sort", "optimistic", "random", "pessimistic")


def evaluation_tie_options(args) -> Dict:
    """Read evaluation options, including configurations saved before tie policies."""
    settings = args if isinstance(args, Mapping) else vars(args)
    seed = settings.get("eval_tie_seed")
    return {
        "tie_policy": settings.get("eval_tie_policy", "sort"),
        "tie_seed": settings.get("random_seed", 0) if seed is None else seed,
    }


class FilteredRanker:
    """One evaluation's tie policy and independent CPU random stream.

    Ties use exact score equality. Random ranks are uniform integers between
    the optimistic and pessimistic ranks, inclusive. Recreate this object for
    each evaluation to replay its random stream without changing model RNGs.
    A supplied generator allows callers to continue the stream across chunks.
    """

    def __init__(self, tie_policy: str = "sort", tie_seed: int = 0,
                 generator: Optional[torch.Generator] = None):
        if tie_policy not in TIE_POLICIES:
            raise ValueError(f"Unknown tie policy {tie_policy!r}; choose from {TIE_POLICIES}")
        self.tie_policy = tie_policy
        self.generator = generator
        if tie_policy == "random":
            if generator is None:
                self.generator = torch.Generator(device="cpu").manual_seed(tie_seed)
            elif generator.device.type != "cpu":
                raise ValueError("Tie-breaking requires a CPU torch.Generator")

    def rank(self, predictions: torch.Tensor, target_idx: SupportsInt,
             filter_indices: Iterable[int], exclude_target: bool = True) -> int:
        target_idx = int(target_idx)
        filtered = predictions.clone()
        filters = set(filter_indices)
        if exclude_target:
            filters.discard(target_idx)
        if filters:
            filtered[list(filters)] = -np.Inf

        if self.tie_policy == "sort":
            # Preserve DICE's original per-vector torch.sort ordering.
            order = torch.sort(filtered, descending=True).indices
            return int(torch.where(order == target_idx)[0].item()) + 1

        # An explicit mask also excludes filtered candidates when the target
        # itself has score -Inf. Masking scores alone would count false ties.
        eligible = torch.ones_like(predictions, dtype=torch.bool)
        if filters:
            eligible[list(filters)] = False
        eligible[target_idx] = False
        other_scores = filtered[eligible]
        target_score = filtered[target_idx]
        if torch.isnan(target_score) or torch.isnan(other_scores).any():
            raise ValueError("Cannot rank NaN prediction scores")
        optimistic = 1 + int((other_scores > target_score).sum().item())
        if self.tie_policy == "optimistic":
            return optimistic
        tied = int((other_scores == target_score).sum().item())
        if self.tie_policy == "pessimistic":
            return optimistic + tied
        if tied == 0:
            return optimistic
        return optimistic + int(torch.randint(tied + 1, (), generator=self.generator).item())

    def rank_batch(self, predictions: torch.Tensor, target_indices: Iterable[SupportsInt],
                   filter_indices_list: List[List[int]]) -> List[int]:
        if self.tie_policy != "sort":
            return self.ranks_from_bounds(self.bounds_batch(predictions, target_indices, filter_indices_list))

        # Existing KvsAll/ensemble evaluators sorted whole batches. Keep that
        # operation intact, including its device-dependent ordering of ties.
        filtered = predictions.clone()
        for i, (target, filters) in enumerate(zip(target_indices, filter_indices_list)):
            target = int(target)
            if len(filters):
                filtered[i, list(filters)] = -np.Inf
            filtered[i, target] = predictions[i, target]
        order = torch.sort(filtered, dim=1, descending=True).indices
        return [int(torch.where(row == int(target))[0].item()) + 1
                for row, target in zip(order, target_indices)]

    def bounds_batch(self, predictions, target_indices, filter_indices_list) -> List[Tuple[int, int]]:
        """Compute optimistic ranks and tie counts together on the score device.

        Only two integers per query leave the GPU. Sampling is deliberately
        separate so grouped graph inference can restore original query order
        before consuming the independent random-tie stream.
        """
        targets = torch.as_tensor(target_indices, device=predictions.device, dtype=torch.long)
        if not len(targets):
            return []
        eligible = torch.ones_like(predictions, dtype=torch.bool)
        rows, columns = [], []
        for row, filters in enumerate(filter_indices_list):
            columns.extend(filters)
            rows.extend([row] * len(filters))
        if columns:
            eligible[torch.tensor(rows, device=predictions.device), torch.tensor(columns, device=predictions.device)] = False
        eligible[torch.arange(len(targets), device=predictions.device), targets] = False
        target_scores = predictions.gather(1, targets[:, None])
        if target_scores.isnan().any() or (predictions.isnan() & eligible).any():
            raise ValueError('Cannot rank NaN prediction scores')
        better = ((predictions > target_scores) & eligible).sum(1) + 1
        tied = ((predictions == target_scores) & eligible).sum(1)
        return [(int(b), int(t)) for b, t in torch.stack((better, tied), 1).cpu().tolist()]

    def ranks_from_bounds(self, bounds: Iterable[Tuple[int, int]]) -> List[int]:
        ranks = []
        for optimistic, tied in bounds:
            if self.tie_policy == 'pessimistic':
                ranks.append(optimistic + tied)
            elif self.tie_policy == 'random' and tied:
                ranks.append(optimistic + int(torch.randint(tied + 1, (), generator=self.generator)))
            else:
                ranks.append(optimistic)
        return ranks


def compute_filtered_rank(
    predictions: torch.Tensor,
    target_idx: int,
    filter_indices: List[int],
    exclude_target: bool = True,
    *,
    tie_policy: str = "sort",
    tie_seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Compute filtered rank for a single prediction vector.

    Applies filtered setting by setting known correct entities to -Inf
    before ranking, then finds the rank of the target entity.

    Args:
        predictions: 1D tensor of prediction scores for all entities.
        target_idx: Index of the target entity to rank.
        filter_indices: Indices of entities to filter out (set to -Inf).
        exclude_target: If True, exclude target from filter_indices.
        tie_policy: 'sort' (legacy), 'optimistic', 'random', or 'pessimistic'.
        tie_seed: Seed for random ties when no generator is supplied.
        generator: Optional CPU generator shared across successive queries.

    Returns:
        1-indexed rank of the target entity (1 = best rank).

    Example:
        >>> predictions = torch.tensor([0.1, 0.9, 0.5, 0.3])
        >>> compute_filtered_rank(predictions, target_idx=2, filter_indices=[1, 2])
        1  # After filtering out index 1, target at index 2 ranks first
    """
    return FilteredRanker(tie_policy, tie_seed, generator).rank(
        predictions, target_idx, filter_indices, exclude_target
    )


def compute_filtered_rank_batch(
    predictions: torch.Tensor,
    target_indices: torch.Tensor,
    filter_indices_list: List[List[int]],
    *,
    tie_policy: str = "sort",
    tie_seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Compute filtered ranks for a batch of predictions.

    Args:
        predictions: (batch_size, num_entities) tensor of scores.
        target_indices: (batch_size,) tensor of target entity indices.
        filter_indices_list: List of filter index lists, one per batch item.
        tie_policy: 'sort' (legacy), 'optimistic', 'random', or 'pessimistic'.
        tie_seed: Seed for random ties when no generator is supplied.
        generator: Optional CPU generator shared across successive batches.

    Returns:
        List of 1-indexed ranks, one per batch item.

    Example:
        >>> predictions = torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
        >>> targets = torch.tensor([2, 0])
        >>> filters = [[1, 2], [0, 2]]
        >>> compute_filtered_rank_batch(predictions, targets, filters)
        [1, 2]
    """
    ranker = FilteredRanker(tie_policy, tie_seed, generator)
    # Preserve the original helper's per-vector sorting for the default policy.
    return [ranker.rank(scores, target, filters) for scores, target, filters in
            zip(predictions, target_indices, filter_indices_list)]


def accumulate_bidirectional_hits(
    hits_dict: Dict[int, List],
    head_rank: int,
    tail_rank: int,
    hits_range: Optional[List[int]] = None
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
