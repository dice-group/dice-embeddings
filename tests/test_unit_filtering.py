"""Unit tests for dicee.evaluation._filtering module.

Tests the filtering and ranking helper functions used across evaluation.
"""

import torch
import pytest
from dicee.evaluation._filtering import (
    compute_filtered_rank,
    compute_filtered_rank_batch,
    accumulate_bidirectional_hits,
    build_bpe_entity_index,
)


class TestComputeFilteredRank:
    """Test compute_filtered_rank function."""

    def test_basic_ranking(self):
        """Test basic filtered ranking without filtering."""
        predictions = torch.tensor([0.1, 0.9, 0.5, 0.3])
        rank = compute_filtered_rank(predictions, target_idx=1, filter_indices=[])
        assert rank == 1, "Highest score (0.9) should rank 1st"

    def test_with_filtering(self):
        """Test filtered ranking with entities filtered out."""
        predictions = torch.tensor([0.1, 0.9, 0.5, 0.3])
        # Filter out index 1 (highest), target at index 2 should become rank 1
        rank = compute_filtered_rank(predictions, target_idx=2, filter_indices=[1])
        assert rank == 1, "After filtering index 1, target at 2 should rank 1st"

    def test_target_not_filtered(self):
        """Test that target is excluded from filter list."""
        predictions = torch.tensor([0.1, 0.9, 0.5, 0.3])
        # Even though target is in filter list, it should not be filtered
        rank = compute_filtered_rank(predictions, target_idx=2, filter_indices=[1, 2])
        assert rank == 1, "Target should not be filtered even if in filter list"

    def test_multiple_filtered_entities(self):
        """Test with multiple entities filtered."""
        predictions = torch.tensor([0.8, 0.9, 0.5, 0.7])
        # Filter indices 0 and 1, target at 3 should rank 1st among remaining
        rank = compute_filtered_rank(predictions, target_idx=3, filter_indices=[0, 1])
        assert rank == 1, "Target should rank 1st after filtering"

    def test_middle_rank(self):
        """Test target ranking in the middle."""
        predictions = torch.tensor([0.9, 0.8, 0.5, 0.7])
        rank = compute_filtered_rank(predictions, target_idx=3, filter_indices=[])
        assert rank == 3, "Target with 3rd highest score should rank 3rd"

    def test_worst_rank(self):
        """Test target with worst score."""
        predictions = torch.tensor([0.9, 0.8, 0.7, 0.1])
        rank = compute_filtered_rank(predictions, target_idx=3, filter_indices=[])
        assert rank == 4, "Target with lowest score should rank last"

    def test_exclude_target_false(self):
        """Test with exclude_target=False."""
        predictions = torch.tensor([0.1, 0.9, 0.5, 0.3])
        # When exclude_target=False, target IS filtered and gets low rank
        rank = compute_filtered_rank(
            predictions, target_idx=2, filter_indices=[2], exclude_target=False
        )
        # After setting index 2 to -Inf, it should rank last
        assert rank == 4, "When exclude_target=False, target can be filtered"


class TestComputeFilteredRankBatch:
    """Test compute_filtered_rank_batch function."""

    def test_batch_ranking(self):
        """Test batch filtered ranking."""
        predictions = torch.tensor([
            [0.1, 0.9, 0.5, 0.3],
            [0.3, 0.2, 0.8, 0.4]
        ])
        targets = torch.tensor([1, 2])
        filters = [[], []]
        ranks = compute_filtered_rank_batch(predictions, targets, filters)
        assert ranks == [1, 1], "Both targets should rank 1st"

    def test_batch_with_filtering(self):
        """Test batch ranking with filtering."""
        predictions = torch.tensor([
            [0.1, 0.9, 0.5, 0.3],
            [0.3, 0.9, 0.8, 0.4]
        ])
        targets = torch.tensor([2, 2])
        filters = [[1], [1]]  # Filter out index 1 in both
        ranks = compute_filtered_rank_batch(predictions, targets, filters)
        assert ranks[0] == 1, "After filtering, target should rank 1st"
        assert ranks[1] == 1, "After filtering, target should rank 1st"

    def test_batch_different_filters(self):
        """Test batch with different filters per item."""
        predictions = torch.tensor([
            [0.1, 0.9, 0.5, 0.3],
            [0.3, 0.2, 0.8, 0.4]
        ])
        targets = torch.tensor([2, 3])
        filters = [[1], [2]]
        ranks = compute_filtered_rank_batch(predictions, targets, filters)
        assert len(ranks) == 2
        assert all(r >= 1 for r in ranks), "All ranks should be at least 1"


class TestAccumulateBidirectionalHits:
    """Test accumulate_bidirectional_hits function."""

    def test_both_hit_at_1(self):
        """Test when both head and tail rank 1."""
        hits = {}
        accumulate_bidirectional_hits(hits, head_rank=1, tail_rank=1)
        assert hits[1] == [2], "Both head and tail hit@1 = 2"
        assert hits[10] == [2], "Both also hit@10"

    def test_one_hits_at_1(self):
        """Test when only head hits at 1."""
        hits = {}
        accumulate_bidirectional_hits(hits, head_rank=1, tail_rank=5)
        assert hits[1] == [1], "Only head hits@1"
        assert hits[5] == [2], "Both hit@5"
        assert hits[10] == [2], "Both hit@10"

    def test_neither_hits_at_1(self):
        """Test when neither hits at 1."""
        hits = {}
        accumulate_bidirectional_hits(hits, head_rank=3, tail_rank=5)
        assert 1 not in hits, "Neither hits@1"
        assert hits[3] == [1], "Only head hits@3"
        assert hits[5] == [2], "Both hit@5"

    def test_custom_hits_range(self):
        """Test with custom hits range."""
        hits = {}
        accumulate_bidirectional_hits(
            hits, head_rank=2, tail_rank=4, hits_range=[1, 3, 5, 10]
        )
        assert 1 not in hits
        assert hits[3] == [1], "Only head hits@3"
        assert hits[5] == [2], "Both hit@5"

    def test_accumulation_over_multiple_calls(self):
        """Test accumulating hits over multiple predictions."""
        hits = {}
        accumulate_bidirectional_hits(hits, head_rank=1, tail_rank=1)
        accumulate_bidirectional_hits(hits, head_rank=2, tail_rank=3)
        assert hits[1] == [2], "First call: both hit@1"
        assert hits[2] == [2, 1], "First call: both hit@2, second call: only head"
        assert hits[3] == [2, 2], "First call: both, second call: both"

    def test_high_ranks_no_contribution(self):
        """Test that high ranks don't contribute to low k values."""
        hits = {}
        accumulate_bidirectional_hits(hits, head_rank=15, tail_rank=20)
        for k in range(1, 11):
            assert k not in hits, f"No hits@{k} when ranks are > 10"


class TestBuildBpeEntityIndex:
    """Test build_bpe_entity_index function."""

    def test_tuple_format(self):
        """Test with (str_entity, bpe_entity, shaped_bpe_entity) tuples."""
        entities = [
            ("ent1", [1, 2], (1, 2)),
            ("ent2", [3, 4], (3, 4)),
            ("ent3", [5, 6], (5, 6)),
        ]
        idx_map, tensor = build_bpe_entity_index(entities)
        
        assert len(idx_map) == 3
        assert idx_map[(1, 2)] == 0
        assert idx_map[(3, 4)] == 1
        assert idx_map[(5, 6)] == 2
        assert tensor.shape == (3, 2)
        assert torch.equal(tensor, torch.LongTensor([[1, 2], [3, 4], [5, 6]]))

    def test_direct_shaped_format(self):
        """Test with shaped entities directly."""
        entities = [(1, 2), (3, 4), (5, 6)]
        idx_map, tensor = build_bpe_entity_index(entities)
        
        assert len(idx_map) == 3
        assert idx_map[(1, 2)] == 0
        assert tensor.shape == (3, 2)

    def test_custom_key_function(self):
        """Test with custom shaped_key_fn."""
        entities = [
            {"name": "ent1", "bpe": (1, 2)},
            {"name": "ent2", "bpe": (3, 4)},
        ]
        idx_map, tensor = build_bpe_entity_index(
            entities, shaped_key_fn=lambda x: x["bpe"]
        )
        
        assert len(idx_map) == 2
        assert idx_map[(1, 2)] == 0
        assert idx_map[(3, 4)] == 1

    def test_single_entity(self):
        """Test with single entity."""
        entities = [("ent1", [1, 2], (1, 2))]
        idx_map, tensor = build_bpe_entity_index(entities)
        
        assert len(idx_map) == 1
        assert tensor.shape == (1, 2)

    def test_preserves_order(self):
        """Test that index mapping preserves input order."""
        entities = [
            ("z", [9], (9,)),
            ("a", [1], (1,)),
            ("m", [5], (5,)),
        ]
        idx_map, tensor = build_bpe_entity_index(entities)
        
        assert idx_map[(9,)] == 0, "First entity gets index 0"
        assert idx_map[(1,)] == 1, "Second entity gets index 1"
        assert idx_map[(5,)] == 2, "Third entity gets index 2"


class TestFilteringIntegration:
    """Integration tests combining multiple filtering functions."""

    def test_full_evaluation_workflow(self):
        """Test simulating a mini link prediction evaluation."""
        # Setup: 4 entities, 1 triple to evaluate
        num_entities = 4
        predictions = torch.tensor([0.3, 0.9, 0.5, 0.7])
        
        # Triple: (h=0, r=0, t=2)
        # Known tails for (h=0, r=0): [1, 2] -> filter out 1
        # Known heads for (r=0, t=2): [0] -> no filtering needed for this example
        
        target_tail = 2
        filter_tails = [1, 2]  # Target will be auto-excluded
        
        tail_rank = compute_filtered_rank(predictions, target_tail, filter_tails)
        
        # After filtering index 1, remaining scores: [0.3, -inf, 0.5, 0.7]
        # Sorted descending: 0.7(idx=3), 0.5(idx=2), 0.3(idx=0), -inf(idx=1)
        # Target at index 2 ranks 2nd
        assert tail_rank == 2

    def test_bidirectional_evaluation(self):
        """Test full bidirectional evaluation with hits accumulation."""
        predictions_tails = torch.tensor([0.1, 0.9, 0.5, 0.3])
        predictions_heads = torch.tensor([0.8, 0.2, 0.3, 0.9])
        
        # Triple: (h=0, r=0, t=2)
        filter_tails = [1]
        filter_heads = [3]
        
        tail_rank = compute_filtered_rank(predictions_tails, 2, filter_tails)
        head_rank = compute_filtered_rank(predictions_heads, 0, filter_heads)
        
        hits = {}
        accumulate_bidirectional_hits(hits, head_rank, tail_rank)
        
        # Verify hits were accumulated
        assert len(hits) > 0
        assert all(isinstance(v, list) for v in hits.values())
