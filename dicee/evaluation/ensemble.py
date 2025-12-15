"""Ensemble evaluation functions.

This module provides functions for evaluating ensemble models,
including weighted averaging and score normalization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import compute_metrics_from_ranks_simple


@torch.no_grad()
def evaluate_ensemble_link_prediction_performance(
    models: List,
    triples,
    er_vocab: Dict[Tuple, List],
    weights: Optional[List[float]] = None,
    batch_size: int = 512,
    weighted_averaging: bool = True,
    normalize_scores: bool = True
) -> Dict[str, float]:
    """Evaluate link prediction performance of an ensemble of KGE models.

    Combines predictions from multiple models using weighted or simple
    averaging, with optional score normalization.

    Args:
        models: List of KGE models (e.g., snapshots from training).
        triples: Test triples as numpy array or list, shape (N, 3),
            with integer indices (head, relation, tail).
        er_vocab: Mapping (head_idx, rel_idx) -> list of tail indices
            for filtered evaluation.
        weights: Weights for model averaging. Required if weighted_averaging
            is True. Must sum to 1 for proper averaging.
        batch_size: Batch size for processing triples.
        weighted_averaging: If True, use weighted averaging of predictions.
            If False, use simple mean.
        normalize_scores: If True, normalize scores to [0, 1] range per
            sample before averaging.

    Returns:
        Dictionary with H@1, H@3, H@10, and MRR metrics.

    Raises:
        AssertionError: If weighted_averaging is True but weights are not
            provided or have wrong length.

    Example:
        >>> from dicee.evaluation import evaluate_ensemble_link_prediction_performance
        >>> models = [model1, model2, model3]
        >>> weights = [0.5, 0.3, 0.2]
        >>> results = evaluate_ensemble_link_prediction_performance(
        ...     models, test_triples, er_vocab,
        ...     weights=weights, weighted_averaging=True
        ... )
        >>> print(f"MRR: {results['MRR']:.4f}")
    """
    num_triples = len(triples)
    ranks = []
    hits_range = list(range(1, 11))
    hits = {i: [] for i in hits_range}
    n_models = len(models)

    # Validate weights for weighted averaging
    if weighted_averaging:
        assert weights is not None, "Weights must be provided for weighted averaging."
        assert len(weights) == n_models, "Number of weights must match number of models."
        weights_tensor = torch.FloatTensor(weights)

    for i in range(0, num_triples, batch_size):
        data_batch = np.array(triples[i:i + batch_size])
        e1_idx_r_idx = torch.LongTensor(data_batch[:, [0, 1]])
        e2_idx = torch.LongTensor(data_batch[:, 2])

        # Collect predictions from all models
        preds_list = []
        for model in models:
            model.eval()
            preds_raw = model(e1_idx_r_idx)  # [batch, n_entities]

            if normalize_scores:
                # Min-max normalization per sample
                preds_min = preds_raw.min(dim=1, keepdim=True)[0]
                preds_max = preds_raw.max(dim=1, keepdim=True)[0]
                preds = (preds_raw - preds_min) / (preds_max - preds_min + 1e-8)
            else:
                preds = preds_raw

            preds_list.append(preds)

        # Stack predictions: [n_models, batch, n_entities]
        preds_stack = torch.stack(preds_list, dim=0)

        # Aggregate predictions
        if weighted_averaging:
            avg_preds = torch.sum(
                preds_stack * weights_tensor.view(-1, 1, 1),
                dim=0
            )
        else:
            avg_preds = torch.mean(preds_stack, dim=0)

        # Apply filtering for each sample in batch
        for j in range(data_batch.shape[0]):
            id_e, id_r, id_e_target = data_batch[j]
            filt = er_vocab.get((id_e, id_r), [])
            target_value = avg_preds[j, id_e_target].item()

            if len(filt) > 0:
                avg_preds[j, filt] = -np.Inf
            avg_preds[j, id_e_target] = target_value

        # Compute ranks
        _, sort_idxs = torch.sort(avg_preds, dim=1, descending=True)

        for j in range(data_batch.shape[0]):
            rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
            ranks.append(rank)

            for hits_level in hits_range:
                if rank <= hits_level:
                    hits[hits_level].append(1.0)

    assert len(triples) == len(ranks) == num_triples
    return compute_metrics_from_ranks_simple(ranks, num_triples, hits)
