"""Evaluation module for knowledge graph embedding models.

This module provides comprehensive evaluation capabilities for KGE models,
including link prediction, literal prediction, and ensemble evaluation.

Modules:
    link_prediction: Functions for evaluating link prediction performance
    literal_prediction: Functions for evaluating literal/attribute prediction
    ensemble: Functions for ensemble model evaluation
    evaluator: Main Evaluator class for integrated evaluation
    utils: Shared utility functions for evaluation

Example:
    >>> from dicee.evaluation import Evaluator
    >>> from dicee.evaluation.link_prediction import evaluate_link_prediction_performance
    >>> from dicee.evaluation.ensemble import evaluate_ensemble_link_prediction_performance
"""

from .evaluator import Evaluator
from .link_prediction import (
    evaluate_link_prediction_performance,
    evaluate_link_prediction_performance_with_reciprocals,
    evaluate_link_prediction_performance_with_bpe,
    evaluate_link_prediction_performance_with_bpe_reciprocals,
    evaluate_lp,
    evaluate_lp_bpe_k_vs_all,
    evaluate_bpe_lp,
)
from .literal_prediction import evaluate_literal_prediction
from .ensemble import evaluate_ensemble_link_prediction_performance
from .utils import compute_metrics_from_ranks, make_iterable_verbose

__all__ = [
    # Main evaluator class
    "Evaluator",
    # Link prediction functions
    "evaluate_link_prediction_performance",
    "evaluate_link_prediction_performance_with_reciprocals",
    "evaluate_link_prediction_performance_with_bpe",
    "evaluate_link_prediction_performance_with_bpe_reciprocals",
    "evaluate_lp",
    "evaluate_lp_bpe_k_vs_all",
    "evaluate_bpe_lp",
    # Literal prediction
    "evaluate_literal_prediction",
    # Ensemble evaluation
    "evaluate_ensemble_link_prediction_performance",
    # Utilities
    "compute_metrics_from_ranks",
    "make_iterable_verbose",
]
