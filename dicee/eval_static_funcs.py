"""Static evaluation functions for KGE models.

This module provides backward compatibility by re-exporting from the
new dicee.evaluation module.

.. deprecated::
    Use ``dicee.evaluation`` submodules instead. This module will be
    removed in a future version.
"""

# Re-export from new evaluation module for backward compatibility
from .evaluation.link_prediction import (
    evaluate_link_prediction_performance,
    evaluate_link_prediction_performance_with_reciprocals,
    evaluate_link_prediction_performance_with_bpe,
    evaluate_link_prediction_performance_with_bpe_reciprocals,
    evaluate_lp_bpe_k_vs_all,
)
from .evaluation.literal_prediction import evaluate_literal_prediction
from .evaluation.ensemble import evaluate_ensemble_link_prediction_performance

__all__ = [
    "evaluate_link_prediction_performance",
    "evaluate_link_prediction_performance_with_reciprocals",
    "evaluate_link_prediction_performance_with_bpe",
    "evaluate_link_prediction_performance_with_bpe_reciprocals",
    "evaluate_lp_bpe_k_vs_all",
    "evaluate_literal_prediction",
    "evaluate_ensemble_link_prediction_performance",
]
