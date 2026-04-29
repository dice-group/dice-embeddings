"""Training-related static functions.

This module provides backward compatibility by re-exporting evaluation
functions from the new dicee.evaluation module, along with training utilities.

.. deprecated::
    Evaluation functions have moved to ``dicee.evaluation``. Use that module
    for new code. This module will continue to export training utilities.
"""

# Re-export from new evaluation module for backward compatibility
from .evaluation.link_prediction import evaluate_lp, evaluate_bpe_lp
from .evaluation.utils import make_iterable_verbose, efficient_zero_grad

__all__ = [
    "evaluate_lp",
    "evaluate_bpe_lp",
    "make_iterable_verbose",
    "efficient_zero_grad",
]
