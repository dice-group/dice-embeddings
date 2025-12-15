"""Evaluator module for knowledge graph embedding models.

This module provides backward compatibility by re-exporting from the
new dicee.evaluation module.

.. deprecated::
    Use ``dicee.evaluation.Evaluator`` instead. This module will be
    removed in a future version.
"""

# Re-export from new location for backward compatibility
from .evaluation.evaluator import Evaluator

__all__ = ["Evaluator"]
