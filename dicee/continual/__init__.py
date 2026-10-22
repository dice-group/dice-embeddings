"""Controlled Continual Knowledge Graph Embedding (CKGE) experimental framework.

This sub-package extends DICE Embeddings with a modular pipeline for studying
how a KGE model adapts to incrementally-arriving facts while preserving
previously learned predictions. It is built entirely on top of existing
``dicee`` primitives (models, datasets, evaluation utilities) -- no KGE model
is reimplemented here.

See ``dicee/continual/README.md`` for the full methodology description and
usage instructions.

Modules
-------
graph_changes
    Compute added triples and active/new entity & relation sets between two
    cumulative graph snapshots.
update_sequence
    Build reproducible continual-learning update sequences from a static
    train split (scenarios: ``existing_entities``, ``new_entities``, ``mixed``).
chebyshev
    Model-agnostic Chebyshev polynomial propagation over the entity graph.
influence
    Edit signals + Chebyshev propagation -> normalized entity influence
    scores and per-query preservation weights.
continual_losses
    Adaptation loss (reuses ``BaseKGE`` scoring/loss) and preservation
    (distillation) loss against a frozen snapshot of the previous model.
initialization
    Safe entity-embedding table expansion (preserve existing rows, add new
    ones) + optional neighbor-informed initialization.
continual_trainer
    Orchestrates the sequential update procedure and the three baselines
    (fine-tuning, uniform distillation, Chebyshev-weighted distillation).
evaluation
    Filtered link-prediction metrics (reusing ``dicee.evaluation``) plus
    preservation metrics (Delta-MRR on stable facts, score drift).
"""

from .graph_changes import GraphSnapshot, diff_snapshots
from .update_sequence import UpdateStep, generate_update_sequence
from .chebyshev import build_normalized_adjacency, chebyshev_propagate
from .influence import EditSignals, compute_edit_signals, compute_influence_scores
from .initialization import extend_entity_embeddings
from .continual_losses import adaptation_loss, preservation_loss, preservation_weights
from .continual_trainer import ContinualTrainer, ContinualConfig
from .evaluation import evaluate_snapshot, preservation_metrics

__all__ = [
    "GraphSnapshot",
    "diff_snapshots",
    "UpdateStep",
    "generate_update_sequence",
    "build_normalized_adjacency",
    "chebyshev_propagate",
    "EditSignals",
    "compute_edit_signals",
    "compute_influence_scores",
    "extend_entity_embeddings",
    "adaptation_loss",
    "preservation_loss",
    "preservation_weights",
    "ContinualTrainer",
    "ContinualConfig",
    "evaluate_snapshot",
    "preservation_metrics",
]
