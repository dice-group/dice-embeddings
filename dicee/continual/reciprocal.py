"""Reciprocal-relation data augmentation for continual KGE experiments.

Both FastKGE and IncDE (the two published continual-KGE baselines this
framework is benchmarked against) train every triple ``(h, r, t)`` together
with its inverse ``(t, r + num_relations, h)`` using a doubled relation
vocabulary. This is a standard trick (used by ConvE, RotatE-style reciprocal
training, etc.) that gives the model an explicit, dedicated set of
parameters for "tail-style" prediction in the reverse direction instead of
relying solely on the (harder) raw head-prediction scoring path, and tends
to noticeably improve head-prediction metrics in particular.

This module only *augments training data*; it does not change evaluation.
Filtered test/validation triples keep their original (non-reciprocal)
relation ids, since `dicee.continual.evaluation.evaluate_snapshot` already
scores both the head and the tail side of every original relation directly.
"""

from __future__ import annotations

import numpy as np


def add_reciprocal_triples(triples: np.ndarray, num_relations: int) -> np.ndarray:
    """Return ``triples`` concatenated with their inverse ``(t, r+R, h)`` copies.

    Parameters
    ----------
    triples : np.ndarray
        ``(N, 3)`` int array of ``(head_idx, relation_idx, tail_idx)``.
    num_relations : int
        The *original* (pre-augmentation) number of relations ``R``; inverse
        relations are assigned ids ``r + R`` for ``r`` in ``[0, R)``, so the
        effective relation vocabulary size after augmentation is ``2*R``.

    Returns
    -------
    np.ndarray
        ``(2N, 3)`` array: the original triples followed by their inverses.
    """
    triples = np.asarray(triples, dtype=np.int64)
    if len(triples) == 0:
        return triples.reshape(0, 3)
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    inverse = np.stack([t, r + num_relations, h], axis=1)
    return np.concatenate([triples, inverse], axis=0)


def effective_num_relations(num_relations: int, use_reciprocal: bool) -> int:
    """Relation-vocabulary size a model should be built with."""
    return num_relations * 2 if use_reciprocal else num_relations
