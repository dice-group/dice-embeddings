"""Entity influence estimation via Chebyshev-propagated edit signals.

Pipeline (Section 7 of the research spec):

1. Build the propagation graph at update ``i`` as the union of entity
   connections in the previous (``T_{i-1}``) and current (``T_i``)
   snapshots -- i.e. simply the normalized adjacency of ``T_i`` itself,
   since ``T_i`` already contains ``T_{i-1}`` under the cumulative
   protocol used by :mod:`dicee.continual.update_sequence`.
2. Build two initial edit-signal channels per entity:
   ``added_count(v)`` and ``new_entity_count(v)`` (optionally a third,
   ``is_new_entity(v)``).
3. Propagate the signal with :func:`dicee.continual.chebyshev.chebyshev_propagate`.
4. Combine propagated signals into a single scalar influence score per
   entity and min-max-style normalize it.

The resulting score ``I_hat(v)`` is a **structural influence heuristic**:
it reflects how topologically close an entity is to the newly edited
region of the graph. It is **not** a probability and **not** a prediction
of the exact embedding update magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from .chebyshev import build_normalized_adjacency, chebyshev_propagate

Triple = tuple


@dataclass
class EditSignals:
    """Initial per-entity edit-signal channels for one update step.

    Attributes
    ----------
    added_count : np.ndarray
        ``(num_entities,)`` -- number of newly added triples incident to
        each entity.
    new_entity_count : np.ndarray
        ``(num_entities,)`` -- number of newly added triples incident to
        each entity that involve an entity introduced at this update.
    is_new_entity : np.ndarray
        ``(num_entities,)`` -- 1.0 if the entity itself was introduced at
        this update, else 0.0.
    """

    added_count: np.ndarray
    new_entity_count: np.ndarray
    is_new_entity: np.ndarray

    def as_matrix(self) -> np.ndarray:
        """Stack the three channels into an ``(n, 3)`` signal matrix."""
        return np.stack([self.added_count, self.new_entity_count, self.is_new_entity], axis=1)


def compute_edit_signals(
    added_triples: np.ndarray,
    new_entity_ids: Iterable[int],
    num_entities: int,
) -> EditSignals:
    """Build the initial edit-signal channels ``S_0`` for one update step.

    Parameters
    ----------
    added_triples : np.ndarray
        ``(n, 3)`` array of ``DeltaT_i`` (the triples added at this step).
    new_entity_ids : Iterable[int]
        Entity ids introduced for the first time at this update step.
    num_entities : int
        Global entity vocabulary size (signal vectors are indexed by the
        stable global entity id so that scores line up across steps).
    """
    added_count = np.zeros(num_entities, dtype=np.float64)
    new_entity_count = np.zeros(num_entities, dtype=np.float64)
    is_new_entity = np.zeros(num_entities, dtype=np.float64)

    new_ids = set(int(e) for e in new_entity_ids)
    for e in new_ids:
        if 0 <= e < num_entities:
            is_new_entity[e] = 1.0

    for h, _r, t in added_triples:
        h, t = int(h), int(t)
        added_count[h] += 1
        added_count[t] += 1
        involves_new = (h in new_ids) or (t in new_ids)
        if involves_new:
            new_entity_count[h] += 1
            new_entity_count[t] += 1

    return EditSignals(added_count=added_count, new_entity_count=new_entity_count, is_new_entity=is_new_entity)


def compute_influence_scores(
    cumulative_triples: np.ndarray,
    added_triples: np.ndarray,
    new_entity_ids: Iterable[int],
    num_entities: int,
    *,
    K: int = 3,
    rho: float = 0.9,
    epsilon: float = 1e-8,
) -> Dict[int, float]:
    """Compute normalized structural influence scores ``I_hat(v)``.

    Parameters
    ----------
    cumulative_triples : np.ndarray
        ``T_i`` (current cumulative snapshot). Used to build the
        propagation graph; since ``T_i`` is a superset of ``T_{i-1}`` under
        the cumulative protocol, this already equals "the union of entity
        connections in the previous and updated snapshots".
    added_triples : np.ndarray
        ``DeltaT_i``, used to build the initial edit signal.
    new_entity_ids : Iterable[int]
        Entities introduced at this update step.
    num_entities : int
        Global entity vocabulary size.
    K : int
        Highest Chebyshev polynomial order (``K >= 0``).
    rho : float
        Decay rate for combining polynomial orders.
    epsilon : float
        Numerical-stability constant in the normalization denominator.

    Returns
    -------
    Dict[int, float]
        Mapping ``entity_id -> I_hat(entity_id)`` for every entity id in
        ``[0, num_entities)``.
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    l_tilde = build_normalized_adjacency(cumulative_triples, num_entities)
    signals = compute_edit_signals(added_triples, new_entity_ids, num_entities)
    s0 = signals.as_matrix()

    z_list = chebyshev_propagate(l_tilde, s0, K)

    denom = sum(rho ** j for j in range(K + 1))
    alphas = [(rho ** k) / denom for k in range(K + 1)]

    influence = np.zeros(num_entities, dtype=np.float64)
    for k, (alpha, z_k) in enumerate(zip(alphas, z_list)):
        z_k_np = z_k.numpy() if hasattr(z_k, "numpy") else np.asarray(z_k)
        influence += abs(alpha) * np.linalg.norm(z_k_np, ord=2, axis=1)

    max_influence = influence.max() if len(influence) else 0.0
    influence_hat = influence / (max_influence + epsilon)

    return {int(v): float(influence_hat[v]) for v in range(num_entities)}
