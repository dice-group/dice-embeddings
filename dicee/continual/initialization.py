"""Safe entity-embedding table expansion for new-entity acquisition.

DICE's ``BaseKGE`` models store entity embeddings in a plain
``torch.nn.Embedding(num_entities, dim)`` (see
``dicee/models/base_model.py:init_entity_embeddings``). Calling
``init_entity_embeddings`` again with a larger ``num_entities`` would
**reinitialize every row from scratch**, silently discarding everything the
model has learned so far -- unsafe for continual learning.

:func:`extend_entity_embeddings` instead creates a new, larger
``nn.Embedding`` that:

* copies all existing rows verbatim (identity preserved);
* initializes only the newly appended rows (uniform/model default, or an
  optional neighbor-informed average);
* is swapped into the model in place.

Optimizer-state note (Section 10)
----------------------------------
PyTorch optimizers (Adam, SGD, ...) key their per-parameter state
(``exp_avg``, ``exp_avg_sq``, momentum buffers, ...) by the **parameter
object's identity** (``id(tensor)``), not by name. Replacing
``model.entity_embeddings`` with a brand-new ``nn.Embedding`` therefore
detaches the old parameter from the optimizer: any accumulated moment
estimates for the *existing* rows are lost unless explicitly migrated.

This module implements the smallest safe extension that preserves
*parameter values* (the embeddings themselves) but does **not** attempt to
migrate per-row optimizer state, because partial-row optimizer-state
migration is optimizer-implementation-specific and out of scope for this
milestone (Section 15). The accompanying tests
(``tests/continual/test_initialization.py``) explicitly verify that after
extension (a) old embedding rows are bit-identical, (b) the optimizer must
be rebuilt (``model.parameters()`` is re-collected) for gradients to reach
the new rows, and (c) training after extension still updates both old and
new rows.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def extend_entity_embeddings(
    model: torch.nn.Module,
    new_num_entities: int,
    *,
    neighbor_informed_init: bool = False,
    new_entity_neighbors: Optional[dict] = None,
) -> torch.nn.Embedding:
    """Grow ``model.entity_embeddings`` to ``new_num_entities`` rows in place.

    Parameters
    ----------
    model : torch.nn.Module
        A ``BaseKGE`` instance (or any module exposing ``entity_embeddings``
        as an ``nn.Embedding`` and ``num_entities``/``embedding_dim``).
    new_num_entities : int
        Target vocabulary size. Must be ``>= model.num_entities``.
    neighbor_informed_init : bool
        If True, initialize each new row as the mean embedding of its
        known neighbors (via ``new_entity_neighbors``) instead of the
        model's default initializer. This is a configurable baseline, not
        a theoretically-justified default for every KGE architecture
        (e.g. it is a reasonable heuristic for translational/bilinear
        models but not necessarily for Clifford-algebra models).
    new_entity_neighbors : Optional[dict]
        Mapping ``new_entity_id -> list[known_entity_id]``, required when
        ``neighbor_informed_init=True``.

    Returns
    -------
    torch.nn.Embedding
        The new embedding table (also assigned to
        ``model.entity_embeddings``).
    """
    old_embedding: torch.nn.Embedding = model.entity_embeddings
    old_num_entities, dim = old_embedding.weight.shape

    if new_num_entities < old_num_entities:
        raise ValueError(
            f"new_num_entities ({new_num_entities}) must be >= current "
            f"num_entities ({old_num_entities}); shrinking is not supported."
        )
    if new_num_entities == old_num_entities:
        logger.info("extend_entity_embeddings: no new entities, returning table unchanged.")
        return old_embedding

    new_embedding = torch.nn.Embedding(new_num_entities, dim)
    param_init = getattr(model, "param_init", None)
    if callable(param_init):
        param_init(new_embedding.weight.data)
    else:
        torch.nn.init.xavier_normal_(new_embedding.weight.data)

    with torch.no_grad():
        # Preserve existing rows exactly (identity of learned parameters).
        new_embedding.weight.data[:old_num_entities] = old_embedding.weight.data.clone()

        if neighbor_informed_init:
            if new_entity_neighbors is None:
                raise ValueError("new_entity_neighbors is required when neighbor_informed_init=True")
            for new_id in range(old_num_entities, new_num_entities):
                neighbors = new_entity_neighbors.get(new_id, [])
                known_neighbors = [n for n in neighbors if n < old_num_entities]
                if known_neighbors:
                    idx = torch.tensor(known_neighbors, dtype=torch.long)
                    new_embedding.weight.data[new_id] = old_embedding.weight.data[idx].mean(dim=0)
                # else: keep the default (param_init) initialization already set above.

    model.entity_embeddings = new_embedding
    model.num_entities = new_num_entities
    if isinstance(model.args, dict):
        model.args["num_entities"] = new_num_entities

    logger.info(
        "Extended entity_embeddings from %s to %s rows (dim=%s, neighbor_informed_init=%s).",
        old_num_entities, new_num_entities, dim, neighbor_informed_init,
    )
    return new_embedding
