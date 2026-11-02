"""Continual-learning training objectives.

Four objectives are implemented here:

``adaptation_loss``
    Delegates directly to the model's own ``loss_function`` (configured in
    ``BaseKGE.__init__`` from ``scoring_technique``/``loss_fn``). Used only
    for the *initial* ``T_0`` training (``ContinualTrainer.train_initial``),
    which is a standard, non-continual training run.

``sample_filtered_negatives`` / ``margin_adaptation_loss``
    The "Learning added facts" objective:

        L_add = sum_{q in DeltaT+} [gamma - f(q)]_+
              + sum_{q- in N_new}  [f(q-) + gamma]_+

    Negatives for the newly added triples are sampled by head/tail
    corruption and *filtered* against the known positive triples at
    ``T_{t+1}`` (``known_positive_triples``) so that true-but-unseen facts
    are never accidentally used as negative supervision. This is what
    actually teaches the model about new facts (including facts involving
    brand-new entities) during an update step; it replaces plain KvsAll
    training of ``DeltaT_i`` for that purpose, because a KvsAll multi-hot
    target built only from ``DeltaT_i`` silently mislabels any
    previously-known tail for the same ``(h, r)`` pair as a negative
    (it is absent from that step's restricted target vector), which is
    especially damaging in the ``new_entities``/``mixed`` scenarios where
    a new entity's only positive facts are a handful of triples spread
    across possibly multiple ``(h, r)`` keys.

``preservation_loss``
    A stable-query distillation loss against a frozen snapshot of the
    previous model:

        L_pres = sum_q w(q) * (f_new(q) - f_old(q))**2

    computed in minibatches (never materializing scores for the whole
    graph at once), with ``f_old`` evaluated under ``torch.no_grad()`` so
    gradients never flow into the frozen model.

Note on ``L_del``: the continual-learning protocol implemented in this
package (:mod:`dicee.continual.update_sequence`) is strictly cumulative --
``T_{i-1} subset T_i`` -- so there are no explicitly-invalidated facts to
unlearn (``num_removed_entities`` is always 0, see ``UpdateStep``). The
invalidation loss from the design document is therefore intentionally not
implemented; it would require a non-cumulative update protocol (facts
explicitly marked false, as opposed to merely absent) to be meaningful.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Set, Tuple, Union

import numpy as np
import torch


def encode_triples_as_keys(triples: np.ndarray, num_entities: int, num_relations: int) -> np.ndarray:
    """Pack ``(h, r, t)`` triples into unique sorted ``int64`` keys.

    ``key = h * (num_relations * num_entities) + r * num_entities + t`` is a
    bijection from valid ``(h, r, t)`` id-triples (``0 <= h, t < num_entities``,
    ``0 <= r < num_relations``) to a dense range of ``int64`` values, so
    set membership ("is this triple a known positive?") can be checked with
    a single vectorized ``np.searchsorted`` against a *sorted* key array
    instead of a Python-level ``tuple in set`` loop per candidate -- the
    membership check previously dominated :func:`sample_filtered_negatives`'
    wall-clock time at FB_CKGE/WN_CKGE scale (tens of thousands of added
    triples x ``num_negatives``, repeated every minibatch of every epoch).

    Returns
    -------
    np.ndarray
        Sorted, de-duplicated ``int64`` array of keys, ready to pass as
        ``known_positive_triples`` to :func:`sample_filtered_negatives`.
    """
    if len(triples) == 0:
        return np.empty((0,), dtype=np.int64)
    h = triples[:, 0].astype(np.int64)
    r = triples[:, 1].astype(np.int64)
    t = triples[:, 2].astype(np.int64)
    keys = h * (np.int64(num_relations) * np.int64(num_entities)) + r * np.int64(num_entities) + t
    return np.unique(keys)


def _keys_contains(keys_sorted: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    """Vectorized ``query_keys[i] in keys_sorted`` for every ``i`` (keys_sorted must be sorted)."""
    if len(keys_sorted) == 0:
        return np.zeros(len(query_keys), dtype=bool)
    idx = np.searchsorted(keys_sorted, query_keys)
    idx_clamped = np.clip(idx, 0, len(keys_sorted) - 1)
    return keys_sorted[idx_clamped] == query_keys


def adaptation_loss(model, yhat_batch: torch.Tensor, y_batch: torch.Tensor,
                     current_epoch: int = 0) -> torch.Tensor:
    """New-fact adaptation loss: reuses ``model.loss_function`` as-is.

    ``yhat_batch``/``y_batch`` must already be in whatever layout the
    model's configured scoring technique expects (e.g. KvsAll multi-hot
    targets, or NegSample binary targets) -- construct them with the same
    dataset classes / vocabularies DICE already uses
    (``dicee.static_preprocess_funcs.get_er_vocab`` etc.), not bespoke code.
    """
    return model.loss_function(yhat_batch, y_batch, current_epoch=current_epoch)


def sample_filtered_negatives(
    pos_triples: np.ndarray,
    num_entities: int,
    known_positive_triples: Union[Set[Tuple[int, int, int]], np.ndarray],
    num_negatives: int,
    rng: np.random.Generator,
    max_tries: int = 10,
    num_relations: Optional[int] = None,
) -> np.ndarray:
   
    n = len(pos_triples)
    total = n * num_negatives
    if total == 0:
        return np.empty((0, 3), dtype=np.int64)

    if isinstance(known_positive_triples, np.ndarray):
        known_keys = known_positive_triples
        if num_relations is None:
            num_relations = int(pos_triples[:, 1].max()) + 1 if n else 1
    else:
        # Legacy / test path: a plain Set[Tuple[int, int, int]].
        if num_relations is None:
            rels = [r for _, r, _ in known_positive_triples] + pos_triples[:, 1].tolist()
            num_relations = (max(rels) + 1) if rels else 1
        if known_positive_triples:
            known_arr = np.array(list(known_positive_triples), dtype=np.int64)
            known_keys = encode_triples_as_keys(known_arr, num_entities, num_relations)
        else:
            known_keys = np.empty((0,), dtype=np.int64)

    # Repeat each positive triple num_negatives times (block layout
    # identical to the original: block of `num_negatives` per input row).
    h0 = np.repeat(pos_triples[:, 0].astype(np.int64), num_negatives)
    r0 = np.repeat(pos_triples[:, 1].astype(np.int64), num_negatives)
    t0 = np.repeat(pos_triples[:, 2].astype(np.int64), num_negatives)

    cand_h = h0.copy()
    cand_t = t0.copy()
    pending = np.ones(total, dtype=bool)  # True while still unresolved (not yet accepted)

    mult = np.int64(num_relations) * np.int64(num_entities)
    for _try in range(max_tries):
        idx = np.nonzero(pending)[0]
        if len(idx) == 0:
            break
        corrupt_tail = rng.random(len(idx)) < 0.5
        sampled_ids = rng.integers(0, num_entities, size=len(idx))
        new_h = np.where(corrupt_tail, h0[idx], sampled_ids)
        new_t = np.where(corrupt_tail, sampled_ids, t0[idx])
        cand_h[idx] = new_h
        cand_t[idx] = new_t
        # Fully vectorized membership check (no per-candidate Python loop):
        # pack (h, r, t) into the same int64 key encoding as known_keys and
        # look each one up with a single np.searchsorted call.
        query_keys = new_h.astype(np.int64) * mult + r0[idx] * np.int64(num_entities) + new_t.astype(np.int64)
        is_known = _keys_contains(known_keys, query_keys)
        pending[idx] = is_known
        if not pending.any():
            break

    negs = np.empty((total, 3), dtype=np.int64)
    negs[:, 0] = cand_h
    negs[:, 1] = r0
    negs[:, 2] = cand_t
    return negs


def margin_adaptation_loss(
    model,
    pos_triples: torch.LongTensor,
    neg_triples: torch.LongTensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """``L_add`` -- margin loss for newly added facts.

        L_add = mean_{q in pos} [margin - f(q)]_+
              + mean_{q- in neg} [f(q-) + margin]_+

    ``pos_triples``/``neg_triples`` are ``(n, 3)`` long tensors of
    ``(head, relation, tail)`` ids; scored with ``model.forward_triples``
    (the generic single-triple scoring path every ``BaseKGE`` model
    implements, independent of the scoring technique the model happens to
    be configured for).

    Deviation from the literal paper formula: terms are mean-reduced
    (rather than summed) so the loss scale is invariant to batch size /
    the number of negatives per positive, which keeps ``margin`` and any
    outer loss weights (e.g. ``lambda_pres``) meaningful across update
    steps with very different ``len(DeltaT_i)``.
    """
    if len(pos_triples) > 0:
        f_pos = model.forward_triples(pos_triples)
        pos_term = torch.clamp(margin - f_pos, min=0.0).mean()
    else:
        pos_term = torch.tensor(0.0, device=neg_triples.device if len(neg_triples) else None)

    if len(neg_triples) > 0:
        f_neg = model.forward_triples(neg_triples)
        neg_term = torch.clamp(f_neg + margin, min=0.0).mean()
    else:
        neg_term = torch.tensor(0.0, device=pos_triples.device if len(pos_triples) else None)

    return pos_term + neg_term


def preservation_weights(
    heads: torch.LongTensor,
    tails: torch.LongTensor,
    influence_hat: Mapping[int, float],
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute ``w(q) = exp(-beta * max(I_hat(h), I_hat(t)))`` for a batch of queries.

    Parameters
    ----------
    heads, tails : torch.LongTensor
        ``(batch,)`` entity ids for the stable-query batch.
    influence_hat : Mapping[int, float]
        Normalized influence scores, as returned by
        :func:`dicee.continual.influence.compute_influence_scores`. Entities
        not present in the mapping default to influence ``0.0`` (maximally
        preserved), which is the conservative choice for brand-new
        entities that were never touched by the propagation graph used to
        build ``influence_hat``.
    beta : float
        Preservation strength: larger beta => high-influence entities are
        preserved much more weakly than low-influence ones.
    device : torch.device
        Device for the returned tensor.
    """
    h_inf = torch.tensor(
        [influence_hat.get(int(h), 0.0) for h in heads.tolist()], dtype=torch.float32, device=device
    )
    t_inf = torch.tensor(
        [influence_hat.get(int(t), 0.0) for t in tails.tolist()], dtype=torch.float32, device=device
    )
    max_inf = torch.maximum(h_inf, t_inf)
    return torch.exp(-beta * max_inf)


def preservation_loss(
    model_new,
    model_old,
    heads: torch.LongTensor,
    relations: torch.LongTensor,
    tails: torch.LongTensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted-squared-score-difference distillation loss for one minibatch.

        L_pres = sum_q w(q) * (f_new(q) - f_old(q))**2  (mean-reduced here)

    ``model_old`` is evaluated under ``torch.no_grad()``; no gradients ever
    flow into it (it is a frozen snapshot of the model *before* the current
    update step).
    """
    x = torch.stack([heads, relations, tails], dim=1)
    f_new = model_new.forward_triples(x)

    with torch.no_grad():
        was_training = model_old.training
        model_old.eval()
        f_old = model_old.forward_triples(x).detach()
        if was_training:
            model_old.train()

    diff_sq = (f_new - f_old) ** 2
    return (weights * diff_sq).sum() #(For WN_18, mean reduction is much better than sum reduction.)
