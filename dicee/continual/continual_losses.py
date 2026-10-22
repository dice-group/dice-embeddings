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

from typing import Iterable, Mapping, Optional, Set, Tuple

import numpy as np
import torch


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
    known_positive_triples: Set[Tuple[int, int, int]],
    num_negatives: int,
    rng: np.random.Generator,
    max_tries: int = 10,
) -> np.ndarray:
    """Head/tail-corrupt ``pos_triples``, filtered against known positives.

    For every positive triple ``(h, r, t)`` in ``pos_triples``, draws
    ``num_negatives`` corrupted triples by replacing the head or the tail
    (chosen uniformly at random) with a uniformly-sampled entity id, and
    rejects candidates that are themselves in ``known_positive_triples``
    (re-drawing up to ``max_tries`` times). ``known_positive_triples``
    should be every triple known to be true at the *target* snapshot
    (``T_{t+1}``, i.e. ``step.cumulative_triples``), not just the newly
    added ones, so that previously-learned facts are never used as
    negative supervision.

    If ``max_tries`` is exhausted without finding a non-positive candidate
    (only plausible for very small/dense graphs), the last sampled
    candidate is kept as-is; this is a documented, rare edge case rather
    than a correctness guarantee.

    Returns
    -------
    np.ndarray
        ``(len(pos_triples) * num_negatives, 3)`` int64 array of negative
        triples, one block of ``num_negatives`` per input positive triple
        (same order as ``pos_triples``).
    """
    negs = np.empty((len(pos_triples) * num_negatives, 3), dtype=np.int64)
    idx = 0
    for row in pos_triples:
        h, r, t = int(row[0]), int(row[1]), int(row[2])
        for _ in range(num_negatives):
            candidate = (h, r, t)
            for _ in range(max_tries):
                corrupt_tail = bool(rng.random() < 0.5)
                sampled_id = int(rng.integers(0, num_entities))
                candidate = (h, r, sampled_id) if corrupt_tail else (sampled_id, r, t)
                if candidate not in known_positive_triples:
                    break
            negs[idx] = candidate
            idx += 1
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
    return (weights * diff_sq).mean()
