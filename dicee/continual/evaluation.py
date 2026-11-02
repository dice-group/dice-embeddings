"""Evaluation for continual KGE experiments (Section 11).

Reuses DICE's existing filtered-ranking primitives
(``dicee.evaluation._filtering.FilteredRanker``) directly on integer entity
ids -- continual experiments operate purely on integer-indexed triples (no
string round-trip needed), so the string-keyed convenience wrappers in
``dicee.evaluation.link_prediction`` are not used here, but the exact same
filtered-ranking algorithm is.

Two kinds of metrics are produced:

* standard filtered link-prediction metrics (MRR, Hits@1/3/10) on a held-out
  triple set, via :func:`evaluate_snapshot`;
* preservation metrics between two model checkpoints on a shared "stable"
  query set, via :func:`preservation_metrics` (``Delta_MRR_stable`` and
  ``ScoreDrift``, Section 11).

Performance note (evaluation bottleneck fix)
---------------------------------------------
After vectorizing continual-step training (see
``dicee.continual.continual_losses.sample_filtered_negatives``), evaluation
became the dominant per-snapshot cost. The original implementation scored
every candidate triple by materializing a full ``(batch_size * num_entities,
3)`` index tensor and calling the generic, per-triple ``model.forward_triples``
-- this forces THREE independent embedding gathers (head, relation, tail),
each of shape ``(batch_size * num_entities, embedding_dim)``, i.e. the full
entity table duplicated/gathered ``batch_size`` times over, for every single
evaluation batch, on both the head and the tail side.

Tail-side scoring (fixed head+relation, vary tail) is now routed through
``model.forward_k_vs_all`` instead: every ``BaseKGE`` subclass used in the
continual experiments (DistMult, ComplEx, QMult, OMult, Keci -- see
``dicee.continual.experiments.run_experiment.MODEL_REGISTRY``) already
implements this method for KvsAll training, and it computes the IDENTICAL
score (``model.score(h, r, e)`` for every candidate entity ``e``) via a
single matmul against the already-materialized entity embedding matrix,
with no embedding gather/duplication at all. This is an exact, not
approximate, speedup: ``forward_k_vs_all`` is the same scoring function
DICE already uses to train/evaluate every model with the KvsAll technique.

Head-side scoring (fixed relation+tail, vary head) has no such official
"swap" method in the base framework (only a handful of graph-model classes
implement ``forward_k_vs_all_heads``; see ``dicee/models/graph_model.py``).
For DistMult specifically, the triple score ``sum(h * r * t)`` is fully
commutative/symmetric in ``h`` and ``t``, so the EXACT SAME
``k_vs_all_score`` matmul can be reused with head and tail swapped (this is
verified to reproduce ``forward_triples`` bit-for-bit under eval mode with
``normalization=None``, which is what every continual experiment config
uses -- see ``tests/continual/test_evaluation_fastpath.py``). This fast path
is applied ONLY when it is provably exact (model name is ``"DistMult"`` and
``normalization`` is unset); every other model falls back to the original,
slower-but-always-correct ``forward_triples``-based head-side computation,
since e.g. ComplEx's Hermitian score and QMult/OMult's non-commutative
Hamilton-product scores are NOT generally symmetric under a head/tail swap.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ..evaluation._filtering import FilteredRanker
from ..evaluation.utils import ALL_HITS_RANGE, compute_metrics_from_ranks_simple, create_hits_dict, update_hits
from ..models.base_model import BaseKGE


def _overrides_forward_k_vs_all(model) -> bool:
    """True iff ``model``'s class actually overrides ``forward_k_vs_all``.

    ``BaseKGE.forward_k_vs_all`` exists on every model (it just raises
    ``ValueError``), so a plain ``hasattr`` check always returns ``True``;
    we must compare the bound method's underlying function against the
    base class's to detect a genuine override.
    """
    return type(model).forward_k_vs_all is not BaseKGE.forward_k_vs_all

# Models whose triple score is exactly symmetric/commutative in (head, tail)
# -- i.e. score(h, r, t) == score(t, r, h) as a function of the embeddings
# -- so that the tail-side k_vs_all_score matmul can be reused verbatim for
# head-side scoring by swapping which entity is "fixed". Verified against
# DistMult.score: f(h,r,t) = sum(h * r * t), fully commutative. Do NOT add
# models here without re-deriving the formula: ComplEx/ConEx's Hermitian
# inner product and QMult/OMult's Hamilton-product-based scores are NOT
# symmetric in general.
_HEAD_TAIL_SYMMETRIC_MODELS = {"DistMult"}


def _normalization_is_identity(model) -> bool:
    """True iff the model's (head/relation/tail) normalizer is a no-op.

    The fast head-side swap path (and the tail-side ``forward_k_vs_all``
    path for the raw, un-normalized candidate table) are only exact when
    ``normalize_tail_entity_embeddings`` -- applied in ``forward_triples``'s
    ``get_triple_representation`` but NOT in ``k_vs_all_score``'s raw
    ``entity_embeddings.weight`` -- is itself a no-op. Every continual
    experiment config hardcodes ``normalization=None`` (see
    ``dicee.continual.experiments.run_experiment._make_model``), which maps
    to ``IdentityClass``; this check makes that assumption explicit and
    verifiable rather than silently relying on it.
    """
    args = getattr(model, "args", None)
    normalization = args.get("normalization") if isinstance(args, dict) else None
    return normalization in (None, "None")


@torch.no_grad()
def evaluate_snapshot(
    model,
    triples: np.ndarray,
    er_vocab: Dict[Tuple[int, int], List[int]],
    re_vocab: Dict[Tuple[int, int], List[int]],
    *,
    tie_policy: str = "sort",
    tie_seed: int = 0,
    batch_size: Optional[int] = None,
    max_candidate_rows: int = 2_000_000,
) -> Dict[str, float]:
    """Filtered head+tail link-prediction evaluation on integer-indexed triples.

    Parameters
    ----------
    model : torch.nn.Module
        A ``BaseKGE`` instance exposing ``forward_triples``.
    triples : np.ndarray
        ``(n, 3)`` int array of held-out ``(head, relation, tail)`` ids.
    er_vocab, re_vocab : dict
        ``(head, relation) -> [tails]`` / ``(relation, tail) -> [heads]``
        filtering vocabularies built from the *currently visible* graph
        (e.g. via ``dicee.static_preprocess_funcs.get_er_vocab``/``get_re_vocab``
        on the cumulative training triples at the current step, unioned
        with the evaluation split itself for standard filtered evaluation).
    batch_size : Optional[int]
        Number of queries scored per forward pass. Each query expands into
        ``num_entities`` candidate triples, so the actual tensor built per
        batch has ``batch_size * num_entities`` rows. If ``None``
        (default), it is derived from ``max_candidate_rows`` so the
        candidate tensor stays within a fixed memory budget regardless of
        vocabulary size.
    max_candidate_rows : int
        Soft cap on ``batch_size * num_entities`` used to auto-derive
        ``batch_size`` when not given explicitly. Lower this if you hit
        out-of-memory errors on very large vocabularies / GPUs with little
        memory; raise it on large-memory GPUs to further reduce the number
        of forward passes (and therefore wall-clock time).

    Returns
    -------
    Dict[str, float]
        ``{"MRR", "H@1", "H@3", "H@10", "runtime_seconds"}``.
    """
    start = time.time()
    model.eval()
    ranker = FilteredRanker(tie_policy, tie_seed)
    num_entities = model.num_entities
    device = next(model.parameters()).device

    if batch_size is None:
        batch_size = max(1, max_candidate_rows // max(1, num_entities))

    ranks: List[int] = []
    hits = create_hits_dict(ALL_HITS_RANGE)

    # Fast-path eligibility, decided once per call (not per batch).
    use_fast_tail = _overrides_forward_k_vs_all(model) and _normalization_is_identity(model)
    use_fast_head = (
        use_fast_tail
        and hasattr(model, "k_vs_all_score")
        and getattr(model, "name", None) in _HEAD_TAIL_SYMMETRIC_MODELS
    )

    all_e = torch.arange(num_entities, dtype=torch.long, device=device)
    n = len(triples)
    show_progress = n > batch_size

    for start_idx in tqdm(range(0, n, batch_size), desc="Evaluating", unit="batch", leave=False,
                           disable=not show_progress):
        chunk = triples[start_idx:start_idx + batch_size]
        b = len(chunk)
        h = torch.as_tensor(chunk[:, 0], dtype=torch.long, device=device)
        r = torch.as_tensor(chunk[:, 1], dtype=torch.long, device=device)
        t = torch.as_tensor(chunk[:, 2], dtype=torch.long, device=device)
        h_list, r_list, t_list = chunk[:, 0].tolist(), chunk[:, 1].tolist(), chunk[:, 2].tolist()

        # --- Tail-side: score every (h_i, r_i, *) against all candidate entities at once. ---
        if use_fast_tail:
            # Exact match to forward_triples under eval()+normalization=None:
            # this is literally the same scoring function DICE's own KvsAll
            # training already uses, computed via one matmul against the
            # entity table instead of a (b*num_entities)-row gather.
            tail_scores = model.forward_k_vs_all(torch.stack([h, r], dim=1))
        else:
            e_rep = all_e.repeat(b)
            tail_x = torch.stack([h.repeat_interleave(num_entities), r.repeat_interleave(num_entities), e_rep], dim=1)
            tail_scores = model.forward_triples(tail_x).view(b, num_entities)
        filt_tails = [er_vocab.get((hh, rr), []) for hh, rr in zip(h_list, r_list)]
        ranks_tail = ranker.rank_batch(tail_scores, t_list, filt_tails)

        # --- Head-side: score every (*, r_i, t_i) against all candidate entities at once. ---
        if use_fast_head:
            # Only reachable for models with a provably head/tail-symmetric
            # score (see _HEAD_TAIL_SYMMETRIC_MODELS): reuse the exact same
            # matmul-based k_vs_all_score with head and tail swapped.
            emb_t, emb_r = model.get_head_relation_representation(torch.stack([t, r], dim=1))
            head_scores = model.k_vs_all_score(emb_h=emb_t, emb_r=emb_r, emb_E=model.entity_embeddings.weight)
        else:
            e_rep = all_e.repeat(b)
            head_x = torch.stack([e_rep, r.repeat_interleave(num_entities), t.repeat_interleave(num_entities)], dim=1)
            head_scores = model.forward_triples(head_x).view(b, num_entities)
        filt_heads = [re_vocab.get((rr, tt), []) for rr, tt in zip(r_list, t_list)]
        ranks_head = ranker.rank_batch(head_scores, h_list, filt_heads)

        for rank_tail, rank_head in zip(ranks_tail, ranks_head):
            ranks.append(rank_tail)
            update_hits(hits, rank_tail, ALL_HITS_RANGE)
            ranks.append(rank_head)
            update_hits(hits, rank_head, ALL_HITS_RANGE)

    metrics = compute_metrics_from_ranks_simple(ranks, len(ranks), hits)
    metrics["runtime_seconds"] = time.time() - start
    return metrics



@torch.no_grad()
def preservation_metrics(
    model_new,
    model_old,
    stable_triples: np.ndarray,
    er_vocab: Dict[Tuple[int, int], List[int]],
    re_vocab: Dict[Tuple[int, int], List[int]],
) -> Dict[str, float]:
    """Preservation metrics on a shared stable-query set (Section 11).

    Returns
    -------
    Dict[str, float]
        ``MRR_new``, ``MRR_old``, ``Delta_MRR_stable`` (signed),
        ``abs_Delta_MRR_stable``, and ``ScoreDrift`` (mean absolute
        per-triple score difference between the two models).
    """
    if len(stable_triples) == 0:
        return {
            "MRR_new": float("nan"), "MRR_old": float("nan"),
            "Delta_MRR_stable": float("nan"), "abs_Delta_MRR_stable": float("nan"),
            "ScoreDrift": float("nan"), "num_stable_triples": 0,
        }

    metrics_new = evaluate_snapshot(model_new, stable_triples, er_vocab, re_vocab)
    metrics_old = evaluate_snapshot(model_old, stable_triples, er_vocab, re_vocab)

    model_new.eval()
    model_old.eval()
    device = next(model_new.parameters()).device
    x = torch.tensor(stable_triples, dtype=torch.long, device=device)
    f_new = model_new.forward_triples(x)
    f_old = model_old.forward_triples(x.to(next(model_old.parameters()).device)).to(device)
    score_drift = (f_new - f_old).abs().mean().item()

    delta = metrics_new["MRR"] - metrics_old["MRR"]
    return {
        "MRR_new": metrics_new["MRR"],
        "MRR_old": metrics_old["MRR"],
        "Delta_MRR_stable": delta,
        "abs_Delta_MRR_stable": abs(delta),
        "ScoreDrift": score_drift,
        "num_stable_triples": len(stable_triples),
    }
