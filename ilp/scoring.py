"""Core scoring primitives for InductiveKGModel (shared by eval and Scorer).

These functions are the *foundation* both the metrics layer (`eval.py`) and the
ergonomic facade (`scorer.py`) build on, so they live here rather than under
either — keeping the dependency graph a clean DAG (`eval → scoring ← scorer`).

The two scoring entry points share one trick: the anchor's subgraph is encoded
ONCE per query (`pooled` is candidate-independent), then only the candidate
representation varies. Single-tower (`score_candidates`) varies a token; dual
(`score_candidates_dual`) varies a precomputed per-entity tower vector.

Anonymization note: an entity's encoded vector is only stable within a single
[Z_*] draw. For `z_mode="runtime"` the caller (eval) redraws and averages across
MC passes, so any candidate table is valid only for the draw it was built with.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch
from tqdm.auto import tqdm

from .dataset import (
    KnowledgeGraph,
    Triple,
    _build_subgraph_rows,
    _pad_subgraph,
)
from .model import InductiveKGModel


def _tokenize_center(
    center: str,
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    max_triples: int,
    z_pool: int,
    device: torch.device,
    *,
    exclude_triple: Triple | None = None,
    collapse_z: bool = False,
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
    center_mode: str = "xtoken",
    inherit_map: dict[str, int] | None = None,
    inherit_z: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], list[int]]:
    """Tokenize `center`'s anonymized k-hop subgraph into batched tensors.

    The single tokenizer: delegates to the *same* `_build_subgraph_rows` the
    training path uses, then pads and adds a batch dim. Returns
    `(triples [1,N,3], hop [1,N,3], mask [1,N], entity_map, z_remaining)` so the
    caller can both encode the subgraph and assign candidate tokens consistently
    with the anonymization. Replaces the assign/BFS blocks formerly copy-pasted
    into `score_candidates` and `_encode_center`.

    `center_mode` must match training (`xtoken` vs `cls_role`). `inherit_map`/
    `inherit_z` (shared anonymization) seed labels from another tower's subgraph.
    """
    triples_tok, hop_tok, _tok, entity_map, z_remaining, none_id = _build_subgraph_rows(
        center, kg, vocab, fixed_values, max_triples, z_pool, random.Random(),
        exclude_triple, collapse_z, subgraph_hops, use_hop_distance_tokens,
        center_mode=center_mode, inherit_map=inherit_map, inherit_z=inherit_z,
    )
    triples_t, hop_t, mask_t = _pad_subgraph(triples_tok, hop_tok, max_triples, none_id)
    return (
        triples_t.unsqueeze(0).to(device),
        hop_t.unsqueeze(0).to(device),
        mask_t.unsqueeze(0).to(device),
        entity_map,
        z_remaining,
    )


@torch.no_grad()
def score_candidates(
    model: InductiveKGModel,
    anchor: str,
    relation: str,
    candidates: Sequence[str],
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    max_triples: int,
    z_pool: int,
    batch_size: int,                  # kept for backwards compat (unused)
    device: torch.device,
    exclude_triple: Triple | None = None,
    collapse_z: bool = False,
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
    center_mode: str = "xtoken",
) -> dict[str, float]:
    """Score every candidate under a single (anchor, relation) query.

    Optimization: `pooled` depends only on the subgraph, not on the candidate.
    So we encode the subgraph ONCE, then vary only the candidate token. This
    replaces the previous per-candidate full forward pass — roughly a 100×
    to 1000× speedup on DBpedia50-scale eval.

    Z-randomization at inference is a single fixed draw per query (versus a
    fresh draw per candidate). That's actually more correct: candidates are
    compared on identical encoded context, not on different anonymizations.
    """
    # 1. Tokenize the anonymized subgraph for this query (candidate-independent).
    triples_t, hop_t, mask_t, entity_map, z_remaining = _tokenize_center(
        anchor, kg, vocab, fixed_values, max_triples, z_pool, device,
        exclude_triple=exclude_triple, collapse_z=collapse_z,
        subgraph_hops=subgraph_hops, use_hop_distance_tokens=use_hop_distance_tokens,
        center_mode=center_mode,
    )

    # 2. Encode subgraph once → pooled vector [1, d]. One shared random
    # anonymization per query (runtime), so subgraph and candidates see the same
    # [Z_*] vectors; _entity_embed handles the override. Runtime MC averaging is
    # done one level up in evaluate_direction (a fresh full draw per pass).
    rand_bank = (
        model._make_rand_bank(device=device)
        if getattr(model, "z_mode", "learned") == "runtime" else None
    )
    pooled = model._encode_subgraph(
        triples_t, mask_t, hop_t if use_hop_distance_tokens else None, rand_bank
    )  # [1, d]

    # 3. Score all candidates in vectorized chunks.
    # Candidate token: use the subgraph's assignment if present, else [VAL_*] for
    # schema, else any available [Z_*]. All [Z_*] embeddings are interchangeable
    # by Z-randomization training — unseen candidates legitimately tie.
    unused_z_id = (
        vocab[f"[Z_{z_remaining[-1]}]"] if z_remaining else vocab["[Z_0]"]
    )
    rel_id = vocab[f"[REL_{relation}]"]

    cand_tokens: list[int] = []
    for c in candidates:
        if c in entity_map:
            cand_tokens.append(entity_map[c])
        elif c in fixed_values:
            cand_tokens.append(vocab[f"[VAL_{c}]"])
        else:
            cand_tokens.append(unused_z_id)

    cand_t = torch.tensor(cand_tokens, dtype=torch.long, device=device)
    rel_t = torch.full((len(candidates),), rel_id, dtype=torch.long, device=device)

    # Process candidates in batches to keep memory bounded.
    chunk = 4096
    out: list[float] = []
    pooled_d = pooled.squeeze(0)  # [d]
    for i in range(0, len(candidates), chunk):
        ct = cand_t[i : i + chunk]
        rt = rel_t[i : i + chunk]
        tr = model.embed(rt)                          # [c, d]
        tt = model._entity_embed(ct, rand_bank)       # [c, d]
        target_emb = tr + tt
        pooled_b = pooled_d.unsqueeze(0).expand(target_emb.size(0), -1)
        feats = torch.cat([pooled_b, target_emb, pooled_b * target_emb], dim=-1)
        logits = model.classifier(feats).squeeze(-1)
        out.extend(logits.float().cpu().tolist())

    return dict(zip(candidates, out))


@torch.no_grad()
def _encode_center(
    model: InductiveKGModel,
    center: str,
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    max_triples: int,
    z_pool: int,
    device: torch.device,
    *,
    exclude_triple: Triple | None = None,
    collapse_z: bool = False,
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
    rand_bank: torch.Tensor | None = None,
    center_mode: str = "xtoken",
    inherit_map: dict[str, int] | None = None,
    inherit_z: list[int] | None = None,
) -> torch.Tensor:                    # [d]
    """Encode `center`'s anonymized k-hop subgraph into a pooled vector.

    Shared by the candidate-table builder and the anchor side of the dual
    scorer. Uses the same `_tokenize_center` tokenizer as `score_candidates`,
    but returns only the pooled [X] representation (no candidate scoring).
    `inherit_map`/`inherit_z` seed shared anonymization from the anchor tower.
    """
    triples_t, hop_t, mask_t, _entity_map, _z_remaining = _tokenize_center(
        center, kg, vocab, fixed_values, max_triples, z_pool, device,
        exclude_triple=exclude_triple, collapse_z=collapse_z,
        subgraph_hops=subgraph_hops, use_hop_distance_tokens=use_hop_distance_tokens,
        center_mode=center_mode, inherit_map=inherit_map, inherit_z=inherit_z,
    )
    return model._encode_subgraph(
        triples_t, mask_t, hop_t if use_hop_distance_tokens else None, rand_bank
    ).squeeze(0)


@torch.no_grad()
def build_candidate_table(
    model: InductiveKGModel,
    kg: KnowledgeGraph,
    entity_pool: Sequence[str],
    vocab: dict[str, int],
    fixed_values: set[str],
    max_triples: int,
    z_pool: int,
    device: torch.device,
    *,
    collapse_z: bool = False,
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
    center_mode: str = "xtoken",
) -> tuple[dict[str, int], torch.Tensor]:
    """Precompute each entity's candidate-tower vector ONCE → (index, table).

    This is what keeps dual-anchored eval fast: the candidate tower depends only
    on the entity's own subgraph (not on the anchor or relation), so we encode
    every entity once (O(|E|) tower passes) and reuse the table across all
    queries. Built over `kg` as-is; the per-query held-out triple is excluded
    only for the gold candidate inside `score_candidates_dual`.
    """
    rand_bank = (
        model._make_rand_bank(device=device)
        if getattr(model, "z_mode", "learned") == "runtime" else None
    )
    index: dict[str, int] = {}
    rows: list[torch.Tensor] = []
    for e in tqdm(entity_pool, desc="cand-table", unit="ent", dynamic_ncols=True):
        index[e] = len(rows)
        rows.append(_encode_center(
            model, e, kg, vocab, fixed_values, max_triples, z_pool, device,
            exclude_triple=None, collapse_z=collapse_z, subgraph_hops=subgraph_hops,
            use_hop_distance_tokens=use_hop_distance_tokens, rand_bank=rand_bank,
            center_mode=center_mode,
        ))
    table = torch.stack(rows, dim=0) if rows else torch.empty(0, model.d_model, device=device)
    return index, table


@torch.no_grad()
def score_candidates_dual(
    model: InductiveKGModel,
    anchor: str,
    relation: str,
    candidates: Sequence[str],
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    max_triples: int,
    z_pool: int,
    device: torch.device,
    *,
    cand_index: dict[str, int] | None,
    cand_table: torch.Tensor | None,
    true_cand: str | None = None,
    exclude_triple: Triple | None = None,
    collapse_z: bool = False,
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
    shared_anonymization: bool = False,
    center_mode: str = "xtoken",
) -> dict[str, float]:
    """Dual-anchored scoring with a precomputed candidate table.

    Encodes the anchor subgraph once (held-out triple excluded), then represents
    each candidate by its precomputed tower vector. The gold `true_cand` is
    re-encoded WITH `exclude_triple` so the held-out edge can't leak into its own
    subgraph; candidates missing from the table are encoded on the fly.

    `shared_anonymization`: the candidate labeling is seeded from the anchor's,
    so candidate vectors are anchor-dependent and the precomputed `cand_table`
    can't be reused — every candidate subgraph is re-encoded per query.
    """
    rand_bank = (
        model._make_rand_bank(device=device)
        if getattr(model, "z_mode", "learned") == "runtime" else None
    )

    if shared_anonymization:
        # Anchor labeling must be available to seed each candidate, so tokenize
        # explicitly (cand_table is invalid under shared anonymization).
        a_tr, a_hop, a_mask, a_map, a_zrem = _tokenize_center(
            anchor, kg, vocab, fixed_values, max_triples, z_pool, device,
            exclude_triple=exclude_triple, collapse_z=collapse_z,
            subgraph_hops=subgraph_hops, use_hop_distance_tokens=use_hop_distance_tokens,
            center_mode=center_mode,
        )
        pooled = model._encode_subgraph(
            a_tr, a_mask, a_hop if use_hop_distance_tokens else None, rand_bank
        ).squeeze(0)
        C = len(candidates)
        cand_mat = torch.empty(C, model.d_model, device=device)
        for i, c in enumerate(candidates):
            cand_mat[i] = _encode_center(
                model, c, kg, vocab, fixed_values, max_triples, z_pool, device,
                exclude_triple=exclude_triple if c == true_cand else None,
                collapse_z=collapse_z, subgraph_hops=subgraph_hops,
                use_hop_distance_tokens=use_hop_distance_tokens, rand_bank=rand_bank,
                center_mode=center_mode, inherit_map=a_map, inherit_z=a_zrem,
            )
        return _score_dual_feats(model, pooled, cand_mat, candidates, relation, vocab, device)

    pooled = _encode_center(
        model, anchor, kg, vocab, fixed_values, max_triples, z_pool, device,
        exclude_triple=exclude_triple, collapse_z=collapse_z, subgraph_hops=subgraph_hops,
        use_hop_distance_tokens=use_hop_distance_tokens, rand_bank=rand_bank,
        center_mode=center_mode,
    )  # [d]

    C = len(candidates)
    cand_mat = torch.empty(C, model.d_model, device=device)
    hit_pos: list[int] = []
    hit_rows: list[int] = []
    for i, c in enumerate(candidates):
        if c != true_cand and c in cand_index:
            hit_pos.append(i)
            hit_rows.append(cand_index[c])
        else:
            # Gold (exclude the held-out edge) or an entity absent from the table.
            cand_mat[i] = _encode_center(
                model, c, kg, vocab, fixed_values, max_triples, z_pool, device,
                exclude_triple=exclude_triple if c == true_cand else None,
                collapse_z=collapse_z, subgraph_hops=subgraph_hops,
                use_hop_distance_tokens=use_hop_distance_tokens, rand_bank=rand_bank,
                center_mode=center_mode,
            )
    if hit_pos:
        cand_mat[torch.tensor(hit_pos, device=device)] = cand_table[
            torch.tensor(hit_rows, device=device)
        ]

    return _score_dual_feats(model, pooled, cand_mat, candidates, relation, vocab, device)


def _score_dual_feats(model, pooled, cand_mat, candidates, relation, vocab, device):
    """Classifier over [pooled, tr+cand, pooled*(tr+cand)] for each candidate vector."""
    rel_id = vocab[f"[REL_{relation}]"]
    C = len(candidates)
    rel_t = torch.full((C,), rel_id, dtype=torch.long, device=device)
    chunk = 4096
    out: list[float] = []
    for i in range(0, C, chunk):
        cm = cand_mat[i : i + chunk]
        tr = model.embed(rel_t[i : i + chunk])
        target_emb = tr + cm
        pooled_b = pooled.unsqueeze(0).expand(cm.size(0), -1)
        feats = torch.cat([pooled_b, target_emb, pooled_b * target_emb], dim=-1)
        logits = model.classifier(feats).squeeze(-1)
        out.extend(logits.float().cpu().tolist())
    return dict(zip(candidates, out))
