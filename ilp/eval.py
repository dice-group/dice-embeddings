"""Filtered MRR/Hits@K evaluation (spec §7.1) + invariance tests (spec §7.2).

Usage:
    python -m ilp.eval --run checkpoints/default
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import (
    KnowledgeGraph,
    Triple,
    augment_with_inverse,
    build_sample,
    collate,
    k_hop_neighborhood,
    read_triples,
    two_hop_neighborhood,
)
from .model import InductiveKGModel
from .vocab import HOP_NONE, hop_distance_token, inverse_relation, load_vocab, z_token_ids


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
    # 1. Build the anonymized subgraph for this query (does not depend on candidate).
    rng = random.Random()
    subgraph, entity_distance = k_hop_neighborhood(anchor, kg, k=subgraph_hops)
    if exclude_triple is not None:
        s_, r_, o_ = exclude_triple
        subgraph.discard((s_, r_, o_))
        subgraph.discard((o_, inverse_relation(r_), s_))
    if len(subgraph) > max_triples:
        subgraph_list = rng.sample(list(subgraph), max_triples)
    else:
        subgraph_list = list(subgraph)

    z_indices = list(range(z_pool))
    rng.shuffle(z_indices)
    entity_map: dict[str, int] = {anchor: vocab["[X]"]}

    def assign(node: str) -> int:
        if node in entity_map:
            return entity_map[node]
        if node in fixed_values:
            return vocab[f"[VAL_{node}]"]
        if not z_indices:
            if collapse_z:
                entity_map[node] = vocab[f"[Z_{rng.randrange(z_pool)}]"]
                return entity_map[node]
            raise RuntimeError(f"Z pool exhausted for anchor={anchor}")
        entity_map[node] = vocab[f"[Z_{z_indices.pop()}]"]
        return entity_map[node]

    triples_tok = [[assign(s), vocab[f"[REL_{rel}]"], assign(o)] for s, rel, o in subgraph_list]
    n = len(triples_tok)
    pad = [[0, 0, 0]] * (max_triples - n)
    triples_t = torch.tensor([triples_tok + pad], dtype=torch.long, device=device)
    mask_t = torch.tensor(
        [[True] * n + [False] * (max_triples - n)], dtype=torch.bool, device=device
    )

    # Per-position hop-distance tokens. When use_hop_distance_tokens=False
    # these are all [HOP_NONE] → uniform bias, matches the no-hop training
    # path (the tok += model.embed(...) below is also gated by the flag).
    none_id = vocab[HOP_NONE]

    def _hop_id(node: str) -> int:
        if not use_hop_distance_tokens:
            return none_id
        return vocab[hop_distance_token(entity_distance.get(node, -1))]

    hop_tok = [[_hop_id(s), none_id, _hop_id(o)] for s, _, o in subgraph_list]
    hop_pad = [[none_id, none_id, none_id]] * (max_triples - n)
    hop_t = torch.tensor([hop_tok + hop_pad], dtype=torch.long, device=device)

    # 2. Encode subgraph once → pooled vector [1, d].
    B, N, _ = triples_t.shape
    tok = model.embed(triples_t) + model.intra_pos
    if use_hop_distance_tokens:
        tok = tok + model.embed(hop_t)
    tok = model.triple_encoder(tok.view(B * N, 3, -1))
    triple_vec = tok.mean(dim=1).view(B, N, -1)
    x_emb = model.embed(
        torch.tensor([[model.x_token_id]], dtype=torch.long, device=device)
    )
    h = torch.cat([x_emb, triple_vec], dim=1)
    mask_ext = torch.cat(
        [torch.ones(B, 1, dtype=torch.bool, device=device), mask_t], dim=1
    )
    h = model.sab_stack(h, src_key_padding_mask=~mask_ext)
    pooled = h[:, 0]  # [1, d]

    # 3. Score all candidates in vectorized chunks.
    # Candidate token: use the subgraph's assignment if present, else [VAL_*] for
    # schema, else any available [Z_*]. All [Z_*] embeddings are interchangeable
    # by Z-randomization training — unseen candidates legitimately tie.
    unused_z_id = (
        vocab[f"[Z_{z_indices[-1]}]"] if z_indices else vocab["[Z_0]"]
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
        tr = model.embed(rt)             # [c, d]
        tt = model.embed(ct)             # [c, d]
        target_emb = tr + tt
        pooled_b = pooled_d.unsqueeze(0).expand(target_emb.size(0), -1)
        feats = torch.cat([pooled_b, target_emb, pooled_b * target_emb], dim=-1)
        logits = model.classifier(feats).squeeze(-1)
        out.extend(logits.float().cpu().tolist())

    return dict(zip(candidates, out))


def evaluate_direction(
    model: InductiveKGModel,
    eval_triples: Sequence[Triple],
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    entity_pool: Sequence[str],
    known_triples: set[Triple],
    direction: str,  # "tail" or "head"
    max_triples: int,
    z_pool: int,
    device: torch.device,
    cand_batch_size: int = 128,
    collapse_z: bool = False,
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
) -> dict[str, float]:
    assert direction in ("tail", "head")
    model.eval()
    ranks: list[int] = []
    entity_pool = list(entity_pool)
    pool_set = set(entity_pool)

    rels: list[str] = []
    reachable_flags: list[bool] = []
    pbar = tqdm(eval_triples, desc=f"eval[{direction}]", unit="q", dynamic_ncols=True)
    for h, r, t in pbar:
        if direction == "tail":
            anchor, true_cand, rel_q = h, t, r
        else:
            anchor, true_cand, rel_q = t, h, inverse_relation(r)
        rels.append(r)
        # Is true_cand reachable in anchor's k-hop AFTER excluding the test triple?
        # Mirrors score_candidates' subgraph construction. Used downstream to
        # report stratified MRR (reachable vs unreachable) so per-relation gaps
        # aren't masked by ties on out-of-subgraph candidates (eval.py:115).
        nb, _ = k_hop_neighborhood(anchor, kg, k=subgraph_hops)
        nb.discard((h, r, t))
        nb.discard((t, inverse_relation(r), h))
        nb_ents = {s for s, _, _ in nb} | {o for _, _, o in nb}
        reachable_flags.append(true_cand in nb_ents)
        if true_cand not in pool_set:
            # Ensure the true candidate is always rankable.
            cands = entity_pool + [true_cand]
        else:
            cands = entity_pool

        scores = score_candidates(
            model, anchor, rel_q, cands, kg, vocab, fixed_values,
            max_triples, z_pool, cand_batch_size, device,
            exclude_triple=(h, r, t),
            collapse_z=collapse_z,
            subgraph_hops=subgraph_hops,
            use_hop_distance_tokens=use_hop_distance_tokens,
        )
        true_score = scores[true_cand]

        worse = 0
        equal = 0
        for c, s in scores.items():
            if c == true_cand:
                continue
            if direction == "tail":
                if (h, r, c) in known_triples:
                    continue
            else:
                if (c, r, t) in known_triples:
                    continue
            if s > true_score:
                worse += 1
            elif s == true_score:
                equal += 1
        # Mid-tie rank: standard KG-completion convention. Optimistic resolution
        # (just `worse + 1`) lets out-of-subgraph candidates falsely rank-1 because
        # they all share `unused_z_id` in score_candidates — see eval.py:115.
        ranks.append(worse + 1 + equal / 2.0)

        # Running MRR / Hits@10 in the progress bar.
        if len(ranks) % 50 == 0:
            r_arr = np.array(ranks, dtype=np.float64)
            pbar.set_postfix(
                MRR=f"{(1.0 / r_arr).mean():.3f}",
                H10=f"{(r_arr <= 10).mean():.3f}",
            )

    ranks_arr = np.array(ranks, dtype=np.float64)
    reach_arr = np.array(reachable_flags, dtype=bool) if reachable_flags else np.array([], dtype=bool)

    def _agg(mask: np.ndarray) -> dict[str, float]:
        r_sub = ranks_arr[mask]
        if len(r_sub) == 0:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "n": 0}
        return {
            "MRR": float((1.0 / r_sub).mean()),
            "Hits@1": float((r_sub <= 1).mean()),
            "Hits@3": float((r_sub <= 3).mean()),
            "Hits@10": float((r_sub <= 10).mean()),
            "n": int(mask.sum()),
        }

    by_rel: dict[str, dict[str, float]] = {}
    if rels:
        rels_arr = np.array(rels)
        for rel in np.unique(rels_arr):
            mask = rels_arr == rel
            by_rel[str(rel)] = {
                **_agg(mask),
                "reachable": _agg(mask & reach_arr),
                "unreachable": _agg(mask & ~reach_arr),
            }
    all_mask = np.ones(len(ranks_arr), dtype=bool)
    return {
        **_agg(all_mask),
        "reachable": _agg(reach_arr),
        "unreachable": _agg(~reach_arr),
        "by_relation": by_rel,
    }


def filter_known_relations(
    triples: Sequence[Triple], vocab: dict[str, int]
) -> tuple[list[Triple], int]:
    """Drop triples whose relation isn't in the trained vocab.

    DBpedia50 is open-world — some test relations never appear in train,
    so they have no trained [REL_*] embedding. Scoring them is meaningless;
    we report how many we skipped so MRR isn't silently inflated.
    """
    kept: list[Triple] = []
    dropped = 0
    for h, r, t in triples:
        if f"[REL_{r}]" in vocab:
            kept.append((h, r, t))
        else:
            dropped += 1
    return kept, dropped


def evaluate(
    model: InductiveKGModel,
    eval_triples: Sequence[Triple],
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    entity_pool: Sequence[str],
    known_triples: Iterable[Triple],
    max_triples: int,
    z_pool: int,
    device: torch.device,
    cand_batch_size: int = 128,
    collapse_z: bool = False,
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
) -> dict[str, float]:
    eval_triples, dropped = filter_known_relations(eval_triples, vocab)
    if dropped:
        print(f"[eval] skipped {dropped} test triples with unseen relations "
              f"(open-world). Evaluating on {len(eval_triples)} triples.")
    known = set(known_triples)
    tail = evaluate_direction(
        model, eval_triples, kg, vocab, fixed_values, entity_pool, known,
        "tail", max_triples, z_pool, device, cand_batch_size, collapse_z,
        subgraph_hops=subgraph_hops, use_hop_distance_tokens=use_hop_distance_tokens,
    )
    head = evaluate_direction(
        model, eval_triples, kg, vocab, fixed_values, entity_pool, known,
        "head", max_triples, z_pool, device, cand_batch_size, collapse_z,
        subgraph_hops=subgraph_hops, use_hop_distance_tokens=use_hop_distance_tokens,
    )
    avg = {k: 0.5 * (tail[k] + head[k]) for k in ("MRR", "Hits@1", "Hits@3", "Hits@10")}
    # Per-relation averages: only relations evaluated in both directions are aggregated.
    avg_by_rel: dict[str, dict[str, float]] = {}
    tail_by = tail.get("by_relation", {})
    head_by = head.get("by_relation", {})
    for rel in set(tail_by) | set(head_by):
        if rel in tail_by and rel in head_by:
            avg_by_rel[rel] = {
                k: 0.5 * (tail_by[rel][k] + head_by[rel][k])
                for k in ("MRR", "Hits@1", "Hits@3", "Hits@10")
            }
            avg_by_rel[rel]["n"] = tail_by[rel]["n"]  # same triples scored both directions
    avg["by_relation"] = avg_by_rel
    return {"tail": tail, "head": head, "avg": avg, "n_skipped": dropped}


# --- Invariance tests (spec §7.2) ----------------------------------------

@torch.no_grad()
def test_permutation_invariance(
    model: InductiveKGModel, sample: dict[str, torch.Tensor], atol: float = 1e-5,
    rng: random.Random | None = None,
) -> tuple[bool, float]:
    """Architectural property — passes from init."""
    rng = rng or random
    model.eval()
    t = sample["triples"].clone()
    m = sample["mask"]
    valid = torch.where(m)[0]
    perm = valid[torch.randperm(len(valid))]
    t_perm = t.clone()
    t_perm[valid] = t[perm]
    args = (sample["target_relation"].unsqueeze(0), sample["target_tail"].unsqueeze(0))
    out1 = model(t.unsqueeze(0), m.unsqueeze(0), *args)
    out2 = model(t_perm.unsqueeze(0), m.unsqueeze(0), *args)
    diff = (out1 - out2).abs().max().item()
    return diff <= atol, diff


@torch.no_grad()
def test_z_relabeling_invariance(
    model: InductiveKGModel,
    sample: dict[str, torch.Tensor],
    vocab: dict[str, int],
    z_pool_size: int,
    atol: float = 1e-3,
    rng: random.Random | None = None,
) -> tuple[bool, float]:
    """Should pass only after Z-randomized training."""
    rng = rng or random
    model.eval()
    z_ids = z_token_ids(vocab, z_pool_size)
    permuted = z_ids.copy()
    rng.shuffle(permuted)
    perm_map = dict(zip(z_ids, permuted))

    def relabel(x: int) -> int:
        return perm_map.get(x, x)

    t_relabeled = sample["triples"].clone().apply_(relabel)
    target_tail_rel = torch.tensor(relabel(int(sample["target_tail"])), dtype=torch.long)

    args1 = (sample["target_relation"].unsqueeze(0), sample["target_tail"].unsqueeze(0))
    args2 = (sample["target_relation"].unsqueeze(0), target_tail_rel.unsqueeze(0))
    out1 = model(sample["triples"].unsqueeze(0), sample["mask"].unsqueeze(0), *args1)
    out2 = model(t_relabeled.unsqueeze(0), sample["mask"].unsqueeze(0), *args2)
    diff = (out1 - out2).abs().max().item()
    return diff <= atol, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Run directory under checkpoints/, e.g. checkpoints/z1")
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint file. Defaults to {run}/model_final.pt")
    args = ap.parse_args()
    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "model_final.pt"

    data_dir = Path(cfg["data_dir"])
    fmt = cfg.get("triple_format", "head_tail_relation")
    train = read_triples(data_dir / "train.txt", fmt=fmt)
    valid = read_triples(data_dir / "valid.txt", fmt=fmt)
    test = read_triples(data_dir / "test.txt", fmt=fmt)
    vocab, fixed_values = load_vocab(run_dir / "vocab.json")

    # Filter the test graph too: any triple whose relation isn't in vocab
    # would crash subgraph encoding. (DBpedia50 open-world quirk.)
    test_for_graph, dropped_graph = filter_known_relations(test, vocab)
    if dropped_graph:
        print(f"[eval] removed {dropped_graph} test KG triples with unseen relations "
              f"before subgraph extraction.")
    test_kg = KnowledgeGraph(augment_with_inverse(test_for_graph))
    test_entities = test_kg.entities

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = InductiveKGModel(
        vocab_size=len(vocab),
        x_token_id=vocab["[X]"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_triple_layers=cfg["n_triple_layers"],
        n_sab=cfg["n_sab"],
        dropout=cfg["dropout"],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    results = evaluate(
        model, test_for_graph, test_kg, vocab, fixed_values, test_entities,
        known_triples=train + valid + test_for_graph,
        max_triples=cfg["max_triples"],
        z_pool=cfg["z_pool_size"],
        device=device,
        collapse_z=cfg.get("collapse_z", False),
    )
    print("Tail :", results["tail"])
    print("Head :", results["head"])
    print("Avg  :", results["avg"])


if __name__ == "__main__":
    main()
