"""Filtered MRR/Hits@K evaluation (spec §7.1) + invariance tests (spec §7.2).

CLI-first: evaluate a self-contained model bundle on a held-out split.

    python -m ilp.eval --model checkpoints/run/model_final.pt --kg-dir KGs/mykg
    python -m ilp.eval --model bundle.pt --kg-dir KGs/kg_inductive \\
        --obs-file KGs/kg_inductive_ind/train.txt --test-file KGs/kg_inductive_ind/test.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
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
from .model import InductiveKGModel, load_bundle
from .vocab import HOP_NONE, hop_distance_token, inverse_relation, z_token_ids


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

    # 2. Encode subgraph once → pooled vector [1, d]. One shared random
    # anonymization per query (runtime), so subgraph and candidates see the same
    # [Z_*] vectors; _entity_embed handles the override. Runtime MC averaging is
    # done one level up in evaluate_direction (a fresh full draw per pass).
    rand_bank = (
        model._make_rand_bank(device=device)
        if getattr(model, "z_mode", "learned") == "runtime" else None
    )
    B, N, _ = triples_t.shape
    tok = model._entity_embed(triples_t, rand_bank) + model.intra_pos
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
        tr = model.embed(rt)                          # [c, d]
        tt = model._entity_embed(ct, rand_bank)       # [c, d]
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
    eval_mc: int = 1,
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

        # runtime: average over `eval_mc` fresh draws to marginalize the random
        # features (one draw is too noisy to rank a large pool). learned: 1 pass.
        mc = max(1, eval_mc) if getattr(model, "z_mode", "learned") == "runtime" else 1
        scores = score_candidates(
            model, anchor, rel_q, cands, kg, vocab, fixed_values,
            max_triples, z_pool, cand_batch_size, device,
            exclude_triple=(h, r, t),
            collapse_z=collapse_z,
            subgraph_hops=subgraph_hops,
            use_hop_distance_tokens=use_hop_distance_tokens,
        )
        for _ in range(mc - 1):
            extra = score_candidates(
                model, anchor, rel_q, cands, kg, vocab, fixed_values,
                max_triples, z_pool, cand_batch_size, device,
                exclude_triple=(h, r, t),
                collapse_z=collapse_z,
                subgraph_hops=subgraph_hops,
                use_hop_distance_tokens=use_hop_distance_tokens,
            )
            for c in scores:
                scores[c] += extra[c]
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
    eval_mc: int = 1,
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
        eval_mc=eval_mc,
    )
    head = evaluate_direction(
        model, eval_triples, kg, vocab, fixed_values, entity_pool, known,
        "head", max_triples, z_pool, device, cand_batch_size, collapse_z,
        subgraph_hops=subgraph_hops, use_hop_distance_tokens=use_hop_distance_tokens,
        eval_mc=eval_mc,
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


# --- reusable eval driver (shared by this CLI and train.py auto-eval) ------

def load_eval_inputs(
    kg_dir: str | Path,
    fmt: str,
    test_file: str | Path | None = None,
    obs_file: str | Path | None = None,
) -> tuple[list[Triple], list[Triple], list[Triple]]:
    """Resolve (test, known, context) triple lists for evaluation.

    - test     : the queries (kg_dir/test.txt unless `test_file` overrides).
    - known    : the filter set for filtered ranking — train + valid + test, plus
      the observed inference graph (`obs_file`) when given, so other true facts
      in the inference graph aren't counted as ranking competitors.
    - context  : triples used to build the subgraph KG. Defaults to the test
      split; with `obs_file` (GraIL-style) it is obs_file + test.
    """
    kg_dir = Path(kg_dir)
    test_path = Path(test_file) if test_file else kg_dir / "test.txt"
    test = read_triples(test_path, fmt=fmt)

    known: list[Triple] = list(test)
    for name in ("train.txt", "valid.txt"):
        p = kg_dir / name
        if p.exists():
            known += read_triples(p, fmt=fmt)

    if obs_file:
        obs = read_triples(Path(obs_file), fmt=fmt)
        known += obs
        # Include the queries so test-internal facts also enter the subgraph;
        # each query's own triple is excluded inside score_candidates.
        context = obs + test
    else:
        context = list(test)
    return test, known, context


def run_eval(
    model: InductiveKGModel,
    vocab: dict[str, int],
    fixed_values: set[str],
    cfg: dict,
    device: torch.device,
    *,
    test_triples: Sequence[Triple],
    known_triples: Sequence[Triple],
    context_triples: Sequence[Triple],
) -> dict:
    """Build the context KG and run filtered MRR/Hits@K evaluation."""
    context_for_graph, dropped = filter_known_relations(context_triples, vocab)
    if dropped:
        print(f"[eval] removed {dropped} context triples with unseen relations "
              f"before building the subgraph KG.")
    kg = KnowledgeGraph(augment_with_inverse(context_for_graph))
    return evaluate(
        model, test_triples, kg, vocab, fixed_values, kg.entities,
        known_triples=known_triples,
        max_triples=cfg["max_triples"], z_pool=cfg["z_pool_size"],
        device=device, collapse_z=cfg.get("collapse_z", False),
        subgraph_hops=cfg.get("subgraph_hops", 2),
        use_hop_distance_tokens=cfg.get("use_hop_distance_tokens", False),
        eval_mc=cfg.get("eval_mc", 8),  # runtime MC draws; ignored for learned
    )


def print_eval_results(results: dict, per_relation: bool = False) -> None:
    for split in ("tail", "head", "avg"):
        m = results[split]
        if split == "avg":
            print(f"{split:5s}  MRR={m['MRR']:.4f}  H@1={m['Hits@1']:.4f}  "
                  f"H@3={m['Hits@3']:.4f}  H@10={m['Hits@10']:.4f}")
        else:
            print(f"{split:5s}  MRR={m['MRR']:.4f}  H@1={m['Hits@1']:.4f}  "
                  f"H@3={m['Hits@3']:.4f}  H@10={m['Hits@10']:.4f}  n={m['n']}")
            if "reachable" in m:
                rm = m["reachable"]; um = m["unreachable"]
                print(f"  reachable    MRR={rm['MRR']:.4f}  H@1={rm['Hits@1']:.4f}  "
                      f"H@10={rm['Hits@10']:.4f}  n={rm['n']}")
                print(f"  unreachable  MRR={um['MRR']:.4f}  H@1={um['Hits@1']:.4f}  "
                      f"H@10={um['Hits@10']:.4f}  n={um['n']}")
    if results.get("n_skipped"):
        print(f"(skipped {results['n_skipped']} test triples with unseen relations)")
    if per_relation:
        _print_per_relation(results)


def _print_per_relation(results: dict, top_n: int = 30) -> None:
    """Per-relation MRR table including inverse rows.

    For each relation r, the inverse row r__inv reuses head's metrics as tail
    and vice versa — predicting the tail of (t, r__inv, h) is the same problem
    as predicting the head of (h, r, t). The flip makes per-relation asymmetry
    obvious: functional relations have one direction near 1 and the other near 0.
    """
    tail_br = results["tail"].get("by_relation", {})
    head_br = results["head"].get("by_relation", {})
    rels = sorted(
        set(tail_br) | set(head_br),
        key=lambda r: -tail_br.get(r, head_br.get(r, {})).get("n", 0),
    )[:top_n]
    print(f"\n--- per-relation MRR (top {len(rels)} by support) ---")
    print(f"{'relation':40s}  {'tail_MRR':>8s}  {'head_MRR':>8s}  {'n':>5s}")
    for r in rels:
        t_mrr = tail_br.get(r, {}).get("MRR", float("nan"))
        h_mrr = head_br.get(r, {}).get("MRR", float("nan"))
        n = tail_br.get(r, head_br.get(r, {})).get("n", 0)
        print(f"{r:40s}  {t_mrr:8.4f}  {h_mrr:8.4f}  {n:5d}")
        print(f"{(r + '__inv'):40s}  {h_mrr:8.4f}  {t_mrr:8.4f}  {n:5d}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ilp.eval",
        description="Filtered MRR / Hits@K evaluation of a model bundle on a held-out split.",
    )
    ap.add_argument("--model", required=True, help="Bundled .pt (model + vocab + cfg).")
    ap.add_argument("--kg-dir", required=True,
                    help="Directory with test.txt (and optionally train.txt/valid.txt for the filter set).")
    ap.add_argument("--test-file", default=None,
                    help="Override path to the eval split (default: {kg-dir}/test.txt).")
    ap.add_argument("--obs-file", default=None,
                    help="Observed-graph triples for subgraph context (GraIL-style inductive eval). "
                         "If omitted, the test split is used as context.")
    ap.add_argument("--json-out", default=None,
                    help="If set, dump the full results dict (incl. per-relation MRR) as JSON.")
    ap.add_argument("--per-relation", action="store_true",
                    help="Print a per-relation MRR table with each relation's inverse row alongside it.")
    ap.add_argument("--device", default=None,
                    help="Override compute device (e.g. 'cuda', 'cpu'). Default: 'cuda' "
                         "if available, else the checkpoint's cfg device.")
    return ap


def main():
    args = build_parser().parse_args()
    model, vocab, fixed_values, cfg, device = load_bundle(args.model)
    if args.device or (device.type == "cpu" and torch.cuda.is_available()):
        device = torch.device(args.device or "cuda")
        model.to(device)
    fmt = cfg.get("triple_format", "head_relation_tail")
    test, known, context = load_eval_inputs(args.kg_dir, fmt, args.test_file, args.obs_file)
    results = run_eval(
        model, vocab, fixed_values, cfg, device,
        test_triples=test, known_triples=known, context_triples=context,
    )
    print_eval_results(results, per_relation=args.per_relation)
    if args.json_out:
        import json
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[eval] wrote full results to {out_path}")


if __name__ == "__main__":
    main()
