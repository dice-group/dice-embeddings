"""Diagnostic: subgraph sizes for head vs tail eval queries, plus a
stratified eval pass that bins ranks by (direction x subsampled?) and
by relation.

Tests two hypotheses for asymmetric head/tail MRR:
  (1) Hub-tail subsampling destroys context for head queries.
  (2) Many-to-one relations make head prediction inherently harder.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from ilp.dataset import KnowledgeGraph, read_triples, two_hop_neighborhood
from ilp.dataset import augment_with_inverse
from ilp.eval import filter_known_relations, score_candidates
from ilp.model import InductiveKGModel
from ilp.vocab import inverse_relation, load_vocab


def pct(arr, qs=(50, 75, 90, 95, 99)):
    return {f"p{q}": int(np.percentile(arr, q)) for q in qs}


def subgraph_size_report(test, kg, max_triples):
    head_sizes = np.array([len(two_hop_neighborhood(h, kg)) for h, _, _ in test])
    tail_sizes = np.array([len(two_hop_neighborhood(t, kg)) for _, _, t in test])

    def report(name, arr):
        over = (arr > max_triples).mean() * 100
        print(f"{name}")
        print(f"  mean={arr.mean():.1f}  median={np.median(arr):.0f}  max={arr.max()}")
        print(f"  {pct(arr)}")
        print(f"  % over budget ({max_triples}): {over:.1f}%\n")

    print("=" * 60)
    print("TAIL prediction (anchor = h, true_cand = t)")
    print("=" * 60)
    report("anchor=h subgraph sizes", head_sizes)
    print("=" * 60)
    print("HEAD prediction (anchor = t, true_cand = h)")
    print("=" * 60)
    report("anchor=t subgraph sizes", tail_sizes)


def filtered_rank(scores, true_cand, true_score, h, r, t, direction, known):
    worse = 0
    for c, s in scores.items():
        if c == true_cand:
            continue
        if direction == "tail":
            if (h, r, c) in known:
                continue
        else:
            if (c, r, t) in known:
                continue
        if s > true_score:
            worse += 1
    return worse + 1


def stratified_eval(model, cfg, vocab, fixed_values, kg, test, known, device,
                    sample_size, seed):
    rng = random.Random(seed)
    sample = rng.sample(test, min(sample_size, len(test)))
    entity_pool = list(kg.entities)
    pool_set = set(entity_pool)
    max_triples = cfg["max_triples"]
    z_pool = cfg["z_pool_size"]

    # rows: (direction, subsampled, rank, relation)
    rows = []

    for h, r, t in tqdm(sample, desc="stratified eval", dynamic_ncols=True):
        for direction in ("tail", "head"):
            if direction == "tail":
                anchor, true_cand, rel_q = h, t, r
            else:
                anchor, true_cand, rel_q = t, h, inverse_relation(r)
            sub_size = len(two_hop_neighborhood(anchor, kg))
            subsampled = sub_size > max_triples

            cands = entity_pool if true_cand in pool_set else entity_pool + [true_cand]
            scores = score_candidates(
                model, anchor, rel_q, cands, kg, vocab, fixed_values,
                max_triples, z_pool, 128, device,
                exclude_triple=(h, r, t),
            )
            rank = filtered_rank(
                scores, true_cand, scores[true_cand], h, r, t, direction, known
            )
            rows.append((direction, subsampled, rank, r))

    return rows


def print_stratified(rows):
    def mrr(ranks):
        if not ranks:
            return float("nan")
        return float(np.mean([1.0 / x for x in ranks]))

    def h10(ranks):
        if not ranks:
            return float("nan")
        return float(np.mean([1.0 if x <= 10 else 0.0 for x in ranks]))

    print("\n" + "=" * 60)
    print("STRATIFIED MRR / Hits@10 by (direction x subsampled)")
    print("=" * 60)
    print(f"{'direction':<10}{'subsampled':<12}{'n':>6}{'MRR':>10}{'H@10':>10}")
    for direction in ("tail", "head"):
        for sub in (False, True):
            ranks = [r for d, s, r, _ in rows if d == direction and s == sub]
            print(f"{direction:<10}{str(sub):<12}{len(ranks):>6}"
                  f"{mrr(ranks):>10.3f}{h10(ranks):>10.3f}")
        all_ranks = [r for d, _, r, _ in rows if d == direction]
        print(f"{direction:<10}{'ALL':<12}{len(all_ranks):>6}"
              f"{mrr(all_ranks):>10.3f}{h10(all_ranks):>10.3f}")
        print()

    # Per-relation breakdown, sorted by abs(tail_MRR - head_MRR)
    by_rel_dir: dict[tuple[str, str], list[int]] = defaultdict(list)
    for d, _, r, rel in rows:
        by_rel_dir[(rel, d)].append(r)

    rels = sorted({rel for rel, _ in by_rel_dir})
    rel_stats = []
    for rel in rels:
        tail_ranks = by_rel_dir.get((rel, "tail"), [])
        head_ranks = by_rel_dir.get((rel, "head"), [])
        if len(tail_ranks) < 10 or len(head_ranks) < 10:
            continue
        tail_mrr = mrr(tail_ranks)
        head_mrr = mrr(head_ranks)
        rel_stats.append((rel, len(tail_ranks), tail_mrr, head_mrr,
                          tail_mrr - head_mrr))

    rel_stats.sort(key=lambda x: -x[4])
    print("=" * 60)
    print(f"PER-RELATION (>=10 queries each direction, "
          f"sorted by tail_MRR - head_MRR desc)")
    print("=" * 60)
    print(f"{'relation':<40}{'n':>6}{'tail_MRR':>10}{'head_MRR':>10}{'gap':>10}")
    for rel, n, tm, hm, gap in rel_stats[:20]:
        print(f"{rel[:38]:<40}{n:>6}{tm:>10.3f}{hm:>10.3f}{gap:>+10.3f}")
    if len(rel_stats) > 20:
        print(f"... and {len(rel_stats) - 20} more relations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Run directory under checkpoints/, e.g. checkpoints/default")
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint file. Defaults to {run}/model_final.pt")
    ap.add_argument("--sample-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-eval", action="store_true",
                    help="Only print subgraph-size stats; skip model-based eval.")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "model_final.pt"
    data_dir = Path(cfg["data_dir"])
    max_triples = cfg["max_triples"]

    fmt = cfg.get("triple_format", "head_tail_relation")
    train = read_triples(data_dir / "train.txt", fmt=fmt)
    valid = read_triples(data_dir / "valid.txt", fmt=fmt)
    test = read_triples(data_dir / "test.txt", fmt=fmt)
    vocab, fixed_values = load_vocab(run_dir / "vocab.json")

    test, dropped = filter_known_relations(test, vocab)
    print(f"Test triples: {len(test)}  (dropped {dropped} for unseen rels)")
    print(f"max_triples (subsample budget): {max_triples}\n")

    kg = KnowledgeGraph(augment_with_inverse(test))
    subgraph_size_report(test, kg, max_triples)

    if args.skip_eval:
        return

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
    model.eval()

    known = set(train) | set(valid) | set(test)
    rows = stratified_eval(
        model, cfg, vocab, fixed_values, kg, test, known, device,
        args.sample_size, args.seed,
    )
    print_stratified(rows)


if __name__ == "__main__":
    main()
