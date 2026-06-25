"""Diagnostic: Check whether the true answer for each query is reachable
within the anchor's k-hop subgraph — under `obs`-only context vs `obs + test`.

A query is *reachable* iff, after
excluding its own triple, the true answer lies in the anchor's k-hop
neighborhood — exactly the flag eval.evaluate_direction computes and uses to
stratify reachable/unreachable MRR. Unreachable queries are essentially
un-rankable by a subgraph model not anchored on candidate entities (ie single anchor mode), so this is an upper bound on attainable recall.

Inductive wiring: context/observed graph = inference graph's train.txt
(--obs-file), queries = inference graph's test.txt (--test-file).

Run from the repo root (the dir containing the `ilp/` package):

    python -m ilp.util.check_reachability --ind-dir KGs/WN18RR_v1_ind --hops 2
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ilp.dataset import (
    KnowledgeGraph,
    augment_with_inverse,
    k_hop_neighborhood,
    read_triples,
)
from ilp.vocab import inverse_relation


def reachable_flags(test, kg, hops):
    """Per-query reachability for both directions, mirroring eval.evaluate_direction.

    Returns rows of (relation, direction, reachable_bool).
    """
    rows = []
    for h, r, t in test:
        for direction in ("tail", "head"):
            anchor, true_cand = (h, t) if direction == "tail" else (t, h)
            nb, _ = k_hop_neighborhood(anchor, kg, k=hops)
            excluded = {(h, r, t), (t, inverse_relation(r), h)}
            nb_ents = {e for s, rr, o in nb if (s, rr, o) not in excluded
                       for e in (s, o)}
            rows.append((r, direction, true_cand in nb_ents))
    return rows


def frac(rows, direction=None):
    sel = [reach for _, d, reach in rows if direction is None or d == direction]
    return (sum(sel) / len(sel), len(sel)) if sel else (0.0, 0)


def print_block(title, rows):
    print(f"\n{title}")
    for d in ("tail", "head", None):
        f, n = frac(rows, d)
        label = d if d else "overall"
        print(f"  {label:<8} reachable {f:6.2%}  ({sum(1 for _,dd,r in rows if (d is None or dd==d) and r)}/{n})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ind-dir", default="KGs/WN18RR_v1_ind",
                    help="Inference graph: train.txt=context, test.txt=queries.")
    ap.add_argument("--obs-file", default=None,
                    help="Observed-graph triples (default: {ind-dir}/train.txt).")
    ap.add_argument("--test-file", default=None,
                    help="Query triples (default: {ind-dir}/test.txt).")
    ap.add_argument("--triple-format", default="head_relation_tail")
    ap.add_argument("--hops", type=int, default=2,
                    help="Subgraph BFS depth (must match training subgraph_hops).")
    ap.add_argument("--per-relation", action="store_true",
                    help="Print per-relation obs-only reachability (overall direction).")
    args = ap.parse_args()

    ind = Path(args.ind_dir)
    fmt = args.triple_format
    obs = read_triples(Path(args.obs_file) if args.obs_file else ind / "train.txt", fmt=fmt)
    test = read_triples(Path(args.test_file) if args.test_file else ind / "test.txt", fmt=fmt)
    print(f"observed (context) triples: {len(obs)}   queries: {len(test)}   hops={args.hops}")

    # Two context KGs, each inverse-augmented exactly like eval.run_eval.
    kg_obs = KnowledgeGraph(augment_with_inverse(obs))
    kg_obs_test = KnowledgeGraph(augment_with_inverse(obs + test))

    rows_obs = reachable_flags(test, kg_obs, args.hops)
    rows_obs_test = reachable_flags(test, kg_obs_test, args.hops)

    print("\n" + "=" * 60)
    print("REACHABILITY of the true answer in the anchor's k-hop subgraph")
    print("=" * 60)
    print_block("context = obs + test   (OLD behavior; test leaks into subgraph)", rows_obs_test)
    print_block("context = obs only     (strict inductive; test never in subgraph)", rows_obs)

    old_f, _ = frac(rows_obs_test)
    new_f, _ = frac(rows_obs)
    print(f"\n  Δ overall reachable: {new_f - old_f:+.2%}  "
          f"({old_f:.2%} → {new_f:.2%})")
    print("  Unreachable queries are an upper bound the model cannot rank correctly.")

    if args.per_relation:
        by_rel = defaultdict(list)
        for r, _, reach in rows_obs:
            by_rel[r].append(reach)
        print("\n" + "-" * 60)
        print(f"PER-RELATION reachability (context=obs only)")
        print("-" * 60)
        print(f"{'relation':<40}{'n':>6}{'reachable':>11}")
        for rel in sorted(by_rel, key=lambda x: sum(by_rel[x]) / len(by_rel[x])):
            v = by_rel[rel]
            print(f"{rel[:38]:<40}{len(v):>6}{sum(v) / len(v):>10.1%}")


if __name__ == "__main__":
    main()
