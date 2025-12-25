#!/usr/bin/env python3
"""
Hub-focused edge pruning for triple files: each line is "head<TAB>relation<TAB>tail".

Definitions (no magic, no vibes):
- Endpoint pair = the unordered node pair {head, tail}. (Direction and relation are ignored here.)
- "Non-redundant triple" = its endpoint pair appears exactly once in the whole file.
  (If {u,v} appears in 3 triples with different relations, ALL 3 are redundant.)
- "Incoming edge to X" = any triple whose tail == X.
- "Hub node" = a node with many incoming NON-REDUNDANT edges (tail == node, triple is non-redundant).

Removals performed:
  (1) All edges BETWEEN hub nodes:
        any triple where head in hubs AND tail in hubs (direction still ignored in the sense we don't care which hub is head/tail)
  (2) A random subset of incoming edges into each hub node (tail == hub),
      sampled from a pool you choose: all incoming vs only non-redundant incoming.

Outputs:
  - kept triples (same 3-column format)
  - removed triples (same 3-column format)
  - removed_annotated.tsv (adds "reason" column)
  - report.txt (hubs, incoming non-redundant edges per hub, hub-hub edges, summary)
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
from typing import Dict, List, Sequence, Tuple, Set, Optional

Triple = Tuple[str, str, str]  # (head, relation, tail)


def read_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                bad += 1
                continue
            h, r, t = parts
            triples.append((h, r, t))
    if bad:
        print(f"[warn] Skipped {bad} malformed lines (expected 3 tab-separated columns).")
    return triples


def write_triples(path: str, triples: Sequence[Triple]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def unordered_pair(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def compute_pair_counts(triples: Sequence[Triple]) -> collections.Counter:
    c = collections.Counter()
    for h, _, t in triples:
        c[unordered_pair(h, t)] += 1
    return c


def compute_nonredundant_mask(triples: Sequence[Triple], pair_counts: collections.Counter) -> List[bool]:
    return [pair_counts[unordered_pair(h, t)] == 1 for (h, _, t) in triples]


def build_incoming_index(triples: Sequence[Triple], mask: Optional[Sequence[bool]] = None) -> Dict[str, List[int]]:
    inc: Dict[str, List[int]] = collections.defaultdict(list)
    for i, (_, _, t) in enumerate(triples):
        if mask is None or mask[i]:
            inc[t].append(i)
    return inc


def pick_hubs_by_incoming_nonredundant(
    inc_nr: Dict[str, List[int]],
    top_k: int,
    min_incoming_nr: int,
) -> List[Tuple[str, int]]:
    scored = [(node, len(idxs)) for node, idxs in inc_nr.items() if len(idxs) >= min_incoming_nr]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hub_hub_edge_indices(triples: Sequence[Triple], hub_set: Set[str]) -> List[int]:
    return [i for i, (h, _, t) in enumerate(triples) if h in hub_set and t in hub_set]


def sample_incoming_edges_per_hub(
    hub_set: Set[str],
    inc_pool: Dict[str, List[int]],
    seed: int,
    exclude: Set[int],
    frac: Optional[float],
    n_per_hub: Optional[int],
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    out: Dict[str, List[int]] = {}

    for hub in sorted(hub_set):
        cand = [i for i in inc_pool.get(hub, []) if i not in exclude]
        if not cand:
            out[hub] = []
            continue

        if n_per_hub is not None:
            k = min(max(n_per_hub, 0), len(cand))
        else:
            if frac is None:
                k = 0
            else:
                if not (0.0 <= frac <= 1.0):
                    raise ValueError(f"--remove-incoming-frac must be in [0,1], got {frac}")
                k = int(math.floor(frac * len(cand)))

        if k <= 0:
            out[hub] = []
        else:
            out[hub] = rng.sample(cand, k)

    return out


def triple_str(tr: Triple) -> str:
    h, r, t = tr
    return f"{h}\t{r}\t{t}"


def greedy_pick_disconnected_hubs(
    triples: Sequence[Triple],
    candidate_hubs: List[str],
    final_k: int,
) -> List[str]:
    """
    Greedy heuristic:
      - start with the best-scoring hub (first in candidate_hubs)
      - repeatedly add the hub that adds the fewest hub-hub edges to the current set
        (ties broken by original candidate order)
    """
    if final_k >= len(candidate_hubs):
        return candidate_hubs[:]

    cand_order = {h: i for i, h in enumerate(candidate_hubs)}
    selected: List[str] = [candidate_hubs[0]]
    selected_set = {selected[0]}

    # Precompute hub-hub adjacency counts between candidates
    pair_count = collections.Counter()
    cand_set = set(candidate_hubs)
    for (h, _, t) in triples:
        if h in cand_set and t in cand_set and h != t:
            pair_count[unordered_pair(h, t)] += 1

    while len(selected) < final_k:
        best = None
        best_added = None
        for h in candidate_hubs:
            if h in selected_set:
                continue
            added = 0
            for s in selected:
                added += pair_count.get(unordered_pair(h, s), 0)
            if best is None or added < best_added or (added == best_added and cand_order[h] < cand_order[best]):
                best = h
                best_added = added
        selected.append(best)
        selected_set.add(best)

    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input triple file (head<TAB>rel<TAB>tail).")

    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of hub CANDIDATES (ranked by incoming non-redundant edges).")
    ap.add_argument("--min-incoming-nr", type=int, default=1,
                    help="Minimum incoming non-redundant edges for a node to be considered a hub candidate.")

    ap.add_argument("--final-hubs", type=int, default=None,
                    help="If set (< top-k), pick this many hubs from the candidates using a greedy 'min hub-hub edges' heuristic.")

    ap.add_argument("--incoming-sample-pool", choices=["all", "nonredundant"], default="nonredundant",
                    help="Pool to sample incoming edges from when pruning within hubs.")

    ap.add_argument("--remove-incoming-frac", type=float, default=0.0,
                    help="Fraction of (pool) incoming edges to remove per hub. Ignored if --remove-incoming-n is set.")
    ap.add_argument("--remove-incoming-n", type=int, default=None,
                    help="Fixed number of (pool) incoming edges to remove per hub (overrides --remove-incoming-frac).")

    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")

    ap.add_argument("--out-prefix", default=None,
                    help="Prefix for outputs. Defaults to <input_basename> with parameters.")

    args = ap.parse_args()

    triples = read_triples(args.input)
    if not triples:
        raise SystemExit("No triples loaded. Check input formatting (3 tab-separated columns).")

    pair_counts = compute_pair_counts(triples)
    nr_mask = compute_nonredundant_mask(triples, pair_counts)

    inc_all = build_incoming_index(triples, mask=None)
    inc_nr = build_incoming_index(triples, mask=nr_mask)

    hubs_scored = pick_hubs_by_incoming_nonredundant(
        inc_nr, top_k=args.top_k, min_incoming_nr=args.min_incoming_nr
    )
    hub_candidates = [h for h, _ in hubs_scored]

    final_k = args.final_hubs
    if final_k is None or final_k >= len(hub_candidates):
        hubs = hub_candidates
    else:
        if final_k <= 0:
            hubs = []
        else:
            hubs = greedy_pick_disconnected_hubs(triples, hub_candidates, final_k=final_k)

    hub_set = set(hubs)

    # (1) remove edges between hubs
    between_idxs = hub_hub_edge_indices(triples, hub_set)
    between_set = set(between_idxs)

    # (2) remove random subset of incoming edges into hubs
    pool = inc_all if args.incoming_sample_pool == "all" else inc_nr
    sampled_by_hub = sample_incoming_edges_per_hub(
        hub_set=hub_set,
        inc_pool=pool,
        seed=args.seed,
        exclude=between_set,  # don't double-remove
        frac=args.remove_incoming_frac if args.remove_incoming_n is None else None,
        n_per_hub=args.remove_incoming_n,
    )

    sampled_idxs = sorted({i for idxs in sampled_by_hub.values() for i in idxs})

    # Merge removals with reasons
    reason_by_idx: Dict[int, str] = {}
    for i in between_idxs:
        reason_by_idx[i] = "between_hubs"
    for hub, idxs in sampled_by_hub.items():
        for i in idxs:
            # if a sampled edge is also between hubs, keep between_hubs label
            reason_by_idx.setdefault(i, f"incoming_sampled_to:{hub}")

    remove_idx = sorted(reason_by_idx.keys())

    kept: List[Triple] = []
    removed: List[Triple] = []
    for i, tr in enumerate(triples):
        if i in reason_by_idx:
            removed.append(tr)
        else:
            kept.append(tr)

    base = os.path.splitext(os.path.basename(args.input))[0]
    if args.out_prefix:
        prefix = args.out_prefix
    else:
        if args.remove_incoming_n is not None:
            samp = f"n{args.remove_incoming_n}"
        else:
            samp = f"frac{args.remove_incoming_frac}"
        final_tag = f"final{len(hubs)}" if (args.final_hubs is not None and args.final_hubs < args.top_k) else f"top{args.top_k}"
        prefix = f"{base}.pruned_{final_tag}_pool{args.incoming_sample_pool}_{samp}_seed{args.seed}"

    out_kept = prefix + ".kept.txt"
    out_removed = prefix + ".removed.txt"
    out_removed_ann = prefix + ".removed_annotated.tsv"
    out_report = prefix + ".report.txt"

    write_triples(out_kept, kept)
    write_triples(out_removed, removed)

    with open(out_removed_ann, "w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\treason\n")
        for i in remove_idx:
            h, r, t = triples[i]
            f.write(f"{h}\t{r}\t{t}\t{reason_by_idx[i]}\n")

    # Report
    between_pairs = collections.Counter(unordered_pair(triples[i][0], triples[i][2]) for i in between_idxs)

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("=== HUBS (ranked by incoming non-redundant edges) ===\n")
        f.write("hub\tincoming_all\tincoming_nonredundant\tsampled_incoming_removed\n")
        for hub, _score in hubs_scored:
            if hub not in hub_set:
                continue
            f.write(
                f"{hub}\t{len(inc_all.get(hub, []))}\t{len(inc_nr.get(hub, []))}\t{len(sampled_by_hub.get(hub, []))}\n"
            )

        f.write("\n=== EDGES: incoming NON-REDUNDANT edges per hub (tail = hub) ===\n")
        for hub in hubs:
            idxs = inc_nr.get(hub, [])
            f.write(f"\n# hub={hub}  incoming_nonredundant={len(idxs)}\n")
            for i in idxs:
                f.write(triple_str(triples[i]) + "\n")

        f.write("\n=== EDGES: between hubs (both endpoints in hub set) ===\n")
        f.write(f"total_between_hubs_edges={len(between_idxs)}\n")
        f.write("\n# pair_counts (unordered endpoints)\n")
        for (u, v), c in between_pairs.most_common():
            f.write(f"{u}\t{v}\tcount={c}\n")

        f.write("\n# actual triples between hubs\n")
        for i in between_idxs:
            f.write(triple_str(triples[i]) + "\n")

        f.write("\n=== REMOVAL SUMMARY ===\n")
        f.write(f"input_triples={len(triples)}\n")
        f.write(f"kept_triples={len(kept)}\n")
        f.write(f"removed_triples={len(removed)}\n")
        f.write(f"removed_between_hubs={len(between_idxs)}\n")
        f.write(f"removed_incoming_sampled={len(sampled_idxs)}\n")

    # Console summary
    print("Selected hubs:")
    for hub in hubs:
        print(f"  {hub}  in_all={len(inc_all.get(hub, []))}  in_nonredundant={len(inc_nr.get(hub, []))}  sampled_removed={len(sampled_by_hub.get(hub, []))}")
    print(f"Edges between hubs removed: {len(between_idxs)}")
    print(f"Incoming sampled edges removed: {len(sampled_idxs)}")
    print(f"Total removed: {len(removed)}   Total kept: {len(kept)}")
    print("Wrote:")
    print("  ", out_kept)
    print("  ", out_removed)
    print("  ", out_removed_ann)
    print("  ", out_report)


if __name__ == "__main__":
    main()
