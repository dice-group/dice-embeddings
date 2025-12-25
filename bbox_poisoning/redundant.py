#!/usr/bin/env python3
"""
Hub/non-redundant edge analysis for a (directed) labeled graph.

Input format (one edge per line, tab-separated):
    head<TAB>relation<TAB>tail

Definitions (matching your request):
- Non-redundant edge: the unordered endpoint pair {head, tail} occurs exactly once
  in the whole graph, regardless of direction AND relation label.
- Hub node: node with many *incoming* non-redundant edges (incoming = appears as 'tail').
- Edges between hubs: edges where BOTH endpoints are hubs (direction kept in output,
  but hub-pair counting treats endpoints as unordered).

Outputs:
- hub_summary.tsv
- hub_incoming_nonredundant_edges.tsv
- hub_pair_counts.tsv
- hub_pair_edges.tsv
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from itertools import combinations
import argparse
import csv
import os
import re
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Edge:
    src: str
    rel: str
    dst: str


def read_edges(path: str) -> List[Edge]:
    edges: List[Edge] = []
    bad_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                # fallback: split on any whitespace
                parts = re.split(r"\s+", line)

            if len(parts) != 3:
                bad_lines += 1
                continue

            edges.append(Edge(parts[0], parts[1], parts[2]))

    if bad_lines:
        print(f"[warn] Skipped {bad_lines} malformed lines from {path}")

    return edges


def unordered_pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def compute_pair_counts(edges: Iterable[Edge]) -> Counter:
    """Count how many edges exist per unordered endpoint pair."""
    c = Counter()
    for e in edges:
        c[unordered_pair_key(e.src, e.dst)] += 1
    return c


def nonredundant_edges(edges: List[Edge], pair_counts: Counter) -> List[Edge]:
    """Edges whose unordered endpoint pair appears exactly once in the graph."""
    return [e for e in edges if pair_counts[unordered_pair_key(e.src, e.dst)] == 1]


def hub_to_incoming_nonredundant(nonred: Iterable[Edge]) -> Dict[str, List[Edge]]:
    """Map: hub node (tail) -> list of incoming non-redundant edges."""
    m: Dict[str, List[Edge]] = defaultdict(list)
    for e in nonred:
        m[e.dst].append(e)
    return m


def compute_hub_stats(hub_map: Dict[str, List[Edge]]) -> List[Tuple[str, int, int]]:
    """
    Return list of (hub, incoming_nonred_count, unique_sources_count),
    sorted descending by count.
    """
    stats: List[Tuple[str, int, int]] = []
    for hub, es in hub_map.items():
        sources = {e.src for e in es}
        stats.append((hub, len(es), len(sources)))

    stats.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
    return stats


def edges_between_hubs(edges: Iterable[Edge], hubs: set[str]) -> Dict[Tuple[str, str], List[Edge]]:
    """Group all hub-hub edges by unordered hub pair."""
    pair_to_edges: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
    for e in edges:
        if e.src in hubs and e.dst in hubs:
            pair_to_edges[unordered_pair_key(e.src, e.dst)].append(e)
    return pair_to_edges


def write_tsv(path: str, header: List[str], rows: Iterable[Tuple]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Path to edge list file (TSV: head<TAB>rel<TAB>tail)")
    ap.add_argument("--top-hubs", type=int, default=5, help="How many hubs to keep (default: 10)")
    ap.add_argument("--min-incoming-nonred", type=int, default=None,
                    help="Optional: only keep hubs with >= this many incoming non-redundant edges")
    ap.add_argument("--out-dir", default="hub_analysis_out", help="Output directory")
    args = ap.parse_args()

    edges = read_edges(args.input)
    pair_counts = compute_pair_counts(edges)
    nonred = nonredundant_edges(edges, pair_counts)

    hub_map = hub_to_incoming_nonredundant(nonred)
    stats = compute_hub_stats(hub_map)

    if args.min_incoming_nonred is not None:
        stats = [s for s in stats if s[1] >= args.min_incoming_nonred]

    stats = stats[: args.top_hubs]
    hubs = {hub for hub, _, _ in stats}

    pair_to_edges = edges_between_hubs(edges, hubs)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) hub summary
    write_tsv(
        os.path.join(args.out_dir, "hub_summary.tsv"),
        ["hub", "incoming_nonredundant_edges", "unique_sources"],
        stats,
    )

    # 2) incoming non-redundant edges for each hub
    hub_edge_rows = []
    for hub, _, _ in stats:
        for e in hub_map.get(hub, []):
            hub_edge_rows.append((hub, e.src, e.rel, e.dst))

    write_tsv(
        os.path.join(args.out_dir, "hub_incoming_nonredundant_edges.tsv"),
        ["hub", "src", "rel", "dst"],
        hub_edge_rows,
    )

    # 3) hub pair counts + 4) hub pair edge list
    pair_count_rows = []
    pair_edge_rows = []
    for a, b in combinations(sorted(hubs), 2):
        k = (a, b)
        es = pair_to_edges.get(k, [])
        pair_count_rows.append((a, b, len(es)))
        for e in es:
            pair_edge_rows.append((a, b, e.src, e.rel, e.dst))

    pair_count_rows.sort(key=lambda x: (x[2], x[0], x[1]))

    write_tsv(
        os.path.join(args.out_dir, "hub_pair_counts.tsv"),
        ["hub1", "hub2", "edge_count"],
        pair_count_rows,
    )
    write_tsv(
        os.path.join(args.out_dir, "hub_pair_edges.tsv"),
        ["hub1", "hub2", "src", "rel", "dst"],
        pair_edge_rows,
    )

    print("Top hubs (incoming non-redundant):")
    for hub, n, u in stats:
        print(f"  {hub}\t{n} edges\t{u} unique sources")

    print(f"\nWrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
