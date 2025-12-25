#!/usr/bin/env python3
"""
Compare removed-edge sets around two hub nodes, using the original graph.

Focus:
- Only triples that connect to one of the hubs:
    'biomedical_occupation_or_discipline'
    'occupation_or_discipline'

For each removed-file F and each hub H, we compute (in the ORIGINAL graph):

- num_edges_removed_to_hub:
    number of removed triples touching H

- num_unique_pairs_removed_to_hub:
    among those removed triples, how many have endpoint pair {hub, other}
    that appears EXACTLY ONCE in the original graph (undirected, any relation)

- avg_backup_count_to_hub:
    average count of edges between hub and other in the original graph
    (again, undirected, across all relations)

- avg_deg_removed_neighbors:
    average total degree (incident edges) of the neighbor entities (other node)
    in the original graph

- avg_deg_removed_neighbors_norm:
    degree centrality approximation: deg / (2 * |E|)

Then we compare all files to a chosen baseline removed-file by subtracting
metrics: metrics[file] - metrics[baseline], per hub.

Outputs (TSVs):

1) metrics_per_file.tsv:
    file, hub, num_edges_removed, num_unique_pairs_removed,
    avg_backup_count_to_hub, avg_deg_removed_neighbors, avg_deg_removed_neighbors_norm

2) metrics_diff_vs_baseline.tsv:
    same columns but each value is (file_metric - baseline_metric)

Assumes all input files are whitespace- or tab-separated triples:
    head  relation  tail
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Tuple


Triple = Tuple[str, str, str]
Pair = Tuple[str, str]


HUBS_DEFAULT = [
    "biomedical_occupation_or_discipline",
    "occupation_or_discipline",
]


@dataclass
class NodeStats:
    deg_total: int = 0
    deg_in: int = 0
    deg_out: int = 0


def undirected_pair(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


def load_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
    return triples


def compute_graph_stats(triples: Iterable[Triple]):
    """
    From the ORIGINAL graph:

    - node_stats[node].deg_total / deg_in / deg_out
    - pair_counts[(u,v)] = number of edges between u and v, undirected,
      across all relations and directions
    - num_edges_total
    """
    node_stats: Dict[str, NodeStats] = defaultdict(NodeStats)
    pair_counts: Counter = Counter()
    num_edges = 0

    for h, r, t in triples:
        num_edges += 1
        node_stats[h].deg_out += 1
        node_stats[t].deg_in += 1
        node_stats[h].deg_total += 1
        node_stats[t].deg_total += 1
        pair_counts[undirected_pair(h, t)] += 1

    return node_stats, pair_counts, num_edges


def gather_removed_neighbors(
    removed_triples: Iterable[Triple],
    hubs: List[str],
):
    """
    From a removed-file, collect for each hub:

    - edges_to_hub[hub] : list of triples (h, r, t) that touch that hub
      (if both endpoints are hubs, that triple will be counted for BOTH hubs)

    - neighbor_nodes[hub] : set of 'other entity' nodes adjacent to hub
    """
    hub_set = set(hubs)
    edges_to_hub: Dict[str, List[Triple]] = {h: [] for h in hubs}
    neighbor_nodes: Dict[str, set] = {h: set() for h in hubs}

    for h, r, t in removed_triples:
        if h in hub_set and t in hub_set:
            # hub-hub edge, counted for both
            for hub in hubs:
                edges_to_hub[hub].append((h, r, t))
                other = t if hub == h else h
                neighbor_nodes[hub].add(other)
        elif h in hub_set:
            hub = h
            edges_to_hub[hub].append((h, r, t))
            neighbor_nodes[hub].add(t)
        elif t in hub_set:
            hub = t
            edges_to_hub[hub].append((h, r, t))
            neighbor_nodes[hub].add(h)
        # else: triple doesn't touch a hub, ignore

    return edges_to_hub, neighbor_nodes


def compute_metrics_for_removed_file(
    removed_path: str,
    hubs: List[str],
    node_stats: Dict[str, NodeStats],
    pair_counts: Counter,
    num_edges_total: int,
):
    """
    For this removed-file, return:
      metrics[hub] = dict of scalars
    """
    removed_triples = load_triples(removed_path)
    edges_to_hub, neighbor_nodes = gather_removed_neighbors(removed_triples, hubs)

    metrics = {}

    for hub in hubs:
        hub_edges = edges_to_hub[hub]
        neighbors = neighbor_nodes[hub]

        num_edges_removed = len(hub_edges)

        # backup counts PER TRIPLE (hub-other pair)
        backup_counts: List[int] = []
        num_unique_pairs_removed = 0

        for h, r, t in hub_edges:
            # find other endpoint for this hub
            if h == hub and t != hub:
                other = t
            elif t == hub and h != hub:
                other = h
            else:
                # hub-hub edge: choose the "other" hub as neighbor
                other = t if h == hub else h

            pc = pair_counts[undirected_pair(hub, other)]
            backup_counts.append(pc)
            if pc == 1:
                num_unique_pairs_removed += 1

        avg_backup = sum(backup_counts) / len(backup_counts) if backup_counts else 0.0

        # neighbor degree stats
        degs = [node_stats[n].deg_total for n in neighbors if n in node_stats]
        if degs:
            avg_deg = sum(degs) / len(degs)
            # crude degree centrality
            avg_deg_norm = avg_deg / (2.0 * num_edges_total)
        else:
            avg_deg = 0.0
            avg_deg_norm = 0.0

        metrics[hub] = {
            "num_edges_removed": float(num_edges_removed),
            "num_unique_pairs_removed": float(num_unique_pairs_removed),
            "avg_backup_count_to_hub": float(avg_backup),
            "avg_deg_removed_neighbors": float(avg_deg),
            "avg_deg_removed_neighbors_norm": float(avg_deg_norm),
        }

    return metrics


def write_metrics_tsv(path: str, metrics_per_file: Dict[str, Dict[str, Dict[str, float]]], hubs: List[str]):
    """
    metrics_per_file[file][hub][metric_name] -> value
    """
    metric_names = [
        "num_edges_removed",
        "num_unique_pairs_removed",
        "avg_backup_count_to_hub",
        "avg_deg_removed_neighbors",
        "avg_deg_removed_neighbors_norm",
    ]

    with open(path, "w", encoding="utf-8") as f:
        header = ["file", "hub"] + metric_names
        f.write("\t".join(header) + "\n")
        for fname in sorted(metrics_per_file.keys()):
            for hub in hubs:
                vals = metrics_per_file[fname][hub]
                row = [fname, hub] + [f"{vals[m]:.6g}" for m in metric_names]
                f.write("\t".join(row) + "\n")


def write_diff_vs_baseline_tsv(
    path: str,
    metrics_per_file: Dict[str, Dict[str, Dict[str, float]]],
    baseline: str,
    hubs: List[str],
):
    metric_names = [
        "num_edges_removed",
        "num_unique_pairs_removed",
        "avg_backup_count_to_hub",
        "avg_deg_removed_neighbors",
        "avg_deg_removed_neighbors_norm",
    ]

    base = metrics_per_file[baseline]

    with open(path, "w", encoding="utf-8") as f:
        header = ["file", "hub"] + [f"delta_{m}" for m in metric_names]
        f.write("\t".join(header) + "\n")

        for fname in sorted(metrics_per_file.keys()):
            if fname == baseline:
                continue
            for hub in hubs:
                vals = metrics_per_file[fname][hub]
                base_vals = base[hub]
                deltas = [vals[m] - base_vals[m] for m in metric_names]
                row = [fname, hub] + [f"{d:.6g}" for d in deltas]
                f.write("\t".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Compare removed triples around two hubs using degree / backup statistics."
    )
    ap.add_argument("--train", required=True, help="Original graph file (train.txt).")
    ap.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline removed file, e.g. train.pruned_top2_poolnonredundant_frac0.5_seed123.removed.txt",
    )
    ap.add_argument(
        "--removed",
        nargs="+",
        required=True,
        help="Other removed files to compare against the baseline.",
    )
    ap.add_argument(
        "--hubs",
        nargs="*",
        default=HUBS_DEFAULT,
        help="Hub nodes to focus on (default: biomedical_occupation_or_discipline, occupation_or_discipline)",
    )
    ap.add_argument(
        "--out-prefix",
        default="removed_comparison",
        help="Prefix for output TSVs (default: removed_comparison.*.tsv)",
    )
    args = ap.parse_args()

    hubs = args.hubs

    # Load original graph and precompute stats
    print(f"Loading original graph from {args.train} ...")
    train_triples = load_triples(args.train)
    node_stats, pair_counts, num_edges_total = compute_graph_stats(train_triples)
    print(f"Original graph: {num_edges_total} edges, {len(node_stats)} nodes.")

    # Collect metrics for each removed file (including baseline)
    files = [args.baseline] + args.removed
    metrics_per_file: Dict[str, Dict[str, Dict[str, float]]] = {}

    for path in files:
        print(f"Computing metrics for removed file: {path}")
        metrics = compute_metrics_for_removed_file(
            removed_path=path,
            hubs=hubs,
            node_stats=node_stats,
            pair_counts=pair_counts,
            num_edges_total=num_edges_total,
        )
        metrics_per_file[os.path.basename(path)] = metrics

    # Write raw metrics
    metrics_tsv = args.out_prefix + ".metrics_per_file.tsv"
    write_metrics_tsv(metrics_tsv, metrics_per_file, hubs)
    print(f"Wrote per-file metrics to: {metrics_tsv}")

    # Write diffs vs baseline
    diff_tsv = args.out_prefix + ".metrics_diff_vs_baseline.tsv"
    write_diff_vs_baseline_tsv(
        diff_tsv,
        metrics_per_file,
        baseline=os.path.basename(args.baseline),
        hubs=hubs,
    )
    print(f"Wrote differences vs baseline to: {diff_tsv}")


if __name__ == "__main__":
    main()
