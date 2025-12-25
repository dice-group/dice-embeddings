import os
from collections import defaultdict, Counter
import math
import networkx as nx

##############################
# CONFIG
##############################

TRAIN_FILE = "./KGs/UMLS/train.txt"

REMOVED_FILES = [
    "train.pruned_top2_poolnonredundant_frac0.3_seed123.removed.txt",
    "train.pruned_top2_poolnonredundant_frac0.4_seed123.removed.txt",
    "train.pruned_top2_poolnonredundant_frac0.5_seed123.removed.txt",
    "train.pruned_top2_poolnonredundant_frac0.6_seed123.removed.txt",
    "train.pruned_top2_poolnonredundant_frac0.7_seed123.removed.txt",
    "train.pruned_top2_poolnonredundant_frac0.8_seed123.removed.txt",
    "train.pruned_top2_poolnonredundant_frac0.9_seed123.removed.txt",
]

HUB_NODES = ["biomedical_occupation_or_discipline", "occupation_or_discipline"]
MAX_BACKUP_PATH_LEN = 3   # k in your earlier definition


##############################
# IO HELPERS
##############################

def read_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # skip garbage
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def build_digraph(triples):
    """
    Directed multigraph collapsed to simple DiGraph:
    edge (h, t) labelled with relation r in edge data.
    For structural stuff we will mostly ignore labels.
    """
    G = nx.DiGraph()
    for h, r, t in triples:
        if not G.has_edge(h, t):
            G.add_edge(h, t, rels=set())
        G[h][t]["rels"].add(r)
    return G


##############################
# METRICS
##############################

def compute_hub_incident_edges(triples, hubs):
    """
    From original triples, extract all edges incident to hub nodes.
    Returns:
        hub_incident_edges[hub] = list of triples (h, r, t)
    """
    hub_incident = {h: [] for h in hubs}
    for h, r, t in triples:
        if h in hubs:
            hub_incident[h].append((h, r, t))
        if t in hubs and t != h:
            hub_incident[t].append((h, r, t))
    return hub_incident


def path_exists_within_k(UG, src, dst, k):
    """
    Check if there's a path of length <= k between src and dst
    in an *undirected* graph UG.
    """
    if src not in UG or dst not in UG:
        return False
    if src == dst:
        return True
    # BFS limited to depth k
    lengths = nx.single_source_shortest_path_length(UG, src, cutoff=k)
    return dst in lengths


def backup_stats_for_removed_edges(
    original_hub_edges, removed_triples_set, UG_pruned, k
):
    """
    Among hub-incident edges that are removed,
    compute how many still have a backup path (<= k) between endpoints
    in the pruned graph.
    """
    stats = {}
    for hub, triples in original_hub_edges.items():
        removed_incident = []
        for (h, r, t) in triples:
            if (h, r, t) in removed_triples_set:
                removed_incident.append((h, r, t))

        total_removed = len(removed_incident)
        with_backup = 0
        without_backup = 0

        for (h, r, t) in removed_incident:
            src, dst = h, t
            # ignore direction for backup
            if path_exists_within_k(UG_pruned, src, dst, k):
                with_backup += 1
            else:
                without_backup += 1

        stats[hub] = {
            "removed_incident_edges": total_removed,
            "removed_with_backup": with_backup,
            "removed_without_backup": without_backup,
        }
    return stats


def hub_centrality_and_connectedness(G, hubs):
    """
    Basic degree/centrality per hub in pruned graph.
    Uses undirected degree for 'connectedness' and total-degree.
    """
    UG = G.to_undirected()
    res = {}
    for hub in hubs:
        if hub not in G:
            res[hub] = {
                "in_deg": 0,
                "out_deg": 0,
                "total_deg": 0,
                "num_neighbors": 0,
                "component_size": 0,
            }
            continue

        in_deg = G.in_degree(hub)
        out_deg = G.out_degree(hub)
        total_deg = in_deg + out_deg

        neighbors = set(G.predecessors(hub)) | set(G.successors(hub))
        num_neighbors = len(neighbors)

        # connected component size in undirected view
        component_size = len(nx.node_connected_component(UG, hub))

        res[hub] = {
            "in_deg": in_deg,
            "out_deg": out_deg,
            "total_deg": total_deg,
            "num_neighbors": num_neighbors,
            "component_size": component_size,
        }
    return res


def neighbor_degree_summary(G, original_triples, hubs):
    """
    For each hub, take all entities adjacent to that hub in the *original* graph,
    then measure their degree distribution in the pruned graph.

    Returns per hub:
        - num_neighbors
        - avg_deg
        - median_deg
        - num_deg1_or_less
    """
    # neighbors in original
    orig_neighbors = {h: set() for h in hubs}
    for h, r, t in original_triples:
        if h in hubs and t != h:
            orig_neighbors[h].add(t)
        if t in hubs and h != t:
            orig_neighbors[t].add(h)

    UG = G.to_undirected()
    summaries = {}
    for hub in hubs:
        neighs = [n for n in orig_neighbors[hub] if n in UG]
        if not neighs:
            summaries[hub] = {
                "num_neighbors_original": len(orig_neighbors[hub]),
                "num_neighbors_present": 0,
                "avg_deg": 0.0,
                "median_deg": 0.0,
                "num_deg1_or_less": 0,
            }
            continue

        degs = [UG.degree(n) for n in neighs]
        degs_sorted = sorted(degs)
        n = len(degs)
        if n % 2 == 1:
            median = degs_sorted[n // 2]
        else:
            median = 0.5 * (degs_sorted[n // 2 - 1] + degs_sorted[n // 2])

        summaries[hub] = {
            "num_neighbors_original": len(orig_neighbors[hub]),
            "num_neighbors_present": len(neighs),
            "avg_deg": sum(degs) / n,
            "median_deg": median,
            "num_deg1_or_less": sum(d <= 1 for d in degs),
        }

    return summaries


##############################
# MAIN ANALYSIS PIPELINE
##############################

def analyze_all():
    # 1. Load original graph
    print("Loading original graph from", TRAIN_FILE)
    orig_triples = read_triples(TRAIN_FILE)
    G_orig = build_digraph(orig_triples)

    # Precompute original hub-incident edges
    original_hub_edges = compute_hub_incident_edges(orig_triples, HUB_NODES)

    per_file_metrics = {}

    for path in REMOVED_FILES:
        if not os.path.exists(path):
            print(f"WARNING: file not found: {path}")
            continue

        label = os.path.basename(path)
        print("\nProcessing removed-file:", label)

        # 2. Read removed triples
        removed_triples = read_triples(path)
        removed_set = set(removed_triples)

        # 3. Build pruned graph = original - removed
        pruned_triples = [tr for tr in orig_triples if tr not in removed_set]
        G_pruned = build_digraph(pruned_triples)
        UG_pruned = G_pruned.to_undirected()

        # 4. Hub centrality / connectedness in pruned graph
        hub_deg_stats = hub_centrality_and_connectedness(G_pruned, HUB_NODES)

        # 5. Backup stats for *removed* hub-incident edges
        backup_stats = backup_stats_for_removed_edges(
            original_hub_edges, removed_set, UG_pruned, MAX_BACKUP_PATH_LEN
        )

        # 6. Neighbor degree summaries (neighbors defined by original graph)
        neigh_stats = neighbor_degree_summary(G_pruned, orig_triples, HUB_NODES)

        per_file_metrics[label] = {
            "hub_degrees": hub_deg_stats,
            "backup_stats": backup_stats,
            "neighbor_degrees": neigh_stats,
        }

    # 7. Print comparison with frac0.5 as reference
    ref_label = "train.pruned_top2_poolnonredundant_frac0.5_seed123.removed.txt"
    if ref_label not in per_file_metrics:
        print(f"\nReference file {ref_label} not found in metrics, skipping comparison.")
        return

    print("\n================ COMPARISON (relative to frac0.5) ================")
    ref = per_file_metrics[ref_label]

    def print_diff_block(metric_name, key_path):
        print(f"\n--- {metric_name} ---")
        for hub in HUB_NODES:
            print(f"\nHub: {hub}")
            ref_val = ref[key_path][hub]
            for label, m in per_file_metrics.items():
                val = m[key_path][hub]
                print(f"  {label}:")
                for k2 in sorted(ref_val.keys()):
                    ref_num = ref_val[k2]
                    cur_num = val[k2]
                    diff = cur_num - ref_num
                    print(f"    {k2}: {cur_num} (Δ vs 0.5: {diff})")

    # Centrality / connectedness differences
    print_diff_block("Hub degree & connectedness", "hub_degrees")

    # Backup statistics differences
    print_diff_block("Backup stats (removed hub edges)", "backup_stats")

    # Neighbor degree distribution summaries
    print_diff_block("Neighbor degree summaries", "neighbor_degrees")


if __name__ == "__main__":
    analyze_all()
