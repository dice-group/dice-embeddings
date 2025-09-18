import networkx as nx
from typing import List, Tuple, Hashable, Optional

Triple = Tuple[Hashable, Hashable, Hashable]

def select_remove_by_edge_betweenness(
    triples: List[Triple],
    budget: int,
    *,
    use_undirected: bool = False,
    treat_relations_as_parallel_edges: bool = True,
    recompute: bool = True,
    approx_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Triple], List[Triple]]:
    """
    Select up to `budget` triples to delete using edge betweenness centrality.
    Returns (removed_triples, remaining_triples).
    """
    budget = max(0, min(budget, len(triples)))
    if budget == 0:
        return [], list(triples)

    # Build a graph that preserves relation identity if requested
    if treat_relations_as_parallel_edges:
        G = nx.MultiDiGraph()
        # map from (u,v,key) -> index in `triples`
        edge2idx = {}
        for idx, (h, r, t) in enumerate(triples):
            # unique key so duplicates remain distinguishable
            key = (r, idx)
            G.add_edge(h, t, key=key)
            edge2idx[(h, t, key)] = idx
    else:
        G = nx.DiGraph()
        edge2idx = {}
        # If multiple triples map to same (h,t), keep the *last* index (arbitrary)
        # Alternative: store list and break ties later.
        for idx, (h, r, t) in enumerate(triples):
            G.add_edge(h, t)
            edge2idx[(h, t)] = idx

    removed_idx = []
    remaining_mask = [True] * len(triples)

    def centrality_dict():
        H = G.to_undirected(as_view=True) if use_undirected else G
        # For MultiGraphs, keys in the dict are (u,v,key); for simple graphs they are (u,v).
        return nx.edge_betweenness_centrality(H, k=approx_k, normalized=True, seed=seed)

    steps = budget if recompute else 1
    for step in range(steps):
        bw = centrality_dict()

        # Turn edge bw into a per-triple score
        scored = []
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            for (u, v, key), score in bw.items():
                idx = edge2idx.get((u, v, key))
                if idx is not None and remaining_mask[idx]:
                    scored.append((score, idx, (u, v, key)))
        else:
            for (u, v), score in bw.items():
                idx = edge2idx.get((u, v))
                if idx is not None and remaining_mask[idx]:
                    scored.append((score, idx, (u, v)))

        if not scored:
            break

        # If not recomputing, we need top-`budget` in one shot
        if not recompute:
            scored.sort(key=lambda x: x[0], reverse=True)
            picks = scored[:budget]
            for _, idx, edge_id in picks:
                removed_idx.append(idx)
            break

        # Greedy: remove one with the max score, then recompute
        score, idx, edge_id = max(scored, key=lambda x: x[0])
        removed_idx.append(idx)
        remaining_mask[idx] = False
        # Remove that specific edge from G
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            u, v, key = edge_id
            if G.has_edge(u, v, key):
                G.remove_edge(u, v, key)
        else:
            u, v = edge_id
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        if len(removed_idx) >= budget:
            break

    removed = [triples[i] for i in removed_idx]
    remaining = [t for i, t in enumerate(triples) if remaining_mask[i]]
    return removed, remaining


def select_remove_by_endpoint_centrality(
    triples: List[Triple],
    budget: int,
    *,
    use_undirected: bool = False,
    metric: str = "harmonic",  # or "closeness"
    recompute: bool = False,
) -> Tuple[List[Triple], List[Triple]]:
    """
    Score a triple by combining endpoint centralities and remove top `budget`.
    metric: "harmonic" or "closeness"
    Returns (removed_triples, remaining_triples).
    """
    budget = max(0, min(budget, len(triples)))
    if budget == 0:
        return [], list(triples)

    remaining = list(triples)
    removed: List[Triple] = []

    def centrality_scores(cur_triples):
        Gd = nx.DiGraph()
        Gd.add_nodes_from({h for h,_,t in cur_triples} | {t for _,_,t in cur_triples})
        Gd.add_edges_from([(h, t) for (h, _, t) in cur_triples])
        G = Gd.to_undirected(as_view=True) if use_undirected else Gd

        if metric == "harmonic":
            node_c = nx.harmonic_centrality(G)  # robust on disconnected graphs
        elif metric == "closeness":
            node_c = nx.closeness_centrality(G) # standard closeness
        else:
            raise ValueError("metric must be 'harmonic' or 'closeness'")

        # Combine endpoint scores; harmonic mean emphasizes low-connected endpoints
        def combine(a, b):
            # geometric-ish: avoid zeros dominating completely
            return (a + b) * 0.5

        scored = [(combine(node_c.get(h, 0.0), node_c.get(t, 0.0)), i)
                  for i, (h, _, t) in enumerate(cur_triples)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    if not recompute:
        scored = centrality_scores(remaining)
        to_remove_idx = [i for _, i in scored[:budget]]
        for i in sorted(to_remove_idx, reverse=True):
            removed.append(remaining.pop(i))
        return removed, remaining

    # Greedy recomputation (stronger)
    for _ in range(budget):
        if not remaining:
            break
        scored = centrality_scores(remaining)
        i = scored[0][1]
        removed.append(remaining.pop(i))
    return removed, remaining
