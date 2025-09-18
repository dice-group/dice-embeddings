# Requires: networkx as nx, torch, torch.nn.functional as F

from typing import List, Tuple, Dict, Set, Optional
import networkx as nx
import torch
import torch.nn.functional as F
from collections import defaultdict

Triple = Tuple[str, str, str]

def build_entity_digraph(triples: List[Triple]) -> nx.DiGraph:
    G = nx.DiGraph()
    for h, _, t in triples:
        G.add_edge(h, t)
    return G

def _existing_sets(triples: List[Triple]):
    triple_set = set(triples)
    ht_set = {(h, t) for h, _, t in triples}
    return triple_set, ht_set

def _rank_nodes(centrality: Dict[str, float], top_k: Optional[int]) -> List[str]:
    items = sorted(centrality.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return [n for n, _ in (items if top_k is None else items[:top_k])]

def _relation_domains_ranges(triples: List[Triple]):
    heads_by_rel = defaultdict(set)
    tails_by_rel = defaultdict(set)
    for h, r, t in triples:
        heads_by_rel[r].add(h)
        tails_by_rel[r].add(t)
    return heads_by_rel, tails_by_rel

def propose_candidates_centrality(
    triples: List[Triple],
    node_centrality: Dict[str, float],
    *,
    mode: str = "both",                  # "head", "tail", "both"
    top_k_nodes: int = 100,
    avoid_existing_edge: bool = False,
    restrict_by_relation: bool = False,
    forbidden: Optional[Set[Triple]] = None
) -> List[Triple]:
    """Generate candidate corrupted triples (no model scoring yet)."""
    forbidden = forbidden or set()
    triple_set, ht_set = _existing_sets(triples)
    top_nodes = _rank_nodes(node_centrality, top_k_nodes)
    heads_by_rel, tails_by_rel = _relation_domains_ranges(triples)

    candidates = []
    seen = set()

    for h, r, t in triples:
        # Precompute allowed sets if schema restriction is on
        allowed_heads = heads_by_rel[r] if restrict_by_relation else top_nodes
        allowed_tails = tails_by_rel[r] if restrict_by_relation else top_nodes

        if mode in ("head", "both"):
            for n in top_nodes:
                if restrict_by_relation and (n not in allowed_heads):
                    continue
                if n == h or n == t:
                    continue
                cand = (n, r, t)
                if cand in triple_set or cand in forbidden or cand in seen:
                    continue
                if avoid_existing_edge and (n, t) in ht_set:
                    continue
                candidates.append(cand)
                seen.add(cand)

        if mode in ("tail", "both"):
            for n in top_nodes:
                if restrict_by_relation and (n not in allowed_tails):
                    continue
                if n == h or n == t:
                    continue
                cand = (h, r, n)
                if cand in triple_set or cand in forbidden or cand in seen:
                    continue
                if avoid_existing_edge and (h, n) in ht_set:
                    continue
                candidates.append(cand)
                seen.add(cand)

    return candidates

def add_corrupted_by_centrality_and_loss(
    triples: List[Triple],
    *,
    oracle,
    budget: int,
    centrality: str = "betweenness",     # "betweenness" | "closeness" | "harmonic"
    undirected: bool = False,
    mode: str = "both",
    top_k_nodes: int = 100,
    avoid_existing_edge: bool = False,
    restrict_by_relation: bool = False,
    forbidden: Optional[Set[Triple]] = None,
    batch_size: int = 65536
) -> List[Triple]:
    """
    Centrality-seeded, MODEL-AWARE poisoning:
    - Propose head/tail corruptions anchored on high-centrality nodes.
    - Score each candidate by positive BCE loss L(z, y=1) under the current model.
    - Return the top-`budget` candidates (to be added as *positives*).
    """
    # 1) Centrality on entity graph
    Gd = build_entity_digraph(triples)
    G = Gd.to_undirected(as_view=True) if undirected else Gd
    if centrality == "betweenness":
        node_cent = nx.betweenness_centrality(G, normalized=True)
    elif centrality == "closeness":
        node_cent = nx.closeness_centrality(G)
    elif centrality == "harmonic":
        node_cent = nx.harmonic_centrality(G)
    else:
        raise ValueError("centrality must be one of {'betweenness','closeness','harmonic'}")

    # 2) Propose candidates structurally
    cands = propose_candidates_centrality(
        triples,
        node_cent,
        mode=mode,
        top_k_nodes=top_k_nodes,
        avoid_existing_edge=avoid_existing_edge,
        restrict_by_relation=restrict_by_relation,
        forbidden=forbidden,
    )
    if not cands:
        return []

    # 3) Score candidates by *positive* training loss (y=1)
    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    device = next(oracle.model.parameters()).device

    # Build index tensor
    idx = torch.empty((len(cands), 3), dtype=torch.long)
    for i, (h, r, t) in enumerate(cands):
        idx[i, 0] = E2I[h]
        idx[i, 1] = R2I[r]
        idx[i, 2] = E2I[t]

    # Batched forward for memory safety
    logits_list = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            batch = idx[s:s+batch_size].to(device)
            z = oracle.model.forward_triples(batch).reshape(-1)
            logits_list.append(z.cpu())
    logits = torch.cat(logits_list, dim=0)
    y_pos = torch.ones_like(logits)
    loss_pos = F.binary_cross_entropy_with_logits(logits, y_pos, reduction="none")

    # 4) Pick top-`budget` by highest positive loss (largest gradient pull)
    budget = max(0, min(budget, len(cands)))
    topk = torch.topk(loss_pos, k=budget, largest=True)
    selected = [cands[i] for i in topk.indices.tolist()]
    return selected
