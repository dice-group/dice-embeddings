from typing import List, Tuple, Dict, Optional, Set, Literal
from collections import defaultdict
import networkx as nx
import torch
import torch.nn.functional as F
from typing import Literal
import torch.nn as nn

Triple = Tuple[str, str, str]

def _resolve_embeddings_or_raise(model, num_entities, num_relations):
    """
    Try to obtain the entity/relation embedding modules from the model.
    First look for `model.entity_embeddings` / `model.relation_embeddings`,
    otherwise search for nn.Embedding modules that match table sizes.
    """
    ent_emb = getattr(model, "entity_embeddings", None)
    rel_emb = getattr(model, "relation_embeddings", None)

    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        # heuristic search
        ent_emb = None
        rel_emb = None
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Embedding):
                if mod.num_embeddings == num_entities and ent_emb is None:
                    ent_emb = mod
                elif mod.num_embeddings == num_relations and rel_emb is None:
                    rel_emb = mod
        if ent_emb is None or rel_emb is None:
            raise AttributeError(
                "Could not locate entity/relation embeddings on the model. "
                "Expected attributes `entity_embeddings` and `relation_embeddings`, "
                "or nn.Embedding modules with sizes matching (|E|, |R|)."
            )
    return ent_emb, rel_emb

@torch.no_grad()
def _norm_vec(x, p = "l2"):
    if p == "l2":
        return float(torch.linalg.vector_norm(x).item())
    elif p == "linf":
        return float(torch.max(torch.abs(x)).item())
    else:
        raise ValueError("p must be 'l2' or 'linf'")


def remove_by_gradient_influence_forward(
    triples,
    *,
    model,                                  
    entity_to_idx,
    relation_to_idx,
    budget,
    label = 1.0,                     
    p = "l2",         
    device = None,
    show_progress = False,            
    progress_every = 10000,
):

    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device)
    model.train(False)  

    e2i, r2i = entity_to_idx, relation_to_idx
    num_entities, num_relations = len(e2i), len(r2i)

    ent_emb, rel_emb = _resolve_embeddings_or_raise(model, num_entities, num_relations)

    for pmt in model.parameters():
        pmt.requires_grad_(False)
    ent_emb.weight.requires_grad_(True)
    rel_emb.weight.requires_grad_(True)

    y = torch.tensor(label, dtype=torch.float32, device=device).view(())

    scores: List[Tuple[float, int]] = [] 

    for i, (h, r, t) in enumerate(triples):
        try:
            hi, ri, ti = e2i[h], r2i[r], e2i[t]
        except KeyError as e:
            raise KeyError(f"Triple contains OOV label at index {i}: {(h,r,t)} :: {e}")

        model.zero_grad(set_to_none=True)

        idx = torch.tensor([[hi, ri, ti]], dtype=torch.long, device=device)
        logit = model.forward_triples(idx).view(())  # scalar

        loss = F.binary_cross_entropy_with_logits(logit, y)
        loss.backward()

        GE = ent_emb.weight.grad 
        GR = rel_emb.weight.grad 

        if GE is None or GR is None:
            raise RuntimeError("Embedding gradients are None; ensure embeddings require_grad is True.")

        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h = GE[hi]
        g_r = GR[ri]
        g_t = GE[ti]

        g_cat = torch.cat([g_h.flatten(), g_r.flatten(), g_t.flatten()])
        gnorm = _norm_vec(g_cat, p=p)

        scores.append((gnorm, i))

        if show_progress and (i % progress_every == 0) and i > 0:
            print(f"[grad-influence] processed {i}/{len(triples)}")

    scores.sort(key=lambda x: x[0], reverse=True)
    pick = [i for _, i in scores[:budget]]

    removed = [triples[i] for i in pick]
    keep_mask = [True] * len(triples)
    for i in pick: keep_mask[i] = False
    kept = [t for i, t in enumerate(triples) if keep_mask[i]]

    ent_emb.weight.requires_grad_(False)
    rel_emb.weight.requires_grad_(False)

    return removed, kept


def triples_to_idx_with_maps(
    triples,
    entity_to_idx,
    relation_to_idx,
):

    idx = torch.empty((len(triples), 3), dtype=torch.long)
    for i, (h, r, t) in enumerate(triples):
        try:
            idx[i, 0] = entity_to_idx[h]
            idx[i, 1] = relation_to_idx[r]
            idx[i, 2] = entity_to_idx[t]
        except KeyError as e:
            raise KeyError(f"Label not found in model maps while indexing {triples[i]}: {e}")
    return idx

def _build_entity_digraph(triples: List[Triple]) -> nx.DiGraph:
    G = nx.DiGraph()
    for h, _, t in triples:
        G.add_edge(h, t)
    return G

def score_triples(cur_triples: List[Triple]):
        Gd = _build_entity_digraph(cur_triples)
        G = Gd.to_undirected(as_view=True) if undirected else Gd

        if metric == "edge_betweenness":
            edge_bw = nx.edge_betweenness_centrality(G, k=approx_k, normalized=True)
            scores = []
            for i, (h, r, t) in enumerate(cur_triples):
                s = edge_bw.get((h, t), 0.0)
                scores.append((s, i))
        elif metric == "endpoint_harmonic":
            node_c = nx.harmonic_centrality(G)
            scores = [(0.5 * (node_c.get(h, 0.0) + node_c.get(t, 0.0)), i)
                      for i, (h, r, t) in enumerate(cur_triples)]
        elif metric == "endpoint_closeness":
            node_c = nx.closeness_centrality(G)
            scores = [(0.5 * (node_c.get(h, 0.0) + node_c.get(t, 0.0)), i)
                      for i, (h, r, t) in enumerate(cur_triples)]
        else:
            raise ValueError("metric must be one of {'edge_betweenness','endpoint_harmonic','endpoint_closeness'}")

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores

def remove_by_simple_centrality(
    triples,
    budget,
    *,
    metric,
    undirected,
    approx_k,
    recompute,
):

    remaining = list(triples)
    removed: List[Triple] = []

    if not recompute:
        scores = score_triples(remaining)
        pick_idx = [i for _, i in scores[:budget]]
        removed = [remaining[i] for i in pick_idx]
        keep_mask = [True] * len(remaining)
        for i in pick_idx: keep_mask[i] = False
        kept = [t for i, t in enumerate(remaining) if keep_mask[i]]
        return removed, kept

    for _ in range(budget):
        if not remaining:
            break
        scores = score_triples(remaining)
        _, i = scores[0]
        removed.append(remaining.pop(i))
    return removed, remaining

@torch.no_grad()
def remove_by_centrality_plus_loss_forward(
    triples: List[Triple],
    *,
    model,                                  
    entity_to_idx,
    relation_to_idx,
    budget,
    centrality,
    undirected,
    mode = "endpoint",  
    alpha = 1.0,                
    batch_size = 10000,
    device = None,
):

    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device).eval()

    Gd = _build_entity_digraph(triples)
    G = Gd.to_undirected(as_view=True) if undirected else Gd

    if mode == "edge":
        edge_c = nx.edge_betweenness_centrality(G, normalized=True)
        cent_vals = [edge_c.get((h, t), 0.0) for (h, _, t) in triples]
    elif centrality == "harmonic":
        node_c = nx.harmonic_centrality(G)
        cent_vals = [0.5 * (node_c.get(h, 0.0) + node_c.get(t, 0.0)) for (h, _, t) in triples]
    elif centrality == "closeness":
        node_c = nx.closeness_centrality(G)
        cent_vals = [0.5 * (node_c.get(h, 0.0) + node_c.get(t, 0.0)) for (h, _, t) in triples]
    else:
        raise ValueError("centrality must be 'harmonic' or 'closeness' when mode='endpoint'")

    c = torch.tensor(cent_vals, dtype=torch.float32)
    cmin, cmax = float(torch.min(c)), float(torch.max(c))
    if cmax > cmin:
        c_norm = (c - cmin) / (cmax - cmin)
    else:
        c_norm = torch.zeros_like(c)

    idx = triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx).to(device)
    logits_list = []
    for s in range(0, idx.size(0), batch_size):
        z = model.forward_triples(idx[s:s+batch_size]).reshape(-1)
        logits_list.append(z.detach().to("cpu"))
    logits = torch.cat(logits_list, dim=0)
    prob = torch.sigmoid(logits)   

    score = (c_norm ** alpha) * prob  
    topk = min(budget, len(triples))
    pick_idx = torch.topk(score, k=topk, largest=True).indices.tolist()

    removed = [triples[i] for i in pick_idx]
    keep_mask = [True] * len(triples)
    for i in pick_idx: keep_mask[i] = False
    kept = [t for i, t in enumerate(triples) if keep_mask[i]]
    return removed, kept

@torch.no_grad()
def remove_by_global_argmax_forward(
    triples,
    *,
    model,                                 
    entity_to_idx,
    relation_to_idx,
    budget,
    criterion = "low_loss",
    batch_size = 10000,
    device = None,
):

    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device).eval()

    idx = triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx).to(device)
    logits_list = []
    for s in range(0, idx.size(0), batch_size):
        z = model.forward_triples(idx[s:s+batch_size]).reshape(-1)
        logits_list.append(z.detach().to("cpu"))
    logits = torch.cat(logits_list, dim=0)

    if criterion == "low_loss":
        scores = -F.softplus(-logits)  # higher score => lower loss
    elif criterion == "high_prob":
        scores = torch.sigmoid(logits) # higher score => higher prob
    else:
        raise ValueError("criterion must be 'low_loss' or 'high_prob'")

    topk = min(budget, len(triples))
    pick_idx = torch.topk(scores, k=topk, largest=True).indices.tolist()

    removed = [triples[i] for i in pick_idx]
    keep_mask = [True] * len(triples)
    for i in pick_idx: keep_mask[i] = False
    kept = [t for i, t in enumerate(triples) if keep_mask[i]]
    return removed, kept

def build_digraph(triples):
    G = nx.DiGraph()
    for h, r, t in triples:
        if not G.has_node(h): G.add_node(h)
        if not G.has_node(t): G.add_node(t)
        if not G.has_edge(h, t): G.add_edge(h, t)
    return G

def remove_by_endpoint_closeness(triples, budget, undirected=False):
    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    node_cl = nx.closeness_centrality(G)
    scored = []
    for i, (h, r, t) in enumerate(triples):
        s = 0.5 * (node_cl.get(h, 0.0) + node_cl.get(t, 0.0))
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for (_, _, _, _, i) in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]

def remove_by_edge_betweenness(triples, budget, approx_k=None):
    G = build_digraph(triples)
    edge_bw = nx.edge_betweenness_centrality(G, k=approx_k, normalized=True)
    scored = []
    for i, (h, r, t) in enumerate(triples):
        s = edge_bw.get((h, t), 0.0)
        scored.append((s, h, r, t, i))
    scored.sort(reverse=True)
    top_idx = [i for (_, _, _, _, i) in scored[:min(budget, len(scored))]]
    return [triples[i] for i in top_idx]