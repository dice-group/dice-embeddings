from typing import List, Tuple, Dict, Optional, Set, Literal
from collections import defaultdict
import networkx as nx
import torch
import torch.nn.functional as F
from typing import Literal
import torch.nn as nn
import pandas as pd
from pathlib import Path

Triple = Tuple[str, str, str]

import pandas as pd
import networkx as nx

def triples_and_scores_to_csv(
    triples,
    scores,
    out_csv_path="triples_with_scores_and_centrality.csv",
):

    if len(triples) != len(scores):
        raise ValueError(f"len(triples)={len(triples)} must equal len(scores)={len(scores)}")

    Gd = nx.DiGraph()
    Gu = nx.Graph()

    for (h, r, t) in triples:
        Gd.add_edge(h, t)
        Gu.add_edge(h, t)

    deg_dir_total = dict(Gd.degree())      # in+out
    deg_dir_in = dict(Gd.in_degree())
    deg_dir_out = dict(Gd.out_degree())
    deg_undir = dict(Gu.degree())

    ebc_dir = nx.edge_betweenness_centrality(Gd, normalized=False)
    ebc_undir_raw = nx.edge_betweenness_centrality(Gu, normalized=False)
    ebc_undir = {frozenset((u, v)): v for (u, v), v in ebc_undir_raw.items()}

    rows = []
    for (h, r, t), score in zip(triples, scores):
        h_deg_dir = deg_dir_total.get(h, 0)
        t_deg_dir = deg_dir_total.get(t, 0)

        node_deg_avg_dir = (h_deg_dir + t_deg_dir) / 2.0
        edge_degree_centrality_dir = (h_deg_dir + t_deg_dir - 2)   # "edge centrality" via endpoints
        edge_betweenness_dir = ebc_dir.get((h, t), 0.0)

        # Undirected
        h_deg_undir = deg_undir.get(h, 0)
        t_deg_undir = deg_undir.get(t, 0)

        node_deg_avg_undir = (h_deg_undir + t_deg_undir) / 2.0
        edge_degree_centrality_undir = (h_deg_undir + t_deg_undir - 2)
        edge_betweenness_undir = ebc_undir.get(frozenset((h, t)), 0.0)

        rows.append({
            "h": h, "r": r, "t": t,
            "score": float(score),

            # directed metrics
            "edge_betweenness_dir": edge_betweenness_dir,
            "edge_degree_centrality_dir": edge_degree_centrality_dir,
            "node_degree_avg_dir": node_deg_avg_dir,
            "h_in_degree": deg_dir_in.get(h, 0),
            "h_out_degree": deg_dir_out.get(h, 0),
            "t_in_degree": deg_dir_in.get(t, 0),
            "t_out_degree": deg_dir_out.get(t, 0),

            # undirected metrics
            "edge_betweenness_undir": edge_betweenness_undir,
            "edge_degree_centrality_undir": edge_degree_centrality_undir,
            "node_degree_avg_undir": node_deg_avg_undir,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    return df

def _resolve_embeddings_or_raise(model, num_entities, num_relations):
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

def canon(x):
    s = str(x)
    
    return s

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

    e2i = {canon(k): v for k, v in entity_to_idx.items()}
    r2i = {canon(k): v for k, v in relation_to_idx.items()}

    num_entities, num_relations = len(e2i), len(r2i)

    ent_emb, rel_emb = _resolve_embeddings_or_raise(model, num_entities, num_relations)

    for pmt in model.parameters():
        pmt.requires_grad_(False)
    ent_emb.weight.requires_grad_(True)
    rel_emb.weight.requires_grad_(True)

    y = torch.tensor(label, dtype=torch.float32, device=device).view(())

    scores: List[Tuple[float, int]] = [] 

    for i, (h, r, t) in enumerate(triples):
        h, r, t = canon(h), canon(r), canon(t)

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
            idx[i, 0] = entity_to_idx[str(h)]
            idx[i, 1] = relation_to_idx[str(r)]
            idx[i, 2] = entity_to_idx[str(t)]
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

import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_loss(loss, bins=50, title="Per-triple score distribution",
                   show_stats=True, save_path=None):

    if isinstance(loss, torch.Tensor):
        loss_np = loss.detach().cpu().numpy()
    else:
        loss_np = np.asarray(loss, dtype=np.float32)

    if show_stats:
        mean = float(loss_np.mean())
        std = float(loss_np.std())
        p10 = float(np.percentile(loss_np, 10))
        p50 = float(np.percentile(loss_np, 50))
        p90 = float(np.percentile(loss_np, 90))
        print(f"Loss stats:")
        print(f"  mean    = {mean:.6f}")
        print(f"  std     = {std:.6f}")
        print(f"  p10     = {p10:.6f}")
        print(f"  median  = {p50:.6f}")
        print(f"  p90     = {p90:.6f}")

    plt.figure(figsize=(7, 4))
    plt.hist(loss_np, bins=bins, alpha=0.7)
    plt.axvline(loss_np.mean(), color="red", linestyle="--", label="mean score")
    plt.xlabel("Loss")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)  # <- create folder(s)
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

import random

def _sample_diff(pool, old, rng):
    if len(pool) < 2:
        return old
    new = old
    while new == old:
        new = rng.choice(pool)
    return new

def corrupt_triples_at_indices(
    triples,
    pick_idx,
    entity_to_idx,
    relation_to_idx,
    p_corrupt_head=0.45,
    p_corrupt_rel=0.10,
    p_corrupt_tail=0.45,
    avoid_existing=True,
    max_tries=50,
    seed=None,
):
    rng = random.Random(seed)

    # Pools match triple element types: if triples use strings -> keys, if ints -> idx values
    sample_ent = (lambda x: list(entity_to_idx.keys())) if isinstance(triples[0][0], str) else (lambda x: list(entity_to_idx.values()))
    sample_rel = (lambda x: list(relation_to_idx.keys())) if isinstance(triples[0][1], str) else (lambda x: list(relation_to_idx.values()))
    ent_pool = sample_ent(None)
    rel_pool = sample_rel(None)

    noisy = list(triples)
    existing = set(triples) if avoid_existing else None

    # Precompute cumulative probs
    p1 = p_corrupt_head
    p2 = p_corrupt_head + p_corrupt_rel
    p3 = p2 + p_corrupt_tail
    if abs(p3 - 1.0) > 1e-6:
        # normalize if user passed weird values
        s = p3
        p1, p2 = p1 / s, p2 / s

    for i in pick_idx:
        h, r, t = noisy[i]

        for _ in range(max_tries):
            u = rng.random()
            if u < p1:
                h2, r2, t2 = _sample_diff(ent_pool, h, rng), r, t
            elif u < p2:
                h2, r2, t2 = h, _sample_diff(rel_pool, r, rng), t
            else:
                h2, r2, t2 = h, r, _sample_diff(ent_pool, t, rng)

            cand = (h2, r2, t2)
            if (not avoid_existing) or (cand not in existing):
                noisy[i] = cand
                if avoid_existing:
                    existing.add(cand)
                break

    return noisy


#@torch.no_grad()
def remove_by_centrality_plus_loss_forward(
    triples,
    *,
    model,   
    entity_to_idx,
    relation_to_idx,                               
    budget,             
    batch_size = 100,
    device = None,
    model_name,
    db_name,
):
    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device).eval()

    #print(model.loss_function)

    idx = triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx).to(device)
    logits_list = []
    for s in range(0, idx.size(0), batch_size):
        z = model.forward_triples(idx[s:s+batch_size]).reshape(-1)
        logits_list.append(z.detach().to("cpu"))
    logits = torch.cat(logits_list, dim=0)

    prob = logits  
    #prob = torch.sigmoid(logits)   
    
    mu = prob.mean()   
    dist = torch.abs(prob - mu)
    pick_idx = torch.topk(dist, k=budget, largest=False).indices.tolist()

    #ick_idx = torch.topk(logits, k=budget, largest=True).indices.tolist()

    visualize_loss(prob, bins=60, title=f"per-triple score {db_name}-{model_name}", save_path=f"./per-triple score/per-triple score {db_name}-{model_name}.png")

    noisy_triples = corrupt_triples_at_indices(
        triples=triples,
        pick_idx=pick_idx,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        seed=0,                 
        avoid_existing=True
    )

    return [], noisy_triples

    """
    removed = [triples[i] for i in pick_idx]
    keep_mask = [True] * len(triples)
    for i in pick_idx:
        keep_mask[i] = False
    kept = [t for i, t in enumerate(triples) if keep_mask[i]]

    return removed, kept
    """

    """
    dist = prob
    pick_idx = torch.topk(dist, k=budget, largest=True).indices.tolist()

    removed = [triples[i] for i in pick_idx]
    keep_mask = [True] * len(triples)
    for i in pick_idx:
        keep_mask[i] = False
    kept = [t for i, t in enumerate(triples) if keep_mask[i]]

    return removed, kept
    """

    #import torch.nn.functional as F
    #import math
    #import torch.nn as nn

    """
    idx = triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx).to(device)
    N = idx.size(0)
    grad_scores = torch.zeros(N, device=device, dtype=torch.float32)

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    for s in range(0, N, batch_size):
        model.zero_grad(set_to_none=True)

        batch_idx = idx[s:s+batch_size]                     
        z = model.forward_triples(batch_idx).reshape(-1)    
        target = torch.ones_like(z, device=z.device)
        
        #batch_loss = F.binary_cross_entropy_with_logits(
        #    z, target, reduction="none"
        #)                              

        mse_loss = nn.MSELoss()
        batch_loss = mse_loss(z, target)

        loss_scalar = batch_loss.mean()
        loss_scalar.backward()

        total = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.grad is None:
                continue
            total = total + (p.grad.detach() ** 2).sum()

        grad_norm_sq = total.item() 
        grad_scores[s:s+batch_idx.size(0)] = grad_norm_sq   # same score for all triples in this batch

    grad_scores = grad_scores.detach().cpu()
    visualize_loss(grad_scores, bins=60, title="UMLS train per-triple loss", save_path=f"complex_umls_train_grad_hist.png")
    #mu_g = grad_scores.mean()
    #dist = torch.abs(grad_scores - mu_g)  # distance from mean grad norm
    dist = grad_scores 

    pick_idx = torch.topk(dist, k=budget, largest=False).indices.tolist()
    """

    """
    idx = triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx).to(device)
    N = idx.size(0)

    mse_loss = nn.MSELoss(reduction="none") 

    loss_list = []
    for s in range(0, N, batch_size):
        batch_idx = idx[s:s+batch_size]
        z = model.forward_triples(batch_idx).reshape(-1)      
        target = torch.ones_like(z)                          
        batch_loss = mse_loss(z, target)             
        loss_list.append(batch_loss.detach().cpu())

    per_triple_loss = torch.cat(loss_list, dim=0) 

    dist = per_triple_loss 
    pick_idx = torch.topk(dist, k=budget, largest=False).indices.tolist()

    #mu_l = per_triple_loss.mean()
    #dist = torch.abs(per_triple_loss - mu_l)  # distance from mean loss
    #pick_idx = torch.topk(dist, k=budget, largest=False).indices.tolist()
    """
  
    """
    noisy_triples = corrupt_triples_at_indices(
        triples=triples,
        pick_idx=pick_idx,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        seed=0,                 
        avoid_existing=True
    )

    return [], noisy_triples
    """
    
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
