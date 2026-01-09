from typing import List, Tuple, Dict, Optional, Set, Literal
from collections import defaultdict
import networkx as nx
import torch
import torch.nn.functional as F
from typing import Literal
import torch.nn as nn
import pandas as pd
from pathlib import Path
import pandas as pd
import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

Triple = Tuple[str, str, str]

import pandas as pd
import networkx as nx

def triples_and_scores_to_csv(
    triples,
    scores,
    out_csv_path,
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
def score_based_deletion(
    triples,
    *,
    model,   
    entity_to_idx,
    relation_to_idx,                               
    budget,             
    batch_size,
    device,
    model_name,
    db_name,
    q1,
    q2
):
    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device).eval()

    idx = triples_to_idx_with_maps(triples, entity_to_idx, relation_to_idx).to(device)
    logits_list = []
    for s in range(0, idx.size(0), batch_size):
        z = model.forward_triples(idx[s:s+batch_size]).reshape(-1)
        logits_list.append(z.detach().to("cpu"))
    logits = torch.cat(logits_list, dim=0)

    #triples_and_scores_to_csv(triples, logits, f"triple_scores_and_centrality_{db_name}.csv")
    
    """
    mu = logits.mean()   
    dist = torch.abs(logits - mu)
    pick_idx = torch.topk(dist, k=budget, largest=False).indices.tolist()
    visualize_loss(logits, bins=60, title=f"per-triple score (mid) {db_name}-{model_name}", save_path=f"./per-triple-score/per-triple score (mid) {db_name}-{model_name}.png")

    """

    q_low  = torch.quantile(logits, q1)
    q_high = torch.quantile(logits, q2)

    cand = torch.where((logits >= q_low) & (logits <= q_high))[0]

    #if cand.numel() >= budget:
    pick_idx = cand[torch.randperm(cand.numel(), device=cand.device)[:budget]].tolist()
    #else:
    #    # fallback: fill with next-highest if band too small
    #    pick_idx = torch.topk(logits, k=min(budget, logits.numel()), largest=True).indices.tolist()

    #visualize_loss(logits, bins=60, title=f"per-triple score (mid-high) {db_name}-{model_name}", save_path=f"./per-triple-score/per-triple score (mid-high) {db_name}-{model_name}.png")
    
    noisy_triples = corrupt_triples_at_indices(
        triples=triples,
        pick_idx=pick_idx,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        seed=0,                 
        avoid_existing=True
    )

    return noisy_triples

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
    
