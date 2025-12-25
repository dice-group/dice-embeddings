from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import networkx as nx
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set, Literal
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

Triple = Tuple[str, str, str]


def _resolve_embeddings_or_raise(model, num_entities, num_relations):

    ent_emb = getattr(model, "entity_embeddings", None)
    rel_emb = getattr(model, "relation_embeddings", None)

    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        ent_emb, rel_emb = None, None
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Embedding):
                if mod.num_embeddings == num_entities and ent_emb is None:
                    ent_emb = mod
                elif mod.num_embeddings == num_relations and rel_emb is None:
                    rel_emb = mod
        if ent_emb is None or rel_emb is None:
            raise AttributeError(
                "Could not locate entity/relation embeddings on the model. "
                "Expected `entity_embeddings` / `relation_embeddings` or tables matching |E|/|R|."
            )
    return ent_emb, rel_emb

def _fgsm_step(grad, eps, norm):
    if norm == "linf":
        return eps * grad.sign()
    if norm == "l2":
        return eps * grad / (grad.norm() + 1e-12)
    raise ValueError("norm must be 'linf' or 'l2'")

@torch.no_grad()
def _topk_nearest_excluding(vec: torch.Tensor, table: torch.Tensor, exclude_idx: int, k: int) -> torch.Tensor:

    d = torch.cdist(vec.unsqueeze(0), table)[0] 
    d[exclude_idx] = float("inf")
    k = min(k, max(0, table.size(0) - 1))
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=table.device)
    return torch.topk(d, k=k, largest=False).indices

# ---------- FGSM add-attack ----------

@torch.no_grad()
def _try_trip_no_nest(
    h_id,
    r_id,
    t_id,
    *,
    model,
    device,
    I2E,
    I2R,
    train_set,
    forbidden,
    avoid_existing_edge,
    ht_set,
):
    trip = (I2E[h_id], I2R[r_id], I2E[t_id])

    if trip in train_set or trip in forbidden:
        return None
    if avoid_existing_edge and (trip[0], trip[2]) in ht_set:
        return None

    z = model.forward_triples(torch.tensor([[h_id, r_id, t_id]], device=device)).view(())
    L = F.softplus(-z).item()  # positive BCE loss
    return (L, trip)


def _consider_candidate(
    h_id,
    r_id,
    t_id,
    best,
    *,
    model,
    device,
    I2E,
    I2R,
    train_set,
    forbidden,
    avoid_existing_edge,
    ht_set,
):
    cand = _try_trip_no_nest(
        h_id, r_id, t_id,
        model=model, device=device, I2E=I2E, I2R=I2R,
        train_set=train_set, forbidden=forbidden,
        avoid_existing_edge=avoid_existing_edge, ht_set=ht_set,
    )
    if cand is None:
        return best
    if best is None or cand[0] > best[0]:
        return cand
    return best


def _pick_best_for_pattern(
    pattern,
    *,
    hi,
    ri,
    ti,
    h_cands,
    r_cands,
    t_cands,
    model,
    device,
    I2E,
    I2R,
    train_set,
    forbidden,
    avoid_existing_edge,
    ht_set,
):
    best = None

    if pattern == "best-of-three":
        for hh in (h_cands.tolist() if len(h_cands) > 0 else []):
            best = _consider_candidate(hh, ri, ti, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)
        for rr in (r_cands.tolist() if len(r_cands) > 0 else []):
            best = _consider_candidate(hi, rr, ti, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)
        for tt in (t_cands.tolist() if len(t_cands) > 0 else []):
            best = _consider_candidate(hi, ri, tt, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)

    elif pattern == "head":
        for hh in (h_cands.tolist() if len(h_cands) > 0 else []):
            best = _consider_candidate(hh, ri, ti, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)

    elif pattern == "rel":
        for rr in (r_cands.tolist() if len(r_cands) > 0 else []):
            best = _consider_candidate(hi, rr, ti, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)

    elif pattern == "tail":
        for tt in (t_cands.tolist() if len(t_cands) > 0 else []):
            best = _consider_candidate(hi, ri, tt, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)

    elif pattern in ("head-tail", "head-rel", "rel-tail", "all"):
        hh = int(h_cands[0].item()) if len(h_cands) > 0 else hi
        rr = int(r_cands[0].item()) if len(r_cands) > 0 else ri
        tt = int(t_cands[0].item()) if len(t_cands) > 0 else ti

        if pattern == "head-tail":
            best = _consider_candidate(hh, ri, tt, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)
        elif pattern == "head-rel":
            best = _consider_candidate(hh, rr, ti, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)
        elif pattern == "rel-tail":
            best = _consider_candidate(hi, rr, tt, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)
        elif pattern == "all":
            best = _consider_candidate(hh, rr, tt, best,
                model=model, device=device, I2E=I2E, I2R=I2R,
                train_set=train_set, forbidden=forbidden,
                avoid_existing_edge=avoid_existing_edge, ht_set=ht_set)
    else:
        raise ValueError("Invalid pattern")

    return best

def canon(x: object) -> str:
    s = str(x)
    return s

def add_corrupted_by_fgsm_forward(
    triples,
    *,
    model,
    entity_to_idx,
    relation_to_idx,
    budget,
    eps=0.25,
    norm="linf",
    pattern="best-of-three",
    topk_neighbors=32,
    avoid_existing_edge=True,
    restrict_by_relation=False,
    forbidden=None,
    device=None,
    progress_every=None,
):


    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device)
    model.train(False)
    

    e2i = {canon(k): v for k, v in entity_to_idx.items()}   
    r2i = {canon(k): v for k, v in relation_to_idx.items()}

    I2E = {v: canon(k) for k, v in entity_to_idx.items()}
    I2R = {v: canon(k) for k, v in relation_to_idx.items()}

    triples = [(canon(h), canon(r), canon(t)) for (h, r, t) in triples]

    num_entities, num_relations = len(e2i), len(r2i)

    ent_emb, rel_emb = _resolve_embeddings_or_raise(model, num_entities, num_relations)
    for p in model.parameters():
        p.requires_grad_(False)
    ent_emb.weight.requires_grad_(True)
    rel_emb.weight.requires_grad_(True)

    E = ent_emb.weight.detach()
    R = rel_emb.weight.detach()

    train_set = set(triples)
    ht_set = {(h, t) for h, _, t in triples}

    heads_by_rel, tails_by_rel = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        heads_by_rel[r].add(h)
        tails_by_rel[r].add(t)
    heads_by_rel_ids = {r2i[r]: {e2i[h] for h in H} for r, H in heads_by_rel.items()}
    tails_by_rel_ids = {r2i[r]: {e2i[t] for t in T} for r, T in tails_by_rel.items()}

    forbidden = forbidden or set()
    candidates = []

    for idx_anchor, (h, r, t) in enumerate(triples):
        try:
            hi, ri, ti = e2i[h], r2i[r], e2i[t]
        except KeyError:
            continue

        model.zero_grad(set_to_none=True)
        idx = torch.tensor([[hi, ri, ti]], dtype=torch.long, device=device)
        logit = model.forward_triples(idx).view(())
        loss = F.binary_cross_entropy_with_logits(logit, torch.ones((), device=device))
        loss.backward()

        GE = ent_emb.weight.grad
        GR = rel_emb.weight.grad
        if GE is None or GR is None:
            raise RuntimeError("Embedding gradients are None; ensure embeddings require_grad is True.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h, g_r, g_t = GE[hi], GR[ri], GE[ti]

        Eh_adv = (E[hi] + _fgsm_step(g_h, eps, norm)).detach()
        Er_adv = (R[ri] + _fgsm_step(g_r, eps, norm)).detach()
        Et_adv = (E[ti] + _fgsm_step(g_t, eps, norm)).detach()

        h_cands = _topk_nearest_excluding(Eh_adv, E, hi, topk_neighbors)
        r_cands = _topk_nearest_excluding(Er_adv, R, ri, topk_neighbors)
        t_cands = _topk_nearest_excluding(Et_adv, E, ti, topk_neighbors)

        if restrict_by_relation:
            if ri in heads_by_rel_ids and len(h_cands) > 0:
                mask = torch.tensor([int(hh.item() in heads_by_rel_ids[ri]) for hh in h_cands],
                                    dtype=torch.bool, device=h_cands.device)
                h_cands = h_cands[mask]
            if ri in tails_by_rel_ids and len(t_cands) > 0:
                mask = torch.tensor([int(tt.item() in tails_by_rel_ids[ri]) for tt in t_cands],
                                    dtype=torch.bool, device=t_cands.device)
                t_cands = t_cands[mask]

        best = _pick_best_for_pattern(
            pattern,
            hi=hi, ri=ri, ti=ti,
            h_cands=h_cands, r_cands=r_cands, t_cands=t_cands,
            model=model, device=device, I2E=I2E, I2R=I2R,
            train_set=train_set, forbidden=forbidden,
            avoid_existing_edge=avoid_existing_edge, ht_set=ht_set
        )

        if best is not None:
            candidates.append(best)

        if progress_every and idx_anchor % progress_every == 0 and idx_anchor > 0:
            print(f"[FGSM-add] processed {idx_anchor}/{len(triples)} anchors")

    ent_emb.weight.requires_grad_(False)
    rel_emb.weight.requires_grad_(False)

    if not candidates:
        return []

    best_by_trip = {}
    for L, trip in candidates:
        prev = best_by_trip.get(trip)
        if (prev is None) or (L > prev):
            best_by_trip[trip] = L

    items = sorted(best_by_trip.items(), key=lambda kv: kv[1], reverse=True)
    return [trip for trip, _ in items[:budget]]

######################

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

def _rank_nodes(centrality: Dict[str, float], top_k: Optional[int]) -> List[str]:
    items = sorted(centrality.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return [n for n, _ in (items if top_k is None else items[:top_k])]

def _existing_sets(triples: List[Triple]):
    triple_set = set(triples)
    ht_set = {(h, t) for h, _, t in triples} 
    return triple_set, ht_set

def _relation_domains_ranges(triples: List[Triple]):
    heads_by_rel = defaultdict(set)
    tails_by_rel = defaultdict(set)
    for h, r, t in triples:
        heads_by_rel[r].add(h)
        tails_by_rel[r].add(t)
    return heads_by_rel, tails_by_rel

def propose_candidates_centrality(
    triples,
    node_centrality,
    *,
    mode = "both",                 
    top_k_nodes = 1000,
    avoid_existing_edge = True,
    restrict_by_relation = False,
    forbidden = None
):
    forbidden = forbidden or set()
    triple_set, ht_set = _existing_sets(triples)
    top_nodes = _rank_nodes(node_centrality, top_k_nodes)
    heads_by_rel, tails_by_rel = _relation_domains_ranges(triples)

    cands = []
    seen = set()

    for h, r, t in triples:
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
                cands.append(cand)
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
                cands.append(cand)
                seen.add(cand)

    return cands

@torch.no_grad()
def add_corrupted_by_centrality_and_loss_forward(
    triples,
    *,
    model,                              
    entity_to_idx,          
    relation_to_idx,       
    budget,
    centrality = "harmonic",          
    undirected = True,
    mode = "both",
    top_k_nodes,
    avoid_existing_edge = True,
    restrict_by_relation = False,
    forbidden = None,
    batch_size,
    device = None,
):

    e2i = {canon(k): v for k, v in entity_to_idx.items()}
    r2i = {canon(k): v for k, v in relation_to_idx.items()}
    triples = [(canon(h), canon(r), canon(t)) for (h, r, t) in triples]

    if forbidden:
        forbidden = {(canon(h), canon(r), canon(t)) for (h, r, t) in forbidden}
    
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    model = model.to(device).eval()

    Gd = _build_entity_digraph(triples)
    G = Gd.to_undirected(as_view=True) if undirected else Gd
    
    if centrality == "betweenness":
        node_cent = nx.betweenness_centrality(G, normalized=True)
    elif centrality == "closeness":
        node_cent = nx.closeness_centrality(G)
    elif centrality == "harmonic":
        node_cent = nx.harmonic_centrality(G)
    else:
        raise ValueError("centrality must be in {'betweenness','closeness','harmonic'}")

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

    cands = [(canon(h), canon(r), canon(t)) for (h, r, t) in cands]
    

    cands = [c for c in cands if c[0] in e2i and c[1] in r2i and c[2] in e2i]
    if not cands:
        return []

    idx = triples_to_idx_with_maps(cands, e2i, r2i).to(device)
    logits = []
    for s in range(0, idx.size(0), batch_size):
        z = model.forward_triples(idx[s:s+batch_size]).reshape(-1)
        logits.append(z.detach().to("cpu"))
    logits = torch.cat(logits, dim=0)

    loss_pos = F.softplus(-logits)
    k = min(budget, len(cands))
    topk_vals, topk_idx = torch.topk(loss_pos, k=k, largest=True)
    selected = [cands[i] for i in topk_idx.tolist()]
    return selected


@torch.no_grad()
def add_corrupted_by_global_argmax_forward(
    triples,
    *,
    model,                                 
    entity_to_idx,          
    relation_to_idx,        
    budget,
    mode = "both",                     
    avoid_existing_edge = True,
    forbidden = None,
    per_anchor_topk = 1,               
    batch_size = 10000,
    anchor_cap = None,      
    device = None,
):

    device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu"))
    model = model.to(device).eval()

    e2i, r2i = entity_to_idx, relation_to_idx
    I2E = {i: e for e, i in e2i.items()}
    I2R = {i: r for r, i in r2i.items()}

    num_entities = len(e2i)
    all_e_idx = torch.arange(num_entities, dtype=torch.long, device=device)

    train_triple_set: Set[Triple] = set(triples)
    ht_set = {(h, t) for h, _, t in triples}

    heads_for_tail_ids = defaultdict(set) 
    tails_for_head_ids = defaultdict(set)  
    for h, _, t in triples:
        hi, ti = e2i[h], e2i[t]
        heads_for_tail_ids[ti].add(hi)
        tails_for_head_ids[hi].add(ti)

    orig_heads_by_rt_ids: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
    orig_tails_by_hr_ids: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
    anchors_head: Set[Tuple[int,int]] = set()
    anchors_tail: Set[Tuple[int,int]] = set()

    for h, r, t in triples:
        hi, ri, ti = e2i[h], r2i[r], e2i[t]
        anchors_head.add((ri, ti))
        anchors_tail.add((hi, ri))
        orig_heads_by_rt_ids[(ri, ti)].add(hi)
        orig_tails_by_hr_ids[(hi, ri)].add(ti)

    forbidden = forbidden or set()
    forb_heads_by_rt_ids: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
    forb_tails_by_hr_ids: Dict[Tuple[int,int], Set[int]] = defaultdict(set)
    for h, r, t in forbidden:
        if h in e2i and r in r2i and t in e2i:
            hi, ri, ti = e2i[h], r2i[r], e2i[t]
            forb_heads_by_rt_ids[(ri, ti)].add(hi)
            forb_tails_by_hr_ids[(hi, ri)].add(ti)

    if anchor_cap is not None:
        head_counts = {k: len(v) for k, v in orig_heads_by_rt_ids.items()}
        tail_counts = {k: len(v) for k, v in orig_tails_by_hr_ids.items()}
        if mode in ("head", "both"):
            anchors_head = set(sorted(anchors_head, key=lambda k: head_counts.get(k, 0), reverse=True)[:anchor_cap])
        else:
            anchors_head = set()
        if mode in ("tail", "both"):
            anchors_tail = set(sorted(anchors_tail, key=lambda k: tail_counts.get(k, 0), reverse=True)[:anchor_cap])
        else:
            anchors_tail = set()
    else:
        if mode == "head":
            anchors_tail = set()
        elif mode == "tail":
            anchors_head = set()

    candidates: List[Tuple[float, Triple]] = []  

    for (ri, ti) in anchors_head:
        allow = torch.ones(num_entities, dtype=torch.bool, device=device)
        if orig_heads_by_rt_ids[(ri, ti)]:
            idxs = torch.tensor(list(orig_heads_by_rt_ids[(ri, ti)]), dtype=torch.long, device=device)
            allow[idxs] = False
        if avoid_existing_edge and heads_for_tail_ids.get(ti):
            idxs = torch.tensor(list(heads_for_tail_ids[ti]), dtype=torch.long, device=device)
            allow[idxs] = False
        if forb_heads_by_rt_ids.get((ri, ti)):
            idxs = torch.tensor(list(forb_heads_by_rt_ids[(ri, ti)]), dtype=torch.long, device=device)
            allow[idxs] = False

        if allow.sum().item() == 0:
            continue

        cand_heads = all_e_idx[allow]  
        losses = []
        for s in range(0, cand_heads.numel(), batch_size):
            hh = cand_heads[s:s+batch_size]
            rr = torch.full_like(hh, ri)
            tt = torch.full_like(hh, ti)
            idx = torch.stack([hh, rr, tt], dim=1)   
            z = model.forward_triples(idx).reshape(-1)
            L = F.softplus(-z)  # BCEWithLogits(z, 1)
            losses.append(L.detach())
        losses = torch.cat(losses, dim=0)

        k = min(per_anchor_topk, losses.numel())
        top_vals, top_idx = torch.topk(losses, k=k, largest=True)
        for v, j in zip(top_vals.tolist(), top_idx.tolist()):
            h_id = int(cand_heads[j].item())
            trip = (I2E[h_id], I2R[ri], I2E[ti])
            
            if avoid_existing_edge and (trip[0], trip[2]) in ht_set:
                continue
            if trip in train_triple_set or trip in forbidden:
                continue
            candidates.append((v, trip))

    for (hi, ri) in anchors_tail:
        allow = torch.ones(num_entities, dtype=torch.bool, device=device)
        if orig_tails_by_hr_ids[(hi, ri)]:
            idxs = torch.tensor(list(orig_tails_by_hr_ids[(hi, ri)]), dtype=torch.long, device=device)
            allow[idxs] = False
        if avoid_existing_edge and tails_for_head_ids.get(hi):
            idxs = torch.tensor(list(tails_for_head_ids[hi]), dtype=torch.long, device=device)
            allow[idxs] = False
        if forb_tails_by_hr_ids.get((hi, ri)):
            idxs = torch.tensor(list(forb_tails_by_hr_ids[(hi, ri)]), dtype=torch.long, device=device)
            allow[idxs] = False

        if allow.sum().item() == 0:
            continue

        cand_tails = all_e_idx[allow]
        losses = []
        for s in range(0, cand_tails.numel(), batch_size):
            tt = cand_tails[s:s+batch_size]
            hh = torch.full_like(tt, hi)
            rr = torch.full_like(tt, ri)
            idx = torch.stack([hh, rr, tt], dim=1)
            z = model.forward_triples(idx).reshape(-1)
            L = F.softplus(-z)
            losses.append(L.detach())
        losses = torch.cat(losses, dim=0)

        k = min(per_anchor_topk, losses.numel())
        top_vals, top_idx = torch.topk(losses, k=k, largest=True)
        for v, j in zip(top_vals.tolist(), top_idx.tolist()):
            t_id = int(cand_tails[j].item())
            trip = (I2E[hi], I2R[ri], I2E[t_id])
            if avoid_existing_edge and (trip[0], trip[2]) in ht_set:
                continue
            if trip in train_triple_set or trip in forbidden:
                continue
            candidates.append((v, trip))

    if not candidates:
        return []

    best_by_trip: Dict[Triple, float] = {}
    for loss_val, trip in candidates:
        prev = best_by_trip.get(trip)
        if (prev is None) or (loss_val > prev):
            best_by_trip[trip] = loss_val

    items = sorted(best_by_trip.items(), key=lambda kv: kv[1], reverse=True)
    selected = [trip for trip, _ in items[:min(budget, len(items))]]
    return selected

# -------------------------------------------------------

def build_digraph(triples):
    G = nx.DiGraph()
    for h, _, t in triples:
        if not G.has_edge(h, t):
            G.add_edge(h, t)
    return G

def _existing_sets2(triples):
    triple_set = set(triples)
    ht_set = set((h, t) for h, _, t in triples)
    return triple_set, ht_set

def _rank_nodes2(centrality_dict):
    return sorted(centrality_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

def _propose_corruptions(triples, node_centrality, budget, mode="both", top_k_nodes=100, avoid_existing_edge=False):
    G_nodes_ranked = _rank_nodes2(node_centrality)
    if top_k_nodes is not None:
        G_nodes_ranked = G_nodes_ranked[:top_k_nodes]
    top_nodes = [n for n, _ in G_nodes_ranked]

    triple_set, ht_set = _existing_sets2(triples)
    candidates = []
    seen = set()

    for h, r, t in triples:
        if mode in ("head", "both"):
            for n in top_nodes:
                if n == h or n == t:
                    continue
                cand = (n, r, t)
                if cand in triple_set or cand in seen:
                    continue
                if avoid_existing_edge and (n, t) in ht_set:
                    continue
                score = 0.5 * (node_centrality.get(n, 0.0) + node_centrality.get(t, 0.0))
                candidates.append((score, cand))
                seen.add(cand)
        if mode in ("tail", "both"):
            for n in top_nodes:
                if n == h or n == t:
                    continue
                cand = (h, r, n)
                if cand in triple_set or cand in seen:
                    continue
                if avoid_existing_edge and (h, n) in ht_set:
                    continue
                score = 0.5 * (node_centrality.get(h, 0.0) + node_centrality.get(n, 0.0))
                candidates.append((score, cand))
                seen.add(cand)

    candidates.sort(key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]), reverse=True)
    top = [trip for _, trip in candidates[:budget]]
    return top

def add_corrupted_by_betweenness(triples, budget, mode="both", top_k_nodes=100, avoid_existing_edge=False):
    G = build_digraph(triples)
    node_cent = nx.betweenness_centrality(G, normalized=True)
    return _propose_corruptions(
        triples, node_cent, budget, mode=mode, top_k_nodes=top_k_nodes, avoid_existing_edge=avoid_existing_edge
    )

def add_corrupted_by_closeness(triples, budget, mode="both", top_k_nodes=100, undirected=False, avoid_existing_edge=False):
    Gd = build_digraph(triples)
    G = Gd.to_undirected() if undirected else Gd
    node_cent = nx.closeness_centrality(G)
    return _propose_corruptions(
        triples, node_cent, budget, mode=mode, top_k_nodes=top_k_nodes, avoid_existing_edge=avoid_existing_edge
    )
