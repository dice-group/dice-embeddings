import random
from typing import List, Tuple, Set
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- you said you have these already; included here for completeness ---
def fgsm_delta(grad: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:
    if not torch.isfinite(grad).all():
        grad = torch.nan_to_num(grad)
    n = norm.lower()
    if n == "linf":
        return eps * grad.sign()
    if n == "l2":
        g = torch.norm(grad)
        if (not torch.isfinite(g)) or g.item() == 0.0:
            return torch.zeros_like(grad)
        return eps * (grad / g)
    raise ValueError("norm must be 'linf' or 'l2'")

def logits_for_indices(model, h_i, r_i, t_i, device):
    idxs = torch.tensor([[int(h_i), int(r_i), int(t_i)]], dtype=torch.long, device=device)  # [1,3]
    return model.forward_triples(idxs)


# --- helpers (top-level only) ---
def _ensure_dense(x: torch.Tensor) -> torch.Tensor:
    return x.to_dense() if x.is_sparse else x

def _nearest_k_excluding(vec: torch.Tensor, table: torch.Tensor, exclude_idx: int, k: int) -> torch.Tensor:
    # vec: [d], table: [N,d]  -> returns LongTensor[k]
    d = torch.cdist(vec.unsqueeze(0), table)[0]    # [N]
    d[exclude_idx] = float("inf")
    k = min(k, table.size(0) - 1)
    vals, idxs = torch.topk(-d, k)  # negative distances => largest are nearest
    return idxs

def _collect_hard_negs_for_triple(
    h_i: int,
    r_i: int,
    t_i: int,
    E: torch.Tensor,
    R: torch.Tensor,
    triples_set_idx: Set[Tuple[int,int,int]],
    k_head: int,
    k_tail: int,
    k_rel: int
) -> List[Tuple[int,int,int]]:
    cands: List[Tuple[int,int,int]] = []
    # nearest alternatives
    nh = _nearest_k_excluding(E[h_i], E, h_i, k_head)
    nt = _nearest_k_excluding(E[t_i], E, t_i, k_tail)
    nr = _nearest_k_excluding(R[r_i], R, r_i, k_rel)

    # single-component corruptions (classic)
    for hh in nh.tolist():
        trip = (hh, r_i, t_i)
        if trip not in triples_set_idx:
            cands.append(trip)
    for tt in nt.tolist():
        trip = (h_i, r_i, tt)
        if trip not in triples_set_idx:
            cands.append(trip)
    for rr in nr.tolist():
        trip = (h_i, rr, t_i)
        if trip not in triples_set_idx:
            cands.append(trip)

    # simple combos (stronger hard-negs, but keep small)
    if len(nh) > 0 and len(nt) > 0:
        trip = (int(nh[0]), r_i, int(nt[0]))
        if trip not in triples_set_idx:
            cands.append(trip)
    if len(nh) > 0 and len(nr) > 0:
        trip = (int(nh[0]), int(nr[0]), t_i)
        if trip not in triples_set_idx:
            cands.append(trip)
    if len(nr) > 0 and len(nt) > 0:
        trip = (h_i, int(nr[0]), int(nt[0]))
        if trip not in triples_set_idx:
            cands.append(trip)

    # de-dup
    seen = set()
    unique = []
    for tr in cands:
        if tr not in seen:
            unique.append(tr)
            seen.add(tr)
    return unique

# --- main method: FGSM hard-negative support removal ---
def select_adversarial_removals_fgsm_hardneg(
    train_triples: List[Tuple[str, str, str]],
    oracle,
    seed: int = 0,
    eps: float = 1e-2,
    norm: str = "linf",
    k_head: int = 4,
    k_tail: int = 4,
    k_rel: int = 2,
    require_high_conf: bool = False,
    conf_threshold: float = 0.70
):
    """
    Rank training triples by how much an FGSM nudge on their OWN rows increases
    the loss on a batch of hard negatives built from nearest neighbors.
    Return: [ ((h,r,t), delta_neg_loss, neg_loss_before, neg_loss_after, p_pos_before) ] sorted desc by delta_neg_loss.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    device = next(oracle.model.parameters()).device
    model = oracle.model
    model.train(False)

    # maps
    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}
    I2R = {i: r for r, i in R2I.items()}

    # embeddings
    ent_emb = model.entity_embeddings
    rel_emb = model.relation_embeddings
    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        raise RuntimeError("Expected nn.Embedding at model.entity_embeddings / relation_embeddings")

    E = ent_emb.weight.detach()  # [n_ent, d]
    R = rel_emb.weight.detach()  # [n_rel, d]

    # fast membership check to avoid sampling true triples as negatives
    triples_set_idx = {(E2I[h], R2I[r], E2I[t]) for (h, r, t) in train_triples}

    results: List[Tuple[Tuple[str,str,str], float, float, float, float]] = []
    sort_by = itemgetter(1)

    for (h, r, t) in train_triples:
        h_i = E2I[h]; r_i = R2I[r]; t_i = E2I[t]

        # baseline pos prob (optional filter)
        with torch.no_grad():
            logit_pos = logits_for_indices(model, h_i, r_i, t_i, device)
            p_before = torch.sigmoid(logit_pos.squeeze()).item()
        if require_high_conf and p_before < conf_threshold:
            continue

        # build a small batch of hard negatives for THIS triple
        hard_negs = _collect_hard_negs_for_triple(
            h_i, r_i, t_i, E, R, triples_set_idx,
            k_head=k_head, k_tail=k_tail, k_rel=k_rel
        )
        if len(hard_negs) == 0:
            # fallback: random single head corruption
            rand_h = (h_i + 1) % E.shape[0]
            cand = (rand_h, r_i, t_i)
            if cand in triples_set_idx:
                cand = (h_i, r_i, (t_i + 1) % E.shape[0])
            hard_negs = [cand]

        # neg loss BEFORE
        with torch.no_grad():
            hn = torch.tensor(hard_negs, dtype=torch.long, device=device)
            logits_neg = model.forward_triples(hn)                     # [B]
            loss_neg_before = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
            lnb = float(loss_neg_before.item())

        # gradient on candidate triple (pos label)
        for p in model.parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = logits_for_indices(model, h_i, r_i, t_i, device)
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss.backward()

        GE = _ensure_dense(ent_emb.weight.grad)
        GR = _ensure_dense(rel_emb.weight.grad)
        if GE is None or GR is None:
            raise RuntimeError("Embedding grads are None; ensure forward_triples keeps grads.")

        # FGSM deltas on SAME rows
        dh = fgsm_delta(GE[h_i], eps, norm)
        dr = fgsm_delta(GR[r_i], eps, norm)
        dt = fgsm_delta(GE[t_i], eps, norm)

        # snapshot and apply virtual perturbation
        with torch.no_grad():
            Eh0 = ent_emb.weight[h_i].clone()
            Er0 = rel_emb.weight[r_i].clone()
            Et0 = ent_emb.weight[t_i].clone()
            ent_emb.weight[h_i] = Eh0 + dh
            rel_emb.weight[r_i] = Er0 + dr
            ent_emb.weight[t_i] = Et0 + dt

        # neg loss AFTER FGSM
        logits_neg_adv = model.forward_triples(hn)
        loss_neg_after = F.binary_cross_entropy_with_logits(logits_neg_adv, torch.zeros_like(logits_neg_adv))
        lna = float(loss_neg_after.item())

        # restore params
        with torch.no_grad():
            ent_emb.weight[h_i] = Eh0
            rel_emb.weight[r_i] = Er0
            ent_emb.weight[t_i] = Et0

        delta = lna - lnb  # how much the candidate supports pushing negatives down

        results.append(((I2E[h_i], I2R[r_i], I2E[t_i]), float(delta), lnb, lna, float(p_before)))

    results.sort(key=sort_by, reverse=True)  # higher Δneg-loss => better to remove
    return results
