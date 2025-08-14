import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, random
import math
from typing import List, Tuple, Optional, Set
import numpy as np
from typing import Dict
import numpy as np
from typing import List, Tuple, Optional, Set

Triple = Tuple[str, str, str]
Result = Tuple[Triple, Triple, float]  # (corrupted, clean, prob_after)

def poison_score_from_prob(prob_after: float, eps: float = 1e-12) -> float:
    # Higher is worse for the model if added as positive
    return -math.log(max(prob_after, eps))

def select_k_top_loss(results: List[Result],
                      k: int,
                      forbidden: Optional[Set[Triple]] = None) -> List[Result]:
    """
    Pick k triples with largest positive-label loss (i.e., smallest prob_after).
    `forbidden` can be a set of triples to exclude (e.g., any that already exist in train/valid/test).
    """
    filtered = []
    for corrupted, clean, p in results:
        if forbidden is not None and (corrupted in forbidden):
            continue
        score = poison_score_from_prob(p)
        filtered.append((corrupted, clean, p, score))
    filtered.sort(key=lambda x: x[3], reverse=True)  # highest loss first
    out = [(c, cl, p) for (c, cl, p, s) in filtered[:k]]
    return out

# -------- Optional: diversify with a simple MMR-style selection --------

def triple_token_set(tri: Triple) -> Set[str]:
    # structural similarity proxy (shares head/rel/tail penalized)
    return {tri[0], tri[1], tri[2]}

def jaccard_sim(a: Triple, b: Triple) -> float:
    sa = triple_token_set(a)
    sb = triple_token_set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return 0.0 if union == 0 else inter / union

def select_k_mmr(results: List[Result],
                 k: int,
                 lambda_div: float = 0.5,
                 forbidden: Optional[Set[Triple]] = None) -> List[Result]:
    """
    Greedy Maximal Marginal Relevance:
      objective(cand) = (1 - λ)*score  -  λ*max_{sel} sim(cand, sel)
    where score = -log(prob_after). λ∈[0,1].
    """
    pool = []
    for corrupted, clean, p in results:
        if forbidden is not None and (corrupted in forbidden):
            continue
        score = poison_score_from_prob(p)
        pool.append((corrupted, clean, p, score))
    pool.sort(key=lambda x: x[3], reverse=True)

    selected: List[Tuple[Triple, Triple, float, float]] = []
    while len(selected) < k and len(pool) > 0:
        if len(selected) == 0:
            selected.append(pool.pop(0))
            continue

        best_idx = None
        best_obj = -1e30
        for i, (c, cl, p, s) in enumerate(pool):
            max_sim = 0.0
            for (cs, _, _, _) in selected:
                max_sim = max(max_sim, jaccard_sim(c, cs))
            obj = (1.0 - lambda_div) * s - lambda_div * max_sim
            if obj > best_obj:
                best_obj = obj
                best_idx = i
        selected.append(pool.pop(best_idx))

    out = [(c, cl, p) for (c, cl, p, s) in selected]
    return out


def logits_for_indices(model, h_i: int, r_i: int, t_i: int, device) -> torch.Tensor:
    idxs = torch.tensor([[h_i, r_i, t_i]], dtype=torch.long, device=device)
    return model.forward_triples(idxs)  # raw logits expected (do not sigmoid here)


def fgsm_delta(grad: torch.Tensor, eps: float, norm: str = "linf") -> torch.Tensor:
    if norm == "linf":
        return eps * torch.sign(grad)
    if norm == "l2":
        g = grad
        n = torch.linalg.vector_norm(g)
        return eps * (g / (n + 1e-12))
    raise ValueError("norm must be 'linf' or 'l2'")

def nearest_idx_excluding(vec: torch.Tensor, table: torch.Tensor, exclude_idx: int) -> int:
    d = torch.cdist(vec.unsqueeze(0), table, p=2)[0]  # [n_items]
    d[exclude_idx] = float("inf")
    return int(torch.argmin(d).item())

# --- main function ---

def select_adversarial_triples_fgsm_simple(
    triples,
    oracle,
    seed,
    eps: float = 1e-2,
    norm: str = "linf",
):
    """
    Minimal FGSM attack for KGE models.
    - Computes grad on the clean (h, r, t)
    - Takes an FGSM step for h/r/t embeddings
    - Snaps each to its nearest *different* entity/relation
    - Returns list of (corrupted_triple, clean_triple, prob_after)

    Expects:
      - oracle.model.entity_embeddings / relation_embeddings are nn.Embedding
      - oracle.entity_to_idx / relation_to_idx dicts
      - a helper: logits_for_indices(model, h_i, r_i, t_i, device) -> raw logits (scalar tensor)
    """
    device = next(oracle.model.parameters()).device

    random.seed(seed)
    torch.manual_seed(seed)

    # maps
    E2I = oracle.entity_to_idx
    R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}
    I2R = {i: r for r, i in R2I.items()}

    # embeddings
    ent_emb = oracle.model.entity_embeddings
    rel_emb = oracle.model.relation_embeddings
    if not isinstance(ent_emb, nn.Embedding) or not isinstance(rel_emb, nn.Embedding):
        raise RuntimeError("Expected nn.Embedding at model.entity_embeddings / relation_embeddings")

    # frozen tables to snap against
    E = ent_emb.weight.detach()  # [n_ent, d]
    R = rel_emb.weight.detach()  # [n_rel, d]

    oracle.model.train(False)  # deterministic forward (no dropout)
    results = []

    for (h, r, t) in triples:
        h_i, r_i, t_i = E2I[h], R2I[r], E2I[t]

        # grads on clean triple
        for p in oracle.model.parameters():
            p.requires_grad_(True)
        oracle.model.zero_grad(set_to_none=True)

        logits_clean = logits_for_indices(oracle.model, h_i, r_i, t_i, device)  # scalar tensor
        target_pos = torch.ones_like(logits_clean)
        loss = F.binary_cross_entropy_with_logits(logits_clean, target_pos)
        loss.backward()

        GE = ent_emb.weight.grad
        GR = rel_emb.weight.grad
        if GE is None or GR is None:
            raise RuntimeError("Embedding grads are None; ensure forward doesn't detach grads.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        # FGSM step for each component
        dh = fgsm_delta(GE[h_i], eps, norm)
        dr = fgsm_delta(GR[r_i], eps, norm)
        dt = fgsm_delta(GE[t_i], eps, norm)

        Eh = E[h_i] + dh
        Er = R[r_i] + dr
        Et = E[t_i] + dt

        # snap to nearest valid symbols (exclude originals)
        h_adv_i = nearest_idx_excluding(Eh, E, h_i)
        r_adv_i = nearest_idx_excluding(Er, R, r_i)
        t_adv_i = nearest_idx_excluding(Et, E, t_i)

        # evaluate and store
        logits_adv = logits_for_indices(oracle.model, h_adv_i, r_adv_i, t_adv_i, device)
        prob_after = torch.sigmoid(logits_adv).item()

        corrupted_triple = (I2E[h_adv_i], I2R[r_adv_i], I2E[t_adv_i])
        results.append((corrupted_triple, (h, r, t), prob_after))

    return results

#Triple = Tuple[str, str, str]
#Result = Tuple[Triple, Triple, float]  # (corrupted, clean, prob_after)

def select_k_top_loss_fast(results: List[Result],
                           k: int,
                           forbidden: Optional[Set[Triple]] = None) -> List[Result]:
    n = len(results)
    if n == 0 or k <= 0:
        return []

    if forbidden is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.array([res[0] not in forbidden for res in results], dtype=bool)

    idx_pool = np.nonzero(mask)[0]
    if idx_pool.size == 0:
        return []
    if idx_pool.size <= k:
        # Already <= k after filtering: return sorted by loss
        probs_all = np.array([results[i][2] for i in idx_pool], dtype=np.float64)
        scores_all = -np.log(np.clip(probs_all, 1e-12, 1.0))
        order = np.argsort(-scores_all)  # descending score
        return [results[i] for i in idx_pool[order]]

    probs = np.array([results[i][2] for i in idx_pool], dtype=np.float64)
    scores = -np.log(np.clip(probs, 1e-12, 1.0))

    topk_unsorted = np.argpartition(scores, -k)[-k:]
    topk_scores = scores[topk_unsorted]
    order = np.argsort(-topk_scores)  # sort top-k only, descending
    chosen = idx_pool[topk_unsorted[order]]

    final_output = [results[i] for i in chosen]
    corrupted_tripls = [item[0] for item in final_output]

    return corrupted_tripls


def _build_token_ids(triples: List[Triple],
                     tok2id: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if tok2id is None:
        tok2id = {}
    H, R, T = [], [], []
    next_id = len(tok2id)
    for h, r, t in triples:
        if h not in tok2id:
            tok2id[h] = next_id; next_id += 1
        if r not in tok2id:
            tok2id[r] = next_id; next_id += 1
        if t not in tok2id:
            tok2id[t] = next_id; next_id += 1
        H.append(tok2id[h]); R.append(tok2id[r]); T.append(tok2id[t])
    return np.array(H, np.int64), np.array(R, np.int64), np.array(T, np.int64), tok2id


def select_k_mmr_fast(results: List[Result],
                      k: int,
                      lambda_div: float = 0.5,
                      forbidden: Optional[Set[Triple]] = None,
                      pool_size: Optional[int] = None,
                      tok2id: Optional[Dict[str, int]] = None) -> List[Result]:
    """
    Vectorized MMR:
      objective = (1 - λ) * score  -  λ * max_sim_to_selected
    where score = -log(prob_after), and sim(cand, sel) = (# of equal positions among h,r,t) / 3.

    pool_size: if set, first keep only top-`pool_size` by score before running MMR (big speed win).
    tok2id: optional shared map so repeated calls don't rebuild IDs.
    """
    n = len(results)
    if n == 0 or k <= 0:
        return []

    triples = [res[0] for res in results]
    probs = np.array([res[2] for res in results], dtype=np.float64)
    scores_all = -np.log(np.clip(probs, 1e-12, 1.0))

    if forbidden is None:
        keep_mask = np.ones(n, dtype=bool)
    else:
        keep_mask = np.array([tri not in forbidden for tri in triples], dtype=bool)

    idx_pool = np.nonzero(keep_mask)[0]
    if idx_pool.size == 0:
        return []

    # Optional candidate pool reduction by score
    if pool_size is not None and idx_pool.size > pool_size:
        pool_scores = scores_all[idx_pool]
        keep_local = np.argpartition(pool_scores, -pool_size)[-pool_size:]
        idx_pool = idx_pool[keep_local]

    H, R, T, tok2id = _build_token_ids([triples[i] for i in idx_pool], tok2id)
    scores = scores_all[idx_pool]

    m = idx_pool.size
    k = min(k, m)

    # max similarity to selected so far (initialize zeros)
    max_sim = np.zeros(m, dtype=np.float32)

    # To avoid reselection, we'll mark picked items by setting their score to -inf after each pick.
    chosen_local = []

    for _ in range(k):
        obj = (1.0 - lambda_div) * scores - lambda_div * max_sim
        j = int(np.argmax(obj))
        chosen_local.append(j)

        # Update max_sim against the newly selected item (vectorized)
        sim = ( (H == H[j]).astype(np.float32)
              + (R == R[j]).astype(np.float32)
              + (T == T[j]).astype(np.float32) ) / 3.0
        max_sim = np.maximum(max_sim, sim)

        # Exclude the selected index
        scores[j] = -np.inf
        max_sim[j] = np.inf

    chosen_global = idx_pool[np.array(chosen_local, dtype=int)]
    return [results[i] for i in chosen_global]

