from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F

Triple = Tuple[str, str, str]

@dataclass
class FGSMRemovalRecord:
    clean_triple: Triple
    worst_corruption: Triple
    corruption_type: str          # "head" | "rel" | "tail"
    clean_loss: float
    worst_loss: float
    delta_loss: float
    adv_prob: float               # sigmoid(logit) on the worst corruption
    grad_norm: float              # ||[g_h,g_r,g_t]||_2

def _fgsm_step(grad: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    if norm == "linf":
        return eps * grad.sign()
    elif norm == "l2":
        return eps * grad / (grad.norm() + 1e-12)
    else:
        raise ValueError("norm must be 'linf' or 'l2'")

def _logits_for_indices(model, h_i: int, r_i: int, t_i: int, device) -> torch.Tensor:
    idxs = torch.tensor([[h_i, r_i, t_i]], dtype=torch.long, device=device)
    return model.forward_triples(idxs).view(())

def _topk_nearest_excluding(vec: torch.Tensor,
                            table: torch.Tensor,
                            exclude_idx: int,
                            k: int) -> torch.Tensor:
    # vec: (d,), table: (N,d)
    with torch.no_grad():
        d = torch.cdist(vec.unsqueeze(0), table)[0]  # (N,)
        d[exclude_idx] = float("inf")
        k = min(k, table.size(0) - 1)
        nn_idx = torch.topk(d, k=k, largest=False).indices
    return nn_idx  # (k,)

def select_triples_to_remove_fgsm(
    triples: List[Triple],
    oracle,
    budget: int,
    *,
    eps: float = 0.25,
    norm: str = "linf",                  # "linf" or "l2"
    topk_neighbors: int = 32,            # k for discrete snapping
    seed: int = 0,
    return_audit: bool = True
) -> Tuple[List[Triple], List[Triple], Optional[List[FGSMRemovalRecord]]]:
    """
    Rank triples by FGSM attackability and select top-`budget` to remove.
    Returns (to_remove, kept, audit_records) where `audit_records` is detailed per-triple info (optional).

    Notes:
    - This is a one-shot scoring pass with the current model parameters.
    - For a stronger deletion attack, delete and then retrain/fine-tune your model on the pruned KG.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    model = oracle.model
    device = next(model.parameters()).device
    model.train(False)

    E2I: Dict[str, int] = oracle.entity_to_idx
    R2I: Dict[str, int] = oracle.relation_to_idx
    I2E: Dict[int, str] = {i: e for e, i in E2I.items()}
    I2R: Dict[int, str] = {i: r for r, i in R2I.items()}

    ent_emb = model.entity_embeddings
    rel_emb = model.relation_embeddings
    # Frozen copies for neighbor search
    E = ent_emb.weight.detach()
    R = rel_emb.weight.detach()

    # Enable grads for this pass
    for p in model.parameters():
        p.requires_grad_(True)

    records: List[FGSMRemovalRecord] = []

    for (h, r, t) in triples:
        h_i, r_i, t_i = E2I[h], R2I[r], E2I[t]

        model.zero_grad(set_to_none=True)
        # 1) Clean loss
        z_clean = _logits_for_indices(model, h_i, r_i, t_i, device)
        y_pos = torch.ones((), device=device)
        loss_clean = F.binary_cross_entropy_with_logits(z_clean, y_pos)
        loss_clean.backward()

        # 2) Grads wrt embedding rows
        GE = ent_emb.weight.grad
        GR = rel_emb.weight.grad
        if GE is None or GR is None:
            raise RuntimeError("Embedding grads are None (ensure model exposes grads).")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h, g_r, g_t = GE[h_i], GR[r_i], GE[t_i]
        grad_norm = torch.linalg.vector_norm(torch.cat([g_h.flatten(), g_r.flatten(), g_t.flatten()])).item()

        # 3) FGSM steps in embedding space (untargeted: maximize loss)
        Eh_adv = (E[h_i] + _fgsm_step(g_h, eps, norm)).detach()
        Er_adv = (R[r_i] + _fgsm_step(g_r, eps, norm)).detach()
        Et_adv = (E[t_i] + _fgsm_step(g_t, eps, norm)).detach()

        # 4) Discrete neighbors (top-k), excluding the original indices
        h_cands = _topk_nearest_excluding(Eh_adv, E, h_i, topk_neighbors).tolist()
        r_cands = _topk_nearest_excluding(Er_adv, R, r_i, topk_neighbors).tolist()
        t_cands = _topk_nearest_excluding(Et_adv, E, t_i, topk_neighbors).tolist()

        # 5) Score head-only / rel-only / tail-only corruptions; pick the worst loss
        best_loss = -float("inf")
        best_cand = (h_i, r_i, t_i)
        best_type = "none"

        with torch.no_grad():
            # head
            for hh in h_cands:
                z = _logits_for_indices(model, hh, r_i, t_i, device)
                L = F.binary_cross_entropy_with_logits(z, y_pos).item()
                if L > best_loss:
                    best_loss = L
                    best_cand = (hh, r_i, t_i)
                    best_type = "head"
            # relation
            for rr in r_cands:
                z = _logits_for_indices(model, h_i, rr, t_i, device)
                L = F.binary_cross_entropy_with_logits(z, y_pos).item()
                if L > best_loss:
                    best_loss = L
                    best_cand = (h_i, rr, t_i)
                    best_type = "rel"
            # tail
            for tt in t_cands:
                z = _logits_for_indices(model, h_i, r_i, tt, device)
                L = F.binary_cross_entropy_with_logits(z, y_pos).item()
                if L > best_loss:
                    best_loss = L
                    best_cand = (h_i, r_i, tt)
                    best_type = "tail"

            adv_prob = torch.sigmoid(
                _logits_for_indices(model, *best_cand, device)
            ).item()

        rec = FGSMRemovalRecord(
            clean_triple=(h, r, t),
            worst_corruption=(I2E[best_cand[0]], I2R[best_cand[1]], I2E[best_cand[2]]),
            corruption_type=best_type,
            clean_loss=loss_clean.item(),
            worst_loss=best_loss,
            delta_loss=(best_loss - loss_clean.item()),
            adv_prob=adv_prob,
            grad_norm=grad_norm
        )
        records.append(rec)

    # 6) Rank original triples by delta loss and pick top-`budget` to remove
    records.sort(key=lambda z: (z.delta_loss, z.worst_loss, z.grad_norm), reverse=True)
    budget = max(0, min(budget, len(records)))
    to_remove = [rec.clean_triple for rec in records[:budget]]
    kept = [rec.clean_triple for rec in records[budget:]]

    return to_remove, kept, (records if return_audit else None)
