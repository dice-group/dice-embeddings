# new: utility
def fgsm_step(grad, eps, norm, targeted=False):
    if norm == "linf":
        step = grad.sign()
    elif norm == "l2":
        step = grad / (grad.norm() + 1e-12)
    else:
        raise ValueError("norm must be 'linf' or 'l2'")
    return (-eps * step) if targeted else (eps * step)

# new: pick top-k nearest indices and return the one that maximizes (or minimizes) loss
def best_neighbor_by_loss(vec_adv, table, orig_idx, score_fn, k=32, maximize=True):
    # vec_adv: (d,), table: (N, d)
    with torch.no_grad():
        d = torch.cdist(vec_adv.unsqueeze(0), table)[0]
        d[orig_idx] = float("inf")
        k = min(k, (d.numel() - 1))
        cand_idx = torch.topk(d, k=k, largest=False).indices  # k nearest
    # Evaluate objective and pick best
    best_i, best_obj = None, None
    for i in cand_idx.tolist():
        obj = score_fn(i)  # should return a scalar (loss or -loss depending on objective)
        if best_obj is None or (obj > best_obj if maximize else obj < best_obj):
            best_obj, best_i = obj, i
    return best_i

def select_adversarial_triples_fgsm(
    triples,
    corruption_type,
    oracle,
    seed,
    eps,
    norm,
    targeted=False,         # new: targeted FGSM
    topk_nn=32              # new: evaluate top-K neighbors by loss
):
    random.seed(seed); torch.manual_seed(seed)
    device = next(oracle.model.parameters()).device

    E2I = oracle.entity_to_idx; R2I = oracle.relation_to_idx
    I2E = {i: e for e, i in E2I.items()}; I2R = {i: r for r, i in R2I.items()}

    base_seen = {(E2I[h], R2I[r], E2I[t]) for (h, r, t) in triples}
    seen = set(base_seen)

    ent_emb = oracle.model.entity_embeddings
    rel_emb = oracle.model.relation_embeddings
    E = ent_emb.weight.detach()
    R = rel_emb.weight.detach()

    out = []

    for (h, r, t) in triples:
        h_i, r_i, t_i = E2I[h], R2I[r], E2I[t]
        for p in oracle.model.parameters(): p.requires_grad_(True)
        oracle.model.zero_grad(set_to_none=True); oracle.model.train(False)

        logits = logits_for_indices(oracle.model, h_i, r_i, t_i, device)
        y = torch.ones_like(logits)      # true triple
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        GE, GR = ent_emb.weight.grad, rel_emb.weight.grad
        if GE is None or GR is None: raise RuntimeError("Embedding grads are None.")
        if GE.is_sparse: GE = GE.to_dense()
        if GR.is_sparse: GR = GR.to_dense()

        g_h, g_r, g_t = GE[h_i], GR[r_i], GE[t_i]
        dh = fgsm_step(g_h, eps, norm, targeted)
        dr = fgsm_step(g_r, eps, norm, targeted)
        dt = fgsm_step(g_t, eps, norm, targeted)

        Eh = (E[h_i] + dh).detach()
        Er = (R[r_i] + dr).detach()
        Et = (E[t_i] + dt).detach()

        # define how to score a candidate index for each field
        with torch.no_grad():
            def loss_head(i):
                z = logits_for_indices(oracle.model, i, r_i, t_i, device)
                return F.binary_cross_entropy_with_logits(z, y).item()

            def loss_rel(i):
                z = logits_for_indices(oracle.model, h_i, i, t_i, device)
                return F.binary_cross_entropy_with_logits(z, y).item()

            def loss_tail(i):
                z = logits_for_indices(oracle.model, h_i, r_i, i, device)
                return F.binary_cross_entropy_with_logits(z, y).item()

        # choose corruption pattern
        if corruption_type == "all":
            # pick each field from top-K by loss
            h_adv_i = best_neighbor_by_loss(Eh, E, h_i, loss_head, k=topk_nn, maximize=not targeted)
            r_adv_i = best_neighbor_by_loss(Er, R, r_i, loss_rel,  k=topk_nn, maximize=not targeted)
            t_adv_i = best_neighbor_by_loss(Et, E, t_i, loss_tail, k=topk_nn, maximize=not targeted)
            cand = (h_adv_i, r_adv_i, t_adv_i)
        elif corruption_type == "head":
            h_adv_i = best_neighbor_by_loss(Eh, E, h_i, loss_head, k=topk_nn, maximize=not targeted)
            cand = (h_adv_i, r_i, t_i)
        elif corruption_type == "rel":
            r_adv_i = best_neighbor_by_loss(Er, R, r_i, loss_rel, k=topk_nn, maximize=not targeted)
            cand = (h_i, r_adv_i, t_i)
        elif corruption_type == "tail":
            t_adv_i = best_neighbor_by_loss(Et, E, t_i, loss_tail, k=topk_nn, maximize=not targeted)
            cand = (h_i, r_i, t_adv_i)
        elif corruption_type == "head-tail":
            h_adv_i = best_neighbor_by_loss(Eh, E, h_i, loss_head, k=topk_nn, maximize=not targeted)
            t_adv_i = best_neighbor_by_loss(Et, E, t_i, loss_tail, k=topk_nn, maximize=not targeted)
            cand = (h_adv_i, r_i, t_adv_i)
        elif corruption_type == "head-rel":
            h_adv_i = best_neighbor_by_loss(Eh, E, h_i, loss_head, k=topk_nn, maximize=not targeted)
            r_adv_i = best_neighbor_by_loss(Er, R, r_i, loss_rel, k=topk_nn, maximize=not targeted)
            cand = (h_adv_i, r_adv_i, t_i)
        elif corruption_type == "random-one":
            # unchanged, though you could also use the top-K machinery here
            with torch.no_grad():
                lh = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, nearest_idx_excluding(Eh, E, h_i), r_i, t_i, device), y).item()
                lr = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, h_i, nearest_idx_excluding(Er, R, r_i), t_i, device), y).item()
                lt = F.binary_cross_entropy_with_logits(
                    logits_for_indices(oracle.model, h_i, r_i, nearest_idx_excluding(Et, E, t_i), device), y).item()
            if (not targeted and lh >= lr and lh >= lt) or (targeted and lh <= lr and lh <= lt):
                cand = (nearest_idx_excluding(Eh, E, h_i), r_i, t_i)
            elif (not targeted and lr >= lt) or (targeted and lr <= lt):
                cand = (h_i, nearest_idx_excluding(Er, R, r_i), t_i)
            else:
                cand = (h_i, r_i, nearest_idx_excluding(Et, E, t_i))
        else:
            raise ValueError("Invalid corruption_type")

        # ensure new and unseen; if equal, try second-best neighbor(s) before giving up (omitted for brevity)

        if cand == (h_i, r_i, t_i) or cand in seen:
            continue
        seen.add(cand)

        with torch.no_grad():
            logits_adv = logits_for_indices(oracle.model, *cand, device)
            pred_prob = torch.sigmoid(logits_adv).item()

        out.append(((I2E[cand[0]], I2R[cand[1]], I2E[cand[2]]), (h, r, t), pred_prob))

    # ...centrality bookkeeping unchanged...
    return out
