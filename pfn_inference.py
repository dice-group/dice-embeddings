"""Inference and evaluation routines for GraphPFN.

Exposes
-------
- ``evaluate`` — transductive link-prediction evaluation (MRR, MR, Hits@k).
- ``infer``    — top-k tail prediction for a (head, relation, ?) query.
- ``score_triple`` — Monte Carlo N-pass probability estimate for a given triple.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from pfn_dataset import _encode_strings
from pfn_model import TriplePFN


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(
    model: TriplePFN,
    train_file: str,
    test_file: str,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate a trained GraphPFN on transductive link-prediction.

    Implements the two-phase GraphPFN workflow:

    1. **Pre-training** (:func:`train`): meta-train on diverse KG episodes so
       the model learns in-context triple scoring from a single forward pass.
    2. **Evaluation** (this function): provide the full ``train.txt`` as the
       in-context support and rank every test triple against all entity
       candidates by binary score.

    For each test query ``(h, r, t*)`` the model scores ``(h, r, t_i)`` for
    **every entity** ``t_i`` in the training vocabulary and ranks ``t*`` by
    its score.  No constraint is imposed on whether ``t*`` appears in the
    support: the model is free to score any entity.

    Parameters
    ----------
    model : TriplePFN
        A trained model returned by :func:`train` or loaded from a checkpoint.
    train_file : str
        Path to ``train.txt`` (whitespace-separated string tokens per line).
        All triples are loaded as the shared in-context support.
    test_file : str
        Path to ``test.txt``.
    device : torch.device or None
        Inference device.  Defaults to the device of model parameters.
    batch_size : int
        Number of candidate triples scored per forward pass per test query.
        Reduce if OOM.

    Returns
    -------
    dict
        Keys: ``"MRR"``, ``"MR"``, ``"Hits@1"``, ``"Hits@3"``, ``"Hits@10"``.

    Examples
    --------
    >>> model = train(num_epochs=3000, kg_dir="KGs/")
    >>> results = evaluate(model, "KGs/UMLS/train.txt", "KGs/UMLS/test.txt")
    >>> print(results)
    {'MRR': 0.82, 'MR': 3.1, 'Hits@1': 0.71, 'Hits@3': 0.93, 'Hits@10': 0.99}
    """
    if device is None:
        device = next(model.parameters()).device

    # ── 1. Parse train.txt and build vocab ───────────────────────────────────
    entity_vocab: Dict[str, int] = {}     # string token → local 0-based int
    relation_vocab: Dict[str, int] = {}
    train_triples: List[Tuple[int, int, int]] = []
    with open(train_file) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            h_s, r_s, t_s = parts
            for tok in (h_s, t_s):
                if tok not in entity_vocab:
                    entity_vocab[tok] = len(entity_vocab)
            if r_s not in relation_vocab:
                relation_vocab[r_s] = len(relation_vocab)
            train_triples.append(
                (entity_vocab[h_s], relation_vocab[r_s], entity_vocab[t_s])
            )

    n_ent = len(entity_vocab)
    n_rel = len(relation_vocab)
    print(f"  Train: {len(train_triples):,} triples | {n_ent} entities | {n_rel} relations")

    # ── 2. Encode entity/relation strings via frozen SentenceTransformer ──────
    entity_strings   = [tok for tok, _ in sorted(entity_vocab.items(),   key=lambda x: x[1])]
    relation_strings = [tok for tok, _ in sorted(relation_vocab.items(), key=lambda x: x[1])]
    print("  Encoding entity/relation strings with SentenceTransformer...")
    entity_embs   = _encode_strings(entity_strings).to(device)    # (n_ent, ST_DIM)
    relation_embs = _encode_strings(relation_strings).to(device)  # (n_rel, ST_DIM)

    # ── 3. Build support tensor from all training triples ─────────────────────
    h_list = [h for h, _r, _t in train_triples]
    r_list = [_r for _h, _r, _t in train_triples]
    t_list = [_t for _h, _r, _t in train_triples]
    sup_h = entity_embs[h_list]    # (S, ST_DIM)
    sup_r = relation_embs[r_list]  # (S, ST_DIM)
    sup_t = entity_embs[t_list]    # (S, ST_DIM)
    support_tensor = torch.stack([sup_h, sup_r, sup_t], dim=1)  # (S, 3, ST_DIM)
    sup = support_tensor.unsqueeze(0)  # (1, S, 3, ST_DIM)

    # ── 4. Parse test.txt ─────────────────────────────────────────────────────
    test_queries: List[Tuple[int, int, int]] = []   # (h_idx, r_idx, t_idx) local
    n_skipped = 0
    with open(test_file) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            h_s, r_s, t_s = parts
            if h_s not in entity_vocab or t_s not in entity_vocab or r_s not in relation_vocab:
                n_skipped += 1    # OOV (inductive) — skip
                continue
            test_queries.append((
                entity_vocab[h_s],
                relation_vocab[r_s],
                entity_vocab[t_s],
            ))

    if n_skipped:
        print(f"  Skipped {n_skipped} test triples with OOV tokens.")
    print(f"  Test:  {len(test_queries):,} queries")
    print(f"  Scoring each query against {n_ent} entity candidates...")

    # ── 5. Rank each test query ───────────────────────────────────────────────
    # For every test triple (h, r, t*) score (h, r, t_i) for every entity t_i
    # in the vocabulary by batching over candidates.
    model.eval()

    ranks: List[float] = []
    hits1 = hits3 = hits10 = 0

    for qi, (q_h, q_r, q_t) in enumerate(test_queries):
        if qi % 100 == 0 and qi > 0:
            print(f"    {qi}/{len(test_queries)} queries done")

        q_h_emb = entity_embs[q_h]    # (ST_DIM,)
        q_r_emb = relation_embs[q_r]  # (ST_DIM,)

        # Score all (q_h, q_r, t_i) for t_i in [0, n_ent) in batches.
        all_scores: List[float] = []
        for cstart in range(0, n_ent, batch_size):
            c_embs = entity_embs[cstart : cstart + batch_size]  # (C, ST_DIM)
            C = c_embs.shape[0]

            sup_c    = sup.expand(C, -1, -1, -1)                          # (C, S, 3, ST_DIM)
            q_h_exp  = q_h_emb.unsqueeze(0).expand(C, -1)                # (C, ST_DIM)
            q_r_exp  = q_r_emb.unsqueeze(0).expand(C, -1)                # (C, ST_DIM)
            q_triples = torch.stack([q_h_exp, q_r_exp, c_embs], dim=1)   # (C, 3, ST_DIM)

            with torch.no_grad():
                scores = model(sup_c, q_triples)   # (C,) logits
            all_scores.extend(scores.tolist())

        # Rank the true tail q_t directly by its index (no permutation needed).
        tgt_score = all_scores[q_t]
        rank = sum(1 for s in all_scores if s > tgt_score) + 1   # 1-based

        ranks.append(rank)
        hits1  += int(rank <= 1)
        hits3  += int(rank <= 3)
        hits10 += int(rank <= 10)

    total = len(ranks)
    if total == 0:
        return {"MRR": 0.0, "MR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    return {
        "MRR":     sum(1.0 / r for r in ranks) / total,
        "MR":      sum(ranks) / total,
        "Hits@1":  hits1  / total,
        "Hits@3":  hits3  / total,
        "Hits@10": hits10 / total,
    }


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def infer(
    model: TriplePFN,
    query_h: str,
    query_r: str,
    support: List[Tuple[str, str, str]],
    k: int = 5,
    device: Optional[torch.device] = None,
) -> List[Tuple[str, float]]:
    """Predict the top-k tail entities for a (query_h, query_r, ?) query.

    All inputs are plain string tokens — no integer ID mapping required.
    Entity/relation strings are encoded via the frozen all-MiniLM-L6-v2
    SentenceTransformer.  Candidates are every entity (head or tail) that
    appears in the support context.

    Parameters
    ----------
    model : TriplePFN
        A meta-trained TriplePFN instance.
    query_h : str
        Head entity string token (e.g. ``"slovakia"``).
    query_r : str
        Relation string token (e.g. ``"neighbor"``).
    support : list of (head, relation, tail) string triples
        In-context triples that form the observed knowledge base for this
        query.  Must be non-empty.
    k : int, default 5
        Number of top candidates to return.
    device : torch.device or None
        Inference device.  Defaults to the device of model parameters.

    Returns
    -------
    list of (entity_str, score)
        Top-k ``(entity_string, logit)`` pairs sorted by descending score.

    Examples
    --------
    >>> from graph_pfn import train
    >>> from pfn_inference import infer
    >>> model = train(num_epochs=3000, kg_dir="KGs/")
    >>> support = [
    ...     ("slovakia", "neighbor", "ukraine"),
    ...     ("slovakia", "neighbor", "hungary"),
    ...     ("slovakia", "neighbor", "austria"),
    ... ]
    >>> top3 = infer(model, "slovakia", "neighbor", support, k=3)
    >>> for entity, score in top3:
    ...     print(f"{entity:<20}  {score:.4f}")
    """
    if device is None:
        device = next(model.parameters()).device

    if not support:
        raise ValueError("'support' must contain at least one triple.")

    # Collect unique entity/relation strings in insertion order.
    unique_entities  = list(dict.fromkeys(
        tok for h, _r, t in support for tok in (h, t)
    ))
    unique_relations = list(dict.fromkeys(_r for _h, _r, _t in support))

    entity_to_idx   = {e: i for i, e in enumerate(unique_entities)}
    relation_to_idx = {r: i for i, r in enumerate(unique_relations)}

    # Encode support strings; encode query tokens separately if novel.
    entity_embs   = _encode_strings(unique_entities).to(device)   # (n_ent, ST_DIM)
    relation_embs = _encode_strings(unique_relations).to(device)  # (n_rel, ST_DIM)

    extra_entity_str   = [] if query_h in entity_to_idx else [query_h]
    extra_relation_str = [] if query_r in relation_to_idx else [query_r]

    if extra_entity_str:
        extra_e = _encode_strings(extra_entity_str).to(device)
        entity_to_idx[query_h]   = len(unique_entities)
        entity_embs = torch.cat([entity_embs, extra_e], dim=0)

    if extra_relation_str:
        extra_r = _encode_strings(extra_relation_str).to(device)
        relation_to_idx[query_r] = len(unique_relations)
        relation_embs = torch.cat([relation_embs, extra_r], dim=0)

    # Build support tensor (S, 3, ST_DIM).
    sup_h = entity_embs[[entity_to_idx[h] for h, _r, _t in support]]    # (S, ST_DIM)
    sup_r = relation_embs[[relation_to_idx[r] for _h, r, _t in support]] # (S, ST_DIM)
    sup_t = entity_embs[[entity_to_idx[t] for _h, _r, t in support]]    # (S, ST_DIM)
    support_tensor = torch.stack([sup_h, sup_r, sup_t], dim=1)           # (S, 3, ST_DIM)

    # Candidates: every entity from the support.
    candidates = unique_entities   # ordered list of strings
    cand_embs  = entity_embs[:len(unique_entities)]   # (C, ST_DIM) — support entities only
    C = len(candidates)

    q_h_emb = entity_embs[entity_to_idx[query_h]]    # (ST_DIM,)
    q_r_emb = relation_embs[relation_to_idx[query_r]] # (ST_DIM,)

    sup_c    = support_tensor.unsqueeze(0).expand(C, -1, -1, -1)  # (C, S, 3, ST_DIM)
    q_h_exp  = q_h_emb.unsqueeze(0).expand(C, -1)                 # (C, ST_DIM)
    q_r_exp  = q_r_emb.unsqueeze(0).expand(C, -1)                 # (C, ST_DIM)
    q_triples = torch.stack([q_h_exp, q_r_exp, cand_embs], dim=1) # (C, 3, ST_DIM)

    model.eval()
    with torch.no_grad():
        scores = model(sup_c, q_triples)   # (C,) logits

    k = min(k, C)
    topk_scores, topk_idx = torch.topk(scores, k)
    return [
        (candidates[int(i)], float(s))
        for i, s in zip(topk_idx.tolist(), topk_scores.tolist())
    ]


# ---------------------------------------------------------------------------
# TRIPLE SCORING
# ---------------------------------------------------------------------------

def score_triple(
    model: TriplePFN,
    triple_h: str,
    triple_r: str,
    triple_t: str,
    data_file: str,
    n: int = 10,
    context_size: int = 32,
    device: Optional[torch.device] = None,
) -> float:
    """Estimate the probability that a given triple is true via Monte Carlo context sampling.

    Loads triples from *data_file*, then performs *n* forward passes over the
    same query triple ``(triple_h, triple_r, triple_t)``.  Each pass uses a
    freshly and independently sampled support context of *context_size* triples
    drawn uniformly at random from the file.  The returned score is the average
    sigmoid probability across all *n* passes.

    Parameters
    ----------
    model : TriplePFN
        A meta-trained :class:`TriplePFN` instance.
    triple_h : str
        Head entity string of the triple to score.
    triple_r : str
        Relation string of the triple to score.
    triple_t : str
        Tail entity string of the triple to score.
    data_file : str
        Path to a whitespace-separated triple file (one ``h r t`` per line).
        Used both as the candidate pool for random context sampling.
    n : int, default 10
        Number of independent forward passes (context re-samples).
    context_size : int, default 32
        Number of support triples sampled per forward pass.
    device : torch.device or None
        Inference device.  Defaults to the device of model parameters.

    Returns
    -------
    float
        Average sigmoid probability P(triple is true | context) in [0, 1],
        averaged over *n* random contexts.

    Examples
    --------
    **Python API**:

    .. code-block:: python

        from graph_pfn import train
        from pfn_inference import score_triple

        model = train(num_epochs=3000, kg_dir="KGs/")

        p_true = score_triple(
            model,
            triple_h="slovakia",
            triple_r="neighbor",
            triple_t="austria",
            data_file="KGs/Countries-S1/train.txt",
            n=10,
            context_size=32,
        )
        print(f"P(slovakia neighbor austria) = {p_true:.4f}")
    """
    if device is None:
        device = next(model.parameters()).device

    # ── Load triples from file ────────────────────────────────────────────
    all_rows: List[Tuple[str, str, str]] = []
    with open(data_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                all_rows.append((parts[0], parts[1], parts[2]))

    if not all_rows:
        raise ValueError(f"No valid triples found in '{data_file}'.")

    # ── Encode all unique tokens once ─────────────────────────────────────
    all_entity_strings   = list(dict.fromkeys(tok for h, _r, t in all_rows for tok in (h, t)))
    all_relation_strings = list(dict.fromkeys(_r for _h, _r, _t in all_rows))

    for extra_tok, lst in (
        (triple_h, all_entity_strings),
        (triple_t, all_entity_strings),
        (triple_r, all_relation_strings),
    ):
        if extra_tok not in lst:
            lst.append(extra_tok)

    entity_to_idx   = {e: i for i, e in enumerate(all_entity_strings)}
    relation_to_idx = {r: i for i, r in enumerate(all_relation_strings)}

    entity_embs   = _encode_strings(all_entity_strings).to(device)    # (n_ent, ST_DIM)
    relation_embs = _encode_strings(all_relation_strings).to(device)  # (n_rel, ST_DIM)

    # Pre-build query embedding (constant across passes).
    q_h_emb = entity_embs[entity_to_idx[triple_h]]    # (ST_DIM,)
    q_r_emb = relation_embs[relation_to_idx[triple_r]] # (ST_DIM,)
    q_t_emb = entity_embs[entity_to_idx[triple_t]]    # (ST_DIM,)
    query_emb = torch.stack([q_h_emb, q_r_emb, q_t_emb], dim=0).unsqueeze(0)  # (1, 3, ST_DIM)

    # Pre-index rows for fast sampling.
    h_idxs = [entity_to_idx[h]   for h, _r, _t in all_rows]
    r_idxs = [relation_to_idx[r] for _h, r, _t in all_rows]
    t_idxs = [entity_to_idx[t]   for _h, _r, t in all_rows]
    N_rows = len(all_rows)
    ctx = min(context_size, N_rows)

    model.eval()
    logits: List[float] = []
    with torch.no_grad():
        for _ in range(n):
            idxs = random.sample(range(N_rows), ctx) if N_rows > ctx else list(range(N_rows))
            sup_h = entity_embs[[h_idxs[i] for i in idxs]]    # (ctx, ST_DIM)
            sup_r = relation_embs[[r_idxs[i] for i in idxs]]  # (ctx, ST_DIM)
            sup_t = entity_embs[[t_idxs[i] for i in idxs]]    # (ctx, ST_DIM)
            support_tensor = torch.stack([sup_h, sup_r, sup_t], dim=1).unsqueeze(0)  # (1, ctx, 3, ST_DIM)
            logit = model(support_tensor, query_emb)           # scalar or (1,)
            logits.append(float(logit.item() if logit.dim() == 0 else logit[0].item()))

    avg_logit = sum(logits) / len(logits)
    return float(torch.sigmoid(torch.tensor(avg_logit)).item())
