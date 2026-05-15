"""Inference and evaluation routines for GraphPFN.

Exposes
-------
- ``evaluate`` — transductive link-prediction evaluation (MRR, MR, Hits@k).
- ``infer``    — top-k tail prediction for a (head, relation, ?) query.
- ``score_triple`` — Monte Carlo N-pass probability estimate for a given triple.

This module also provides a CLI:

    python pfn_inference.py infer --model model.pt --train-file KGs/Countries-S1/train.txt --head slovakia --relation neighbor --k 5
    python pfn_inference.py score --model model.pt --data KGs/Countries-S1/train.txt --triple slovakia neighbor austria --n 10
"""

import argparse
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

from pfn.dataset import _encode_strings
from pfn.model import TriplePFN


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def visualize_triple_scoring(
    model: TriplePFN,
    query_h: str,
    query_r: str,
    query_t: str,
    support: List[Tuple[str, str, str]],
    device: Optional[torch.device] = None,
    top_k_support: int = 10,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """Visualize how the model scores a query triple given support context.

    This function provides insight into the model's reasoning by showing:
    1. The support context (in-context knowledge)
    2. The query triple being scored
    3. The model's score/logit
    4. Attention patterns (which support triples the model focuses on)
    5. Matplotlib visualization of attention weights

    Parameters
    ----------
    model : TriplePFN
        A meta-trained TriplePFN instance.
    query_h : str
        Head entity of the query triple.
    query_r : str
        Relation of the query triple.
    query_t : str
        Tail entity of the query triple.
    support : list of (head, relation, tail) string triples
        In-context support triples (the knowledge base).
    device : torch.device or None
        Inference device. Defaults to model's device.
    top_k_support : int, default 10
        Number of top support triples to display (by attention weight).
    save_path : str or None
        Path to save the attention visualization plot (e.g., "attention.png").
    show_plot : bool, default True
        Whether to display the plot interactively.

    Examples
    --------
    >>> from pfn import TriplePFN
    >>> from pfn.inference import visualize_triple_scoring
    >>> model = TriplePFN()
    >>> model.load_state_dict(torch.load("model.pt"))
    >>> support = [
    ...     ("slovakia", "neighbor", "ukraine"),
    ...     ("slovakia", "neighbor", "austria"),
    ...     ("austria", "neighbor", "germany"),
    ... ]
    >>> visualize_triple_scoring(
    ...     model, "slovakia", "neighbor", "poland",
    ...     support, top_k_support=5
    ... )
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if device is None:
        device = next(model.parameters()).device

    if not support:
        raise ValueError("'support' must contain at least one triple.")

    model.eval()
    model.to(device)
    
    # Use the new forward_with_attention method
    logit, attention_weights = model.forward_with_attention(
        support_triples=support,
        query_triple=(query_h, query_r, query_t)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
    
    # attention_weights shape: (num_layers, num_heads, S+1, S+1)
    # Extract attention from query token (last position) to support tokens
    # Average over all layers and heads
    attn_from_query = attention_weights[:, :, -1, :-1].cpu().numpy()  # (num_layers, num_heads, S)
    attn_avg = attn_from_query.mean(axis=(0, 1))  # (S,)

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION OUTPUT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("  GraphPFN Triple Scoring Visualization")
    print("═" * 80)

    print("\n┌─ QUERY TRIPLE " + "─" * 64)
    print(f"│  ({query_h}, {query_r}, {query_t})")
    print(f"│")
    print(f"│  Model Score (logit):  {float(logit):>8.4f}")
    print(f"│  Probability:          {float(prob):>8.4f}  {'✓ LIKELY TRUE' if prob > 0.5 else '✗ LIKELY FALSE'}")
    print("└" + "─" * 79)

    print("\n┌─ SUPPORT CONTEXT " + "─" * 61)
    print(f"│  Total support triples: {len(support)}")
    print("│")
    
    # Show top-k support triples SORTED BY ATTENTION
    # Sort by attention weight (descending)
    sorted_indices = np.argsort(attn_avg)[::-1]
    k = min(top_k_support, len(support))
    
    print(f"│  Support triples (showing top-{k} by attention):")
    print("│  " + "─" * 76)
    print(f"│  {'#':<4}  {'Head':<20}  {'Relation':<20}  {'Tail':<20}  {'Attn%':<6}")
    print("│  " + "─" * 76)
    
    for i, idx in enumerate(sorted_indices[:k], start=1):
        h, r, t = support[idx]
        attn_pct = attn_avg[idx] * 100
        
        # Highlight triples that share entities/relations with query
        marker = ""
        if h == query_h or t == query_h or h == query_t or t == query_t:
            marker = " ◄─ entity"
        elif r == query_r:
            marker = " ◄─ relation"
        
        print(f"│  {i:<4}  {h:<20}  {r:<20}  {t:<20}  {attn_pct:5.2f}{marker}")
    
    if len(support) > k:
        print(f"│  ...  ({len(support) - k} more triples)")
    
    print("└" + "─" * 79)

    print("\n┌─ MODEL INSIGHTS " + "─" * 62)
    print("│")
    print("│  How the model works:")
    print("│  • No positional encodings → treats support as an UNORDERED SET")
    print("│  • Self-attention aggregates evidence from all support triples")
    print("│  • Variable context size: trained on S=32, can handle S=32 to S=1000+")
    print("│  • Attention weights reveal which support triples are most relevant")
    print("│")
    print("│  Key architectural features:")
    print(f"│  • Embedding dim: {model.embed_dim}")
    print(f"│  • Transformer layers: {len(model.transformer.layers)}")
    print(f"│  • Attention heads: {model.transformer.layers[0].self_attn.num_heads}")
    print("│")
    
    # Analyze support relevance
    shared_entities = sum(1 for h, r, t in support if h in (query_h, query_t) or t in (query_h, query_t))
    shared_relations = sum(1 for h, r, t in support if r == query_r)
    
    print(f"│  Context relevance analysis:")
    print(f"│  • Triples sharing entities with query:   {shared_entities:>3} / {len(support)} ({100*shared_entities/len(support):.1f}%)")
    print(f"│  • Triples with same relation as query:   {shared_relations:>3} / {len(support)} ({100*shared_relations/len(support):.1f}%)")
    print("│")
    print("└" + "─" * 79)
    
    print("\n" + "═" * 80 + "\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MATPLOTLIB ATTENTION VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(support) * 0.25)))
    
    # LEFT: Attention scores bar chart (top K support triples)
    display_k = min(20, len(support))
    top_indices = sorted_indices[:display_k]
    top_attention = attn_avg[top_indices]
    
    triple_labels = []
    for idx in top_indices:
        h, r, t = support[idx]
        label = f"{idx+1:2d}. {h[:15]:<15} {r[:12]:<12} {t[:15]:<15}"
        if h == query_h or t == query_t or h == query_t or t == query_h:
            label += " ◄ entity"
        elif r == query_r:
            label += " ◄ relation"
        triple_labels.append(label)
    
    y_pos = np.arange(len(triple_labels))
    colors = ['steelblue' if attn > attn_avg.mean() else 'lightsteelblue' for attn in top_attention]
    ax1.barh(y_pos, top_attention, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(triple_labels, fontsize=9, family='monospace')
    ax1.set_xlabel('Attention Weight (avg over layers & heads)', fontsize=11)
    ax1.set_title(f'Top-{display_k} Support Triples by Attention\\nQuery: ({query_h}, {query_r}, {query_t})', 
                  fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=attn_avg.mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {attn_avg.mean():.4f}')
    ax1.legend(loc='lower right', fontsize=9)
    
    # RIGHT: Full attention matrix heatmap
    # Show attention from all tokens to all tokens (averaged over layers & heads)
    full_attn = attention_weights.mean(dim=(0, 1)).cpu().numpy()  # (S+1, S+1)
    
    im = ax2.imshow(full_attn, cmap='YlOrRd', aspect='auto', interpolation='nearest', vmin=0)
    ax2.set_xlabel('To Token', fontsize=11)
    ax2.set_ylabel('From Token', fontsize=11)
    ax2.set_title('Full Attention Matrix\\n(all layers & heads averaged)', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Mark query token position
    query_pos = len(support)
    ax2.axhline(y=query_pos - 0.5, color='blue', linewidth=2.5, linestyle='--', alpha=0.7, label='Query token')
    ax2.axvline(x=query_pos - 0.5, color='blue', linewidth=2.5, linestyle='--', alpha=0.7)
    
    # Ticks
    if len(support) <= 20:
        tick_positions = list(range(len(support))) + [query_pos]
        tick_labels = [str(i+1) for i in range(len(support))] + ['Q']
    else:
        tick_step = max(1, len(support) // 10)
        tick_positions = list(range(0, len(support), tick_step)) + [query_pos]
        tick_labels = [str(i+1) for i in range(0, len(support), tick_step)] + ['Q']
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.suptitle(
        f'GraphPFN Attention Visualization\n'
        f'Score: {logit:.4f} | Probability: {prob:.4f} | Support Context: {len(support)} triples',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Attention visualization saved to: {save_path}\n")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(
    model: TriplePFN,
    train_file: str,
    test_file: str,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    context_size: Optional[int] = None,
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
        Retained for API compatibility. Evaluation now scores one candidate
        tail at a time to minimise peak GPU memory.
    context_size : int or None
        Maximum number of support triples fed to the model per query.  Should
        match the ``context_size`` used during training so the transformer sees
        sequences of the same length as those it was trained on.  When ``None``
        all training triples are used as support, which will cause a
        distribution shift if ``len(train_triples) > training context_size``.

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
    _ = batch_size  # Kept for backward compatibility; not used.

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

    # ── 3. Build support tensor from training triples ───────────────────────────
    h_list = [h for h, _r, _t in train_triples]
    r_list = [_r for _h, _r, _t in train_triples]
    t_list = [_t for _h, _r, _t in train_triples]

    if context_size is not None and len(train_triples) > context_size:
        sample_idxs = random.sample(range(len(train_triples)), context_size)
        h_list = [h_list[i] for i in sample_idxs]
        r_list = [r_list[i] for i in sample_idxs]
        t_list = [t_list[i] for i in sample_idxs]
        print(
            f"  Support capped: {len(train_triples):,} → {context_size} triples "
            f"(pass context_size=None to use all)."
        )
    elif context_size is None and len(train_triples) > 256:
        print(
            f"  Warning: using all {len(train_triples):,} train triples as support. "
            f"Pass context_size=<training context_size> to match training distribution."
        )

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
    # one candidate at a time to keep memory usage bounded.
    model.eval()

    ranks: List[float] = []
    hits1 = hits3 = hits10 = 0
    running_rr = 0.0
    progress = tqdm(test_queries, desc="Evaluating queries", unit="query")
    for qi, (q_h, q_r, q_t) in enumerate(progress, start=1):

        q_h_emb = entity_embs[q_h]    # (ST_DIM,)
        q_r_emb = relation_embs[q_r]  # (ST_DIM,)

        # Score all (q_h, q_r, t_i) for t_i in [0, n_ent), one by one.
        all_scores: List[float] = []
        with torch.no_grad():
            for cand_idx in range(n_ent):
                c_emb = entity_embs[cand_idx].unsqueeze(0)                 # (1, ST_DIM)
                q_triple = torch.stack(
                    [q_h_emb.unsqueeze(0), q_r_emb.unsqueeze(0), c_emb],
                    dim=1,
                )                                                           # (1, 3, ST_DIM)
                score = model(sup, q_triple)                                # scalar or (1,)
                all_scores.append(float(score.item() if score.dim() == 0 else score[0].item()))

        # Rank the true tail q_t directly by its index (no permutation needed).
        tgt_score = all_scores[q_t]
        rank = sum(1 for s in all_scores if s > tgt_score) + 1   # 1-based

        ranks.append(rank)
        hits1  += int(rank <= 1)
        hits3  += int(rank <= 3)
        hits10 += int(rank <= 10)
        running_rr += 1.0 / rank

        progress.set_postfix(
            {
                "MRR": f"{running_rr / qi:.4f}",
                "Hits@1": f"{hits1 / qi:.4f}",
                "Hits@3": f"{hits3 / qi:.4f}",
                "Hits@10": f"{hits10 / qi:.4f}",
            }
        )

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

    model.eval()
    sup_1 = support_tensor.unsqueeze(0)   # (1, S, 3, ST_DIM) — reused for every candidate
    scores_list: List[float] = []
    with torch.no_grad():
        for cand_idx in range(C):
            c_emb = cand_embs[cand_idx].unsqueeze(0)                        # (1, ST_DIM)
            q_triple = torch.stack(
                [q_h_emb.unsqueeze(0), q_r_emb.unsqueeze(0), c_emb], dim=1
            )                                                                # (1, 3, ST_DIM)
            score = model(sup_1, q_triple)                                   # scalar or (1,)
            scores_list.append(float(score.item() if score.dim() == 0 else score[0].item()))

    scores = torch.tensor(scores_list)

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
    unnormalized logit across all *n* passes.

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
        Average unnormalized logit (raw model score),
        averaged over *n* random contexts.

    Examples
    --------
    **Python API**:

    .. code-block:: python

        from graph_pfn import train
        from pfn_inference import score_triple

        model = train(num_epochs=3000, kg_dir="KGs/")

        logit = score_triple(
            model,
            triple_h="slovakia",
            triple_r="neighbor",
            triple_t="austria",
            data_file="KGs/Countries-S1/train.txt",
            n=10,
            context_size=32,
        )
        print(f"Logit(slovakia neighbor austria) = {logit:.4f}")
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
    print(f"Loaded {len(all_rows):,} triples from '{data_file}' for context sampling.")
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
    return avg_logit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: torch.device) -> Tuple[TriplePFN, dict]:
    """Load a TriplePFN checkpoint; return (model, run_config)."""
    ckpt = torch.load(model_path, map_location=device)
    run_cfg = ckpt.get("run_config", {}) if isinstance(ckpt, dict) else {}
    if isinstance(ckpt, dict) and "hparams" in ckpt:
        model = TriplePFN(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = TriplePFN()
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, run_cfg


def main() -> None:
    """CLI entrypoint for inference and triple scoring."""
    parser = argparse.ArgumentParser(description="GraphPFN inference and scoring.")
    subparsers = parser.add_subparsers(dest="command")

    infer_parser = subparsers.add_parser(
        "infer",
        help="Predict top-k tail entities for a (head, relation, ?) query.",
    )
    infer_parser.add_argument("--model", type=str, default="model.pt", help="Path to model checkpoint (.pt).")
    infer_parser.add_argument(
        "--train-file",
        "--data",
        dest="train_file",
        type=str,
        default="KGs/Countries-S1/train.txt",
        metavar="TRAIN_TXT",
        help="Path to train.txt whose triples are used as in-context support.",
    )
    infer_parser.add_argument("--head", type=str, default="slovakia", help="Head entity string for the query.")
    infer_parser.add_argument("--relation", type=str, default="neighbor", help="Relation string for the query.")
    infer_parser.add_argument(
        "--query",
        type=str,
        nargs=2,
        required=False,
        metavar=("HEAD", "RELATION"),
        help="Backward-compatible: --query <head> <relation>.",
    )
    infer_parser.add_argument("--k", type=int, default=5, help="Number of top-k predictions to display.")
    infer_parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        metavar="N",
        help="Max support triples used. Defaults to the value stored in the checkpoint.",
    )
    infer_parser.add_argument(
        "--support-size",
        type=int,
        default=None,
        metavar="N",
        help="Optional additional cap on support triples (applied after --context-size).",
    )
    infer_parser.add_argument("--show-support", action="store_true", help="Print support triples before scoring.")

    score_parser = subparsers.add_parser(
        "score",
        help="Compute unnormalized logit for a triple via Monte Carlo context sampling.",
    )
    score_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt).")
    score_parser.add_argument(
        "--triple",
        type=str,
        nargs=3,
        required=True,
        metavar=("HEAD", "RELATION", "TAIL"),
        help="Triple to score.",
    )
    score_parser.add_argument("--data", type=str, required=True, help="Data file for context sampling.")
    score_parser.add_argument("--n", type=int, default=10, help="Number of independent passes.")
    score_parser.add_argument("--context-size", type=int, default=32, help="Support triples per pass.")

    if len(sys.argv) == 1 or sys.argv[1] not in ("infer", "score", "-h", "--help"):
        sys.argv.insert(1, "infer")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "infer":
        if args.query is not None:
            if args.head is None:
                args.head = args.query[0]
            if args.relation is None:
                args.relation = args.query[1]

        missing = []
        if args.train_file is None:
            missing.append("--train-file/--data")
        if args.head is None:
            missing.append("--head or --query")
        if args.relation is None:
            missing.append("--relation or --query")
        if missing:
            infer_parser.error("the following arguments are required: " + ", ".join(missing))
        if args.support_size is not None and args.support_size <= 0:
            infer_parser.error("--support-size must be a positive integer")

        model, run_cfg = _load_model(args.model, device)

        if args.context_size is None:
            args.context_size = int(run_cfg.get("context_size", 128))
            print(f"Using checkpoint context_size={args.context_size} for inference.")
        if args.context_size <= 0:
            infer_parser.error("--context-size must be a positive integer")

        if run_cfg:
            print(
                "Loaded checkpoint run settings: "
                f"context_size={run_cfg.get('context_size')}, "
                f"support_sampler={run_cfg.get('support_sampler')}, "
                f"permute_support={run_cfg.get('permute_support')}"
            )

        support: List[Tuple[str, str, str]] = []
        with open(args.train_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 3:
                    support.append((parts[0], parts[1], parts[2]))

        total_support = len(support)
        effective_cap = args.context_size
        if args.support_size is not None:
            effective_cap = min(effective_cap, args.support_size)

        if total_support > effective_cap:
            support = support[:effective_cap]
            print(
                f"Support capped: {total_support:,} -> {len(support):,} triples "
                f"(context_size={args.context_size}, support_size={args.support_size})."
            )
        else:
            print(f"Support loaded: {len(support):,} triples from '{args.train_file}'.")

        if args.show_support:
            print(f"\n{'─'*52}")
            print(f"  {'#':<5}  {'Head':<20}  {'Relation':<15}  Tail")
            print(f"  {'─'*5}  {'─'*20}  {'─'*15}  {'─'*20}")
            for i, (h, r, t) in enumerate(support, start=1):
                print(f"  {i:<5}  {h:<20}  {r:<15}  {t}")
            print(f"{'─'*52}\n")

        results = infer(model, args.head, args.relation, support, k=args.k, device=device)

        print(f"\nTop-{args.k} predictions for ({args.head}, {args.relation}, ?):")
        print(f"  {'Entity':<30}  Logit")
        print(f"  {'-' * 30}  -----------")
        for rank, (entity, raw_score) in enumerate(results, start=1):
            print(f"  {rank}. {entity:<28}  {raw_score:>11.4f}")

    elif args.command == "score":
        ckpt = torch.load(args.model, map_location="cpu")
        if isinstance(ckpt, dict) and "hparams" in ckpt:
            model = TriplePFN(**ckpt["hparams"])
            model.load_state_dict(ckpt["state_dict"])
        else:
            model = TriplePFN()
            model.load_state_dict(ckpt)
        model.eval()

        h, r, t = args.triple
        logit = score_triple(model, h, r, t, args.data, n=args.n, context_size=args.context_size)
        print(f"Logit({h} {r} {t}) = {logit:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
