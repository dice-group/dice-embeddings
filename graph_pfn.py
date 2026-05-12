"""
Graph Prior-Fitted Network (GraphPFN) for in-context link prediction.

Model
-----
GraphPFN is a **Prior-Fitted Network (PFN)** for knowledge graph link
prediction.  A PFN is meta-trained over a prior so that, at inference time,
it performs in-context prediction purely through a single forward pass —
without any gradient update on the target graph.

**Problem statement.**
Given a knowledge graph G = (E, R, T) with entity set E, relation set R,
and observed triples T ⊆ E × R × E, and a query triple (h_q, r_q, t_q),
predict whether the triple is true:

    s(h_q, r_q, t_q | T_ctx)  =  σ( f_θ(T_ctx, (h_q, r_q, t_q)) )

where T_ctx ⊆ T is a context window of S observed triples, σ is the
sigmoid function, and f_θ is the learned scoring network.  At evaluation
time, all candidate tails are scored and the target is ranked accordingly.

**Embeddings.**
Entity and relation tokens are represented using the frozen pre-trained
SentenceTransformer **all-MiniLM-L6-v2** (384-dimensional, no gradients).
No learnable embedding tables are maintained::

    e  →  ST(e)  ∈ ℝ^{384}   (frozen, from all-MiniLM-L6-v2)
    r  →  ST(r)  ∈ ℝ^{384}   (frozen, from all-MiniLM-L6-v2)

Two learned linear projections bring these down to the model working
dimension d::

    entity_proj   :  ℝ^{384} → ℝ^d
    relation_proj :  ℝ^{384} → ℝ^d

**Triple encoder.**
A two-layer MLP with GELU activations encodes any triple (h, r, t) —
whether a support triple or the complete query triple — into a single
token, preserving the distinct roles of head, relation, and tail:

    τ(h, r, t)  =  MLP_enc( [proj_e(ST(h)) ; proj_r(ST(r)) ; proj_e(ST(t))] )  ∈ ℝ^d

    MLP_enc :  ℝ^{3d}  →  ℝ^{2d}  →  ℝ^d
               Linear, GELU, Linear

**In-context sequence.**
The S support tokens and the query token are concatenated into a sequence
of length S + 1 and passed through a pre-norm Transformer encoder
(norm_first=True, L layers, H heads):

    Z  =  LayerNorm-drop( [τ(sup_1) ; … ; τ(sup_S) ; τ(query)] )
    Z' =  TransformerEncoder(Z)                              ∈ ℝ^{(S+1) × d}

The last token Z'_{S+1} aggregates information from all support triples
via full bidirectional self-attention.

**Prediction head (binary scorer).**
A two-layer MLP (``score_head``) maps the aggregated query token to a
scalar logit:

    f_θ(T_ctx, q)  =  MLP_score( Z'_{S+1} )                 ∈ ℝ

    MLP_score :  ℝ^d  →  ℝ^d  →  ℝ^1
                 Linear, GELU, Linear

    P(triple is true | T_ctx)  =  σ( f_θ(T_ctx, q) )

**Training objective.**
The model is meta-trained on tasks sampled from :class:`RichSubgraphPrior`
using binary cross-entropy loss:

    L  =  -  𝔼_{(T_ctx, q, y) ~ P}  [ y · log σ(f) + (1-y) · log(1-σ(f)) ]

Each episode samples a context window of triples from a real knowledge
graph.  The query triple is either a real triple (y=1) or a corrupted
triple with a random tail (y=0).  Because entity/relation identity is
carried by frozen SentenceTransformer embeddings, **no ID re-randomisation
is needed** — the model learns structural patterns directly from semantics.

Optimisation uses AdamW with cosine learning-rate annealing and gradient
clipping (‖g‖ ≤ 1).

**Ranking at evaluation time.**
For each test query (h, r, t*) the model scores (h, r, t_i) for every
entity t_i in the training vocabulary and ranks t* by its score.  This
removes the constraint that the answer must appear in the context.

Command-line usage
-------------------
The examples below use ``KGs/Countries-S1/train.txt``, which ships with
this repository.  Each line is ``head<TAB>relation<TAB>tail``, e.g.::

    western_africa  locatedin  africa
    slovakia        neighbor   ukraine
    belize          locatedin  americas

Train a model and save it::

    python graph_pfn.py train --epochs 3000 --save model.pt

Infer with an explicit support set (real triples from Countries-S1)::

    python graph_pfn.py infer --model model.pt \
        --query slovakia neighbor \
        --support slovakia,neighbor,ukraine slovakia,neighbor,hungary \
                  slovakia,neighbor,austria slovakia,neighbor,czechia \
        --k 5

Infer by sampling the support automatically from the training file::

    python graph_pfn.py infer --model model.pt \
        --query slovakia neighbor \
        --data KGs/Countries-S1/train.txt --context-size 32 --k 5

The data file must have one triple per line (tab- or space-separated
string tokens)::

    # KGs/Countries-S1/train.txt (excerpt)
    western_africa  locatedin  africa
    slovakia        neighbor   ukraine
    niger           neighbor   chad

Quick-start: inference (Python API)
-------------------------------------
After meta-training, use a :class:`TriplePFN` model to predict the missing
tail entity given a small support set of ``(head, relation, tail)`` **string**
triples and a query ``(head_str, relation_str, ?)``:

.. code-block:: python

    from graph_pfn import train, infer

    # 1. Meta-train the model (or load a checkpoint)
    model = train(num_epochs=3000, kg_dir="KGs/")

    # 2. Build support context as string triples — no ID mapping needed
    support = [
        ("slovakia", "neighbor", "ukraine"),
        ("slovakia", "neighbor", "hungary"),
        ("slovakia", "neighbor", "austria"),
        ("slovakia", "neighbor", "czechia"),
    ]

    # 3. Run inference — returns (entity_string, logit) pairs
    top_k = infer(model, "slovakia", "neighbor", support, k=3)
    for entity, score in top_k:
        print(f"{entity:<20}  score={score:.4f}")
"""

import argparse
import os
import random
import sys
from collections import defaultdict  # noqa: F401  (used by subclasses / external callers)
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# ── Sentence-Transformer embedding cache ────────────────────────────────────
_ST_MODEL: Optional[SentenceTransformer] = None
_ST_DIM = 384   # output dimension of all-MiniLM-L6-v2


def _get_st_model() -> SentenceTransformer:
    """Lazy-load and cache sentence-transformers/all-MiniLM-L6-v2."""
    global _ST_MODEL
    if _ST_MODEL is None:
        print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _ST_MODEL.eval()
        for p in _ST_MODEL.parameters():
            p.requires_grad_(False)
    return _ST_MODEL


def _encode_strings(strings: List[str]) -> torch.Tensor:
    """Encode a list of strings → FloatTensor (N, 384) via all-MiniLM-L6-v2."""
    st = _get_st_model()
    embs = st.encode(strings, batch_size=512, show_progress_bar=False, convert_to_numpy=True)
    return torch.from_numpy(np.array(embs)).float()


# ---------------------------------------------------------------------------
# 1.  DATA PRIOR
# ---------------------------------------------------------------------------

def _load_real_triples(
    kg_dir: str,
    max_per_kg: int = 500,
) -> List[Tuple[List[Tuple[int, int, int]], torch.Tensor, torch.Tensor]]:
    """Load all ``train.txt`` files under *kg_dir* as episodic task pools.

    Each KG is kept separate so that tasks sample context triples from a single
    graph (preserving relational structure).  Entity and relation tokens are
    encoded to 384-dimensional vectors by the frozen all-MiniLM-L6-v2
    SentenceTransformer and stored as float tensors.  No ID re-randomisation
    is performed — semantic identity is captured by the embedding itself.

    Parameters
    ----------
    kg_dir : str
        Root directory to walk recursively for ``train.txt`` files.
    max_per_kg : int
        Maximum number of triples to retain per KG.  Large KGs are randomly
        downsampled so they do not dominate the task distribution.

    Returns
    -------
    List of ``(triples, entity_embs, relation_embs)`` tuples — one per KG.
    ``triples`` is a list of ``(h_idx, r_idx, t_idx)`` integer tuples that
    index into ``entity_embs`` / ``relation_embs``.
    ``entity_embs`` and ``relation_embs`` are FloatTensors of shape
    ``(n_ent, 384)`` and ``(n_rel, 384)`` produced by all-MiniLM-L6-v2.
    """
    pools = []
    for root, _dirs, files in os.walk(kg_dir):     # walk the directory tree recursively
        if "train.txt" not in files:               # skip folders that have no training file
            continue
        path = os.path.join(root, "train.txt")     # full path to this KG's training file
        raw: List[Tuple[str, str, str]] = []       # will hold raw string triples
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()       # split on any whitespace (tab or space)
                if len(parts) == 3:                # skip blank lines / malformed rows
                    raw.append((parts[0], parts[1], parts[2]))
        if not raw:                                # entirely empty file — nothing to learn from
            continue
        if len(raw) > max_per_kg:                  # large KGs (YAGO, FB15k) would otherwise
            raw = random.sample(raw, max_per_kg)   # dominate the task distribution; downsample

        entity_vocab: Dict[str, int] = {}          # string entity → local integer index
        relation_vocab: Dict[str, int] = {}        # string relation → local integer index
        triples: List[Tuple[int, int, int]] = []   # indexed (h, r, t) triples for this KG
        for h, r, t in raw:
            if h not in entity_vocab:              # assign new index the first time we see this entity
                entity_vocab[h] = len(entity_vocab)
            if t not in entity_vocab:              # same for the tail entity
                entity_vocab[t] = len(entity_vocab)
            if r not in relation_vocab:            # same for the relation
                relation_vocab[r] = len(relation_vocab)
            triples.append((entity_vocab[h], relation_vocab[r], entity_vocab[t]))

        kg_name = os.path.basename(root)           # short name for logging (folder name)

        # Encode entity and relation token strings to 384-dim ST embeddings.
        entity_strings   = [tok for tok, _ in sorted(entity_vocab.items(), key=lambda x: x[1])]
        relation_strings = [tok for tok, _ in sorted(relation_vocab.items(), key=lambda x: x[1])]
        entity_embs   = _encode_strings(entity_strings)    # (n_ent, ST_DIM) – CPU
        relation_embs = _encode_strings(relation_strings)  # (n_rel, ST_DIM) – CPU

        print(
            f"  {kg_name:<30s}  {len(triples):>6d} triples  "
            f"{len(entity_vocab):>5d} entities  {len(relation_vocab):>4d} relations"
        )
        pools.append((triples, entity_embs, relation_embs))

    return pools

class RichSubgraphPrior:
    """Generates binary triple-scoring episodes from real knowledge graphs.

    Each episode consists of:

    - A **support set** of ``context_size`` triples sampled uniformly from
      one randomly chosen KG (the in-context knowledge base), represented as
      FloatTensor ``(context_size, 3, 384)`` of SentenceTransformer embeddings.
    - A **query triple** ``(h, r, t)`` — a fully specified triple, represented
      as FloatTensor ``(3, 384)``.
    - A **label** — ``1.0`` if the query triple is a real KG triple,
      ``0.0`` if it is corrupted (tail replaced with a random entity).

    This binary scoring formulation removes the constraint that the target
    entity must appear in the support context.  The model learns to assign
    high logit to triples that are consistent with the observed context
    graph and low logit to corrupted ones.

    At evaluation time the model receives the full ``train.txt`` as context
    and scores every candidate completion of a test query ``(h, r, ?)``,
    ranking the true tail by score.

    Entity and relation identity is captured by frozen SentenceTransformer
    embeddings, so **no ID re-randomisation** is needed during training.
    """

    def __init__(
        self,
        max_hop: int = 3,
        kg_pools: Optional[list] = None,
    ):
        self.max_hop = max_hop
        self.kg_pools = kg_pools  # list of (triples, entity_embs, relation_embs) per KG

    def _generate_real_task(
        self,
        context_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled episode from a real KG.

        Steps:

        a. Select a random KG.
        b. Select a random positive triple ``(h, r, t)`` from the KG.
        c. Sample ``context_size`` triples uniformly as the support context.
        d. With probability 0.5, keep the query as the real triple (label=1).
           Otherwise corrupt the tail with a random entity from the KG (label=0).
        e. Look up pre-computed SentenceTransformer embeddings (no ID re-randomisation
           needed — semantic identity is captured by the embedding itself).

        Returns
        -------
        support      : FloatTensor (context_size, 3, ST_DIM)
        query_triple : FloatTensor (3, ST_DIM)  — [head_emb, rel_emb, tail_emb]
        label        : FloatTensor scalar  — 1.0 = real, 0.0 = corrupted
        """
        # a. Pick one KG; unpack triples + pre-computed ST embedding matrices.
        triples, entity_embs, relation_embs = random.choice(self.kg_pools)
        n     = len(triples)
        n_ent = entity_embs.shape[0]

        # b. Pick the positive query triple uniformly.
        tgt_idx = random.randrange(n)
        q_h, q_r, q_t = triples[tgt_idx]

        # c. Sample context_size support triples (independent draw, no constraint).
        if n <= context_size:
            ctx_idxs = list(range(n))
            while len(ctx_idxs) < context_size:
                ctx_idxs.append(random.randrange(n))
        else:
            ctx_idxs = random.sample(range(n), context_size)
        support_raw = [triples[i] for i in ctx_idxs]

        # d. Decide label: 50 % positive, 50 % corrupted negative.
        label = int(random.random() < 0.5)

        # e. Build float support tensor from pre-computed embeddings.
        h_list = [h for h, _r, _t in support_raw]
        r_list = [_r for _h, _r, _t in support_raw]
        t_list = [_t for _h, _r, _t in support_raw]
        sup_h = entity_embs[h_list]    # (S, ST_DIM)
        sup_r = relation_embs[r_list]  # (S, ST_DIM)
        sup_t = entity_embs[t_list]    # (S, ST_DIM)
        support_tensor = torch.stack([sup_h, sup_r, sup_t], dim=1)  # (S, 3, ST_DIM)

        # Build query triple embedding.
        q_h_emb = entity_embs[q_h]    # (ST_DIM,)
        q_r_emb = relation_embs[q_r]  # (ST_DIM,)
        if label == 1:
            q_t_emb = entity_embs[q_t]
        else:
            neg_idx = random.choice([i for i in range(n_ent) if i != q_t])
            q_t_emb = entity_embs[neg_idx]

        query_tensor = torch.stack([q_h_emb, q_r_emb, q_t_emb], dim=0)  # (3, ST_DIM)

        return support_tensor, query_tensor, torch.tensor(float(label))

    def generate_task(
        self,
        context_size: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one binary-labelled training episode.

        Delegates to :meth:`_generate_real_task`.  ``kg_pools`` must be set
        (pass ``kg_dir`` to :func:`train`).

        Returns
        -------
        support      : FloatTensor (context_size, 3, ST_DIM)  — context triples
        query_triple : FloatTensor (3, ST_DIM)                — candidate triple
        label        : FloatTensor scalar                     — 1.0 real, 0.0 corrupted
        """
        if not self.kg_pools:
            raise ValueError(
                "kg_pools is empty.  Pass --kg-dir (CLI) or kg_dir (API) to load real KGs."
            )
        return self._generate_real_task(context_size)


# ---------------------------------------------------------------------------
# 2.  MODEL
# ---------------------------------------------------------------------------

class TripleEncoder(nn.Module):
    """Maps (h, r, t) embeddings to a single token via a two-layer MLP.

    Concatenating then projecting preserves the distinct structural roles
    of head, relation, and tail that plain addition h+r+t destroys.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # h, r, t: (..., D)
        return self.net(torch.cat([h, r, t], dim=-1))


class TriplePFN(nn.Module):
    """
    Prior-Fitted Network that scores a fully-specified triple
    ``(head, relation, tail)`` against an in-context support set.

    Entity and relation tokens are embedded by the frozen all-MiniLM-L6-v2
    SentenceTransformer (384-dim, no gradients) and projected to the model
    working dimension *d* by two small learned linear layers:

    - ``entity_proj``   : ℝ^{384} → ℝ^d
    - ``relation_proj`` : ℝ^{384} → ℝ^d

    Each triple is then encoded by :class:`TripleEncoder` (a 2-layer MLP over
    the concatenation of the three projected embeddings) into a single token
    of shape ``(d,)``.  The S support tokens plus the query token form a
    sequence fed to a pre-norm Transformer encoder.  The last token (query
    position) is projected to a scalar logit by ``score_head``.

    Unlike a multi-class classifier, this model takes a **complete** triple
    as input and predicts whether it is a true KG triple given the context.
    At evaluation time, all candidate tail entities are enumerated and
    ranked by score.

    Forward interface::

        logit = model(support_triples, query_triple)  # (B,) or scalar
        prob  = torch.sigmoid(logit)

    Accepts both unbatched ``(S, 3, 384)`` / ``(3, 384)`` and batched
    ``(B, S, 3, 384)`` / ``(B, 3, 384)`` float-tensor inputs.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Project from SentenceTransformer output dim to model working dim.
        # These are the only learned "embedding" parameters; the 384-dim
        # semantic representations come from the frozen all-MiniLM-L6-v2 model.
        self.entity_proj   = nn.Linear(_ST_DIM, embed_dim)
        self.relation_proj = nn.Linear(_ST_DIM, embed_dim)

        self.triple_encoder = TripleEncoder(embed_dim)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm: more stable gradients for deeper stacks
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Read the representation of the query token (last position) and
        # project to a single logit: positive → real triple, negative → corrupted.
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        support_triples: torch.Tensor,    # (B, S, 3, ST_DIM) or (S, 3, ST_DIM)
        query_triple: torch.Tensor,       # (B, 3, ST_DIM)   or (3, ST_DIM)
    ) -> torch.Tensor:                    # (B,)     or scalar
        """Score a query triple given a support context.

        Parameters
        ----------
        support_triples : FloatTensor, shape (B, S, 3, ST_DIM) or (S, 3, ST_DIM)
            Context triples as SentenceTransformer embeddings stacked on dim-2:
            ``[:, :, 0, :]`` = head embs, ``[:, :, 1, :]`` = relation embs,
            ``[:, :, 2, :]`` = tail embs.
        query_triple : FloatTensor, shape (B, 3, ST_DIM) or (3, ST_DIM)
            The candidate triple to score.

        Returns
        -------
        Tensor, shape (B,) or scalar
            Unnormalised logit.  Apply ``torch.sigmoid`` for probability.
        """
        unbatched = support_triples.dim() == 3
        if unbatched:
            support_triples = support_triples.unsqueeze(0)   # (1, S, 3, ST_DIM)
            query_triple    = query_triple.unsqueeze(0)      # (1, 3, ST_DIM)

        B, S, _, _ = support_triples.shape
        D = self.embed_dim

        # Project 384-dim ST embeddings → model working dim.
        h = self.entity_proj(support_triples[:, :, 0, :])    # (B, S, D)
        r = self.relation_proj(support_triples[:, :, 1, :])  # (B, S, D)
        t = self.entity_proj(support_triples[:, :, 2, :])    # (B, S, D)
        support_tok = self.triple_encoder(
            h.reshape(B * S, D), r.reshape(B * S, D), t.reshape(B * S, D)
        ).reshape(B, S, D)                                   # (B, S, D)

        q_h = self.entity_proj(query_triple[:, 0, :])        # (B, D)
        q_r = self.relation_proj(query_triple[:, 1, :])      # (B, D)
        q_t = self.entity_proj(query_triple[:, 2, :])        # (B, D)
        query_tok = self.triple_encoder(q_h, q_r, q_t).unsqueeze(1)  # (B, 1, D)

        seq = torch.cat([support_tok, query_tok], dim=1)     # (B, S+1, D)
        seq = self.embed_drop(self.input_norm(seq))

        out = self.transformer(seq)                          # (B, S+1, D)

        logit = self.score_head(out[:, -1, :]).squeeze(-1)   # (B,)
        return logit.squeeze(0) if unbatched else logit


# ---------------------------------------------------------------------------
# 3.  TRAINING
# ---------------------------------------------------------------------------

def _collate(
    tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of (support, query_triple, label) tuples into batched tensors."""
    return (
        torch.stack([t[0] for t in tasks]),   # (B, S, 3, ST_DIM)
        torch.stack([t[1] for t in tasks]),   # (B, 3, ST_DIM)
        torch.stack([t[2] for t in tasks]),   # (B,)  float labels
    )


def train(
    num_epochs: int = 3000,
    batch_size: int = 8,
    context_size: int = 64,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    lr: float = 1e-4,
    eval_every: int = 200,
    kg_dir: Optional[str] = None,
) -> TriplePFN:
    """Meta-train a :class:`TriplePFN` on binary triple-scoring episodes.

    Tasks are sampled on-the-fly from :class:`RichSubgraphPrior`.  Each
    episode consists of:

    - A **support set** of ``context_size`` triples drawn uniformly from a
      randomly chosen KG (the in-context knowledge base).
    - A **query triple** ``(h, r, t)`` — either a real KG triple (label=1)
      or a corrupted one with the tail replaced by a random entity (label=0).

    The model is trained with binary cross-entropy to assign high scores to
    real triples and low scores to corrupted ones.  No constraint is imposed
    on whether the query tail appears in the support context.

    Entity and relation identity is captured by frozen SentenceTransformer
    embeddings, so no ID re-randomisation is needed.  The model learns
    structural patterns directly from the semantic representations.

    After pre-training, call :func:`evaluate` to benchmark on any KG by
    providing ``train.txt`` as the full in-context support.

    Parameters
    ----------
    num_epochs : int
        Total number of training iterations (one batch per iteration).
    batch_size : int
        Number of tasks per gradient step.
    context_size : int
        Number of support triples per task.
    embed_dim : int
        Hidden dimensionality of the projection layers and the Transformer.
        Entity/relation strings are encoded to 384-dim by the frozen
        all-MiniLM-L6-v2 model and projected down to this dimension.
    num_heads : int
        Number of attention heads in the Transformer encoder.
    num_layers : int
        Number of Transformer encoder layers.
    lr : float
        Peak learning rate for AdamW (decayed via cosine annealing).
    eval_every : int
        Print accuracy on a held-out batch of 512 tasks every this many epochs.
    kg_dir : str or None
        Root directory to scan for ``train.txt`` files (e.g. ``KGs/``).
        :func:`_load_real_triples` loads all KGs (capped at ``max_per_kg``
        triples, default 500) and encodes their tokens with the frozen
        all-MiniLM-L6-v2 SentenceTransformer.

    Returns
    -------
    TriplePFN
        The trained model (on CPU or CUDA depending on availability).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kg_pools = None
    if kg_dir is not None:
        print(f"Loading real KG triples from: {kg_dir}")
        print(f"  {'KG':<30s}  {'triples':>6s}  {'entities':>8s}  {'rels':>4s}")
        print(f"  {'-'*30}  {'-'*6}  {'-'*8}  {'-'*4}")
        kg_pools = _load_real_triples(kg_dir)
        total_triples = sum(len(p[0]) for p in kg_pools)
        print(f"  {'─'*57}")
        print(f"  {len(kg_pools)} KGs loaded   {total_triples:,} total triples\n")
    prior = RichSubgraphPrior(
        kg_pools=kg_pools,
    )
    model = TriplePFN(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()   # binary cross-entropy with logits

    print(f"Device : {device}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
    print("Starting GraphPFN binary triple-scoring pre-training...\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        supports, query_triples, labels = _collate(
            [prior.generate_task(context_size=context_size) for _ in range(batch_size)]
        )
        supports      = supports.to(device)       # (B, S, 3, ST_DIM)
        query_triples = query_triples.to(device)  # (B, 3, ST_DIM)
        labels        = labels.to(device)         # (B,)  float {0., 1.}

        optimizer.zero_grad()
        logits = model(supports, query_triples)   # (B,)  scalar logits
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                e_sup, e_qry, e_lbl = _collate(
                    [prior.generate_task(context_size=context_size) for _ in range(512)]
                )
                e_sup = e_sup.to(device)
                e_qry = e_qry.to(device)
                e_lbl = e_lbl.to(device)
                e_logits = model(e_sup, e_qry)                    # (512,)
                e_preds  = (e_logits > 0).float()                # threshold at 0
                acc = (e_preds == e_lbl).float().mean().item()
            print(f"Epoch {epoch:5d} | Loss {loss.item():.4f} | Acc {acc:.3f}")

    print("\nTraining complete.")
    return model


def evaluate(
    model: "TriplePFN",
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
# 4.  INFERENCE
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
    >>> from graph_pfn import train, infer
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
# 5.  CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph_pfn.py",
        description="GraphPFN — in-context link prediction on knowledge graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "  # Train and save\n"
            "  python graph_pfn.py train --epochs 3000 --save model.pt\n"
            "\n"
            "  # Infer using Countries-S1: which countries neighbour slovakia?\n"
            "  # (explicit support triples, string tokens)\n"
            "  python graph_pfn.py infer --model model.pt \\\n"
            "      --query slovakia neighbor \\\n"
            "      --support slovakia,neighbor,ukraine slovakia,neighbor,hungary \\\n"
            "               slovakia,neighbor,austria slovakia,neighbor,czechia \\\n"
            "      --k 5\n"
            "\n"
            "  # Infer using Countries-S1: auto-sample context from training file\n"
            "  # (which region does morocco belong to? answer: africa)\n"
            "  python graph_pfn.py infer --model model.pt \\\n"
            "      --query morocco locatedin \\\n"
            "      --data KGs/Countries-S1/train.txt --context-size 32 --k 5\n"
            "\n"
            "  # Infer using Countries-S1: use entire test file as support context\n"
            "  python graph_pfn.py infer --model model.pt \\\n"
            "      --query slovakia neighbor \\\n"
            "      --test KGs/Countries-S1/test.txt --k 5\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")
    parser.set_defaults(command="train")

    # ── train ──────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Meta-train a TriplePFN model.")
    tr.add_argument("--epochs",        type=int,   default=3000,  help="Training epochs (default: 3000)")
    tr.add_argument("--batch-size",    type=int,   default=8,     help="Tasks per batch (default: 8; reduce if OOM with large contexts)")
    tr.add_argument("--context-size",  type=int,   default=64,    help="Support triples per task (default: 64; increase if VRAM allows)")
    tr.add_argument("--embed-dim",     type=int,   default=256,   help="Embedding dimension (default: 256)")
    tr.add_argument("--num-heads",     type=int,   default=8,     help="Transformer attention heads (default: 8)")
    tr.add_argument("--num-layers",    type=int,   default=6,     help="Transformer layers (default: 6)")
    tr.add_argument("--lr",            type=float, default=1e-4,  help="Learning rate (default: 1e-4)")
    tr.add_argument("--eval-every",    type=int,   default=200,   help="Evaluate every N epochs (default: 200)")
    tr.add_argument("--save",          type=str,   default=None,  help="Path to save the trained model state-dict (.pt)")
    tr.add_argument(
        "--kg-dir", type=str, default="KGs", metavar="DIR",
        help=(
            "Directory to scan recursively for train.txt files (e.g. KGs/). "
            "When provided, real KG episodes are mixed into training as a third "
            "task family (1/3 real, 1/3 lookup, 1/3 rich-subgraph). "
            "Each KG is capped at 20 000 triples to avoid large-KG dominance."
        ),
    )

    # ── infer ──────────────────────────────────────────────────────────────
    inf = sub.add_parser(
        "infer",
        help="Predict the top-k tail entities for a (head, relation, ?) query.",
    )
    inf.add_argument(
        "--model", type=str, required=True,
        help="Path to a saved state-dict (.pt) produced by the train command.",
    )
    inf.add_argument(
        "--query", type=str, nargs=2, required=True, metavar=("HEAD", "REL"),
        help=(
            "Query as two tokens: head-entity and relation. "
            "Can be string names (e.g. 'Paris locatedIn') when --data or --support "
            "contains string labels, or plain integers for raw indexed triples."
        ),
    )
    inf.add_argument(
        "--support", type=str, nargs="+", default=None, metavar="H,R,T",
        help=(
            "Explicit support triples, each as 'h,r,t' (comma-separated). "
            "Tokens can be string names or integers. "
            "Example: --support Paris,locatedIn,France London,locatedIn,UK"
        ),
    )
    inf.add_argument(
        "--data", type=str, default=None, metavar="FILE",
        help=(
            "Path to a triple file (one triple per line, whitespace-separated: h r t). "
            "Columns can be string names or integers — detected automatically. "
            "String columns are mapped to integer indices; results are shown with their "
            "original string labels. Used to sample the support when --support is not given."
        ),
    )
    inf.add_argument(
        "--test", type=str, default=None, metavar="FILE",
        help=(
            "Path to a triple file whose triples are used directly as the full support "
            "context (all lines loaded — no sampling).  This is the file-based equivalent "
            "of --support.  Useful when you have a pre-built context set on disk, e.g. "
            "KGs/Countries-S1/test.txt.  Tokens can be string names or integers."
        ),
    )
    inf.add_argument(
        "--context-size", type=int, default=32,
        help="Number of triples to sample from --data (default: 32).",
    )
    inf.add_argument(
        "--k", type=int, default=5,
        help="Number of top predictions to display (default: 5).",
    )
    # Model architecture flags must match those used during training
    inf.add_argument("--embed-dim",     type=int, default=256)
    inf.add_argument("--num-heads",     type=int, default=8)
    inf.add_argument("--num-layers",    type=int, default=6)

    # ── eval ───────────────────────────────────────────────────────────────
    ev = sub.add_parser(
        "eval",
        help="Evaluate a trained model on a KG test set (MRR, MR, Hits@1/3/10).",
        description=(
            "Load the full train.txt as the support context and rank each "
            "test.txt triple.  Architecture flags must match the trained model."
        ),
    )
    ev.add_argument(
        "--model", type=str, required=True,
        help="Path to a saved state-dict (.pt) produced by the train command.",
    )
    ev.add_argument(
        "--train", type=str, required=True, metavar="FILE",
        help="Path to train.txt (used as the full support context for every test query).",
    )
    ev.add_argument(
        "--test", type=str, required=True, metavar="FILE",
        help="Path to test.txt (queries to rank).",
    )
    ev.add_argument("--batch-size",    type=int, default=64,    help="Queries per forward pass (default: 64; reduce if OOM).")
    ev.add_argument("--embed-dim",     type=int, default=256,   help="Must match the trained model (default: 256).")
    ev.add_argument("--num-heads",     type=int, default=8,     help="Must match the trained model (default: 8).")
    ev.add_argument("--num-layers",    type=int, default=6,     help="Must match the trained model (default: 6).")

    return parser


if __name__ == "__main__":
    # Default to 'train' when no subcommand is given (convenient for debugging).
    if len(sys.argv) == 1:
        sys.argv.append("train")

    args = _build_parser().parse_args()

    # ── TRAIN ──────────────────────────────────────────────────────────────
    if args.command == "train":
        model = train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            context_size=args.context_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            lr=args.lr,
            eval_every=args.eval_every,
            kg_dir=args.kg_dir,
        )
        if args.save:
            torch.save(model.state_dict(), args.save)
            print(f"Model saved to {args.save}")

    # ── INFER ──────────────────────────────────────────────────────────────
    elif args.command == "infer":
        model = TriplePFN(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))

        q_head_tok, q_rel_tok = args.query

        # ── Build support as a list of string triples ───────────────────────
        support_rows: List[Tuple[str, str, str]] = []

        if args.support:
            for triple_str in args.support:
                parts = triple_str.split(",")
                if len(parts) != 3:
                    raise ValueError(f"Expected h,r,t but got: '{triple_str}'")
                support_rows.append((parts[0], parts[1], parts[2]))

        elif args.test:
            with open(args.test) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        raise ValueError(f"Expected 3 columns per line, got: '{line}'")
                    support_rows.append((parts[0], parts[1], parts[2]))

        elif args.data:
            import random as _rnd
            all_rows: List[Tuple[str, str, str]] = []
            with open(args.data) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        raise ValueError(f"Expected 3 columns per line, got: '{line}'")
                    all_rows.append((parts[0], parts[1], parts[2]))
            # Prioritise triples sharing query head or query relation.
            relevant = [t for t in all_rows if t[0] == q_head_tok or t[1] == q_rel_tok]
            others   = [t for t in all_rows if t[0] != q_head_tok and t[1] != q_rel_tok]
            _rnd.shuffle(relevant)
            _rnd.shuffle(others)
            support_rows = (relevant + others)[:args.context_size]

        if not support_rows:
            raise ValueError(
                "No support triples found.  Provide --support, --test, or --data."
            )

        results = infer(model, q_head_tok, q_rel_tok, support_rows, k=args.k)

        print(f"\nTop-{args.k} predictions for query ({q_head_tok}, {q_rel_tok}, ?):")
        print(f"  {'Rank':<6} {'Entity':>20}  {'Score':>10}")
        print(f"  {'-'*6} {'-'*20}  {'-'*10}")
        for rank, (entity_str, score) in enumerate(results, 1):
            print(f"  #{rank:<5} {entity_str:>20}  {score:>10.4f}")

    # ── EVAL ───────────────────────────────────────────────────────────────
    elif args.command == "eval":
        model = TriplePFN(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))

        print(f"Evaluating on  : {args.test}")
        print(f"Support (train): {args.train}\n")
        results = evaluate(model, args.train, args.test, batch_size=args.batch_size)

        print("\nLink Prediction Results (raw / unfiltered):")
        print(f"  {'Metric':<10s}  {'Value':>8s}")
        print(f"  {'-'*10}  {'-'*8}")
        for metric, value in results.items():
            fmt = f"{value:>8.4f}" if metric != "MR" else f"{value:>8.2f}"
            print(f"  {metric:<10s}  {fmt}")
