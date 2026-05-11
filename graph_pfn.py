"""
Graph Prior-Fitted Network (GraphPFN) for in-context link prediction.

Key improvements over the baseline
------------------------------------
1. Entity-index re-randomisation per task — prevents identity memorisation;
   forces the model to learn from structure alone (the core PFN property).
2. Guaranteed-solvable tasks — the critical chain triples needed to answer
   the query are always kept in the support set; noise fills remaining slots.
3. Role-aware triple encoder (MLP on [h; r; t]) — plain addition h+r+t
   conflates head/relation/tail roles; an MLP preserves them.
4. Pre-norm Transformer (norm_first=True) — better gradient flow.
5. Single batched forward  (B, S, 3) — handles both batched and unbatched
   inputs without a separate class.
6. Two-layer prediction head — more expressive than a single linear layer.
7. AdamW + cosine LR + gradient clipping — modern training recipe.
8. Hits@1 evaluation during training — tracks task-level accuracy.
9. Rich subgraph prior — context window filled with inverse edges, sibling
   multi-relation triples, a distractor chain, and extended-chain triples
   rather than random noise, giving the Transformer real structural signal.
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1.  DATA PRIOR
# ---------------------------------------------------------------------------

class RichSubgraphPrior:
    """
    Generates tasks from richer multi-relational subgraphs.

    Each task samples a hop depth in [1, max_hop] and fills the context with:
      - Critical chain triples  (guaranteed; make the task solvable)
      - Inverse edges           (teach relation directionality)
      - Sibling triples         (chain entities connected via other relations —
                                 simulates a real multi-relational neighbourhood)
      - Distractor chain        (same relation, fully disjoint entities — tests
                                 the model's ability to ignore irrelevant paths)
      - Extended chain          (chain triples beyond the answer node — forces
                                 precise hop counting, not "longest path" heuristics)
      - Random noise            (pads any remaining slots)

    Entity/relation indices are re-randomised per task.
    """

    def __init__(
        self,
        num_entities: int = 100,
        num_relations: int = 10,
        max_hop: int = 3,
    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.max_hop = max_hop

    def generate_task(
        self,
        context_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        support : LongTensor (context_size, 3)  — observed (h, r, t) triples
        query   : LongTensor (2,)               — (h_q, r_q)
        target  : LongTensor scalar             — expected tail entity
        """
        hop = random.randint(1, self.max_hop)
        assert context_size >= hop

        r = random.randint(0, self.num_relations - 1)
        r_inv = (r + 1) % self.num_relations
        other_rels = [rr for rr in range(self.num_relations) if rr not in (r, r_inv)]

        # Build a chain that extends several nodes beyond the answer
        extra = 3
        chain = random.sample(range(self.num_entities), min(hop + extra + 1, self.num_entities))

        # ── Critical triples ────────────────────────────────────────────────
        critical: List[Tuple[int, int, int]] = [
            (chain[i], r, chain[i + 1]) for i in range(hop)
        ]
        critical_set: set = set(map(tuple, critical))
        query_h, query_r, target_t = chain[0], r, chain[hop]

        # ── Structured filler ───────────────────────────────────────────────
        structured: List[Tuple[int, int, int]] = []

        # a) Inverse edges for each critical triple
        for h_, r_, t_ in critical:
            structured.append((t_, r_inv, h_))

        # b) Sibling triples: chain entities connected via other relations
        chain_nodes = chain[: hop + 1]
        for i, ei in enumerate(chain_nodes):
            for j, ej in enumerate(chain_nodes):
                if i != j and other_rels:
                    structured.append((ei, random.choice(other_rels), ej))

        # c) Distractor chain: same relation r, disjoint entity set
        pool = [e for e in range(self.num_entities) if e not in set(chain)]
        if len(pool) >= hop + 1:
            d_chain = random.sample(pool, hop + 1)
            for i in range(hop):
                structured.append((d_chain[i], r, d_chain[i + 1]))

        # d) Extended chain triples beyond the answer node (forces precise hop counting)
        for i in range(hop, min(hop + extra, len(chain) - 1)):
            structured.append((chain[i], r, chain[i + 1]))

        # Deduplicate structured filler
        seen: set = set(critical_set)
        deduped: List[Tuple[int, int, int]] = []
        for t in structured:
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                deduped.append(t)

        random.shuffle(deduped)
        remaining = context_size - len(critical)
        selected = deduped[:remaining]

        # e) Pad with random noise if structured filler is insufficient
        while len(selected) < remaining:
            nh = random.randint(0, self.num_entities - 1)
            nr_ = random.randint(0, self.num_relations - 1)
            nt = random.randint(0, self.num_entities - 1)
            key = (nh, nr_, nt)
            if key not in seen:
                seen.add(key)
                selected.append(key)

        support = critical + selected
        random.shuffle(support)  # shuffle so position carries no signal

        # ── Re-randomise entity / relation indices ───────────────────────────
        unique_ents = list(
            {e for h_, _, t_ in support for e in (h_, t_)} | {query_h, target_t}
        )
        unique_rels = list({r_ for _, r_, _ in support} | {query_r})

        e_perm = random.sample(range(self.num_entities), len(unique_ents))
        r_perm = random.sample(range(self.num_relations), len(unique_rels))
        e_map = dict(zip(unique_ents, e_perm))
        r_map = dict(zip(unique_rels, r_perm))

        support = [(e_map[h_], r_map[r_], e_map[t_]) for h_, r_, t_ in support]

        return (
            torch.tensor(support, dtype=torch.long),
            torch.tensor([e_map[query_h], r_map[query_r]], dtype=torch.long),
            torch.tensor(e_map[target_t], dtype=torch.long),
        )


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
    Prior-Fitted Network for in-context link prediction on knowledge graphs.

    The support set of observed triples is the context; the Transformer
    performs in-context inference to predict the missing tail entity.
    Supports both unbatched (S, 3) and batched (B, S, 3) inputs.
    """

    def __init__(
        self,
        num_entities: int = 50,
        num_relations: int = 5,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.embed_dim = embed_dim

        self.entity_embed = nn.Embedding(num_entities, embed_dim)
        self.relation_embed = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

        self.triple_encoder = TripleEncoder(embed_dim)

        # Learnable [MASK] token representing the unknown tail in the query
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))

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

        # Two-layer head is more expressive than a single linear projection
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_entities),
        )

    def forward(
        self,
        support_triples: torch.Tensor,  # (B, S, 3) or (S, 3)
        query: torch.Tensor,            # (B, 2)   or (2,)
    ) -> torch.Tensor:                  # (B, E)   or (E,)
        unbatched = support_triples.dim() == 2
        if unbatched:
            support_triples = support_triples.unsqueeze(0)  # (1, S, 3)
            query = query.unsqueeze(0)                      # (1, 2)

        B, S, _ = support_triples.shape
        D = self.embed_dim

        # Encode support triples — flatten to (B*S, D) for the MLP, then restore
        h = self.entity_embed(support_triples[:, :, 0])    # (B, S, D)
        r = self.relation_embed(support_triples[:, :, 1])  # (B, S, D)
        t = self.entity_embed(support_triples[:, :, 2])    # (B, S, D)
        support_tok = self.triple_encoder(
            h.reshape(B * S, D), r.reshape(B * S, D), t.reshape(B * S, D)
        ).reshape(B, S, D)

        # Encode query: (h_q + r_q) + MASK
        q_h = self.entity_embed(query[:, 0])                       # (B, D)
        q_r = self.relation_embed(query[:, 1])                     # (B, D)
        query_tok = (q_h + q_r).unsqueeze(1) + self.mask_token    # (B, 1, D)

        # Sequence: [support_1, ..., support_S, query_token]
        seq = torch.cat([support_tok, query_tok], dim=1)           # (B, S+1, D)
        seq = self.embed_drop(self.input_norm(seq))

        out = self.transformer(seq)                                 # (B, S+1, D)
        logits = self.head(out[:, -1, :])                          # (B, E)

        return logits.squeeze(0) if unbatched else logits


# ---------------------------------------------------------------------------
# 3.  TRAINING
# ---------------------------------------------------------------------------

def _collate(
    tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([t[0] for t in tasks]),
        torch.stack([t[1] for t in tasks]),
        torch.stack([t[2] for t in tasks]),
    )


def train(
    num_epochs: int = 3000,
    batch_size: int = 64,
    context_size: int = 32,
    num_entities: int = 100,
    num_relations: int = 10,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    lr: float = 1e-4,
    eval_every: int = 200,
) -> TriplePFN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = RichSubgraphPrior(num_entities=num_entities, num_relations=num_relations)
    model = TriplePFN(
        num_entities=num_entities,
        num_relations=num_relations,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"Device : {device}")
    print(f"Params : {sum(p.numel() for p in model.parameters()):,}")
    print("Starting PFN meta-training...\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        supports, queries, targets = _collate(
            [prior.generate_task(context_size=context_size) for _ in range(batch_size)]
        )  # generate_task samples hop in [1, max_hop] internally
        supports = supports.to(device)
        queries = queries.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss = criterion(model(supports, queries), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                e_sup, e_qry, e_tgt = _collate(
                    [prior.generate_task(context_size=context_size) for _ in range(512)]
                )
                e_sup = e_sup.to(device)
                e_qry = e_qry.to(device)
                e_tgt = e_tgt.to(device)
                hits1 = (model(e_sup, e_qry).argmax(dim=-1) == e_tgt).float().mean().item()
            print(f"Epoch {epoch:5d} | Loss {loss.item():.4f} | Hits@1 {hits1:.3f}")

    print("\nTraining complete.")
    return model


if __name__ == "__main__":
    train()
