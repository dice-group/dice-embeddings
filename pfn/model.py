"""TriplePFN model definition for GraphPFN.

Exposes
-------
- ``TripleEncoder`` — 2-layer MLP that maps (h, r, t) projected embeddings to a
  single token.
- ``TriplePFN`` — full Prior-Fitted Network: projection layers, triple encoder,
  pre-norm Transformer encoder, and binary scoring head.
"""

from typing import Optional

import torch
import torch.nn as nn

from pfn.dataset import _ST_DIM


# ---------------------------------------------------------------------------
# MODEL
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
