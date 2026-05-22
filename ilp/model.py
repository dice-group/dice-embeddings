"""InductiveKGModel: per-triple encoder + permutation-invariant SAB stack (spec §5)."""
from __future__ import annotations

import torch
import torch.nn as nn


class InductiveKGModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        x_token_id: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_triple_layers: int = 2,
        n_sab: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.x_token_id = x_token_id
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Intra-triple positional encoding (subject / relation / object).
        # Inter-triple position is intentionally absent — that's what makes
        # the SAB stack permutation-invariant.
        self.intra_pos = nn.Parameter(torch.randn(3, d_model) * 0.02)

        triple_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.triple_encoder = nn.TransformerEncoder(triple_layer, num_layers=n_triple_layers)

        sab_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.sab_stack = nn.TransformerEncoder(sab_layer, num_layers=n_sab)

        self.classifier = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        triples: torch.Tensor,         # [B, N, 3] long
        mask: torch.Tensor,            # [B, N]    bool, True = valid
        target_relation: torch.Tensor, # [B]       long
        target_tail: torch.Tensor,     # [B]       long
        hop_distances: torch.Tensor | None = None,  # [B, N, 3] long; optional per-entity hop-distance tokens
    ) -> torch.Tensor:                 # [B]       float (logits)
        B, N, _ = triples.shape

        tok = self.embed(triples) + self.intra_pos          # [B, N, 3, d]
        if hop_distances is not None:
            # Hop-distance tokens share the main embedding table — adds zero
            # new parameters while letting the model use per-entity BFS
            # distance from the anchor.
            tok = tok + self.embed(hop_distances)
        tok = self.triple_encoder(tok.view(B * N, 3, -1))   # [B*N, 3, d]
        triple_vec = tok.mean(dim=1).view(B, N, -1)         # [B, N, d]

        x_emb = self.embed(
            torch.full((B, 1), self.x_token_id, device=triples.device, dtype=torch.long)
        )  # [B, 1, d]
        h = torch.cat([x_emb, triple_vec], dim=1)           # [B, 1+N, d]
        mask_ext = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
        )
        # src_key_padding_mask: True positions are *masked out*.
        h = self.sab_stack(h, src_key_padding_mask=~mask_ext)
        pooled = h[:, 0]                                    # [B, d]

        tr = self.embed(target_relation)                    # [B, d]
        tt = self.embed(target_tail)                        # [B, d]
        target_emb = tr + tt
        feats = torch.cat([pooled, target_emb, pooled * target_emb], dim=-1)
        return self.classifier(feats).squeeze(-1)
