"""InductiveKGModel: per-triple encoder + permutation-invariant SAB stack (spec §5)."""
from __future__ import annotations

from pathlib import Path

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
        z_mode: str = "learned",
        z_start: int = 0,
        z_pool: int = 0,
    ):
        super().__init__()
        self.x_token_id = x_token_id # usually 1
        self.d_model = d_model # e.g. 128

        # Anonymous [Z_*] slots: `learned` = trained embedding rows; `runtime` =
        # fresh random vector per slot, redrawn every forward (RNI, Abboud 2021).
        # `z_start`/`z_pool` mark the [Z_0..Z_{z_pool-1}] block in the vocab.
        if z_mode not in ("learned", "runtime"):
            raise ValueError(f"z_mode must be learned or runtime, got {z_mode!r}")
        self.z_mode = z_mode
        self.z_start = z_start
        self.z_pool = z_pool
        if z_mode == "runtime" and z_pool <= 0:
            raise ValueError("z_mode='runtime' requires z_pool > 0")

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0) # shape [vocab_size, d_model] -> first row all 0s
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

    def _make_rand_bank(
        self, batch: int | None = None, device: torch.device | None = None,
    ) -> torch.Tensor:
        """Draw a unit-norm random bank for `runtime` mode.

        `batch=None` → a shared [z_pool, d] bank (one anonymization for a whole
        query; used at scoring time so candidates share the subgraph's draw).
        `batch=B`    → an independent [B, z_pool, d] bank per sample (training:
        a different anonymization per example, which is the RNI regularizer).
        """
        shape = (self.z_pool, self.d_model) if batch is None else (batch, self.z_pool, self.d_model)
        bank = torch.randn(shape, device=device)
        return bank / bank.norm(dim=-1, keepdim=True)

    def _entity_embed(
        self, ids: torch.Tensor, rand_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed token ids, overriding [Z_*] positions per the active z_mode.

        `learned` returns the table embedding unchanged. `runtime` substitutes
        `rand_bank` (shape [z_pool, d] shared, or [B, z_pool, d] per-sample where
        B == ids.shape[0]). Non-Z tokens always keep their table embedding.
        """
        emb = self.embed(ids)
        if self.z_mode == "learned":
            return emb
        is_z = (ids >= self.z_start) & (ids < self.z_start + self.z_pool)
        z_local = (ids - self.z_start).clamp(0, self.z_pool - 1)
        if rand_bank.dim() == 2:                          # shared [z_pool, d]
            z_emb = rand_bank[z_local]
        else:                                             # per-sample [B, z_pool, d]
            B, d = ids.shape[0], self.d_model
            flat = z_local.reshape(B, -1)
            g = torch.gather(rand_bank, 1, flat.unsqueeze(-1).expand(-1, -1, d))
            z_emb = g.reshape(*ids.shape, d)
        return torch.where(is_z.unsqueeze(-1), z_emb, emb)

    def _encode_subgraph(
        self,
        triples: torch.Tensor,         # [B, N, 3] long
        mask: torch.Tensor,            # [B, N]    bool, True = valid
        hop_distances: torch.Tensor | None,
        rand_bank: torch.Tensor | None,
    ) -> torch.Tensor:                 # [B, d]    pooled [X] representation
        """Encode one anonymized subgraph (centered on its [X] token) → pooled vec.

        Shared by the anchor tower and the optional candidate tower, so the two
        sides are siamese: identical weights, zero extra parameters.
        """
        B, N, _ = triples.shape
        tok = self._entity_embed(triples, rand_bank) + self.intra_pos  # [B, N, 3, d]
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
        return h[:, 0]                                       # [B, d]

    def forward(
        self,
        triples: torch.Tensor,         # [B, N, 3] long
        mask: torch.Tensor,            # [B, N]    bool, True = valid
        target_relation: torch.Tensor, # [B]       long
        target_tail: torch.Tensor,     # [B]       long
        hop_distances: torch.Tensor | None = None,  # [B, N, 3] long; optional per-entity hop-distance tokens
        cand_triples: torch.Tensor | None = None,        # [B, M, 3] long; candidate subgraph
        cand_mask: torch.Tensor | None = None,           # [B, M]    bool
        cand_hop_distances: torch.Tensor | None = None,  # [B, M, 3] long
    ) -> torch.Tensor:                 # [B]       float (logits)
        """Score (anchor-subgraph, relation, candidate).

        Single tower (default): the candidate is a *token* — its embedding is
        meaningful only when it appears in the anchor's subgraph, otherwise it's
        an ungrounded [Z]. Pass `cand_triples`/`cand_mask` to instead represent
        the candidate by the pooled vector of *its own* k-hop subgraph (the
        candidate tower), which grounds candidates that lie outside the anchor's
        neighborhood. The two towers share weights and never attend to each
        other, so each entity's subgraph can be encoded once and cached at eval.
        """
        B = triples.shape[0]

        # One independent random anonymization per sample for runtime mode.
        rand_bank = self._make_rand_bank(B, triples.device) if self.z_mode == "runtime" else None

        pooled = self._encode_subgraph(triples, mask, hop_distances, rand_bank)  # [B, d]

        tr = self.embed(target_relation)                    # [B, d]
        if cand_triples is not None:
            # Candidate tower: same encoder run on the candidate's subgraph.
            tt = self._encode_subgraph(cand_triples, cand_mask, cand_hop_distances, rand_bank)
        else:
            tt = self._entity_embed(target_tail, rand_bank)  # [B, d]
        # TODO: we should consider a more sophistacted interaction for the feature vector  
        target_emb = tr + tt
        feats = torch.cat([pooled, target_emb, pooled * target_emb], dim=-1) # [B, 3*d] 
        return self.classifier(feats).squeeze(-1)


# --- bundle helpers (shared by train / eval / predict) --------------------

def resolve_device(cfg: dict) -> torch.device:
    """Honour cfg['device'] but fall back to CPU when CUDA is unavailable."""
    return torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict, vocab: dict[str, int], device: torch.device) -> "InductiveKGModel":
    """Construct an InductiveKGModel from a config dict and vocab.

    `cfg['z_mode']` (default 'learned') selects how [Z_*] slots are embedded;
    `runtime` locates the [Z_0..] block via the vocab, sized from `z_pool_size`.
    """
    z_mode = cfg.get("z_mode", "learned")
    z_pool = int(cfg["z_pool_size"]) if z_mode != "learned" else 0
    z_start = vocab.get("[Z_0]", 0)
    return InductiveKGModel(
        vocab_size=len(vocab),
        x_token_id=vocab["[X]"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_triple_layers=cfg["n_triple_layers"],
        n_sab=cfg["n_sab"],
        dropout=cfg["dropout"],
        z_mode=z_mode,
        z_start=z_start,
        z_pool=z_pool,
    ).to(device)


def load_state_dict_compat(model: "InductiveKGModel", state_dict: dict) -> None:
    """load_state_dict that tolerates a legacy unused `z_bank` buffer.

    Early `compare_anonymization` checkpoints always saved a `z_bank` (even in
    learned mode, where the unified model registers none). Drop it when the
    target model has no such buffer so those checkpoints still load strictly.
    """
    if "z_bank" in state_dict and not hasattr(model, "z_bank"):
        state_dict = {k: v for k, v in state_dict.items() if k != "z_bank"}
    model.load_state_dict(state_dict)


def load_bundle(
    model_path: str | Path,
) -> tuple["InductiveKGModel", dict, set[str], dict, torch.device]:
    """Load a self-contained {model, vocab, fixed_values, cfg} bundle (.pt).

    Returns (model in eval mode, vocab, fixed_values, cfg, device).
    """
    bundle = torch.load(model_path, map_location="cpu")
    cfg = dict(bundle["cfg"])
    # compare_anonymization checkpoints carry the scheme in a top-level "mode"
    # key rather than cfg["z_mode"]; honour it so eval reconstructs the right
    # model (learned vs runtime) without any extra wiring.
    if "mode" in bundle and "z_mode" not in cfg:
        cfg["z_mode"] = bundle["mode"]
    device = resolve_device(cfg)
    vocab = bundle["vocab"]
    model = build_model(cfg, vocab, device)
    load_state_dict_compat(model, bundle["model"])
    model.eval()
    return model, vocab, set(bundle["fixed_values"]), cfg, device
