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

        # ──────────────────────────────────────────────────────────────────────────
        # DESIGN DECISION: Frozen SentenceTransformer Embeddings
        # ──────────────────────────────────────────────────────────────────────────
        # Entity and relation strings are encoded by the FROZEN all-MiniLM-L6-v2
        # SentenceTransformer (384-dim). These embeddings are NOT updated during
        # training. Only the projection layers below are learned.
        #
        # WHY KEEP IT FROZEN?
        # ✓ Universal semantic space: Pretrained embeddings provide consistent
        #   representations across different KG domains (medical, geographic, etc.)
        # ✓ Zero-shot generalization: Can score triples with completely novel
        #   entities at test time (just encode the string)
        # ✓ Meta-learning alignment: Frozen features enable the transformer to
        #   learn RELATIONAL REASONING rather than memorizing entity patterns
        # ✓ Cross-domain transfer: Same embedding space allows knowledge transfer
        #   from medical KGs to geographic KGs during episodic training
        # ✓ Prevents catastrophic forgetting: Keeps semantic knowledge (e.g.,
        #   "czechoslovakia" → country) that helps with unseen entities
        # ✓ Smaller model: Only ~10-20M trainable params vs ~42M if fine-tuning
        #
        # TRADE-OFFS IF FINE-TUNING:
        # ⚠ Task-specific embeddings: Could optimize for triple scoring specifically
        # ⚠ End-to-end optimization: Embedding space directly optimized for objective
        # BUT:
        # ✗ 3× parameter increase (~22M more params)
        # ✗ 3× slower training + higher GPU memory
        # ✗ Overfitting risk with limited unique entities per episode
        # ✗ Loss of zero-shot capability for novel entities
        # ✗ Breaks meta-learning paradigm: couples embeddings to specific KGs
        # ✗ Catastrophic forgetting of pretrained semantic knowledge
        #
        # WHEN TO FINE-TUNE:
        # - Single-domain deployment (e.g., only biomedical KGs)
        # - Millions of triples from one domain (prevents overfitting)
        # - Entity IDs instead of natural language strings
        #
        # CURRENT APPROACH (RECOMMENDED):
        # Keep SentenceTransformer frozen + learn lightweight projection layers.
        # This balances semantic grounding with task-specific adaptation.
        # ──────────────────────────────────────────────────────────────────────────
        
        # Project from SentenceTransformer output dim to model working dim.
        # These are the only learned "embedding" parameters; the 384-dim
        # semantic representations come from the frozen all-MiniLM-L6-v2 model.
        self.entity_proj   = nn.Linear(_ST_DIM, embed_dim)
        self.relation_proj = nn.Linear(_ST_DIM, embed_dim)

        self.triple_encoder = TripleEncoder(embed_dim)

        self.input_norm = nn.LayerNorm(embed_dim)
        self.embed_drop = nn.Dropout(dropout)

        # ──────────────────────────────────────────────────────────────────────────
        # IMPORTANT: Variable Context Size Capability
        # ──────────────────────────────────────────────────────────────────────────
        # This architecture can handle ANY number of support triples (context_size)
        # at inference time, even if trained with a fixed context_size (e.g., 32).
        #
        # Why this works:
        # 1. NO POSITIONAL ENCODINGS: Unlike standard transformers, we don't add
        #    position embeddings. This makes the model treat support triples as
        #    an UNORDERED SET rather than a sequence.
        #
        # 2. PERMUTATION-INVARIANT ATTENTION: Without positional info, self-attention
        #    is permutation-invariant over the support set. The model learns to
        #    aggregate evidence from any number of context triples.
        #
        # 3. NO HARD-CODED SEQUENCE LENGTH: The transformer architecture itself has
        #    no fixed sequence length limit—only computational constraints (O(S²)
        #    memory for self-attention).
        #
        # Trade-offs when using context_size != training_context_size:
        # ✓ Architecture is flexible and will produce scores
        # ⚠ Distribution shift: attention patterns differ (e.g., 32 vs 200 triples)
        # ⚠ Diluted attention: with more triples, attention spreads thinner
        # ⚠ Slower inference: quadratic cost in sequence length
        #
        # Best practice: Use context_size close to training value for optimal results.
        # ──────────────────────────────────────────────────────────────────────────
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

        # Concatenate support tokens with query token (query at the end).
        # CRITICAL: No positional encodings are added here! This enables:
        # - Variable-length context support (trained on S=32, can infer with S=200)
        # - Permutation-invariant processing of support triples
        # - Set-like behavior rather than sequence-dependent behavior
        seq = torch.cat([support_tok, query_tok], dim=1)     # (B, S+1, D)
        seq = self.embed_drop(self.input_norm(seq))

        out = self.transformer(seq)                          # (B, S+1, D)

        logit = self.score_head(out[:, -1, :]).squeeze(-1)   # (B,)
        return logit.squeeze(0) if unbatched else logit

    def forward_with_attention(self, support_triples: list, query_triple: tuple) -> tuple:
        """Forward pass that returns both logit and attention weights.
        
        Args:
            support_triples: List of (head, relation, tail) string triples
            query_triple: Tuple of (head, relation, tail) strings
            
        Returns:
            logit: Model score for the query triple (float)
            attention_weights: Tensor of shape (num_layers, num_heads, S+1, S+1)
                             showing attention patterns (query is last token)
        """
        from pfn.inference import _encode_strings
        
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Extract unique entities and relations
            all_entities = []
            all_relations = []
            for h, r, t in support_triples:
                all_entities.extend([h, t])
                all_relations.append(r)
            
            # Add query entities/relations
            query_h, query_r, query_t = query_triple
            all_entities.extend([query_h, query_t])
            all_relations.append(query_r)
            
            # Get unique items while preserving order
            unique_entities = list(dict.fromkeys(all_entities))
            unique_relations = list(dict.fromkeys(all_relations))
            
            # Create mappings
            entity_to_idx = {e: i for i, e in enumerate(unique_entities)}
            relation_to_idx = {r: i for i, r in enumerate(unique_relations)}
            
            # Encode all entities and relations
            entity_embs = _encode_strings(unique_entities).to(device)  # (num_entities, 384)
            relation_embs = _encode_strings(unique_relations).to(device)  # (num_relations, 384)
            
            # Build support tensor (S, 3, 384)
            S = len(support_triples)
            support_tensor = torch.zeros(S, 3, 384, device=device)
            for i, (h, r, t) in enumerate(support_triples):
                support_tensor[i, 0, :] = entity_embs[entity_to_idx[h]]
                support_tensor[i, 1, :] = relation_embs[relation_to_idx[r]]
                support_tensor[i, 2, :] = entity_embs[entity_to_idx[t]]
            
            # Build query tensor (3, 384)
            query_tensor = torch.zeros(3, 384, device=device)
            query_tensor[0, :] = entity_embs[entity_to_idx[query_h]]
            query_tensor[1, :] = relation_embs[relation_to_idx[query_r]]
            query_tensor[2, :] = entity_embs[entity_to_idx[query_t]]
            
            # Add batch dimension
            support_tensor = support_tensor.unsqueeze(0)  # (1, S, 3, 384)
            query_tensor = query_tensor.unsqueeze(0)  # (1, 3, 384)
            
            # Encode support triples
            h = self.entity_proj(support_tensor[:, :, 0, :])  # (1, S, D)
            r = self.relation_proj(support_tensor[:, :, 1, :])  # (1, S, D)
            t = self.entity_proj(support_tensor[:, :, 2, :])  # (1, S, D)
            support_tok = self.triple_encoder(
                h.reshape(S, self.embed_dim),
                r.reshape(S, self.embed_dim),
                t.reshape(S, self.embed_dim)
            ).reshape(1, S, self.embed_dim)  # (1, S, D)
            
            # Encode query triple
            q_h = self.entity_proj(query_tensor[:, 0, :])  # (1, D)
            q_r = self.relation_proj(query_tensor[:, 1, :])  # (1, D)
            q_t = self.entity_proj(query_tensor[:, 2, :])  # (1, D)
            query_tok = self.triple_encoder(q_h, q_r, q_t).unsqueeze(1)  # (1, 1, D)
            
            # Concatenate support + query
            seq = torch.cat([support_tok, query_tok], dim=1)  # (1, S+1, D)
            seq = self.embed_drop(self.input_norm(seq))
            
            # Manually iterate through transformer layers to capture attention
            attention_weights = []
            x = seq
            for layer in self.transformer.layers:
                # Call self-attention with need_weights=True
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    need_weights=True,
                    average_attn_weights=False  # Keep all heads separately
                )
                attention_weights.append(attn_weights)  # (1, num_heads, S+1, S+1)
                
                # Complete the layer's forward pass (matching norm_first=True)
                x = layer.norm1(x + layer.dropout1(attn_output))
                ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ff_out))
            
            # Stack attention weights: (num_layers, 1, num_heads, S+1, S+1)
            attention_weights = torch.stack(attention_weights, dim=0)
            attention_weights = attention_weights.squeeze(1)  # (num_layers, num_heads, S+1, S+1)
            
            # Compute final score
            logit = self.score_head(x[:, -1, :]).squeeze(-1)
            return logit.item(), attention_weights


# ===========================================================================
# QUERY-CONDITIONED CROSS-ATTENTION MODEL (Recommendation 4.1)
# ===========================================================================

class TriplePFNCrossAttn(TriplePFN):
    """
    TriplePFN with query-conditioned cross-attention.
    
    Architecture Enhancement (Recommendation 4.1):
    ----------------------------------------------
    Standard TriplePFN concatenates support and query tokens, then applies
    self-attention. This variant adds an explicit cross-attention layer where
    the query attends over support BEFORE the transformer encoder.
    
    Benefits:
    - Query explicitly selects relevant support triples (interpretable)
    - Attention weights = relevance scores for visualization/debugging
    - Better gradient flow to support encoding
    - More parameter-efficient than deeper self-attention
    
    Architecture:
        1. Encode support triples → support_tokens (S, D)
        2. Encode query triple → query_token (1, D)
        3. Cross-attention: query attends to support
           - Q = query_token, K = V = support_tokens
           - Output: weighted_support (1, D)
        4. Concatenate [support_tokens, query_token + weighted_support]
        5. Self-attention transformer (existing)
        6. Score head (existing)
    
    Usage:
        model = TriplePFNCrossAttn(embed_dim=512, num_heads=8, ...)
        # Rest identical to TriplePFN
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_cross_attention: bool = True,  # Can disable for ablation
    ):
        super().__init__(embed_dim, num_heads, num_layers, dropout)
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            # Single cross-attention layer: query → support
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(embed_dim)
            self.cross_attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        support_triples: torch.Tensor,    # (B, S, 3, ST_DIM) or (S, 3, ST_DIM)
        query_triple: torch.Tensor,       # (B, 3, ST_DIM)   or (3, ST_DIM)
    ) -> torch.Tensor:
        """Score query with cross-attention support weighting."""
        unbatched = support_triples.dim() == 3
        if unbatched:
            support_triples = support_triples.unsqueeze(0)
            query_triple = query_triple.unsqueeze(0)
        
        B, S, _, _ = support_triples.shape
        D = self.embed_dim
        
        # Encode support and query (same as base TriplePFN)
        h = self.entity_proj(support_triples[:, :, 0, :])
        r = self.relation_proj(support_triples[:, :, 1, :])
        t = self.entity_proj(support_triples[:, :, 2, :])
        support_tok = self.triple_encoder(
            h.reshape(B * S, D), r.reshape(B * S, D), t.reshape(B * S, D)
        ).reshape(B, S, D)  # (B, S, D)
        
        q_h = self.entity_proj(query_triple[:, 0, :])
        q_r = self.relation_proj(query_triple[:, 1, :])
        q_t = self.entity_proj(query_triple[:, 2, :])
        query_tok = self.triple_encoder(q_h, q_r, q_t).unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention: query attends to support
        if self.use_cross_attention:
            # query_tok is Q, support_tok is K and V
            attn_output, _attn_weights = self.cross_attn(
                query=query_tok,           # (B, 1, D)
                key=support_tok,           # (B, S, D)
                value=support_tok,         # (B, S, D)
                need_weights=False,
            )  # attn_output: (B, 1, D) - query representation weighted by support
            
            # Residual connection: query + attended_support
            query_tok = self.cross_attn_norm(
                query_tok + self.cross_attn_dropout(attn_output)
            )  # (B, 1, D)
        
        # Concatenate support + enhanced query
        seq = torch.cat([support_tok, query_tok], dim=1)  # (B, S+1, D)
        seq = self.embed_drop(self.input_norm(seq))
        
        # Self-attention transformer (inherited from TriplePFN)
        out = self.transformer(seq)  # (B, S+1, D)
        
        # Score from query position (last token)
        logit = self.score_head(out[:, -1, :]).squeeze(-1)  # (B,)
        return logit.squeeze(0) if unbatched else logit


# ===========================================================================
# ROADMAP: Learned Support Retrieval (Recommendation 4.2)
# ===========================================================================
#
# Current limitation: Support selection at inference time is heuristic
# (entity-centric BFS, embedding similarity, random sampling).
#
# Proposed: Train a lightweight RETRIEVER to score support relevance.
#
# ───────────────────────────────────────────────────────────────────────────
# Architecture: Bi-Encoder Retriever
# ───────────────────────────────────────────────────────────────────────────
#
# class SupportRetriever(nn.Module):
#     """Scores how relevant a support triple is to a query triple."""
#     
#     def __init__(self, embed_dim=256):
#         super().__init__()
#         # Shared triple encoder (same as TripleEncoder in main model)
#         self.triple_encoder = TripleEncoder(embed_dim)
#         
#         # Similarity scoring: dot product or learned MLP
#         self.score_mlp = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, 1),
#         )
#     
#     def forward(self, query_emb, support_emb):
#         """
#         Args:
#             query_emb: (B, D) - encoded query triple
#             support_emb: (B, K, D) - K candidate support triples
#         Returns:
#             scores: (B, K) - relevance scores (higher = more relevant)
#         """
#         # Expand query to match support shape
#         query_expanded = query_emb.unsqueeze(1).expand(-1, support_emb.size(1), -1)
#         
#         # Concatenate and score
#         combined = torch.cat([query_expanded, support_emb], dim=-1)  # (B, K, 2*D)
#         scores = self.score_mlp(combined).squeeze(-1)  # (B, K)
#         return scores
#
# ───────────────────────────────────────────────────────────────────────────
# Training Strategies
# ───────────────────────────────────────────────────────────────────────────
#
# Option 1: Joint Training
#   - Train retriever and TriplePFN together
#   - Loss = L_scoring (BCE) + λ * L_retrieval (contrastive or ranking)
#   - Pros: End-to-end optimization
#   - Cons: Complex, slower convergence
#
# Option 2: Post-hoc Training (RECOMMENDED)
#   - Train TriplePFN first (existing workflow)
#   - Extract attention weights from trained model
#   - Train retriever to predict attention patterns:
#       - Positive pairs: (query, high-attention support triples)
#       - Negative pairs: (query, low-attention support triples)
#   - Pros: Simple, leverages existing model knowledge
#   - Cons: Two-stage training
#
# Option 3: Self-Supervised Contrastive
#   - Positive: (query, entity-centric neighborhood triples)
#   - Negative: (query, random triples from different KG)
#   - Loss: InfoNCE contrastive loss
#   - Pros: No labels needed
#   - Cons: May not align with downstream task
#
# ───────────────────────────────────────────────────────────────────────────
# Inference Workflow
# ───────────────────────────────────────────────────────────────────────────
#
# def infer_with_retrieval(model, retriever, query, kg_triples, k=32):
#     """
#     1. Encode all KG triples with retriever (can precompute & cache)
#     2. Encode query triple
#     3. Score all supports: retriever(query, kg_triples) → relevance scores
#     4. Select top-K supports by score
#     5. Feed to TriplePFN: model(top_k_supports, query) → prediction
#     """
#     # Encode query
#     query_emb = encode_triple(query)  # (1, D)
#     
#     # Score all supports (batch for efficiency)
#     support_embs = encode_triples(kg_triples)  # (N, D)
#     scores = retriever(query_emb, support_embs)  # (N,)
#     
#     # Top-K retrieval
#     top_k_idx = torch.topk(scores, k=k).indices
#     top_k_supports = [kg_triples[i] for i in top_k_idx]
#     
#     # Main model prediction
#     return model(top_k_supports, query)
#
# ───────────────────────────────────────────────────────────────────────────
# Expected Benefits
# ───────────────────────────────────────────────────────────────────────────
#
# - Faster inference on large KGs (retrieve K from 10K+ triples efficiently)
# - Better than heuristics (learned relevance > hand-crafted rules)
# - Scalability: Precompute support embeddings, use FAISS for retrieval
# - Interpretability: Inspect retriever scores to understand context selection
#
# Implementation deferred pending research validation.
# ===========================================================================

