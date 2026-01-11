import torch
from torch import nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Su et al. 2021
    
    Applies rotation to query and key vectors based on their absolute positions.
    This allows the model to learn relative positions through dot product properties.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin cache for efficiency
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions [0, seq_len)"""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        # Outer product: (seq_len, dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate for real/imaginary pairs: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0):
        """
        Apply rotary embeddings to q and k.
        
        Args:
            q: Query tensor of shape (B, n_head, T, head_dim)
            k: Key tensor of shape (B, n_head, T, head_dim)
            position_offset: Starting position (for KV cache support)
            
        Returns:
            Rotated q and k tensors
        """
        seq_len = q.size(2)
        
        # Extend cache if needed
        if position_offset + seq_len > self.max_seq_len:
            self._build_cache(position_offset + seq_len)
            # Move buffers to correct device
            self.cos_cached = self.cos_cached.to(q.device)
            self.sin_cached = self.sin_cached.to(q.device)
        
        # Get cos/sin for the current positions
        cos = self.cos_cached[position_offset : position_offset + seq_len]
        sin = self.sin_cached[position_offset : position_offset + seq_len]
        
        # Apply rotation
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Apply rotary embedding to tensor x.
        
        Uses the rotation formula:
        x_rot = x * cos + rotate_half(x) * sin
        """
        # x shape: (B, n_head, T, head_dim)
        # cos/sin shape: (T, head_dim)
        
        # Reshape for broadcasting: (1, 1, T, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Rotate half: swap and negate pairs
        x_rotated = self._rotate_half(x)
        
        return (x * cos) + (x_rotated * sin)
    
    def _rotate_half(self, x: torch.Tensor):
        """Rotate half the hidden dims: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj._is_residual_proj = True
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        
        # Position embedding type
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        
        # Initialize RoPE if needed
        if self.position_embedding_type == 'rope':
            self.rotary_emb = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_seq_len=config.block_size,
                base=getattr(config, 'rope_base', 10000)
            )
        else:
            self.rotary_emb = None

    def forward(self, x, past_kv=None, position_offset: int = 0):
        B, T, C = x.size()
        
        # Calculate q, k, v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE before concatenating with cache
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_offset=position_offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        present_kv = (k, v)
        
        # Handle attention masking
        if past_kv is None:
            # No cache: use standard causal mask
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True
            )
        elif T == 1:
            # Single token generation: can attend to all cached tokens, no mask needed
            # This path uses FlashAttention when available
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Multiple new tokens with cache: need custom mask that allows full attention 
            # to prefix but causal attention among new tokens
            # Note: This path falls back to memory-efficient attention (no FlashAttention)
            T_new = T  # Number of new query positions
            T_total = k.size(2)  # Total KV length (prefix + new)
            T_prefix = T_total - T_new
            
            # Build attention mask: (T_new, T_total)
            # - All positions can attend to all prefix positions
            # - Causal mask among new tokens
            prefix_mask = torch.ones(T_new, T_prefix, dtype=torch.bool, device=x.device)
            causal_mask = torch.tril(torch.ones(T_new, T_new, dtype=torch.bool, device=x.device))
            attn_mask = torch.cat([prefix_mask, causal_mask], dim=1)  # (T_new, T_total)
            
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False  # We provide our own mask
            )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present_kv


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj._is_residual_proj = True
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ GPT-Style Block """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, position_offset: int = 0):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, position_offset=position_offset)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv
