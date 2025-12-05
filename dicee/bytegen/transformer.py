import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_p = config.dropout

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size()
        
        # Calculate q, k, v
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Concatenate past key-values if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        present_kv = (k, v) if use_cache else None

        # Flash Attention
        attn_mask = None
        is_causal = True
        
        if past_kv is not None:
            # If we are generating token-by-token (T=1), we can attend to everything
            if T == 1:
                is_causal = False
            else:
                # If processing a chunk (T > 1) with history, we need a mask
                # to prevent attending to future tokens within the chunk.
                # Q index i matches K index (past_len + i).
                # We want mask[i, j] = 0 if j <= past_len + i else -inf
                
                past_len = past_k.size(2)
                total_len = k.size(2)
                
                # Create mask: (B, 1, T, total_len) or broadcastable
                # Indices for Q: [0, ..., T-1]
                # Indices for K: [0, ..., past_len + T - 1]
                
                # Standard causal mask logic adapted for offset
                # Using manual mask with is_causal=False
                mask = torch.ones((T, total_len), dtype=torch.bool, device=x.device)
                mask_q = torch.arange(T, device=x.device).unsqueeze(1) # (T, 1)
                mask_k = torch.arange(total_len, device=x.device).unsqueeze(0) # (1, total_len)
                
                # Allow attention if k_idx <= q_idx + past_len
                mask = mask_k <= (mask_q + past_len)
                
                attn_mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, T, total_len)
                is_causal = False

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y)), present_kv

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
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

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv