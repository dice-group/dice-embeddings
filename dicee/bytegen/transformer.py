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

    def forward(self, x, past_kv=None):
        B, T, C = x.size()
        
        # Calculate q, k, v
        # shape: (B, T, n_head, head_dim)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # Concat along time dimension
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
        else:
            # With cache: need custom mask that allows full attention to prefix
            # but causal attention among new tokens
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

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv