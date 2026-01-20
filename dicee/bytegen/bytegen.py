import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal
from dicee.bytegen.transformer import Block

# Configuration
@dataclass
class ByteGenConfig:
    block_size: int = 256
    vocab_size: int = 260  # 256 bytes + PAD(256) + SEP_HR(257) + SEP_RT(258) + EOS(259)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 0.0  # L2 regularization for AdamW optimizer
    batch_size: int = 32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Position embedding configuration
    position_embedding_type: Literal['absolute', 'rope'] = 'absolute'
    rope_base: int = 10000  # Base frequency for RoPE


class ByteGenModel(nn.Module):
    def __init__(self, config: ByteGenConfig):
        super().__init__()
        self.config = config
        
        # Build transformer modules
        modules = {
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        }
        
        # Position embedding only for absolute mode
        if config.position_embedding_type == 'absolute':
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            
            # Special scaled init to the residual projections, per GPT-2 paper
            if getattr(module, '_is_residual_proj', False):
                scale = 1.0 / (2 * self.config.n_layer) ** 0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, past_kvs=None):
        device = idx.device
        b, t = idx.size()
        
        # Calculate position offset for KV cache
        if past_kvs is not None:
            position_offset = past_kvs[0][0].size(2)
        else:
            position_offset = 0
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx)
        
        # Add position embeddings only for absolute mode
        if self.config.position_embedding_type == 'absolute':
            pos = torch.arange(position_offset, position_offset + t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # For RoPE, no additive position embedding - positions are applied in attention
            x = self.transformer.drop(tok_emb)

        new_past_key_values = []
        
        for i, block in enumerate(self.transformer.h):
            layer_past = past_kvs[i] if past_kvs is not None else None
            x, layer_present = block(x, past_kv=layer_past, position_offset=position_offset)
            new_past_key_values.append(layer_present)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_past_key_values

    def generate(self, idx, tokenizer, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        past_kvs = None
        
        with torch.no_grad():
            # Initial forward pass with full context
            logits, past_kvs = self(idx)
            
            for _ in range(max_new_tokens):
                # Get logits for the last token
                next_logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                if idx_next.item() == tokenizer.pad_token_id:
                    break
                    
                idx = torch.cat((idx, idx_next), dim=1)
                
                # Check if we've exceeded block size - if so, reset cache
                if idx.size(1) > self.config.block_size:
                    idx = idx[:, -self.config.block_size:]
                    logits, past_kvs = self(idx)
                else:
                    # Use KV cache - only process the new token
                    logits, past_kvs = self(idx_next, past_kvs=past_kvs)
        
        return idx
