import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse

from .base_model import BaseKGE
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear

class Transformer2CL(nn.Module):
            def __init__(self, in_channels, n_blades):
                super(Transformer2CL, self).__init__()
                self.in_channels = in_channels
                self.n_blades = n_blades

            def forward(self, x):
                return x.view(-1, self.in_channels, self.n_blades)
            
class OutputReshape(nn.Module):
    def __init__(self, out_layer):
        super(OutputReshape, self).__init__()
        self.out = out_layer

    def forward(self, x):
        return self.out(x)[:,:,0]

class Transformer(nn.Module):
    """Transformer for knowledge graph embeddings
    
    Takes entity and relation embeddings and processes them through transformer layers
    to predict entity scores.
    
    The input embeddings can be arranged in different ways:
    - (batch_size, 2d, 1): 2d tokens with embedding size 1
    - (batch_size, d, 2): d tokens with embedding size 2
    - (batch_size, d/2, 4): d/2 tokens with embedding size 4
    etc.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_layer = config.n_layer
        self.bias = config.bias
        
        self.return_attention = config.return_attention
        # Entity-relation embedding into logits
        self.lm_head = nn.Linear(in_features=config.in_features,
                                out_features=config.out_features)
        
        self.token_size = config.in_features // config.n_embd  # Number of tokens derived from embedding size
        assert config.in_features % self.token_size == 0, \
            f"in_features ({config.in_features}) must be divisible by token_size ({self.token_size})"

        # Layer normalization weights and biases for all layers
        self.ln_weights = nn.ParameterList([nn.Parameter(torch.ones(config.n_embd)) for _ in range(2 * config.n_layer)])
        self.ln_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None
            for _ in range(2 * config.n_layer)
        ])

        # Attention projections for all layers
        self.attn_projections = nn.ModuleList([
            nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            for _ in range(config.n_layer)
        ])
        self.attn_output_projections = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            for _ in range(config.n_layer)
        ])

        # MLP projections for all layers
        self.mlp_fc = nn.ModuleList([
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            for _ in range(config.n_layer)
        ])
        self.mlp_proj = nn.ModuleList([
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            for _ in range(config.n_layer)
        ])

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.mlp_dropout = nn.Dropout(config.dropout)

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and not self.return_attention
        if not self.flash:
            # Use a persistent buffer for the causal mask if flash is unavailable
            self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.token_size, self.token_size, dtype=torch.bool)).view(1, 1, self.token_size, self.token_size),
            persistent=False
            )

    def layer_norm(self, x, layer_idx):
        """Apply layer normalization with optional bias"""
        weight = self.ln_weights[layer_idx]
        bias = self.ln_biases[layer_idx] if self.bias else None
        return F.layer_norm(x, weight.shape, weight, bias, 1e-5)

    def causal_self_attention(self, x, layer_idx):
        """Apply causal self-attention for a specific layer"""
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.attn_projections[layer_idx](x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.causal_mask == 0, float('-inf'))  # use causal_mask here
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs

        # Output projection
        y = self.resid_dropout(self.attn_output_projections[layer_idx](y))
        return y

    def mlp(self, x, layer_idx):
        """Apply MLP for a specific layer"""
        x = self.mlp_fc[layer_idx](x)
        x = F.gelu(x)
        x = self.mlp_proj[layer_idx](x)
        x = self.mlp_dropout(x)
        return x

    def transformer_block(self, x, layer_idx):
        """Apply a complete transformer block (attention + MLP)"""
        # Layer norm indices: layer_idx * 2 for attention, layer_idx * 2 + 1 for MLP
        attn_ln_idx = layer_idx * 2
        mlp_ln_idx = layer_idx * 2 + 1

        # Attention with residual connection
        x = x + self.causal_self_attention(self.layer_norm(x, attn_ln_idx), layer_idx)

        # MLP with residual connection
        x = x + self.mlp(self.layer_norm(x, mlp_ln_idx), layer_idx)

        return x

    def forward(self,x):
        """Forward pass through all transformer blocks
        
        Parameters
        ----------
        x : torch.FloatTensor
            Input tensor of shape (batch_size, [emb_h; emb_r])
        emb_h : torch.FloatTensor
            Head entity embeddings of shape (batch_size, embedding_dim)
        emb_r : torch.FloatTensor
            Relation embeddings of shape (batch_size, embedding_dim)
            
        Returns
        -------
        torch.FloatTensor
            Logits for all entities of shape (batch_size, num_entities)
        """
        # Concatenate head and relation embeddings
        # Shape: (batch_size, 2*embedding_dim)
        # x = torch.cat([emb_h, emb_r], dim=1)
        
        # Reshape into tokens
        # Shape: (batch_size, token_size, inner_embedding_size)
        x = x.view(x.size(0), -1, self.n_embd)  # last dimension = 1

        # Apply transformer blocks
        for layer_idx in range(self.n_layer):
            x = self.transformer_block(x, layer_idx)

        # Flatten and project to entity scores
        return torch.flatten(x, start_dim=1)
        # a =x.shape
        # return self.lm_head(x)


class TapireCL(BaseKGE):
    """
    TapireCL: Transformer-based Clifford Linear Knowledge Graph Embedding Model.

    This model combines a transformer architecture with Clifford linear layers to capture
    complex relationships between entities and relations in a knowledge graph.

    Workflow:
        1. Takes a batch of (head, relation) pairs as input.
        2. Retrieves and concatenates their embeddings.
        3. Reshapes the concatenated embeddings into a sequence of tokens.
        4. Processes the sequence through a transformer network.
        5. Applies Clifford linear layers and normalization.
        6. Outputs scores for all possible tail entities.

    Args:
        args (dict): Configuration dictionary with keys:
            - embedding_dim: Dimension of entity/relation embeddings.
            - num_entities: Number of entities.
            - num_relations: Number of relations.
            - inner_embedding_size: Size of token embeddings.
            - n_layer: Number of transformer layers.
            - n_head: Number of attention heads.
            - dropout: Dropout probability.

    Methods:
        k_vs_all_score(emb_h, emb_r): Computes scores for all tail entities.
        forward_k_vs_all(x): Forward pass for k-vs-all scoring.
        forward_k_vs_sample(x, target_entity_idx): Forward pass for k-vs-sample scoring (not implemented).
        score(h, r, t): Computes the score for a specific triple (not implemented).
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'TapireCL'

        # () inner_embedding_size determines the size of the token embeddings.
        # () Remaining part of embedding_dim is used as the number of tokens.

        # Create configuration
        n_layers = args.get('n_layer',8 )
        n_heads = args.get('n_head', 2)
        embedding_dim = args.get('embedding_dim', 32)
        in_features = embedding_dim * 2  # Concatenated head and relation embeddings
        out_features =   in_features  # Output features match input features
        dropout = args.get('dropout', 0)
        inner_embedding_dim = args.get('inner_embedding_size', 8)

        assert n_layers > 0, "n_layer must be greater than 0"
        assert n_heads > 0, "n_head must be greater than 0"

        # assert embedding_dim == inner_embedding_dim or \
        #     embedding_dim == inner_embedding_dim * 2 or \
        #     embedding_dim == inner_embedding_dim // 2, \
        #     "inner_embedding_dim must be equal to, half of, or double embedding_dim"

        assert in_features % inner_embedding_dim == 0, \
            f"in_features ({in_features}) must be divisible by inner_embedding_size ({inner_embedding_dim})"
        assert inner_embedding_dim % n_heads == 0, \
            f"inner_embedding_size ({inner_embedding_dim}) must be divisible by n_head ({n_heads})"

        
        # Create transformer model
        transformer_config = argparse.Namespace(dropout=dropout,
                                                n_layer=n_layers,
                                                n_head=n_heads,
                                                n_embd=inner_embedding_dim,
                                                bias=True,
                                                in_features=in_features,
                                                out_features=out_features,
                                                return_attention=True)
        
        
        self.g = [-1]  # Clifford algebra basis elements
        self.n_blades = 2 ** len(self.g)
        self.in_channels = (
            self.embedding_dim * 2 // self.n_blades
        )

        self.tapireCL = nn.Sequential(
            Transformer(config=transformer_config),
            Transformer2CL(
                in_channels=self.in_channels,
                n_blades=self.n_blades
            ),
            CliffordLinear(
            g=self.g,
            in_channels=self.in_channels,
            out_channels=self.embedding_dim * 2,
            bias=True,
            ),
            torch.nn.LayerNorm([self.embedding_dim * 2, self.n_blades]),
           
            OutputReshape( CliffordLinear(
            g=self.g,
            in_channels=self.embedding_dim * 2,
            out_channels=self.num_entities,
            bias=True,
            ))
        )


    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor):
        """
        Parameters
        ----------
        emb_h : n by d tensor
        emb_r : n by d tensor

        Returns
        -------

        """
        return self.tapireCL(torch.cat((emb_h, emb_r), dim=1))

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        # (b,d),     (b,d)
        raise NotImplemented


    def score(self, h, r, t):
        raise NotImplementedError("score method is not implemented in TapireCL model.")
        return (self.tapireCL(torch.cat((h, r), dim=1)) * t).sum(dim=1)