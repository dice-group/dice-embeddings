from dataclasses import dataclass
from .base_model import BaseKGE
from typing import Optional, Tuple
import torch
import numpy as np
from dicee.models.transformers import Block
from torch import nn


class DistMult(BaseKGE):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    https://arxiv.org/abs/1412.6575"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'DistMult'

    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor):
        """

        Parameters
        ----------
        emb_h
        emb_r
        emb_E

        Returns
        -------

        """
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)), emb_E.transpose(1, 0))

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel, emb_E=self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        # (b,d),     (b,d)
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        # (b, d)
        hr = torch.einsum('bd, bd -> bd', emb_head_real, emb_rel_real)
        # (b, k, d)
        t = self.entity_embeddings(target_entity_idx)
        return torch.einsum('bd, bkd -> bk', hr, t)


    def score(self, h, r, t):
        return (self.hidden_dropout(self.hidden_normalizer(h * r)) * t).sum(dim=1)


class TransE(BaseKGE):
    """
    Translating Embeddings for Modeling
    Multi-relational Data
    https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'TransE'
        self._norm = 2
        self.margin = 4

    def score(self, head_ent_emb, rel_ent_emb, tail_ent_emb):
        # Original d:=|| s+p - t||_2 \approx 0 distance, if true
        # if d =0 sigma(5-0) => 1
        # if d =5 sigma(5-5) => 0.5
        # Update: sigmoid( \gamma - d)
        return self.margin - torch.nn.functional.pairwise_distance(head_ent_emb + rel_ent_emb, tail_ent_emb,
                                                                   p=self._norm)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        distance = torch.nn.functional.pairwise_distance(torch.unsqueeze(emb_head_real + emb_rel_real, 1),
                                                         self.entity_embeddings.weight, p=self._norm)
        return self.margin - distance


class Shallom(BaseKGE):
    """ A shallow neural model for relation prediction (https://arxiv.org/abs/2101.09090) """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Shallom'
        shallom_width = int(2 * self.embedding_dim)
        self.shallom = torch.nn.Sequential(torch.nn.Dropout(self.input_dropout_rate),
                                           torch.nn.Linear(self.embedding_dim * 2, shallom_width),
                                           self.normalizer_class(shallom_width),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(self.hidden_dropout_rate),
                                           torch.nn.Linear(shallom_width, self.num_relations))

    def get_embeddings(self) -> Tuple[np.ndarray, None]:
        return self.entity_embeddings.weight.data.detach(), None

    def forward_k_vs_all(self, x) -> torch.FloatTensor:
        e1_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, e2_idx = x[:, 0], x[:, 1]
        emb_s, emb_o = self.entity_embeddings(e1_idx), self.entity_embeddings(e2_idx)
        return self.shallom(torch.cat((emb_s, emb_o), 1))

    def forward_triples(self, x) -> torch.FloatTensor:
        """

        :param x:
        :return:
        """

        n, d = x.shape
        assert d == 3
        scores_for_all_relations = self.forward_k_vs_all(x[:, [0, 2]])
        return scores_for_all_relations[:, x[:, 1]].flatten()


class Pyke(BaseKGE):
    """ A Physical Embedding Model for Knowledge Graphs """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Pyke'
        self.dist_func = torch.nn.PairwiseDistance(p=2)
        self.margin = 1.0

    def forward_triples(self, x: torch.LongTensor):
        # (1) get embeddings for a batch of entities and relations
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Compute the Euclidean distance from head to relation
        dist_head_rel = self.dist_func(head_ent_emb, rel_ent_emb)
        dist_rel_tail = self.dist_func(rel_ent_emb, tail_ent_emb)
        avg_dist = (dist_head_rel + dist_rel_tail) / 2
        return self.margin - avg_dist

@dataclass
class KGTransformerConfig:
    """
    Configuration for Transformer-based KGE backbones.
    
    Attributes:
        block_size: Sequence length (e.g., 3 for triples: h, r, t/mask)
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
        causal: Whether to use causal attention masking
    """
    # Architecture Dimensions
    n_layer: int = 6             
    n_head: int = 8               
    n_embd: int = 256   # Default value (previously None)          
    block_size: int = 3 
    
    # Regularization
    dropout: float = 0.3          
    bias: bool = True             
    causal: bool = False  
    vocab_size: Optional[int] = None

class CoKE(BaseKGE):
    """
    Contextualized Knowledge Graph Embedding (CoKE) model.
    Based on: https://arxiv.org/pdf/1911.02168.
    
    CoKE uses a transformer encoder to learn contextualized representations of entities and relations.
    For link prediction, it predicts masked elements in (head, relation, tail) triples using
    bidirectional attention, similar to BERT's masked language modeling approach.
    
    The model creates a sequence [head_emb, relation_emb, mask_emb], adds positional embeddings,
    and processes it through transformer layers to predict the tail entity.
    """
    def __init__(self, args, config: KGTransformerConfig = KGTransformerConfig()):
        super().__init__(args)
        self.name = 'CoKE'

        # Configure model dimensions
        self.config = config
        self.config.vocab_size = self.num_entities + self.num_relations
        self.config.n_embd = self.embedding_dim
    
        # Positional and mask embeddings
        self.pos_emb = torch.nn.Embedding(config.block_size, self.embedding_dim)
        self.mask_emb = torch.nn.Parameter(torch.zeros(self.embedding_dim))

        # Transformer layers
        self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embedding_dim)

        self.coke_dropout = nn.Dropout(config.dropout)

    def forward_k_vs_all(self, x: torch.Tensor):
        device = x.device
        b = x.size(dim=0)

        # Get embeddings for head and relation
        head_emb, rel_emb = self.get_head_relation_representation(x)  # (b, dim), (b, dim)
        mask_emb = self.mask_emb.unsqueeze(0).expand(b, -1)  # (b, dim)
        
        # Create sequence: [head, relation, mask]
        seq = torch.stack([head_emb, rel_emb, mask_emb], dim=1)  # (b, 3, dim)
        
        # Add positional embeddings
        pos_ids = torch.arange(0, 3, device=device)  # (3,) -> TODO: LF: here 3 has to change according to voacb size (in case we want multi-hop)
        pos_ids = pos_ids.unsqueeze(0).expand(b, 3)  # (b, 3) TODO: LF: same as above
        pos_emb = self.pos_emb(pos_ids)  # (b, 3, dim)
        x_tok = seq + pos_emb  # (b, 3, dim)

        # Pass through transformer layers
        for block in self.blocks:
            x_tok = block(x_tok)
        x_tok = self.ln_f(x_tok)

        # Extract the mask token's hidden state (position 2)
        h_mask = x_tok[:, 2, :]
        h_mask = self.coke_dropout(h_mask)

        # Score against all entity embeddings
        E = self.entity_embeddings.weight
        E = self.normalize_tail_entity_embeddings(E)
        scores = h_mask.mm(E.t())

        return scores 

    def score(self, emb_h, emb_r, emb_t):
        b = emb_h.size(0)
        device = emb_h.device
        
        # Create sequence with mask token
        mask_emb = self.mask_emb.unsqueeze(0).expand(b, -1)
        seq = torch.stack([emb_h, emb_r, mask_emb], dim=1)
        
        # Add positional embeddings
        pos_ids = torch.arange(0, 3, device=device).unsqueeze(0).expand(b, 3)
        pos_emb = self.pos_emb(pos_ids)
        x_tok = seq + pos_emb

        # Pass through transformer
        for block in self.blocks:
            x_tok = block(x_tok)
        x_tok = self.ln_f(x_tok)
        
        # Extract mask token hidden state
        h_mask = x_tok[:, 2, :]
        h_mask = self.coke_dropout(h_mask)

        # Compute similarity between mask representation and tail embedding
        score = torch.einsum('bd,bd -> b', h_mask, emb_t)
        return score

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        emb_head, emb_rel = self.get_head_relation_representation(x)
        b = emb_head.size(0)
        emb_tail = self.entity_embeddings(target_entity_idx)  # (b, k, dim)
        device = emb_head.device
        
        # Create sequence with mask token
        mask_emb = self.mask_emb.unsqueeze(0).expand(b, -1)
        seq = torch.stack([emb_head, emb_rel, mask_emb], dim=1)
        
        # Add positional embeddings
        pos_ids = torch.arange(0, 3, device=device).unsqueeze(0).expand(b, 3)
        pos_emb = self.pos_emb(pos_ids)
        x_tok = seq + pos_emb
        
        # Pass through transformer
        for block in self.blocks:
            x_tok = block(x_tok)
        x_tok = self.ln_f(x_tok)
        
        # Extract mask token hidden state
        h_mask = x_tok[:, 2, :]
        h_mask = self.coke_dropout(h_mask)

        scores = torch.einsum('bd, bkd -> bk', h_mask, emb_tail) # dot product between each batch (how simlar is mask to all k tails in batch x)
                                                         #output: (b,k) -> k scores per batch

        return scores

class ByteEncoder(nn.Module):
    def __init__(self, d_model, n_head=4, n_layer=2, max_len=128):
        super().__init__()

        #we need 257 tokens, ie each byte one and than the padding token at index 256 -> 260 could help performance
        self.byte_emb = nn.Embedding(260, d_model, padding_idx=256)

        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
                                    d_model=d_model, 
                                    nhead=n_head,
                                    dim_feedforward=4*d_model, # expansion ratio
                                    dropout=0.1,
                                    activation='gelu',         
                                    batch_first=True,          # Expects (Batch, Seq, Dim)
                                    norm_first=True            # makes it "Pre-LN" 
                                )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model) 

    def forward(self, x, mask=None):
        device = x.device
        b, t = x.size()

        # get position_ids upto seq_len t 
        pos_idx = torch.arange(0, t, device=device).unsqueeze(0)

        # get embeddings and add position embeddings for all tokens
        x_emb = self.byte_emb(x) + self.pos_emb(pos_idx)

        # forward pass trough transformer (masking paddings for attention computation)
        out = self.transformer(x_emb, src_key_padding_mask=(x == 256))

        # mask where non padding tokens get true
        mask_expanded = (x != 256).unsqueeze(-1)
        # sum embedded tokens (only if non-padding token)
        sum_embeddings = torch.sum(out * mask_expanded, dim=1)
        # count how many tokens are non padding tokens 
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # take average and apply layer nomralization
        return self.ln_f(sum_embeddings / sum_mask) 


class BET(BaseKGE):
    """
    BET: Byte Entity Transformer
    
    A Knowledge Graph Embedding model that constructs entity representations 
    directly from raw bytes using a hierarchical Transformer.
    
    Note: BET requires byte_level_encoding=True and works with NegSample scoring.
    """
    def __init__(self, args, config: KGTransformerConfig = KGTransformerConfig()):
        super().__init__(args)
        self.name = 'BET'

        # Configure model dimensions
        self.config = config
        self.config.n_embd = self.embedding_dim
        
        dataset_max_len = getattr(args, 'max_byte_sequence_length', 128)

        self.byte_encoder = ByteEncoder(d_model=self.config.n_embd, max_len=dataset_max_len) 
         
        # Positional and mask embeddings
        self.pos_emb = torch.nn.Embedding(self.config.block_size, self.config.n_embd)
        self.mask_emb = torch.nn.Parameter(torch.zeros(self.config.n_embd))

        # Transformer layers
        self.blocks = torch.nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)

        self.dropout = nn.Dropout(self.config.dropout)
    
    def forward(self, x):
            """
            Overwrites BaseKGE.forward to handle byte inputs.
            
            param x: Tensor of shape (Batch_Size, 3, Seq_Len)
                    containing (Head_Bytes, Relation_Bytes, Tail_Bytes)
            """
            # Unpack the triplet dimension
            # x[:, 0] -> Head Bytes
            # x[:, 1] -> Relation Bytes
            # x[:, 2] -> Tail Bytes
            h_bytes = x[:, 0, :]
            r_bytes = x[:, 1, :]
            t_bytes = x[:, 2, :]
            
            # Call your byte-based scoring function
            return self.score(h_bytes, r_bytes, t_bytes)
        
    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass for triples with byte-encoded inputs.
        Called by forward() - delegates to score().
        
        Args:
            x: Byte-encoded tensor of shape (batch, 3, seq_len)
        
        Returns:
            Scores for each triple
        """
        return self.forward(x)

    def score(self, h_bytes, r_bytes, t_bytes):
        emb_h = self.byte_encoder(h_bytes)
        emb_r = self.byte_encoder(r_bytes)

        b = emb_h.size(0)
        device = emb_h.device

        mask_emb = self.mask_emb.unsqueeze(0).expand(b, -1)

        seq = torch.stack([emb_h, emb_r, mask_emb], dim=1)

        pos_ids = torch.arange(0, 3, device=device).unsqueeze(0).expand(b, 3)
        x_tok = seq + self.pos_emb(pos_ids)

        for block in self.blocks:
            x_tok = block(x_tok)
        
        h_mask = self.ln_f(x_tok[:,2,:])

        emb_t = self.byte_encoder(t_bytes)
        scores = torch.sum(h_mask * emb_t, dim=1)
        return scores
