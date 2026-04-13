from dataclasses import dataclass
from .base_model import BaseKGE
from typing import Tuple
import torch
import numpy as np
import os
import torch.distributed as dist
from dicee.models.transformers import Block
from torch import nn


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DistMult(BaseKGE):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    https://arxiv.org/abs/1412.6575"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'DistMult'
        self.manual_sharded_entity_training = self.defer_large_embeddings
        self.local_entity_embeddings = None
        self.local_entity_start = 0
        self.local_entity_end = self.num_entities
        self.local_entity_count = self.num_entities
        if self.manual_sharded_entity_training:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
            shard_size = (self.num_entities + world_size - 1) // world_size
            self.local_entity_start = rank * shard_size
            self.local_entity_end = min(self.local_entity_start + shard_size, self.num_entities)
            self.local_entity_count = max(0, self.local_entity_end - self.local_entity_start)
            self.local_entity_embeddings = torch.nn.Embedding(self.local_entity_count, self.embedding_dim, sparse=True)
            self.param_init(self.local_entity_embeddings.weight.data)

    def configure_optimizers(self, parameters=None):
        if not self.manual_sharded_entity_training:
            return super().configure_optimizers(parameters=parameters)

        dense_parameters = [param for name, param in self.named_parameters() if name != "local_entity_embeddings.weight"]
        return super().configure_optimizers(parameters=dense_parameters)

    def _entity_lookup(self, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        if not self.manual_sharded_entity_training:
            return self.entity_embeddings(entity_ids)

        outputs = torch.zeros(
            entity_ids.shape[0],
            self.embedding_dim,
            device=entity_ids.device,
            dtype=self.local_entity_embeddings.weight.dtype,
        )
        mask = (entity_ids >= self.local_entity_start) & (entity_ids < self.local_entity_end)
        if mask.any():
            local_ids = entity_ids[mask] - self.local_entity_start
            outputs[mask] = self.local_entity_embeddings(local_ids)
        return _AllReduceSum.apply(outputs)

    def get_triple_representation(self, idx_hrt):
        if not self.manual_sharded_entity_training:
            return super().get_triple_representation(idx_hrt)
        idx_head_entity, idx_relation, idx_tail_entity = idx_hrt[:, 0], idx_hrt[:, 1], idx_hrt[:, 2]
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self._entity_lookup(idx_head_entity))
        )
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        tail_ent_emb = self.normalize_tail_entity_embeddings(self._entity_lookup(idx_tail_entity))
        return head_ent_emb, rel_ent_emb, tail_ent_emb

    def get_head_relation_representation(self, indexed_triple):
        if not self.manual_sharded_entity_training:
            return super().get_head_relation_representation(indexed_triple)
        idx_head_entity, idx_relation = indexed_triple[:, 0], indexed_triple[:, 1]
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self._entity_lookup(idx_head_entity))
        )
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return head_ent_emb, rel_ent_emb

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
        if self.manual_sharded_entity_training:
            raise NotImplementedError("Sharded DistMult currently supports only NegSample training.")
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel, emb_E=self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        if self.manual_sharded_entity_training:
            t = self._entity_lookup(target_entity_idx.reshape(-1)).reshape(target_entity_idx.shape[0], target_entity_idx.shape[1], -1)
            emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
            hr = torch.einsum('bd, bd -> bd', emb_head_real, emb_rel_real)
            return torch.einsum('bd, bkd -> bk', hr, t)
        # (b,d),     (b,d)
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        # (b, d)
        hr = torch.einsum('bd, bd -> bd', emb_head_real, emb_rel_real)
        # (b, k, d)
        t = self.entity_embeddings(target_entity_idx)
        return torch.einsum('bd, bkd -> bk', hr, t)


    def score(self, h, r, t):
        return (self.hidden_dropout(self.hidden_normalizer(h * r)) * t).sum(dim=1)

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.manual_sharded_entity_training:
            return super().get_embeddings()
        raise RuntimeError("Full entity embeddings are materialized by the trainer on rank 0 after training.")


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
class CoKEConfig:
    """
    Configuration for the CoKE (Contextualized Knowledge Graph Embedding) model.
    
    Attributes:
        block_size: Sequence length for transformer (3 for triples: head, relation, tail)
        vocab_size: Total vocabulary size (num_entities + num_relations)
        n_layer: Number of transformer layers
        n_head: Number of attention heads per layer
        n_embd: Embedding dimension (set to match model embedding_dim)
        dropout: Dropout rate applied throughout the model
        bias: Whether to use bias in linear layers
        causal: Whether to use causal masking (False for bidirectional attention)
    """
    block_size: int = 3           # triples -> TODO: LF: for multi-hop this needs to be bigger
    vocab_size: int = None        # Must be set to num_entities + num_relations before initializing CoKE
    n_layer: int = 6             
    n_head: int = 8               
    n_embd: int = None             
    dropout: float = 0.3          # according to paper in [0.1 - 0.5]
    bias: bool = True             # idk if better with false?
    causal: bool = False          # non-causal so that we gather information in mask token 

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
    def __init__(self, args, config: CoKEConfig = CoKEConfig()):
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
