from dataclasses import dataclass
from .base_model import BaseKGE
from typing import Tuple
import torch
import numpy as np
from dicee.models.transformers import Block
from torch import nn


class DistMult(BaseKGE):
    """DistMult: bilinear diagonal knowledge graph embedding.

    Scores a triple ``(h, r, t)`` as the element-wise product of the head,
    relation, and tail embeddings summed over the embedding dimension::

        f(h, r, t) = \\sum_i  h_i  \\cdot  r_i  \\cdot  t_i

    Simple yet effective baseline; incapable of modelling asymmetric
    relations.

    References
    ----------
    Yang et al., *Embedding Entities and Relations for Learning and Inference
    in Knowledge Bases*, ICLR 2015.
    https://arxiv.org/abs/1412.6575
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'DistMult'

    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor,
                       emb_E: torch.FloatTensor) -> torch.FloatTensor:
        """Score a head/relation batch against all entity embeddings.

        Computes ``(h * r) @ E^T`` after applying hidden dropout and
        normalisation to the element-wise product.

        Parameters
        ----------
        emb_h : torch.FloatTensor
            Head entity embeddings, shape ``(batch_size, embedding_dim)``.
        emb_r : torch.FloatTensor
            Relation embeddings, shape ``(batch_size, embedding_dim)``.
        emb_E : torch.FloatTensor
            All entity embeddings, shape ``(num_entities, embedding_dim)``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_entities)`` score matrix.
        """
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)), emb_E.transpose(1, 0))

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """KvsAll forward pass: score head/relation against all entities.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 2)`` integer tensor ``[head_idx, relation_idx]``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_entities)`` score matrix.
        """
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel, emb_E=self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor) -> torch.FloatTensor:
        """KvsSample forward pass: score head/relation against a sampled entity subset.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 2)`` integer tensor ``[head_idx, relation_idx]``.
        target_entity_idx : torch.LongTensor
            Shape ``(batch_size, k)`` indices of the *k* target entities per
            sample.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, k)`` score matrix.
        """
        # (b,d),     (b,d)
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        # (b, d)
        hr = torch.einsum('bd, bd -> bd', emb_head_real, emb_rel_real)
        # (b, k, d)
        t = self.entity_embeddings(target_entity_idx)
        return torch.einsum('bd, bkd -> bk', hr, t)


    def score(self, h: torch.FloatTensor, r: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        """Score a batch of ``(head, relation, tail)`` embedding triples.

        Parameters
        ----------
        h, r, t : torch.FloatTensor
            Each has shape ``(batch_size, embedding_dim)``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size,)`` triple scores.
        """
        return (self.hidden_dropout(self.hidden_normalizer(h * r)) * t).sum(dim=1)


class TransE(BaseKGE):
    """TransE: translation-based knowledge graph embedding.

    Models a relation *r* as a translation in embedding space such that
    ``h + r \u2248 t`` for a true triple ``(h, r, t)``.  The score function is
    defined as::

        f(h, r, t) = margin - ||h + r - t||_2

    TransE is effective for 1-to-1 relations but struggles with reflexive,
    one-to-many, and many-to-one patterns.

    References
    ----------
    Bordes et al., *Translating Embeddings for Modeling Multi-relational
    Data*, NeurIPS 2013.
    https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'TransE'
        self._norm = 2
        self.margin = 4

    def score(self, head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor,
              tail_ent_emb: torch.FloatTensor) -> torch.FloatTensor:
        """Score a batch of triples using the TransE margin-distance formula.

        Parameters
        ----------
        head_ent_emb, rel_ent_emb, tail_ent_emb : torch.FloatTensor
            Each has shape ``(batch_size, embedding_dim)``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size,)`` scores equal to
            ``margin - ||h + r - t||_2``.
        """
        # Original d:=|| s+p - t||_2 \approx 0 distance, if true
        # if d =0 sigma(5-0) => 1
        # if d =5 sigma(5-5) => 0.5
        # Update: sigmoid( \gamma - d)
        return self.margin - torch.nn.functional.pairwise_distance(head_ent_emb + rel_ent_emb, tail_ent_emb,
                                                                   p=self._norm)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """KvsAll forward pass: score head/relation against all entities.

        Computes ``margin - ||h + r - e||_2`` for every entity embedding *e*.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, 2)`` integer tensor ``[head_idx, relation_idx]``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_entities)`` score matrix.
        """
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        distance = torch.nn.functional.pairwise_distance(torch.unsqueeze(emb_head_real + emb_rel_real, 1),
                                                         self.entity_embeddings.weight, p=self._norm)
        return self.margin - distance


class Shallom(BaseKGE):
    """Shallom: shallow neural model for relation prediction.

    Represents each triple as the concatenation of head and tail entity
    embeddings and feeds it through a two-layer MLP to predict the
    relation.  Designed for the ``RelationPrediction`` labelling form.

    References
    ----------
    Demir et al., *A Shallow Neural Model for Relation Prediction*,
    ISWC 2021.  https://arxiv.org/abs/2101.09090
    """

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
        """Score a batch of triples by looking up relation scores from ``forward_k_vs_all``.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 3)`` integer tensor
            ``[head_idx, relation_idx, tail_idx]``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size,)`` triple scores.
        """
        n, d = x.shape
        assert d == 3
        scores_for_all_relations = self.forward_k_vs_all(x[:, [0, 2]])
        return scores_for_all_relations[:, x[:, 1]].flatten()


class Pyke(BaseKGE):
    """Pyke: Physical Embedding Model for Knowledge Graphs.

    Scores a triple ``(h, r, t)`` based on the average pairwise distance
    between head-to-relation and relation-to-tail in embedding space::

        f(h, r, t) = margin - (||h - r||_2 + ||r - t||_2) / 2

    The model encodes geometric proximity between entities and the
    relations that connect them.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Pyke'
        self.dist_func = torch.nn.PairwiseDistance(p=2)
        self.margin = 1.0

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        """Score a batch of triples using the Pyke distance formula.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 3)`` integer tensor
            ``[head_idx, relation_idx, tail_idx]``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size,)`` triple scores.
        """
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


