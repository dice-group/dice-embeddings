from .base_model import BaseKGE
from typing import Tuple, Optional
import torch
import numpy as np
from .transformers import Transformer
import argparse
from dataclasses import dataclass

@dataclass
class TapireConfig:
    """Configuration for Tapire model
    
    Attributes:
        embedding_dim: Dimension of entity and relation embeddings
        num_entities: Number of entities in the knowledge graph
        num_relations: Number of relations in the knowledge graph
        token_size: Number of tokens to split embeddings into (default: 2)
        inner_embedding_size: Size of inner embeddings for each token (default: 1)
        n_layer: Number of transformer layers (default: 2)
        n_head: Number of attention heads (default: 1)
        dropout: Dropout rate (default: 0.0)
        bias: Whether to use bias in linear layers (default: False)
    """
    embedding_dim: int
    num_entities: int
    num_relations: int
    token_size: int = 2
    inner_embedding_size: int = 1
    n_layer: int = 2
    n_head: int = 1
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self):
        # Validate configuration
        assert self.embedding_dim % self.token_size == 0, \
            f"embedding_dim ({self.embedding_dim}) must be divisible by token_size ({self.token_size})"
        
        # Calculate dimensions for transformer
        self.in_features = self.embedding_dim * 2  # Combined head and relation embeddings
        self.out_features = self.num_entities
        self.n_embd = self.inner_embedding_size

class Tapire(BaseKGE):
    """TrAnsformerPAIRE
    
    (1) A batch of tuples [(h,r),...,(h,r)]_b
    (2) Retrieve embeddings emb_h, emb_r
    (3) Concat emb_h and emb_r horizontally into emb_hr :(batch_size, 2d)
    (4) Reshape emb_hr into (batch_size, token_size, inner_embedding_size)
    (5) Apply transformer operation with a single classifier to compute logits for all entities


    Potential operations (3)
    We may want to explore shapes
     ***(batch_size,2d,1)***
     ***(batch_size,d,2)***
     ***(batch_size,d//2,4)***
     
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'Tapire'
        
        # Create configuration
        self.config = TapireConfig(
            embedding_dim=self.embedding_dim,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            token_size=args.get('token_size', 2),
            inner_embedding_size=args.get('inner_embedding_size', 4),
            n_layer=args.get('n_layer', 2),
            n_head=args.get('n_head', 1),
            dropout=args.get('dropout', 0.0),
            bias=args.get('bias', False)
        )
        
        # Create transformer model
        transformer_config = argparse.Namespace(
            dropout=self.config.dropout,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            bias=self.config.bias,
            in_features=self.config.in_features,
            out_features=self.config.out_features
        )
        self.transformer_model = Transformer(config=transformer_config)


    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor):
        """

        Parameters
        ----------
        emb_h : n by d tensor
        emb_r : n by d tensor

        Returns
        -------

        """
        
        return self.transformer_model(emb_h,emb_r)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(emb_h=emb_head, emb_r=emb_rel)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        # (b,d),     (b,d)
        raise NotImplemented
        self.k_vs_all_score(self.get_head_relation_representation(x))
        # (b, d)
        hr = torch.einsum('bd, bd -> bd', emb_head_real, emb_rel_real)
        # (b, k, d)
        t = self.entity_embeddings(target_entity_idx)
        return torch.einsum('bd, bkd -> bk', hr, t)


    def score(self, h, r, t):
        self.transformer_model(h,r)
        return (self.transformer_model(h,r) * t).sum(dim=1)


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
        emb_h : n by d tensor
        emb_r : n by d tensor
        emb_E : num_entity by d tensor

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
