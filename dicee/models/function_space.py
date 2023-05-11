from .base_model import BaseKGE
from .static_funcs import quaternion_mul
from ..types import Tuple, Union, torch, np


class FMult(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'FMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = 50
        # self.gamma = torch.rand(self.k, self.num_sample) [0,1) uniform=> worse results
        self.gamma = torch.randn(self.k, self.num_sample)  # N(0,1)

    def compute_func(self, weights: torch.FloatTensor, x) -> torch.FloatTensor:
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.reshape(n, self.k, self.k)
        w2 = w2.reshape(n, self.k, self.k)
        # (2) Forward Pass
        out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
        out2 = w2 @ out1
        return out2  # no non-linearity => better results

    def chain_func(self, weights, x: torch.FloatTensor):
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.reshape(n, self.k, self.k)
        w2 = w2.reshape(n, self.k, self.k)
        # (2) Perform the forward pass
        out1 = torch.tanh(torch.bmm(w1, x))
        out2 = torch.bmm(w2, out1)
        return out2

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        # (2) Compute NNs on \Gamma
        # Logits via FDistMult...
        # h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # r_x = self.compute_func(rel_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # out = h_x * r_x * t_x  # batch, \mathbb{R}^k, |gamma|
        # (2) Compute NNs on \Gamma
        h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.chain_func(weights=rel_ent_emb, x=h_x)  # batch, \mathbb{R}^k, |\Gamma|
        # (3) Compute |\Gamma| predictions
        out = torch.sum(r_h_x * t_x, dim=1)  # batch, |gamma| #
        # (4) Average (3) over \Gamma
        out = torch.mean(out, dim=1)  # batch
        return out
