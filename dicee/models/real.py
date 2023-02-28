import torch
from .base_model import *
import numpy as np
from math import sqrt
from .static_funcs import quaternion_mul


class DistMult(BaseKGE):
    """
    Embedding Entities and Relations for Learning and Inference in Knowledge Bases
    https://arxiv.org/abs/1412.6575"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'DistMult'
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Compute the score
        return (self.hidden_dropout(self.hidden_normalizer(head_ent_emb * rel_ent_emb)) * tail_ent_emb).sum(dim=1)

    def forward_k_vs_all(self, x: torch.LongTensor):
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_head_real * emb_rel_real)),
                        self.entity_embeddings.weight.transpose(1, 0))

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        hr = self.hidden_dropout(self.hidden_normalizer(emb_head_real * emb_rel_real)).unsqueeze(1)
        t = self.entity_embeddings(target_entity_idx).transpose(1, 2)
        return torch.bmm(hr, t).squeeze(1)


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
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # Original d:=|| s+p - t||_2 \approx 0 distance, if true
        # if d =0 sigma(5-0) => 1
        # if d =5 sigma(5-5) => 0.5
        # Update: sigmoid( \gamma - d)
        distance = self.margin - torch.nn.functional.pairwise_distance(head_ent_emb + rel_ent_emb, tail_ent_emb,
                                                                       p=self._norm)
        return distance

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
        # Fixed
        shallom_width = int(2 * self.embedding_dim)
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data)
        self.shallom = nn.Sequential(nn.Dropout(self.input_dropout_rate),
                                     torch.nn.Linear(self.embedding_dim * 2, shallom_width),
                                     self.normalizer_class(shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(self.hidden_dropout_rate),
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
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def get_embeddings(self) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        return self.entity_embeddings.weight.data.data.detach(), None

    def loss_function(self, x: torch.FloatTensor, y=None) -> torch.FloatTensor:
        anchor, positive, negative = x
        return self.loss(anchor, positive, negative)

    def forward_sequence(self, x: torch.LongTensor):
        # (1) Anchor node Embedding: N, D
        anchor = self.entity_embeddings(x[:, 0])
        # (2) Positives and Negatives
        pos, neg = torch.hsplit(x[:, 1:], 2)
        # (3) Embeddings for Pos N, K, D
        pos_emb = self.entity_embeddings(pos)
        # (4) Embeddings for Negs N, K, D
        neg_emb = self.entity_embeddings(neg)
        # (5) Mean.
        # N, D
        mean_pos_emb = pos_emb.mean(dim=1)
        mean_neg_emb = neg_emb.mean(dim=1)
        return anchor, mean_pos_emb, mean_neg_emb

