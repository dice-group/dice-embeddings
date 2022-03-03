import torch

from .base_model import *


def octonion_mul(*, O_1, O_2):
    x0, x1, x2, x3, x4, x5, x6, x7 = O_1
    y0, y1, y2, y3, y4, y5, y6, y7 = O_2
    x = x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3 - x4 * y4 - x5 * y5 - x6 * y6 - x7 * y7
    e1 = x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2 + x4 * y5 - x5 * y4 - x6 * y7 + x7 * y6
    e2 = x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1 + x4 * y6 + x5 * y7 - x6 * y4 - x7 * y5
    e3 = x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0 + x4 * y7 - x5 * y6 + x6 * y5 - x7 * y4
    e4 = x0 * y4 - x1 * y5 - x2 * y6 - x3 * y7 + x4 * y0 + x5 * y1 + x6 * y2 + x7 * y3
    e5 = x0 * y5 + x1 * y4 - x2 * y7 + x3 * y6 - x4 * y1 + x5 * y0 - x6 * y3 + x7 * y2
    e6 = x0 * y6 + x1 * y7 + x2 * y4 - x3 * y5 - x4 * y2 + x5 * y3 + x6 * y0 - x7 * y1
    e7 = x0 * y7 - x1 * y6 + x2 * y5 + x3 * y4 - x4 * y3 - x5 * y2 + x6 * y1 + x7 * y0

    return x, e1, e2, e3, e4, e5, e6, e7


def octonion_mul_norm(*, O_1, O_2):
    x0, x1, x2, x3, x4, x5, x6, x7 = O_1
    y0, y1, y2, y3, y4, y5, y6, y7 = O_2

    # Normalize the relation to eliminate the scaling effect, may cause Nan due to floating point.
    denominator = torch.sqrt(y0 ** 2 + y1 ** 2 + y2 ** 2 + y3 ** 2 + y4 ** 2 + y5 ** 2 + y6 ** 2 + y7 ** 2)
    y0 = y0 / denominator
    y1 = y1 / denominator
    y2 = y2 / denominator
    y3 = y3 / denominator
    y4 = y4 / denominator
    y5 = y5 / denominator
    y6 = y6 / denominator
    y7 = y7 / denominator

    x = x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3 - x4 * y4 - x5 * y5 - x6 * y6 - x7 * y7
    e1 = x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2 + x4 * y5 - x5 * y4 - x6 * y7 + x7 * y6
    e2 = x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1 + x4 * y6 + x5 * y7 - x6 * y4 - x7 * y5
    e3 = x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0 + x4 * y7 - x5 * y6 + x6 * y5 - x7 * y4
    e4 = x0 * y4 - x1 * y5 - x2 * y6 - x3 * y7 + x4 * y0 + x5 * y1 + x6 * y2 + x7 * y3
    e5 = x0 * y5 + x1 * y4 - x2 * y7 + x3 * y6 - x4 * y1 + x5 * y0 - x6 * y3 + x7 * y2
    e6 = x0 * y6 + x1 * y7 + x2 * y4 - x3 * y5 - x4 * y2 + x5 * y3 + x6 * y0 - x7 * y1
    e7 = x0 * y7 - x1 * y6 + x2 * y5 + x3 * y4 - x4 * y3 - x5 * y2 + x6 * y1 + x7 * y0

    return x, e1, e2, e3, e4, e5, e6, e7


class OMult(BaseKGE):

    def __init__(self, args):
        super().__init__(learning_rate=args.learning_rate)
        self.name = 'OMult'
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.apply_unit_norm = args.apply_unit_norm
        # Octonion embeddings of entities
        self.emb_ent_e0 = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_ent_e1 = nn.Embedding(args.num_entities, args.embedding_dim)  # e1
        self.emb_ent_e2 = nn.Embedding(args.num_entities, args.embedding_dim)  # e2
        self.emb_ent_e3 = nn.Embedding(args.num_entities, args.embedding_dim)  # e3
        self.emb_ent_e4 = nn.Embedding(args.num_entities, args.embedding_dim)  # e3
        self.emb_ent_e5 = nn.Embedding(args.num_entities, args.embedding_dim)  # e4
        self.emb_ent_e6 = nn.Embedding(args.num_entities, args.embedding_dim)  # e6
        self.emb_ent_e7 = nn.Embedding(args.num_entities, args.embedding_dim)  # e7
        xavier_normal_(self.emb_ent_e0.weight.data), xavier_normal_(self.emb_ent_e1.weight.data)
        xavier_normal_(self.emb_ent_e2.weight.data), xavier_normal_(self.emb_ent_e3.weight.data)
        xavier_normal_(self.emb_ent_e4.weight.data), xavier_normal_(self.emb_ent_e5.weight.data)
        xavier_normal_(self.emb_ent_e6.weight.data), xavier_normal_(self.emb_ent_e7.weight.data)

        # Octonion embeddings of relations
        self.emb_rel_e0 = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        self.emb_rel_e1 = nn.Embedding(args.num_relations, args.embedding_dim)  # e1
        self.emb_rel_e2 = nn.Embedding(args.num_relations, args.embedding_dim)  # e2
        self.emb_rel_e3 = nn.Embedding(args.num_relations, args.embedding_dim)  # e3
        self.emb_rel_e4 = nn.Embedding(args.num_relations, args.embedding_dim)  # e4
        self.emb_rel_e5 = nn.Embedding(args.num_relations, args.embedding_dim)  # e5
        self.emb_rel_e6 = nn.Embedding(args.num_relations, args.embedding_dim)  # e6
        self.emb_rel_e7 = nn.Embedding(args.num_relations, args.embedding_dim)  # e7
        xavier_normal_(self.emb_rel_e0.weight.data), xavier_normal_(self.emb_rel_e1.weight.data)
        xavier_normal_(self.emb_rel_e2.weight.data), xavier_normal_(self.emb_rel_e3.weight.data)
        xavier_normal_(self.emb_rel_e4.weight.data), xavier_normal_(self.emb_rel_e5.weight.data)
        xavier_normal_(self.emb_rel_e6.weight.data), xavier_normal_(self.emb_rel_e7.weight.data)

        # Dropouts for octonion embeddings of subject entities.
        self.input_dp_ent_e0 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e1 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e2 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e3 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e4 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e5 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e6 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e7 = torch.nn.Dropout(args.input_dropout_rate)
        # Dropouts for octonion embeddings of relations.
        self.input_dp_rel_e0 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e1 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e2 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e3 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e4 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e5 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e6 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e7 = torch.nn.Dropout(args.input_dropout_rate)
        # Dropouts for octonion embeddings obtained from octonion multiplication.
        self.hidden_dp_e0 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e1 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e2 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e3 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e4 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e5 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e6 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e7 = torch.nn.Dropout(args.hidden_dropout_rate)
        # Batch normalization for octonion embeddings of subject entities.
        self.bn_ent_e0 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_e1 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_e2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_e3 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_e4 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_e5 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_e6 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_e7 = torch.nn.BatchNorm1d(args.embedding_dim)
        # Batch normalization for octonion embeddings of relations.
        self.bn_rel_e0 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_e1 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_e2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_e3 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_e4 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_e5 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_e6 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_e7 = torch.nn.BatchNorm1d(args.embedding_dim)


        # Batch normalization for octonion embeddings of relations.
        self.bn_hidden_e0 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_e1 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_e2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_e3 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_e4 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_e5 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_e6 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_hidden_e7 = torch.nn.BatchNorm1d(args.embedding_dim)

    def get_embeddings(self):
        entity_emb = torch.cat((
            self.emb_ent_e0.weight.data, self.emb_ent_e1.weight.data,
            self.emb_ent_e2.weight.data, self.emb_ent_e3.weight.data,
            self.emb_ent_e4.weight.data, self.emb_ent_e5.weight.data,
            self.emb_ent_e6.weight.data, self.emb_ent_e7.weight.data), 1)
        rel_emb = torch.cat((
            self.emb_rel_e0.weight.data, self.emb_rel_e1.weight.data,
            self.emb_rel_e2.weight.data, self.emb_rel_e3.weight.data,
            self.emb_rel_e4.weight.data, self.emb_rel_e5.weight.data,
            self.emb_rel_e6.weight.data, self.emb_rel_e7.weight.data), 1)

        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

    def forward_k_vs_all(self, x):
        """
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
            [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
            Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]

        # (1)
        # (1.1) Octonion embeddings of head entities
        emb_head_e0 = self.emb_ent_e0(e1_idx)
        emb_head_e1 = self.emb_ent_e1(e1_idx)
        emb_head_e2 = self.emb_ent_e2(e1_idx)
        emb_head_e3 = self.emb_ent_e3(e1_idx)
        emb_head_e4 = self.emb_ent_e4(e1_idx)
        emb_head_e5 = self.emb_ent_e5(e1_idx)
        emb_head_e6 = self.emb_ent_e6(e1_idx)
        emb_head_e7 = self.emb_ent_e7(e1_idx)
        # (1.2) Octonion embeddings of relations
        emb_rel_e0 = self.emb_rel_e0(rel_idx)
        emb_rel_e1 = self.emb_rel_e1(rel_idx)
        emb_rel_e2 = self.emb_rel_e2(rel_idx)
        emb_rel_e3 = self.emb_rel_e3(rel_idx)
        emb_rel_e4 = self.emb_rel_e4(rel_idx)
        emb_rel_e5 = self.emb_rel_e5(rel_idx)
        emb_rel_e6 = self.emb_rel_e6(rel_idx)
        emb_rel_e7 = self.emb_rel_e7(rel_idx)

        if self.apply_unit_norm:
            # (2) Octonion  multiplication of (1.1) and unit normalized (1.2).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul_norm(
                O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                     emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                     emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
            # (3) Inner product of (2) with ALL entities.
            e0_score = torch.mm(e0, self.emb_ent_e0.weight.transpose(1, 0))
            e1_score = torch.mm(e1, self.emb_ent_e1.weight.transpose(1, 0))
            e2_score = torch.mm(e2, self.emb_ent_e2.weight.transpose(1, 0))
            e3_score = torch.mm(e3, self.emb_ent_e3.weight.transpose(1, 0))
            e4_score = torch.mm(e4, self.emb_ent_e4.weight.transpose(1, 0))
            e5_score = torch.mm(e5, self.emb_ent_e5.weight.transpose(1, 0))
            e6_score = torch.mm(e6, self.emb_ent_e6.weight.transpose(1, 0))
            e7_score = torch.mm(e7, self.emb_ent_e7.weight.transpose(1, 0))
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2) relations.
            # (2.2.) Apply octonion  multiplication of (1.1) and (2.1).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
                O_1=(self.input_dp_ent_e0(self.bn_ent_e0(emb_head_e0)),
                     self.input_dp_ent_e1(self.bn_ent_e1(emb_head_e1)),
                     self.input_dp_ent_e2(self.bn_ent_e2(emb_head_e2)),
                     self.input_dp_ent_e3(self.bn_ent_e3(emb_head_e3)),
                     self.input_dp_ent_e4(self.bn_ent_e4(emb_head_e4)),
                     self.input_dp_ent_e5(self.bn_ent_e5(emb_head_e5)),
                     self.input_dp_ent_e6(self.bn_ent_e6(emb_head_e6)),
                     self.input_dp_ent_e7(self.bn_ent_e7(emb_head_e7))),
                O_2=(self.input_dp_rel_e0(self.bn_rel_e0(emb_rel_e0)),
                     self.input_dp_rel_e1(self.bn_rel_e1(emb_rel_e1)),
                     self.input_dp_rel_e2(self.bn_rel_e2(emb_rel_e2)),
                     self.input_dp_rel_e3(self.bn_rel_e3(emb_rel_e3)),
                     self.input_dp_rel_e4(self.bn_rel_e4(emb_rel_e4)),
                     self.input_dp_rel_e5(self.bn_rel_e5(emb_rel_e5)),
                     self.input_dp_rel_e6(self.bn_rel_e6(emb_rel_e6)),
                     self.input_dp_rel_e7(self.bn_rel_e7(emb_rel_e7))))
            # (3)
            # (3.1) Dropout on (2)-result of octonion multiplication.
            # (3.2) Apply BN + DP on ALL entities.
            # (3.3) Inner product
            e0_score = torch.mm(self.hidden_dp_e0(self.bn_hidden_e0(e0)), self.emb_ent_e0.weight.transpose(1, 0))

            e1_score = torch.mm(self.hidden_dp_e1(self.bn_hidden_e1(e1)), self.emb_ent_e1.weight.transpose(1, 0))
            e2_score = torch.mm(self.hidden_dp_e2(self.bn_hidden_e2(e2)), self.emb_ent_e2.weight.transpose(1, 0))
            e3_score = torch.mm(self.hidden_dp_e3(self.bn_hidden_e3(e3)),
                                self.emb_ent_e3.weight.transpose(1, 0))
            e4_score = torch.mm(self.hidden_dp_e4(self.bn_hidden_e4(e4)), self.emb_ent_e4.weight.transpose(1, 0))
            e5_score = torch.mm(self.hidden_dp_e5(self.bn_hidden_e5(e5)), self.emb_ent_e5.weight.transpose(1, 0))
            e6_score = torch.mm(self.hidden_dp_e6(self.bn_hidden_e6(e6)), self.emb_ent_e6.weight.transpose(1, 0))
            e7_score = torch.mm(self.hidden_dp_e7(self.bn_hidden_e7(e7)), self.emb_ent_e7.weight.transpose(1, 0))

        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1],x[:, 2]
        # (1)
        # (1.1) Octonion embeddings of head entities
        emb_head_e0 = self.emb_ent_e0(e1_idx)
        emb_head_e1 = self.emb_ent_e1(e1_idx)
        emb_head_e2 = self.emb_ent_e2(e1_idx)
        emb_head_e3 = self.emb_ent_e3(e1_idx)
        emb_head_e4 = self.emb_ent_e4(e1_idx)
        emb_head_e5 = self.emb_ent_e5(e1_idx)
        emb_head_e6 = self.emb_ent_e6(e1_idx)
        emb_head_e7 = self.emb_ent_e7(e1_idx)
        # (1.2) Octonion embeddings of relations
        emb_rel_e0 = self.emb_rel_e0(rel_idx)
        emb_rel_e1 = self.emb_rel_e1(rel_idx)
        emb_rel_e2 = self.emb_rel_e2(rel_idx)
        emb_rel_e3 = self.emb_rel_e3(rel_idx)
        emb_rel_e4 = self.emb_rel_e4(rel_idx)
        emb_rel_e5 = self.emb_rel_e5(rel_idx)
        emb_rel_e6 = self.emb_rel_e6(rel_idx)
        emb_rel_e7 = self.emb_rel_e7(rel_idx)

        # (1.3) Octonion embeddings of tail entities
        emb_tail_e0 = self.emb_ent_e0(e2_idx)
        emb_tail_e1 = self.emb_ent_e1(e2_idx)
        emb_tail_e2 = self.emb_ent_e2(e2_idx)
        emb_tail_e3 = self.emb_ent_e3(e2_idx)
        emb_tail_e4 = self.emb_ent_e4(e2_idx)
        emb_tail_e5 = self.emb_ent_e5(e2_idx)
        emb_tail_e6 = self.emb_ent_e6(e2_idx)
        emb_tail_e7 = self.emb_ent_e7(e2_idx)

        if self.apply_unit_norm:
            # (2) Octonion  multiplication of (1.1) and unit normalized (1.2).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul_norm(
                O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                     emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                     emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
            # (3) Inner product of (2) with ALL entities.
            e0_score = (self.hidden_dp_e0(e0) * emb_tail_e0).sum(dim=1)
            e1_score = (self.hidden_dp_e1(e1) * emb_tail_e1).sum(dim=1)
            e2_score = (self.hidden_dp_e2(e2) * emb_tail_e2).sum(dim=1)
            e3_score = (self.hidden_dp_e3(e3) * emb_tail_e3).sum(dim=1)
            e4_score = (self.hidden_dp_e4(e4) * emb_tail_e4).sum(dim=1)
            e5_score = (self.hidden_dp_e5(e5) * emb_tail_e5).sum(dim=1)
            e6_score = (self.hidden_dp_e6(e6) * emb_tail_e6).sum(dim=1)
            e7_score = (self.hidden_dp_e7(e7) * emb_tail_e7).sum(dim=1)
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2) relations.
            # (2.2.) Apply octonion  multiplication of (1.1) and (2.1).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
                O_1=(self.input_dp_ent_e0(self.bn_ent_e0(emb_head_e0)),
                     self.input_dp_ent_e1(self.bn_ent_e1(emb_head_e1)),
                     self.input_dp_ent_e2(self.bn_ent_e2(emb_head_e2)),
                     self.input_dp_ent_e3(self.bn_ent_e3(emb_head_e3)),
                     self.input_dp_ent_e4(self.bn_ent_e4(emb_head_e4)),
                     self.input_dp_ent_e5(self.bn_ent_e5(emb_head_e5)),
                     self.input_dp_ent_e6(self.bn_ent_e6(emb_head_e6)),
                     self.input_dp_ent_e7(self.bn_ent_e7(emb_head_e7))),
                O_2=(self.input_dp_rel_e0(self.bn_rel_e0(emb_rel_e0)),
                     self.input_dp_rel_e1(self.bn_rel_e1(emb_rel_e1)),
                     self.input_dp_rel_e2(self.bn_rel_e2(emb_rel_e2)),
                     self.input_dp_rel_e3(self.bn_rel_e3(emb_rel_e3)),
                     self.input_dp_rel_e4(self.bn_rel_e4(emb_rel_e4)),
                     self.input_dp_rel_e5(self.bn_rel_e5(emb_rel_e5)),
                     self.input_dp_rel_e6(self.bn_rel_e6(emb_rel_e6)),
                     self.input_dp_rel_e7(self.bn_rel_e7(emb_rel_e7))))
            # (3)
            # (3.1) Dropout on (2)-result of octonion multiplication.
            # (3.2) Apply BN + DP on ALL entities.
            # (3.3) Inner product
            e0_score = (self.hidden_dp_e0(e0) * emb_tail_e0).sum(dim=1)
            e1_score = (self.hidden_dp_e1(e1) * emb_tail_e1).sum(dim=1)
            e2_score = (self.hidden_dp_e2(e2) * emb_tail_e2).sum(dim=1)
            e3_score = (self.hidden_dp_e3(e3) * emb_tail_e3).sum(dim=1)
            e4_score = (self.hidden_dp_e4(e4) * emb_tail_e4).sum(dim=1)
            e5_score = (self.hidden_dp_e5(e5) * emb_tail_e5).sum(dim=1)
            e6_score = (self.hidden_dp_e6(e6) * emb_tail_e6).sum(dim=1)
            e7_score = (self.hidden_dp_e7(e7) * emb_tail_e7).sum(dim=1)

        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score


class ConvO(BaseKGE):
    def __init__(self, args):
        super().__init__(learning_rate=args.learning_rate)
        self.name = 'ConvO'
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.apply_unit_norm = args.apply_unit_norm
        self.embedding_dim = args.embedding_dim  # for reshaping in the residual.
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations

        # Octonion embeddings of entities
        self.emb_ent_e0 = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_e1 = nn.Embedding(self.num_entities, self.embedding_dim)  # e1
        self.emb_ent_e2 = nn.Embedding(self.num_entities, self.embedding_dim)  # e2
        self.emb_ent_e3 = nn.Embedding(self.num_entities, self.embedding_dim)  # e3
        self.emb_ent_e4 = nn.Embedding(self.num_entities, self.embedding_dim)  # e3
        self.emb_ent_e5 = nn.Embedding(self.num_entities, self.embedding_dim)  # e4
        self.emb_ent_e6 = nn.Embedding(self.num_entities, self.embedding_dim)  # e6
        self.emb_ent_e7 = nn.Embedding(self.num_entities, self.embedding_dim)  # e7
        xavier_normal_(self.emb_ent_e0.weight.data), xavier_normal_(self.emb_ent_e1.weight.data)
        xavier_normal_(self.emb_ent_e2.weight.data), xavier_normal_(self.emb_ent_e3.weight.data)
        xavier_normal_(self.emb_ent_e4.weight.data), xavier_normal_(self.emb_ent_e5.weight.data)
        xavier_normal_(self.emb_ent_e6.weight.data), xavier_normal_(self.emb_ent_e7.weight.data)

        # Octonion embeddings of relations
        self.emb_rel_e0 = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_e1 = nn.Embedding(self.num_relations, self.embedding_dim)  # e1
        self.emb_rel_e2 = nn.Embedding(self.num_relations, self.embedding_dim)  # e2
        self.emb_rel_e3 = nn.Embedding(self.num_relations, self.embedding_dim)  # e3
        self.emb_rel_e4 = nn.Embedding(self.num_relations, self.embedding_dim)  # e4
        self.emb_rel_e5 = nn.Embedding(self.num_relations, self.embedding_dim)  # e5
        self.emb_rel_e6 = nn.Embedding(self.num_relations, self.embedding_dim)  # e6
        self.emb_rel_e7 = nn.Embedding(self.num_relations, self.embedding_dim)  # e7
        xavier_normal_(self.emb_rel_e0.weight.data), xavier_normal_(self.emb_rel_e1.weight.data)
        xavier_normal_(self.emb_rel_e2.weight.data), xavier_normal_(self.emb_rel_e3.weight.data)
        xavier_normal_(self.emb_rel_e4.weight.data), xavier_normal_(self.emb_rel_e5.weight.data)
        xavier_normal_(self.emb_rel_e6.weight.data), xavier_normal_(self.emb_rel_e7.weight.data)

        # Dropouts for octonion embeddings of entities.
        self.input_dp_ent_e0 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e1 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e2 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e3 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e4 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e5 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e6 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_e7 = torch.nn.Dropout(args.input_dropout_rate)

        # Dropouts for octonion embeddings of relations.
        self.input_dp_rel_e0 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e1 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e2 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e3 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e4 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e5 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e6 = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_e7 = torch.nn.Dropout(args.input_dropout_rate)

        # Dropouts for octonion embeddings obtained from octonion multiplication.
        self.hidden_dp_e0 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e1 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e2 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e3 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e4 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e5 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e6 = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_e7 = torch.nn.Dropout(args.hidden_dropout_rate)

        # Batch normalization for octonion embeddings of ALL entities.
        self.bn_ent_e0 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_e1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_e2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_e3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_e4 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_e5 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_e6 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_e7 = torch.nn.BatchNorm1d(self.embedding_dim)
        # Batch normalization for octonion embeddings of relations.
        self.bn_rel_e0 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_e1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_e2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_e3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_e4 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_e5 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_e6 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_e7 = torch.nn.BatchNorm1d(self.embedding_dim)

        # Convolution
        self.kernel_size = args.kernel_size
        self.num_of_output_channels = args.num_of_output_channels

        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = args.embedding_dim * 16 * self.num_of_output_channels  # 8 because of 8 real values in 2 quaternions
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim * 8)  # Hard compression.
        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(args.embedding_dim * 8)

        # Convolution Dropout
        self.feature_map_dropout = torch.nn.Dropout2d(args.feature_map_dropout_rate)

    def get_embeddings(self):
        entity_emb = torch.cat((
            self.emb_ent_e0.weight.data, self.emb_ent_e1.weight.data,
            self.emb_ent_e2.weight.data, self.emb_ent_e3.weight.data,
            self.emb_ent_e4.weight.data, self.emb_ent_e5.weight.data,
            self.emb_ent_e6.weight.data, self.emb_ent_e7.weight.data), 1)
        rel_emb = torch.cat((
            self.emb_rel_e0.weight.data, self.emb_rel_e1.weight.data,
            self.emb_rel_e2.weight.data, self.emb_rel_e3.weight.data,
            self.emb_rel_e4.weight.data, self.emb_rel_e5.weight.data,
            self.emb_rel_e6.weight.data, self.emb_rel_e7.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

    def residual_convolution(self, O_1, O_2):
        emb_ent_e0, emb_ent_e1, emb_ent_e2, emb_ent_e3, emb_ent_e4, emb_ent_e5, emb_ent_e6, emb_ent_e7 = O_1
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = O_2
        x = torch.cat([emb_ent_e0.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_e1.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_e2.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_e3.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_e4.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_e5.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_e6.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_e7.view(-1, 1, 1, self.embedding_dim),  # entities
                       emb_rel_e0.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_e1.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_e2.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_e3.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_e4.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_e5.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_e6.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_e7.view(-1, 1, 1, self.embedding_dim), ], 2)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = self.fc1(x)
        x = self.bn_conv2(x)
        x = F.relu(x)
        return torch.chunk(x, 8, dim=1)

    def forward_k_vs_all(self, x:torch.Tensor):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]

        # (1)
        # (1.1) Octonion embeddings of head entities
        emb_head_e0 = self.emb_ent_e0(e1_idx)
        emb_head_e1 = self.emb_ent_e1(e1_idx)
        emb_head_e2 = self.emb_ent_e2(e1_idx)
        emb_head_e3 = self.emb_ent_e3(e1_idx)
        emb_head_e4 = self.emb_ent_e4(e1_idx)
        emb_head_e5 = self.emb_ent_e5(e1_idx)
        emb_head_e6 = self.emb_ent_e6(e1_idx)
        emb_head_e7 = self.emb_ent_e7(e1_idx)
        # (1.2) Octonion embeddings of relations
        emb_rel_e0 = self.emb_rel_e0(rel_idx)
        emb_rel_e1 = self.emb_rel_e1(rel_idx)
        emb_rel_e2 = self.emb_rel_e2(rel_idx)
        emb_rel_e3 = self.emb_rel_e3(rel_idx)
        emb_rel_e4 = self.emb_rel_e4(rel_idx)
        emb_rel_e5 = self.emb_rel_e5(rel_idx)
        emb_rel_e6 = self.emb_rel_e6(rel_idx)
        emb_rel_e7 = self.emb_rel_e7(rel_idx)
        # (2) Apply convolution operation on (1.1) and (1.2).
        O_3 = self.residual_convolution(O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                                             emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                                        O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                                             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
        conv_e0, conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7 = O_3

        if self.apply_unit_norm:
            # (3) Octonion multiplication of (1.1) and unit normalized (1.2).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul_norm(
                O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                     emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                     emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Inner product of (4.1) with ALL entities.
            e0_score = torch.mm(conv_e0 * e0, self.emb_ent_e0.weight.transpose(1, 0))
            e1_score = torch.mm(conv_e1 * e1, self.emb_ent_e1.weight.transpose(1, 0))
            e2_score = torch.mm(conv_e2 * e2, self.emb_ent_e2.weight.transpose(1, 0))
            e3_score = torch.mm(conv_e3 * e3, self.emb_ent_e3.weight.transpose(1, 0))
            e4_score = torch.mm(conv_e4 * e4, self.emb_ent_e4.weight.transpose(1, 0))
            e5_score = torch.mm(conv_e5 * e5, self.emb_ent_e5.weight.transpose(1, 0))
            e6_score = torch.mm(conv_e6 * e6, self.emb_ent_e6.weight.transpose(1, 0))
            e7_score = torch.mm(conv_e7 * e7, self.emb_ent_e7.weight.transpose(1, 0))
        else:
            # (3)
            # (3.1) Apply BN + Dropout on (1.2)-relations.
            # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
                O_1=(self.input_dp_ent_e0(self.bn_ent_e0(emb_head_e0)),
                     self.input_dp_ent_e1(self.bn_ent_e1(emb_head_e1)),
                     self.input_dp_ent_e2(self.bn_ent_e2(emb_head_e2)),
                     self.input_dp_ent_e3(self.bn_ent_e3(emb_head_e3)),
                     self.input_dp_ent_e4(self.bn_ent_e0(emb_head_e4)),
                     self.input_dp_ent_e5(self.bn_ent_e0(emb_head_e5)),
                     self.input_dp_ent_e6(self.bn_ent_e0(emb_head_e6)),
                     self.input_dp_ent_e7(self.bn_ent_e0(emb_head_e7))),
                O_2=(self.input_dp_rel_e0(self.bn_rel_e0(emb_rel_e0)),
                     self.input_dp_rel_e1(self.bn_rel_e1(emb_rel_e1)),
                     self.input_dp_rel_e2(self.bn_rel_e2(emb_rel_e2)),
                     self.input_dp_rel_e3(self.bn_rel_e3(emb_rel_e3)),
                     self.input_dp_rel_e4(self.bn_rel_e4(emb_rel_e4)),
                     self.input_dp_rel_e5(self.bn_rel_e5(emb_rel_e5)),
                     self.input_dp_rel_e6(self.bn_rel_e6(emb_rel_e6)),
                     self.input_dp_rel_e7(self.bn_rel_e7(emb_rel_e7))))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Dropout on (4.1).
            # (4.3) Apply BN + DP on ALL entities.
            # (4.4) Inner product
            e0_score = torch.mm(self.hidden_dp_e0(conv_e0 * e0),
                                self.emb_ent_e0.weight.transpose(1, 0))
            e1_score = torch.mm(self.hidden_dp_e1(conv_e1 * e1), self.emb_ent_e1.weight.transpose(1, 0))
            e2_score = torch.mm(self.hidden_dp_e2(conv_e2 * e2), self.emb_ent_e2.weight.transpose(1, 0))
            e3_score = torch.mm(self.hidden_dp_e3(conv_e3 * e3), self.emb_ent_e3.weight.transpose(1, 0))
            e4_score = torch.mm(self.hidden_dp_e4(conv_e4 * e4), self.emb_ent_e4.weight.transpose(1, 0))
            e5_score = torch.mm(self.hidden_dp_e5(conv_e5 * e5), self.emb_ent_e5.weight.transpose(1, 0))
            e6_score = torch.mm(self.hidden_dp_e6(conv_e6 * e6), self.emb_ent_e6.weight.transpose(1, 0))
            e7_score = torch.mm(self.hidden_dp_e7(conv_e7 * e7), self.emb_ent_e7.weight.transpose(1, 0))
        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1],x[:, 2]
        # (1)
        # (1.1) Octonion embeddings of head entities
        emb_head_e0 = self.emb_ent_e0(e1_idx)
        emb_head_e1 = self.emb_ent_e1(e1_idx)
        emb_head_e2 = self.emb_ent_e2(e1_idx)
        emb_head_e3 = self.emb_ent_e3(e1_idx)
        emb_head_e4 = self.emb_ent_e4(e1_idx)
        emb_head_e5 = self.emb_ent_e5(e1_idx)
        emb_head_e6 = self.emb_ent_e6(e1_idx)
        emb_head_e7 = self.emb_ent_e7(e1_idx)
        # (1.2) Octonion embeddings of relations
        emb_rel_e0 = self.emb_rel_e0(rel_idx)
        emb_rel_e1 = self.emb_rel_e1(rel_idx)
        emb_rel_e2 = self.emb_rel_e2(rel_idx)
        emb_rel_e3 = self.emb_rel_e3(rel_idx)
        emb_rel_e4 = self.emb_rel_e4(rel_idx)
        emb_rel_e5 = self.emb_rel_e5(rel_idx)
        emb_rel_e6 = self.emb_rel_e6(rel_idx)
        emb_rel_e7 = self.emb_rel_e7(rel_idx)

        # (1.3) Octonion embeddings of tail entities
        emb_tail_e0 = self.emb_ent_e0(e2_idx)
        emb_tail_e1 = self.emb_ent_e1(e2_idx)
        emb_tail_e2 = self.emb_ent_e2(e2_idx)
        emb_tail_e3 = self.emb_ent_e3(e2_idx)
        emb_tail_e4 = self.emb_ent_e4(e2_idx)
        emb_tail_e5 = self.emb_ent_e5(e2_idx)
        emb_tail_e6 = self.emb_ent_e6(e2_idx)
        emb_tail_e7 = self.emb_ent_e7(e2_idx)

        # (2) Apply convolution operation on (1.1) and (1.2).
        O_3 = self.residual_convolution(O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                                             emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                                        O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                                             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
        conv_e0, conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7 = O_3

        if self.apply_unit_norm:
            # (3) Octonion multiplication of (1.1) and unit normalized (1.2).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul_norm(
                O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                     emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                     emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Inner product of (4.1) with ALL entities.
            # (3) Inner product of (2) with ALL entities.
            e0_score = (conv_e0 * e0 * emb_tail_e0).sum(dim=1)
            e1_score = (conv_e1 * e1 * emb_tail_e1).sum(dim=1)
            e2_score = (conv_e2 * e2 * emb_tail_e2).sum(dim=1)
            e3_score = (conv_e3 * e3 * emb_tail_e3).sum(dim=1)
            e4_score = (conv_e4 * e4 * emb_tail_e4).sum(dim=1)
            e5_score = (conv_e5 * e5 * emb_tail_e5).sum(dim=1)
            e6_score = (conv_e6 * e6 * emb_tail_e6).sum(dim=1)
            e7_score = (conv_e7 * e7 * emb_tail_e7).sum(dim=1)

        else:
            # (3)
            # (3.1) Apply BN + Dropout on (1.2)-relations.
            # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
            e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
                O_1=(self.input_dp_ent_e0(self.bn_ent_e0(emb_head_e0)),
                     self.input_dp_ent_e1(self.bn_ent_e1(emb_head_e1)),
                     self.input_dp_ent_e2(self.bn_ent_e2(emb_head_e2)),
                     self.input_dp_ent_e3(self.bn_ent_e3(emb_head_e3)),
                     self.input_dp_ent_e4(self.bn_ent_e0(emb_head_e4)),
                     self.input_dp_ent_e5(self.bn_ent_e0(emb_head_e5)),
                     self.input_dp_ent_e6(self.bn_ent_e0(emb_head_e6)),
                     self.input_dp_ent_e7(self.bn_ent_e0(emb_head_e7))),
                O_2=(self.input_dp_rel_e0(self.bn_rel_e0(emb_rel_e0)),
                     self.input_dp_rel_e1(self.bn_rel_e1(emb_rel_e1)),
                     self.input_dp_rel_e2(self.bn_rel_e2(emb_rel_e2)),
                     self.input_dp_rel_e3(self.bn_rel_e3(emb_rel_e3)),
                     self.input_dp_rel_e4(self.bn_rel_e4(emb_rel_e4)),
                     self.input_dp_rel_e5(self.bn_rel_e5(emb_rel_e5)),
                     self.input_dp_rel_e6(self.bn_rel_e6(emb_rel_e6)),
                     self.input_dp_rel_e7(self.bn_rel_e7(emb_rel_e7))))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Dropout on (4.1).
            # (4.3) Apply BN + DP on ALL entities.
            # (4.4) Inner product
            e0_score = (self.hidden_dp_e0(conv_e0 * e0) * emb_tail_e0).sum(dim=1)
            e1_score = (self.hidden_dp_e1(conv_e1 * e1) * emb_tail_e1).sum(dim=1)
            e2_score = (self.hidden_dp_e2(conv_e2 * e2) * emb_tail_e2).sum(dim=1)
            e3_score = (self.hidden_dp_e3(conv_e3 * e3) * emb_tail_e3).sum(dim=1)
            e4_score = (self.hidden_dp_e4(conv_e4 * e4) * emb_tail_e4).sum(dim=1)
            e5_score = (self.hidden_dp_e5(conv_e5 * e5) * emb_tail_e5).sum(dim=1)
            e6_score = (self.hidden_dp_e6(conv_e6 * e6) * emb_tail_e6).sum(dim=1)
            e7_score = (self.hidden_dp_e7(conv_e7 * e7) * emb_tail_e7).sum(dim=1)

        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score
