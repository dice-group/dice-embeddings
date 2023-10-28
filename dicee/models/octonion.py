import torch
from .base_model import BaseKGE, IdentityClass


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
        super().__init__(args)
        self.name = 'OMult'

    @staticmethod
    def octonion_normalizer(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6,
                            emb_rel_e7):
        denominator = torch.sqrt(
            emb_rel_e0 ** 2 + emb_rel_e1 ** 2 + emb_rel_e2 ** 2 + emb_rel_e3 ** 2 + emb_rel_e4 ** 2
            + emb_rel_e5 ** 2 + emb_rel_e6 ** 2 + emb_rel_e7 ** 2)
        y0 = emb_rel_e0 / denominator
        y1 = emb_rel_e1 / denominator
        y2 = emb_rel_e2 / denominator
        y3 = emb_rel_e3 / denominator
        y4 = emb_rel_e4 / denominator
        y5 = emb_rel_e5 / denominator
        y6 = emb_rel_e6 / denominator
        y7 = emb_rel_e7 / denominator
        return y0, y1, y2, y3, y4, y5, y6, y7

    def score(self, head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor, tail_ent_emb: torch.FloatTensor):
        # (2) Split (1) into real and imaginary parts.
        emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7 = torch.hsplit(
            head_ent_emb, 8)
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = torch.hsplit(
            rel_ent_emb,
            8)
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            (emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
             emb_rel_e5, emb_rel_e6, emb_rel_e7) = self.octonion_normalizer(emb_rel_e0,
                                                                            emb_rel_e1, emb_rel_e2, emb_rel_e3,
                                                                            emb_rel_e4, emb_rel_e5, emb_rel_e6,
                                                                            emb_rel_e7)

        emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 = torch.hsplit(
            tail_ent_emb, 8)
        # (3) Octonion Multiplication
        e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
            O_1=(
                emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
            O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
        # (4)
        # (4.3) Inner product
        e0_score = (e0 * emb_tail_e0).sum(dim=1)
        e1_score = (e1 * emb_tail_e1).sum(dim=1)
        e2_score = (e2 * emb_tail_e2).sum(dim=1)
        e3_score = (e3 * emb_tail_e3).sum(dim=1)
        e4_score = (e4 * emb_tail_e4).sum(dim=1)
        e5_score = (e5 * emb_tail_e5).sum(dim=1)
        e6_score = (e6 * emb_tail_e6).sum(dim=1)
        e7_score = (e7 * emb_tail_e7).sum(dim=1)

        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score

    def k_vs_all_score(self, bpe_head_ent_emb, bpe_rel_ent_emb, E):

        # (2) Split (1) into real and imaginary parts.
        # (2) Split (1) into real and imaginary parts.
        emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7 = torch.hsplit(
            bpe_head_ent_emb, 8)
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = torch.hsplit(
            bpe_rel_ent_emb,
            8)
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            (emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7) = self.octonion_normalizer(emb_rel_e0, emb_rel_e1,
                                                                                        emb_rel_e2, emb_rel_e3,
                                                                                        emb_rel_e4, emb_rel_e5,
                                                                                        emb_rel_e6, emb_rel_e7)

        # (3)Apply octonion multiplication
        e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
            O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4,
                 emb_head_e5, emb_head_e6, emb_head_e7),
            O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
                 emb_rel_e5, emb_rel_e6, emb_rel_e7))

        # Prepare all entities.
        emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 = torch.hsplit(
            E, 8)
        emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 \
            = emb_tail_e0.transpose(1, 0), emb_tail_e1.transpose(1, 0), \
            emb_tail_e2.transpose(1, 0), emb_tail_e3.transpose(1, 0), \
            emb_tail_e4.transpose(1, 0), emb_tail_e5.transpose(1, 0), \
            emb_tail_e6.transpose(1, 0), emb_tail_e7.transpose(1, 0)

        # (4)
        # (4.4) Inner product
        e0_score = torch.mm(e0, emb_tail_e0)
        e1_score = torch.mm(e1, emb_tail_e1)
        e2_score = torch.mm(e2, emb_tail_e2)
        e3_score = torch.mm(e3, emb_tail_e3)
        e4_score = torch.mm(e4, emb_tail_e4)
        e5_score = torch.mm(e5, emb_tail_e5)
        e6_score = torch.mm(e6, emb_tail_e6)
        e7_score = torch.mm(e7, emb_tail_e7)
        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score

    def forward_k_vs_all(self, x):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        return self.k_vs_all_score(head_ent_emb, rel_ent_emb, self.entity_embeddings.weight)


class ConvO(BaseKGE):
    def __init__(self, args: dict):
        super().__init__(args=args)
        self.name = 'ConvO'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim)  # Hard compression.
        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.norm_fc1 = self.normalizer_class(self.embedding_dim)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    @staticmethod
    def octonion_normalizer(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6,
                            emb_rel_e7):
        denominator = torch.sqrt(
            emb_rel_e0 ** 2 + emb_rel_e1 ** 2 + emb_rel_e2 ** 2 + emb_rel_e3 ** 2 + emb_rel_e4 ** 2 +
            emb_rel_e5 ** 2 + emb_rel_e6 ** 2 + emb_rel_e7 ** 2)
        y0 = emb_rel_e0 / denominator
        y1 = emb_rel_e1 / denominator
        y2 = emb_rel_e2 / denominator
        y3 = emb_rel_e3 / denominator
        y4 = emb_rel_e4 / denominator
        y5 = emb_rel_e5 / denominator
        y6 = emb_rel_e6 / denominator
        y7 = emb_rel_e7 / denominator
        return y0, y1, y2, y3, y4, y5, y6, y7

    def residual_convolution(self, O_1, O_2):
        emb_ent_e0, emb_ent_e1, emb_ent_e2, emb_ent_e3, emb_ent_e4, emb_ent_e5, emb_ent_e6, emb_ent_e7 = O_1
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = O_2
        x = torch.cat([emb_ent_e0.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e1.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e2.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e3.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e4.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e5.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e6.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e7.view(-1, 1, 1, self.embedding_dim // 8),  # entities
                       emb_rel_e0.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e1.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e2.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e3.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e4.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e5.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e6.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e7.view(-1, 1, 1, self.embedding_dim // 8), ], 2)
        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        return torch.chunk(x, 8, dim=1)

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7 = torch.hsplit(
            head_ent_emb, 8)
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = torch.hsplit(
            rel_ent_emb,
            8)
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            (emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7) = self.octonion_normalizer(
                emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7)

        (emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3,
         emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7) = torch.hsplit(
            tail_ent_emb, 8)

        # (2) Apply convolution operation on (1.1) and (1.2).
        O_3 = self.residual_convolution(O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                                             emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                                        O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                                             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
        conv_e0, conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7 = O_3

        # (3)
        # (3.1) Apply quaternion multiplication.
        e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
            O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4,
                 emb_head_e5, emb_head_e6, emb_head_e7),
            O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
                 emb_rel_e5, emb_rel_e6, emb_rel_e7))
        # (4)
        # (4.4) Inner product
        e0_score = (conv_e0 * e0 * emb_tail_e0).sum(dim=1)
        e1_score = (conv_e1 * e1 * emb_tail_e1).sum(dim=1)
        e2_score = (conv_e2 * e2 * emb_tail_e2).sum(dim=1)
        e3_score = (conv_e3 * e3 * emb_tail_e3).sum(dim=1)
        e4_score = (conv_e4 * e4 * emb_tail_e4).sum(dim=1)
        e5_score = (conv_e5 * e5 * emb_tail_e5).sum(dim=1)
        e6_score = (conv_e6 * e6 * emb_tail_e6).sum(dim=1)
        e7_score = (conv_e7 * e7 * emb_tail_e7).sum(dim=1)
        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score

    def forward_k_vs_all(self, x: torch.Tensor):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        # (2) Split (1) into real and imaginary parts.
        emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7 = torch.hsplit(
            head_ent_emb, 8)
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = torch.hsplit(
            rel_ent_emb,
            8)
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            (emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7) = self.octonion_normalizer(
                emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7)

        # (2) Apply convolution operation on (1.1) and (1.2).
        O_3 = self.residual_convolution(O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                                             emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                                        O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                                             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
        conv_e0, conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7 = O_3

        # (3)
        # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
        e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
            O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4,
                 emb_head_e5, emb_head_e6, emb_head_e7),
            O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
                 emb_rel_e5, emb_rel_e6, emb_rel_e7))

        emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 = torch.hsplit(
            self.entity_embeddings.weight, 8)
        emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 = \
            emb_tail_e0.transpose(1, 0), emb_tail_e1.transpose(1, 0), \
                emb_tail_e2.transpose(1, 0), emb_tail_e3.transpose(1, 0), \
                emb_tail_e4.transpose(1, 0), emb_tail_e5.transpose(1, 0), emb_tail_e6.transpose(1,
                                                                                                0), emb_tail_e7.transpose(
                1, 0)

        # (4)
        # (4.4) Inner product
        e0_score = torch.mm(conv_e0 * e0, emb_tail_e0)
        e1_score = torch.mm(conv_e1 * e1, emb_tail_e1)
        e2_score = torch.mm(conv_e2 * e2, emb_tail_e2)
        e3_score = torch.mm(conv_e3 * e3, emb_tail_e3)
        e4_score = torch.mm(conv_e4 * e4, emb_tail_e4)
        e5_score = torch.mm(conv_e5 * e5, emb_tail_e5)
        e6_score = torch.mm(conv_e6 * e6, emb_tail_e6)
        e7_score = torch.mm(conv_e7 * e7, emb_tail_e7)
        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score


class AConvO(BaseKGE):
    """ Additive Convolutional Octonion Knowledge Graph Embeddings """

    def __init__(self, args: dict):
        super().__init__(args=args)
        self.name = 'AConvO'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim)  # Hard compression.
        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.norm_fc1 = self.normalizer_class(self.embedding_dim)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    @staticmethod
    def octonion_normalizer(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6,
                            emb_rel_e7):
        denominator = torch.sqrt(
            emb_rel_e0 ** 2 + emb_rel_e1 ** 2 + emb_rel_e2 ** 2 +
            emb_rel_e3 ** 2 + emb_rel_e4 ** 2 + emb_rel_e5 ** 2 + emb_rel_e6 ** 2 + emb_rel_e7 ** 2)
        y0 = emb_rel_e0 / denominator
        y1 = emb_rel_e1 / denominator
        y2 = emb_rel_e2 / denominator
        y3 = emb_rel_e3 / denominator
        y4 = emb_rel_e4 / denominator
        y5 = emb_rel_e5 / denominator
        y6 = emb_rel_e6 / denominator
        y7 = emb_rel_e7 / denominator
        return y0, y1, y2, y3, y4, y5, y6, y7

    def residual_convolution(self, O_1, O_2):
        emb_ent_e0, emb_ent_e1, emb_ent_e2, emb_ent_e3, emb_ent_e4, emb_ent_e5, emb_ent_e6, emb_ent_e7 = O_1
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = O_2
        x = torch.cat([emb_ent_e0.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e1.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e2.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e3.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e4.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e5.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e6.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_ent_e7.view(-1, 1, 1, self.embedding_dim // 8),  # entities
                       emb_rel_e0.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e1.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e2.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e3.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e4.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e5.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e6.view(-1, 1, 1, self.embedding_dim // 8),
                       emb_rel_e7.view(-1, 1, 1, self.embedding_dim // 8), ], 2)
        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        return torch.chunk(x, 8, dim=1)

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7 = torch.hsplit(
            head_ent_emb, 8)
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = torch.hsplit(
            rel_ent_emb,
            8)
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            (emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
             emb_rel_e5, emb_rel_e6, emb_rel_e7) = self.octonion_normalizer(emb_rel_e0, emb_rel_e1,
                                                                            emb_rel_e2, emb_rel_e3,
                                                                            emb_rel_e4, emb_rel_e5,
                                                                            emb_rel_e6, emb_rel_e7)

        (emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6,
         emb_tail_e7) = torch.hsplit(
            tail_ent_emb, 8)

        # (2) Apply convolution operation on (1.1) and (1.2).
        O_3 = self.residual_convolution(O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                                             emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                                        O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                                             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
        conv_e0, conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7 = O_3

        # (3)
        # (3.1) Apply quaternion multiplication.
        e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
            O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4,
                 emb_head_e5, emb_head_e6, emb_head_e7),
            O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
                 emb_rel_e5, emb_rel_e6, emb_rel_e7))
        # (4)
        # (4.4) Inner product
        e0_score = (conv_e0 + e0 * emb_tail_e0).sum(dim=1)
        e1_score = (conv_e1 + e1 * emb_tail_e1).sum(dim=1)
        e2_score = (conv_e2 + e2 * emb_tail_e2).sum(dim=1)
        e3_score = (conv_e3 + e3 * emb_tail_e3).sum(dim=1)
        e4_score = (conv_e4 + e4 * emb_tail_e4).sum(dim=1)
        e5_score = (conv_e5 + e5 * emb_tail_e5).sum(dim=1)
        e6_score = (conv_e6 + e6 * emb_tail_e6).sum(dim=1)
        e7_score = (conv_e7 + e7 * emb_tail_e7).sum(dim=1)
        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score

    def forward_k_vs_all(self, x: torch.Tensor):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        # (2) Split (1) into real and imaginary parts.
        (emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4,
         emb_head_e5, emb_head_e6, emb_head_e7) = torch.hsplit(
            head_ent_emb, 8)
        emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7 = torch.hsplit(
            rel_ent_emb,
            8)
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            (emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7) = self.octonion_normalizer(emb_rel_e0, emb_rel_e1,
                                                                                        emb_rel_e2, emb_rel_e3,
                                                                                        emb_rel_e4, emb_rel_e5,
                                                                                        emb_rel_e6, emb_rel_e7)

        # (2) Apply convolution operation on (1.1) and (1.2).
        O_3 = self.residual_convolution(O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3,
                                             emb_head_e4, emb_head_e5, emb_head_e6, emb_head_e7),
                                        O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3,
                                             emb_rel_e4, emb_rel_e5, emb_rel_e6, emb_rel_e7))
        conv_e0, conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7 = O_3

        # (3)
        # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
        e0, e1, e2, e3, e4, e5, e6, e7 = octonion_mul(
            O_1=(emb_head_e0, emb_head_e1, emb_head_e2, emb_head_e3, emb_head_e4,
                 emb_head_e5, emb_head_e6, emb_head_e7),
            O_2=(emb_rel_e0, emb_rel_e1, emb_rel_e2, emb_rel_e3, emb_rel_e4,
                 emb_rel_e5, emb_rel_e6, emb_rel_e7))

        emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 = \
            torch.hsplit(self.entity_embeddings.weight, 8)
        emb_tail_e0, emb_tail_e1, emb_tail_e2, emb_tail_e3, emb_tail_e4, emb_tail_e5, emb_tail_e6, emb_tail_e7 = \
            emb_tail_e0.transpose(1, 0), emb_tail_e1.transpose(1, 0), \
                emb_tail_e2.transpose(1, 0), emb_tail_e3.transpose(1, 0), emb_tail_e4.transpose(
                1, 0), emb_tail_e5.transpose(1, 0), emb_tail_e6.transpose(1, 0), emb_tail_e7.transpose(1, 0)

        # (4)
        # (4.4) Inner product
        e0_score = torch.mm(conv_e0 + e0, emb_tail_e0)
        e1_score = torch.mm(conv_e1 + e1, emb_tail_e1)
        e2_score = torch.mm(conv_e2 + e2, emb_tail_e2)
        e3_score = torch.mm(conv_e3 + e3, emb_tail_e3)
        e4_score = torch.mm(conv_e4 + e4, emb_tail_e4)
        e5_score = torch.mm(conv_e5 + e5, emb_tail_e5)
        e6_score = torch.mm(conv_e6 + e6, emb_tail_e6)
        e7_score = torch.mm(conv_e7 + e7, emb_tail_e7)
        return e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score
