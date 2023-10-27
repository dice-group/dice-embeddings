import torch
from .static_funcs import quaternion_mul
from .base_model import BaseKGE, IdentityClass


def quaternion_mul_with_unit_norm(*, Q_1, Q_2):
    a_h, b_h, c_h, d_h = Q_1  # = {a_h + b_h i + c_h j + d_h k : a_r, b_r, c_r, d_r \in R^k}
    a_r, b_r, c_r, d_r = Q_2  # = {a_r + b_r i + c_r j + d_r k : a_r, b_r, c_r, d_r \in R^k}

    # Normalize the relation to eliminate the scaling effect
    denominator = torch.sqrt(a_r ** 2 + b_r ** 2 + c_r ** 2 + d_r ** 2)
    p = a_r / denominator
    q = b_r / denominator
    u = c_r / denominator
    v = d_r / denominator
    #  Q'=E Hamilton product R
    r_val = a_h * p - b_h * q - c_h * u - d_h * v
    i_val = a_h * q + b_h * p + c_h * v - d_h * u
    j_val = a_h * u - b_h * v + c_h * p + d_h * q
    k_val = a_h * v + b_h * u - c_h * q + d_h * p
    return r_val, i_val, j_val, k_val


class QMult(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'QMult'
        self.explicit = True
        if self.explicit is False:
            _1, _i, _j, _k = 0, 1, 2, 3
            self.multiplication_table = torch.zeros(4, 4, 4)
            for i, j, k, v in [
                # 1 * ? = ?; ? * 1 = ?
                (_1, _1, _1, 1),
                (_1, _i, _i, 1),
                (_1, _j, _j, 1),
                (_1, _k, _k, 1),
                (_i, _1, _i, 1),
                (_j, _1, _j, 1),
                (_k, _1, _k, 1),
                # i**2 = j**2 = k**2 = -1
                (_i, _i, _1, -1),
                (_j, _j, _1, -1),
                (_k, _k, _1, -1),
                # i * j = k; i * k = -j
                (_i, _j, _k, 1),
                (_i, _k, _j, -1),
                # j * i = -k, j * k = i
                (_j, _i, _k, -1),
                (_j, _k, _i, 1),
                # k * i = j; k * j = -i
                (_k, _i, _j, 1),
                (_k, _j, _i, -1),
            ]:
                self.multiplication_table[i, j, k] = v

    def quaternion_multiplication_followed_by_inner_product(self, h, r, t):
        """
        :param h: shape: (`*batch_dims`, dim)
            The head representations.
        :param r: shape: (`*batch_dims`, dim)
            The head representations.
        :param t: shape: (`*batch_dims`, dim)
            The tail representations.
        :return:
            Triple scores.
        """
        n, d = h.shape
        h = h.reshape(n, d // 4, 4)
        r = r.reshape(n, d // 4, 4)
        t = t.reshape(n, d // 4, 4)
        return -torch.einsum("...di, ...dj, ...dk, ijk -> ...", h, r, t, self.multiplication_table)

    @staticmethod
    def quaternion_normalizer(x: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Normalize the length of relation vectors, if the forward constraint has not been applied yet.

        Absolute value of a quaternion

        .. math::

            |a + bi + cj + dk| = \sqrt{a^2 + b^2 + c^2 + d^2}

        L2 norm of quaternion vector:

        .. math::
            \|x\|^2 = \sum_{i=1}^d |x_i|^2
                     = \sum_{i=1}^d (x_i.re^2 + x_i.im_1^2 + x_i.im_2^2 + x_i.im_3^2)
        :param x:
            The vector.

        :return:
            The normalized vector.
        """
        # Normalize relation embeddings
        shape = x.shape
        x = x.view(*shape[:-1], -1, 4)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x.view(*shape)

    def score(self, head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor, tail_ent_emb: torch.FloatTensor):
        # (1.1) If No normalization set, we need to apply quaternion normalization
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            rel_ent_emb = self.quaternion_normalizer(rel_ent_emb)
        if self.explicit is False:
            return self.quaternion_multiplication_followed_by_inner_product(head_ent_emb, rel_ent_emb, tail_ent_emb)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(tail_ent_emb, 4)
        # (2)
        # (2.1) Apply quaternion multiplication on (1.1) and (2.1).
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # (3)
        # (3.1) Inner product
        real_score = torch.sum(r_val * emb_tail_real, dim=1)
        i_score = torch.sum(i_val * emb_tail_i, dim=1)
        j_score = torch.sum(j_val * emb_tail_j, dim=1)
        k_score = torch.sum(k_val * emb_tail_k, dim=1)
        return real_score + i_score + j_score + k_score

    def k_vs_all_score(self, bpe_head_ent_emb, bpe_rel_ent_emb, E):
        """

        Parameters
        ----------
        bpe_head_ent_emb
        bpe_rel_ent_emb
        E

        Returns
        -------

        """
        # (1.1) If No normalization set, we need to apply quaternion normalization
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            bpe_rel_ent_emb = self.quaternion_normalizer(bpe_rel_ent_emb)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(bpe_head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(bpe_rel_ent_emb, 4)
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(E, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = emb_tail_real.transpose(1, 0), emb_tail_i.transpose(1, 0), \
            emb_tail_j.transpose(1, 0), emb_tail_k.transpose(1, 0)

        # (3)
        # (3.1) Inner product
        real_score = torch.mm(r_val, emb_tail_real)
        i_score = torch.mm(i_val, emb_tail_i)
        j_score = torch.mm(j_val, emb_tail_j)
        k_score = torch.mm(k_val, emb_tail_k)
        return real_score + i_score + j_score + k_score

    def forward_k_vs_all(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        return self.k_vs_all_score(head_ent_emb, rel_ent_emb,self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x, target_entity_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (1.1) If No normalization set, we need to apply quaternion normalization
        if isinstance(self.normalize_relation_embeddings, IdentityClass):
            rel_ent_emb = self.quaternion_normalizer(rel_ent_emb)

        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

        # (batch size, num. selected entity, dimension)
        tail_entity_emb = self.entity_embeddings(target_entity_idx)
        # quaternion vectors
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.tensor_split(tail_entity_emb, 4, dim=2)

        emb_tail_real = emb_tail_real.transpose(1, 2)
        emb_tail_i = emb_tail_i.transpose(1, 2)
        emb_tail_j = emb_tail_j.transpose(1, 2)
        emb_tail_k = emb_tail_k.transpose(1, 2)

        # (batch size, 1, dimension)
        r_val = r_val.unsqueeze(1)
        i_val = i_val.unsqueeze(1)
        j_val = j_val.unsqueeze(1)
        k_val = k_val.unsqueeze(1)

        real_score = torch.bmm(r_val, emb_tail_real)
        i_score = torch.bmm(i_val, emb_tail_i)
        j_score = torch.bmm(j_val, emb_tail_j)
        k_score = torch.bmm(k_val, emb_tail_k)

        return (real_score + i_score + j_score + k_score).squeeze(1)


class ConvQ(BaseKGE):
    """ Convolutional Quaternion Knowledge Graph Embeddings

    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'ConvQ'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels  # 8 because of 8 real values in 2 quaternions
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim)  # Hard compression.

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = self.normalizer_class(self.embedding_dim)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, Q_1, Q_2):
        emb_ent_real, emb_ent_imag_i, emb_ent_imag_j, emb_ent_imag_k = Q_1
        emb_rel_real, emb_rel_imag_i, emb_rel_imag_j, emb_rel_imag_k = Q_2
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_j.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_k.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_j.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_k.view(-1, 1, 1, self.embedding_dim // 4)], 2)

        # n, c_in, h_in, w_in x.shape before conv. h_in=8, w_in embeddings
        x = self.conv2d(x)
        # n, c_out, h_out, w_out x.shape after conv.
        x = self.bn_conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.bn_conv2(self.fc1(x)))
        return torch.chunk(x, 4, dim=1)

    def forward_triples(self, indexed_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(indexed_triple)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(tail_ent_emb, 4)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3
        # (3)
        # (3.1) Apply quaternion multiplication on (1.1) and (3.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # (4)
        # (4.1) Hadamard product of (2) with (3) and inner product with tails
        real_score = torch.sum(conv_real * r_val * emb_tail_real, dim=1)
        i_score = torch.sum(conv_imag_i * i_val * emb_tail_i, dim=1)
        j_score = torch.sum(conv_imag_j * j_val * emb_tail_j, dim=1)
        k_score = torch.sum(conv_imag_k * k_val * emb_tail_k, dim=1)
        return real_score + i_score + j_score + k_score

    def forward_k_vs_all(self, x: torch.Tensor):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3

        # (3)
        # (3.1) Apply quaternion multiplication.
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # Prepare all entity embeddings.
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(self.entity_embeddings.weight, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = emb_tail_real.transpose(1, 0), \
            emb_tail_i.transpose(1, 0), emb_tail_j.transpose(
            1, 0), emb_tail_k.transpose(1, 0)

        # (4)
        # (4.1) Hadamard product of (2) with (3) and inner product with tails
        real_score = torch.mm(conv_real * r_val, emb_tail_real)
        i_score = torch.mm(conv_imag_i * i_val, emb_tail_i)
        j_score = torch.mm(conv_imag_j * j_val, emb_tail_j)
        k_score = torch.mm(conv_imag_k * k_val, emb_tail_k)

        return real_score + i_score + j_score + k_score


class AConvQ(BaseKGE):
    """ Additive Convolutional Quaternion Knowledge Graph Embeddings """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'AConvQ'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels  # 8 because of 8 real values in 2 quaternions
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim)  # Hard compression.

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = self.normalizer_class(self.embedding_dim)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, Q_1, Q_2):
        emb_ent_real, emb_ent_imag_i, emb_ent_imag_j, emb_ent_imag_k = Q_1
        emb_rel_real, emb_rel_imag_i, emb_rel_imag_j, emb_rel_imag_k = Q_2
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_j.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_ent_imag_k.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_j.view(-1, 1, 1, self.embedding_dim // 4),
                       emb_rel_imag_k.view(-1, 1, 1, self.embedding_dim // 4)], 2)

        # n, c_in, h_in, w_in x.shape before conv. h_in=8, w_in embeddings
        x = self.conv2d(x)
        # n, c_out, h_out, w_out x.shape after conv.
        x = self.bn_conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.bn_conv2(self.fc1(x)))
        return torch.chunk(x, 4, dim=1)

    def forward_triples(self, indexed_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(indexed_triple)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(tail_ent_emb, 4)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3
        # (3)
        # (3.1) Apply quaternion multiplication on (1.1) and (3.1).
        r_val, i_val, j_val, k_val = quaternion_mul(
            Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
            Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # (4)
        # (4.1) Hadamard product of (2) with (3) and inner product with tails
        real_score = torch.sum(conv_real + r_val * emb_tail_real, dim=1)
        i_score = torch.sum(conv_imag_i + i_val * emb_tail_i, dim=1)
        j_score = torch.sum(conv_imag_j + j_val * emb_tail_j, dim=1)
        k_score = torch.sum(conv_imag_k + k_val * emb_tail_k, dim=1)
        return real_score + i_score + j_score + k_score

    def forward_k_vs_all(self, x: torch.Tensor):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(head_ent_emb, 4)
        emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(rel_ent_emb, 4)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3

        # (3)
        # (3.1) Apply quaternion multiplication.
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                    Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        # Prepare all entity embeddings.
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(self.entity_embeddings.weight, 4)
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = emb_tail_real.transpose(1, 0), \
            emb_tail_i.transpose(1, 0), emb_tail_j.transpose(
            1, 0), emb_tail_k.transpose(1, 0)

        # (4)
        # (4.1) Hadamard product of (2) with (3) and inner product with tails
        real_score = torch.mm(conv_real + r_val, emb_tail_real)
        i_score = torch.mm(conv_imag_i + i_val, emb_tail_i)
        j_score = torch.mm(conv_imag_j + j_val, emb_tail_j)
        k_score = torch.mm(conv_imag_k + k_val, emb_tail_k)

        return real_score + i_score + j_score + k_score
