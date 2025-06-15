from typing import Tuple
import torch
from .base_model import BaseKGE


class ConEx(BaseKGE):
    """ Convolutional ComplEx Knowledge Graph Embeddings"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'ConEx'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim)  # Hard compression.
        self.norm_fc1 = self.normalizer_class(self.embedding_dim)

        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, C_1: Tuple[torch.Tensor, torch.Tensor],
                             C_2: Tuple[torch.Tensor, torch.Tensor]) -> torch.FloatTensor:
        """
        Compute residual score of two complex-valued embeddings.
        :param C_1: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :param C_2: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :return:
        """
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # Think of x a n image of two complex numbers.
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 2)], 2)

        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b = C_3
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (4)
        real_real_real = torch.mm(a * emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(a * emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(b * emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(b * emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b = C_3
        # (3) Compute hermitian inner product.
        real_real_real = (a * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (a * emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (b * emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (b * emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_sample(self, x: torch.Tensor, target_entity_idx: torch.Tensor):
        # @OTOD: Double check later.
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (2).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b = C_3

        # (batch size, num. selected entity, dimension)
        # tail_entity_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(target_entity_idx))
        tail_entity_emb = self.entity_embeddings(target_entity_idx)
        # complex vectors
        emb_tail_real, emb_tail_i = torch.tensor_split(tail_entity_emb, 2, dim=2)

        emb_tail_real = emb_tail_real.transpose(1, 2)
        emb_tail_i = emb_tail_i.transpose(1, 2)

        real_real_real = torch.bmm((a * emb_head_real * emb_rel_real).unsqueeze(1), emb_tail_real)
        real_imag_imag = torch.bmm((a * emb_head_real * emb_rel_imag).unsqueeze(1), emb_tail_i)
        imag_real_imag = torch.bmm((b * emb_head_imag * emb_rel_real).unsqueeze(1), emb_tail_i)
        imag_imag_real = torch.bmm((b * emb_head_imag * emb_rel_imag).unsqueeze(1), emb_tail_real)
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        return score.squeeze(1)


class AConEx(BaseKGE):
    """ Additive Convolutional ComplEx Knowledge Graph Embeddings """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'AConEx'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim + self.embedding_dim)  # Hard compression.
        self.norm_fc1 = self.normalizer_class(self.embedding_dim + self.embedding_dim)

        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, C_1: Tuple[torch.Tensor, torch.Tensor],
                             C_2: Tuple[torch.Tensor, torch.Tensor]) -> torch.FloatTensor:
        """
        Compute residual score of two complex-valued embeddings.
        :param C_1: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :param C_2: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :return:
        """
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # (N,C,H,W) : A single channel 2D image.
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 2)], 2)

        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        #
        return torch.chunk(x, 4, dim=1)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b, c, d = C_3
        # (4) Retrieve tail entity embeddings.
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        # (5) Transpose (4).
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (6) Hermitian inner product with additive Conv2D connection.
        real_real_real = torch.mm(a + emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(b + emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(c + emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(d + emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)
        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b, c, d = C_3
        # (3) Hermitian inner product with additive Conv2D connection.
        real_real_real = (a + emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (b + emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (c + emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (d + emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_sample(self, x: torch.Tensor, target_entity_idx: torch.Tensor):
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (2).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b, c, d = C_3

        # (4) Retrieve selected tail entity embeddings
        tail_entity_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(target_entity_idx))
        # (5) Split (4) into real and imaginary parts.
        emb_tail_real, emb_tail_i = torch.tensor_split(tail_entity_emb, 2, dim=2)
        # (6) Transpose (5)
        emb_tail_real = emb_tail_real.transpose(1, 2)
        emb_tail_i = emb_tail_i.transpose(1, 2)
        # (7) Hermitian inner product with additive Conv2D connection
        # (7.1) Elementwise multiply (2) according to the Hermitian Inner Product order
        # (7.2) Additive connection: Add (3) into (7.1)
        # (7.3) Batch matrix multiplication (7.2) and tail entity embeddings.
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        # input.shape (N, 1, D), mat2.shape (N,D,1)
        real_real_real = torch.bmm((a + emb_head_real * emb_rel_real).unsqueeze(1), emb_tail_real)
        real_imag_imag = torch.bmm((b + emb_head_real * emb_rel_imag).unsqueeze(1), emb_tail_i)
        imag_real_imag = torch.bmm((c + emb_head_imag * emb_rel_real).unsqueeze(1), emb_tail_i)
        imag_imag_real = torch.bmm((d + emb_head_imag * emb_rel_imag).unsqueeze(1), emb_tail_real)
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        # (N,1,1) => (N,1).
        return score.squeeze(1)


class ComplEx(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'ComplEx'

    @staticmethod
    def score(head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor, tail_ent_emb: torch.FloatTensor):
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)
        # (3) Compute hermitian inner product.
        real_real_real = (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    @staticmethod
    def k_vs_all_score(emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor):
        """

        Parameters
        ----------
        emb_h
        emb_r
        emb_E

        Returns
        -------

        """
        emb_head_real, emb_head_imag = torch.hsplit(emb_h, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(emb_r, 2)
        # (3) Transpose Entity embedding matrix to perform matrix multiplications in Hermitian Product.
        emb_tail_real, emb_tail_imag = torch.hsplit(emb_E, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (4) Compute hermitian inner product on embedding vectors.
        real_real_real = torch.mm(emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        return self.k_vs_all_score(head_ent_emb,rel_ent_emb,self.entity_embeddings.weight)