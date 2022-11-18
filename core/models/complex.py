import numpy as np
import torch
from typing import Tuple
from .base_model import *


class SumConEx(BaseKGE):
    """
    Output of Residual connection is distributed over hermitian product
    Differs from original Conex.

    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'SumConEx'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim * 2)
        self.norm_fc1 = self.normalizer_class(self.embedding_dim * 2)
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

        x = F.relu(self.conv2d(x))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.norm_fc1(self.fc1(x)))
        return torch.chunk(x, 4, dim=1)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b, c, d = C_3
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (4)
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
        # This can be decomposed into 4 as well
        a, b, c, d = C_3

        # (3) Compute hermitian inner product.
        real_real_real = (a + emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (b + emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (c + emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (d + emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)

        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real


class ConEx(BaseKGE):

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

        x = F.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.norm_fc1(self.fc1(x)))
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
        real_real_real = torch.mm(a + emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(a + emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(b + emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(b + emb_head_imag * emb_rel_imag, emb_tail_real)
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
        # This can be decomposed into 4 as well
        a, b = C_3

        # (3) Compute hermitian inner product.

        real_real_real = (a * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (a * emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (b * emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (b * emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)

        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real


class ComplEx(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'ComplEx'

    def forward_triples(self, x: torch.Tensor)-> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)
        # (3) Compute hermitian inner product.
        real_real_real = (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_all(self, x: torch.Tensor)-> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Transpose Entity embedding matrix to perform matrix multiplications in Hermitian Product.
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (4) Compute hermitian inner product on embedding vectors.
        real_real_real = torch.mm(emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
