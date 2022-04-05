import numpy as np
import torch
from typing import Tuple
from .base_model import *

class ConEx(BaseKGE):
    """
    Output of Residual connection is distributed over hermitian product
    Differs from original Conex.

    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'ConEx'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim*2)
        self.norm_fc1 = self.normalizer_class(self.embedding_dim*2)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, C_1: Tuple[torch.Tensor, torch.Tensor],
                             C_2: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
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

class OriginalConEx(BaseKGE):
    """



    """

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
                             C_2: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_k_vs_all(self, x: torch.Tensor):
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


class KDComplEx(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'KPDistMult'
        # Init Embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary i
        self.emb_rel_real = nn.Embedding(args.num_relations, int(sqrt(args.embedding_dim)))  # real
        self.emb_rel_i = nn.Embedding(args.num_relations, int(sqrt(args.embedding_dim)))  # imaginary i
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(args.input_dropout_rate)

        self.hidden_dp_a = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_b = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_c = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_d = torch.nn.Dropout(args.hidden_dropout_rate)

        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(int(sqrt(args.embedding_dim)))
        self.bn_rel_i = torch.nn.BatchNorm1d(int(sqrt(args.embedding_dim)))

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb.data.detach(), rel_emb.data.detach()

    def forward_k_vs_all(self, x):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        emb_head_i = self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i(e1_idx)))

        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        # (2) Retrieve  relation embeddings and apply kronecker_product
        emb_rel_real = batch_kronecker_product(emb_rel_real.unsqueeze(1), emb_rel_real.unsqueeze(1)).flatten(1)

        emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(self.emb_rel_i(rel_idx)))
        emb_rel_i = batch_kronecker_product(emb_rel_i.unsqueeze(1), emb_rel_i.unsqueeze(1)).flatten(1)

        real_real_real = torch.mm(self.hidden_dp_a(emb_head_real * emb_rel_real),
                                  self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(self.hidden_dp_b(emb_head_real * emb_rel_i), self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(self.hidden_dp_c(emb_head_i * emb_rel_real), self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(self.hidden_dp_d(emb_head_i * emb_rel_i), self.emb_ent_real.weight.transpose(1, 0))

        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
