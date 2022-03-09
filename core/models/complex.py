
import numpy as np
import torch
from typing import Tuple
from .base_model import *


class ConEx(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'ConEx'

        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)

        # Convolution
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        fc_num_input = self.embedding_dim * 4 * self.num_of_output_channels  # 4 because of 4 real values in 2 complex numbers
        self.fc = torch.nn.Linear(fc_num_input, self.embedding_dim * 2)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(self.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(self.embedding_dim)

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(self.embedding_dim * 2)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def get_embeddings(self) -> Tuple[np.array, np.array]:
        """
        prepare embeddings
        :return:
        """
        # Combine real and complex parts
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

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
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim)], 2)

        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_k_vs_all(self, x:torch.Tensor) -> torch.Tensor:
        """
        Compute scores of all entities
        :param x:
        :param e1_idx:
        :param rel_idx:
        :return:
        """
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.bn_ent_real(self.emb_ent_real(e1_idx))
        emb_head_i = self.bn_ent_i(self.emb_ent_i(e1_idx))
        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.bn_rel_real(self.emb_rel_real(rel_idx))
        emb_rel_i = self.bn_rel_i(self.emb_rel_i(rel_idx))

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_i),
                                        C_2=(emb_rel_real, emb_rel_i))
        a, b = C_3

        # (3) Apply dropout out on (1).
        emb_head_real = self.input_dp_ent_real(emb_head_real)
        emb_head_i = self.input_dp_ent_i(emb_head_i)
        emb_rel_real = self.input_dp_rel_real(emb_rel_real)
        emb_rel_i = self.input_dp_rel_i(emb_rel_i)

        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and all entities.
        real_real_real = torch.mm(a * emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(a * emb_head_real * emb_rel_i, self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(b * emb_head_i * emb_rel_real, self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(b * emb_head_i * emb_rel_i, self.emb_ent_real.weight.transpose(1, 0))
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1],x[:, 2]
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        # (1.2) Complex embeddings of relations.
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)

        # (1.2) Complex embeddings of tail entities.
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_i),
                                        C_2=(emb_rel_real, emb_rel_i))
        a, b = C_3

        # (3) Apply dropout out on (1).
        emb_head_real = self.input_dp_ent_real(emb_head_real)
        emb_head_i = self.input_dp_ent_i(emb_head_i)
        emb_rel_real = self.input_dp_rel_real(emb_rel_real)
        emb_rel_i = self.input_dp_rel_i(emb_rel_i)
        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and tail entities
        # Compute multi-linear product embeddings
        real_real_real = (a * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (a * emb_head_real * emb_rel_i * emb_tail_i).sum(dim=1)
        imag_real_imag = (b * emb_head_i * emb_rel_real * emb_tail_i).sum(dim=1)
        imag_imag_real = (b * emb_head_i * emb_rel_i * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real


class ComplEx(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'ComplEx'
        # Init Embeddings
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(self.input_dropout_rate)

        self.hidden_dp_a = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_b = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_c = torch.nn.Dropout(self.hidden_dropout_rate)
        self.hidden_dp_d = torch.nn.Dropout(self.hidden_dropout_rate)

        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(self.embedding_dim)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

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
        emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(self.emb_rel_i(rel_idx)))

        real_real_real = torch.mm(self.hidden_dp_a(emb_head_real * emb_rel_real),
                                  self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(self.hidden_dp_b(emb_head_real * emb_rel_i), self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(self.hidden_dp_c(emb_head_i * emb_rel_real), self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(self.hidden_dp_d(emb_head_i * emb_rel_i), self.emb_ent_real.weight.transpose(1, 0))

        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1],x[:, 2]
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        emb_head_i = self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i(e1_idx)))

        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(self.emb_rel_i(rel_idx)))

        # (1.3) Complex embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)

        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and tail entities
        # Compute multi-linear product embeddings
        real_real_real = (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (emb_head_real * emb_rel_i * emb_tail_i).sum(dim=1)
        imag_real_imag = (emb_head_i * emb_rel_real * emb_tail_i).sum(dim=1)
        imag_imag_real = (emb_head_i * emb_rel_i * emb_tail_real).sum(dim=1)
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
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

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
