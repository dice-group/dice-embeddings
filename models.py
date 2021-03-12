from dataset import KvsAll, RelationPredictionDataset

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.metrics.functional import accuracy
from typing import List, Any
from torch.nn.init import xavier_normal_

"""

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x1_batch, x2_batch, y_batch = batch
        train_loss = self.loss_function(self(x1_batch, x2_batch), y_batch)
        return {'loss': train_loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        predictions = self(x1_batch, x2_batch)
        val_loss = self.loss_function(predictions, y_batch)
        val_accuracy = accuracy(predictions, y_batch)
        return {'val_acc': val_accuracy, 'val_loss': val_loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        x = [[x['val_acc'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_loss_per_epoch', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc_per_epoch', avg_val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        test_accuracy = accuracy(self(x1_batch, x2_batch), y_batch)
        return {'test_accuracy': test_accuracy}

    def test_epoch_end(self, outputs: List[Any]):
        avg_test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('avg_test_accuracy', avg_test_accuracy, on_epoch=True, prog_bar=True)

"""

class BaseKGE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.name = 'Not init'

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x1_batch, x2_batch, y_batch = batch
        train_loss = self.loss_function(self(x1_batch, x2_batch), y_batch)
        return {'loss': train_loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        predictions = self(x1_batch, x2_batch)
        val_loss = self.loss_function(predictions, y_batch)
        val_accuracy = accuracy(predictions, y_batch)
        return {'val_acc': val_accuracy, 'val_loss': val_loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        x = [[x['val_acc'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_loss_per_epoch', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc_per_epoch', avg_val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        test_accuracy = accuracy(self(x1_batch, x2_batch), y_batch)
        return {'test_accuracy': test_accuracy}

    def test_epoch_end(self, outputs: List[Any]):
        avg_test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('avg_test_accuracy', avg_test_accuracy, on_epoch=True, prog_bar=True)


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


def quaternion_mul(*, Q_1, Q_2):
    a_h, b_h, c_h, d_h = Q_1  # = {a_h + b_h i + c_h j + d_h k : a_r, b_r, c_r, d_r \in R^k}
    a_r, b_r, c_r, d_r = Q_2  # = {a_r + b_r i + c_r j + d_r k : a_r, b_r, c_r, d_r \in R^k}
    r_val = a_h * a_r - b_h * b_r - c_h * c_r - d_h * d_r
    i_val = a_h * b_r + b_h * a_r + c_h * d_r - d_h * c_r
    j_val = a_h * c_r - b_h * d_r + c_h * a_r + d_h * b_r
    k_val = a_h * d_r + b_h * c_r - c_h * b_r + d_h * a_r
    return r_val, i_val, j_val, k_val


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


class Shallom(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'Shallom'
        shallom_width = int(args.shallom_width_ratio_of_emb * args.embedding_dim)
        self.loss = torch.nn.BCELoss()
        self.entity_embeddings = nn.Embedding(args.num_entities, args.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data)
        self.shallom = nn.Sequential(nn.Dropout(args.input_dropout_rate),
                                     torch.nn.Linear(args.embedding_dim * 2, shallom_width),
                                     nn.BatchNorm1d(shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(args.hidden_dropout_rate),
                                     torch.nn.Linear(shallom_width, args.num_relations))

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.detach().numpy()

    def forward(self, s, o):
        emb_s, emb_o = self.entity_embeddings(s), self.entity_embeddings(o)
        return torch.sigmoid(self.shallom(torch.cat((emb_s, emb_o), 1)))

    """

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)
    def training_step(self, batch, batch_idx):
        x1_batch, x2_batch, y_batch = batch
        train_loss = self.loss_function(self(x1_batch, x2_batch), y_batch)
        return {'loss': train_loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_loss_per_epoch', avg_loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        predictions = self(x1_batch, x2_batch)
        val_loss = self.loss_function(predictions, y_batch)
        val_accuracy = accuracy(predictions, y_batch)
        return {'val_acc': val_accuracy, 'val_loss': val_loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        x = [[x['val_acc'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_loss_per_epoch', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc_per_epoch', avg_val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        test_accuracy = accuracy(self(x1_batch, x2_batch), y_batch)
        return {'test_accuracy': test_accuracy}

    def test_epoch_end(self, outputs: List[Any]):
        avg_test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('avg_test_accuracy', avg_test_accuracy, on_epoch=True, prog_bar=True)


    """


class ConEx(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'ConEx'
        self.loss = torch.nn.BCELoss()
        # Init Embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary i
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary i
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)

        # Init Conv.
        self.kernel_size = args.kernel_size  # Square filter.
        self.num_of_output_channels = args.num_of_output_channels
        # Convolution
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=args.num_of_output_channels,
                                     kernel_size=(args.kernel_size, args.kernel_size), stride=1, padding=1, bias=True)

        fc_num_input = args.embedding_dim * 4 * args.num_of_output_channels  # 4 because of 4 real values in 2 complex numbers
        self.fc = torch.nn.Linear(fc_num_input, args.embedding_dim * 2)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(args.embedding_dim)

        self.bn_conv1 = torch.nn.BatchNorm2d(args.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(args.embedding_dim * 2)
        self.feature_map_dropout = torch.nn.Dropout2d(args.feature_map_dropout_rate)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

    def residual_convolution(self, C_1, C_2):
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

    def forward(self, e1_idx, rel_idx):
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
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

        return torch.sigmoid(score)


class QMult(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'QMult'
        self.loss = torch.nn.BCELoss()
        self.apply_unit_norm = args.apply_unit_norm

        # Quaternion embeddings of entities
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary i
        self.emb_ent_j = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary j
        self.emb_ent_k = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_ent_j.weight.data), xavier_normal_(self.emb_ent_k.weight.data)

        # Quaternion embeddings of relations.
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary i
        self.emb_rel_j = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary j
        self.emb_rel_k = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)
        xavier_normal_(self.emb_rel_j.weight.data), xavier_normal_(self.emb_rel_k.weight.data)

        # Dropouts for quaternion embeddings of ALL entities.
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_j = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_k = torch.nn.Dropout(args.input_dropout_rate)
        # Dropouts for quaternion embeddings of relations.
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_j = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_k = torch.nn.Dropout(args.input_dropout_rate)
        # Dropouts for quaternion embeddings obtained from quaternion multiplication.
        self.hidden_dp_real = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_i = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_j = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_k = torch.nn.Dropout(args.hidden_dropout_rate)

        # Batch normalization for quaternion embeddings of ALL entities.
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_j = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_k = torch.nn.BatchNorm1d(args.embedding_dim)
        # Batch normalization for quaternion embeddings of relations.
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_j = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_k = torch.nn.BatchNorm1d(args.embedding_dim)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data,
                                self.emb_ent_j.weight.data, self.emb_ent_k.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data,
                             self.emb_rel_j.weight.data, self.emb_rel_k.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

    def forward(self, e1_idx, rel_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)

        if self.apply_unit_norm:
            # (2) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (3) Inner product of (2) with all entities.
            real_score = torch.mm(r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(i_val, self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(j_val, self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(k_val, self.emb_ent_k.weight.transpose(1, 0))
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2)-relations.
            # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(emb_head_real)),
                     self.input_dp_ent_i(self.bn_ent_i(emb_head_i)),
                     self.input_dp_ent_j(self.bn_ent_j(emb_head_j)),
                     self.input_dp_ent_k(self.bn_ent_k(emb_head_k))),
                Q_2=(
                    self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                    self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                    self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                    self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))

            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Inner product
            real_score = torch.mm(self.hidden_dp_real(r_val), self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(i_val), self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(self.hidden_dp_j(j_val), self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(self.hidden_dp_k(k_val), self.emb_ent_k.weight.transpose(1, 0))

        score = real_score + i_score + j_score + k_score
        return torch.sigmoid(score)


class OMult(BaseKGE):

    def __init__(self, args):
        super().__init__()
        self.name = 'OMult'
        self.loss = torch.nn.BCELoss()

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

    def forward(self, e1_idx, rel_idx):
        """
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
            [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
            Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
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
            e0_score = torch.mm(self.hidden_dp_e0(e0), self.emb_ent_e0.weight.transpose(1, 0))

            e1_score = torch.mm(self.hidden_dp_e1(e1), self.emb_ent_e1.weight.transpose(1, 0))
            e2_score = torch.mm(self.hidden_dp_e2(e2), self.emb_ent_e2.weight.transpose(1, 0))
            e3_score = torch.mm(self.hidden_dp_e3(e3),
                                self.emb_ent_e3.weight.transpose(1, 0))
            e4_score = torch.mm(self.hidden_dp_e4(e4), self.emb_ent_e4.weight.transpose(1, 0))
            e5_score = torch.mm(self.hidden_dp_e5(e5), self.emb_ent_e5.weight.transpose(1, 0))
            e6_score = torch.mm(self.hidden_dp_e6(e6), self.emb_ent_e6.weight.transpose(1, 0))
            e7_score = torch.mm(self.hidden_dp_e7(e7), self.emb_ent_e7.weight.transpose(1, 0))
        score = e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score
        return torch.sigmoid(score)


class ConvQ(BaseKGE):
    """ Convolutional Quaternion Knowledge Graph Embeddings"""

    def __init__(self, args):
        super().__init__()
        self.name = 'ConvQ'
        self.loss = torch.nn.BCELoss()
        self.apply_unit_norm = args.apply_unit_norm
        self.embedding_dim = args.embedding_dim  # for reshaping in the residual.

        # Quaternion embeddings of entities
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary i
        self.emb_ent_j = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary j
        self.emb_ent_k = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_ent_j.weight.data), xavier_normal_(self.emb_ent_k.weight.data)

        # Quaternion embeddings of relations.
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary i
        self.emb_rel_j = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary j
        self.emb_rel_k = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary k
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)
        xavier_normal_(self.emb_rel_j.weight.data), xavier_normal_(self.emb_rel_k.weight.data)

        # Dropouts for quaternion embeddings of ALL entities.
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_j = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_k = torch.nn.Dropout(args.input_dropout_rate)
        # Dropouts for quaternion embeddings of relations.
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_j = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_k = torch.nn.Dropout(args.input_dropout_rate)
        # Dropouts for quaternion embeddings obtained from quaternion multiplication.
        self.hidden_dp_real = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_i = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_j = torch.nn.Dropout(args.hidden_dropout_rate)
        self.hidden_dp_k = torch.nn.Dropout(args.hidden_dropout_rate)

        # Batch normalization for quaternion embeddings of ALL entities.
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_j = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_k = torch.nn.BatchNorm1d(args.embedding_dim)
        # Batch normalization for quaternion embeddings of relations.
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_j = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_k = torch.nn.BatchNorm1d(args.embedding_dim)

        self.kernel_size = args.kernel_size
        self.num_of_output_channels = args.num_of_output_channels

        # Convolution
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = args.embedding_dim * 8 * self.num_of_output_channels  # 8 because of 8 real values in 2 quaternions
        self.fc1 = torch.nn.Linear(self.fc_num_input, args.embedding_dim * 4)  # Hard compression.

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(args.embedding_dim * 4)
        self.feature_map_dropout = torch.nn.Dropout2d(args.feature_map_dropout_rate)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data,
                                self.emb_ent_j.weight.data, self.emb_ent_k.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data,
                             self.emb_rel_j.weight.data, self.emb_rel_k.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

    def residual_convolution(self, Q_1, Q_2):
        emb_ent_real, emb_ent_imag_i, emb_ent_imag_j, emb_ent_imag_k = Q_1
        emb_rel_real, emb_rel_imag_i, emb_rel_imag_j, emb_rel_imag_k = Q_2
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_j.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_k.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_j.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_k.view(-1, 1, 1, self.embedding_dim)], 2)

        # Think of x a n image of two quaternions.
        # Batch norms after fully connnect and Conv layers
        # and before nonlinearity.
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc1(x)))
        return torch.chunk(x, 4, dim=1)

    def forward(self, e1_idx, rel_idx):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(e1_idx)
        emb_head_i = self.emb_ent_i(e1_idx)
        emb_head_j = self.emb_ent_j(e1_idx)
        emb_head_k = self.emb_ent_k(e1_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(rel_idx)
        emb_rel_i = self.emb_rel_i(rel_idx)
        emb_rel_j = self.emb_rel_j(rel_idx)
        emb_rel_k = self.emb_rel_k(rel_idx)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3
        if self.apply_unit_norm:
            # (3) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Inner product of (4.1) with ALL entities.
            real_score = torch.mm(conv_real * r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(conv_imag_i * i_val, self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(conv_imag_j * j_val, self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(conv_imag_k * k_val, self.emb_ent_k.weight.transpose(1, 0))
        else:
            # (3)
            # (3.1) Apply BN + Dropout on (1.2).
            # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(emb_head_real)),
                     self.input_dp_ent_i(self.bn_ent_i(emb_head_i)),
                     self.input_dp_ent_j(self.bn_ent_j(emb_head_j)),
                     self.input_dp_ent_k(self.bn_ent_k(emb_head_k))),
                Q_2=(self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                     self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                     self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                     self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Dropout on (4.1).
            # (4.3) Inner product
            real_score = torch.mm(self.hidden_dp_real(conv_real * r_val), self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(conv_imag_i * i_val), self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(self.hidden_dp_j(conv_imag_j * j_val),
                               self.input_dp_ent_j(self.bn_ent_j(self.emb_ent_j.weight)).transpose(1, 0))
            k_score = torch.mm(self.hidden_dp_k(conv_imag_k * k_val), self.emb_ent_k.weight.transpose(1, 0))

        score = real_score + i_score + j_score + k_score
        return torch.sigmoid(score)


class ConvO(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'ConvO'
        self.loss = torch.nn.BCELoss()
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

    def forward(self, e1_idx, rel_idx):
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
        score = e0_score + e1_score + e2_score + e3_score + e4_score + e5_score + e6_score + e7_score
        return torch.sigmoid(score)


class ComplEx(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'ComplEx'
        self.loss = torch.nn.BCELoss()
        # Init Embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(args.num_entities, args.embedding_dim)  # imaginary i
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(args.num_relations, args.embedding_dim)  # imaginary i
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data), xavier_normal_(self.emb_rel_i.weight.data)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_ent_i = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_i = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(args.embedding_dim)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()

    def forward(self, e1_idx, rel_idx):
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        emb_head_i = self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i(e1_idx)))

        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(self.emb_rel_i(rel_idx)))

        real_real_real = torch.mm(emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(emb_head_real * emb_rel_i, self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(emb_head_i * emb_rel_real, self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(emb_head_i * emb_rel_i, self.emb_ent_real.weight.transpose(1, 0))

        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        return torch.sigmoid(score)


class DistMult(BaseKGE):
    def __init__(self, args):
        super().__init__()
        self.name = 'DistMult'
        self.loss = torch.nn.BCELoss()
        # Init Embeddings
        self.embedding_dim = args.embedding_dim
        self.emb_ent_real = nn.Embedding(args.num_entities, args.embedding_dim)  # real
        self.emb_rel_real = nn.Embedding(args.num_relations, args.embedding_dim)  # real
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(args.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(args.input_dropout_rate)
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(args.embedding_dim)

    def get_embeddings(self):
        return self.emb_ent_real.weight.data.data.detach().numpy(), self.emb_rel_real.weight.data.detach().numpy()

    def forward(self, e1_idx, rel_idx):
        # (1)
        # (1.1) Real embeddings of head entities
        emb_head_real = self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real(e1_idx)))
        # (1.2) Real embeddings of relations
        emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(self.emb_rel_real(rel_idx)))
        real_score = torch.mm(emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        score = real_score
        return torch.sigmoid(score)