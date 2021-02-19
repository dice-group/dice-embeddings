from dataset import KvsAll, RelationPredictionDataset

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.metrics.functional import accuracy
from typing import List, Any
from torch.nn.init import xavier_normal_


class Shallom(pl.LightningModule):
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

    def forward(self, s, o):
        emb_s, emb_o = self.entity_embeddings(s), self.entity_embeddings(o)
        return torch.sigmoid(self.shallom(torch.cat((emb_s, emb_o), 1)))

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

    def get_embeddings(self):
        return self.entity_embeddings.weight.data.detach().numpy()


class ConEx(pl.LightningModule):
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

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

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb.data.detach().numpy(), rel_emb.data.detach().numpy()
