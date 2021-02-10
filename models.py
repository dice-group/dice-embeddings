from dataset import KvsAll, RelationPredictionDataset

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.metrics.functional import accuracy
from typing import List, Any

class Shallom(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.name = 'Shallom'
        shallom_width = int(args.shallom_width_ratio_of_emb * args.embedding_dim)
        self.loss = torch.nn.BCELoss()

        self.entity_embeddings = nn.Embedding(args.num_entities, args.embedding_dim)
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)

        self.shallom = nn.Sequential(nn.Dropout(0.1),
                                     torch.nn.Linear(args.embedding_dim * 2, shallom_width),
                                     nn.BatchNorm1d(shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
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
        self.log('avg_loss', avg_loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        predictions = self(x1_batch, x2_batch)
        val_loss = self.loss_function(predictions, y_batch)
        val_accuracy = accuracy(predictions, y_batch)
        return {'_val_acc.': val_accuracy, 'val_loss': val_loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        x = [[x['_val_acc.'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        test_accuracy = accuracy(self(x1_batch, x2_batch), y_batch)
        return {'test_accuracy': test_accuracy}

    def test_epoch_end(self, outputs: List[Any]):
        avg_test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('avg_test_accuracy', avg_test_accuracy, on_epoch=True,  prog_bar=True)
