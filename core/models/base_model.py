import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy as accuracy
from typing import List, Any, Tuple
from torch.nn.init import xavier_normal_


class BaseKGE(pl.LightningModule):

    def __init__(self, learning_rate=.1):
        super().__init__()
        self.name = 'Not init'
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

    def forward_triples(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_triples function')

    def forward_k_vs_all(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_all function')

    def forward(self, x):
        if len(x) == 3:
            h, r, t = x[0], x[1], x[2]
            return self.forward_triples(h, r, t)
        elif len(x) == 2:
            h, y = x[0], x[1]
            # Note that y can be relation or tail entity.
            return self.forward_k_vs_all(h, y)
        else:
            raise ValueError('Not valid input')

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        pred_batch = self.forward(x_batch)
        train_loss = self.loss_function(pred_batch, y_batch)
        return {'loss': train_loss}

    # def training_epoch_end(self, outputs) -> None:
    #    """ DBpedia debugging removed."""
    #    #avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #    #self.log('avg_loss', avg_loss, on_epoch=False, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            h, r, t, y_batch = batch
            predictions = self.forward_triples(h, r, t)
        else:
            h, x, y_batch = batch[:, 0], batch[:, 1], batch[:, 2]
            predictions = self.forward_k_vs_all(h, x)

        val_loss = self.loss_function(predictions, y_batch)
        val_accuracy = accuracy(predictions, y_batch)
        return {'val_acc': val_accuracy, 'val_loss': val_loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        x = [[x['val_acc'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_loss_per_epoch', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc_per_epoch', avg_val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:
            h, r, t, y_batch = batch
            predictions = self.forward_triples(h, r, t)
        else:
            h, x, y_batch = batch[:, 0], batch[:, 1], batch[:, 2]
            predictions = self.forward_k_vs_all(h, x)
        test_accuracy = accuracy(predictions, y_batch)
        return {'test_accuracy': test_accuracy}

    def test_epoch_end(self, outputs: List[Any]):
        avg_test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('avg_test_accuracy', avg_test_accuracy, on_epoch=True, prog_bar=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass
