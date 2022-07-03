import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy as accuracy
from typing import List, Any, Tuple
from torch.nn.init import xavier_normal_
import numpy as np


class BaseKGE(pl.LightningModule):

    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.embedding_dim = None
        self.num_entities = None
        self.num_relations = None
        self.learning_rate = None
        self.apply_unit_norm = None
        self.input_dropout_rate = None
        self.hidden_dropout_rate = None
        self.optimizer_name = None
        self.feature_map_dropout_rate = None
        self.kernel_size = None
        self.num_of_output_channels = None
        self.weight_decay = None
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.selected_optimizer = None
        self.normalizer_class = None
        self.normalize_head_entity_embeddings = None
        self.normalize_relation_embeddings = None
        self.normalize_tail_entity_embeddings = None
        self.init_params_with_sanity_checking()

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data), xavier_normal_(self.relation_embeddings.weight.data)

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(self.input_dropout_rate)

    def init_params_with_sanity_checking(self):
        assert self.args['model'] in ['DistMult', 'ComplEx', 'QMult', 'OMult', 'ConvQ', 'ConvO',
                                      'ConEx', 'Shallom']
        if self.args.get('weight_decay'):
            self.weight_decay = self.args['weight_decay']
        else:
            self.weight_decay = 0.0
        if self.args.get('embedding_dim'):
            self.embedding_dim = self.args['embedding_dim']
        else:
            self.embedding_dim = 1

        if self.args.get('num_entities'):
            self.num_entities = self.args['num_entities']
        else:
            self.num_entities = 1

        if self.args.get('num_relations'):
            self.num_relations = self.args['num_relations']
        else:
            self.num_relations = 1

        if self.args.get('learning_rate'):
            self.learning_rate = self.args['learning_rate']
        else:
            self.learning_rate = .1

        if self.args.get("input_dropout_rate"):
            self.input_dropout_rate = self.args['input_dropout_rate']
        else:
            self.input_dropout_rate = 0.0
        if self.args.get("hidden_dropout_rate"):
            self.hidden_dropout_rate = self.args['hidden_dropout_rate']
        else:
            self.hidden_dropout_rate = 0.0

        if self.args['model'] in ['QMult', 'OMult', 'ConvQ', 'ConvO']:
            if self.args.get("apply_unit_norm"):
                self.apply_unit_norm = self.args['apply_unit_norm']
            else:
                self.apply_unit_norm = False

        if self.args['model'] in ['ConvQ', 'ConvO', 'ConEx']:
            if self.args.get("kernel_size"):
                self.kernel_size = self.args['kernel_size']
            else:
                self.kernel_size = 3
            if self.args.get("num_of_output_channels"):
                self.num_of_output_channels = self.args['num_of_output_channels']
            else:
                self.num_of_output_channels = 3
            if self.args.get("feature_map_dropout_rate"):
                self.feature_map_dropout_rate = self.args['feature_map_dropout_rate']
            else:
                self.feature_map_dropout_rate = 0.0

        if self.args.get("normalization") == 'LayerNorm':
            self.normalizer_class = torch.nn.LayerNorm
            self.normalize_head_entity_embeddings = self.normalizer_class(self.embedding_dim)
            self.normalize_relation_embeddings = self.normalizer_class(self.embedding_dim)
            if self.args['scoring_technique'] == 'NegSample':
                self.normalize_tail_entity_embeddings = self.normalizer_class(self.embedding_dim)
        elif self.args.get("normalization") == 'BatchNorm1d':
            # https://twitter.com/karpathy/status/1299921324333170689/photo/1
            # to decrease the memory usage.
            self.normalizer_class = torch.nn.BatchNorm1d
            self.normalize_head_entity_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
            self.normalize_relation_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
            if self.args['scoring_technique'] == 'NegSample':
                self.normalize_tail_entity_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
        else:
            raise NotImplementedError()

        if self.args.get("optim") in ['NAdam', 'Adam', 'SGD']:
            self.optimizer_name = self.args['optim']
        else:
            print(self.args)
            raise NotImplementedError()

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.entity_embeddings.weight.data.data.detach(), self.relation_embeddings.weight.data.detach()

    def configure_optimizers(self):

        if self.optimizer_name == 'SGD':
            self.selected_optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learning_rate,
                                                      momentum=0.05, dampening=0, weight_decay=self.weight_decay,
                                                      nesterov=False)
        elif self.optimizer_name == 'Adam':
            self.selected_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                                       weight_decay=self.weight_decay)

        elif self.optimizer_name == 'NAdam':
            self.selected_optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),
                                                        eps=1e-08, weight_decay=self.weight_decay, momentum_decay=0.004)
        else:
            raise KeyError()
        return self.selected_optimizer

    def loss_function(self, yhat_batch, y_batch):
        return self.loss(input=yhat_batch, target=y_batch)

    def forward_triples(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_triples function')

    def forward_k_vs_all(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_all function')

    def forward(self, x: torch.Tensor):
        """

        :param x:
        :return:
        """
        batch_size, dim = x.shape
        if dim == 3:
            return self.forward_triples(x)
        elif dim == 2:
            # h, y = x[0], x[1]
            # Note that y can be relation or tail entity.
            return self.forward_k_vs_all(x=x)
        else:
            raise ValueError('Not valid input')

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        yhat_batch = self.forward(x_batch)
        train_loss = self.loss_function(yhat_batch=yhat_batch, y_batch=y_batch)
        return train_loss

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

    def get_triple_representation(self, indexed_triple):
        # (1) Split input into indexes.
        idx_head_entity, idx_relation, idx_tail_entity = indexed_triple[:, 0], indexed_triple[:, 1], indexed_triple[:,
                                                                                                     2]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        tail_ent_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(idx_tail_entity))
        return head_ent_emb, rel_ent_emb, tail_ent_emb

    def get_head_relation_representation(self, indexed_triple):
        # (1) Split input into indexes.
        idx_head_entity, idx_relation = indexed_triple[:, 0], indexed_triple[:, 1]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return head_ent_emb, rel_ent_emb
