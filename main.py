from argparse import ArgumentParser
from dataset import KG, KvsAll, RelationPredictionDataset
from pytorch_lightning.metrics import Accuracy

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.model_selection import KFold
from funcs import sanity_checking_with_arguments
import numpy as np


def k_fold_cv_training(model, dataset, args):
    kf = KFold(n_splits=args.num_folds_for_cv)
    print(f'KFold training with {args.num_folds_for_cv} folds starts')

    for train_index, test_index in kf.split(dataset):
        k_fold_loader_training = DataLoader(dataset.create_fold(train_index), batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers)
        k_fold_loader_test = DataLoader(dataset.create_fold(test_index), batch_size=args.batch_size,
                                        num_workers=args.num_workers)

        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, train_dataloader=k_fold_loader_training)
        trainer.test(model, k_fold_loader_test)


class Shallom(pl.LightningModule):
    def __init__(self, args, dataset):
        super().__init__()
        self.name = 'Shallom'
        self.acc = Accuracy()
        self.args = args
        self.dataset = dataset
        self.num_entities = len(self.dataset.entities)
        self.num_relations = len(self.dataset.relations)

        self.embedding_dim = self.args.embedding_dim
        self.shallom_width = int(self.args.shallom_width_ratio_of_emb * self.embedding_dim)
        self.loss = torch.nn.BCELoss()

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)

        self.shallom = nn.Sequential(nn.Dropout(0.1),
                                     torch.nn.Linear(self.embedding_dim * 2, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     torch.nn.Linear(self.shallom_width, self.num_relations))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

    def forward(self, s, o):
        emb_s, emb_o = self.entity_embeddings(s), self.entity_embeddings(o)
        return torch.sigmoid(self.shallom(torch.cat((emb_s, emb_o), 1)))

    def training_step(self, batch, batch_idx):
        x1_batch, x2_batch, y_batch = batch
        loss = self.loss_function(self(x1_batch, x2_batch), y_batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        valid_acc = self.acc(self(x1_batch, x2_batch), y_batch)
        return {'Validation Accuracy': valid_acc}

    def test_step(self, batch, batch_idx):
        # s,p,o => s,o predict relation.
        x1_batch, x2_batch, y_batch = batch
        test_acc = self.acc(self(x1_batch, x2_batch), y_batch)
        return {'Test Accuracy': test_acc}

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self) -> DataLoader:
        train_set = KvsAll(self.dataset.train, entity_idxs=self.dataset.entity_idxs,
                           relation_idxs=self.dataset.relation_idxs, form='RelationPrediction')

        return DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self) -> DataLoader:
        val = [[self.dataset.entity_idxs[s], self.dataset.relation_idxs[p], self.dataset.entity_idxs[o]] for s, p, o in
               self.dataset.valid]
        return DataLoader(RelationPredictionDataset(val, target_dim=self.num_relations),
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers)

    def test_dataloader(self) -> DataLoader:
        test = [[self.dataset.entity_idxs[s], self.dataset.relation_idxs[p], self.dataset.entity_idxs[o]] for s, p, o in
                self.dataset.test]
        return DataLoader(RelationPredictionDataset(test, target_dim=self.num_relations),
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers)


def initialize(args):
    dataset = KG(data_dir=args.path_dataset_folder)
    trainer = pl.Trainer.from_argparse_args(args)
    if args.model == 'Shallom':
        model = Shallom(args=args, dataset=dataset)
    else:
        # @TODOs ConEx, QMult, OMult etc.
        raise ValueError

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during loadingIncrease ')
    parser.add_argument('--kvsall', default=True)
    parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_folds_for_cv', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embedding_dim', type=int, default=25)
    parser.add_argument("--model", type=str, default='Shallom', help="Models:Shallom")
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.0, help='With of the hidden layer')
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS')
    initialize(sanity_checking_with_arguments(parser.parse_args()))
