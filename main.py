from argparse import ArgumentParser
from dataset import KG, KvsAllDataset
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import KFold


class Shallom(pl.LightningModule):
    def __init__(self, param):
        # @TODO Add self.width
        super().__init__()
        self.name = 'Shallom'
        self.param = param
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = self.param['num_entities']
        self.num_relations = self.param['num_relations']

        self.width = self.embedding_dim
        self.loss = torch.nn.BCELoss()

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.fc1 = torch.nn.Linear(self.embedding_dim * 2, self.width)
        self.fc2 = torch.nn.Linear(self.width, self.num_relations)
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)

    def forward(self, s, o):
        # @TODO add dropouts and batchnorms.
        emb_s = self.entity_embeddings(s)
        emb_o = self.entity_embeddings(o)
        x = torch.cat((emb_s, emb_o), 1)
        x = self.fc2(F.relu(self.fc1(x)))
        return torch.sigmoid(x)

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x1_batch, x2_batch, y_batch = batch
        return {'loss': self.loss_function(self(x1_batch, x2_batch), y_batch)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def sanity_checking_with_arguments(args):
    try:
        assert args.num_folds_for_cv >= 2
    except AssertionError:
        print(f'num_folds_for_cv must be greater or equal to two. Currently:{args.num_folds_for_cv}')
        raise
    try:
        assert not (args.kvsall is True and args.negative_sample_ratio > 0)
    except AssertionError:
        print(f'Training  strategy: If args.kvsall is TRUE, args.negative_sample_ratio must be 0'
              f'args.kvsall:{args.kvsall} and args.negative_sample_ratio:{args.negative_sample_ratio}.')
        raise
    try:
        assert os.path.isfile(args.path_dataset)
    except AssertionError:
        print(f'The file does not exist in {args.path_dataset}')
        raise

    return args


def initialize(args):
    dataset = KvsAllDataset(triples=KG(data_dir=args.path_dataset))

    if args.model == 'Shallom':
        model = Shallom(
            param={'embedding_dim': args.embedding_dim,
                   'num_entities': len(dataset.entities),
                   'num_relations': len(dataset.relations)})

        dataset.labelling(form='RelationPrediction')
    else:
        # @TODOs ConEx, QMult, OMult etc.
        raise ValueError

    start_training(model, dataset, args)


def start_training(model, dataset, args):
    kf = KFold(n_splits=args.num_folds_for_cv)
    for train_index, test_index in kf.split(dataset):
        k_fold_loader_training = DataLoader(dataset.create_fold(train_index), batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers)
        k_fold_loader_test = DataLoader(dataset.create_fold(test_index), batch_size=args.batch_size,
                                        num_workers=args.num_workers)

        trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=args.fast_dev_run)
        trainer.fit(model, train_dataloader=k_fold_loader_training)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--num_workers', default=2, help='Number of cpus used during loadingIncrease ')
    parser.add_argument('--kvsall', default=True)
    parser.add_argument('--negative_sample_ratio', default=0)
    parser.add_argument('--num_folds_for_cv', default=2)
    parser.add_argument('--batch_size', default=1024)
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--embedding_dim', default=50)
    parser.add_argument("--model", type=str, default='Shallom', help="Models:Shallom")
    parser.add_argument("--path_dataset", type=str, default='KGs/YAGO3-10/train.txt')
    initialize(sanity_checking_with_arguments(parser.parse_args()))
