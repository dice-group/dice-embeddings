import argparse
from dataset import KG, StandardDataModule, KvsAll, RelationPredictionDataset
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold
from static_funcs import *
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import json


class Execute:
    def __init__(self, args):
        sanity_checking_with_arguments(args)
        self.args = args
        self.dataset = KG(data_dir=args.path_dataset_folder, add_reciprical=args.add_reciprical)
        self.args.num_entities, self.args.num_relations = self.dataset.num_entities, self.dataset.num_relations
        self.storage_path = create_experiment_folder(folder_name=args.storage_path)
        self.logger = create_logger(name=self.args.model, p=self.storage_path)

    def standard_training(self):
        self.logger.info('\nTraining starts')
        trainer = pl.Trainer.from_argparse_args(self.args)
        model, form_of_labelling = select_model(self.args)
        dataset = StandardDataModule(dataset=self.dataset, form=form_of_labelling,
                                     batch_size=self.args.batch_size, num_workers=self.args.num_workers)

        trainer.fit(model, train_dataloader=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())
        trainer.test(model, test_dataloaders=dataset.test_dataloader())

        mrr = self.evaluate(model, self.dataset.test_set)
        self.logger.info(f"Raw MRR at testing => {mrr:.3f}")
        return model

    def k_fold_cross_validation(self) -> pl.LightningModule:
        """
        Perform K-fold Cross-Validation

        1. Obtain K train and test splits.
        2. For each split,
            2.1 initialize trainer and model
            2.2. Train model with configuration provided in args.
            2.3. Compute the mean reciprocal rank (MRR) score of the model on the test respective split.
        3. Report the mean and average MRR .

        :param self:
        :return: model
        """
        if self.args.num_folds_for_cv < 2:
            self.logger.info(
                f'k-fold cross-validation requires at least one train/test split, but got only ***num_folds_for_cv*** => {args.num_folds_for_cv}.num_folds_for_cv is now set to 10.')
            self.args.num_folds_for_cv = 10
        self.logger.info(f'{self.args.num_folds_for_cv}-fold cross-validation starts')
        kf = KFold(n_splits=self.args.num_folds_for_cv, shuffle=True)
        train_set = np.array(self.dataset.train_set)
        mrr_for_folds = []
        model = None
        for train_index, test_index in kf.split(train_set):
            trainer = pl.Trainer.from_argparse_args(self.args)
            model, form_of_labelling = select_model(self.args)

            train_set_for_i_th_fold, test_set_for_i_th_fold = train_set[train_index], train_set[test_index]

            train_dataset_loader = DataLoader(KvsAll(train_set_for_i_th_fold, entity_idxs=self.dataset.entity_to_idx,
                                                     relation_idxs=self.dataset.relation_to_idx,
                                                     form=form_of_labelling),
                                              batch_size=self.args.batch_size, shuffle=True,
                                              num_workers=self.args.num_workers, drop_last=True)
            trainer.fit(model, train_dataloader=train_dataset_loader)

            raw_mrr = self.evaluate(model, test_set_for_i_th_fold)
            mrr_for_folds.append(raw_mrr)

        mrr_for_folds = np.array(mrr_for_folds)
        self.logger.info(
            f'Mean and standard deviation of raw MRR in {self.args.num_folds_for_cv}-fold cross validation => {mrr_for_folds.mean():.3f}, {mrr_for_folds.std():.3f}')
        assert model is not None
        return model

    def evaluate(self, trained_model, triples):
        trained_model.eval()
        trained_model.cpu()

        if trained_model.name == 'Shallom':
            return compute_mrr_based_on_relation_ranking(trained_model, triples, self.dataset.entity_to_idx,
                                                         self.dataset.relations)
        else:
            return compute_mrr_based_on_entity_ranking(trained_model, triples, self.dataset.entity_to_idx,
                                                       self.dataset.relation_to_idx, self.dataset.entities)

    def start(self):
        if self.dataset.is_valid_test_available():
            trained_model = self.standard_training()
        else:
            trained_model = self.k_fold_cross_validation()

        self.store(trained_model)

    def store(self, trained_model):
        with open(self.storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            temp.pop('gpus')
            temp.pop('tpu_cores')
            json.dump(temp, file_descriptor)

        if trained_model.name == 'Shallom':
            entity_emb = trained_model.get_embeddings()
        else:
            entity_emb, relation_ebm = trained_model.get_embeddings()
            pd.DataFrame(relation_ebm, index=self.dataset.relations).to_csv(
                self.storage_path + '/' + trained_model.name + '_relation_embeddings.csv')
        pd.DataFrame(entity_emb, index=self.dataset.entities).to_csv(
            self.storage_path + '/' + trained_model.name + '_entity_embeddings.csv')


def argparse_default():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')
    parser.add_argument('--kvsall', default=True)
    parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_folds_for_cv', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=25)
    parser.add_argument('--input_dropout_rate', type=float, default=0.2)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.2)
    parser.add_argument("--model", type=str, default='Shallom',
                        help="Models:Shallom")
    parser.add_argument("--kernel_size", type=int, default=3, help="Square kernel size for ConEx")
    parser.add_argument("--num_of_output_channels", type=int, default=32,
                        help="Number of output channels for a convolution operation")
    parser.add_argument("--feature_map_dropout_rate", type=int, default=.3,
                        help="Dropout rate to be applied on feature map produced by a convolution operation")
    parser.add_argument("--max_num_epochs", type=int, default=10)
    parser.add_argument("--shallom_width_ratio_of_emb", type=float, default=1.5,
                        help='The ratio of the size of the first affine transformation with respect to size of the embeddings')
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS')
    parser.add_argument("--check_val_every_n_epochs", type=int, default=1000)
    parser.add_argument("--storage_path", type=str, default='DAIKIRI_Storage')
    parser.add_argument("--add_reciprical", type=bool, default=False)
    return parser


def preprocesses_input_args(arg):
    # To update the default value of Trainer in pytorch-lightnings
    arg.max_epochs = arg.max_num_epochs
    del arg.max_num_epochs
    arg.check_val_every_n_epoch = arg.check_val_every_n_epochs
    del arg.check_val_every_n_epochs

    arg.checkpoint_callback = False
    arg.logger = False
    return arg


if __name__ == '__main__':
    Execute(preprocesses_input_args(argparse_default().parse_args())).start()
