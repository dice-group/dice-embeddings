import warnings

warnings.simplefilter("ignore", UserWarning)
from dataset import KG, StandardDataModule, KvsAll
import torch
from torch import nn
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
        """
        if self.args.num_folds_for_cv < 2:
            self.logger.info(
                f'k-fold cross-validation requires at least one train/test split, but got only ***num_folds_for_cv*** => {args.num_folds_for_cv}.num_folds_for_cv is now set to 10.')
            self.args.num_folds_for_cv = 10
        """

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

    def only_train(self) -> pl.LightningModule:
        """

        :return:
        """
        train_set = np.array(self.dataset.train_set)
        trainer = pl.Trainer.from_argparse_args(self.args)
        model, form_of_labelling = select_model(self.args)
        train_dataset_loader = DataLoader(KvsAll(train_set, entity_idxs=self.dataset.entity_to_idx,
                                                 relation_idxs=self.dataset.relation_to_idx,
                                                 form=form_of_labelling),
                                          batch_size=self.args.batch_size, shuffle=True,
                                          num_workers=self.args.num_workers, drop_last=False)
        trainer.fit(model, train_dataloader=train_dataset_loader)

        return model

    def start(self):
        if self.args.batch_size > len(self.dataset.train_set):
            self.args.batch_size = len(self.dataset.train_set)

        if self.dataset.is_valid_test_available():
            trained_model = self.standard_training()
        else:
            if self.args.num_folds_for_cv < 2:
                self.logger.info(
                    f'No test set is found and k-fold cross-validation is set to less than 2 (***num_folds_for_cv*** => {self.args.num_folds_for_cv}). Hence we do not evaluate the model')
                trained_model = self.only_train()

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
