import time
import pytorch_lightning as pl
import gc
from core.models.base_model import BaseKGE
from core.static_funcs import select_model
from core.callbacks import *
from core.dataset_classes import StandardDataModule
from .torch_trainer import TorchTrainer
from .torch_trainer_ddp import TorchDDPTrainer
import os
import torch
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
from core.helper_classes import LabelRelaxationLoss, BatchRelaxedvsAllLoss
import pandas as pd
from sklearn.model_selection import KFold
import copy
from typing import List, Tuple


def initialize_trainer(args, callbacks: List, plugins: List) -> pl.Trainer:
    """ Initialize Trainer from input arguments """
    if args.trainer == 'torchCPUTrainer':
        print('Initialize TorchTrainer CPU Trainer')
        return TorchTrainer(args, callbacks=callbacks)
    elif args.trainer == 'torchDDP':
        if torch.cuda.is_available():
            print('Initialize TorchDDPTrainer GPU')
            return TorchDDPTrainer(args, callbacks=callbacks)
        else:
            print('Initialize TorchTrainer CPU Trainer')
            return TorchTrainer(args, callbacks=callbacks)
    elif args.trainer == 'PL':
        print('Initialize Pytorch-lightning Trainer')
        # Pytest with PL problem https://github.com/pytest-dev/pytest/discussions/7995
        return pl.Trainer.from_argparse_args(args,
                                             strategy=DDPStrategy(find_unused_parameters=False))
    else:
        print('Initialize TorchTrainer CPU Trainer')
        return TorchTrainer(args, callbacks=callbacks)


def get_callbacks(args):
    callbacks = [PrintCallback(),
                 KGESaveCallback(every_x_epoch=args.save_model_at_every_epoch,
                                 max_epochs=args.max_epochs,
                                 path=args.full_storage_path),
                 AccumulateEpochLossCallback(path=args.full_storage_path)
                 ]
    for i in args.callbacks:
        if 'PPE' in i:
            if "PPE" == i:
                callbacks.append(
                    PPE(num_epochs=args.num_epochs, path=args.full_storage_path, last_percent_to_consider=None))
            elif len(i) > 3:
                name, param = i[:3], i[3:]
                assert name == 'PPE'
                assert int(param)
                callbacks.append(PPE(num_epochs=args.num_epochs,
                                     path=args.full_storage_path,
                                     last_percent_to_consider=int(param)))
            else:
                raise KeyError(f'Unexpected input for callbacks ***\t{i}\t***')
    return callbacks


class DICE_Trainer:
    """
    DICE_Trainer implement
    1- Pytorch Lightning trainer (https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
    2- Multi-GPU Trainer(https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
    3- CPU Trainer
    """

    def __init__(self, executor, evaluator=None):
        self.executor = executor
        self.report = dict()
        self.args = self.executor.args
        self.trainer = None
        self.dataset = self.executor.dataset
        self.is_continual_training = self.executor.is_continual_training
        self.storage_path = self.executor.storage_path
        # Required for CV.
        self.evaluator = evaluator
        print(
            f'# of CPUs:{os.cpu_count()} | # of GPUs:{torch.cuda.device_count()} | # of CPUs for dataloader:{self.args.num_core}')

        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))

    def old_start(self) -> Tuple[BaseKGE, str]:
        """ Start training process"""
        self.executor.report['num_train_triples'] = len(self.executor.dataset.train_set)
        self.executor.report['num_entities'] = self.executor.dataset.num_entities
        self.executor.report['num_relations'] = self.executor.dataset.num_relations
        print('------------------- Train -------------------')
        return self.train()

    def start(self) -> Tuple[BaseKGE, str]:
        """ Train selected model via the selected training strategy """
        self.executor.report['num_train_triples'] = len(self.executor.dataset.train_set)
        self.executor.report['num_entities'] = self.executor.dataset.num_entities
        self.executor.report['num_relations'] = self.executor.dataset.num_relations
        print('------------------- Train -------------------')

        # (1) Perform K-fold CV
        if self.args.num_folds_for_cv >= 2:
            return self.k_fold_cross_validation()
        else:
            # (2) Initialize Trainer.
            self.trainer = initialize_trainer(self.args, callbacks=get_callbacks(self.args), plugins=[])
            # (3) Select the training strategy.
            if self.args.scoring_technique == 'NegSample':
                return self.training_negative_sampling()
            elif self.args.scoring_technique == 'KvsAll':
                return self.training_kvsall()
            elif self.args.scoring_technique == 'KvsSample':
                return self.training_KvsSample()
            elif self.args.scoring_technique == '1vsAll':
                return self.training_1vsall()
            else:
                raise ValueError(f'Invalid argument: {self.args.scoring_technique}')

    def training_kvsall(self) -> BaseKGE:
        """
        Train models with KvsAll
        D= {(x,y)_i }_i ^n where
        1. x denotes a tuple of indexes of a head entity and a relation
        2. y denotes a vector of probabilities, y_j corresponds to probability of j.th indexed entity
        :return: trained BASEKGE
        """
        # (1) Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(vars(self.args), self.executor.is_continual_training,
                                                self.executor.storage_path)
        # (2) Create training data.
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_core,
                                     label_smoothing_rate=self.args.label_smoothing_rate)
        # (3) Train model.
        train_dataloaders = dataset.train_dataloader()
        del dataset
        self.release_mem()
        self.trainer.fit(model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def release_mem(self):
        if self.args.eval_model is None:
            del self.dataset, self.executor.dataset
            del self.evaluator, self.executor.evaluator
        gc.collect()

    def training_1vsall(self) -> BaseKGE:
        # (1) Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(vars(self.args), self.executor.is_continual_training,
                                                self.executor.storage_path)
        print(f'1vsAll training starts: {model.name}')
        # (2) Create training data.
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_core)
        if self.args.label_smoothing_rate:
            model.loss = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_rate)
        else:
            model.loss = torch.nn.CrossEntropyLoss()
        # (3) Train model
        train_dataloaders = dataset.train_dataloader()
        del dataset
        self.release_mem()
        self.trainer.fit(model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def training_negative_sampling(self):
        """
        Train models with Negative Sampling
        """
        assert self.args.neg_ratio > 0
        # (1) Select the model
        model, _ = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        del _
        form_of_labelling = 'NegativeSampling'
        print(f'Training starts: {model.name}-labeling:{form_of_labelling}')
        # 3. Train model
        train_dataloaders = StandardDataModule(train_set_idx=self.dataset.train_set,
                                               valid_set_idx=self.dataset.valid_set,
                                               test_set_idx=self.dataset.test_set,
                                               entity_to_idx=self.dataset.entity_to_idx,
                                               relation_to_idx=self.dataset.relation_to_idx,
                                               form=form_of_labelling,
                                               neg_sample_ratio=self.args.neg_ratio,
                                               batch_size=self.args.batch_size,
                                               num_workers=self.args.num_core,
                                               label_smoothing_rate=self.args.label_smoothing_rate).train_dataloader()

        self.release_mem()
        self.trainer.fit(model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def training_KvsSample(self) -> BaseKGE:
        """ A memory efficient variant of KvsAll training regime.

        Let D= {(x_i,y_i) }_i ^n where
        1. x denotes a tuple of indexes of a head entity and a relation
        2. y\in {0,1}^neg_sample_ratio

        Compared to KvsAll, KvsSample uses a subset of entities instead of using all entities.
        :return: trained BASEKGE
        """
        # (1) Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        form_of_labelling = 'KvsSample'
        print(f'KvsSample training starts: {model.name}')  # -labeling:{form_of_labelling}')
        # (2) Create training data.
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_core,
                                     label_smoothing_rate=self.args.label_smoothing_rate)
        # (3) Train model.
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        self.release_mem()
        self.trainer.fit(model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def k_fold_cross_validation(self) -> Tuple[BaseKGE, str]:
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
        print(f'{self.args.num_folds_for_cv}-fold cross-validation')
        kf = KFold(n_splits=self.args.num_folds_for_cv, shuffle=True, random_state=1)
        model = None
        eval_folds = []

        for (ith, (train_index, test_index)) in enumerate(kf.split(self.dataset.train_set)):
            # Need to create a new copy for the callbacks
            args = copy.copy(self.args)
            trainer = initialize_trainer(args, get_callbacks(args), plugins=[])
            model, form_of_labelling = select_model(vars(args), self.is_continual_training, self.storage_path)
            print(f'{form_of_labelling} training starts: {model.name}')

            train_set_for_i_th_fold, test_set_for_i_th_fold = self.dataset.train_set[train_index], \
                                                              self.dataset.train_set[
                                                                  test_index]

            dataset = StandardDataModule(train_set_idx=train_set_for_i_th_fold,
                                         entity_to_idx=self.dataset.entity_to_idx,
                                         relation_to_idx=self.dataset.relation_to_idx,
                                         form=form_of_labelling,
                                         neg_sample_ratio=self.args.neg_ratio,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_core,
                                         label_smoothing_rate=self.args.label_smoothing_rate)
            # 3. Train model
            train_dataloaders = dataset.train_dataloader()
            del dataset
            trainer.fit(model, train_dataloaders=train_dataloaders)

            res = self.evaluator.eval_with_data(model, test_set_for_i_th_fold, form_of_labelling=form_of_labelling)
            # res = self.evaluator.evaluate_lp_k_vs_all(model, test_set_for_i_th_fold, form_of_labelling=form_of_labelling)
            eval_folds.append([res['MRR'], res['H@1'], res['H@3'], res['H@10']])
        eval_folds = pd.DataFrame(eval_folds, columns=['MRR', 'H@1', 'H@3', 'H@10'])
        self.evaluator.report = eval_folds.to_dict()
        print(eval_folds)
        print(eval_folds.describe())
        # results = {'H@1': eval_folds['H@1'].mean(), 'H@3': eval_folds['H@3'].mean(), 'H@10': eval_folds['H@10'].mean(),
        #           'MRR': eval_folds['MRR'].mean()}
        # print(f'KFold Cross Validation Results: {results}')
        return model, form_of_labelling
