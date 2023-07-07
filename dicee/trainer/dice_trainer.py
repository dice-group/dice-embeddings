import time
import pytorch_lightning as pl
import gc
from typing import Union
from dicee.models.base_model import BaseKGE
from dicee.static_funcs import select_model
from dicee.callbacks import *
from dicee.dataset_classes import construct_dataset, reload_dataset
from .torch_trainer import TorchTrainer
from .torch_trainer_ddp import TorchDDPTrainer
from ..static_funcs import timeit
import os
import torch
import numpy as np
from pytorch_lightning.strategies import DDPStrategy
import pandas as pd
from sklearn.model_selection import KFold
import copy
from typing import List, Tuple
from ..knowledge_graph import KG

def initialize_trainer(args, callbacks):
    if args.trainer == 'torchCPUTrainer':
        print('Initializing TorchTrainer CPU Trainer...', end='\t')
        return TorchTrainer(args, callbacks=callbacks)
    elif args.trainer == 'torchDDP':
        if torch.cuda.is_available():
            print('Initializing TorchDDPTrainer GPU', end='\t')
            return TorchDDPTrainer(args, callbacks=callbacks)
        else:
            print('Initializing TorchTrainer CPU Trainer', end='\t')
            return TorchTrainer(args, callbacks=callbacks)
    elif args.trainer == 'PL':
        print('Initializing Pytorch-lightning Trainer', end='\t')
        # Pytest with PL problem https://github.com/pytest-dev/pytest/discussions/7995
        return pl.Trainer.from_argparse_args(args,
                                             strategy=DDPStrategy(find_unused_parameters=False))
    else:
        print('Initialize TorchTrainer CPU Trainer', end='\t')
        return TorchTrainer(args, callbacks=callbacks)


def get_callbacks(args):
    callbacks = [PrintCallback(),
                 KGESaveCallback(every_x_epoch=args.save_model_at_every_epoch,
                                 max_epochs=args.max_epochs,
                                 path=args.full_storage_path),
                 AccumulateEpochLossCallback(path=args.full_storage_path)
                 ]
    for i in args.callbacks:
        if i=='KronE':
            callbacks.append(KronE())
        elif i=='Search':
            callbacks.append(Search(num_epochs=args.num_epochs,embedding_dim=args.embedding_dim))
        # @TODO: Rename it
        elif i=='Eval':
            callbacks.append(Eval(path=args.full_storage_path))
        elif 'FPPE' in i:
            if i == 'FPPE':
                callbacks.append(
                    FPPE(num_epochs=args.num_epochs, path=args.full_storage_path, last_percent_to_consider=None))
            elif 'FPPE' == i[:4] and len(i) > 3:
                name, param = i[:4], i[4:]
                assert name == 'FPPE'
                assert int(param)
                callbacks.append(FPPE(num_epochs=args.num_epochs,
                                      path=args.full_storage_path,
                                      last_percent_to_consider=int(param)))
            else:
                raise KeyError(f'Unexpected input for callbacks ***\t{i}\t***')
        elif 'PPE' in i:
            if "PPE" == i:
                callbacks.append(
                    PPE(num_epochs=args.num_epochs, path=args.full_storage_path, last_percent_to_consider=None))
            elif 'PPE' == i[:3] and len(i) > 3:
                name, param = i[:3], i[3:]
                assert name == 'PPE'
                assert int(param)
                callbacks.append(PPE(num_epochs=args.num_epochs,
                                     path=args.full_storage_path,
                                     last_percent_to_consider=int(param)))
            else:
                raise KeyError(f'Unexpected input for callbacks ***\t{i}\t***')
        else:
            raise KeyError(f'Unexpected input for callbacks ***\t{i}\t***')

    return callbacks


class DICE_Trainer:
    """
   DICE_Trainer implement
    1- Pytorch Lightning trainer (https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
    2- Multi-GPU Trainer(https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
    3- CPU Trainer

    Parameter
    ---------
    args

    is_continual_training:bool

    storage_path:str

    evaluator:

    Returns
    -------
    report:dict
    """

    def __init__(self, args, is_continual_training, storage_path, evaluator=None,dataset=None):
        self.report = dict()
        self.args = args
        self.trainer = None
        self.is_continual_training = is_continual_training
        self.storage_path = storage_path
        # Required for CV.
        self.evaluator = evaluator
        self.form_of_labelling=None
        self.dataset=dataset
        print(
            f'# of CPUs:{os.cpu_count()} | # of GPUs:{torch.cuda.device_count()} | # of CPUs for dataloader:{self.args.num_core}')

        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))

    def continual_start(self):
        """
        (1) Initialize training.
        (2) Load model
        (3) Load trainer
        (3) Fit model

        Parameter
        ---------

        Returns
        -------
        model:
        form_of_labelling: str
        """

        self.trainer = self.initialize_trainer(callbacks=get_callbacks(self.args), plugins=[])
        model, form_of_labelling = self.initialize_or_load_model()
        assert form_of_labelling in ['EntityPrediction', 'RelationPrediction', 'Pyke']
        assert self.args.scoring_technique in ['KvsSample', '1vsAll', 'KvsAll', 'NegSample']
        train_loader = self.initialize_dataloader(
            reload_dataset(path=self.storage_path, form_of_labelling=form_of_labelling,
                           scoring_technique=self.args.scoring_technique,
                           neg_ratio=self.args.neg_ratio,
                           label_smoothing_rate=self.args.label_smoothing_rate))
        self.trainer.fit(model, train_dataloaders=train_loader)
        return model, form_of_labelling

    @timeit
    def initialize_trainer(self, callbacks: List, plugins: List) -> pl.Trainer:
        """ Initialize Trainer from input arguments """
        return initialize_trainer(self.args, callbacks)

    @timeit
    def initialize_or_load_model(self):
        print('Initializing Model...', end='\t')
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path,self.dataset)
        self.report['form_of_labelling'] = form_of_labelling
        assert form_of_labelling in ['EntityPrediction', 'RelationPrediction']
        return model, form_of_labelling

    @timeit
    def initialize_dataloader(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        print('Initializing Dataloader...', end='\t')
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.batch_size,
                                           shuffle=True, collate_fn=dataset.collate_fn,
                                           num_workers=self.args.num_core, persistent_workers=False)
    @timeit
    def initialize_dataset(self, dataset, form_of_labelling) -> torch.utils.data.Dataset:
        print('Initializing Dataset...', end='\t')
        train_dataset = construct_dataset(train_set=dataset.train_set,
                                          valid_set=dataset.valid_set,
                                          test_set=dataset.test_set,
                                          entity_to_idx=dataset.entity_to_idx,
                                          relation_to_idx=dataset.relation_to_idx,
                                          form_of_labelling=form_of_labelling,
                                          scoring_technique=self.args.scoring_technique,
                                          neg_ratio=self.args.neg_ratio,
                                          label_smoothing_rate=self.args.label_smoothing_rate)
        if self.args.eval_model is None:
            del dataset.train_set
            gc.collect()
        # pickle.PicklingError: memo id too large for LONG_BINPUT
        # torch.save(train_loader, self.storage_path + '/TrainDataloader.pth')
        # @TODO: SaveDataset
        return train_dataset

    def start(self, dataset:KG) -> Tuple[BaseKGE, str]:
        """ Train selected model via the selected training strategy """
        print('------------------- Train -------------------')
        # (1) Perform K-fold CV
        if self.args.num_folds_for_cv >= 2:
            return self.k_fold_cross_validation(dataset)
        else:
            self.trainer: Union[TorchTrainer, TorchDDPTrainer, pl.Trainer]
            self.trainer = self.initialize_trainer(callbacks=get_callbacks(self.args), plugins=[])
            model, form_of_labelling = self.initialize_or_load_model()
            self.trainer.evaluator=self.evaluator
            self.trainer.dataset = dataset
            self.trainer.form_of_labelling = form_of_labelling
            print(model)
            self.trainer.fit(model, train_dataloaders=self.initialize_dataloader(self.initialize_dataset(dataset, form_of_labelling)))
            return model, form_of_labelling

    def k_fold_cross_validation(self, dataset) -> Tuple[BaseKGE, str]:
        """
        Perform K-fold Cross-Validation

        1. Obtain K train and test splits.
        2. For each split,
            2.1 initialize trainer and model
            2.2. Train model with configuration provided in args.
            2.3. Compute the mean reciprocal rank (MRR) score of the model on the test respective split.
        3. Report the mean and average MRR .

        :param self:
        :param dataset:
        :return: model
        """
        print(f'{self.args.num_folds_for_cv}-fold cross-validation')
        # (1) Create Kfold data
        kf = KFold(n_splits=self.args.num_folds_for_cv, shuffle=True, random_state=1)
        model = None
        eval_folds = []
        # (2) Iterate over (1)
        for (ith, (train_index, test_index)) in enumerate(kf.split(dataset.train_set)):
            # (2.1) Create a new copy for the callbacks
            args = copy.copy(self.args)
            trainer = initialize_trainer(args, get_callbacks(args))
            model, form_of_labelling = select_model(vars(args), self.is_continual_training, self.storage_path)
            print(f'{form_of_labelling} training starts: {model.name}')

            train_set_for_i_th_fold, test_set_for_i_th_fold = dataset.train_set[train_index], dataset.train_set[
                test_index]

            trainer.fit(model, train_dataloaders=self.initialize_dataloader(
                construct_dataset(train_set=train_set_for_i_th_fold,
                                  entity_to_idx=dataset.entity_to_idx,
                                  relation_to_idx=dataset.relation_to_idx,
                                  form_of_labelling=form_of_labelling,
                                  scoring_technique=self.args.scoring_technique,
                                  neg_ratio=self.args.neg_ratio,
                                  label_smoothing_rate=self.args.label_smoothing_rate)))

            res = self.evaluator.eval_with_data(dataset=dataset, trained_model=model, triple_idx=test_set_for_i_th_fold,
                                                form_of_labelling=form_of_labelling)
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
