import pytorch_lightning as pl
import gc
from typing import Union
from dicee.models.base_model import BaseKGE
from dicee.static_funcs import select_model
from dicee.callbacks import (PPE, FPPE, Eval, KronE, PrintCallback, KGESaveCallback, AccumulateEpochLossCallback,
                             Perturb)
from dicee.dataset_classes import construct_dataset, reload_dataset
from .torch_trainer import TorchTrainer
from .torch_trainer_ddp import TorchDDPTrainer
from ..static_funcs import timeit
import os
import torch
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
        return pl.Trainer.from_argparse_args(args,
                                             callbacks=callbacks,
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
    if isinstance(args.callbacks, list):
        return callbacks
    for k, v in args.callbacks.items():
        if k == "Perturb":
            callbacks.append(Perturb(**v))
        elif k == 'FPP':
            callbacks.append(
                FPPE(num_epochs=args.num_epochs, path=args.full_storage_path,
                     last_percent_to_consider=v.get('last_percent_to_consider')))
        elif k == 'PPE':
            callbacks.append(
                PPE(num_epochs=args.num_epochs, path=args.full_storage_path,
                    last_percent_to_consider=v.get('last_percent_to_consider')))
        elif k == 'KronE':
            callbacks.append(KronE())
        elif k == 'Eval':
            callbacks.append(Eval(path=args.full_storage_path, epoch_ratio=v.get('epoch_ratio')))
        else:
            raise RuntimeError(f'Incorrect callback:{k}')
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

    def __init__(self, args, is_continual_training, storage_path, evaluator=None):
        self.report = dict()
        self.args = args
        self.trainer = None
        self.is_continual_training = is_continual_training
        self.storage_path = storage_path
        # Required for CV.
        self.evaluator = evaluator
        self.form_of_labelling = None
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

        self.trainer = self.initialize_trainer(callbacks=get_callbacks(self.args))
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
    def initialize_trainer(self, callbacks: List) -> pl.Trainer:
        """ Initialize Trainer from input arguments """
        return initialize_trainer(self.args, callbacks)

    @timeit
    def initialize_or_load_model(self):
        print('Initializing Model...', end='\t')
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
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
    def initialize_dataset(self, dataset: KG, form_of_labelling) -> torch.utils.data.Dataset:
        print('Initializing Dataset...', end='\t')
        train_dataset = construct_dataset(train_set=dataset.train_set,
                                          valid_set=dataset.valid_set,
                                          test_set=dataset.test_set,
                                          train_target_indices=dataset.train_target_indices,
                                          target_dim=dataset.target_dim,
                                          ordered_bpe_entities=dataset.ordered_bpe_entities,
                                          entity_to_idx=dataset.entity_to_idx,
                                          relation_to_idx=dataset.relation_to_idx,
                                          form_of_labelling=form_of_labelling,
                                          scoring_technique=self.args.scoring_technique,
                                          neg_ratio=self.args.neg_ratio,
                                          label_smoothing_rate=self.args.label_smoothing_rate,
                                          byte_pair_encoding=self.args.byte_pair_encoding)
        if self.args.eval_model is None:
            del dataset.train_set
            gc.collect()
        return train_dataset

    def start(self, knowledge_graph: KG) -> Tuple[BaseKGE, str]:
        """ Train selected model via the selected training strategy """
        print('------------------- Train -------------------')

        if self.args.num_folds_for_cv == 0:
            # Initialize Trainer
            self.trainer: Union[TorchTrainer, TorchDDPTrainer, pl.Trainer]
            self.trainer = self.initialize_trainer(callbacks=get_callbacks(self.args))
            # Initialize or load model
            model, form_of_labelling = self.initialize_or_load_model()
            self.trainer.evaluator = self.evaluator
            self.trainer.dataset = knowledge_graph
            self.trainer.form_of_labelling = form_of_labelling
            self.trainer.fit(model, train_dataloaders=self.initialize_dataloader(
                self.initialize_dataset(knowledge_graph, form_of_labelling)))
            return model, form_of_labelling
        else:
            return self.k_fold_cross_validation(knowledge_graph)

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
        form_of_labelling = None
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
