"""DICE Trainer module for knowledge graph embedding training.

Provides the DICE_Trainer class which supports multiple training backends
including PyTorch Lightning, DDP, and custom CPU/GPU trainers.
"""
import copy
import os
import time
from typing import List, Optional, Tuple, Union

import lightning as pl
import numpy as np
import pandas as pd
import polars
import torch

from dicee.callbacks import (
    AccumulateEpochLossCallback,
    Eval,
    KronE,
    LRScheduler,
    PeriodicEvalCallback,
    Perturb,
    PrintCallback,
)
from dicee.dataset_classes import construct_dataset
from dicee.knowledge_graph import KG
from dicee.models.base_model import BaseKGE
from dicee.static_funcs import save_numpy_ndarray, select_model, timeit
from dicee.weight_averaging import ASWA, EMA, SWA, SWAG, TWA

from ..models.ensemble import EnsembleKGE
from .model_parallelism import TensorParallel
from .torch_trainer import TorchTrainer
from .torch_trainer_ddp import TorchDDPTrainer

def load_term_mapping(file_path: str) -> polars.DataFrame:
    """Load term-to-index mapping from CSV file.

    Args:
        file_path: Base path without extension.

    Returns:
        Polars DataFrame containing the mapping.
    """
    return polars.read_csv(f"{file_path}.csv")


def initialize_trainer(
    args,
    callbacks: List
) -> Union[TorchTrainer, TensorParallel, TorchDDPTrainer, pl.Trainer]:
    """Initialize the appropriate trainer based on configuration.

    Args:
        args: Configuration arguments containing trainer type.
        callbacks: List of training callbacks.

    Returns:
        Initialized trainer instance.

    Raises:
        AssertionError: If trainer is None after initialization.
    """
    trainer: Optional[Union[TorchTrainer, TensorParallel, TorchDDPTrainer, pl.Trainer]] = None
    if args.trainer == 'torchCPUTrainer':
        print('Initializing TorchTrainer CPU Trainer...', end='\t')
        trainer = TorchTrainer(args, callbacks=callbacks)
    elif args.trainer == 'TP':
        print('Initializing TensorParallel...', end='\t')
        trainer= TensorParallel(args, callbacks=callbacks)
    elif args.trainer == 'torchDDP':
        assert torch.cuda.is_available()
        print('Initializing TorchDDPTrainer GPU', end='\t')
        trainer = TorchDDPTrainer(args, callbacks=callbacks)
    elif args.trainer == 'PL':
        print('Initializing Pytorch-lightning Trainer', end='\t')
        kwargs = {**vars(args), **(getattr(args, "pl_trainer_kwargs", {}) or {})}
        # NOTE: PyTorch Lightning Trainer has many optional parameters
        # See: https://lightning.ai/docs/pytorch/stable/common/trainer.html
        trainer = pl.Trainer(accelerator=kwargs.get("accelerator", "auto"),
                          strategy=kwargs.get("strategy", "auto"),
                          num_nodes=kwargs.get("num_nodes", 1),
                          precision=kwargs.get("precision", None),
                          logger=kwargs.get("logger", None),
                          callbacks=callbacks,
                          fast_dev_run=kwargs.get("fast_dev_run", False),
                          max_epochs=kwargs["num_epochs"],
                          min_epochs=kwargs["num_epochs"],
                          max_steps=kwargs.get("max_step", -1),
                          min_steps=kwargs.get("min_steps", None),
                          detect_anomaly=False,
                          barebones=False,
                          enable_checkpointing=not kwargs.get('disable_checkpointing', False))
    else:
        print('Initializing TorchTrainer CPU Trainer...', end='\t')
        trainer = TorchTrainer(args, callbacks=callbacks)
    assert trainer is not None
    return trainer


def get_callbacks(args) -> List:
    """Create list of callbacks based on configuration.

    Args:
        args: Configuration arguments.

    Returns:
        List of callback instances.
    """
    callbacks = [
        pl.pytorch.callbacks.ModelSummary(),
        PrintCallback(),
        AccumulateEpochLossCallback(path=args.full_storage_path)
    ]

    # Weight averaging callbacks (mutually exclusive)
    if args.swa:
        print(f"Starting Stochastic Weight Averaging (SWA) at Epoch: {args.swa_start_epoch}")
        callbacks.append(SWA(
            swa_start_epoch=args.swa_start_epoch,
            lr_init=args.lr,
            max_epochs=args.num_epochs,
            swa_c_epochs=args.swa_c_epochs
        ))
    elif args.swag:
        print(f"Starting Stochastic Weight Averaging-Gaussian (SWA-G) at Epoch: {args.swa_start_epoch}")
        callbacks.append(SWAG(
            swa_start_epoch=args.swa_start_epoch,
            lr_init=args.lr,
            max_epochs=args.num_epochs,
            swa_c_epochs=args.swa_c_epochs
        ))
    elif args.ema:
        print(f"Starting Exponential Moving Average (EMA) at Epoch: {args.swa_start_epoch}")
        callbacks.append(EMA(
            ema_start_epoch=args.swa_start_epoch,
            max_epochs=args.num_epochs,
            ema_c_epochs=args.swa_c_epochs
        ))
    elif args.twa:
        print(f"Starting Trainable Weight Averaging at Epoch: {args.swa_start_epoch}")
        callbacks.append(TWA(
            twa_start_epoch=args.swa_start_epoch,
            lr_init=args.lr,
            max_epochs=args.num_epochs,
            twa_c_epochs=args.swa_c_epochs
        ))
    elif args.adaptive_swa:
        callbacks.append(ASWA(num_epochs=args.num_epochs, path=args.full_storage_path))
    elif args.adaptive_lr:
        callbacks.append(LRScheduler(
            adaptive_lr_config=args.adaptive_lr,
            total_epochs=args.num_epochs,
            experiment_dir=args.full_storage_path,
            eta_max=args.lr
        ))

    # Periodic evaluation callback
    if args.eval_every_n_epochs > 0 or args.eval_at_epochs is not None:
        callbacks.append(PeriodicEvalCallback(
            experiment_path=args.full_storage_path,
            max_epochs=args.num_epochs,
            eval_every_n_epoch=args.eval_every_n_epochs,
            eval_at_epochs=args.eval_at_epochs,
            save_model_every_n_epoch=args.save_every_n_epochs,
            n_epochs_eval_model=args.n_epochs_eval_model
        ))

    if isinstance(args.callbacks, list):
        return callbacks

    for k, v in args.callbacks.items():
        if k == "Perturb":
            callbacks.append(Perturb(**v))
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

    def __init__(self, args, is_continual_training:bool, storage_path, evaluator=None):
        self.report = dict()
        self.args = args
        self.trainer = None
        self.is_continual_training = is_continual_training
        self.storage_path = storage_path
        # Required for CV.
        self.evaluator = evaluator
        self.form_of_labelling = None
        print(f'# of CPUs:{os.cpu_count()} |'
              f' # of GPUs:{torch.cuda.device_count()} |'
              f' # of CPUs for dataloader:{self.args.num_core}')
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))

    def continual_start(self,knowledge_graph):
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
        # TODO: Here we need to load memory pag
        self.trainer.evaluator = self.evaluator
        self.trainer.dataset = knowledge_graph
        self.trainer.form_of_labelling = form_of_labelling
        self.trainer.fit(model, train_dataloaders=self.init_dataloader(self.init_dataset()))
        return model, form_of_labelling

    @timeit
    def initialize_trainer(self, callbacks: List) -> pl.Trainer | TensorParallel | TorchTrainer | TorchDDPTrainer:
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
    def init_dataloader(self, dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        print('Initializing Dataloader...', end='\t')
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.batch_size,
                                           shuffle=True, collate_fn=dataset.collate_fn,
                                           num_workers=self.args.num_core, persistent_workers=False)

    @timeit
    def init_dataset(self) -> torch.utils.data.Dataset:
        print('Initializing Dataset...', end='\t')
        if isinstance(self.trainer.dataset,KG):
            # Create a memory map of training dataset to reduce the memory usage
            path_memory_map=self.trainer.dataset.path_for_serialization + '/memory_map_train_set.npy'
            if not os.path.exists(path_memory_map):
                train_set_shape=self.trainer.dataset.train_set.shape
                train_set_dtype=self.trainer.dataset.train_set.dtype
                path_memory_map=self.trainer.dataset.path_for_serialization + '/memory_map_train_set.npy'
                memmap_kg = np.memmap(path_memory_map, dtype=train_set_dtype, mode='w+', shape=train_set_shape)
                memmap_kg[:] = self.trainer.dataset.train_set[:]
                memmap_kg[:].flush()
                del memmap_kg
                self.trainer.dataset.train_set = np.memmap(path_memory_map,
                                                mode='r',
                                                dtype=train_set_dtype,
                                                shape=train_set_shape)
            train_dataset = construct_dataset(train_set=self.trainer.dataset.train_set,
                                              valid_set=self.trainer.dataset.valid_set,
                                              test_set=self.trainer.dataset.test_set,
                                              train_target_indices=self.trainer.dataset.train_target_indices,
                                              target_dim=self.trainer.dataset.target_dim,
                                              ordered_bpe_entities=self.trainer.dataset.ordered_bpe_entities,
                                              entity_to_idx=self.trainer.dataset.entity_to_idx,
                                              relation_to_idx=self.trainer.dataset.relation_to_idx,
                                              form_of_labelling=self.trainer.form_of_labelling,
                                              scoring_technique=self.args.scoring_technique,
                                              neg_ratio=self.args.neg_ratio,
                                              label_smoothing_rate=self.args.label_smoothing_rate,
                                              byte_pair_encoding=self.args.byte_pair_encoding,
                                              block_size=self.args.block_size)
        else:
            assert isinstance(self.trainer.dataset, np.memmap), ("Train dataset must be an instance of memmap. "
                                                                 f"Currently, {type(np.memmap)}!")
            if self.args.continual_learning:
                path = self.args.continual_learning
            else:
                path = self.args.path_to_store_single_run

            train_dataset = construct_dataset(train_set=self.trainer.dataset,
                                              valid_set=None,
                                              test_set=None,
                                              train_target_indices=None,
                                              target_dim=None,
                                              ordered_bpe_entities=None,
                                              entity_to_idx=pd.read_csv(f"{path}/entity_to_idx.csv",index_col=0),
                                              relation_to_idx=pd.read_csv(f"{path}/relation_to_idx.csv",index_col=0),
                                              form_of_labelling=self.trainer.form_of_labelling,
                                              scoring_technique=self.args.scoring_technique,
                                              neg_ratio=self.args.neg_ratio,
                                              label_smoothing_rate=self.args.label_smoothing_rate,
                                              byte_pair_encoding=self.args.byte_pair_encoding,
                                              block_size=self.args.block_size,
                                              seed=self.args.random_seed)


        return train_dataset

    def start(self, knowledge_graph: Union[KG,np.memmap]) -> Tuple[BaseKGE, str]:
        """
        Start the training

        (1) Initialize Trainer
        (2) Initialize or load a pretrained KGE model

        in DDP setup, we need to load the memory map of already read/index KG.
        """
        """ Train selected model via the selected training strategy """
        print('------------------- Train -------------------')
        assert isinstance(knowledge_graph, np.memmap) or isinstance(knowledge_graph, KG), \
            f"knowledge_graph must be an instance of KG or np.memmap. Currently {type(knowledge_graph)}"
        if self.args.num_folds_for_cv == 0:
            self.trainer: Union[TensorParallel, TorchTrainer, TorchDDPTrainer, pl.Trainer]
            self.trainer = self.initialize_trainer(callbacks=get_callbacks(self.args))
            model, form_of_labelling = self.initialize_or_load_model()
            self.trainer.evaluator = self.evaluator
            self.trainer.dataset = knowledge_graph
            self.trainer.form_of_labelling = form_of_labelling
            # TODO: Later, maybe we should write a callback to save the models in disk

            if isinstance(self.trainer, TensorParallel):
                assert isinstance(model, EnsembleKGE), type(model)

                model = self.trainer.fit(model, train_dataloaders=self.init_dataloader(self.init_dataset()))
                assert isinstance(model,EnsembleKGE)
            else:
                self.trainer.fit(model, train_dataloaders=self.init_dataloader(self.init_dataset()))


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
        merged_train_set = self._collect_cv_pool(dataset)
        relation_filter_ids = self._resolve_relation_filter_ids(
            relation_to_idx=dataset.relation_to_idx,
            substrings=getattr(self.args, 'cv_relation_filter_substrings', None),
        )
        relation_filtered_indices = self._select_relation_filtered_indices(merged_train_set, relation_filter_ids)

        args = copy.copy(self.args)
        _, form_of_labelling = select_model(vars(args), self.is_continual_training, self.storage_path)

        # Decide whether to run relation-filtered CV: if the user provided
        # `cv_relation_filter_substrings` and matching indices exist, run
        # relation-filtered CV which restricts test/val selection to those
        # relation-containing triples. Otherwise use the full merged pool.
        use_relation_filtered_cv = relation_filtered_indices.size > 0 and (
            getattr(self.args, 'cv_relation_filter_substrings', None) is not None
        )

        if use_relation_filtered_cv:
            print(f'CV predicate filter matched {relation_filtered_indices.size} triples.')
            cv_source = merged_train_set[relation_filtered_indices]
        else:
            cv_source = merged_train_set
        # Prepare a handy inverse relation mapping for readable logs
        try:
            relation_map = self._relation_mapping(dataset.relation_to_idx)
            inverse_relation_map = {v: k for k, v in relation_map.items()}
        except Exception:
            inverse_relation_map = {}

        # (1) Create Kfold data
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.args.num_folds_for_cv, shuffle=True, random_state=1)
        model = None
        eval_folds = []
        cv_models_dir = os.path.join(self.args.full_storage_path, 'k-models')
        os.makedirs(cv_models_dir, exist_ok=True)

        print(f'Starting {self.args.num_folds_for_cv}-fold CV. Output dir: {cv_models_dir}')
        print(f'Using relation-filtered CV: {bool(use_relation_filtered_cv)}; CV pool size: {len(cv_source)}')

        fold_splits = list(kf.split(cv_source))
        # (2) Iterate over (1)
        for ith, (train_index, test_index) in enumerate(fold_splits):
            fold_start_time = time.time()
            print('-' * 60)
            print(f'Starting fold {ith + 1}/{len(fold_splits)}')
            if relation_filtered_indices.size > 0:
                val_index = fold_splits[(ith + 1) % len(fold_splits)][1]
                held_out_rel_idx = np.unique(np.concatenate([test_index, val_index]))

                # Defensive bounds check: ensure indices from KFold are within
                # the range of relation_filtered_indices. If out-of-range
                # indices are found, log and ignore them so CV can continue.
                if held_out_rel_idx.size > 0:
                    max_idx = int(held_out_rel_idx.max())
                else:
                    max_idx = -1

                if max_idx >= relation_filtered_indices.size:
                    print(
                        'Warning: held_out_rel_idx contains values >= relation_filtered_indices.size',
                        f'(max held_out_rel_idx={max_idx}, relation_filtered_indices.size={relation_filtered_indices.size})'
                    )
                    print(f'len(cv_source)={len(cv_source)}; filtering out invalid held-out indices')
                    valid_mask = held_out_rel_idx < relation_filtered_indices.size
                    invalid = held_out_rel_idx[~valid_mask]
                    if invalid.size > 0:
                        print(f'Skipping {invalid.size} out-of-range indices (examples): {invalid[:5]}')
                    held_out_rel_idx = held_out_rel_idx[valid_mask]

                if held_out_rel_idx.size == 0:
                    held_out_indices = np.asarray([], dtype=int)
                else:
                    held_out_indices = relation_filtered_indices[held_out_rel_idx]

                train_mask = np.ones(len(merged_train_set), dtype=bool)
                if held_out_indices.size > 0:
                    train_mask[held_out_indices] = False
                train_set_for_i_th_fold = merged_train_set[train_mask]
                test_set_for_i_th_fold = merged_train_set[relation_filtered_indices[test_index]]
                val_set_for_i_th_fold = merged_train_set[relation_filtered_indices[val_index]]
            else:
                val_index = fold_splits[(ith + 1) % len(fold_splits)][1]
                train_set_for_i_th_fold = cv_source[train_index]
                test_set_for_i_th_fold = cv_source[test_index]
                val_set_for_i_th_fold = cv_source[val_index]
            # Log fold dataset sizes
            try:
                train_n = len(train_set_for_i_th_fold)
                test_n = len(test_set_for_i_th_fold)
                val_n = len(val_set_for_i_th_fold)
            except Exception:
                train_n = np.asarray(train_set_for_i_th_fold).shape[0]
                test_n = np.asarray(test_set_for_i_th_fold).shape[0]
                val_n = np.asarray(val_set_for_i_th_fold).shape[0]

            print(f'Fold {ith + 1}: train={train_n}, test={test_n}, val={val_n}')
            if use_relation_filtered_cv:
                # report unique relation ids in held-out (test+val)
                try:
                    rels = np.unique(np.concatenate([test_set_for_i_th_fold[:, 1], val_set_for_i_th_fold[:, 1]]))
                    rel_names = [inverse_relation_map.get(int(r), str(r)) for r in rels[:10]]
                    print(f'Fold {ith + 1}: held-out relations count={len(rels)}; examples={rel_names}')
                except Exception:
                    pass

            # (2.1) Create a new copy for the callbacks
            args = copy.copy(self.args)
            trainer = initialize_trainer(args, get_callbacks(args))
            model, form_of_labelling = select_model(vars(args), self.is_continual_training, self.storage_path)
            print(f'{form_of_labelling} training starts: {model.name}')

            # Save each fold split
            save_numpy_ndarray(data=train_set_for_i_th_fold, file_path=f'{cv_models_dir}/train_set_{ith}_fold.npy')
            save_numpy_ndarray(data=test_set_for_i_th_fold, file_path=f'{cv_models_dir}/test_set_{ith}_fold.npy')
            save_numpy_ndarray(data=val_set_for_i_th_fold, file_path=f'{cv_models_dir}/val_set_{ith}_fold.npy')

            trainer.fit(model, train_dataloaders=self.init_dataloader(
                construct_dataset(train_set=train_set_for_i_th_fold,
                                  entity_to_idx=dataset.entity_to_idx,
                                  relation_to_idx=dataset.relation_to_idx,
                                  form_of_labelling=form_of_labelling,
                                  scoring_technique=self.args.scoring_technique,
                                  neg_ratio=self.args.neg_ratio,
                                  label_smoothing_rate=self.args.label_smoothing_rate)))

            fold_model_path = os.path.join(cv_models_dir, f'model_fold_{ith + 1}.pt')
            torch.save(model.state_dict(), fold_model_path)

            if self.args.eval_model is not None:
                res = self.evaluator.eval_with_data(dataset=dataset, trained_model=model, triple_idx=test_set_for_i_th_fold,
                                                    form_of_labelling=form_of_labelling)
                fold_record = {
                    'MRR': res['MRR'],
                    'H@1': res['H@1'],
                    'H@3': res['H@3'],
                    'H@10': res['H@10'],
                }
                eval_folds.append(fold_record)

        if self.args.eval_model is not None:
            eval_folds = pd.DataFrame(eval_folds)
            self.evaluator.report = eval_folds.to_dict()
            print(eval_folds)
            print(eval_folds.describe())

            # Save results to csv
            eval_folds.to_csv(f'{self.args.full_storage_path}/kfold_results.csv', index=False)
            eval_folds.describe().to_csv(f'{self.args.full_storage_path}/kfold_result_stats.csv')
        # results = {'H@1': eval_folds['H@1'].mean(), 'H@3': eval_folds['H@3'].mean(), 'H@10': eval_folds['H@10'].mean(),
        #           'MRR': eval_folds['MRR'].mean()}
        # print(f'KFold Cross Validation Results: {results}')
        return model, form_of_labelling

    @staticmethod
    def _collect_cv_pool(dataset) -> np.ndarray:
        splits = [np.asarray(dataset.train_set)]
        if getattr(dataset, 'valid_set', None) is not None:
            splits.append(np.asarray(dataset.valid_set))
        if getattr(dataset, 'test_set', None) is not None:
            splits.append(np.asarray(dataset.test_set))
        if len(splits) == 1:
            return splits[0]
        return np.concatenate(splits, axis=0)

    @staticmethod
    def _relation_mapping(relation_to_idx) -> dict:
        if isinstance(relation_to_idx, dict):
            return relation_to_idx
        if isinstance(relation_to_idx, pd.DataFrame):
            if relation_to_idx.shape[1] == 1:
                relation_col = relation_to_idx.columns[0]
                return dict(zip(relation_to_idx[relation_col].tolist(), relation_to_idx.index.tolist()))
            if 'relation' in relation_to_idx.columns and 'index' in relation_to_idx.columns:
                return dict(zip(relation_to_idx['relation'].tolist(), relation_to_idx['index'].tolist()))
            if relation_to_idx.shape[1] >= 2:
                first_col, second_col = relation_to_idx.columns[:2]
                return dict(zip(relation_to_idx[second_col].tolist(), relation_to_idx[first_col].tolist()))
        if isinstance(relation_to_idx, polars.DataFrame):
            if 'relation' in relation_to_idx.columns and 'index' in relation_to_idx.columns:
                return dict(zip(relation_to_idx['relation'].to_list(), relation_to_idx['index'].to_list()))
            if len(relation_to_idx.columns) >= 2:
                first_col, second_col = relation_to_idx.columns[:2]
                return dict(zip(relation_to_idx[second_col].to_list(), relation_to_idx[first_col].to_list()))
        raise TypeError(f'Unsupported relation_to_idx type: {type(relation_to_idx)}')

    @classmethod
    def _resolve_relation_filter_ids(cls, relation_to_idx, substrings) -> List[int]:
        if not substrings:
            substrings = ['resistant_to', 'sensitive_to']
        mapping = cls._relation_mapping(relation_to_idx)
        lowered_substrings = tuple(str(item).lower() for item in substrings)
        return [relation_idx for relation, relation_idx in mapping.items()
                if any(substr in str(relation).lower() for substr in lowered_substrings)]

    @staticmethod
    def _select_relation_filtered_indices(merged_train_set: np.ndarray, relation_filter_ids: List[int]) -> np.ndarray:
        if not relation_filter_ids:
            return np.asarray([], dtype=np.int64)
        relation_column = np.asarray(merged_train_set)[:, 1]
        return np.flatnonzero(np.isin(relation_column, np.asarray(relation_filter_ids)))

    @staticmethod
    def _macro_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> dict:
        precision_scores = []
        recall_scores = []
        f1_scores = []

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        for label in labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            false_positive = np.sum((y_true != label) & (y_pred == label))
            false_negative = np.sum((y_true == label) & (y_pred != label))

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return {
            'PredicatePrecision': float(np.mean(precision_scores)) if precision_scores else 0.0,
            'PredicateRecall': float(np.mean(recall_scores)) if recall_scores else 0.0,
            'PredicateF1': float(np.mean(f1_scores)) if f1_scores else 0.0,
        }

    @classmethod
    def _compute_relation_prediction_macro_metrics(
        cls,
        *,
        model,
        triple_idx: np.ndarray,
        ee_vocab,
        relation_ids: List[int],
        batch_size: int,
    ) -> dict:
        if hasattr(ee_vocab, 'result'):
            ee_vocab = ee_vocab.result()

        y_true = []
        y_pred = []

        model.eval()
        with torch.no_grad():
            for i in range(0, len(triple_idx), batch_size):
                data_batch = triple_idx[i:i + batch_size]
                e1_idx_e2_idx = torch.LongTensor(data_batch[:, [0, 2]])
                r_idx = torch.LongTensor(data_batch[:, 1])

                predictions = model.forward_k_vs_all(x=e1_idx_e2_idx)

                for j in range(data_batch.shape[0]):
                    filt = ee_vocab[(int(data_batch[j][0]), int(data_batch[j][2]))]
                    target_value = predictions[j, r_idx[j]].item()
                    predictions[j, filt] = -np.Inf
                    predictions[j, r_idx[j]] = target_value

                predicted_relation_idx = torch.argmax(predictions, dim=1).cpu().numpy()
                y_true.extend(r_idx.cpu().numpy().tolist())
                y_pred.extend(predicted_relation_idx.tolist())

        return cls._macro_precision_recall_f1(np.asarray(y_true), np.asarray(y_pred), relation_ids)
