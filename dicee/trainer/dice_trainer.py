import argparse
import lightning as pl
import gc
from typing import Any, Dict, Optional, Union

from dicee.models.base_model import BaseKGE
from dicee.static_funcs import select_model
from dicee.callbacks import (
    ASWA,
    Eval,
    KronE,
    PrintCallback,
    AccumulateEpochLossCallback,
    Perturb,
)
from dicee.dataset_classes import construct_dataset, reload_dataset
from .torch_trainer import TorchTrainer
from .torch_trainer_ddp import TorchDDPTrainer
from ..static_funcs import timeit
import os
import torch
import pandas as pd
import copy
from typing import List, Tuple
from ..knowledge_graph import KG


def initialize_trainer(args: Dict[str, Any], callbacks: List[Any]) -> Any:
    """
    Initialize the trainer for knowledge graph embedding.

    This function initializes and returns a trainer object based on the specified training configuration.

    Parameters
    ----------
    args : dict
        A dictionary containing the training configuration parameters.
    callbacks : list
        A list of callback objects to be used during training.

    Returns
    -------
    Any
        An initialized trainer object based on the specified configuration.
    """
    if args.trainer == "torchCPUTrainer":
        print("Initializing TorchTrainer CPU Trainer...", end="\t")
        return TorchTrainer(args, callbacks=callbacks)
    elif args.trainer == "torchDDP":
        if torch.cuda.is_available():
            print("Initializing TorchDDPTrainer GPU", end="\t")
            return TorchDDPTrainer(args, callbacks=callbacks)
        else:
            print("Initializing TorchTrainer CPU Trainer", end="\t")
            return TorchTrainer(args, callbacks=callbacks)
    elif args.trainer == "PL":
        print("Initializing Pytorch-lightning Trainer", end="\t")
        kwargs = vars(args)
        # kwargs["callbacks"] = callbacks
        """
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        plugins: Optional[Union[_PLUGIN_INPUT, List[_PLUGIN_INPUT]]] = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,)
        """
        # @TODO: callbacks need to be ad
        return pl.Trainer(
            accelerator=kwargs.get("accelerator", "auto"),
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
            barebones=False
        )
    else:
        print("Initialize TorchTrainer CPU Trainer", end="\t")
        return TorchTrainer(args, callbacks=callbacks)


def get_callbacks(args: Dict[str, Any]) -> List[Any]:
    """
    Get a list of callback objects based on the specified training configuration.

    This function constructs and returns a list of callback objects to be used during training.

    Parameters
    ----------
    args : dict
        A dictionary containing the training configuration parameters.

    Returns
    -------
    list
        A list of callback objects.
    """
    callbacks = [
        pl.pytorch.callbacks.ModelSummary(),
        PrintCallback(),
        AccumulateEpochLossCallback(path=args.full_storage_path)
    ]
    if args.swa:
        callbacks.append(pl.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=args.lr, swa_epoch_start=1))
    elif args.adaptive_swa:
        callbacks.append(ASWA(num_epochs=args.num_epochs, path=args.full_storage_path))
    else:
        """No SWA or ASWA applied"""

    if isinstance(args.callbacks, list):
        return callbacks

    for k, v in args.callbacks.items():
        if k == "Perturb":
            callbacks.append(Perturb(**v))
        elif k == 'KronE':
            callbacks.append(KronE())
        elif k == "Eval":
            callbacks.append(
                Eval(path=args.full_storage_path, epoch_ratio=v.get("epoch_ratio"))
            )
        else:
            raise RuntimeError(f"Incorrect callback:{k}")
    return callbacks


class DICE_Trainer:
    """
    Implements a training framework for knowledge graph embedding models using [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html),
    supporting [multi-GPU](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and CPU training. This trainer can handle continual training scenarios and supports
    different forms of labeling and evaluation methods.

    Parameters
    ----------
    args : Namespace
        Command line arguments or configurations specifying training parameters and model settings.
    is_continual_training : bool
        Flag indicating whether the training session is part of a continual learning process.
    storage_path : str
        Path to the directory where training checkpoints and models are stored.
    evaluator : object, optional
        An evaluation object responsible for model evaluation. This can be any object that implements
        an `eval` method accepting model predictions and returning evaluation metrics.

    Attributes
    ----------
    report : dict
        A dictionary to store training reports and metrics.
    trainer : lightening.Trainer or None
        The PyTorch Lightning Trainer instance used for model training.
    form_of_labelling : str or None
        The form of labeling used during training, which can be "EntityPrediction", "RelationPrediction", or "Pyke".

    Methods
    -------
    continual_start()
        Initializes and starts the training process, including model loading and fitting.
    initialize_trainer(callbacks: List) -> lightening.Trainer
        Initializes a PyTorch Lightning Trainer instance with the specified callbacks.
    initialize_or_load_model()
        Initializes or loads a model for training based on the training configuration.
    initialize_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader
        Initializes a DataLoader for the given dataset.
    initialize_dataset(dataset: KG, form_of_labelling) -> torch.utils.data.Dataset
        Prepares and initializes a dataset for training.
    start(knowledge_graph: KG) -> Tuple[BaseKGE, str]
        Starts the training process for a given knowledge graph.
    k_fold_cross_validation(dataset) -> Tuple[BaseKGE, str]
        Performs K-fold cross-validation on the dataset and returns the trained model and form of labelling.
    """

    def __init__(
        self,
        args,
        is_continual_training: bool,
        storage_path: str,
        evaluator: Optional[object] = None,
    ):
        self.report = dict()
        self.args = args
        self.trainer = None
        self.is_continual_training = is_continual_training
        self.storage_path = storage_path
        # Required for CV.
        self.evaluator = evaluator
        self.form_of_labelling = None
        print(
            f"# of CPUs:{os.cpu_count()} | # of GPUs:{torch.cuda.device_count()} | # of CPUs for dataloader:{self.args.num_core}"
        )

        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))

    def continual_start(self):
        """
        Initializes and starts the training process, including model loading and fitting.
        This method is specifically designed for continual training scenarios.

        Returns
        -------
        model : BaseKGE
            The trained knowledge graph embedding model instance. `BaseKGE` is a placeholder
            for the actual model class, which should be a subclass of the base model class
            used in your framework.
        form_of_labelling : str
            The form of labeling used during the training. This can indicate the type of
            prediction task the model is trained for, such as "EntityPrediction",
            "RelationPrediction", or other custom labeling forms defined in your implementation.
        """

        self.trainer = self.initialize_trainer(callbacks=get_callbacks(self.args))
        model, form_of_labelling = self.initialize_or_load_model()
        assert form_of_labelling in ["EntityPrediction", "RelationPrediction", "Pyke"]
        assert self.args.scoring_technique in [
            "AllvsAll",
            "KvsSample",
            "1vsAll",
            "KvsAll",
            "NegSample",
        ]
        train_loader = self.initialize_dataloader(
            reload_dataset(
                path=self.storage_path,
                form_of_labelling=form_of_labelling,
                scoring_technique=self.args.scoring_technique,
                neg_ratio=self.args.neg_ratio,
                label_smoothing_rate=self.args.label_smoothing_rate,
            )
        )
        self.trainer.fit(model, train_dataloaders=train_loader)
        return model, form_of_labelling

    @timeit
    def initialize_trainer(self, callbacks: List) -> pl.Trainer:
        """
        Initializes a PyTorch Lightning Trainer instance.

        Parameters
        ----------
        callbacks : List
            A list of PyTorch Lightning callbacks to be used during training.

        Returns
        -------
        pl.Trainer
            The initialized PyTorch Lightning Trainer instance.
        """
        return initialize_trainer(self.args, callbacks)

    @timeit
    def initialize_or_load_model(self) -> Tuple[BaseKGE, str]:
        """
        Initializes or loads a knowledge graph embedding model based on the training configuration.
        This method decides whether to start training from scratch or to continue training from a
        previously saved model state, depending on the `is_continual_training` attribute.

        Returns
        -------
        model : BaseKGE
            The model instance that is either initialized from scratch or loaded from a saved state.
            `BaseKGE` is a generic placeholder for the actual model class, which is a subclass of the
            base knowledge graph embedding model class used in your implementation.
        form_of_labelling : str
            A string indicating the type of prediction task the model is configured for. Possible values
            include "EntityPrediction" and "RelationPrediction", which signify whether the model is
            trained to predict missing entities or relations in a knowledge graph. The actual values
            depend on the specific tasks supported by your implementation.

        Notes
        -----
        The method uses the `is_continual_training` attribute to determine if the model should be loaded
        from a saved state. If `is_continual_training` is True, the method attempts to load the model and its
        configuration from the specified `storage_path`. If `is_continual_training` is False or the model
        cannot be loaded, a new model instance is initialized.

        This method also sets the `form_of_labelling` attribute based on the model's configuration, which
        is used to inform downstream training and evaluation processes about the type of prediction task.
        """
        print("Initializing Model...", end="\t")
        model, form_of_labelling = select_model(
            vars(self.args), self.is_continual_training, self.storage_path
        )
        self.report["form_of_labelling"] = form_of_labelling
        assert form_of_labelling in ["EntityPrediction", "RelationPrediction"]
        return model, form_of_labelling

    @timeit
    def initialize_dataloader(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.DataLoader:
        """
        Initializes and returns a PyTorch DataLoader object for the given dataset.

        This DataLoader is configured based on the training arguments provided,
        including batch size, shuffle status, and the number of workers.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to be loaded into the DataLoader. This dataset should already
            be processed and ready for training or evaluation.

        Returns
        -------
        torch.utils.data.DataLoader
            A DataLoader instance ready for training or evaluation, configured with the
            appropriate batch size, shuffle setting, and number of workers.
        """
        print("Initializing Dataloader...", end="\t")
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=self.args.num_core,
            persistent_workers=False,
        )

    @timeit
    def initialize_dataset(
        self, dataset: KG, form_of_labelling: str
    ) -> torch.utils.data.Dataset:
        """
        Initializes and returns a dataset suitable for training or evaluation, based on the
        knowledge graph data and the specified form of labelling.

        Parameters
        ----------
        dataset : KG
            The knowledge graph data used to construct the dataset. This should include
            training, validation, and test sets along with any other necessary information
            like entity and relation mappings.
        form_of_labelling : str
            The form of labelling to be used for the dataset, indicating the prediction
            task (e.g., "EntityPrediction", "RelationPrediction").

        Returns
        -------
        torch.utils.data.Dataset
            A processed dataset ready for use with a PyTorch DataLoader, tailored to the
            specified form of labelling and containing all necessary data for training
            or evaluation.
        """
        print("Initializing Dataset...", end="\t")
        train_dataset = construct_dataset(
            train_set=dataset.train_set,
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
            byte_pair_encoding=self.args.byte_pair_encoding,
            block_size=self.args.block_size
        )
        if self.args.eval_model is None:
            del dataset.train_set
            gc.collect()
        return train_dataset

    def start(self, knowledge_graph: KG) -> Tuple[BaseKGE, str]:
        """
        Starts the training process for the selected model using the provided knowledge graph data.
        The method selects and trains the model based on the configuration specified in the arguments.

        Parameters
        ----------
        knowledge_graph : KG
            The knowledge graph data containing entities, relations, and triples, which will be used
            for training the model.

        Returns
        -------
        Tuple[BaseKGE, str]
            A tuple containing the trained model instance and the form of labelling used during
            training. The form of labelling indicates the type of prediction task.
        """
        print("------------------- Train -------------------")

        if self.args.num_folds_for_cv == 0:
            # Initialize Trainer
            self.trainer: Union[TorchTrainer, TorchDDPTrainer, pl.Trainer]
            self.trainer = self.initialize_trainer(callbacks=get_callbacks(self.args))
            # Initialize or load model
            model, form_of_labelling = self.initialize_or_load_model()
            self.trainer.evaluator = self.evaluator
            self.trainer.dataset = knowledge_graph
            self.trainer.form_of_labelling = form_of_labelling
            self.trainer.fit(
                model,
                train_dataloaders=self.initialize_dataloader(
                    self.initialize_dataset(knowledge_graph, form_of_labelling)
                ),
            )
            return model, form_of_labelling
        else:
            return self.k_fold_cross_validation(knowledge_graph)

    def k_fold_cross_validation(self, dataset: KG) -> Tuple[BaseKGE, str]:
        """
        Conducts K-fold cross-validation on the provided dataset to assess the performance
        of the model specified in the training arguments. The process involves partitioning
        the dataset into K distinct subsets, iteratively using one subset for testing and
        the remainder for training. The model's performance is evaluated on each test split
        to compute the Mean Reciprocal Rank (MRR) scores.

        Steps:
        1. The dataset is divided into K train and test splits.
        2. For each split:
        2.1. A trainer and model are initialized based on the provided configuration.
        2.2. The model is trained using the training portion of the split.
        2.3. The MRR score of the trained model is computed using the test portion of the split.
        3. The process aggregates the MRR scores across all splits to report the mean and standard deviation
        of the MRR, providing a comprehensive evaluation of the model's performance.

        Parameters
        ----------
        dataset : KG
            The dataset to be used for K-fold cross-validation. This dataset should include
            the triples (head entity, relation, tail entity) for the entire knowledge graph.

        Returns
        -------
        Tuple[BaseKGE, str]
            A tuple containing:
            - The trained model instance from the last fold of the cross-validation.
            - The form of labelling used during training, indicating the prediction task
            (e.g., "EntityPrediction", "RelationPrediction").

        Notes
        -----
        The function assumes the presence of a predefined number of folds (K) specified in
        the training arguments. It utilizes PyTorch Lightning for model training and evaluation,
        leveraging GPU acceleration if available. The final output includes the model trained
        on the last fold and a summary of the cross-validation performance metrics.
        """
        print(f"{self.args.num_folds_for_cv}-fold cross-validation")
        # (1) Create Kfold data
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.args.num_folds_for_cv, shuffle=True, random_state=1)
        model = None
        eval_folds = []
        form_of_labelling = None
        # (2) Iterate over (1)
        for ith, (train_index, test_index) in enumerate(kf.split(dataset.train_set)):
            # (2.1) Create a new copy for the callbacks
            args = copy.copy(self.args)
            trainer = initialize_trainer(args, get_callbacks(args))
            model, form_of_labelling = select_model(
                vars(args), self.is_continual_training, self.storage_path
            )
            print(f"{form_of_labelling} training starts: {model.name}")

            train_set_for_i_th_fold, test_set_for_i_th_fold = (
                dataset.train_set[train_index],
                dataset.train_set[test_index],
            )

            trainer.fit(
                model,
                train_dataloaders=self.initialize_dataloader(
                    construct_dataset(
                        train_set=train_set_for_i_th_fold,
                        entity_to_idx=dataset.entity_to_idx,
                        relation_to_idx=dataset.relation_to_idx,
                        form_of_labelling=form_of_labelling,
                        scoring_technique=self.args.scoring_technique,
                        neg_ratio=self.args.neg_ratio,
                        label_smoothing_rate=self.args.label_smoothing_rate,
                    )
                ),
            )

            res = self.evaluator.eval_with_data(
                dataset=dataset,
                trained_model=model,
                triple_idx=test_set_for_i_th_fold,
                form_of_labelling=form_of_labelling,
            )
            # res = self.evaluator.evaluate_lp_k_vs_all(model, test_set_for_i_th_fold, form_of_labelling=form_of_labelling)
            eval_folds.append([res["MRR"], res["H@1"], res["H@3"], res["H@10"]])
        eval_folds = pd.DataFrame(eval_folds, columns=["MRR", "H@1", "H@3", "H@10"])
        self.evaluator.report = eval_folds.to_dict()
        print(eval_folds)
        print(eval_folds.describe())
        # results = {'H@1': eval_folds['H@1'].mean(), 'H@3': eval_folds['H@3'].mean(), 'H@10': eval_folds['H@10'].mean(),
        #           'MRR': eval_folds['MRR'].mean()}
        # print(f'KFold Cross Validation Results: {results}')
        return model, form_of_labelling
