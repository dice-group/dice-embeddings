import lightning as pl
import polars
from typing import Union
from dicee.models.base_model import BaseKGE
from dicee.static_funcs import select_model
from dicee.weight_averaging import  SWA, SWAG, EMA, TWA
from dicee.callbacks import ASWA, Eval, KronE, PrintCallback, AccumulateEpochLossCallback, Perturb, PeriodicEvalCallback, LRScheduler
from dicee.dataset_classes import construct_dataset
from .torch_trainer import TorchTrainer
from .torch_trainer_ddp import TorchDDPTrainer
from .model_parallelism import TensorParallel
from ..models.ensemble import EnsembleKGE
from ..static_funcs import timeit
import os
import torch
import pandas as pd
import copy
from typing import List, Tuple
from ..knowledge_graph import KG
import numpy as np

def load_term_mapping(file_path=str):
    return polars.read_csv(file_path + ".csv")


def initialize_trainer(args, callbacks)->TorchTrainer | TensorParallel | TorchDDPTrainer | pl.Trainer:
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
        trainer= pl.Trainer(accelerator=kwargs.get("accelerator", "auto"),
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
                          barebones=False)
    else:
        print('Initializing TorchTrainer CPU Trainer...', end='\t')
        trainer = TorchTrainer(args, callbacks=callbacks)
    assert trainer is not None
    return trainer



def get_callbacks(args):
    callbacks = [
        pl.pytorch.callbacks.ModelSummary(),
        PrintCallback(),
        AccumulateEpochLossCallback(path=args.full_storage_path)
    ]
    if args.swa:
        print(f"Starting Stochastic Weight Averaging (SWA) at Epoch: {args.swa_start_epoch}")
        callbacks.append(SWA(swa_start_epoch=args.swa_start_epoch, lr_init=args.lr,
                            max_epochs=args.num_epochs, swa_c_epochs= args.swa_c_epochs))
    elif args.swag:
        print(f"Starting Stochastic Weight Averaging-Gaussian (SWA-G) at Epoch: {args.swa_start_epoch}")
        callbacks.append(SWAG(swa_start_epoch=args.swa_start_epoch, lr_init=args.lr,
                            max_epochs=args.num_epochs, swa_c_epochs= args.swa_c_epochs))
    elif args.ema:
        print(f"Starting Exponential Moving Average (EMA) at Epoch: {args.swa_start_epoch}")
        callbacks.append(EMA(ema_start_epoch=args.swa_start_epoch, max_epochs=args.num_epochs, ema_c_epochs= args.swa_c_epochs))
    elif args.twa:
        print(f"Starting Trainable Weight Averaging at Epoch: {args.swa_start_epoch}")
        callbacks.append(TWA(twa_start_epoch=args.swa_start_epoch,lr_init=args.lr, max_epochs=args.num_epochs, twa_c_epochs= args.swa_c_epochs))
    elif args.adaptive_swa:
        callbacks.append(ASWA(num_epochs=args.num_epochs, path=args.full_storage_path))
    elif args.adaptive_lr:
        callbacks.append(LRScheduler(adaptive_lr_config=args.adaptive_lr, total_epochs=args.num_epochs,
                        experiment_dir=args.full_storage_path, eta_max=args.lr))
    
    if args.eval_every_n_epochs > 0 or args.eval_at_epochs is not None:
        callbacks.append(PeriodicEvalCallback(experiment_path=args.full_storage_path, max_epochs=args.num_epochs,
                        eval_every_n_epoch=args.eval_every_n_epochs, eval_at_epochs=args.eval_at_epochs,
                        save_model_every_n_epoch=args.save_every_n_epochs, n_epochs_eval_model=args.n_epochs_eval_model))

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
                self.trainer.dataset.train_se = np.memmap(path_memory_map,
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
                                              entity_to_idx={v["entity"]:k for k,v in pd.read_csv(f"{path}/entity_to_idx.csv",index_col=0).to_dict(orient='index').items()},
                                              relation_to_idx={v["relation"]:k for k,v in pd.read_csv(f"{path}/relation_to_idx.csv",index_col=0).to_dict(orient='index').items()},
                                              form_of_labelling=self.trainer.form_of_labelling,
                                              scoring_technique=self.args.scoring_technique,
                                              neg_ratio=self.args.neg_ratio,
                                              label_smoothing_rate=self.args.label_smoothing_rate,
                                              byte_pair_encoding=self.args.byte_pair_encoding,
                                              block_size=self.args.block_size)


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
        # (1) Create Kfold data
        from sklearn.model_selection import KFold
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

            trainer.fit(model, train_dataloaders=self.init_dataloader(
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
