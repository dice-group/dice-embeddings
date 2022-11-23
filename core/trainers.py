""" Custom Trainer Class

* DataParallelTrainer implements a trainer class as in pytorch lightning based on torch.nn.DataParallel

* DistributedDataParallelTrainer implements a trainer class based on torch.nn.parallel.DistributedDataParallel

Although DistributedDataParallel is faster than DataParallel, the former is more memory extensive.

"""
import torch
from tqdm import tqdm
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import os
import tempfile
from core.custom_opt.sls import Sls
from core.custom_opt.adam_sls import AdamSLS
from pytorch_lightning.callbacks import ModelSummary
import torch
import numpy as np
import json
import pytorch_lightning as pl
from core.typings import *
from core.models.base_model import BaseKGE
from core.callbacks import PrintCallback, KGESaveCallback, PseudoLabellingCallback, PolyakCallback
from pytorch_lightning.strategies import DDPStrategy
from core.dataset_classes import StandardDataModule
from core.helper_classes import LabelRelaxationLoss, BatchRelaxedvsAllLoss
from core.static_funcs import select_model, model_fitting
from core.abstracts import AbstractTrainer
import pandas as pd
from sklearn.model_selection import KFold


class DICE_Trainer:
    """
    DICE_Trainer implement
    1- PL training schema
    2- Pytorch Dataparalell
    3- Pytorch DistributedDataParallel
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

    def training_process(self) -> BaseKGE:
        """
        Training and evaluation procedure

        (1) Collect Callbacks to be used during training
        (2) Initialize Pytorch-lightning Trainer
        (3) Train a KGE modal via (2)
        (4) Eval trained model
        (5) Return trained model
        """
        self.executor.report['num_train_triples'] = len(self.executor.dataset.train_set)
        self.executor.report['num_entities'] = self.executor.dataset.num_entities
        self.executor.report['num_relations'] = self.executor.dataset.num_relations
        print('------------------- Train & Eval -------------------')
        # (1) Collect Callbacks to be used during training
        # (2) Initialize Trainer
        self.trainer = initialize_trainer(self.args, callbacks=get_callbacks(self.args), plugins=[])
        # (3) Use (2) to train a KGE model
        trained_model, form_of_labelling = self.train()
        # (5) Return trained model
        return trained_model, form_of_labelling

    def start(self):
        return self.training_process()

    def train(self):  # -> Tuple[BaseKGE, str]:
        """ Train selected model via the selected training strategy """
        print("Train selected model via the selected training strategy ")
        if self.args.num_folds_for_cv >= 2:
            return self.k_fold_cross_validation()
        else:
            if self.args.scoring_technique == 'NegSample':
                return self.training_negative_sampling()
            elif self.args.scoring_technique == 'KvsAll':
                return self.training_kvsall()
            elif self.args.scoring_technique == 'KvsSample':
                return self.training_KvsSample()
            elif self.args.scoring_technique == 'PvsAll':
                return self.training_PvsAll()
            elif self.args.scoring_technique == 'CCvsAll':
                return self.training_CCvsAll()
            elif self.args.scoring_technique == '1vsAll':
                return self.training_1vsall()
            elif self.args.scoring_technique == "BatchRelaxedKvsAll" or self.args.scoring_technique == "BatchRelaxed1vsAll":
                return self.train_relaxed_k_vs_all()
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
        print(f'KvsAll training starts: {model.name}')  # -labeling:{form_of_labelling}')
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
        if self.args.eval_model is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
        """
        # @TODO Model Calibration
        from laplace import Laplace
        from laplace.utils.subnetmask import ModuleNameSubnetMask
        from laplace.utils import ModuleNameSubnetMask
        from laplace import Laplace
        # No change in link prediciton results
        subnetwork_mask = ModuleNameSubnetMask(model, module_names=['emb_ent_real'])
        subnetwork_mask.select()
        subnetwork_indices = subnetwork_mask.indices
        la = Laplace(model, 'classification',
                     subset_of_weights='subnetwork',
                     hessian_structure='full',
                     subnetwork_indices=subnetwork_indices)
        # la.fit(dataset.train_dataloader())
        # la.optimize_prior_precision(method='CV', val_loader=dataset.val_dataloader())
        """

        return model, form_of_labelling

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
        if self.args.label_relaxation_rate:
            model.loss = LabelRelaxationLoss(alpha=self.args.label_relaxation_rate)
            # model.loss=LabelSmoothingLossCanonical()
        elif self.args.label_smoothing_rate:
            model.loss = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_rate)
        else:
            model.loss = torch.nn.CrossEntropyLoss()
        # (3) Train model
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        if self.args.eval_model is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def training_negative_sampling(self) -> pl.LightningModule:
        """
        Train models with Negative Sampling
        """
        assert self.args.neg_ratio > 0
        # (1) Select the model
        model, _ = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        form_of_labelling = 'NegativeSampling'
        print(f'Training starts: {model.name}-labeling:{form_of_labelling}')
        print('Creating training data...', end='\t')
        start_time = time.time()
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_core)
        print(f'Done ! {time.time() - start_time:.3f} seconds\n')
        # 3. Train model
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        if self.args.eval_model is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def train_relaxed_k_vs_all(self) -> pl.LightningModule:
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        print(f'{self.args.scoring_technique}training starts: {model.name}')  # -labeling:{form_of_labelling}')
        # 2. Create training data.)
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=self.args.scoring_technique,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_core,
                                     label_smoothing_rate=self.args.label_smoothing_rate)
        # 3. Train model.
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        if self.args.eval_model is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None

        model.loss = BatchRelaxedvsAllLoss()
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
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
        if self.args.eval_model is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
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
            # trainer = pl.Trainer.from_argparse_args(self.args)
            trainer = initialize_trainer(self.args, get_callbacks(self.args), plugins=[])
            model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
            print(f'{form_of_labelling} training starts: {model.name}')  # -labeling:{form_of_labelling}')

            train_set_for_i_th_fold, test_set_for_i_th_fold = self.dataset.train_set[train_index], \
                                                              self.dataset.train_set[
                                                                  test_index]

            dataset = StandardDataModule(train_set_idx=train_set_for_i_th_fold,
                                         entity_to_idx=self.dataset.entity_to_idx,
                                         relation_to_idx=self.dataset.relation_to_idx,
                                         form=form_of_labelling,
                                         neg_sample_ratio=self.args.neg_ratio,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_core)
            # 3. Train model
            train_dataloaders = dataset.train_dataloader()
            del dataset
            model_fitting(trainer=trainer, model=model, train_dataloaders=train_dataloaders)

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


class DataParallelTrainer(AbstractTrainer):
    """ A Trainer based on torch.nn.DataParallel (https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)"""

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.use_closure = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_function = None
        self.optimizer = None
        self.model = None
        self.is_global_zero = True
        torch.manual_seed(self.seed_for_computation)
        torch.cuda.manual_seed_all(self.seed_for_computation)

        print(self.attributes)

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args
        self.model = model
        self.model.to(self.device)
        self.on_fit_start(trainer=self, pl_module=self.model)
        dataset = kwargs['train_dataloaders'].dataset
        self.loss_function = model.loss_function
        # self.model = torch.nn.DataParallel(model)
        self.optimizer = self.model.configure_optimizers()

        if isinstance(self.optimizer, Sls) or isinstance(self.optimizer, AdamSLS):
            self.use_closure = True
        else:
            self.use_closure = False

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.num_core,
                                                  collate_fn=dataset.collate_fn)

        num_total_batches = len(data_loader)
        print_period = max(num_total_batches // 10, 1)
        print(f'Number of batches for an epoch:{num_total_batches}\t printing period:{print_period}')

        for epoch in (pbar := tqdm(range(self.attributes['max_epochs']))):
            epoch_loss = 0
            start_time = time.time()
            i: int
            batch: list
            for i, batch in enumerate(data_loader):
                # (1) Zero the gradients.
                self.optimizer.zero_grad()
                # (2) Extract Input and Outputs.
                x_batch, y_batch = self.extract_input_outputs(batch)
                # (3) Loss Forward and Backward w.r.t the batch.
                batch_loss = self.compute_forward_loss_backward(x_batch, y_batch)

                epoch_loss += batch_loss.item()
            epoch_loss /= num_total_batches
            pbar.set_postfix_str(
                f"{epoch + 1} epoch: Runtime: {(time.time() - start_time) / 60:.3f} mins \tEpoch loss: {epoch_loss:.8f}")
            self.model.loss_history.append(epoch_loss)
            # Fit on epochs e
            self.on_train_epoch_end(self, self.model)
            # Write a callback to store
            # print(self.optimizer.state['step_size'])

        self.on_fit_end(self, self.model)

    def compute_forward_loss_backward(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """ Compute the forward, loss and backward """
        if self.use_closure:
            batch_loss = self.optimizer.step(closure=lambda: self.loss_function(self.model(x_batch), y_batch))
            return batch_loss
        else:
            # (4) Backpropagate the gradient of (3) w.r.t. parameters.
            batch_loss = self.loss_function(self.model(x_batch), y_batch)
            # Backward pass
            batch_loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            return batch_loss

    def extract_input_outputs(self, z: list) -> tuple:
        """ Construct inputs and outputs from a batch of inputs with outputs From a batch of inputs and put """
        if len(z) == 2:
            x_batch, y_batch = z
            return x_batch.to(self.device), y_batch.to(self.device)
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            x_batch, y_idx_batch, y_batch = x_batch.to(self.device), y_idx_batch.to(self.device), y_batch.to(
                self.device)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            raise ValueError('Unexpected batch shape..')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group, nccl
    # gloo, mpi or ncclhttps://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


def distributed_training(rank: int, *args):
    """
    distributed_training is called as the entrypoint of the spawned process.
    This function must be defined at the top level of a module so it can be pickled and spawned.
    This is a requirement imposed by multiprocessing.

    The function is called as ``fn(i, *args)``, where ``i`` is the process index and ``args`` is the passed through tuple of arguments.
    """
    world_size, model, dataset, batch_size, max_epochs, lr = args
    print(f"Running basic DDP example on rank {rank}.")
    print(f"torch.utils.data.get_worker_info():{torch.utils.data.get_worker_info()}")
    print(f"torch.initial_seed():{torch.initial_seed()}")
    setup(rank, world_size)
    # Move the model to GPU with id rank
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=lr)
    # https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html
    # Note: ZeroRedundancy Increases the computation time quite a bit. DBpedia/10 => 3mins
    # Without ZeroReundancy optimizer we have 0.770 minutes
    # optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(),optimizer_class=torch.optim.SGD, lr=lr )

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    # worker_init_fn?
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=0,
                                              collate_fn=dataset.collate_fn,
                                              sampler=train_sampler)  # , pin_memory=False)
    num_total_batches = len(data_loader)
    print_period = max(num_total_batches // 10, 1)
    print(f'Number of batches for an epoch:{num_total_batches}\t printing period:{print_period}')
    for epoch in range(max_epochs):
        epoch_loss = 0
        start_time = time.time()
        for i, z in enumerate(data_loader):
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            x_batch, y_batch = z
            # the data transfer should be overlapped by the kernel execution
            x_batch, y_batch = x_batch.to(rank, non_blocking=True), y_batch.to(rank, non_blocking=True)
            yhat_batch = model(x_batch)
            batch_loss = loss_function(yhat_batch, y_batch)
            epoch_loss += batch_loss.item()
            if i > 0 and i % print_period == 0:
                print(
                    f"Batch:{i}\t avg. batch loss until now:\t{epoch_loss / i}\t TotalRuntime:{(time.time() - start_time) / 60:.3f} minutes")
            # Backward pass
            batch_loss.backward()
            # Adjust learning weights
            optimizer.step()

        print(f"Epoch took {(time.time() - start_time) / 60:.3f} minutes")
        if i > 0:
            print(f"{epoch} epoch: Average batch loss:{epoch_loss / i}")
        else:
            print(f"{epoch} epoch: Average batch loss:{epoch_loss}")

        if rank == 0:
            torch.save(ddp_model.module.state_dict(), "model.pt")


class DistributedDataParallelTrainer(AbstractTrainer):
    """ A Trainer based on torch.nn.parallel.DistributedDataParallel (https://pytorch.org/docs/stable/notes/ddp.html#ddp)"""

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.attributes = vars(args)
        self.callbacks = callbacks

        torch.manual_seed(self.seed_for_computation)
        torch.cuda.manual_seed_all(self.seed_for_computation)

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args

        # nodes * gpus
        world_size = self.num_nodes * torch.cuda.device_count()
        dataset = kwargs['train_dataloaders'].dataset
        print(model)
        mp.spawn(fn=distributed_training, args=(world_size, model, dataset, self.batch_size, self.max_epochs, self.lr),
                 nprocs=world_size,
                 join=True)

        model = model.load_state_dict(torch.load('model.pt'))
        os.remove('model.pt')
        self.model = model
        self.on_fit_end(self, self.model)

    def training_CCvsAll(self) -> BaseKGE:
        """ Conformal Credal Self-Supervised Learning for KGE
        D:= {(x,y)}, where
        x is an input is a head-entity & relation pair
        y is a one-hot vector
        """
        model, form_of_labelling = select_model(vars(self.args), self.executor.is_continual_training,
                                                self.executor.storage_path)
        print(f'Conformal Credal Self Training starts: {model.name}')
        # Split the training triples into train, calibration and unlabelled.
        train_set, calibration_set, unlabelled_set = semi_supervised_split(self.dataset.train_set,
                                                                           train_split_ratio=.3,
                                                                           calibration_split_ratio=.2)
        model.calibration_set = torch.LongTensor(calibration_set)
        model.unlabelled_set = torch.LongTensor(unlabelled_set)

        variant = 0
        non_conf_score_fn = non_conformity_score_diff  # or  non_conformity_score_prop
        print('Variant:', variant)
        print('non_conf_score_fn:', non_conf_score_fn)

        dataset = StandardDataModule(train_set_idx=train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form='CCvsAll',
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_core)

        def on_epoch_start(self, *args, **kwargs):
            """ Update non-conformity scores"""
            with torch.no_grad():
                # (1.1) Compute non-conformity scores on calibration dataset per epoch.
                self.non_conf_scores = non_conformity_score_diff(
                    torch.nn.functional.softmax(self.forward(self.calibration_set[:, [0, 1]])),
                    self.calibration_set[:, 2])

        setattr(BaseKGE, 'on_epoch_start', on_epoch_start)

        # Define a new raining set
        def training_step(self, batch, batch_idx):
            # (1) SUPERVISED PART
            # (1.1) Extract inputs and labels from a given batch (\mathcal{B}_l)
            x_batch, y_batch = batch
            # (1.2) Compute the supervised Loss
            train_loss = self.loss_function(yhat_batch=self.forward(x_batch), y_batch=y_batch)
            """
            # (1.3.2) Via KL divergence
            yhat = torch.clip(torch.softmax(logits_x, dim=-1), 1e-5, 1.)
            one_hot_targets = torch.clip(y_batch, 1e-5, 1.)
            train_loss = F.kl_div(yhat.log(), one_hot_targets, log_target=False, reduction='batchmean')
            """
            # (2) UNSUPERVISED PART
            # (2.1) Obtain unlabelled batch (\mathcal{B}_u), (x:=(s,p))
            unlabelled_input_batch = self.unlabelled_set[
                                         torch.randint(low=0, high=len(unlabelled_set), size=(len(x_batch),))][:,
                                     [0, 1]]
            # (2.2) Predict unlabelled batch \mathcal{B}_u
            with torch.no_grad():
                # TODO:Important moving this code outside of the no_grad improved the results a lot.
                # (2.2) Predict unlabelled batch \mathcal{B}_u
                pseudo_label = torch.nn.functional.softmax(self.forward(unlabelled_input_batch).detach())
                # (2.3) Construct p values given non-conformity scores and pseudo labels
                p_values = construct_p_values(self.non_conf_scores, pseudo_label.detach(),
                                              non_conf_score_fn=non_conformity_score_diff)
                # (2.4) Normalize (2.3)
                norm_p_values = norm_p_value(p_values, variant=0)

            unlabelled_loss = gen_lr(pseudo_label, norm_p_values)

            return train_loss + unlabelled_loss

        # Dynamically update
        setattr(BaseKGE, 'training_step', training_step)

        if self.args.eval_model is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=dataset.train_dataloader())
        return model, form_of_labelling

    def training_PvsAll(self) -> BaseKGE:
        """ Pseudo Labelling for KGE """

        # (1) Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        print(f'PvsAll training starts: {model.name}')
        train_set, calibration_set, unlabelled_set = semi_supervised_split(self.dataset.train_set,
                                                                           train_split_ratio=.5,
                                                                           calibration_split_ratio=.4)
        # Pseudo Labeling : unlabeled set consts of calibration_set, unlabelled_set
        model.unlabelled_set = torch.LongTensor(np.concatenate((calibration_set, unlabelled_set), axis=0))
        # (2) Create training data.
        dataset = StandardDataModule(train_set_idx=train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form='PvsAll',
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_core)

        # Define a new raining set
        def training_step(self, batch, batch_idx):
            # (1) SUPERVISED PART
            # (1.1) Extract inputs and labels from a given batch
            x_batch, y_batch = batch
            # (1.2) Predictions
            logits_x = self.forward(x_batch)
            # (1.3) Compute the supervised Loss
            # (1.3.1) Via Cross Entropy
            supervised_loss = self.loss_function(yhat_batch=logits_x, y_batch=y_batch)
            # (2) UNSUPERVISED PART
            # (2.1) Obtain unlabelled batch (\mathcal{B}_u)
            random_idx = torch.randint(low=0, high=len(self.unlabelled_set), size=(len(x_batch),))
            # (2.2) Batch of head entity and relation
            unlabelled_x = self.unlabelled_set[random_idx][:, [0, 1]]
            # (2.3) Create Pseudo-Labels
            with torch.no_grad():
                # (2.2) Compute loss
                _, max_pseudo_tail = torch.max(self.forward(unlabelled_x), dim=1)
                pseudo_labels = F.one_hot(max_pseudo_tail, num_classes=y_batch.shape[1]).float()

            unlabelled_loss = self.loss_function(yhat_batch=self.forward(unlabelled_x), y_batch=pseudo_labels)

            return supervised_loss + unlabelled_loss

        # Dynamically update
        setattr(BaseKGE, 'training_step', training_step)

        if self.args.eval_model is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=dataset.train_dataloader())
        return model, form_of_labelling


def initialize_trainer(args, callbacks: List, plugins: List) -> pl.Trainer:
    """ Initialize Trainer from input arguments """
    if args.torch_trainer == 'DataParallelTrainer':
        print('Initialize DataParallelTrainer Trainer')
        return DataParallelTrainer(args, callbacks=callbacks)
    elif args.torch_trainer == 'DistributedDataParallelTrainer':
        return DistributedDataParallelTrainer(args, callbacks=callbacks)
    else:
        print('Initialize Pytorch-lightning Trainer')
        # Pytest with PL problem https://github.com/pytest-dev/pytest/discussions/7995
        return pl.Trainer.from_argparse_args(args,
                                             strategy=DDPStrategy(find_unused_parameters=False),
                                             plugins=plugins, callbacks=callbacks)


# @TODO: Move the static
def get_callbacks(args):
    callbacks = [PrintCallback(),
                 KGESaveCallback(every_x_epoch=args.save_model_at_every_epoch,
                                 max_epochs=args.max_epochs,
                                 path=args.full_storage_path),
                 ModelSummary(max_depth=-1)]
    for i in args.callbacks:
        if i == 'Polyak':
            callbacks.append(PolyakCallback(max_epochs=args.max_epochs, path=args.full_storage_path))
    return callbacks
