import os
import sys
import torch
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

import numpy as np
from dicee.abstracts import AbstractTrainer
from dicee.static_funcs_training import efficient_zero_grad
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.distributed import init_process_group, destroy_process_group


# DDP with gradiant accumulation https://gist.github.com/mcarilli/bf013d2d2f4b4dd21ade30c9b52d5e2e
def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


class TorchDDPTrainer(AbstractTrainer):
    """
        A Trainer based on torch.nn.parallel.DistributedDataParallel

        Arguments
       ----------
       train_set_idx
           Indexed triples for the training.
       entity_idxs
           mapping.
       relation_idxs
           mapping.
       form
           ?
       store
            ?
       label_smoothing_rate
            Using hard targets (0,1) drives weights to infinity.
            An outlier produces enormous gradients.

       Returns
       -------
       torch.utils.data.Dataset
       """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)

    def fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        model, = args
        # (1) Run the fit the start callback.
        self.on_fit_start(self, model)
        # (2) Setup DDP.
        torch.distributed.init_process_group(backend="nccl")
        train_dataset_loader = kwargs['train_dataloaders']
        # (1) Create DATA LOADER.
        train_dataset_loader = DataLoader(train_dataset_loader.dataset, batch_size=self.attributes.batch_size,
                                          pin_memory=True, shuffle=False, num_workers=self.attributes.num_core,
                                          persistent_workers=False,
                                          collate_fn=kwargs['train_dataloaders'].dataset.collate_fn,
                                          sampler=torch.utils.data.distributed.DistributedSampler(
                                              train_dataset_loader.dataset))

        # (2) Initialize OPTIMIZER.
        optimizer = model.configure_optimizers()
        # (3) Start NodeTrainer.
        NodeTrainer(model, train_dataset_loader, optimizer, self.callbacks, self.attributes.num_epochs).train()
        torch.distributed.destroy_process_group()
        self.on_fit_end(self, model)

    def old_fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        model, = args
        # (1) Run the fit the start callback.
        self.on_fit_start(self, model)
        # (2) Compute the world size nodes * gpus.
        world_size = self.attributes.num_nodes * torch.cuda.device_count()
        # @TODO: torchrun is required
        # https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html
        # https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py
        # (3) Spawn the function across processes. nprocs => 1 process for each GPU.
        mp.spawn(fn=distributed_training,
                 args=(world_size, model, kwargs['train_dataloaders'], self.callbacks, self.attributes),
                 nprocs=world_size,
                 join=True, )
        model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        os.remove('model.pt')
        self.on_fit_end(self, model)


class NodeTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 callbacks,
                 num_epochs: int) -> None:
        # (1) Local and Global Ranks. 
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        # (2) Send model to local trainer. (Check whether it is uncesseary as we wrap it with DDP
        self.model = model.to(self.local_rank)
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = self.model.loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        # (3) Wrap the model with DDP() along with GPU ID that model lives on.
        self.model = DDP(model, device_ids=[self.local_rank])
        self.num_epochs = num_epochs
        print_peak_memory("Max memory allocated after creating DDP local local_rank:", self.local_rank)
        print(f'Global Rank {self.global_rank}\t Local Rank:{self.local_rank}')
        print(self.model)
        print(self.optimizer)
        print(
                f'Global:{self.global_rank} | Local:{self.local_rank} | NumOfDataPoints:{len(self.train_dataset_loader.dataset)} | NumOfEpochs:{self.num_epochs} | LearningRate:{self.model.module.learning_rate} | BatchSize:{self.train_dataset_loader.batch_size} | EpochBatchsize:{len(self.train_dataset_loader)}')

        self.loss_history = []

    def _load_snapshot(self, snapshot_path):
        raise NotImplementedError

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        return batch_loss

    def extract_input_outputs(self, z: list):
        if len(z) == 2:
            x_batch, y_batch = z
            return x_batch.to(self.local_rank), y_batch.to(self.local_rank)
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            x_batch, y_idx_batch, y_batch = x_batch.to(self.local_rank), y_idx_batch.to(self.local_rank), y_batch.to(
                self.local_rank)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            raise ValueError('Unexpected batch shape..')

    def _run_epoch(self, epoch):
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        i = 0
        construct_mini_batch_time = None
        for i, z in enumerate(self.train_dataset_loader):
            source, targets = self.extract_input_outputs(z)
            start_time = time.time()
            if construct_mini_batch_time:
                construct_mini_batch_time = start_time - construct_mini_batch_time
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
            if True:  # self.local_rank == self.global_rank==0:
                if construct_mini_batch_time:
                    print(
                        f"Global:{self.global_rank} | Local:{self.local_rank} | Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}sec | BatchConst.:{construct_mini_batch_time:.2f}sec")
                else:
                    print(
                        f"Global:{self.global_rank} | Local:{self.local_rank} | Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}secs")
            construct_mini_batch_time = time.time()
        return epoch_loss / (i + 1)

    def train(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            epoch_loss = self._run_epoch(epoch)

            print(f"Epoch:{epoch + 1} | Loss:{epoch_loss:.8f} | Runtime:{(time.time() - start_time) / 60:.3f}mins")
            if True:#self.local_rank == self.global_rank == 0:
                #print(f"Epoch:{epoch + 1} | Loss:{epoch_loss:.8f} | Runtime:{(time.time() - start_time) / 60:.3f}mins")
                self.model.module.loss_history.append(epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(None, self.model.module)


def distributed_training(rank: int, world_size, model, train_dataset_loader, callbacks, args):
    """
    distributed_training is called as the entrypoint of the spawned process.
    This function must be defined at the top level of a module so it can be pickled and spawned.
    This is a requirement imposed by multiprocessing.
    args: dictionary
    callbacks:list of callback objects
    The function is called as ``fn(i, *args)``, where ``i`` is the process index and ``args`` is the passed through tuple of arguments.
    """
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    # (1) Create DATA LOADER.
    train_dataset_loader = DataLoader(train_dataset_loader.dataset, batch_size=args.batch_size,
                                      pin_memory=True, shuffle=False, num_workers=args.num_core,
                                      persistent_workers=False, collate_fn=train_dataset_loader.dataset.collate_fn,
                                      sampler=torch.utils.data.distributed.DistributedSampler(
                                          train_dataset_loader.dataset))

    # (2) Initialize OPTIMIZER.
    optimizer = model.configure_optimizers()
    # (3) Create a static DDB Trainer.
    trainer = DDPTrainer(model, train_dataset_loader, optimizer, rank, callbacks, args.num_epochs)
    trainer.train()
    if rank == 0:
        trainer.model.loss_history = trainer.loss_history
        torch.save(trainer.model.module.state_dict(), "model.pt")
    dist.destroy_process_group()


class DDPTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int, callbacks, num_epochs) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = self.model.loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        # (1) Wrap the model with DDP() along with GPU ID that model lives on.
        self.model = DDP(model, device_ids=[gpu_id])
        self.num_epochs = num_epochs
        print_peak_memory("Max memory allocated after creating DDP:", gpu_id)
        print('GPU:{self.gpu_id}')
        print(self.model)
        print(self.optimizer)
        print(
            f'NumOfDataPoints:{len(self.train_dataset_loader.dataset)} | NumOfEpochs:{self.num_epochs} | LearningRate:{self.model.module.learning_rate} | BatchSize:{self.train_dataset_loader.batch_size} | EpochBatchsize:{len(self.train_dataset_loader)}')

        self.loss_history = []

    def _run_batch(self, source, targets):
        # (1) Zero the gradients.
        # self.optimizer.zero_grad()
        efficient_zero_grad(self.model)
        output = self.model(source)
        loss = self.loss_func(output, targets)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        # @TODO: Tips to decrease mem usage
        #  https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        #  torch.cuda.empty_cache()
        return batch_loss

    def extract_input_outputs(self, z: list):
        if len(z) == 2:
            x_batch, y_batch = z
            return x_batch.to(self.gpu_id), y_batch.to(self.gpu_id)
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            x_batch, y_idx_batch, y_batch = x_batch.to(self.gpu_id), y_idx_batch.to(self.gpu_id), y_batch.to(
                self.gpu_id)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            raise ValueError('Unexpected batch shape..')

    def _run_epoch(self, epoch):
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        i = 0
        construct_mini_batch_time = None
        for i, z in enumerate(self.train_dataset_loader):
            source, targets = self.extract_input_outputs(z)
            start_time = time.time()
            if construct_mini_batch_time:
                construct_mini_batch_time = start_time - construct_mini_batch_time
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
            if self.gpu_id == 0:
                if construct_mini_batch_time:
                    print(
                        f"Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}sec | BatchConst.:{construct_mini_batch_time:.2f}sec")
                else:
                    print(
                        f"Epoch:{epoch + 1} | Batch:{i + 1} | Loss:{batch_loss} |ForwardBackwardUpdate:{(time.time() - start_time):.2f}secs")
            construct_mini_batch_time = time.time()
        return epoch_loss / (i + 1)

    def train(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            epoch_loss = self._run_epoch(epoch)
            if self.gpu_id == 0:
                print(f"Epoch:{epoch + 1} | Loss:{epoch_loss:.8f} | Runtime:{(time.time() - start_time) / 60:.3f}mins")
                self.model.module.loss_history.append(epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(None, self.model.module)
