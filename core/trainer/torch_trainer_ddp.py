import torch
from tqdm import tqdm
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import os
import numpy as np
import pytorch_lightning as pl
from core.typings import *
from core.abstracts import AbstractTrainer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys

class TorchDDPTrainer(AbstractTrainer):
    """ A Trainer based on torch.nn.parallel.DistributedDataParallel (https://pytorch.org/docs/stable/notes/ddp.html#ddp)"""

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)

    def fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        model, = args
        # (1) Fit start.
        self.on_fit_start(trainer=self, pl_module=model)
        # nodes * gpus
        world_size = self.attributes.num_nodes * torch.cuda.device_count()
        train_dataset = kwargs['train_dataloaders'].dataset
        mp.spawn(fn=distributed_training,
                 args=(world_size, model, train_dataset, self.callbacks, self.attributes),
                 nprocs=world_size,
                 join=True,  # ?
                 )
        model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        os.remove('model.pt')
        self.on_fit_end(None, model)

def distributed_training(rank: int, world_size, model, train_dataset, callbacks, args):
    """
    distributed_training is called as the entrypoint of the spawned process.
    This function must be defined at the top level of a module so it can be pickled and spawned.
    This is a requirement imposed by multiprocessing.

    args: dictionary
    callbacks:list of callback objects
    The function is called as ``fn(i, *args)``, where ``i`` is the process index and ``args`` is the passed through tuple of arguments.
    """
    ddp_setup(rank, world_size)

    print(f"Running basic DDP example on rank {rank}.")
    print(f"torch.utils.data.get_worker_info():{torch.utils.data.get_worker_info()}")
    print(f"torch.initial_seed():{torch.initial_seed()}")
    # (1) Create DATA LOADER.
    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      pin_memory=True, shuffle=False,
                                      sampler=torch.utils.data.distributed.DistributedSampler(train_dataset))

    # (2) Initialize OPTIMIZER.
    optimizer = model.configure_optimizers()
    # (3) Create a static DDB Trainer.
    trainer = Trainer(model, train_dataset_loader, optimizer, rank, callbacks)
    trainer.train(args.num_epochs)
    if rank == 0:
        trainer.model.loss_history = trainer.loss_history
        torch.save(trainer.model.module.state_dict(), "model.pt")
    dist.destroy_process_group()
    """
    # Move the model to GPU with id rank
    # https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html
    # Note: ZeroRedundancy Increases the computation time quite a bit. DBpedia/10 => 3mins
    # Without ZeroReundancy optimizer we have 0.770 minutes
    # optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(),optimizer_class=torch.optim.SGD, lr=lr )
    """


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int, callbacks) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.model = DDP(model, device_ids=[gpu_id])
        print(self.model)
        self.loss_history = []

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        return batch_loss

    def _run_epoch(self, epoch):
        # b_sz = len(next(iter(self.train_dataset_loader))[0])
        # print(f"[GPU {self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Number of Batches per Epoch:{len(self.train_dataset_loader)}")
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        for i, (source, targets) in enumerate(self.train_dataset_loader):
            source, targets = source.to(self.gpu_id, non_blocking=True), targets.to(self.gpu_id, non_blocking=True)
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
        return epoch_loss / len(self.train_dataset_loader)

    def train(self, max_epochs: int):
        #         for epoch in (pbar := tqdm(range(self.attributes['max_epochs']), file=sys.stdout)):
        for epoch in (pbar := tqdm(range(max_epochs),file=sys.stdout)):
            start_time = time.time()
            epoch_loss = self._run_epoch(epoch)
            #print(f"{epoch + 1} epoch: Runtime: {(time.time() - start_time) / 60:.3f} min\tEpoch loss: {epoch_loss:.8f}")
            self.loss_history.append(epoch_loss)
            
            pbar.set_description(f'Epoch {epoch + 1}')
            pbar.set_postfix_str(
                f"runtime:{(time.time() - start_time) / 60:.3f}mins, loss={epoch_loss:.8f}")
            pbar.update(1)

            if self.gpu_id == 0:
                self.model.module.loss_history.append(epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(None, self.model.module)


def ddp_setup(rank: int, world_size: int):
    """ Setup for Distributed  Data Parallel
    world size total number of process in a group.
    rank is a unique identifier assigned to each process
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group, nccl
    # gloo, mpi or ncclhttps://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    dist.init_process_group(backend='nccl',  # NVIDIA Collection Communication Library
                            rank=rank,
                            world_size=world_size)
