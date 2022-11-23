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


def cleanup():
    dist.destroy_process_group()


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
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
        b_sz = len(next(iter(self.train_dataset_loader))[0])
        print(
            f"[GPU {self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Number of Batches per Epoch:{len(self.train_dataset_loader)}")
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        for i, (source, targets) in (pbar := tqdm(enumerate(self.train_dataset_loader))):
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            batch_loss = self._run_batch(source, targets)
            pbar.set_description_str(f"{epoch + 1}. epoch | {i + 1}.batch | Loss: {batch_loss:.8f}")
            epoch_loss += batch_loss
        return epoch_loss / len(self.train_data)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            start_time = time.time()
            epoch_loss = self._run_epoch(epoch)
            print(
                f"{epoch + 1} epoch: Runtime: {(time.time() - start_time) / 60:.3f} mins\tEpoch loss: {epoch_loss:.8f}")
            self.loss_history.append(epoch_loss)


def distributed_training(rank: int, world_size, model, train_dataset, batch_size, max_epochs, lr):
    """
    distributed_training is called as the entrypoint of the spawned process.
    This function must be defined at the top level of a module so it can be pickled and spawned.
    This is a requirement imposed by multiprocessing.

    The function is called as ``fn(i, *args)``, where ``i`` is the process index and ``args`` is the passed through tuple of arguments.
    """
    #
    ddp_setup(rank, world_size)

    print(f"Running basic DDP example on rank {rank}.")
    print(f"torch.utils.data.get_worker_info():{torch.utils.data.get_worker_info()}")
    print(f"torch.initial_seed():{torch.initial_seed()}")
    # (1)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                      sampler=torch.utils.data.distributed.DistributedSampler(train_dataset))

    # (2) Create Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainer = Trainer(model, train_dataset_loader, optimizer, rank)
    trainer.train(max_epochs)
    if rank == 0:
        torch.save(trainer.model.module.state_dict(), "model.pt")
        pd.Series(trainer.loss_history).to_csv('epoch_losses.csv')
    dist.destroy_process_group()
    """
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
                                              batch_size=batch_size,  #
                                              shuffle=False,
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
        """


class DistributedDataParallelTrainer(AbstractTrainer):
    """ A Trainer based on torch.nn.parallel.DistributedDataParallel (https://pytorch.org/docs/stable/notes/ddp.html#ddp)"""

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.attributes = vars(args)
        self.callbacks = callbacks
        self.model = None
        torch.manual_seed(self.seed_for_computation)
        torch.cuda.manual_seed_all(self.seed_for_computation)

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args

        # nodes * gpus
        world_size = self.num_nodes * torch.cuda.device_count()
        train_dataset = kwargs['train_dataloaders'].dataset
        if world_size == 0:
            print('#' * 10)
            print('Can not compute distributed computing')
            print('#' * 10)
            return
        # pickle
        # Save the model
        mp.spawn(fn=distributed_training,
                 args=(world_size, model, train_dataset, self.batch_size, self.max_epochs, self.lr),
                 nprocs=world_size,
                 join=True,  # ?
                 )
        model = model.load_state_dict(torch.load('model.pt'))
        os.remove('model.pt')
        self.model = model
        self.model.loss_history = pd.read_csv('epoch_losses.csv').values.tolist()
        self.on_fit_end(self, self.model)
