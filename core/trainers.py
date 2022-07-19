""" Custom Trainer Class

* DataParallelTrainer implements a trainer class as in pytorch lightning based on torch.nn.DataParallel

* DistributedDataParallelTrainer implements a trainer class based on torch.nn.parallel.DistributedDataParallel

Although DistributedDataParallel is faster than DataParallel, the former is more memory extensive.

"""
import torch
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import os
import tempfile


class AbstractTrainer:
    def __init__(self, args, callbacks):
        self.attributes = vars(args)
        self.callbacks = callbacks
        print(self.attributes)

    def __getattr__(self, attr):
        return self.attributes[attr]

    def on_fit_start(self, *args, **kwargs):
        """ """

        for c in self.callbacks:
            c.on_fit_start(*args, **kwargs)

    def on_fit_end(self, *args, **kwargs):
        """ """
        for c in self.callbacks:
            c.on_fit_end(*args, **kwargs)

    @staticmethod
    def save_checkpoint(path):
        print('no checkpoint saving')


class DataParallelTrainer(AbstractTrainer):
    """ A Trainer based on torch.nn.DataParallel (https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)"""

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
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
        self.on_fit_start(trainer=self, pl_module=model)
        self.model = model
        print(kwargs)
        dataset = kwargs['train_dataloaders'].dataset
        self.loss_function = model.loss_function
        self.optimizer = model.configure_optimizers()
        self.model = torch.nn.DataParallel(model)
        self.model.to(self.device)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.num_core,
                                                  collate_fn=dataset.collate_fn)

        num_total_batches = len(data_loader)
        print_period = max(num_total_batches // 10, 1)
        print(f'Number of batches for an epoch:{num_total_batches}\t printing period:{print_period}')
        for epoch in range(self.attributes['max_epochs']):
            epoch_loss = 0
            start_time = time.time()
            for i, z in enumerate(data_loader):
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                if len(z) == 3:
                    x_batch, y_idx, y_batch = z
                    # the data transfer should be overlapped by the kernel execution
                    x_batch, y_idx, y_batch = x_batch.to(self.device, non_blocking=True), y_idx.to(self.device,
                                                                                                   non_blocking=True), y_batch.to(
                        self.device,
                        non_blocking=True)
                    yhat_batch = self.model(x_batch, y_idx)
                elif len(z) == 2:
                    x_batch, y_batch = z
                    # the data transfer should be overlapped by the kernel execution
                    x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device,
                                                                                              non_blocking=True)
                    yhat_batch = self.model(x_batch)
                else:
                    raise ValueError(len(z))

                batch_loss = self.loss_function(yhat_batch, y_batch)

                epoch_loss += batch_loss.item()
                if i > 0 and i % print_period == 0:
                    print(
                        f"Batch:{i}\t avg. batch loss until now:\t{epoch_loss / i}\t TotalRuntime:{(time.time() - start_time) / 60:.3f} minutes")

                # Backward pass
                batch_loss.backward()
                # Adjust learning weights
                self.optimizer.step()
            print(f"Epoch took {(time.time() - start_time) / 60:.3f} minutes")
            if i > 0:
                print(f"{epoch} epoch: Average batch loss:{epoch_loss / i}")
            else:
                print(f"{epoch} epoch: Average batch loss:{epoch_loss:.3f}")

        self.on_fit_end(self, self.model)

    def compute_forward(self, z):
        if len(z) == 2:
            x_batch, y_batch = z
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            return self.model.forward(x=x_batch), y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            x_batch, y_idx_batch, y_batch = x_batch.to(self.device), y_idx_batch.to(self.device), y_batch.to(
                self.device)
            return self.model.forward(x=x_batch, y_idx=y_idx_batch), y_batch

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
    #optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(),optimizer_class=torch.optim.SGD, lr=lr )

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
