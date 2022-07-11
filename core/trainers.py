import torch
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os


class CustomTrainer:
    """ Custom Trainer"""

    def __init__(self, args):
        self.attributes = vars(args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_function = None
        self.optimizer = None
        self.model = None
        torch.manual_seed(self.seed_for_computation)
        torch.cuda.manual_seed_all(self.seed_for_computation)

        print(self.attributes)

    def __getattr__(self, attr):
        return self.attributes[attr]

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args
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
                x_batch, y_batch = z
                # the data transfer should be overlapped by the kernel execution
                x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device,
                                                                                          non_blocking=True)
                yhat_batch = self.model(x_batch)
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
                print(f"{epoch} epoch: Average batch loss:{epoch_loss / i:.3f}")
            else:
                print(f"{epoch} epoch: Average batch loss:{epoch_loss:.3f}")

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

    @staticmethod
    def save_checkpoint(path):
        print('no checkpoint saving')


def distributed_training(rank, *args):
    """
    distributed_training is called as the entrypoint of the spawned process.
    This function must be defined at the top level of a module so it can be pickled and spawned.
    This is a requirement imposed by multiprocessing.

    The function is called as ``fn(i, *args)``, where ``i`` is the process index and ``args`` is the passed through tuple of arguments.
    """
    model, = args
    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    torch.distributed.init_process_group(backend='nccl', init_method=None, world_size=- 1, rank=- 1)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=world_size,
                                                              rank=rank,
                                                              shuffle=True,
                                                              seed=0,
                                                              drop_last=False)


class CustomDistributedTrainer:
    """ Custom Trainer"""

    def __init__(self, args):
        self.attributes = vars(args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_function = None
        self.optimizer = None
        self.model = None

    def __getattr__(self, attr):
        return self.attributes[attr]

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args

        # nodes * gpus
        world_size = 2 * 1
        """
        Args:
            fn (function): Function is called as the entrypoint of the
                spawned process. This function must be defined at the top
                level of a module so it can be pickled and spawned. This
                is a requirement imposed by multiprocessing.

                The function is called as ``fn(i, *args)``, where ``i`` is
                the process index and ``args`` is the passed through tuple
                of arguments.

            args (tuple): Arguments passed to ``fn``.
        """

        mp.spawn(fn=self.train, args=(world_size,), nprocs=world_size, join=True)

        self.model = model
        print(kwargs)
        data_loader = kwargs['train_dataloaders']
        self.loss_function = model.loss_function
        self.optimizer = model.configure_optimizers()

    @staticmethod
    def save_checkpoint(path):
        print('no checkpoint saving')
