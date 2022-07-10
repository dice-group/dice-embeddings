import torch
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()




class CustomTrainer:
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
        self.model = model
        print(kwargs)
        data_loader = kwargs['train_dataloaders']
        self.loss_function = model.loss_function
        self.optimizer = model.configure_optimizers()

        self.model = torch.nn.DataParallel(model)
        self.model.to(self.device)
        
        num_total_batches = len(data_loader)
        print_period=num_total_batches // 10
        print(f'Number of batches for an epoch:{num_total_batches}\t printing period:{print_period}')
        for epoch in range(self.attributes['max_epochs']):
            epoch_loss = 0
            start_time=time.time()
            for i, z in enumerate(data_loader):
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                x_batch, y_batch = z
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                yhat_batch = self.model(x_batch)#self.compute_forward(z)
                batch_loss = self.loss_function(yhat_batch, y_batch)

                epoch_loss += batch_loss.item()
                if i > 0 and i % print_period == 0:
                    print(f"Batch:{i}\t avg. batch loss until now:\t{epoch_loss / i}\t TotalRuntime:{(time.time()-start_time)/60:.3f} minutes")
        
                # Backward pass
                batch_loss.backward()
                # Adjust learning weights
                self.optimizer.step()
            print(f"Epoch took {(time.time()-start_time ) / 60:.3f} minutes")
            if i>0:
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

