import torch
from ..abstracts import AbstractTrainer
from ..static_funcs_training import make_iterable_verbose

import os
import sys
import torch
import torch.nn as nn

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel
)

from torch.distributed._tensor import Shard

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)


from torch.distributed._tensor.device_mesh import init_device_mesh


class MP(AbstractTrainer):
    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)

    def get_ensemble(self):
        return self.models

    def fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        model, = args
        # (1) Run the fit the start callback.
        self.on_fit_start(self, model)
        # (2) Setup DDP.
        optimizer = model.configure_optimizers()
        num_gpus = torch.cuda.device_count()
        for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                        verbose=True, position=0, leave=True)):
            epoch_loss = 0
            num_of_batches = len(kwargs['train_dataloaders'])
            for i, (x_batch, y_batch) in enumerate(kwargs['train_dataloaders']):
                # Define a large batch into small batches
                x_splits = torch.chunk(x_batch, num_gpus)
                y_splits = torch.chunk(y_batch, num_gpus)

                # Forward pass. We need to paralelize it
                gpu_losses=[]
                for gpu_id, (x_split, y_split) in enumerate(zip(x_splits, y_splits)):
                    y_split = y_split.to(f"cuda:{gpu_id}")
                    h_emb, r_emb, t_emb = model.get_triple_representation(x_split)
                    h_emb, r_emb,t_emb = h_emb.pin_memory().to(f"cuda:{gpu_id}", non_blocking=True), r_emb.pin_memory().to(f"cuda:{gpu_id}", non_blocking=True), t_emb.pin_memory().to(f"cuda:{gpu_id}", non_blocking=True)
                    yhat = model.score(h_emb, r_emb, t_emb)
                    gpu_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(yhat, y_split).cpu())

                loss=sum(gpu_losses)/len(gpu_losses)

                loss.backward()
                batch_loss = loss.item() 
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                epoch_loss += batch_loss
                
                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")


    def extract_input_outputs(self, z: list):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        if len(z) == 2:
            x_batch, y_batch = z
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            #x_batch, y_batch = x_batch.to("cuda", non_blocking=True), y_batch.pin_memory().to("cuda", non_blocking=True)
            return x_batch, y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            #x_batch, y_batch,y_idx_batch = x_batch.pin_memory().to("cuda", non_blocking=True), y_batch.pin_memory().to("cuda", non_blocking=True),y_idx_batch.pin_memory().to("cuda", non_blocking=True)
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError('Unexpected batch shape..')

