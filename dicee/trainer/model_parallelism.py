import torch
from ..abstracts import AbstractTrainer
from ..static_funcs_training import make_iterable_verbose

import os
import sys
import torch
import torch.nn as nn

import torch.distributed as dist

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
        torch.distributed.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        torch.manual_seed(rank)
        model.param_init(model.entity_embeddings.weight.data), model.param_init(model.relation_embeddings.weight.data)

        device = torch.device(f'cuda:{rank}')
        model.to(device)
        optimizer = model.configure_optimizers()


        for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                        verbose=True, position=0, leave=True)):
            epoch_loss = 0
            num_of_batches = len(kwargs['train_dataloaders'])
            for i, z in enumerate(kwargs['train_dataloaders']):
                optimizer.zero_grad()

                inputs,targets=self.extract_input_outputs(z,device)
                
                yhats=model(inputs)   
                # https://github.com/pytorch/pytorch/issues/58005 bug in pytoch
                dist.all_reduce(yhats,op=dist.ReduceOp.SUM)
                
                loss = torch.nn.functional.binary_cross_entropy_with_logits(yhats, targets)
                loss.backward()

                batch_loss = loss.item()
                optimizer.step()
                epoch_loss +=batch_loss
                if rank==0 and hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")
        
        torch.distributed.destroy_process_group()
        self.on_fit_end(self, model)

    def extract_input_outputs(self, z: list,rank):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        if len(z) == 2:
            x_batch, y_batch = z
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x_batch, y_batch = x_batch.to(rank, non_blocking=True), y_batch.pin_memory().to(rank, non_blocking=True)
            return x_batch, y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            x_batch, y_batch,y_idx_batch = x_batch.pin_memory().to(rank, non_blocking=True), y_batch.pin_memory().to(rank, non_blocking=True),y_idx_batch.pin_memory().to(rank, non_blocking=True)
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError('Unexpected batch shape..')

