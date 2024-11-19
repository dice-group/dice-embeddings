import torch
from ..abstracts import AbstractTrainer
from ..static_funcs_training import make_iterable_verbose
import os
import sys
import torch.nn as nn
import torch.distributed as dist
from ..models.ensemble import EnsembleKGE
import copy

class MP(AbstractTrainer):
    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.models=[]

    def get_ensemble(self):
        return self.models
    
    def fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        seed_model, = args
        # () TODO: Send ensemble_model
        self.on_fit_start(self, seed_model)
        # () Init. ensemble model
        ensemble_model=EnsembleKGE(seed_model)
        # () Number of available GPUs
        assert torch.cuda.device_count()== len(ensemble_model)

        for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                        verbose=True, position=0, leave=True)):
            epoch_loss = 0
            num_of_batches = len(kwargs['train_dataloaders'])
            # Iterate over batches
            for i, z in enumerate(kwargs['train_dataloaders']):
                # Get the i-th batch of data points.
                x_batch, y_batch = self.extract_input_outputs(z)
                # Move the batch of labels into the master GPU : GPU-0 
                y_batch = y_batch.to("cuda:0")
                # Forward Pass on the batch. Yhat located on the master GPU.
                yhat = ensemble_model(x_batch)
                # Compute the loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat, y_batch)
                # Compute the gradient of the loss w.r.t. parameters.
                loss.backward()
                # Parameter update.
                ensemble_model.step()
                # Report he batch and epoch losses.
                batch_loss = loss.item()
                epoch_loss += batch_loss

                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(
                            f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")
            ensemble_model.loss_history.append(epoch_loss)
        
        self.on_fit_end(self, ensemble_model)
    
    def batchwisefit(self, *args, **kwargs):
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
                gpu_losses = []
                for gpu_id, (x_split, y_split) in enumerate(zip(x_splits, y_splits)):
                    y_split = y_split.to(f"cuda:{gpu_id}")
                    h_emb, r_emb, t_emb = model.get_triple_representation(x_split)
                    h_emb, r_emb, t_emb = h_emb.pin_memory().to(f"cuda:{gpu_id}",
                                                                non_blocking=True), r_emb.pin_memory().to(f"cuda:{gpu_id}", non_blocking=True), t_emb.pin_memory().to(f"cuda:{gpu_id}", non_blocking=True)
                    yhat = model.score(h_emb, r_emb, t_emb)
                    gpu_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(yhat, y_split).to("cuda:0"))

                loss = sum(gpu_losses) / len(gpu_losses)

                loss.backward()
                batch_loss = loss.item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                epoch_loss += batch_loss

                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(
                            f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")


    def torch_buggy_fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        model, = args
        # () Run the fit the start callback.
        self.on_fit_start(self, model)
        # () Init Process Group with NCCL.
        torch.distributed.init_process_group(backend="nccl")
        # () Get Rank and World Size.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # () Reinitialize Rank based on manuel seed rank.
        torch.manual_seed(rank)
        model.param_init(model.entity_embeddings.weight.data) 
        model.param_init(model.relation_embeddings.weight.data)
        # () .
        device = torch.device(f'cuda:{rank}')
        model.to(device)
        # () .
        optimizer = model.configure_optimizers()
        # () .
        for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                        verbose=True, position=0, leave=True)):
            epoch_loss = 0
            num_of_batches = len(kwargs['train_dataloaders'])
            # () .
            for i, z in enumerate(kwargs['train_dataloaders']):
                optimizer.zero_grad()
                # () Get batch and move it on GPUs .
                inputs,targets = self.extract_input_outputs(z,device)
                # () Predict .
                yhats = model(inputs)   
                # () TODO: Pytorch Bug https://github.com/pytorch/pytorch/issues/58005 .
                dist.all_reduce(yhats,op=dist.ReduceOp.SUM)
                # () Compute loss .
                loss = torch.nn.functional.binary_cross_entropy_with_logits(yhats, targets)
                # () Backward .
                loss.backward()
                # () .
                batch_loss = loss.item()
                # () .
                optimizer.step()
                # () .
                epoch_loss +=batch_loss
                # () .
                if rank==0 and hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")
        # () .
        torch.distributed.destroy_process_group()
        # () .
        self.on_fit_end(self, model)

    def extract_input_outputs(self, z: list,device=None):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        if len(z) == 2:
            x_batch, y_batch = z
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            if device:
                x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.pin_memory().to(device, non_blocking=True)
            return x_batch, y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            if device:
                x_batch, y_batch,y_idx_batch = x_batch.pin_memory().to(device, non_blocking=True), y_batch.pin_memory().to(device, non_blocking=True),y_idx_batch.pin_memory().to(device, non_blocking=True)
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError('Unexpected batch shape..')
