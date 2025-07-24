import torch
from ..abstracts import AbstractTrainer
from ..static_funcs_training import make_iterable_verbose
from ..models.ensemble import EnsembleKGE
from typing import Tuple
import time

def extract_input_outputs(z: list, device=None):
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    if len(z) == 2:
        x_batch, y_batch = z
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        if device:
            x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.pin_memory().to(device,
                                                                                              non_blocking=True)
        return x_batch, y_batch
    elif len(z) == 3:
        x_batch, y_idx_batch, y_batch, = z
        if device:
            x_batch, y_batch, y_idx_batch = x_batch.pin_memory().to(device,
                                                                    non_blocking=True), y_batch.pin_memory().to(
                device, non_blocking=True), y_idx_batch.pin_memory().to(device, non_blocking=True)
        return (x_batch, y_idx_batch), y_batch
    else:
        raise ValueError('Unexpected batch shape..')


def find_good_batch_size(train_loader,tp_ensemble_model):
    # () Initial batch size.
    initial_batch_size=train_loader.batch_size
    # () # of training data points.
    training_dataset_size=len(train_loader.dataset)
    # () Batch is large enough.
    if initial_batch_size >= training_dataset_size:
        return training_dataset_size, None
    # () Log the number of training data points.
    print("Number of training data points:",training_dataset_size)

    def increase_batch_size_until_cuda_out_of_memory(ensemble_model, train_loader, batch_size,delta: int = None):
        assert delta is not None, "delta cannot be None."
        assert isinstance(delta, int), "delta must be a positive integer."
        # () Store the batch sizes and GPU memory usages in a tuple.
        batch_sizes_and_mem_usages = []
        # () Increase the batch size until a stopping criterion is reached.
        try:
            while True:
                start_time=time.time()
                # () Initialize a dataloader with a current batch_size
                train_dataloaders = torch.utils.data.DataLoader(train_loader.dataset,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                sampler=None,
                                                                batch_sampler=None,
                                                                num_workers=train_loader.num_workers,
                                                                collate_fn=train_loader.dataset.collate_fn,
                                                                pin_memory=False,
                                                                drop_last=False,
                                                                timeout=0,
                                                                worker_init_fn=None,
                                                                persistent_workers=False)
                
                batch_loss = None
                for i, batch_of_training_data in enumerate(train_dataloaders):
                    batch_loss = forward_backward_update_loss(batch_of_training_data, ensemble_model)
                    break
                
                global_free_memory, total_memory = torch.cuda.mem_get_info(device="cuda:0")
                percentage_used_gpu_memory = (total_memory - global_free_memory) / total_memory
                rt=time.time()-start_time

                print(f"Random Batch Loss: {batch_loss:0.4}\tGPU Usage: {percentage_used_gpu_memory:0.3}\tRuntime: {rt:.3f}\tBatch Size: {batch_size}")

                # Store the batch size and the runtime
                batch_sizes_and_mem_usages.append((batch_size, rt))
                
                # ()
                # https://github.com/pytorch/pytorch/issues/21819
                # CD: as we reach close to 1.0 GPU memory usage, we observe RuntimeError: CUDA error: an illegal memory access was encountered.
                # CD: To avoid this problem, we add the following condition as a temp solution.
                if percentage_used_gpu_memory > 0.9:
                    # Mimik out of memory error
                    return batch_sizes_and_mem_usages, False
                if batch_size < training_dataset_size:
                    # Increase the batch size.
                    batch_size += int(batch_size / delta)
                else:
                    return batch_sizes_and_mem_usages,True
                        
        except torch.OutOfMemoryError as e:
            print(f"torch.OutOfMemoryError caught! {e}\n\n")
            return batch_sizes_and_mem_usages, False

    history_batch_sizes_and_mem_usages=[]
    batch_size=initial_batch_size

    for delta in range(1,5,1):
        result,flag= increase_batch_size_until_cuda_out_of_memory(tp_ensemble_model, train_loader, batch_size,delta=delta)
        
        history_batch_sizes_and_mem_usages.extend(result)

        if flag:
            batch_size, batch_rt = history_batch_sizes_and_mem_usages[-1]
        else:
            assert len(history_batch_sizes_and_mem_usages)>2, "GPU memory errorin the first try"
            # CUDA ERROR Observed 
            batch_size, batch_rt=history_batch_sizes_and_mem_usages[-2]
            # https://github.com/pytorch/pytorch/issues/21819
            break

        if batch_size>=training_dataset_size:
            batch_size=training_dataset_size
            break
        else:
            continue

    return batch_size, batch_rt


def forward_backward_update_loss(z:Tuple, ensemble_model)->float:
    # () Get a random batch of data points (z).
    x_batch, y_batch = extract_input_outputs(z)
    # () Move the batch of labels into the master GPU : GPU-0.
    y_batch = y_batch.to("cuda:0")
    # () Forward pas on the batch of input data points (yhat on the master GPU).
    yhat = ensemble_model(x_batch)
    # () Compute the loss.
    loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat, y_batch)
    # () Compute the gradient of the loss w.r.t. parameters.
    loss.backward()
    # () Parameter update.
    ensemble_model.step()
    return loss.item()

def update_embedding_layer(ensemble_model):
    # () Update the embedding layer.
    for model in ensemble_model:
        if hasattr(model, 'update_embedding_layer'):
            model.update_embedding_layer()
    return ensemble_model

class TensorParallel(AbstractTrainer):
    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)


    def fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        ensemble_model, = args
        assert isinstance(ensemble_model,EnsembleKGE), (f"Selected model must "
                                                        f"be an instance of EnsembleKGE{type(ensemble_model)}")
        # () Run on_fit_start callbacks.
        self.on_fit_start(self, ensemble_model)
        # () Sanity checking
        assert torch.cuda.device_count()== len(ensemble_model)
        # () Get DataLoader
        train_dataloader = kwargs['train_dataloaders']
        # () Find a batch size so that available GPU memory is *almost* fully used.
        if self.attributes.auto_batch_finding:
            batch_size, batch_rt=find_good_batch_size(train_dataloader, ensemble_model)

            train_dataloader = torch.utils.data.DataLoader(train_dataloader.dataset,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            sampler=None,
                                                            batch_sampler=None,
                                                            num_workers=self.attributes.num_core,
                                                            collate_fn=train_dataloader.dataset.collate_fn,
                                                            pin_memory=False,
                                                            drop_last=False,
                                                            timeout=0,
                                                            worker_init_fn=None,
                                                            persistent_workers=False)
            #if batch_rt is not None:
            #    expected_training_time=batch_rt * len(train_dataloader) * self.attributes.num_epochs
            # print(f"Exp.Training Runtime: {expected_training_time/60 :.3f} in mins\t|\tBatch Size:{batch_size}\t|\tBatch RT:{batch_rt:.3f}\t|\t # of batches:{len(train_dataloader)}\t|\t# of epochs:{self.attributes.num_epochs}")

        # --- Use stacking if vocab_size is set ---
        if getattr(self.attributes, "vocab_size", None) is not None:
            threshold = int(0.9 * self.attributes.vocab_size)
            for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                            verbose=True, position=0, leave=True)):
                epoch_loss = 0
                processed_indices = set()
                while True:
                    stacked_triples_tensor, stacked_labels_tensor, batch_indices = self.stack_triples_until_threshold(
                        train_dataloader, threshold)
                    if stacked_triples_tensor is None:
                        break
                    # Mark these batches as processed
                    processed_indices.update(batch_indices)
                    # Create a new DataLoader for the stacked triples
                    stacked_dataset = torch.utils.data.TensorDataset(stacked_triples_tensor, stacked_labels_tensor)
                    stacked_loader = torch.utils.data.DataLoader(
                        stacked_dataset,
                        batch_size=train_dataloader.batch_size,
                        shuffle=True,
                        num_workers=self.attributes.num_core,
                        collate_fn=train_dataloader.dataset.collate_fn,
                        pin_memory=False,
                        drop_last=False,
                        timeout=0,
                        worker_init_fn=None,
                        persistent_workers=False 
                    )
                    for i, z in enumerate(stacked_loader):
                        batch_loss = forward_backward_update_loss(z, ensemble_model)
                        epoch_loss += batch_loss
                        if hasattr(tqdm_bar, 'set_description_str'):
                            tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                            tqdm_bar.set_postfix_str(
                                f"stack_size={len(stacked_triples_tensor)}, loss_step={batch_loss:.5f}")
                    # Remove processed batches from train_dataloader for next stacking
                    if len(processed_indices) >= len(train_dataloader):
                        break
                ensemble_model.loss_history.append(epoch_loss)
        else:
            # () Number of batches to reach a single epoch.
            num_of_batches = len(train_dataloader)
            # () Start training.
            for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                            verbose=True, position=0, leave=True)):
                # () Accumulate the batch losses.
                epoch_loss = 0
                # () Iterate over batches.
                for i, z in enumerate(train_dataloader):
                    # if i>0:
                    #     ensemble_model = update_embedding_layer(ensemble_model)
                    # () Forward, Loss, Backward, and Update on a given batch of data points.
                    batch_loss = forward_backward_update_loss(z,ensemble_model)
                    # () Accumulate the batch losses to compute the epoch loss.
                    epoch_loss += batch_loss
                    # if verbose=TRue, show info.
                    if hasattr(tqdm_bar, 'set_description_str'):
                        tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                        if i > 0:
                            tqdm_bar.set_postfix_str(
                                f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                        else:
                            tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")
                # Store the epoch loss
                ensemble_model.loss_history.append(epoch_loss)
        # Run on_fit_end callbacks after the training is done.
        self.on_fit_end(self, ensemble_model)
        # TODO: Later, maybe we should write a callback to save the models in disk
        return ensemble_model
    
    def stack_triples_until_threshold(self, dataloader, threshold):
            """
            Stacks triples and labels from the dataloader until the number of unique heads and tails
            reaches the threshold. Returns a list of (triples, labels) tensors and a set of processed batch indices.
            """
            stacked_triples = []
            stacked_labels = []
            unique_heads = set()
            unique_tails = set()
            processed_indices = set()
            for i, z in enumerate(dataloader):
                x_batch, y_batch = extract_input_outputs(z)
                if isinstance(x_batch, tuple):
                    x_data = x_batch[0]
                else:
                    x_data = x_batch
                stacked_triples.append(x_data)
                stacked_labels.append(y_batch)
                unique_heads.update(x_data[:, 0].tolist())
                unique_tails.update(x_data[:, 2].tolist())
                processed_indices.add(i)
                if len(unique_heads) >= threshold and len(unique_tails) >= threshold:
                    break
            if stacked_triples:
                stacked_triples_tensor = torch.cat(stacked_triples, dim=0)
                stacked_labels_tensor = torch.cat(stacked_labels, dim=0)
                return stacked_triples_tensor, stacked_labels_tensor, processed_indices
            else:
                return None, None, processed_indices
    
    """
    
    def batchwisefit(self, *args, **kwargs):
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
                inputs,targets = extract_input_outputs(z,device)
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
    """
