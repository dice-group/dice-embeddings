import torch
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
