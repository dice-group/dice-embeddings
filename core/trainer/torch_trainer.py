import torch
from typing import Tuple
from core.abstracts import AbstractTrainer
from core.custom_opt.sls import Sls
from core.custom_opt.adam_sls import AdamSLS
from core.static_funcs_training import efficient_zero_grad
import time


class TorchTrainer(AbstractTrainer):
    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.use_closure = None
        self.loss_function = None
        self.optimizer = None
        self.model = None
        self.is_global_zero = True
        torch.manual_seed(self.attributes.seed_for_computation)
        torch.cuda.manual_seed_all(self.attributes.seed_for_computation)

        if self.attributes.gpus:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

    def run_batch(self, step: int, batch: Tuple) -> float:
        """
            Forward anc Backward according to a mini-batch

            Arguments
           ----------
           step:int
               step of operation per epoch
           batch:Tuple
               a batch of datapoints.
           Returns
           -------
           batch loss (float)
       """
        # (1) Extract Input and Outputs.
        x_batch, y_batch = self.extract_input_outputs(batch)

        if self.attributes.gradient_accumulation_steps > 1:
            # Update parameters every gradient_accumulation_steps mini-batch
            if step % self.attributes.gradient_accumulation_steps == 0:
                efficient_zero_grad(self.model)
        else:
            # (2) Do not accumulate gradient, zero the gradients per batch.
            efficient_zero_grad(self.model)
        # (3) Loss Forward and Backward w.r.t the batch.
        return self.compute_forward_loss_backward(x_batch, y_batch).item()

    def fit(self, *args, **kwargs) -> None:
        """
            Training starts

            Arguments
           ----------
           args:tuple
           (BASEKGE,)
           kwargs:Tuple
               empty dictionary
           Returns
           -------
           batch loss (float)
       """
        assert len(args) == 1
        model, = args
        self.model = model
        self.model.to(self.device)

        dataset = kwargs['train_dataloaders'].dataset
        self.loss_function = model.loss_function
        self.optimizer = self.model.configure_optimizers()
        # (1) Start running callbacks
        self.on_fit_start(self, self.model)

        if isinstance(self.optimizer, Sls) or isinstance(self.optimizer, AdamSLS):
            self.use_closure = True
        else:
            self.use_closure = False
        # (2) Creat Data loader.
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.attributes.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.attributes.num_core,
                                                  collate_fn=dataset.collate_fn)

        num_total_batches = len(data_loader)
        # Creates once at the beginning of training
        for epoch in range(self.attributes.max_epochs):
            epoch_loss = 0
            start_time = time.time()
            i: int
            batch: list
            batch_loss = -1
            construct_mini_batch_time = None
            for step, batch in enumerate(data_loader):
                s_time = time.time()
                if construct_mini_batch_time:
                    construct_mini_batch_time = s_time - construct_mini_batch_time
                batch_loss = self.run_batch(step, batch)
                epoch_loss += batch_loss
                # (1) Start running callbacks
                self.on_train_batch_end(self, self.model)
                # (4) Accumulate a batch loss.
                # (6) Print a info.
                if construct_mini_batch_time:
                    print(
                        f"\tEpoch:{epoch + 1} | Batch {step + 1} Loss:{batch_loss} | Runtime:{(time.time() - s_time):.2f}sec | BatchConst.: {construct_mini_batch_time:.2f}sec")
                else:
                    print(
                        f"\tEpoch:{epoch + 1} | Batch:{step + 1} Loss:{batch_loss} | Runtime:{(time.time() - s_time):.4f}sec")
                construct_mini_batch_time = time.time()
            # (5) Average (4).
            epoch_loss /= num_total_batches
            # (6) Print a info.
            print(f"Epoch:{epoch + 1} | Loss:{epoch_loss} | Runtime:{(time.time() - start_time):.3f}sec")
            # (7) Store epoch losses
            self.model.loss_history.append(epoch_loss)
            self.on_train_epoch_end(self, self.model)
            # Write a callback to store
            # print(self.optimizer.state['step_size'])
        self.on_fit_end(self, self.model)

    def compute_forward_loss_backward(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """
            Compute forward, loss, backward, and parameter update

            Arguments
           ----------
           x_batch:(torch.Tensor) mini-batch inputs
           y_batch:(torch.Tensor) mini-batch outputs

           Returns
           -------
           batch loss (float)
       """
        if self.use_closure:
            batch_loss = self.optimizer.step(closure=lambda: self.loss_function(self.model(x_batch), y_batch))
            return batch_loss
        else:
            #with torch.autocast(device_type=self.device,dtype=torch.bfloat16):
            # (4) Backpropagate the gradient of (3) w.r.t. parameters.
            batch_loss = self.loss_function(self.model(x_batch), y_batch)
            # Backward pass
            batch_loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            return batch_loss

    def extract_input_outputs(self, z: list) -> tuple:
        """
            Construct inputs and outputs from a batch of inputs with outputs From a batch of inputs and put

            Arguments
           ----------
           z: (list) mini-batch inputs on CPU

           Returns
           -------
           (tuple) mini-batch on select device
       """
        """ Construct inputs and outputs from a batch of inputs with outputs From a batch of inputs and put """
        if len(z) == 2:
            x_batch, y_batch = z
            return x_batch.to(self.device), y_batch.to(self.device)
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            x_batch, y_idx_batch, y_batch = x_batch.to(self.device), y_idx_batch.to(self.device), y_batch.to(
                self.device)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            raise ValueError('Unexpected batch shape..')
