import torch
from core.typings import Tuple
from core.abstracts import AbstractTrainer
from core.custom_opt.sls import Sls
from core.custom_opt.adam_sls import AdamSLS
from tqdm import tqdm
import time
import sys

# Consider using https://github.com/huggingface/accelerate ?

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
           float
           """
        # (1) Extract Input and Outputs.
        x_batch, y_batch = self.extract_input_outputs(batch)

        if self.attributes.gradient_accumulation_steps > 1 and step % self.attributes.gradient_accumulation_steps == 0:
            # (2) Accumulate gradients.
            pass
        else:
            # (2) Zero the gradients.
            self.optimizer.zero_grad()
        # (3) Loss Forward and Backward w.r.t the batch.
        batch_loss = self.compute_forward_loss_backward(x_batch, y_batch).item()
        return batch_loss

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args
        self.model = model
        self.model.to(self.device)

        dataset = kwargs['train_dataloaders'].dataset
        self.loss_function = model.loss_function
        self.optimizer = self.model.configure_optimizers()
        self.on_fit_start(self, self.model)

        if isinstance(self.optimizer, Sls) or isinstance(self.optimizer, AdamSLS):
            self.use_closure = True
        else:
            self.use_closure = False

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
                epoch_loss += self.run_batch(step, batch)
                # (4) Accumulate a batch loss.
                # (6) Print a info.
                if construct_mini_batch_time:
                    print(
                        f"\tEpoch:{epoch + 1} | Batch:{step + 1} | Runtime:{(time.time() - s_time):.2f}sec | BatchConst.: {construct_mini_batch_time:.2f}sec")
                else:
                    print(f"\tEpoch:{epoch + 1} | Batch:{step + 1} | Runtime:{(time.time() - s_time):.4f}sec")
                construct_mini_batch_time = time.time()
            # (5) Average (4).
            epoch_loss /= num_total_batches
            # (6) Print a info.
            print(f"Epoch:{epoch + 1} | Loss:{epoch_loss:.8f} | Runtime:{(time.time() - start_time):.3f}sec")
            # (7) Store epoch losses
            self.model.loss_history.append(epoch_loss)
            self.on_train_epoch_end(self, self.model)
            # Write a callback to store
            # print(self.optimizer.state['step_size'])
        self.on_fit_end(self, self.model)

    def compute_forward_loss_backward(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """ Compute the forward, loss and backward """
        if self.use_closure:
            batch_loss = self.optimizer.step(closure=lambda: self.loss_function(self.model(x_batch), y_batch))
            return batch_loss
        else:
            # (4) Backpropagate the gradient of (3) w.r.t. parameters.
            batch_loss = self.loss_function(self.model(x_batch), y_batch)
            # Backward pass
            batch_loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            return batch_loss

    def extract_input_outputs(self, z: list) -> tuple:
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
