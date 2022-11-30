import torch

from core.abstracts import AbstractTrainer
from core.custom_opt.sls import Sls
from core.custom_opt.adam_sls import AdamSLS
from tqdm import tqdm
import time
import sys

class TorchTrainer(AbstractTrainer):
    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.use_closure = None
        self.device = torch.device("cpu")
        self.loss_function = None
        self.optimizer = None
        self.model = None
        self.is_global_zero = True
        torch.manual_seed(self.seed_for_computation)
        torch.cuda.manual_seed_all(self.seed_for_computation)

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args
        self.model = model
        self.model.to(self.device)
        self.on_fit_start(trainer=self, pl_module=self.model)
        dataset = kwargs['train_dataloaders'].dataset
        self.loss_function = model.loss_function
        self.optimizer = self.model.configure_optimizers()

        if isinstance(self.optimizer, Sls) or isinstance(self.optimizer, AdamSLS):
            self.use_closure = True
        else:
            self.use_closure = False

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.num_core,
                                                  collate_fn=dataset.collate_fn)

        num_total_batches = len(data_loader)
        print_period = max(num_total_batches // 10, 1)
        print(f'Number of batches for an epoch:{num_total_batches}\t printing period:{print_period}')
        for epoch in (pbar := tqdm(range(self.attributes['max_epochs']), file=sys.stdout)):
            epoch_loss = 0
            start_time = time.time()
            i: int
            batch: list
            batch_loss = -1
            for i, batch in enumerate(data_loader):
                # (1) Zero the gradients.
                self.optimizer.zero_grad()
                # (2) Extract Input and Outputs.
                x_batch, y_batch = self.extract_input_outputs(batch)
                # (3) Loss Forward and Backward w.r.t the batch.
                batch_loss = self.compute_forward_loss_backward(x_batch, y_batch)
                # (4) Accumulate a batch loss.
                epoch_loss += batch_loss.item()
            # (5) Average (4).
            epoch_loss /= num_total_batches
            # (6) Print a info.
            pbar.set_description(f'Epoch {epoch + 1}')
            pbar.set_postfix_str(
                f"runtime:{(time.time() - start_time) / 60:.3f}mins, loss={epoch_loss:.8f}")
            pbar.update(1)
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
