import torch
from typing import Tuple
from dicee.abstracts import AbstractTrainer
import time
import os
import psutil


class TorchTrainer(AbstractTrainer):
    """
    A trainer class for PyTorch models that supports training on a single GPU or multiple CPUs.

    Parameters
    ----------
    args : dict
        Configuration arguments for training, including model hyperparameters and training options.
    callbacks : List[Callable]
        List of callback functions to be called at various points of the training process.

    Attributes
    ----------
    loss_function : Callable
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    model : torch.nn.Module
        The PyTorch model being trained.
    train_dataloaders : torch.utils.data.DataLoader
        torch.utils.data.DataLoader providing access to the training data.
    training_step : Callable
        The training step function defining the forward pass and loss computation.
    device : torch.device
        The device (CPU or GPU) on which training is performed.

    Methods
    -------
    _run_batch(i: int, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        Executes a training step for a single batch and returns the loss value.
    _run_epoch(epoch: int) -> float:
        Executes training for one epoch and returns the average loss.
    fit(*args, train_dataloaders: torch.utils.data.DataLoader, **kwargs) -> None:
        Starts the training process for the given model and data.
    forward_backward_update(x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        Performs the forward pass, computes the loss, and updates model weights.
    extract_input_outputs_set_device(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        Prepares and moves batch data to the appropriate device.
    """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.loss_function = None
        self.optimizer = None
        self.model = None
        self.train_dataloaders = None
        self.training_step = None
        torch.manual_seed(self.attributes.random_seed)
        torch.cuda.manual_seed_all(self.attributes.random_seed)
        if hasattr(self.attributes,"gpus") and self.attributes.gpus and torch.cuda.is_available():
            self.device = torch.device(
                f"cuda:{self.attributes.gpus}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = "cpu"
        
        # https://psutil.readthedocs.io/en/latest/#psutil.Process
        self.process = psutil.Process(os.getpid())

    def _run_batch(self, i: int, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """
        Executes a training step for a single batch and returns the loss value.

        Parameters
        ----------
        i : int
            The index of the current batch within the epoch.
        x_batch : torch.Tensor
            The batch of input features, already moved to the correct device.
        y_batch : torch.Tensor
            The batch of target outputs, already moved to the correct device.

        Returns
        -------
        float
            The loss value computed for the batch.
        """
        if self.attributes.gradient_accumulation_steps > 1:
            # (1) Update parameters every gradient_accumulation_steps mini-batch.
            if i % self.attributes.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
        else:
            # (2) Do not accumulate gradient, zero the gradients per batch.
            self.optimizer.zero_grad(set_to_none=True)
        # (3) Loss Forward and Backward w.r.t the batch.
        return self.forward_backward_update(x_batch, y_batch)

    def _run_epoch(self, epoch: int) -> float:
        """
        Executes training for one epoch and returns the average loss across all batches.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        float
            The average loss value across all batches in the epoch.
        """
        epoch_loss = 0
        i = 0
        construct_mini_batch_time = None
        batch: list
        for i, batch in enumerate(self.train_dataloaders):
            # (1) Extract Input and Outputs and set them on the dice
            x_batch, y_batch = self.extract_input_outputs_set_device(batch)
            start_time = time.time()
            if construct_mini_batch_time:
                construct_mini_batch_time = start_time - construct_mini_batch_time
            # (2) Forward-Backward-Update.
            batch_loss = self._run_batch(i, x_batch, y_batch)
            epoch_loss += batch_loss
            if construct_mini_batch_time:
                print(
                    f"Epoch:{epoch + 1} "
                    f"| Batch:{i + 1} "
                    f"| Loss:{batch_loss:.10f} "
                    f"| ForwardBackwardUpdate:{(time.time() - start_time):.2f}sec "
                    f"| BatchConst.:{construct_mini_batch_time:.2f}sec "
                    f"| Mem. Usage {self.process.memory_info().rss / 1_000_000: .5}MB "
                    f" ({psutil.virtual_memory().percent} %)"
                )
            else:
                print(
                    f"Epoch:{epoch + 1} "
                    f"| Batch:{i + 1} "
                    f"| Loss:{batch_loss} "
                    f"| ForwardBackwardUpdate:{(time.time() - start_time):.2f}secs "
                    f"| Mem. Usage {self.process.memory_info().rss / 1_000_000: .5}MB "
                )
            construct_mini_batch_time = time.time()
        return epoch_loss / (i + 1)

    def fit(
        self, *args, train_dataloaders: torch.utils.data.DataLoader, **kwargs
    ) -> None:
        """
        Starts the training process for the given model and training data.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        train_dataloaders : torch.utils.data.DataLoader
            A DataLoader instance providing access to the training data.
        """
        assert len(args) == 1
        (model,) = args
        self.model = model
        self.model.to(self.device)
        self.train_dataloaders = train_dataloaders
        self.loss_function = model.loss_function
        self.optimizer = self.model.configure_optimizers()
        self.training_step = self.model.training_step
        # (1) Start running callbacks
        self.on_fit_start(self, self.model)

        print(
            f"NumOfDataPoints:{len(self.train_dataloaders.dataset)} "
            f"| NumOfEpochs:{self.attributes.max_epochs} "
            f"| LearningRate:{self.model.learning_rate} "
            f"| BatchSize:{self.train_dataloaders.batch_size} "
            f"| EpochBatchsize:{len(train_dataloaders)}"
        )
        for epoch in range(self.attributes.max_epochs):
            start_time = time.time()

            avg_epoch_loss = self._run_epoch(epoch)
            print(
                f"Epoch:{epoch + 1} "
                f"| Loss:{avg_epoch_loss:.8f} "
                f"| Runtime:{(time.time() - start_time) / 60:.3f} mins"
            )
            """
            # Autobatch Finder: Double the current batch size if memory allows and repeat this process at mast 5 times.
            if self.attributes.auto_batch_finder and psutil.virtual_memory().percent < 30.0 and counter < 5:
                self.train_dataloaders = DataLoader(dataset=self.train_dataloaders.dataset,
                                                    batch_size=self.train_dataloaders.batch_size
                                                               + self.train_dataloaders.batch_size,
                                                    shuffle=True, collate_fn=self.train_dataloaders.dataset.collate_fn,
                                                    num_workers=self.train_dataloaders.num_workers,
                                                    persistent_workers=False)
                print(
                    f'NumOfDataPoints:{len(self.train_dataloaders.dataset)} '
                    f'| NumOfEpochs:{self.attributes.max_epochs} '
                    f'| LearningRate:{self.model.learning_rate} '
                    f'| BatchSize:{self.train_dataloaders.batch_size} '
                    f'| EpochBatchsize:{len(train_dataloaders)}')
                counter += 1
            """
            self.model.loss_history.append(avg_epoch_loss)
            self.on_train_epoch_end(self, self.model)
        self.on_fit_end(self, self.model)

    def forward_backward_update(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> float:
        """
        Performs the forward pass, computes the loss, performs the backward pass to compute gradients,
        and updates the model weights.

        Parameters
        ----------
        x_batch : torch.Tensor
            The batch of input features.
        y_batch : torch.Tensor
            The batch of target outputs.

        Returns
        -------
        float
            The loss value computed for the batch.
        """
        batch_loss = self.training_step(batch=(x_batch, y_batch))
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item()

    def extract_input_outputs_set_device(
        self, batch: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares a batch by extracting inputs and outputs and moving them to the correct device.

        Parameters
        ----------
        batch : list
            A list containing inputs and outputs for the batch.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the batch of input features and target outputs,
            both moved to the appropriate device.
        """
        if len(batch) == 2:
            x_batch, y_batch = batch

            if isinstance(x_batch, tuple):
                # Triple and Byte
                return x_batch, y_batch
            else:
                # (1) NegSample: x is a triple, y is a float
                x_batch, y_batch = batch
                return x_batch.to(self.device), y_batch.to(self.device)
        elif len(batch) == 3:
            (
                x_batch,
                y_idx_batch,
                y_batch,
            ) = batch
            x_batch, y_idx_batch, y_batch = (
                x_batch.to(self.device),
                y_idx_batch.to(self.device),
                y_batch.to(self.device),
            )
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            raise ValueError("Unexpected batch shape..")
