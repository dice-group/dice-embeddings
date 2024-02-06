import os
from typing import Callable, List, Tuple
import lightning as pl
import torch
import time
from torch.nn.parallel import DistributedDataParallel as DDP

from dicee.abstracts import AbstractTrainer
from dicee.static_funcs_training import efficient_zero_grad
from torch.utils.data import DataLoader


# DDP with gradiant accumulation https://gist.github.com/mcarilli/bf013d2d2f4b4dd21ade30c9b52d5e2e
def print_peak_memory(prefix: str, device: int) -> None:
    """
    Prints the peak memory usage for the specified device during the execution.

    Parameters
    ----------
    prefix : str
        A prefix string to include in the print statement for context or identification
        of the memory usage check point.
    device : int
        The device index for which to check the peak memory usage. This is typically
        used for CUDA devices. For example, `device=0` refers to the first CUDA device.

    Returns
    -------
    None

    Notes
    -----
    This function is specifically useful for monitoring the peak memory usage of GPU
    devices in CUDA context. The memory usage is reported in megabytes (MB). This can
    help in debugging memory issues or for optimizing memory usage in deep learning models.
    It requires PyTorch's CUDA utilities to be available and will print the peak allocated
    memory on the specified CUDA device. If the device is not a CUDA device or if PyTorch
    is not compiled with CUDA support, this function will not display memory usage.
    """
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


class TorchDDPTrainer(AbstractTrainer):
    """
    A Trainer class that leverages PyTorch's DistributedDataParallel (DDP) for distributed training across
    multiple GPUs. This trainer is designed for training models in a distributed fashion using multiple
    GPUs either on a single machine or across multiple nodes.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments namespace, containing training hyperparameters and configurations.
    callbacks : List[lightening.Callback]
        A list of PyTorch Lightning Callbacks to be called during the training process.

    Attributes
    ----------
    train_set_idx : np.ndarray
        An array of indexed triples for training the model.
    entity_idxs : Dict[str, int]
        A dictionary mapping entity names to their corresponding indexes.
    relation_idxs : Dict[str, int]
        A dictionary mapping relation names to their corresponding indexes.
    form : str
        The form of training to be used. This parameter specifies how the training data is presented
        to the model, e.g., 'EntityPrediction', 'RelationPrediction'.
    store : str
        The path to where the trained model and other artifacts are stored.
    label_smoothing_rate : float
        The rate of label smoothing to apply to the loss function. Using label smoothing helps in
        regularizing the model and preventing overfitting by softening the hard targets.

    Methods
    -------
    fit(self, *args, **kwargs):
        Trains the model using distributed data parallelism. This method initializes the distributed
        process group, creates a distributed data loader, and starts the training process using a
        NodeTrainer instance. It handles the setup and teardown of the distributed training environment.

    Notes
    -----
    - This trainer requires the PyTorch library and is designed to work with GPUs.
    - Proper setup of the distributed environment variables (e.g., WORLD_SIZE, RANK, LOCAL_RANK) is
      necessary before using this trainer.
    - The 'nccl' backend is used for GPU-based distributed training.
    - It's important to ensure that the same number of batches is available across all participating
      processes to avoid hanging issues.
    """

    def __init__(self, args, callbacks: List[pl.Callback]):
        super().__init__(args, callbacks)

    def fit(self, *args, **kwargs):
        """
        Trains the model using Distributed Data Parallel (DDP). This method initializes the
        distributed environment, creates a distributed sampler for the DataLoader, and starts
        the training process.

        Parameters
        ----------
        *args : Model
            The model to be trained. Passed as a positional argument.
        **kwargs : dict
            Additional keyword arguments, including:
            - train_dataloaders: DataLoader
                The DataLoader for the training dataset. Must contain a 'dataset' attribute.

        Raises
        ------
        AssertionError
            If the number of arguments is not equal to 1 (i.e., the model is not provided).

        Returns
        -------
        None
        """
        assert len(args) == 1
        (model,) = args
        # (1) Run the fit the start callback.
        self.on_fit_start(self, model)
        # (2) Setup DDP.
        torch.distributed.init_process_group(backend="nccl")
        train_dataset_loader = kwargs["train_dataloaders"]
        # (1) Create DATA LOADER.
        train_dataset_loader = DataLoader(
            train_dataset_loader.dataset,
            batch_size=self.attributes.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.attributes.num_core,
            persistent_workers=False,
            collate_fn=kwargs["train_dataloaders"].dataset.collate_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(
                train_dataset_loader.dataset
            ),
        )

        # (2) Initialize OPTIMIZER.
        optimizer = model.configure_optimizers()
        # (3) Start NodeTrainer.
        NodeTrainer(
            self,
            model,
            train_dataset_loader,
            optimizer,
            self.callbacks,
            self.attributes.num_epochs,
        ).train()
        torch.distributed.destroy_process_group()
        self.on_fit_end(self, model)


class NodeTrainer:
    """
    Manages the training process of a PyTorch model on a single node in a distributed training setup using
    DistributedDataParallel (DDP). This class orchestrates the training process across multiple GPUs on the node,
    handling batch processing, loss computation, and optimizer steps.

    Parameters
    ----------
    trainer : AbstractTrainer
        The higher-level trainer instance managing the overall training process.
    model : torch.nn.Module
        The PyTorch model to be trained.
    train_dataset_loader : DataLoader
        The DataLoader providing access to the training data, properly batched and shuffled.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model parameters.
    callbacks : list
        A list of callbacks to be executed during training, such as model checkpointing.
    num_epochs : int
        The total number of epochs to train the model.

    Attributes
    ----------
    local_rank : int
        The rank of the GPU on the current node, used for GPU-specific operations.
    global_rank : int
        The global rank of the process in the distributed training setup.
    loss_func : callable
        The loss function used to compute the difference between the model predictions and targets.
    loss_history : list
        A list to record the history of loss values over epochs.

    Methods
    -------
    _run_batch(self, source, targets):
        Processes a single batch of data, performing a forward pass, loss computation, and an optimizer step.
    extract_input_outputs(self, z):
        Extracts and sends input data and targets to the appropriate device.
    _run_epoch(self, epoch):
        Performs a single pass over the training dataset, returning the average loss for the epoch.
    train(self):
        Executes the training process, iterating over epochs and managing DDP-specific configurations.
    """

    def __init__(
        self,
        trainer,
        model: torch.nn.Module,
        train_dataset_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        callbacks,
        num_epochs: int,
    ) -> None:
        # (1) Trainer.
        self.trainer = trainer
        # (2) Local and Global Ranks.
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        # (3) Send model to local trainer.
        self.model = model.to(self.local_rank)
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = self.model.loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        # (3) Wrap the model with DDP() along with GPU ID that model lives on.
        self.model = DDP(model, device_ids=[self.local_rank])
        self.num_epochs = num_epochs
        print_peak_memory(
            "Max memory allocated after creating DDP local local_rank:", self.local_rank
        )
        print(f"Global Rank {self.global_rank}\t Local Rank:{self.local_rank}")
        print(self.model)
        print(self.optimizer)
        print(
            f"Global:{self.global_rank}"
            f" | Local:{self.local_rank}"
            f" | NumOfDataPoints:{len(self.train_dataset_loader.dataset)}"
            f" | NumOfEpochs:{self.num_epochs}"
            f" | LearningRate:{self.model.module.learning_rate}"
            f" | BatchSize:{self.train_dataset_loader.batch_size}"
            f" | EpochBatchsize:{len(self.train_dataset_loader)}"
        )

        self.loss_history = []

    def _load_snapshot(self, snapshot_path):
        raise NotImplementedError

    def _run_batch(self, source: torch.LongTensor, targets: torch.FloatTensor) -> float:
        """
        Executes the forward pass, loss computation, and optimizer step for a single batch of data.

        Parameters
        ----------
        source : torch.LongTensor
            The input data tensor.
        targets : torch.FloatTensor
            The target data tensor.

        Returns
        -------
        float
            The loss value for the batch.
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        return batch_loss

    def extract_input_outputs(self, z: list) -> tuple:
        """
        Processes the batch data, ensuring it is on the correct device.

        Parameters
        ----------
        z : list
            The batch data, which can vary in structure depending on the training setup.

        Returns
        -------
        tuple
            The processed input and output data, ready for model training.
        """
        if len(z) == 2:
            x_batch, y_batch = z
            return x_batch.to(self.local_rank), y_batch.to(self.local_rank)
        elif len(z) == 3:
            (
                x_batch,
                y_idx_batch,
                y_batch,
            ) = z
            x_batch, y_idx_batch, y_batch = (
                x_batch.to(self.local_rank),
                y_idx_batch.to(self.local_rank),
                y_batch.to(self.local_rank),
            )
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError("Unexpected batch shape..")

    def _run_epoch(self, epoch: int) -> float:
        """
        Completes one training epoch, iterating over the DataLoader to process each batch.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        float
            The average loss over all batches in the epoch.
        """
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        i = 0
        construct_mini_batch_time = None
        for i, z in enumerate(self.train_dataset_loader):
            source, targets = self.extract_input_outputs(z)
            start_time = time.time()
            if construct_mini_batch_time:
                construct_mini_batch_time = start_time - construct_mini_batch_time
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
            if True:  # self.local_rank == self.global_rank==0:
                if construct_mini_batch_time:
                    print(
                        f"Global:{self.global_rank}"
                        f" | Local:{self.local_rank}"
                        f" | Epoch:{epoch + 1}"
                        f" | Batch:{i + 1}"
                        f" | Loss:{batch_loss}"
                        f" | ForwardBackwardUpdate:{(time.time() - start_time):.2f}sec"
                        f" | BatchConst.:{construct_mini_batch_time:.2f}sec"
                    )
                else:
                    print(
                        f"Global:{self.global_rank}"
                        f" | Local:{self.local_rank}"
                        f" | Epoch:{epoch + 1}"
                        f" | Batch:{i + 1}"
                        f" | Loss:{batch_loss}"
                        f" | ForwardBackwardUpdate:{(time.time() - start_time):.2f}secs"
                    )
            construct_mini_batch_time = time.time()
        return epoch_loss / (i + 1)

    def train(self) -> None:
        """
        The main training loop. Iterates over all epochs, processing each batch of data.

        Returns
        -------
        None
        """
        for epoch in range(self.num_epochs):
            start_time = time.time()
            epoch_loss = self._run_epoch(epoch)

            print(
                f"Global:{self.global_rank}"
                f" | Local:{self.local_rank}"
                f" | Epoch:{epoch + 1}"
                f" | Loss:{epoch_loss:.8f}"
                f" | Runtime:{(time.time() - start_time) / 60:.3f}mins"
            )

            if True:  # self.local_rank == self.global_rank == 0:
                self.model.module.loss_history.append(epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(self.trainer, self.model.module)


class DDPTrainer:
    """
    Distributed Data Parallel (DDP) Trainer for PyTorch models. Orchestrates the model training across multiple GPUs
    by wrapping the model with PyTorch's DDP. It manages the training loop, loss computation, and optimization steps.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained in a distributed manner.
    train_dataset_loader : DataLoader
        DataLoader providing access to the training data, properly batched and shuffled.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for updating the model's parameters.
    gpu_id : int
        The GPU identifier where the model is to be placed.
    callbacks : List[Callable]
        A list of callback functions to be called during training.
    num_epochs : int
        The number of epochs for which the model will be trained.

    Attributes
    ----------
    loss_history : list
        Records the history of loss values over training epochs.

    Methods
    -------
    _run_batch(source: torch.Tensor, targets: torch.Tensor) -> float:
        Executes a forward pass, computes the loss, performs a backward pass, and updates the model parameters for
        a single batch of data.
    extract_input_outputs(z: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        Processes the batch data, ensuring it is on the correct device.
    _run_epoch(epoch: int) -> float:
        Completes one full pass over the entire dataset and computes the average loss for the epoch.
    train() -> None:
        Starts the training process, iterating through epochs and managing the distributed training operations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        callbacks: List[Callable],
        num_epochs: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = self.model.loss
        self.optimizer = optimizer
        self.callbacks = callbacks
        # (1) Wrap the model with DDP() along with GPU ID that model lives on.
        self.model = DDP(model, device_ids=[gpu_id])
        self.num_epochs = num_epochs
        print_peak_memory("Max memory allocated after creating DDP:", gpu_id)
        print("GPU:{self.gpu_id}")
        print(self.model)
        print(self.optimizer)
        print(
            f"NumOfDataPoints:{len(self.train_dataset_loader.dataset)}"
            f"|NumOfEpochs:{self.num_epochs}"
            f"|LearningRate:{self.model.module.learning_rate}"
            f"|BatchSize:{self.train_dataset_loader.batch_size}"
            f"|EpochBatchsize:{len(self.train_dataset_loader)}"
        )

        self.loss_history = []

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Executes a training step for a single batch.

        Parameters
        ----------
        source : torch.Tensor
            Input features for the model.
        targets : torch.Tensor
            Target outputs for the model.

        Returns
        -------
        float
            The loss value for the batch.
        """
        # (1) Zero the gradients.
        # self.optimizer.zero_grad()
        efficient_zero_grad(self.model)
        output = self.model(source)
        loss = self.loss_func(output, targets)
        batch_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        # @TODO: Tips to decrease mem usage
        #  https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        #  torch.cuda.empty_cache()
        return batch_loss

    def extract_input_outputs(
        self, z: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts and moves input and target tensors to the correct device.

        Parameters
        ----------
        z : List[torch.Tensor]
            A batch of data from the DataLoader.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Inputs and targets, moved to the correct device.
        """
        if len(z) == 2:
            x_batch, y_batch = z
            return x_batch.to(self.gpu_id), y_batch.to(self.gpu_id)
        elif len(z) == 3:
            (
                x_batch,
                y_idx_batch,
                y_batch,
            ) = z
            x_batch, y_idx_batch, y_batch = (
                x_batch.to(self.gpu_id),
                y_idx_batch.to(self.gpu_id),
                y_batch.to(self.gpu_id),
            )
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError("Unexpected batch shape..")

    def _run_epoch(self, epoch: int) -> float:
        """
        Processes all batches for a single epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        float
            The average loss over all batches in the epoch.
        """
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        i = 0
        construct_mini_batch_time = None
        for i, z in enumerate(self.train_dataset_loader):
            source, targets = self.extract_input_outputs(z)
            start_time = time.time()
            if construct_mini_batch_time:
                construct_mini_batch_time = start_time - construct_mini_batch_time
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
            if self.gpu_id == 0:
                if construct_mini_batch_time:
                    print(
                        f"Epoch:{epoch + 1}|Batch:{i + 1}"
                        f"|Loss:{batch_loss}"
                        f"|ForwardBackwardUpdate:{(time.time() - start_time):.2f}sec"
                        f"|BatchConst.:{construct_mini_batch_time:.2f}sec"
                    )
                else:
                    print(
                        f"Epoch:{epoch + 1}|Batch:{i + 1}"
                        f"|Loss:{batch_loss}"
                        f"|ForwardBackwardUpdate:{(time.time() - start_time):.2f}secs"
                    )
            construct_mini_batch_time = time.time()
        return epoch_loss / (i + 1)

    def train(self) -> None:
        """
        Trains the model across specified epochs and GPUs using DDP.

        Returns
        -------
        None
        """
        for epoch in range(self.num_epochs):
            start_time = time.time()
            epoch_loss = self._run_epoch(epoch)
            if self.gpu_id == 0:
                print(
                    f"Epoch:{epoch + 1} | Loss:{epoch_loss:.8f} | Runtime:{(time.time() - start_time) / 60:.3f}mins"
                )
                self.model.module.loss_history.append(epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(None, self.model.module)
