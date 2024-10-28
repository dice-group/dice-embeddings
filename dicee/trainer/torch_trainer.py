import torch
from typing import Tuple
from dicee.abstracts import AbstractTrainer
import time
import os
import psutil
from tqdm import tqdm

class xMP(AbstractTrainer):
    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)
        self.loss_function = None
        self.optimizer = None
        self.model = None
        self.train_dataloaders = None
        self.training_step = None
        torch.manual_seed(self.attributes.random_seed)
        torch.cuda.manual_seed_all(self.attributes.random_seed)
        assert torch.cuda.is_available(), "CUDA not available"
        self.available_gpus = torch.cuda.device_count()
        self.process = psutil.Process(os.getpid())

    def _run_batch(self, i: int, x_batch, y_batch) -> float:
        """
            Forward anc Backward according to a mini-batch

            Arguments
           ----------
           i : index of a batch
           x_batch: torch.Tensor on selected device
           y_batch: torch.Tensor on selected device
           Returns
           -------
           batch loss (float)
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

    def fit(self, *args, train_dataloaders, **kwargs) -> None:
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

        import torch
        import copy

        self.models=[]
        self.optimizers=[]
        for i in range(0, self.available_gpus):
            i_model=copy.deepcopy(model)
            self.optimizers.append(model.configure_optimizers())
            device = torch.device(f"cuda:{i}")
            self.models.append(i_model.to(device))


        del device
        del i_model
        # Create a copy of models
        self.model = model
        self.model.to(self.device)
        self.train_dataloaders = train_dataloaders
        self.loss_function = model.loss_function
        # self.optimizer = self.model.configure_optimizers()
        # self.training_step = self.model.training_step
        # (1) Start running callbacks
        # self.on_fit_start(self, self.model)

        print(f'NumOfDataPoints:{len(self.train_dataloaders.dataset)} '
              f'| NumOfEpochs:{self.attributes.max_epochs} '
              f'| LearningRate:{self.model.learning_rate} '
              f'| BatchSize:{self.train_dataloaders.batch_size} '
              f'| EpochBatchsize:{len(train_dataloaders)}')

        for epoch in (tqdm_bar := tqdm(range(self.attributes.max_epochs))):
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
                tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                if i > 0:
                    tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                else:
                    tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")
            avg_epoch_loss = epoch_loss / len(self.train_dataloaders)
            self.model.loss_history.append(avg_epoch_loss)
            self.on_train_epoch_end(self, self.model)
        self.on_fit_end(self, self.model)

    def forward_backward_update(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
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
        batch_loss = self.training_step(batch=(x_batch, y_batch))
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item()

    def extract_input_outputs_set_device(self, batch: list) -> Tuple:
        """
            Construct inputs and outputs from a batch of inputs with outputs From a batch of inputs and put

            Arguments
           ----------
           batch: (list) mini-batch inputs on CPU

           Returns
           -------
           (tuple) mini-batch on select device
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
            x_batch, y_idx_batch, y_batch, = batch
            x_batch, y_idx_batch, y_batch = x_batch.to(self.device), y_idx_batch.to(self.device), y_batch.to(
                self.device)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            print("Unexpected batch shape..")
            raise RuntimeError

class TorchTrainer(AbstractTrainer):
    """
        TorchTrainer for using single GPU or multi CPUs on a single node

        Arguments
       ----------
       args: ?

       callbacks: list of Abstract callback instances

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
            self.device = torch.device(f'cuda:{self.attributes.gpus}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        # https://psutil.readthedocs.io/en/latest/#psutil.Process
        self.process = psutil.Process(os.getpid())

    def _run_batch(self, i: int, x_batch, y_batch) -> float:
        """
            Forward anc Backward according to a mini-batch

            Arguments
           ----------
           i : index of a batch
           x_batch: torch.Tensor on selected device
           y_batch: torch.Tensor on selected device
           Returns
           -------
           batch loss (float)
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

    def fit(self, *args, train_dataloaders, **kwargs) -> None:
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
        self.train_dataloaders = train_dataloaders
        self.loss_function = model.loss_function
        self.optimizer = self.model.configure_optimizers()
        self.training_step = self.model.training_step
        # (1) Start running callbacks
        self.on_fit_start(self, self.model)

        print(f'NumOfDataPoints:{len(self.train_dataloaders.dataset)} '
              f'| NumOfEpochs:{self.attributes.max_epochs} '
              f'| LearningRate:{self.model.learning_rate} '
              f'| BatchSize:{self.train_dataloaders.batch_size} '
              f'| EpochBatchsize:{len(train_dataloaders)}')

        for epoch in (tqdm_bar := tqdm(range(self.attributes.max_epochs))):
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
                tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                if i>0:
                    tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={epoch_loss/i:.5f}")
                else:
                    tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")
            avg_epoch_loss = epoch_loss / len(self.train_dataloaders)
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

    def forward_backward_update(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
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
        batch_loss = self.training_step(batch=(x_batch, y_batch))
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss.item()

    def extract_input_outputs_set_device(self, batch: list) -> Tuple:
        """
            Construct inputs and outputs from a batch of inputs with outputs From a batch of inputs and put

            Arguments
           ----------
           batch: (list) mini-batch inputs on CPU

           Returns
           -------
           (tuple) mini-batch on select device
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
            x_batch, y_idx_batch, y_batch, = batch
            x_batch, y_idx_batch, y_batch = x_batch.to(self.device), y_idx_batch.to(self.device), y_batch.to(
                self.device)
            return (x_batch, y_idx_batch), y_batch
        else:
            print(len(batch))
            print("Unexpected batch shape..")
            raise RuntimeError