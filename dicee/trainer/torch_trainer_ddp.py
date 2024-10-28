import os
import torch
from typing import Iterable
from dicee.abstracts import AbstractTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

torch.set_float32_matmul_precision('high')

def make_iterable_verbose(iterable_object, verbose, desc="Default", position=None, leave=True) -> Iterable:
    if verbose:
        return tqdm(iterable_object, desc=desc, position=position, leave=leave)
    else:
        return iterable_object


class TorchDDPTrainer(AbstractTrainer):
    """
        A Trainer based on torch.nn.parallel.DistributedDataParallel

        Arguments
       ----------
       train_set_idx
           Indexed triples for the training.
       entity_idxs
           mapping.
       relation_idxs
           mapping.
       form
           ?
       store
            ?
       label_smoothing_rate
            Using hard targets (0,1) drives weights to infinity.
            An outlier produces enormous gradients.

       Returns
       -------
       torch.utils.data.Dataset
       """

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)

    def fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        model, = args
        # (1) Run the fit the start callback.
        self.on_fit_start(self, model)
        # (2) Setup DDP.
        torch.distributed.init_process_group(backend="nccl")
        train_dataset_loader = kwargs['train_dataloaders']
        # (1) Create DATA LOADER.
        train_dataset_loader = DataLoader(train_dataset_loader.dataset,
                                          batch_size=self.attributes.batch_size,
                                          pin_memory=True,
                                          shuffle=False,
                                          num_workers=self.attributes.num_core,
                                          persistent_workers=False,
                                          collate_fn=kwargs['train_dataloaders'].dataset.collate_fn,
                                          sampler=torch.utils.data.distributed.DistributedSampler(
                                              train_dataset_loader.dataset))
        # (3) Start NodeTrainer.
        NodeTrainer(self, model, train_dataset_loader, self.callbacks, self.attributes.num_epochs).train()
        torch.distributed.destroy_process_group()
        self.on_fit_end(self, model)


class NodeTrainer:
    def __init__(self,
                 trainer,
                 model: torch.nn.Module,
                 train_dataset_loader: DataLoader,
                 callbacks,
                 num_epochs: int) -> None:
        # (1) Trainer.
        self.trainer = trainer
        # (2) Local and Global Ranks.
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.optimizer = model.configure_optimizers()
        # (3) Send model to local trainer.
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = model.loss
        self.callbacks = callbacks
        self.model = torch.compile(model).to(self.local_rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])#, output_device=self.local_rank)
        self.num_epochs = num_epochs
        self.loss_history = []
        # TODO: CD: This should be given as an input param
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}["float16"]
        self.ctx = torch.amp.autocast(device_type="cuda",dtype=ptdtype)
        self.scaler = torch.amp.GradScaler("cuda",enabled=True)

    def _load_snapshot(self, snapshot_path):
        raise NotImplementedError

    def _run_batch(self, source: torch.LongTensor, targets: torch.FloatTensor):
        """
        Forward + Backward + Update over a single batch

        Parameters
        ----------
        source:
        targets

        Returns
        -------
        batch loss

        """
        with self.ctx:
            output = self.model(source)
            loss = self.loss_func(output, targets)
            batch_loss = loss.item()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        self.optimizer.zero_grad(set_to_none=True)
        return batch_loss

    def extract_input_outputs(self, z: list):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        if len(z) == 2:
            x_batch, y_batch = z
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x_batch, y_batch = x_batch.pin_memory().to(self.local_rank, non_blocking=True), y_batch.pin_memory().to(self.local_rank, non_blocking=True)
            return x_batch, y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            x_batch, y_batch,y_idx_batch = x_batch.pin_memory().to(self.local_rank, non_blocking=True), y_batch.pin_memory().to(self.local_rank, non_blocking=True),y_idx_batch.pin_memory().to(self.local_rank, non_blocking=True)
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError('Unexpected batch shape..')

    def _run_epoch(self, epoch: int) -> float:
        """
        Single pass/iteration over the training dataset

        Parameters
        ----------
        epoch:int epoch number of the DistributedSampler

        Returns
        -------
        Average mini batch loss over the training dataset

        """
        self.train_dataset_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        i = 0
        for i, z in enumerate(self.train_dataset_loader):
            source, targets = self.extract_input_outputs(z)
            batch_loss = self._run_batch(source, targets)
            epoch_loss += batch_loss
        return epoch_loss / (i + 1)

    def train(self):
        """
        Training loop for DDP

        Returns
        -------

        """
        num_of_batches=len(self.train_dataset_loader)
        for epoch in (tqdm_bar := make_iterable_verbose(range(self.num_epochs),
                                                      verbose=self.local_rank == self.global_rank == 0,
                                                      position=0,
                                                        leave=True)):
            self.train_dataset_loader.sampler.set_epoch(epoch)
            epoch_loss = 0
            for i, z in enumerate(self.train_dataset_loader):
                source, targets = self.extract_input_outputs(z)
                batch_loss = self._run_batch(source, targets)
                epoch_loss += batch_loss
                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")

            avg_epoch_loss = epoch_loss / num_of_batches

            if self.local_rank == self.global_rank == 0:
                self.model.module.loss_history.append(avg_epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(self.trainer, self.model.module)
