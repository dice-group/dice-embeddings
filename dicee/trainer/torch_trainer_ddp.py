import os
from typing import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from dicee.abstracts import AbstractTrainer
from dicee.trainer.auto_batch_finder import find_good_batch_size

torch.set_float32_matmul_precision('high')


def make_iterable_verbose(iterable_object, verbose, desc="Default", position=None, leave=True) -> Iterable:
    if verbose:
        return tqdm(iterable_object, desc=desc, position=position, leave=leave)
    else:
        return iterable_object


class TorchDDPTrainer(AbstractTrainer):
    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args

        self.on_fit_start(self, model)

        train_dataset_loader = kwargs['train_dataloaders']

        train_dataset_loader = DataLoader(
            train_dataset_loader.dataset,
            batch_size=self.attributes.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.attributes.num_core,
            persistent_workers=False,
            collate_fn=kwargs['train_dataloaders'].dataset.collate_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(
                train_dataset_loader.dataset
            ),
        )

        NodeTrainer(
            self,
            model,
            train_dataset_loader,
            self.callbacks,
            self.attributes.num_epochs
        ).train()

        if dist.is_initialized():
            dist.destroy_process_group()

        self.on_fit_end(self, model)


class NodeTrainer:
    def __init__(self,
                 trainer,
                 model: torch.nn.Module,
                 train_dataset_loader: DataLoader,
                 callbacks,
                 num_epochs: int) -> None:

        self.trainer = trainer
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        self.optimizer = model.configure_optimizers()
        self.train_dataset_loader = train_dataset_loader
        self.loss_func = model.loss
        self.callbacks = callbacks

        device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

        self.model = torch.compile(model).to(device)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None
        )

        self.num_epochs = num_epochs
        self.loss_history = []

        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}["float16"]
        self.ctx = torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=ptdtype)
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    def _run_batch(self, source: torch.LongTensor, targets: torch.FloatTensor):
        with self.ctx:
            output = self.model(source)
            loss = self.loss_func(output, targets)
            batch_loss = loss.item()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        return batch_loss

    def extract_input_outputs(self, z: list):
        device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

        if len(z) == 2:
            x_batch, y_batch = z
            return (
                x_batch.pin_memory().to(device, non_blocking=True),
                y_batch.pin_memory().to(device, non_blocking=True),
            )

        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch = z
            return (
                (x_batch.pin_memory().to(device, non_blocking=True),
                 y_idx_batch.pin_memory().to(device, non_blocking=True)),
                y_batch.pin_memory().to(device, non_blocking=True),
            )

        else:
            raise ValueError('Unexpected batch shape..')

    def train(self):
        # =========================
        # AUTO BATCH FINDING (SAFE)
        # =========================
        if getattr(self.trainer.attributes, "auto_batch_finding", False):

            if self.local_rank == 0:
                device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

                def _training_step_fn(batch):
                    source, targets = self.extract_input_outputs(batch)
                    return self._run_batch(source, targets)

                new_batch_size, _ = find_good_batch_size(
                    self.train_dataset_loader,
                    _training_step_fn,
                    device=device
                )
            else:
                # safe fallback (NOT zero)
                new_batch_size = self.train_dataset_loader.batch_size

            device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

            batch_size_tensor = torch.tensor(
                new_batch_size,
                dtype=torch.long,
                device=device
            )

            try:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                    dist.broadcast(batch_size_tensor, src=0)
                    dist.barrier()

            except Exception as e:
                if self.local_rank == 0:
                    print(f"[DDP ERROR] Broadcast failed: {e}")

                if dist.is_initialized():
                    dist.destroy_process_group()

                raise RuntimeError("DDP broadcast failed — stopping training")

            new_batch_size = int(batch_size_tensor.item())

            if new_batch_size != self.train_dataset_loader.batch_size:
                self.train_dataset_loader = DataLoader(
                    self.train_dataset_loader.dataset,
                    batch_size=new_batch_size,
                    shuffle=False,
                    num_workers=self.trainer.attributes.num_core,
                    collate_fn=self.train_dataset_loader.dataset.collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=False,
                    sampler=torch.utils.data.distributed.DistributedSampler(
                        self.train_dataset_loader.dataset
                    ),
                )

        # =========================
        # TRAIN LOOP
        # =========================
        num_of_batches = len(self.train_dataset_loader)

        for epoch in (tqdm_bar := make_iterable_verbose(
                range(self.num_epochs),
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
                        tqdm_bar.set_postfix_str(
                            f"batch={i}/{num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}"
                        )
                    else:
                        tqdm_bar.set_postfix_str(
                            f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}"
                        )

            avg_epoch_loss = epoch_loss / num_of_batches

            if self.local_rank == self.global_rank == 0:
                self.model.module.loss_history.append(avg_epoch_loss)
                for c in self.callbacks:
                    c.on_train_epoch_end(self.trainer, self.model.module)
