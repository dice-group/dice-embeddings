# 1. Create Pytorch-lightning Trainer object from input configuration
import datetime
import time

from pytorch_lightning.callbacks import Callback
from .static_funcs import store_kge
from typing import Optional


class PrintCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_fit_start(self, trainer, model):
        print(model)
        print(model.summarize())
        print("\n[1 / 1] Training is started..")

    def on_fit_end(self, trainer, pl_module):
        training_time = time.time() - self.start_time
        if 60 > training_time:
            message = f'{training_time:.3f} seconds.'
        elif 60 * 60 > training_time > 60:
            message = f'{training_time / 60:.3f} minutes.'
        elif training_time > 60 * 60:
            message = f'{training_time / (60 * 60):.3f} hours.'
        else:
            message = f'{training_time:.3f} seconds.'
        print(f"Done ! It took {message}\n")


class KGESaveCallback(Callback):
    def __init__(self, every_x_epoch: int, max_epochs: int, path: str):
        super().__init__()
        self.every_x_epoch = every_x_epoch
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.path = path
        if self.every_x_epoch is None:
            self.every_x_epoch = max(self.max_epochs // 2, 1)

    def on_epoch_end(self, trainer, model):
        if self.epoch_counter % self.every_x_epoch == 0 and self.epoch_counter > 1:
            print(f'\nStoring model {self.epoch_counter}...')
            store_kge(model,
                      path=self.path + f'/model_at_{str(self.epoch_counter)}_epoch_{str(str(datetime.datetime.now()))}.pt')
        self.epoch_counter += 1


class PseudoLabellingCallback(Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        pass

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        pass

    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def _prepare_epoch(self, trainer, model, epoch):
        # Increase it size, Now we increase it.

        model.eval()
        if len(self.dataset.train_set_idx) > 100:
            self.dataset.train_set_idx = self.dataset.train_set_idx[:-100]

        trainer.train_dataloader = self.dataset.train_dataloader()
        print(epoch, len(self.dataset.train_set_idx))
        model.train()

    def on_epoch_end(self, trainer, model):
        self._prepare_epoch(trainer, model, trainer.current_epoch + 1)


# https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#persisting-state
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#teardown
class AdaptiveKGECallback(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        pass

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        pass

    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_epoch_end(self, trainer, model):
        print(trainer.callback_metrics)
