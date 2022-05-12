# 1. Create Pytorch-lightning Trainer object from input configuration
import datetime
import time
import numpy as np
import torch
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
    def __init__(self, dataset, kg):
        super().__init__()
        self.dataset = dataset
        self.kg = kg
        self.num_of_epochs = 0

    def create_random_data(self):
        # TODO: maybe sample triples that are not outside of the range and domain ?
        entities = torch.randint(low=0, high=self.kg.num_entities, size=(50, 2))
        relations = torch.randint(low=0, high=self.kg.num_relations, size=(50,))
        # unlabelled triples
        return torch.stack((entities[:, 0], relations, entities[:, 1]), dim=1)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        pass

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        pass

    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_epoch_end(self, trainer, model):
        # Create random triples
        if trainer.current_epoch < 10:
            return None
        # Increase it size, Now we increase it.
        model.eval()
        with torch.no_grad():
            # (1) Create random triples
            unlabelled_data = self.create_random_data()
            # (2) Select (1) s.t. model is too confident
            unlabelled_data = unlabelled_data[model(unlabelled_data) > 3.0]
        # Update dataset
        self.dataset.train_set_idx = np.concatenate((self.dataset.train_set_idx, unlabelled_data.detach().numpy()),
                                                    axis=0)
        trainer.train_dataloader = self.dataset.train_dataloader()
        print(trainer.current_epoch, len(self.dataset.train_set_idx))
        model.train()

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
