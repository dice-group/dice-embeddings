# 1. Create Pytorch-lightning Trainer object from input configuration
import datetime
import time
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from .static_funcs import store_kge
from typing import Optional
import os


class PrintCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_fit_start(self, trainer, pl_module):
        print(pl_module)
        print(pl_module.summarize())
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

    def on_fit_start(self, *args, **kwargs):
        pass

    def on_epoch_end(self, trainer, model):
        if self.epoch_counter % self.every_x_epoch == 0 and self.epoch_counter > 1:
            print(f'\nStoring model {self.epoch_counter}...')
            store_kge(model,
                      path=self.path + f'/model_at_{str(self.epoch_counter)}_epoch_{str(str(datetime.datetime.now()))}.pt')
        self.epoch_counter += 1


class PseudoLabellingCallback(Callback):
    def __init__(self, data_module, kg, batch_size):
        super().__init__()
        self.data_module = data_module
        self.kg = kg
        self.num_of_epochs = 0
        self.unlabelled_size = len(self.kg.unlabelled_set)
        self.batch_size = batch_size

    def create_random_data(self):
        entities = torch.randint(low=0, high=self.kg.num_entities, size=(self.batch_size, 2))
        relations = torch.randint(low=0, high=self.kg.num_relations, size=(self.batch_size,))
        # unlabelled triples
        return torch.stack((entities[:, 0], relations, entities[:, 1]), dim=1)

    def on_epoch_end(self, trainer, model):
        # Create random triples
        # if trainer.current_epoch < 10:
        #    return None
        # Increase it size, Now we increase it.
        model.eval()
        with torch.no_grad():
            # (1) Create random triples
            # unlabelled_input_batch = self.create_random_data()
            # (2) or use unlabelled batch
            unlabelled_input_batch = self.kg.unlabelled_set[
                torch.randint(low=0, high=self.unlabelled_size, size=(self.batch_size,))]
            # (2) Predict unlabelled batch, and use prediction as pseudo-labels
            pseudo_label = torch.sigmoid(model(unlabelled_input_batch))
            selected_triples = unlabelled_input_batch[pseudo_label >= .90]
        if len(selected_triples) > 0:
            # Update dataset
            self.data_module.train_set_idx = np.concatenate(
                (self.data_module.train_set_idx, selected_triples.detach().numpy()),
                axis=0)
            trainer.train_dataloader = self.data_module.train_dataloader()
            print(f'\tEpoch:{trainer.current_epoch}: Pseudo-labelling\t |D|= {len(self.data_module.train_set_idx)}')
        model.train()


class PolyakCallback(Callback):
    def __init__(self, *, path: str, max_epochs: int, polyak_start_ratio=0.75):
        super().__init__()
        self.epoch_counter = 0
        self.polyak_starts = int(max_epochs * polyak_start_ratio)
        self.path = path

    def on_fit_start(self, *args, **kwargs):
        pass

    def on_epoch_end(self, trainer, model):
        # (1) Polyak Save Condition
        if self.epoch_counter > self.polyak_starts:
            torch.save(model.state_dict(), f=f"{self.path}/trainer_checkpoint_{str(self.epoch_counter)}.pt")
        self.epoch_counter += 1

    def on_fit_end(self, trainer, model):
        """ END:Called """
        print('Perform Polyak on weights stored in disk')
        # (1) Set in eval model
        model.eval()
        trained_model.to('cpu')
        last_state = model.state_dict()
        counter = 1.0
        # (2) Accumulate weights
        for i in os.listdir(self.path):
            if '.pt' in i:
                counter += 1
                for k, v in torch.load(f'{self.path}/{i}').items():
                    last_state[k] += v
        # (3) Average (2)
        for k, v in last_state.items():
            if v.dtype != torch.int64:
                last_state[k] /= counter
        # (4) Set (3)
        model.load_state_dict(last_state)


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
