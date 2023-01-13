import datetime
import time
import numpy as np
import torch
from .static_funcs import save_checkpoint_model
from .abstracts import AbstractCallback
from typing import Optional
import os
import pandas as pd


class AccumulateEpochLossCallback(AbstractCallback):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def on_fit_end(self, trainer, model) -> None:
        """
        Store epoch loss


        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pd.DataFrame(model.loss_history, columns=['EpochLoss']).to_csv(f'{self.path}/epoch_losses.csv')


class PrintCallback(AbstractCallback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_fit_start(self, trainer, pl_module):
        print(pl_module)
        print(pl_module.summarize())
        print(pl_module.selected_optimizer)
        print(f"\nTraining is starting {datetime.datetime.now()}...")

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

    def on_train_batch_end(self, *args, **kwargs):
        return

    def on_train_epoch_end(self, *args, **kwargs):
        return


class KGESaveCallback(AbstractCallback):
    def __init__(self, every_x_epoch: int, max_epochs: int, path: str):
        super().__init__()
        self.every_x_epoch = every_x_epoch
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.path = path
        if self.every_x_epoch is None:
            self.every_x_epoch = max(self.max_epochs // 2, 1)

    def on_train_batch_end(self, *args, **kwargs):
        return

    def on_fit_start(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_fit_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, trainer, pl_module):
        if self.epoch_counter % self.every_x_epoch == 0 and self.epoch_counter > 1:
            print(f'\nStoring model {self.epoch_counter}...')
            save_checkpoint_model(pl_module,
                                  path=self.path + f'/model_at_{str(self.epoch_counter)}_epoch_{str(str(datetime.datetime.now()))}.pt')
        self.epoch_counter += 1


class PseudoLabellingCallback(AbstractCallback):
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


def estimate_q(eps):
    """ estimate rate of convergence q from sequence esp"""
    x = np.arange(len(eps) - 1)
    y = np.log(np.abs(np.diff(np.log(eps))))
    line = np.polyfit(x, y, 1)  # fit degree 1 polynomial
    q = np.exp(line[0])  # find q
    return q


def compute_convergence(seq, i):
    assert len(seq) >= i > 0
    return estimate_q(seq[-i:] / (np.arange(i) + 1))


class PPE:
    """ A callback for Polyak Parameter Ensemble Technique

        Maintains a running parameter average for all parameters requiring gradient signals
    """

    def __init__(self, num_epochs, path, last_percent_to_consider=None):
        self.num_epochs = num_epochs
        self.path = path
        self.epoch_counter = 0
        self.sample_counter = 0
        if last_percent_to_consider is None:
            self.epoch_to_start = 1
        else:
            # e.g. Average only last 10 percent
            self.epoch_to_start = self.num_epochs - int(self.num_epochs / last_percent_to_consider)

    def on_fit_start(self, trainer, model):
        torch.save(model.state_dict(), f=f"{self.path}/trainer_checkpoint_main.pt")

    def on_train_epoch_end(self, trainer, model):
        self.epoch_counter += 1
        if self.epoch_to_start < self.epoch_counter:
            # Load averaged model
            device_of_training = model.device
            x = torch.load(f"{self.path}/trainer_checkpoint_main.pt", torch.device(model.device))

            with torch.no_grad():
                # Update the model
                for k, v in model.state_dict().items():
                    x[k] = (x[k] * self.sample_counter + v) / (self.sample_counter + 1)
            # Store the model
            torch.save(x, f=f"{self.path}/trainer_checkpoint_main.pt")
            self.sample_counter += 1

    def on_fit_end(self, trainer, model):
        """ END:Called """
        model.load_state_dict(torch.load(f"{self.path}/trainer_checkpoint_main.pt", torch.device('cpu')))

    def on_train_batch_end(self, *args, **kwargs):
        return


class FPPE:
    """ A callback for Forgetful Polyak Parameter Ensemble Technique

        Maintains a running weighted average of parameters in each epoch interval.
        As i -> N the impact of the parameters at the early stage of the training decreasing.
    """

    def __init__(self, num_epochs, path, last_percent_to_consider):
        self.num_epochs = num_epochs
        self.path = path
        self.epoch_counter = 0
        self.sample_counter = 0
        self.epoch_to_start = 0

        if last_percent_to_consider is None:
            # Initialize Alphas
            self.alphas = np.cumsum(np.ones(self.num_epochs) * (1 / self.num_epochs))
        else:
            # e.g. Average only last 10 percent
            self.epoch_to_start = self.num_epochs - int(self.num_epochs / last_percent_to_consider)
            size_of_alphas = self.num_epochs - self.epoch_to_start - 1
            self.alphas = np.cumsum(np.ones(size_of_alphas) * (1 / size_of_alphas))
        self.alphas /= sum(self.alphas)
        assert 1.00001 >= sum(self.alphas) >= 0.999
        self.alphas = torch.from_numpy(self.alphas)
        print(self.alphas)

    def on_fit_start(self, trainer, model):
        torch.save(model.state_dict(), f=f"{self.path}/trainer_checkpoint_main.pt")

    def on_train_epoch_end(self, trainer, model):

        if self.epoch_to_start < self.epoch_counter:
            # Load averaged model
            x = torch.load(f"{self.path}/trainer_checkpoint_main.pt", torch.device(model.device))

            with torch.no_grad():
                # Update the model
                for k, v in model.state_dict().items():
                    x[k] = x[k] * self.alphas[self.sample_counter] + v
            # Store the model
            torch.save(x, f=f"{self.path}/trainer_checkpoint_main.pt")
            self.sample_counter += 1
        self.epoch_counter += 1

    def on_fit_end(self, trainer, model):
        """ END:Called """
        model.load_state_dict(torch.load(f"{self.path}/trainer_checkpoint_main.pt", torch.device('cpu')))

    def on_train_batch_end(self, *args, **kwargs):
        return
