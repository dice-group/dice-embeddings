# 1. Create Pytorch-lightning Trainer object from input configuration
import time

from pytorch_lightning.callbacks import Callback
from .static_funcs import store_kge


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
    def __init__(self, every_x_epoch, path: str):
        super().__init__()
        self.every_x_epoch = every_x_epoch
        self.epoch_counter = 0
        self.path = path

    def on_epoch_end(self, trainer, model):
        if self.every_x_epoch:
            if self.epoch_counter % self.every_x_epoch == 0 and self.epoch_counter > 0:
                print('\nStoring model..')
                # Could throw an error if mode is in GPU
                store_kge(model, path=self.path + f'/model_at_{str(self.every_x_epoch)}_epoch.pt')
            self.epoch_counter += 1
