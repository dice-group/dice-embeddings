# 1. Create Pytorch-lightning Trainer object from input configuration
from pytorch_lightning.callbacks import Callback
from .static_funcs import store_kge


class PrintCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, model):
        print(model)
        print(model.summarize())
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("\nTraining is done.")


class KGESaveCallback(Callback):
    def __init__(self, every_x_epoch: int, path: str):
        super().__init__()
        assert isinstance(every_x_epoch, int) and every_x_epoch > 0
        self.every_x_epoch = every_x_epoch
        self.epoch_counter = 0
        self.path = path

    def on_epoch_end(self, trainer, model):
        if self.epoch_counter % self.every_x_epoch == 0 and self.epoch_counter > 0:
            print('\nStoring model..')
            # Could throw an error if mode is in GPU
            store_kge(model, path=self.path + f'/model_at_{str(self.every_x_epoch)}_epoch.pt')
        self.epoch_counter += 1
