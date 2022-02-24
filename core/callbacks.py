# 1. Create Pytorch-lightning Trainer object from input configuration
from pytorch_lightning.callbacks import Callback


class PrintCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, model):
        print(model)
        print(model.summarize())
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("\nTraining is done.")
