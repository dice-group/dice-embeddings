from pykeen.contrib.lightning import SLCWALitModule
from .pykeen_Module import *
from pykeen.triples.triples_factory import CoreTriplesFactory
import wandb

class MySLCWALitModule(SLCWALitModule, Pykeen_Module):
    def __init__(self, *, model_name: str, args, **kwargs):
        Pykeen_Module.__init__(self, model_name,kwargs['optimizer'])
        # import pdb; pdb.set_trace()
        super().__init__(**kwargs)
        self.loss_history = []
        self.args=args
        self.train_dataloaders = self.train_dataloader()


    def training_epoch_end(self, training_step_outputs) -> None:
        batch_losses = [i["loss"].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)

        log_dict = {
            'loss':avg,
            "epoch":self.current_epoch+1,
        }
        # print(avg)
        # wandb.log(log_dict)
        print(avg)
        # log_dict = {
        #     'train_loss':training_step_outputs,
        #     "epoch":self.current_epoch+1,
        # }
        wandb.log(log_dict)

