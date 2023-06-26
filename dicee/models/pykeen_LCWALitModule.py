from pykeen.contrib.lightning import LCWALitModule
import torch
from .pykeen_Module import *
from pykeen.triples.triples_factory import CoreTriplesFactory

class MyLCWALitModule(LCWALitModule, Pykeen_Module):

    def __init__(self, *, model_name: str, args, **kwargs):
        Pykeen_Module.__init__(self, model_name, kwargs['optimizer'])
        super().__init__(**kwargs)
        self.loss_history = []
        self.args = args
        self.train_dataloaders = self.train_dataloader()

    def training_epoch_end(self, training_step_outputs) -> None:
        pass

    def _dataloader(
            self, triples_factory: CoreTriplesFactory, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        #@TODO: There shouldn't be fixed arguments
        return torch.utils.data.DataLoader(dataset=triples_factory.create_lcwa_instances(),
                                           batch_size=self.args['batch_size'], shuffle=True,
                                           num_workers=self.args['num_core'], persistent_workers=True)



