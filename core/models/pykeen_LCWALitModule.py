from pykeen.contrib.lightning import LCWALitModule
from pykeen.models import model_resolver
import torch.utils.data

class Pykeen_LCWALitModule(LCWALitModule):
    def __init__(
        self,
        # dataset
        dataset,
        dataset_kwargs,
        mode,
        # model
        model,
        model_kwargs,
        # stored outside of the training loop / optimizer to give access to auto-tuning from Lightning
        batch_size,
        learning_rate,
        label_smoothing,
        # optimizer
        optimizer,
        optimizer_kwargs,
    ):
        super().__init__(dataset,
        dataset_kwargs,
        mode,
        # model
        model,
        model_kwargs,
        # stored outside of the training loop / optimizer to give access to auto-tuning from Lightning
        batch_size,
        learning_rate,
        label_smoothing,
        # optimizer
        optimizer,
        optimizer_kwargs,)
        self.dataset = None
        self.model = model_resolver.make(
            model, model_kwargs, triples_factory=self.dataset.training
        )

    def _dataloader(
        self, triples_factory, shuffle= False
    ): 
        return torch.utils.data.DataLoader(
            dataset=triples_factory.create_lcwa_instances(),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
