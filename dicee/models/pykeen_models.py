from pykeen.contrib.lightning import LCWALitModule, SLCWALitModule
from pykeen.triples.triples_factory import TriplesFactory
import torch
import torch.utils.data
from pykeen import predict
import numpy as np
from typing import Dict, Tuple
from .base_model import *
import pickle
from pykeen.models import model_resolver
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from pykeen.models import model_resolver
# LCWALitModule
import pytorch_lightning
from .base_model import BaseKGE


# @ TODO: Temp solution for the deployment
class MyERModel:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args
        self.model = None

    def load_state_dict(self, weights):
        """
        model = pykeen.models.ERModel(triples_factory=TriplesFactory(mapped_triples=load_numpy(self.args['full_storage_path']+'/train_set.npy'),
                                                                     entity_to_id=load_pickle(file_path=self.args['full_storage_path']+'/entity_to_idx.p'),
                                                                     relation_to_id=load_pickle(file_path=self.args['full_storage_path']+'/relation_to_idx.p')),
                                      entity_representations=self.weights,
                                      relation_representations=None,
                                      interaction=interaction_resolver.make(actual_name),
                                      )
        """
        # @TODO: Find a way to parse models
        from pykeen.models import QuatE
        from pykeen.contrib.lightning import LitModule
        """
        # dataset
        dataset: HintOrType[Dataset] = "nations",
        dataset_kwargs: OptionalKwargs = None,
        mode: Optional[InductiveMode] = None,
        # model
        model: HintOrType[Model] = "distmult",
        model_kwargs: OptionalKwargs = None,
        # stored outside of the training loop / optimizer to give access to auto-tuning from Lightning
        batch_size: int = 32,
        learning_rate: float = 1.0e-03,
        label_smoothing: float = 0.0,
        # optimizer
        optimizer: HintOrType[torch.optim.Optimizer] = None,
        optimizer_kwargs: OptionalKwargs = None,
        """
        self.model = model_resolver.make(self.name, embedding_dim=self.args["embedding_dim"],
                                         random_seed=0,
                                         triples_factory=TriplesFactory(
                                             mapped_triples=load_numpy(
                                                 self.args['full_storage_path'] + '/train_set.npy'),
                                             entity_to_id=load_pickle(
                                                 file_path=self.args['full_storage_path'] + '/entity_to_idx.p'),
                                             relation_to_id=load_pickle(
                                                 file_path=self.args['full_storage_path'] + '/relation_to_idx.p')))
        self.model.load_state_dict(weights)

    def parameters(self):
        for i in self.model.parameters():
            yield i

    def eval(self):
        self.model.eval()

    def __call__(self, x: torch.LongTensor):
        return self.model(mode=None, h_indices=x[:, 0], r_indices=x[:, 1], t_indices=x[:, 2])

def load_numpy(path) -> np.ndarray:
    print('Loading indexed training data...', end='')
    with open(path, 'rb') as f:
        data = np.load(f)
    return data
def load_pickle(*, file_path=str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class PykeenKGE(BaseKGE):
    def __init__(self, *, model_name: str, args, dataset):
        super().__init__(args)
        self.model_kwargs = {'embedding_dim': args['embedding_dim'],
                             'entity_initializer': None if args['init_param'] is None else torch.nn.init.xavier_normal_,
                             # 'entity_constrainer': None, for complex doesn't work but for distmult does
                             # 'regularizer': None works for ComplEx and DistMult but does not work for QuatE
                             }
        self.model_kwargs.update(args['pykeen_model_kwargs'])
        self.name = model_name
        self.model = model_resolver.make(self.name, self.model_kwargs, triples_factory=dataset.training)
        self.loss_history = []
        self.args = args

    def forward_k_vs_all(self, x: torch.LongTensor):
        return self.model.score_t(x)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx):
        raise NotImplementedError()

    def forward_triples(self, x: torch.LongTensor):
        return self.model.score_hrt(x).flatten()

    def forward(self, x: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]],
                y_idx: torch.LongTensor = None):
        """

        :param x: a batch of inputs
        :param y_idx: index of selected output labels.
        :return:
        """
        if isinstance(x, tuple):
            x, y_idx = x
            return self.forward_k_vs_sample(x=x, target_entity_idx=y_idx)
        else:
            batch_size, dim = x.shape
            if dim == 3:
                return self.forward_triples(x)
            elif dim == 2:
                # h, y = x[0], x[1]
                # Note that y can be relation or tail entity.
                return self.forward_k_vs_all(x=x)
            else:
                return self.forward_sequence(x=x)

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        yhat_batch = self.forward(x_batch)
        loss_batch=self.loss_function(yhat_batch, y_batch)
        return loss_batch + self.model.collect_regularization_term()

    def training_epoch_end(self, training_step_outputs):
        batch_losses = [i['loss'].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)
        self.loss_history.append(avg)