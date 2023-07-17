import torch
import torch.utils.data
import numpy as np
from typing import Tuple,Union
import pickle
from pykeen.models import model_resolver
from .base_model import BaseKGE

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