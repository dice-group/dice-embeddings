import torch
import torch.utils.data
import numpy as np
from typing import Tuple, Union
import pickle
from pykeen.models import model_resolver
from .base_model import BaseKGE
import collections


def load_numpy(path) -> np.ndarray:
    print('Loading indexed training data...', end='')
    with open(path, 'rb') as f:
        data = np.load(f)
    return data


def load_pickle(*, file_path=str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class PykeenKGE(BaseKGE):
    """ A class for using knowledge graph embedding models implemented in Pykeen """
    def __init__(self, args: dict):
        super().__init__(args)
        self.model_kwargs = {'embedding_dim': args['embedding_dim'],
                             'entity_initializer': None if args['init_param'] is None else torch.nn.init.xavier_normal_,
                             # 'entity_constrainer': None, for complex doesn't work but for distmult does
                             # 'regularizer': None works for ComplEx and DistMult but does not work for QuatE
                             }
        self.model_kwargs.update(args['pykeen_model_kwargs'])
        self.name = args['model'].split("_")[1]
        model = model_resolver. \
            make(self.name, self.model_kwargs, triples_factory=
        collections.namedtuple('triples_factory', ['num_entities', 'num_relations', 'create_inverse_triples'])(
            self.num_entities, self.num_relations, False))
        self.loss_history = []
        self.args = args

        self.entity_embeddings = None
        self.relation_embeddings = None
        for (k, v) in model.named_modules():
            if "entity_representations" == k:
                self.entity_embeddings = v[0]._embeddings
            elif "relation_representations" == k:
                self.relation_embeddings = v[0]._embeddings
            elif "interaction"==k:
                self.interaction = v
            else:
                pass
        if self.entity_embeddings.embedding_dim == 4 * self.embedding_dim:
            self.last_dim = 4
        elif self.entity_embeddings.embedding_dim == 2 * self.embedding_dim:
            self.last_dim = 2
        elif self.entity_embeddings.embedding_dim == self.embedding_dim:
            self.last_dim = 0
        else:
            raise NotImplementedError(self.entity_embeddings.embedding_dim)

    def forward_k_vs_all(self, x: torch.LongTensor):
        # (1) Retrieve embeddings of heads and relations +  apply Dropout & Normalization if given.
        h, r = self.get_head_relation_representation(x)
        # (2) Reshape (1).
        h = h.reshape(len(x), self.embedding_dim, self.last_dim)
        r = r.reshape(len(x), self.embedding_dim, self.last_dim)
        # (3) Reshape all entities.
        t = self.entity_embeddings.weight.reshape(self.num_entities, self.embedding_dim, self.last_dim)
        # (4) Call the score_t from interactions to generate triple scores.
        return self.interaction.score_t(h=h, r=r, all_entities=t, slice_size=1)

    def forward_triples(self, x: torch.LongTensor):
        # (1) Retrieve embeddings of heads, relations and tails and apply Dropout & Normalization if given.
        h, r, t = self.get_triple_representation(x)
        # (2) Reshape (1).
        h = h.reshape(len(x), self.embedding_dim, self.last_dim)
        r = r.reshape(len(x), self.embedding_dim, self.last_dim)
        t = t.reshape(len(x), self.embedding_dim, self.last_dim)
        # (3) Compute the triple score
        return self.interaction.score(h=h, r=r, t=t, slice_size=None, slice_dim=0)


    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx):
        raise NotImplementedError()

    def forward(self, x: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]],
                y_idx: torch.LongTensor = None):
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