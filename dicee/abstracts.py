import os
import datetime
from .static_funcs import load_model_ensemble, load_model, save_checkpoint_model
import torch
from typing import List, Tuple
import random
from abc import ABC
import pytorch_lightning


class AbstractTrainer:
    """
    Abstract class for Trainer class for knowledge graph embedding models


    Parameter
    ---------
    args : str
        ?

    callbacks: list
            ?
    """

    def __init__(self, args, callbacks):
        self.attributes = args
        self.callbacks = callbacks
        self.is_global_zero = True
        # Set True to use Model summary callback of pl.
        torch.manual_seed(self.attributes.random_seed)
        torch.cuda.manual_seed_all(self.attributes.random_seed)

    def on_fit_start(self, *args, **kwargs):
        """
        A function to call callbacks before the training starts.

        Parameter
        ---------
        args

        kwargs


        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_fit_start(*args, **kwargs)

    def on_fit_end(self, *args, **kwargs):
        """
        A function to call callbacks at the ned of the training.

        Parameter
        ---------
        args

        kwargs


        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_fit_end(*args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        """
        A function to call callbacks at the end of an epoch.

        Parameter
        ---------
        args

        kwargs


        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_train_epoch_end(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        """
        A function to call callbacks at the end of each mini-batch during training.

        Parameter
        ---------
        args

        kwargs


        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_train_batch_end(*args, **kwargs)

    @staticmethod
    def save_checkpoint(full_path: str, model) -> None:
        """
        A static function to save a model into disk

        Parameter
        ---------
        full_path : str

        model:


        Returns
        -------
        None
        """
        torch.save(model.state_dict(), full_path)


class BaseInteractiveKGE:
    """
    Abstract/base class for using knowledge graph embedding models interactively.


    Parameter
    ---------
    path_of_pretrained_model_dir : str
        ?

    construct_ensemble: boolean
            ?

    model_name: str
    apply_semantic_constraint : boolean
    """

    def __init__(self, path: str, construct_ensemble: bool = False, model_name: str = None,
                 apply_semantic_constraint: bool = False):
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise AssertionError(f'Could not find a directory {path}')
        self.path = path
        # (1) Load model...
        self.construct_ensemble = construct_ensemble
        self.apply_semantic_constraint = apply_semantic_constraint
        if construct_ensemble:
            self.model, self.entity_to_idx, self.relation_to_idx = load_model_ensemble(self.path)
        else:
            if model_name:
                self.model, self.entity_to_idx, self.relation_to_idx = load_model(self.path, model_name=model_name)
            else:
                self.model, self.entity_to_idx, self.relation_to_idx = load_model(self.path)
        self.num_entities = len(self.entity_to_idx)
        self.num_relations = len(self.relation_to_idx)
        self.entity_to_idx: dict
        self.relation_to_idx: dict
        assert list(self.entity_to_idx.values()) == list(range(0, len(self.entity_to_idx)))
        assert list(self.relation_to_idx.values()) == list(range(0, len(self.relation_to_idx)))

        self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
        self.idx_to_relations = {v: k for k, v in self.relation_to_idx.items()}
    def get_domain_of_relation(self, rel: str) -> List[str]:
        x = [self.idx_to_entity[i] for i in self.domain_per_rel[self.relation_to_idx[rel]]]
        res = set(x)
        assert len(x) == len(res)
        return res

    def get_range_of_relation(self, rel: str) -> List[str]:
        x = [self.idx_to_entity[i] for i in self.range_per_rel[self.relation_to_idx[rel]]]
        res = set(x)
        assert len(x) == len(res)
        return res

    def set_model_train_mode(self) -> None:
        """
        Setting the model into training mode


        Parameter
        ---------

        Returns
        ---------
        """
        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def set_model_eval_mode(self) -> None:
        """
        Setting the model into eval mode


        Parameter
        ---------

        Returns
        ---------
        """

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def name(self):
        return self.model.name

    def sample_entity(self, n: int) -> List[str]:
        assert isinstance(n, int)
        assert n >= 0
        return random.sample([i for i in self.entity_to_idx.keys()], n)

    def sample_relation(self, n: int) -> List[str]:
        assert isinstance(n, int)
        assert n >= 0
        return random.sample([i for i in self.relation_to_idx.keys()], n)

    def is_seen(self, entity: str = None, relation: str = None) -> bool:
        if entity is not None:
            return True if self.entity_to_idx.get(entity) else False
        if relation is not None:
            return True if self.relation_to_idx.get(relation) else False

    def save(self) -> None:
        t = str(datetime.datetime.now())
        if self.construct_ensemble:
            save_checkpoint_model(self.model, path=self.path + f'/model_ensemble_interactive_{str(t)}.pt')
        else:
            save_checkpoint_model(self.model, path=self.path + f'/model_interactive_{str(t)}.pt')

    def index_triple(self, head_entity: List[str], relation: List[str], tail_entity: List[str]) -> Tuple[
        torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Index Triple

        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.

        Returns: Tuple
        ---------

        pytorch tensor of triple score
        """
        n = len(head_entity)
        assert n == len(relation) == len(tail_entity)
        idx_head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity]).reshape(n, 1)
        idx_relation = torch.LongTensor([self.relation_to_idx[i] for i in relation]).reshape(n, 1)
        idx_tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity]).reshape(n, 1)
        return idx_head_entity, idx_relation, idx_tail_entity

    def add_new_entity_embeddings(self, entity_name: str = None, embeddings: torch.FloatTensor = None):
        assert isinstance(entity_name, str) and isinstance(embeddings, torch.FloatTensor)

        if entity_name in self.entity_to_idx:
            print(f'Entity ({entity_name}) exists..')
        else:
            self.entity_to_idx[entity_name] = len(self.entity_to_idx)
            self.idx_to_entity[self.entity_to_idx[entity_name]] = entity_name
            self.num_entities += 1
            self.model.num_entities += 1
            self.model.entity_embeddings.weight.data = torch.cat(
                (self.model.entity_embeddings.weight.data.detach(), embeddings.unsqueeze(0)), dim=0)
            self.model.entity_embeddings.num_embeddings += 1

    def get_entity_embeddings(self, items: List[str]):
        """
        Return embedding of an entity given its string representation


        Parameter
        ---------
        items:
            entities

        Returns
        ---------
        """
        return self.model.entity_embeddings(torch.LongTensor([self.entity_to_idx[i] for i in items]))

    def get_relation_embeddings(self, items: List[str]):
        """
        Return embedding of a relation given its string representation


        Parameter
        ---------
        items:
            relations

        Returns
        ---------
        """
        return self.model.relation_embeddings(torch.LongTensor([self.relation_to_idx[i] for i in items]))

    def construct_input_and_output(self, head_entity: List[str], relation: List[str], tail_entity: List[str], labels):
        """
        Construct a data point
        :param head_entity:
        :param relation:
        :param tail_entity:
        :param labels:
        :return:
        """
        idx_head_entity, idx_relation, idx_tail_entity = self.index_triple(head_entity, relation, tail_entity)
        x = torch.hstack((idx_head_entity, idx_relation, idx_tail_entity))
        # Hard Labels
        labels: object = torch.FloatTensor(labels)
        return x, labels

    def parameters(self):
        return self.model.parameters()


class AbstractCallback(ABC, pytorch_lightning.callbacks.Callback):
    """
    Abstract class for Callback class for knowledge graph embedding models


    Parameter
    ---------

    """

    def __init__(self):
        pass

    def on_init_start(self, *args, **kwargs):
        """

        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pass

    def on_init_end(self, *args, **kwargs):
        """
        Call at the beginning of the training.

        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pass

    def on_fit_start(self, trainer, model):
        """
        Call at the beginning of the training.

        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        return

    def on_train_epoch_end(self, trainer, model):
        """
        Call at the end of each epoch during training.

        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pass

    def on_train_batch_end(self, *args, **kwargs):
        """
        Call at the end of each mini-batch during the training.


        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pass

    def on_fit_end(self, *args, **kwargs):
        """
        Call at the end of the training.

        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pass


class AbstractPPECallback(AbstractCallback):
    """
    Abstract class for Callback class for knowledge graph embedding models


    Parameter
    ---------

    """

    def __init__(self, num_epochs, path, last_percent_to_consider):
        super(AbstractPPECallback, self).__init__()
        self.num_epochs = num_epochs
        self.path = path
        self.sample_counter = 0
        if last_percent_to_consider is None:
            self.epoch_to_start = 1
            self.num_ensemble_coefficient = self.num_epochs - 1
        else:
            # Compute the last X % of the training
            self.epoch_to_start = self.num_epochs - int(self.num_epochs * last_percent_to_consider / 100)
            self.num_ensemble_coefficient = self.num_epochs - self.epoch_to_start

    def on_fit_start(self, trainer, model):
        pass

    def on_fit_end(self, trainer, model):
        model.load_state_dict(torch.load(f"{self.path}/trainer_checkpoint_main.pt", torch.device('cpu')))

    def on_train_epoch_end(self, trainer, model):
        if self.epoch_to_start <= 0:
            if self.sample_counter == 0:
                torch.save(model.state_dict(), f=f"{self.path}/trainer_checkpoint_main.pt")
            # (1) Load the running parameter ensemble model.
            param_ensemble = torch.load(f"{self.path}/trainer_checkpoint_main.pt", torch.device(model.device))
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype==torch.float:
                        # (2) Update the parameter ensemble model with the current model.
                        param_ensemble[k] += self.alphas[self.sample_counter] * v
            # (3) Save the updated parameter ensemble model.
            torch.save(param_ensemble, f=f"{self.path}/trainer_checkpoint_main.pt")
            self.sample_counter += 1

        self.epoch_to_start -= 1

    def on_train_batch_end(self, *args, **kwargs):
        return
