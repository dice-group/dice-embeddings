import os
import datetime
# import pandas.core.indexes.range
from .static_funcs import load_model_ensemble, load_model, save_checkpoint_model,load_numpy
from .static_preprocess_funcs import create_constraints
import torch
from typing import List, Tuple
import pandas as pd
import numpy as np
import random
from abc import ABC, abstractmethod


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
        torch.manual_seed(self.attributes.seed_for_computation)
        torch.cuda.manual_seed_all(self.attributes.seed_for_computation)

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
            self.model, self.entity_to_idx, self.relation_to_idx = load_model_ensemble(self.path + '/')
        else:
            if model_name:
                self.model, self.entity_to_idx, self.relation_to_idx = load_model(self.path + '/',
                                                                                  model_name=model_name)
            else:
                self.model, self.entity_to_idx, self.relation_to_idx = load_model(self.path + '/')

        self.num_entities = len(self.entity_to_idx)
        self.num_relations = len(self.relation_to_idx)
        self.entity_to_idx: dict
        self.relation_to_idx: dict
        assert list(self.entity_to_idx.values()) == list(range(0, len(self.entity_to_idx)))
        assert list(self.relation_to_idx.values()) == list(range(0, len(self.relation_to_idx)))

        self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
        self.idx_to_relations = {v: k for k, v in self.relation_to_idx.items()}

        self.train_set=load_numpy(path=self.path + '/train_set.npy')

        if self.apply_semantic_constraint:
            self.domain_constraints_per_rel, self.range_constraints_per_rel, self.domain_per_rel, self.range_per_rel = create_constraints(
                self.train_set)

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

    def __predict_missing_head_entity(self, relation: List[str], tail_entity: List[str]) -> Tuple:
        """
        Given a relation and a tail entity, return top k ranked head entity.

        argmax_{e \in E } f(e,r,t), where r \in R, t \in E.

        Parameter
        ---------
        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.


        k: int

        Highest ranked k entities.

        Returns: Tuple
        ---------

        Highest K scores and entities
        """

        head_entity = torch.arange(0, len(self.entity_to_idx))
        relation = torch.LongTensor([self.relation_to_idx[i] for i in relation])
        tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity])
        x = torch.stack((head_entity,
                         relation.repeat(self.num_entities, ),
                         tail_entity.repeat(self.num_entities, )), dim=1)
        return self.model(x)

    def __predict_missing_relations(self, head_entity: List[str], tail_entity: List[str]) -> Tuple:
        """
        Given a head entity and a tail entity, return top k ranked relations.

        argmax_{r \in R } f(h,r,t), where h, t \in E.


        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        tail_entity: List[str]

        String representation of selected entities.


        k: int

        Highest ranked k entities.

        Returns: Tuple
        ---------

        Highest K scores and entities
        """

        head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity])
        relation = torch.arange(0, len(self.relation_to_idx))
        tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity])

        x = torch.stack((head_entity.repeat(self.num_relations, ),
                         relation,
                         tail_entity.repeat(self.num_relations, )), dim=1)
        return self.model(x)
        # scores = self.model(x)
        # sort_scores, sort_idxs = torch.topk(scores, topk)
        # return sort_scores, [self.idx_to_relations[i] for i in sort_idxs.tolist()]

    def __predict_missing_tail_entity(self, head_entity: List[str], relation: List[str]) -> torch.FloatTensor:
        """
        Given a head entity and a relation, return top k ranked entities

        argmax_{e \in E } f(h,r,e), where h \in E and r \in R.


        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        tail_entity: List[str]

        String representation of selected entities.

        Returns: Tuple
        ---------

        scores
        """
        head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity]).unsqueeze(-1)
        relation = torch.LongTensor([self.relation_to_idx[i] for i in relation]).unsqueeze(-1)
        return self.model(torch.cat((head_entity, relation), dim=1))

    def predict(self, *, head_entities: List[str] = None, relations: List[str] = None, tail_entities: List[str] = None):
        # (1) Sanity checking.
        if head_entities is not None:
            assert isinstance(head_entities, list)
            assert isinstance(head_entities[0], str)
        if relations is not None:
            assert isinstance(relations, list)
            assert isinstance(relations[0], str)
        if tail_entities is not None:
            assert isinstance(tail_entities, list)
            assert isinstance(tail_entities[0], str)
        # (2) Predict missing head entity given a relation and a tail entity.
        if head_entities is None:
            assert relations is not None
            assert tail_entities is not None
            # ? r, t
            scores = self.__predict_missing_head_entity(relations, tail_entities)
        # (3) Predict missing relation given a head entity and a tail entity.
        elif relations is None:
            assert head_entities is not None
            assert tail_entities is not None
            # h ? t
            scores = self.__predict_missing_relations(head_entities, tail_entities)
        # (4) Predict missing tail entity given a head entity and a relation
        elif tail_entities is None:
            assert head_entities is not None
            assert relations is not None
            # h r ?
            scores = self.__predict_missing_tail_entity(head_entities, relations)
        else:
            assert len(head_entities) == len(relations) == len(tail_entities)
            scores = self.triple_score(head_entities, relations, tail_entities)
        return torch.sigmoid(scores)

    def predict_topk(self, *, head_entity: List[str] = None, relation: List[str] = None, tail_entity: List[str] = None,
                     topk: int = 10):
        """
        Predict missing item in a given triple.



        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.


        k: int

        Highest ranked k item.

        Returns: Tuple
        ---------

        Highest K scores and items
        """

        # (1) Sanity checking.
        if head_entity is not None:
            assert isinstance(head_entity, list)
        if relation is not None:
            assert isinstance(relation, list)
        if tail_entity is not None:
            assert isinstance(tail_entity, list)
        # (2) Predict missing head entity given a relation and a tail entity.
        if head_entity is None:
            assert relation is not None
            assert tail_entity is not None
            # ? r, t
            scores = self.__predict_missing_head_entity(relation, tail_entity).flatten()
            sort_scores, sort_idxs = torch.topk(scores, topk)
            return torch.sigmoid(sort_scores), [self.idx_to_entity[i] for i in sort_idxs.tolist()]
        # (3) Predict missing relation given a head entity and a tail entity.
        elif relation is None:
            assert head_entity is not None
            assert tail_entity is not None
            # h ? t
            scores = self.__predict_missing_relations(head_entity, tail_entity).flatten()
            sort_scores, sort_idxs = torch.topk(scores, topk)
            return torch.sigmoid(sort_scores), [self.idx_to_relations[i] for i in sort_idxs.tolist()]
        # (4) Predict missing tail entity given a head entity and a relation
        elif tail_entity is None:
            assert head_entity is not None
            assert relation is not None
            # h r ?t
            scores = self.__predict_missing_tail_entity(head_entity, relation).flatten()
            sort_scores, sort_idxs = torch.topk(scores, topk)
            return torch.sigmoid(sort_scores), [self.idx_to_entity[i] for i in sort_idxs.tolist()]
        else:

            assert len(head_entity) == len(relation) == len(tail_entity)
        # @TODO:replace with triple_score
        head = [self.entity_to_idx[i] for i in head_entity]
        relation = [self.relation_to_idx[i] for i in relation]
        tail = [self.entity_to_idx[i] for i in tail_entity]
        x = torch.LongTensor((head, relation, tail)).reshape(len(head), 3)
        return torch.sigmoid(self.model(x))

    def triple_score(self, head_entity: List[str] = None, relation: List[str] = None,
                     tail_entity: List[str] = None, logits=False) -> torch.FloatTensor:
        """
        Predict triple score

        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.

        logits: bool

        If logits is True, unnormalized score returned

        Returns: Tuple
        ---------

        pytorch tensor of triple score
        """
        head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity]).reshape(len(head_entity), 1)
        relation = torch.LongTensor([self.relation_to_idx[i] for i in relation]).reshape(len(relation), 1)
        tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity]).reshape(len(tail_entity), 1)

        x = torch.hstack((head_entity, relation, tail_entity))
        if self.apply_semantic_constraint:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                out = self.model(x)
                if logits:
                    return out
                else:
                    return torch.sigmoid(out)

    @property
    def name(self):
        return self.model.name

    def sample_entity(self, n: int) -> List[str]:
        assert isinstance(n, int)
        assert n >= 0
        return random.sample(self.entity_to_idx.keys(), n)

    def sample_relation(self, n: int) -> List[str]:
        assert isinstance(n, int)
        assert n >= 0
        return random.sample(self.relation_to_idx.keys(), n)

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

    def construct_input_and_output_k_vs_all(self, head_entity, relation):
        raise NotImplementedError()
        # @TODO: Add explanation
        try:
            idx_head_entity = self.entity_to_idx.loc[head_entity]['entity'].values[0]
            idx_relation = self.relation_to_idx.loc[relation]['relation'].values[0]
        except KeyError as e:
            print(f'Exception:\t {str(e)}')
            return None

        print('\nKvsAll Training...')
        print(f'Start:{head_entity}\t {relation}')
        idx_tails: np.array
        idx_tails = self.train_set[
            (self.train_set['subject'] == idx_head_entity) & (self.train_set['relation'] == idx_relation)][
            'object'].values
        print('Num. Tails:\t', self.entity_to_idx.iloc[idx_tails].values.size)
        # Hard Labels
        labels = torch.zeros(1, self.num_entities)
        labels[0, idx_tails] = 1
        x = torch.LongTensor([idx_head_entity, idx_relation]).reshape(1, 2)
        return x, labels, idx_tails

    def get_cooccuring_relations_given_entity(self, entity: str) -> List[str]:
        """
        Given an entity return relations that occur with this entity regarless of its positition
        :param entity:
        :return:
        """
        idx_entity = self.entity_to_idx.loc[entity].values[0]
        idx_relations = self.train_set[
            (self.train_set['subject'] == idx_entity) | (self.train_set['object'] == idx_entity)][
            'relation'].unique()
        # => relation_to_idx must be a dataframe with monotonically increasing
        return self.relation_to_idx.iloc[idx_relations].index.values.tolist()

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


class AbstractCallback(ABC):
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

    def on_train_batch_end(self, trainer, model):
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

    def on_fit_end(self, trainer, model):
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
                    # (2) Update the parameter ensemble model with the current model.
                    param_ensemble[k] += self.alphas[self.sample_counter] * v
            # (3) Save the updated parameter ensemble model.
            torch.save(param_ensemble, f=f"{self.path}/trainer_checkpoint_main.pt")
            self.sample_counter += 1

        self.epoch_to_start -= 1

    def on_train_batch_end(self, *args, **kwargs):
        return
