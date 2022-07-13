import os
import datetime
from .static_funcs import load_model_ensemble, load_model, store_kge, create_constraints
from typing import List
import torch
from typing import List, Tuple, Generator
import pandas as pd


class BaseInteractiveKGE:
    """ Base class for interactive KGE """

    def __init__(self, path_of_pretrained_model_dir, construct_ensemble=False, model_name=None):
        try:
            assert os.path.isdir(path_of_pretrained_model_dir)
        except AssertionError:
            raise AssertionError(f'Could not find a directory {path_of_pretrained_model_dir}')
        self.path = path_of_pretrained_model_dir
        # (1) Load model...
        self.construct_ensemble = construct_ensemble
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
        print('Loading indexed training data...')
        self.train_set = pd.read_parquet(self.path + '/idx_train_df.gzip')

        # TODO: 1 Obtain a mapping from a relation to its ranges
        # TODO: 2 Convert 2 into a mapping from relations to entities outside of their ranges
        self.domain_constraints_per_rel, self.range_constraints_per_rel = create_constraints(self.train_set.to_numpy())
        # TODO 3 Use 2 at predicting scores.

    def set_model_train_mode(self):
        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def set_model_eval_mode(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def __predict_missing_head_entity(self, relation: List[str], tail_entity: List[str], k: int) -> Tuple:
        """ f(? r t) for all entities.
        :param k:
        :param relation: list of URIs
        :param tail_entity: list of URIs
        :return:
        """
        assert k >= 0

        head_entity = torch.LongTensor(self.entity_to_idx['entity'].values.tolist())
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values.tolist())
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values.tolist())
        x = torch.stack((head_entity,
                         relation.repeat(self.num_entities, ),
                         tail_entity.repeat(self.num_entities, )), dim=1)
        scores = self.model(x)
        entities = self.entity_to_idx.index.values
        # sort_scores, sort_idxs = torch.sort(scores, descending=True)
        sort_scores, sort_idxs = torch.topk(scores, k)
        return sort_scores, entities[sort_idxs]

    def __predict_missing_relations(self, head_entity: List[str], tail_entity: List[str], k: int = 3) -> Tuple:
        assert k >= 0

        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values.tolist())
        relation = torch.LongTensor(self.relation_to_idx['relation'].values.tolist())
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values.tolist())
        x = torch.stack((head_entity.repeat(self.num_relations, ),
                         relation,
                         tail_entity.repeat(self.num_relations, )), dim=1)
        scores = self.model(x)
        relations = self.relation_to_idx.index.values
        # sort_scores, sort_idxs = torch.sort(scores, descending=True)
        sort_scores, sort_idxs = torch.topk(scores, k)
        return sort_scores, relations[sort_idxs]

    def __predict_missing_tail_entity(self, head_entity: List[str], relation: List[str], k: int = 3) -> Tuple:
        assert k >= 0
        # Get index of head entity
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values.tolist())
        # Get index of relation
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values.tolist())
        # Get all entity indexes.
        tail_entity = torch.LongTensor(self.entity_to_idx['entity'].values.tolist())

        x = torch.stack((head_entity.repeat(self.num_entities, ),
                         relation.repeat(self.num_entities, ),
                         tail_entity), dim=1)
        scores = self.model(x)
        entities = self.entity_to_idx.index.values
        # sort_scores, sort_idxs = torch.sort(scores, descending=True)
        sort_scores, sort_idxs = torch.topk(scores, k)
        return sort_scores, entities[sort_idxs]

    def predict_topk(self, *, head_entity: List[str] = None, relation: List[str] = None, tail_entity: List[str] = None,
                     k: int = 10) -> Generator:
        """
        Predict missing triples

        :param k: top k prediction
        :param head_entity:
        :param relation:
        :param tail_entity:
        :return:
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
            scores, entities = self.__predict_missing_head_entity(relation, tail_entity, k)
            return torch.sigmoid(scores), entities
        # (3) Predict missing relation given a head entity and a tail entity.
        elif relation is None:
            assert head_entity is not None
            assert tail_entity is not None
            # h ? t
            scores, relations = self.__predict_missing_relations(head_entity, tail_entity, k)
            return torch.sigmoid(scores), relations
        # (4) Predict missing tail entity given a head entity and a relation
        elif tail_entity is None:
            assert head_entity is not None
            assert relation is not None
            # h r ?t
            scores, entities = self.__predict_missing_tail_entity(head_entity, relation, k)
            return torch.sigmoid(scores), entities
        else:

            assert len(head_entity) == len(relation) == len(tail_entity)
        # @TODO:replace with triple_score
        head = self.entity_to_idx.loc[head_entity]['entity'].values.tolist()
        relation = self.relation_to_idx.loc[relation]['relation'].values.tolist()
        tail = self.entity_to_idx.loc[tail_entity]['entity'].values.tolist()
        x = torch.tensor((head, relation, tail)).reshape(len(head), 3)
        return torch.sigmoid(self.model(x))

    def triple_score(self, *, head_entity: List[str] = None, relation: List[str] = None,
                     tail_entity: List[str] = None, logits=False, without_norm=False) -> torch.tensor:
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(len(head_entity),
                                                                                                     1)
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(len(relation), 1)
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(len(tail_entity),
                                                                                                     1)
        x = torch.hstack((head_entity, relation, tail_entity))
        # @ TODO: Apply semantic filtering
        with torch.no_grad():
            if without_norm:
                out = self.model.forward_without_norm(x)
            else:
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
        return self.entity_to_idx.sample(n=n).index.to_list()

    def sample_relation(self, n: int) -> List[str]:
        assert isinstance(n, int)
        assert n >= 0
        return self.relation_to_idx.sample(n=n).index.to_list()

    def is_seen(self, entity: str = None, relation: str = None) -> bool:
        if entity is not None:
            return True if entity in self.entity_to_idx.index else False
        if relation is not None:
            return True if relation in self.relation_to_idx.index else False

    def save(self) -> None:
        t = str(datetime.datetime.now())
        if self.construct_ensemble:
            store_kge(self.model, path=self.path + f'/model_ensemble_interactive_{str(t)}.pt')
        else:
            store_kge(self.model, path=self.path + f'/model_interactive_{str(t)}.pt')

    def index_triple(self, head_entity: List[str], relation: List[str], tail_entity: List[str]):
        """

        :param head_entity:
        :param relation:
        :param tail_entity:
        :return:
        """
        print('Index inputs...')
        n = len(head_entity)
        assert n == len(relation) == len(tail_entity)
        idx_head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(n, 1)
        idx_relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(n, 1)
        idx_tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(n, 1)
        return idx_head_entity, idx_relation, idx_tail_entity

    def construct_input_and_output_k_vs_all(self, head_entity, relation):
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
