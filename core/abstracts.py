import os
import datetime
from .static_funcs import load_model_ensemble, load_model, store_kge
from typing import List
import torch


class BaseInteractiveKGE:
    def __init__(self, path_of_pretrained_model_dir, construct_ensemble=False, model_path=None):
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
            if model_path:
                self.model, self.entity_to_idx, self.relation_to_idx = load_model(self.path + '/',
                                                                                  model_path=model_path)
            else:
                self.model, self.entity_to_idx, self.relation_to_idx = load_model(self.path + '/')

        self.num_entities = len(self.entity_to_idx)
        self.num_relations = len(self.relation_to_idx)

    def predict_missing_head_entity(self, relation, tail_entity):
        """

        :param relation:
        :param tail_entity:
        :return:
        """
        head_entity = torch.LongTensor(self.entity_to_idx['entity'].values.tolist())
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values.tolist())
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values.tolist())
        x = torch.stack((head_entity,
                         relation.repeat(self.num_entities, ),
                         tail_entity.repeat(self.num_entities, )), dim=1)
        scores = self.model.forward_triples(x)
        entities = self.entity_to_idx.index.values
        sort_scores, sort_idxs = torch.sort(scores, descending=True)
        return sort_scores, entities[sort_idxs]

    def predict_missing_relations(self, head_entity, tail_entity):
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values.tolist())
        relation = torch.LongTensor(self.relation_to_idx['relation'].values.tolist())
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values.tolist())

        x = torch.stack((head_entity.repeat(self.num_relations, ),
                         relation,
                         tail_entity.repeat(self.num_relations, )), dim=1)
        scores = self.model.forward_triples(x)
        relations = self.relation_to_idx.index.values
        sort_scores, sort_idxs = torch.sort(scores, descending=True)
        return sort_scores, relations[sort_idxs]

    def predict_missing_tail_entity(self, head_entity, relation):
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values.tolist())
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values.tolist())
        tail_entity = torch.LongTensor(self.entity_to_idx['entity'].values.tolist())

        x = torch.stack((head_entity.repeat(self.num_entities, ),
                         relation.repeat(self.num_entities, ),
                         tail_entity), dim=1)
        scores = self.model.forward_triples(x)
        entities = self.entity_to_idx.index.values
        sort_scores, sort_idxs = torch.sort(scores, descending=True)
        return sort_scores, entities[sort_idxs]

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

    def save(self):
        t = str(datetime.datetime.now())

        if self.construct_ensemble:
            store_kge(self.model, path=self.path + f'/model_ensemble_interactive_{str(t)}.pt')
        else:
            store_kge(self.model, path=self.path + f'/model_interactive_{str(t)}.pt')
