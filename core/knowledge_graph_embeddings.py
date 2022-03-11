import os
from .static_funcs import load_json, load_model
from typing import List
import torch


class KGE:
    def __init__(self, path_of_pretrained_model_dir):
        try:
            assert os.path.isdir(path_of_pretrained_model_dir)
        except AssertionError:
            raise AssertionError(f'Could not find a directory {path_of_pretrained_model_dir}')
        self.path = path_of_pretrained_model_dir

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

    def predict(self, *, head_entity: list = None, relation: list = None, tail_entity: list = None, k=10):
        """

        :param head_entity:
        :param relation:
        :param tail_entity:
        :return:
        """
        if head_entity is None:
            assert relation is not None
            assert tail_entity is not None
            # ? r, t
            scores, entities = self.predict_missing_head_entity(relation, tail_entity)
            return (x for x in zip(torch.sigmoid(scores[:k]), entities[:k]))

        elif relation is None:
            assert head_entity is not None
            assert tail_entity is not None
            # h ? t
            scores, relations = self.predict_missing_relations(head_entity, tail_entity)
            return (x for x in zip(torch.sigmoid(scores[:k]), relations[:k]))
        elif tail_entity is None:
            assert head_entity is not None
            assert relation is not None
            # h r ?t
            scores, entities = self.predict_missing_tail_entity(head_entity, relation)
            return (x for x in zip(torch.sigmoid(scores[:k]), entities[:k]))
        else:
            assert len(head_entity) == len(relation) == len(tail_entity)
        head = self.entity_to_idx.loc[head_entity]['entity'].values.tolist()
        relation = self.relation_to_idx.loc[relation]['relation'].values.tolist()
        tail = self.entity_to_idx.loc[tail_entity]['entity'].values.tolist()
        x = torch.tensor((head, relation, tail)).reshape(len(head), 3)
        return torch.sigmoid(self.model.forward_triples(x))

    @property
    def name(self):
        return self.model.name

    def sample_entity(self, n: int) -> List[str]:
        assert isinstance(n, int)
        assert n >= 0
        return self.entity_to_idx.sample(n=n, random_state=1).index.to_list()

    def sample_relation(self, n: int) -> List[str]:
        assert isinstance(n, int)
        assert n >= 0
        return self.relation_to_idx.sample(n=n, random_state=1).index.to_list()

    def is_seen(self, entity: str = None, relation: str = None) -> bool:
        if entity is not None:
            return True if entity in self.entity_to_idx.index else False
        if relation is not None:
            return True if entity in self.relation_to_idx.index else False
