import os
from .static_funcs import load_model
from typing import List

class BaseInteractiveKGE:
    def __init__(self, path_of_pretrained_model_dir):
        try:
            assert os.path.isdir(path_of_pretrained_model_dir)
        except AssertionError:
            raise AssertionError(f'Could not find a directory {path_of_pretrained_model_dir}')
        self.path = path_of_pretrained_model_dir
        # (1) Load model...
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
