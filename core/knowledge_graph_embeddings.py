import os
from typing import List, Tuple, Generator
import torch
from torch import optim
from torch.utils.data import DataLoader

from .abstracts import BaseInteractiveKGE
from .dataset_classes import TriplePredictionDataset, OneVsAllEntityPredictionDataset
from .static_funcs import load_json, load_model


class KGE(BaseInteractiveKGE):
    """ Knowledge Graph Embedding Class for interactive usage of pre-trained models"""

    def __init__(self, path_of_pretrained_model_dir, construct_ensemble=False, model_path=None):
        super().__init__(path_of_pretrained_model_dir, construct_ensemble=construct_ensemble, model_path=model_path)
        self.is_model_in_train_mode = False

    def set_model_train_mode(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self.model.train()

    def set_model_eval_mode(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def predict_topk(self, *, head_entity: list = None, relation: list = None, tail_entity: list = None,
                     k: int = 10) -> Generator:
        """
        :param k: top k prediction
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
            return torch.sigmoid(scores[:k]), entities[:k]

        elif relation is None:
            assert head_entity is not None
            assert tail_entity is not None
            # h ? t
            scores, relations = self.predict_missing_relations(head_entity, tail_entity)
            return torch.sigmoid(scores[:k]), relations[:k]
        elif tail_entity is None:
            assert head_entity is not None
            assert relation is not None
            # h r ?t
            scores, entities = self.predict_missing_tail_entity(head_entity, relation)
            return torch.sigmoid(scores[:k]), entities[:k]
        else:
            assert len(head_entity) == len(relation) == len(tail_entity)
        head = self.entity_to_idx.loc[head_entity]['entity'].values.tolist()
        relation = self.relation_to_idx.loc[relation]['relation'].values.tolist()
        tail = self.entity_to_idx.loc[tail_entity]['entity'].values.tolist()
        x = torch.tensor((head, relation, tail)).reshape(len(head), 3)
        return torch.sigmoid(self.model.forward_triples(x))

    def triple_score(self, *, head_entity: list = None, relation: list = None,
                     tail_entity: list = None) -> torch.tensor:
        head = self.entity_to_idx.loc[head_entity]['entity'].values.tolist()
        relation = self.relation_to_idx.loc[relation]['relation'].values.tolist()
        tail = self.entity_to_idx.loc[tail_entity]['entity'].values.tolist()
        x = torch.tensor((head, relation, tail)).reshape(len(head), 3)
        return torch.sigmoid(self.model.forward_triples(x))

    def train(self, kg, lr=.1, epoch=10, batch_size=32, neg_sample_ratio=10, num_workers=1) -> None:
        # (1) Create Negative Sampling Setting for training
        print('Creating Dataset...')
        train_set = TriplePredictionDataset(kg.train_set,
                                            num_entities=len(kg.entity_to_idx),
                                            num_relations=len(kg.relation_to_idx),
                                            neg_sample_ratio=neg_sample_ratio)
        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      collate_fn=train_set.collate_fn, pin_memory=True)

        print('First Eval..')
        self.set_model_eval_mode()
        # (2) Eval model on this triples
        first_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            first_avg_loss_per_triple += self.model.loss(pred, y)
        first_avg_loss_per_triple /= len(train_set)
        print(first_avg_loss_per_triple)
        # (3) Prepare Model for Training
        self.set_model_train_mode()
        # (4) Start Training
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('Training Starts...')
        for epoch in range(epoch):  # loop over the dataset multiple times
            for x, y in train_dataloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(x)
                loss = self.model.loss(outputs, y)
                loss.backward()
                optimizer.step()
        # (5) Prepare For Saving
        self.set_model_eval_mode()

        print('Eval starts...')
        # (6) Eval model on training data to check how much an Improvement
        last_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            last_avg_loss_per_triple += self.model.loss(pred, y)
        last_avg_loss_per_triple /= len(train_set)
        print(f'On average Improvement: {first_avg_loss_per_triple - last_avg_loss_per_triple}:.3f')

    def train_triples(self, head_entity, relation, tail_entity, labels, lr=.1):

        assert len(head_entity) == len(relation) == len(tail_entity) == len(labels)
        n = len(head_entity)
        try:
            head = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values.tolist())
            relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values.tolist())
            tail = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values.tolist())
        except KeyError as e:
            print(f'Ensure that {head_entity}, {relation}, {tail_entity} can be found in the input KG.')
            raise e
        x = torch.cat((head, relation, tail), 0).reshape(n, 3)
        y: object = torch.FloatTensor(labels)
        x = x.repeat(4, 1)
        y = y.repeat(4)

        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(10):  # loop over the dataset multiple times
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.model(x)
            loss = self.model.loss(outputs, y)
            loss.backward()
            optimizer.step()

        self.set_model_eval_mode()
