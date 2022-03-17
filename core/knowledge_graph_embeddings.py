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
        # Workaround due to BN output
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
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(len(head_entity),
                                                                                                     1)
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(len(relation), 1)
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(len(tail_entity),
                                                                                                     1)
        x = torch.hstack((head_entity, relation, tail_entity))
        with torch.no_grad():
            return self.model.forward_triples_multiply(x)

    def indexed_triple_score(self, i):
        i = torch.LongTensor(i).reshape(1, 3)
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.LongTensor(i)))

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

    def train_triples(self, head_entity, relation, tail_entity, labels, iteration=2, lr=.1, repeat=2):

        assert len(head_entity) == len(relation) == len(tail_entity) == len(labels)
        n = len(head_entity)
        print('Index inputs...')
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(n, 1)
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(n, 1)
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(n, 1)

        x = torch.hstack((head_entity, relation, tail_entity))
        labels: object = torch.FloatTensor(labels)

        x = x.repeat(repeat, 1)
        labels = labels.repeat(repeat)

        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('Iteration starts.')
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model.forward_triples_multiply(x)
            loss = self.model.loss(outputs, labels)
            print(f"Iteration:{epoch}\t Loss:{loss.item():.4f}\t Outputs:{outputs.detach()}")
            loss.backward()
            optimizer.step()

        self.set_model_eval_mode()
        with torch.no_grad():
            outputs = self.model.forward_triples_multiply(x)
            loss = self.model.loss(outputs, labels)
        print(f"Eval Mode:Loss:{loss.item():.4f}\t Outputs:{outputs.detach()}")

    def train_triples_lbfgs(self, head_entity, relation, tail_entity, labels, iteration=100):

        assert len(head_entity) == len(relation) == len(tail_entity) == len(labels)
        n = len(head_entity)

        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(n, 1)
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(n, 1)
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(n, 1)

        x = torch.hstack((head_entity, relation, tail_entity))
        labels: object = torch.FloatTensor(labels)

        if n == 1:
            x = x.repeat(2, 1)
            labels = labels.repeat(2)

        self.set_model_train_mode()
        optimizer = optim.LBFGS(self.model.parameters())

        for epoch in range(iteration):  # loop over the dataset multiple times
            def closure():
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.model.loss(outputs, labels)
                loss.backward()
                return loss

            # Take step.
            optimizer.step(closure)

        self.set_model_eval_mode()

    def train_vs_all(self, head_entity, relation, tail_entity, iteration=1, lr=.01):
        raise NotImplementedError('Undesired')
        self.set_model_eval_mode()
        assert len(head_entity) == len(relation) == len(tail_entity)
        n = len(head_entity)

        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(n, 1)
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(n, 1)
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(n, 1)

        x = torch.hstack((head_entity, relation))

        outputs = self.model(x)
        y = torch.zeros(outputs.shape)

        y[:, tail_entity] = 1

        if n == 1:
            x = x.repeat(2, 1)
            y = y.repeat(2, 1)

        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(iteration):  # loop over the dataset multiple times
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = self.model(x)
            loss = self.model.loss(outputs, y)
            loss.backward()
            optimizer.step()

        self.set_model_eval_mode()
