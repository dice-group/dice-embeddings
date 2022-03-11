import os
from .static_funcs import load_json, load_model
from typing import List, Tuple
import torch
from torch import optim
from .abstracts import BaseInteractiveKGE

from .dataset_classes import TriplePredictionDataset
from torch.utils.data import DataLoader


class KGE(BaseInteractiveKGE):
    """ Knowledge Graph Embedding Class for interactive usage of pre-trained models"""

    def __init__(self, path_of_pretrained_model_dir):
        super().__init__(path_of_pretrained_model_dir)

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

    def train(self, kg, lr=.01, epoch=3, batch_size=32):
        # (1) Create Negative Sampling Setting for training
        print('Creating Dataset...')
        train_set = TriplePredictionDataset(kg.train_set,
                                            num_entities=len(kg.entity_to_idx),
                                            num_relations=len(kg.relation_to_idx),
                                            neg_sample_ratio=10)
        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=1,
                                      collate_fn=train_set.collate_fn, pin_memory=True)

        print('First Eval..')
        # (2) Eval model on this triples
        first_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            first_avg_loss_per_triple += self.model.loss(pred, y)
        first_avg_loss_per_triple /= len(train_set)
        print(first_avg_loss_per_triple)
        # (3) Prepare Model for Training
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self.model.train()
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
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
        print('Eval starts...')
        # (6) Eval model on training data to check how much an Improvement we achived
        last_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            last_avg_loss_per_triple += self.model.loss(pred, y)
        last_avg_loss_per_triple /= len(train_set)
        print(last_avg_loss_per_triple)
        print(f'On average Improvement: {first_avg_loss_per_triple-last_avg_loss_per_triple}')
        torch.save(self.model, 'ContinualTraining_model.pt')
