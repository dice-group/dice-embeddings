import os
from typing import List, Tuple
import torch
from torch import optim
from torch.utils.data import DataLoader
from .abstracts import BaseInteractiveKGE
from .dataset_classes import TriplePredictionDataset


class KGE(BaseInteractiveKGE):
    """ Knowledge Graph Embedding Class for interactive usage of pre-trained models"""

    def __init__(self, path_of_pretrained_model_dir, construct_ensemble=False, model_path=None):
        super().__init__(path_of_pretrained_model_dir, construct_ensemble=construct_ensemble, model_path=model_path)
        self.is_model_in_train_mode = False

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

    def train_triples(self, head_entity, relation, tail_entity, labels, iteration=2, lr=.1, repeat=2):
        """

        :param head_entity:
        :param relation:
        :param tail_entity:
        :param labels:
        :param iteration:
        :param lr:
        :param repeat:
        :return:
        """
        assert len(head_entity) == len(relation) == len(tail_entity) == len(labels)
        x, labels = self.construct_input_and_output_k_vs_all(head_entity, relation, tail_entity, labels)
        x = x.num_copies_in_batch(repeat, 1)
        labels = labels.num_copies_in_batch(repeat)
        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('Iteration starts.')
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            print(f"Iteration:{epoch}\t Loss:{loss.item():.4f}\t Outputs:{outputs.detach().mean()}")
            loss.backward()
            optimizer.step()
        self.set_model_eval_mode()
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
        print(f"Eval Mode:Loss:{loss.item():.4f}\t Outputs:{outputs.detach()}")

    def train(self, kg, lr=.1, epoch=10, batch_size=32, neg_sample_ratio=10, num_workers=1) -> None:
        """ Retrained a pretrain model on an input KG via negative sampling."""
        # (1) Create Negative Sampling Setting for training
        print('Creating Dataset...')
        train_set = TriplePredictionDataset(kg.train_set,
                                            num_entities=len(kg.entity_to_idx),
                                            num_relations=len(kg.relation_to_idx),
                                            neg_sample_ratio=neg_sample_ratio)
        num_data_point = len(train_set)
        print('Number of data points: ', num_data_point)
        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                      #  shuffle => to have the data reshuffled at every epoc
                                      shuffle=True, num_workers=num_workers,
                                      collate_fn=train_set.collate_fn, pin_memory=True)

        # (2) Go through valid triples + corrupted triples and compute scores.
        # Average loss per triple is stored. This will be used  to indicate whether we learned something.
        print('First Eval..')
        self.set_model_eval_mode()
        first_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            first_avg_loss_per_triple += self.model.loss(pred, y)
        first_avg_loss_per_triple /= num_data_point
        print(first_avg_loss_per_triple)
        # (3) Prepare Model for Training
        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('Training Starts...')
        for epoch in range(epoch):  # loop over the dataset multiple times
            epoch_loss = 0
            for x, y in train_dataloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(x)
                loss = self.model.loss(outputs, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Epoch={epoch}\t Avg. Loss per epoch: {epoch_loss / num_data_point:.3f}')
        # (5) Prepare For Saving
        self.set_model_eval_mode()
        print('Eval starts...')
        # (6) Eval model on training data to check how much an Improvement
        last_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            last_avg_loss_per_triple += self.model.loss(pred, y)
        last_avg_loss_per_triple /= len(train_set)
        print(f'On average Improvement: {first_avg_loss_per_triple - last_avg_loss_per_triple:.3f}')

    def train_k_vs_all(self, head_entity, relation, iteration=1, num_copies_in_batch=2, lr=.001):
        assert len(head_entity) == 1
        out = self.construct_input_and_output_k_vs_all(head_entity, relation)
        if out is None:
            return
        x, labels, idx_tails = out
        x = x.repeat(num_copies_in_batch, 1)
        # TODO: Apply Label Smoothing
        labels = labels.repeat(num_copies_in_batch, 1)
        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('\nIteration starts.')
        converged = False
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            if epoch % 10 == 0:
                if len(idx_tails) > 0:
                    print(
                        f"Iteration:{epoch}\t Loss:{loss.item():.4f}\t Avg. Logits for correct tails: {outputs[0, idx_tails].flatten().mean().detach():.4f}")
                else:
                    print(
                        f"Iteration:{epoch}\t Loss:{loss.item():.4f}\t Avg. Logits for all negatives: {outputs[0].flatten().mean().detach():.4f}")

            loss.backward()
            optimizer.step()
            if loss.item() < .001:
                print(f'loss is {loss.item():.3f}. Converged !!!')
                converged = True
                break
        self.set_model_eval_mode()
        if converged is False:
            with torch.no_grad():
                outputs = self.model(x)
                loss = self.model.loss(outputs, labels)
            print(f"Eval Mode:Loss:{loss.item():.4f}\t Outputs:{outputs[0, idx_tails].flatten().detach()}\n")

    def train_cbd(self, head_entity, iteration=1, num_copies_in_batch=1, lr=.001, label_smoothing_rate=.1):
        """
        Given an head_entity,
        1) Build {r | (h r x) \in G)
        2) Build x:=(h,r), y=[0.....,1]
        3) Construct (2) as a batch
        4) Train
        """
        assert len(head_entity) == 1
        try:
            idx_head_entity = self.entity_to_idx.loc[head_entity]['entity'].values[0]
        except KeyError as e:
            print(f'Exception:\t {str(e)}')
            return
        print('\nKvsAll Training...')

        batch_relations = []
        # (1)
        idx_batch_relations = self.train_set[self.train_set['subject'] == idx_head_entity]['relation'].unique()
        # (2)
        num_unique_relations = len(idx_batch_relations)
        batch_labels = torch.zeros(num_unique_relations, self.num_entities)
        for i, idx_relation in enumerate(idx_batch_relations):
            # Select tails.
            idx_tails = self.train_set[
                (self.train_set['subject'] == idx_head_entity) & (self.train_set['relation'] == idx_relation)][
                'object'].values
            batch_labels[i, idx_tails] = 1
            batch_relations.append(idx_relation)
        # (3) Construct the batch
        x = torch.cat([torch.LongTensor([idx_head_entity]).repeat(num_unique_relations, 1),
                       torch.LongTensor(batch_relations).reshape(len(batch_relations), 1)], dim=1)
        # Label Smoothing
        try:
            batch_labels = batch_labels * (1 - label_smoothing_rate) + (1 / batch_labels.size(0))
        except:
            print('Empty label batch')
            return
        # Create a BATCH via repeating.
        x = x.repeat(num_copies_in_batch, 1)
        batch_labels = batch_labels.repeat(num_copies_in_batch, 1)

        # (4) Train
        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('\nIteration starts.')
        converged = False
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, batch_labels)
            print(f"Iteration:{epoch}\t Loss:{loss.item():.4f}")
            loss.backward()
            optimizer.step()
            if loss.item() < .001:
                print(f'loss is {loss.item():.3f}. Converged !!!')
                converged = True
                break
        self.set_model_eval_mode()
        if converged is False:
            with torch.no_grad():
                outputs = self.model(x)
                loss = self.model.loss(outputs, batch_labels)
                print(f"Eval Mode:Loss:{loss.item():.4f}")

    def train_triples_lbfgs_negative(self, head_entity, relation, tail_entity, iteration=1, repeat=2):
        """ This training regime with LBFGS often takes quite a bit of timeTakes quite some time"""

        n = len(head_entity)
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(n, 1)
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(n, 1)
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(n, 1)
        x = torch.hstack((head_entity, relation, tail_entity))
        labels: object = torch.zeros(n)
        x = x.repeat(repeat, 1)
        labels = labels.repeat(repeat)
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

    def train_triples_lbfgs_positive(self, head_entity, relation, tail_entity, iteration=1, repeat=2):
        """ This training regime with LBFGS often takes quite a bit of timeTakes quite some time"""

        n = len(head_entity)
        head_entity = torch.LongTensor(self.entity_to_idx.loc[head_entity]['entity'].values).reshape(n, 1)
        relation = torch.LongTensor(self.relation_to_idx.loc[relation]['relation'].values).reshape(n, 1)
        tail_entity = torch.LongTensor(self.entity_to_idx.loc[tail_entity]['entity'].values).reshape(n, 1)
        x = torch.hstack((head_entity, relation, tail_entity))
        labels: object = torch.ones(n)
        x = x.repeat(repeat, 1)
        labels = labels.repeat(repeat)
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
