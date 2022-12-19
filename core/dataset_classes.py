import time
from abc import ABCMeta

from torch.utils.data import Dataset
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List
import random


class StandardDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """ Data Class for creating train/val/test datasets depending on the training strategy chosen """

    def __init__(self, train_set_idx, entity_to_idx, relation_to_idx, batch_size, form,
                 num_workers=None, valid_set_idx=None, test_set_idx=None, neg_sample_ratio=None,
                 label_smoothing_rate=None):
        super().__init__()
        assert isinstance(train_set_idx, np.ndarray)

        if valid_set_idx is not None:
            if len(valid_set_idx) > 0:
                assert isinstance(valid_set_idx, np.ndarray)
        if test_set_idx is not None:
            if len(test_set_idx) > 0:
                assert isinstance(test_set_idx, np.ndarray)

        assert isinstance(entity_to_idx, dict)
        assert isinstance(relation_to_idx, dict)

        self.train_set_idx = train_set_idx
        self.valid_set_idx = valid_set_idx
        self.test_set_idx = test_set_idx

        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx

        self.form = form
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.neg_sample_ratio = neg_sample_ratio
        self.label_smoothing_rate = label_smoothing_rate

        if self.form == 'RelationPrediction':
            self.target_dim = len(self.relation_to_idx)
        elif self.form == 'EntityPrediction':
            self.target_dim = len(self.entity_to_idx)
        elif self.form == 'NegativeSampling':  # we can name it as TriplePrediction
            self.dataset_type_class = TriplePredictionDataset
            self.target_dim = 1
            self.neg_sample_ratio = neg_sample_ratio

        elif self.form == '1VsAll':
            # Multi-class
            self.dataset = OnevsAllDataset(self.train_set_idx, entity_idxs=self.entity_to_idx,
                                           relation_idxs=self.relation_to_idx, form=self.form)
        elif self.form == 'CCvsAll':
            # Multi-class
            self.dataset = OnevsAllDataset(self.train_set_idx, entity_idxs=self.entity_to_idx,
                                           relation_idxs=self.relation_to_idx, form=self.form)
        elif self.form == 'PvsAll':
            # Multi-class
            self.dataset = OnevsAllDataset(self.train_set_idx, entity_idxs=self.entity_to_idx,
                                           relation_idxs=self.relation_to_idx, form=self.form)
        elif self.form == 'BatchRelaxedKvsAll':
            # ?
            self.dataset = BatchRelaxedKvsAllDataset(self.train_set_idx, entity_idxs=self.entity_to_idx,
                                                     relation_idxs=self.relation_to_idx, form=self.form)
        elif self.form == 'BatchRelaxed1vsAll':
            # ?
            self.dataset = BatchRelaxed1vsAllDataset(self.train_set_idx, entity_idxs=self.entity_to_idx,
                                                     relation_idxs=self.relation_to_idx, form=self.form)
        elif self.form == 'KvsSample':
            self.dataset = KvsSampleDataset(self.train_set_idx, entity_idxs=self.entity_to_idx,
                                            relation_idxs=self.relation_to_idx, form=self.form,
                                            neg_sample_ratio=self.neg_sample_ratio,
                                            label_smoothing_rate=self.label_smoothing_rate)
        elif self.form == 'Pyke':
            self.dataset = PykeDataset(self.train_set_idx,
                                       entity_idxs=self.entity_to_idx,
                                       relation_idxs=self.relation_to_idx,
                                       form=self.form,
                                       neg_sample_ratio=self.neg_sample_ratio)
        else:
            raise ValueError(f'Invalid input : {self.form}')

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self) -> DataLoader:
        if self.form == 'NegativeSampling':
            train_set = TriplePredictionDataset(self.train_set_idx,
                                                num_entities=len(self.entity_to_idx),
                                                num_relations=len(self.relation_to_idx),
                                                neg_sample_ratio=self.neg_sample_ratio)
            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, collate_fn=train_set.collate_fn)
        elif self.form == 'EntityPrediction' or self.form == 'RelationPrediction':
            train_set = KvsAll(self.train_set_idx, entity_idxs=self.entity_to_idx,
                               relation_idxs=self.relation_to_idx, form=self.form,
                               label_smoothing_rate=self.label_smoothing_rate)
            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        elif self.form in ['KvsSample', 'PvsAll', 'CCvsAll', '1VsAll', 'BatchRelaxedKvsAll', 'BatchRelaxed1vsAll',
                           'Pyke']:
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            raise KeyError(f'{self.form} illegal input.')

    def val_dataloader(self) -> DataLoader:
        if self.form == 'NegativeSampling':
            valid_set = TriplePredictionDataset(self.valid_set_idx,
                                                num_entities=len(self.entity_to_idx),
                                                num_relations=len(self.relation_to_idx),
                                                neg_sample_ratio=0)
            return DataLoader(valid_set, batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers,
                              collate_fn=valid_set.collate_fn)
        elif self.form == 'EntityPrediction' or self.form == 'RelationPrediction':
            valid_set = KvsAll(self.valid_set_idx, entity_idxs=self.entity_to_idx,
                               relation_idxs=self.relation_to_idx, form=self.form,
                               label_smoothing_rate=self.label_smoothing_rate)
            return DataLoader(valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        elif self.form == '1VsAll':
            return DataLoader(OneVsAllEntityPredictionDataset(self.valid_set_idx), batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers)
        else:
            raise KeyError(f'{self.form} illegal input.')

    def test_dataloader(self) -> DataLoader:
        if self.form == 'NegativeSampling':
            test_set = TriplePredictionDataset(self.test_set_idx,
                                               num_entities=len(self.entity_to_idx),
                                               num_relations=len(self.relation_to_idx), )
            return DataLoader(test_set, batch_size=self.batch_size, num_workers=self.num_workers)

        elif self.form == 'EntityPrediction':
            test_set = KvsAll(self.test_set_idx, entity_idxs=self.entity_to_idx,
                              relation_idxs=self.relation_to_idx, form=self.form)
            return DataLoader(test_set, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            raise KeyError(f'{self.form} illegal input.')

    def setup(self, *args, **kwargs):
        pass

    def transfer_batch_to_device(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        # Nothing to be prepared for now.
        pass


class CVDataModule(pl.LightningDataModule):
    """
    train, valid and test sets are available.
    """

    def __init__(self, train_set, num_entities, num_relations, neg_sample_ratio, batch_size, num_workers):
        super().__init__()
        self.train_set = train_set
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        train_set = TriplePredictionDataset(self.train_set,
                                            num_entities=self.num_entities,
                                            num_relations=self.num_relations,
                                            neg_sample_ratio=self.neg_sample_ratio)
        return DataLoader(train_set, batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=train_set.collate_fn)

    def setup(self, *args, **kwargs):
        pass

    def transfer_batch_to_device(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        # Nothing to be prepared for now.
        pass


class OnevsAllDataset(Dataset):
    def __init__(self, train_set_idx, entity_idxs, relation_idxs, form):
        super().__init__()
        assert len(train_set_idx) > 0
        self.train_data = torch.torch.LongTensor(train_set_idx)
        self.target_dim = len(entity_idxs)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_data[idx, 2]] = 1

        return self.train_data[idx, :2], y_vec


class KvsAll(Dataset):
    """
    For entitiy or relation prediciton
    """

    def __init__(self, triples_idx, entity_idxs, relation_idxs, form, store=None, label_smoothing_rate=None):
        super().__init__()
        assert len(triples_idx) > 0
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = label_smoothing_rate
        self.collate_fn = None

        # (1) Create a dictionary of training data pints
        # Either from tuple of entities or tuple of an entity and a relation
        if store is None:
            store = dict()
            if form == 'RelationPrediction':
                self.target_dim = len(relation_idxs)
                for s_idx, p_idx, o_idx in triples_idx:
                    store.setdefault((s_idx, o_idx), list()).append(p_idx)
            elif form == 'EntityPrediction':
                self.target_dim = len(entity_idxs)
                for s_idx, p_idx, o_idx in triples_idx:
                    store.setdefault((s_idx, p_idx), list()).append(o_idx)
            else:
                raise NotImplementedError
        else:
            raise ValueError()
        assert len(store) > 0
        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.train_data = torch.torch.LongTensor(list(store.keys()))

        if sum([len(i) for i in store.values()]) == len(store):
            # if each s,p pair contains at most 1 entity
            self.train_target = np.array(list(store.values()), dtype=np.int64)
            try:
                assert isinstance(self.train_target[0], np.ndarray)
            except IndexError or AssertionError:
                print(self.train_target)
                exit(1)
            assert isinstance(self.train_target[0][0], np.int64)
        else:
            self.train_target = list(store.values())
            assert isinstance(self.train_target[0], list)
        del store

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class KvsSampleDataset(Dataset):
    """
    For entitiy or relation prediciton
    """

    def __init__(self, triples_idx, entity_idxs, relation_idxs, form, store=None, neg_sample_ratio: int = None,
                 label_smoothing_rate=None):
        super().__init__()
        assert len(triples_idx) > 0
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = label_smoothing_rate
        self.neg_sample_ratio = neg_sample_ratio
        self.collate_fn = None
        if self.neg_sample_ratio == 0:
            print(f'neg_sample_ratio is {neg_sample_ratio}')
            self.neg_sample_ratio = 100
        store = dict()
        self.num_entities = len(entity_idxs)
        for s_idx, p_idx, o_idx in triples_idx:
            store.setdefault((s_idx, p_idx), list()).append(o_idx)

        assert len(store) > 0
        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.train_data = torch.torch.LongTensor(list(store.keys()))

        if sum([len(i) for i in store.values()]) == len(store):
            # if each s,p pair contains at most 1 entity
            self.train_target = np.array(list(store.values()), dtype=np.int64)
            try:
                assert isinstance(self.train_target[0], np.ndarray)
            except IndexError or AssertionError:
                print(self.train_target)
                exit(1)
            assert isinstance(self.train_target[0][0], np.int64)
        else:
            self.train_target = list(store.values())
            assert isinstance(self.train_target[0], list)
        del store

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        # (1) Get ith unique (head,relation) pair
        x = self.train_data[idx]
        # (2) Get tail entities given (1)
        positives_idx = self.train_target[idx]
        num_positives = len(positives_idx)
        # (3) Subsample positive examples to generate a batch of same sized inputs
        if num_positives < self.neg_sample_ratio:
            # (3.1) Upsampling positives.
            positives_idx = torch.LongTensor(random.choices(positives_idx, k=self.neg_sample_ratio))
        else:
            # (3.1) Subsample positives.
            positives_idx = torch.LongTensor(random.sample(positives_idx, self.neg_sample_ratio))
        # (4) Generate random entities
        negative_idx = torch.randint(low=0, high=self.num_entities, size=(self.neg_sample_ratio,))
        # (5) Create selected indexes
        y_idx = torch.cat((positives_idx, negative_idx), 0)
        # (6) Create binary labels.
        y_vec = torch.cat((torch.ones(self.neg_sample_ratio), torch.zeros(self.neg_sample_ratio)), 0)
        return x, y_idx, y_vec


class BatchRelaxedKvsAllDataset(Dataset):
    """
    For entitiy or relation prediciton
    """

    def __init__(self, triples_idx, entity_idxs, relation_idxs, form, store=None, label_smoothing_rate=None):
        super().__init__()
        assert len(triples_idx) > 0
        self.train_data = None
        self.train_target = None
        self.range_of_relations = dict()

        # (1) Create a dictionary of training data pints
        # Either from tuple of entitiies or tuple of an entity and a relation
        if store is None:
            store = dict()
            self.target_dim = len(entity_idxs)
            for s_idx, p_idx, o_idx in triples_idx:
                store.setdefault((s_idx, p_idx), list()).append(o_idx)
                self.range_of_relations.setdefault(p_idx, set()).add(o_idx)

        for k, v in self.range_of_relations.items():
            self.range_of_relations[k] = list(v)

        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.train_data = torch.torch.LongTensor(list(store.keys()))

        if sum([len(i) for i in store.values()]) == len(store):
            # if each s,p pair contains at most 1 entity
            self.train_target = np.array(list(store.values()), dtype=np.int64)
            try:
                assert isinstance(self.train_target[0], np.ndarray)
            except IndexError or AssertionError:
                print(self.train_target)
                exit(1)
            assert isinstance(self.train_target[0][0], np.int64)
        else:
            self.train_target = list(store.values())
            assert isinstance(self.train_target[0], list)
        del store

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        # _, rel = self.train_data[idx]
        # y_vec[self.range_of_relations[rel.item()]] = .0001
        y_vec[self.train_target[idx]] = 1.0
        return self.train_data[idx], y_vec


class BatchRelaxed1vsAllDataset(Dataset):
    def __init__(self, triples_idx, entity_idxs, relation_idxs, form, store=None, label_smoothing_rate=None):
        super().__init__()
        assert len(triples_idx) > 0
        self.train_data = torch.torch.LongTensor(triples_idx)

        self.range_of_relations = dict()
        self.target_dim = len(entity_idxs)

        for s_idx, p_idx, o_idx in triples_idx:
            self.range_of_relations.setdefault(p_idx, set()).add(o_idx)

        for k, v in self.range_of_relations.items():
            self.range_of_relations[k] = list(v)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        idx_triple = self.train_data[idx]
        x, y = idx_triple[:2], idx_triple[2]
        # y_vec[self.range_of_relations[x[1].item()]] = .0001
        y_vec[y] = 1
        return x, y_vec


class TriplePredictionDataset(Dataset):
    """ Negative Sampling Class
    (1) \forall (h,r,t) \in G obtain,
    create negative triples{(h,r,x),(,r,t),(h,m,t)}

    (2) Targets
    Using hard targets (0,1) drives weights to infinity. An outlier produces enormous gradients. """

    def __init__(self, triples_idx, num_entities: int, num_relations: int, neg_sample_ratio: int = 1,
                 soft_confidence_rate: float = 0.001):
        """

        :param triples_idx:
        :param num_entities:
        :param num_relations:
        :param neg_sample_ratio:
        :param soft_confidence_rate:  Target/Label should be little but larger than 0 and lower than 1
        """
        start_time = time.time()
        print('Initializing negative sampling dataset batching...', end='\t')
        self.soft_confidence_rate = soft_confidence_rate
        self.neg_sample_ratio = neg_sample_ratio  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        self.triples_idx = triples_idx

        assert num_entities >= max(triples_idx[:, 0]) and num_entities >= max(triples_idx[:, 2])
        # assert num_relations > max(self.rel_idx)
        self.length = len(self.triples_idx)
        self.num_entities = num_entities
        self.num_relations = num_relations
        print(f'Done ! {time.time() - start_time:.3f} seconds\n')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.triples_idx[idx]

    def collate_fn(self, batch):
        batch = torch.LongTensor(batch)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        size_of_batch, _ = batch.shape
        assert size_of_batch > 0
        label = torch.ones((size_of_batch,), ) - self.soft_confidence_rate

        # corrupt head, tail or rel ?!

        # (1) Corrupted Entities:
        corr = torch.randint(0, self.num_entities, (size_of_batch * self.neg_sample_ratio, 2))
        # (2) Head Corrupt:
        h_head_corr = corr[:, 0]
        r_head_corr = r.repeat(self.neg_sample_ratio, )
        t_head_corr = t.repeat(self.neg_sample_ratio, )
        label_head_corr = torch.zeros(len(t_head_corr), ) + self.soft_confidence_rate
        # (3) Tail Corrupt:
        h_tail_corr = h.repeat(self.neg_sample_ratio, )
        r_tail_corr = r.repeat(self.neg_sample_ratio, )
        t_tail_corr = corr[:, 1]
        label_tail_corr = torch.zeros(len(t_tail_corr), ) + self.soft_confidence_rate
        # (4) Relations Corrupt:
        h_rel_corr = h.repeat(self.neg_sample_ratio, )
        r_rel_corr = torch.randint(0, self.num_relations, (size_of_batch * self.neg_sample_ratio, 1))[:, 0]
        t_rel_corr = t.repeat(self.neg_sample_ratio, )
        label_rel_corr = torch.zeros(len(t_rel_corr), ) + self.soft_confidence_rate
        # (5) Stack True and Corrupted Triples
        h = torch.cat((h, h_head_corr, h_tail_corr, h_rel_corr), 0)
        r = torch.cat((r, r_head_corr, r_tail_corr, r_rel_corr), 0)
        t = torch.cat((t, t_head_corr, t_tail_corr, t_rel_corr), 0)
        x = torch.stack((h, r, t), dim=1)
        label = torch.cat((label, label_head_corr, label_tail_corr, label_rel_corr), 0)
        return x, label



class PykeDataset(Dataset):
    def __init__(self, triples_idx, entity_idxs, relation_idxs, form, store=None, neg_sample_ratio: int = None,
                 label_smoothing_rate=None):
        super().__init__()
        self.entity_vocab = dict()
        self.collate_fn = None
        print('Creating mapping..')
        for i in triples_idx:
            s, p, o = i
            self.entity_vocab.setdefault(s, []).extend([o])
        del triples_idx
        # There are KGs therein some entities may not occur  in the training data split
        # To alleviate our of vocab, those entities are also index.
        self.int_to_data_point = dict()
        for ith, (k, v) in enumerate(self.entity_vocab.items()):
            self.int_to_data_point[ith] = k

        n = 0
        for k, v in self.entity_vocab.items():
            n += len(v)
        self.avg_triple_per_vocab = max(n // len(self.entity_vocab), 10)
        # Default
        # (1) Size of the dataset will be the number of unique vocabulary terms (|Entity \lor Rels|)
        # (2) For each term, at most K terms are stored as positives
        # (3) For each term, at most K terms stored as negatives
        # (4) Update: each term should be pulled by K terms and push by K terms

        # Update:
        # (1) (4) implies that a single data point must be (x, Px, Nx).
        # (2) Loss can be defined as should be large x-mean(Nx) x-mean(Px)

        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.positives = list(self.entity_vocab.values())
        self.num_of_vocabs = len(self.int_to_data_point)

    def __len__(self):
        return self.num_of_vocabs

    def __getitem__(self, idx):
        anchor = self.int_to_data_point[idx]
        positives = self.entity_vocab[anchor]
        # sample 10
        if len(positives) < self.avg_triple_per_vocab:
            # Upsampling
            select_positives_idx = torch.LongTensor(random.choices(positives, k=self.avg_triple_per_vocab))
        else:
            # Subsample
            select_positives_idx = torch.LongTensor(random.sample(positives, self.avg_triple_per_vocab))
        select_negative_idx = torch.LongTensor(random.sample(self.entity_vocab.keys(), len(select_positives_idx)))
        x = torch.cat((torch.LongTensor([anchor]), select_positives_idx, select_negative_idx), dim=0)
        return x, torch.LongTensor([0])

"""

class NotusedTripleClassificationDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
"""