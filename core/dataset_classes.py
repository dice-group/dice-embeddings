from torch.utils.data import Dataset
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List

class StandardDataModule(pl.LightningDataModule):
    """
    train, valid and test sets are available.
    """

    def __init__(self, train_set_idx, entity_to_idx, relation_to_idx, batch_size, form,
                 num_workers=32, valid_set_idx=None, test_set_idx=None, neg_sample_ratio=None,
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
            self.dataset_type_class = OneVsAllEntityPredictionDataset
        else:
            raise ValueError(f'Invalid input : {self.form}')

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self) -> DataLoader:
        if self.form == 'NegativeSampling':
            train_set = TriplePredictionDataset(self.train_set_idx,
                                                num_entities=len(self.entity_to_idx),
                                                num_relations=len(self.relation_to_idx),
                                                neg_sample_ratio=self.neg_sample_ratio)
            return DataLoader(train_set, batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              collate_fn=train_set.collate_fn, pin_memory=True
                              )
        elif self.form == 'EntityPrediction' or self.form == 'RelationPrediction':
            train_set = KvsAll(self.train_set_idx, entity_idxs=self.entity_to_idx,
                               relation_idxs=self.relation_to_idx, form=self.form, label_smoothing_rate=self.label_smoothing_rate)

            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                              num_workers=self.num_workers)

        elif self.form == '1VsAll':
            train_set = OnevsAll(self.train_set_idx, entity_idxs=self.entity_to_idx,
                               relation_idxs=self.relation_to_idx, form=self.form, label_smoothing_rate=self.label_smoothing_rate)

            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                              num_workers=self.num_workers)
        else:
            raise KeyError(f'{self.form} illegal input.')

    def val_dataloader(self) -> DataLoader:
        if self.form == 'NegativeSampling':
            valid_set = TriplePredictionDataset(self.valid_set_idx,
                                                num_entities=len(self.entity_to_idx),
                                                num_relations=len(self.relation_to_idx),
                                                neg_sample_ratio=self.neg_sample_ratio)
            return DataLoader(valid_set, batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              collate_fn=valid_set.collate_fn, pin_memory=True
                              )
        elif self.form == 'EntityPrediction' or self.form == 'RelationPrediction':
            valid_set = KvsAll(self.valid_set_idx, entity_idxs=self.entity_to_idx,
                               relation_idxs=self.relation_to_idx, form=self.form,
                               label_smoothing_rate=self.label_smoothing_rate)
            return DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                              num_workers=self.num_workers)

        elif self.form == '1VsAll':
            valid_set = OneVsAllEntityPredictionDataset(self.valid_set_idx)
            return DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                              num_workers=self.num_workers)
        else:
            raise KeyError(f'{self.form} illegal input.')

    def test_dataloader(self) -> DataLoader:
        if self.form == 'NegativeSampling':
            test_set = TriplePredictionDataset(self.test_set_idx,
                                               num_entities=len(self.entity_to_idx),
                                               num_relations=len(self.relation_to_idx), )
            return DataLoader(test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True)

        elif self.form == 'EntityPrediction':
            test_set = KvsAll(self.test_set_idx, entity_idxs=self.entity_to_idx,
                              relation_idxs=self.relation_to_idx, form=self.form)
            return DataLoader(test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True)
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
                          collate_fn=train_set.collate_fn, pin_memory=True)

    def setup(self, *args, **kwargs):
        pass

    def transfer_batch_to_device(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        # Nothing to be prepared for now.
        pass


class KvsAll(Dataset):
    """
    For entitiy or relation prediciton
    """
    def __init__(self, triples_idx, entity_idxs, relation_idxs, form, store=None, label_smoothing_rate=None):
        super().__init__()
        assert len(triples_idx)>0
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = label_smoothing_rate

        # (1) Create a dictionary of training data pints
        # Either from tuple of entitiies or tuple of an entity and a relation
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
            # list of lists where each list has different size
            self.train_target = np.array(list(store.values()), dtype=object)
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


class TriplePredictionDataset(Dataset):

    def __init__(self, triples_idx, num_entities, num_relations, neg_sample_ratio=0):
        self.neg_sample_ratio = neg_sample_ratio  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        triples_idx = torch.LongTensor(triples_idx)
        self.head_idx = triples_idx[:, 0]
        self.rel_idx = triples_idx[:, 1]
        self.tail_idx = triples_idx[:, 2]
        assert self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape
        assert num_entities > max(self.head_idx) and num_entities > max(self.tail_idx)
        assert num_relations > max(self.rel_idx)

        self.length = len(triples_idx)

        self.num_entities = num_entities
        self.num_relations = num_relations

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h = self.head_idx[idx]
        r = self.rel_idx[idx]
        t = self.tail_idx[idx]
        return h, r, t, torch.ones(1, dtype=torch.long)

    def collate_fn(self, batch):
        batch = torch.LongTensor(batch)
        h, r, t, label = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
        size_of_batch, _ = batch.shape
        assert size_of_batch > 0
        label = torch.ones((size_of_batch,))
        # Generate Negative Triples
        corr = torch.randint(0, self.num_entities, (size_of_batch * self.neg_sample_ratio, 2))
        # 2.1 Head Corrupt:
        h_head_corr = corr[:, 0]
        r_head_corr = r.repeat(self.neg_sample_ratio, )
        t_head_corr = t.repeat(self.neg_sample_ratio, )
        label_head_corr = torch.zeros(len(t_head_corr), )

        # 2.2. Tail Corrupt
        h_tail_corr = h.repeat(self.neg_sample_ratio, )
        r_tail_corr = r.repeat(self.neg_sample_ratio, )
        t_tail_corr = corr[:, 1]
        label_tail_corr = torch.zeros(len(t_tail_corr), )

        # 3. Stack True and Corrupted Triples
        h = torch.cat((h, h_head_corr, h_tail_corr), 0)
        r = torch.cat((r, r_head_corr, r_tail_corr), 0)
        t = torch.cat((t, t_head_corr, t_tail_corr), 0)

        x = torch.stack((h, r, t), dim=1)

        label = torch.cat((label, label_head_corr, label_tail_corr), 0)

        return x, label


class OneVsAllEntityPredictionDataset(Dataset):
    def __init__(self, idx_triples):
        super().__init__()
        assert len(idx_triples) > 0

        self.train_data = torch.torch.LongTensor(idx_triples)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx, :2], self.train_data[idx, 2]




class TripleClassificationDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]