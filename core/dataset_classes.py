import time
from abc import ABCMeta

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
import random
from .typings import Dict, List
from .static_preprocess_funcs import mapping_from_first_two_cols_to_third, parallel_mapping_from_first_two_cols_to_third
# LongTensor or IntTensor

def input_data_type_checking(train_set_idx, valid_set_idx, test_set_idx, entity_to_idx: Dict, relation_to_idx: Dict):
    """ Type checking for efficient memory usage"""
    assert isinstance(train_set_idx, np.ndarray)
    assert str(np.dtype(train_set_idx.dtype)) in ['int8', 'int16', 'int32']
    if valid_set_idx is not None:
        if len(valid_set_idx) > 0:
            assert isinstance(valid_set_idx, np.ndarray)
            assert str(np.dtype(valid_set_idx.dtype)) in ['int8', 'int16', 'int32']
    if test_set_idx is not None:
        if len(test_set_idx) > 0:
            assert isinstance(test_set_idx, np.ndarray)
            assert str(np.dtype(test_set_idx.dtype)) in ['int8', 'int16', 'int32']
    assert isinstance(entity_to_idx, dict)
    assert isinstance(relation_to_idx, dict)


def create_tensor(x: np.ndarray):
    str_type = str(np.dtype(x.dtype))
    if str_type == 'int8':
        return torh.CharTensor(x)
    elif str_type == 'int16':
        return torch.ShortTensor(x)
    elif str_type == 'int32':
        return torch.IntTensor(x)
    else:
        raise TypeError(f'x has a type of {str_type}.')


class StandardDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """
    Creat a Dataset for KGE

    Parameters
    ----------
    train_set_idx
        Indexed triples for the training.
    entity_to_idx
        entity to index mapping.
    relation_to_idx
        relation to index mapping.
    batch_size
        int
    form
        ?
    num_workers
        int for https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    valid_set_idx
        Indexed triples for the validation.
    test_set_idx
        Indexed triples for the testing.
    neg_sample_ratio
        int negative triples per a training data
    label_smoothing_rate
    Smoothing binary labels None



    Returns
    -------
    ?
    """

    def __init__(self, train_set_idx: np.ndarray, entity_to_idx, relation_to_idx, batch_size, form,
                 num_workers=None, valid_set_idx=None,
                 test_set_idx=None, neg_sample_ratio=None,
                 label_smoothing_rate: int = 0.0):
        super().__init__()
        input_data_type_checking(train_set_idx=train_set_idx,
                                 valid_set_idx=valid_set_idx,
                                 test_set_idx=test_set_idx,
                                 entity_to_idx=entity_to_idx,
                                 relation_to_idx=relation_to_idx)

        self.train_set_idx = train_set_idx  # create_tensor(train_set_idx)
        self.valid_set_idx = valid_set_idx  # create_tensor(valid_set_idx) if valid_set_idx is not None else valid_set_idx
        self.test_set_idx = test_set_idx  # create_tensor(test_set_idx) if test_set_idx is not None else test_set_idx
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.target_dim = None
        self.form = form
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.neg_sample_ratio = neg_sample_ratio
        self.label_smoothing_rate = label_smoothing_rate
        self.construct_dataset()

    def construct_dataset(self):
        # @TODO: Proof the logic behind
        if self.form == 'RelationPrediction':
            self.target_dim = len(self.relation_to_idx)
        elif self.form == 'EntityPrediction':
            self.target_dim = len(self.entity_to_idx)
        elif self.form == 'NegativeSampling':  # we can name it as TriplePrediction
            # self.dataset_type_class = TriplePredictionDataset
            # self.target_dim = 1
            # self.neg_sample_ratio = neg_sample_ratio
            pass
        elif self.form == '1VsAll':
            # Multi-class
            self.dataset = OnevsAllDataset(self.train_set_idx, entity_idxs=self.entity_to_idx,
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
                                       neg_sample_ratio=self.neg_sample_ratio,
                                       label_smoothing_rate=self.label_smoothing_rate)
        else:
            raise ValueError(f'Invalid input : {self.form}')

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self) -> DataLoader:
        if self.form == 'NegativeSampling':
            train_set = TriplePredictionDataset(self.train_set_idx,
                                                num_entities=len(self.entity_to_idx),
                                                num_relations=len(self.relation_to_idx),
                                                neg_sample_ratio=self.neg_sample_ratio,
                                                label_smoothing_rate=self.label_smoothing_rate)
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
       Create a Dataset for cross validation

       Parameters
       ----------
       train_set_idx
           Indexed triples for the training.
       num_entities
           entity to index mapping.
       num_relations
           relation to index mapping.
       batch_size
           int
       form
           ?
       num_workers
           int for https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader



       Returns
       -------
       ?
       """

    def __init__(self, train_set_idx: np.ndarray, num_entities, num_relations, neg_sample_ratio, batch_size,
                 num_workers):
        super().__init__()
        assert isinstance(train_set_idx, np.ndarray)
        self.train_set_idx = train_set_idx
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_sample_ratio = neg_sample_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        train_set = TriplePredictionDataset(self.train_set_idx,
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
    """
       Dataset for the 1vsALL training strategy

       Parameters
       ----------
       train_set_idx
           Indexed triples for the training.
       entity_idxs
           mapping.
       relation_idxs
           mapping.
       form
           ?
       num_workers
           int for https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader



       Returns
       -------
       torch.utils.data.Dataset
       """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs, form):
        super().__init__()
        assert isinstance(train_set_idx, np.ndarray)
        assert len(train_set_idx) > 0
        self.train_data = train_set_idx
        self.target_dim = len(entity_idxs)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_data[idx, 2]] = 1

        return self.train_data[idx, :2], y_vec


class KvsAll(Dataset):
    """
    KvsAll a Dataset:

        D:= {(x,y)_i}_i ^N, where
            . x:(h,r) is a unique h \in E and a relation r \in R and
            . y \in [0,1]^{|E|} is a binary label. \forall y_i =1 s.t. (h r E_i) \in KG

       Parameters
       ----------
       train_set_idx
           Indexed triples for the training.
       entity_idxs
           mapping.
       relation_idxs
           mapping.
       form
           ?
       store
            ?
       label_smoothing_rate
           ?



       Returns
       -------
       torch.utils.data.Dataset
       """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs, form, store=None,
                 label_smoothing_rate=0.0):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, np.ndarray)
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate, dtype=torch.float16)
        self.collate_fn = None

        # (1) Create a dictionary of training data pints
        # Either from tuple of entities or tuple of an entity and a relation
        if store is None:
            store = dict()
            if form == 'RelationPrediction':
                self.target_dim = len(relation_idxs)
                for s_idx, p_idx, o_idx in train_set_idx:
                    store.setdefault((s_idx, o_idx), list()).append(p_idx)
            elif form == 'EntityPrediction':
                self.target_dim = len(entity_idxs)
                store = mapping_from_first_two_cols_to_third(train_set_idx)
            else:
                raise NotImplementedError
        else:
            raise ValueError()
        assert len(store) > 0
        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.train_data = torch.IntTensor(list(store.keys()))

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
        y_vec = torch.zeros(self.target_dim, dtype=torch.float16)
        y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class KvsSampleDataset(Dataset):
    """
    KvsSample a Dataset:

        D:= {(x,y)_i}_i ^N, where
            . x:(h,r) is a unique h \in E and a relation r \in R and
            . y \in [0,1]^{|E|} is a binary label. \forall y_i =1 s.t. (h r E_i) \in KG

           At each mini-batch construction, we subsample(y), hence n
            |new_y| << |E|
            new_y contains all 1's if sum(y)< neg_sample ratio
            new_y contains

       Parameters
       ----------
       train_set_idx
           Indexed triples for the training.
       entity_idxs
           mapping.
       relation_idxs
           mapping.
       form
           ?
       store
            ?
       label_smoothing_rate
           ?



       Returns
       -------
       torch.utils.data.Dataset
       """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs, form, store=None,
                 neg_sample_ratio: int = None,
                 label_smoothing_rate: float = 0.0):
        super().__init__()
        assert isinstance(train_set_idx, np.ndarray)
        self.train_data = None
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate, dtype=torch.float16)
        self.neg_sample_ratio = neg_sample_ratio
        self.collate_fn = None
        if self.neg_sample_ratio == 0:
            print(f'neg_sample_ratio is {neg_sample_ratio}. It will be set to 10.')
            self.neg_sample_ratio = 10
        print('Constructing training data...')
        self.num_entities = len(entity_idxs)
        store = mapping_from_first_two_cols_to_third(train_set_idx)
        assert len(store) > 0
        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        # Infer its type
        self.train_data = torch.IntTensor(list(store.keys()))
        self.train_target = list(store.values())
        assert isinstance(self.train_target[0], list)
        del store, train_set_idx, entity_idxs

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
            # (3.1)
            positives_idx = torch.IntTensor(positives_idx)
            # (4) Generate random entities
            negative_idx = torch.randint(low=0, high=self.num_entities,
                                         size=(self.neg_sample_ratio + self.neg_sample_ratio - num_positives,),
                                         dtype=torch.int32)
        else:
            # (3.1) Subsample positives without replacement
            # https://docs.python.org/3/library/random.html#random.sample
            positives_idx = torch.IntTensor(random.sample(positives_idx, self.neg_sample_ratio))
            # (4) Generate random entities
            negative_idx = torch.randint(low=0, high=self.num_entities, size=(self.neg_sample_ratio,),
                                         dtype=torch.int32)
        # (5) Create selected indexes
        y_idx = torch.cat((positives_idx, negative_idx), 0)
        # (6) Create binary labels.
        y_vec = torch.cat(
            (torch.ones(len(positives_idx), dtype=torch.float16), torch.zeros(len(negative_idx), dtype=torch.float16)),
            0)
        return x, y_idx, y_vec


class TriplePredictionDataset(Dataset):
    """
    Triple Dataset

        D:= {(x)_i}_i ^N, where
            . x:(h,r, t) \in KG is a unique h \in E and a relation r \in R and
            . collact_fn => Generates negative triples

        collect_fn:  \forall (h,r,t) \in G obtain, create negative triples{(h,r,x),(,r,t),(h,m,t)}

        y:labels are represented in torch.float16
       Parameters
       ----------
       train_set_idx
           Indexed triples for the training.
       entity_idxs
           mapping.
       relation_idxs
           mapping.
       form
           ?
       store
            ?
       label_smoothing_rate


       collate_fn: batch:List[torch.IntTensor]
       Returns
       -------
       torch.utils.data.Dataset
       """

    def __init__(self, train_set_idx: np.ndarray, num_entities: int, num_relations: int, neg_sample_ratio: int = 1,
                 label_smoothing_rate: float = 0.0):
        assert isinstance(train_set_idx, np.ndarray)
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate, dtype=torch.float16)
        self.neg_sample_ratio = neg_sample_ratio  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        self.triples_idx = torch.IntTensor(train_set_idx)

        assert num_entities >= max(self.triples_idx[:, 0]) and num_entities >= max(self.triples_idx[:, 2])
        self.length = len(self.triples_idx)
        self.num_entities = num_entities
        self.num_relations = num_relations

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.triples_idx[idx]

    def collate_fn(self, batch: List[torch.IntTensor]):
        batch = torch.stack(batch, dim=0)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        size_of_batch, _ = batch.shape
        assert size_of_batch > 0
        label = torch.ones((size_of_batch,), dtype=torch.int16) - self.label_smoothing_rate
        # corrupt head, tail or rel ?!
        # (1) Corrupted Entities:
        corr = torch.randint(0, high=self.num_entities, size=(size_of_batch * self.neg_sample_ratio, 2),
                             dtype=torch.int32)
        # (2) Head Corrupt:
        h_head_corr = corr[:, 0]
        r_head_corr = r.repeat(self.neg_sample_ratio, )
        t_head_corr = t.repeat(self.neg_sample_ratio, )
        label_head_corr = torch.zeros(len(t_head_corr), dtype=torch.int16) + self.label_smoothing_rate
        # (3) Tail Corrupt:
        h_tail_corr = h.repeat(self.neg_sample_ratio, )
        r_tail_corr = r.repeat(self.neg_sample_ratio, )
        t_tail_corr = corr[:, 1]
        label_tail_corr = torch.zeros(len(t_tail_corr), dtype=torch.int16) + self.label_smoothing_rate
        # (4) Relations Corrupt:
        h_rel_corr = h.repeat(self.neg_sample_ratio, )
        r_rel_corr = torch.randint(0, self.num_relations, (size_of_batch * self.neg_sample_ratio, 1),
                                   dtype=torch.int32)[:, 0]
        t_rel_corr = t.repeat(self.neg_sample_ratio, )
        label_rel_corr = torch.zeros(len(t_rel_corr), dtype=torch.int16) + self.label_smoothing_rate
        # (5) Stack True and Corrupted Triples
        h = torch.cat((h, h_head_corr, h_tail_corr, h_rel_corr), 0)
        r = torch.cat((r, r_head_corr, r_tail_corr, r_rel_corr), 0)
        t = torch.cat((t, t_head_corr, t_tail_corr, t_rel_corr), 0)
        x = torch.stack((h, r, t), dim=1)
        label = torch.cat((label, label_head_corr, label_tail_corr, label_rel_corr), 0)
        return x, label


class PykeDataset(Dataset):
    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs, form, store=None,
                 neg_sample_ratio: int = None,
                 label_smoothing_rate=None):
        super().__init__()
        assert isinstance(train_set_idx, np.ndarray)
        self.entity_vocab = dict()
        self.collate_fn = None
        print('Creating mapping..')
        for i in train_set_idx:
            s, p, o = i
            self.entity_vocab.setdefault(s, []).extend([o])
        del train_set_idx
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
