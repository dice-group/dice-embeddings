from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class StandardDataModule(pl.LightningDataModule):
    """
    train, valid and test sets are available.
    """

    def __init__(self, dataset, batch_size, form, num_workers=4):
        super().__init__()
        self.dataset = dataset
        self.form = form
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.form == 'RelationPrediction':
            self.dataset_type_class = RelationPredictionDataset
            self.target_dim = self.dataset.num_relations

        elif self.form == 'EntityPrediction':
            self.dataset_type_class = EntityPredictionDataset
            self.target_dim = self.dataset.num_entities
        else:
            raise ValueError

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self) -> DataLoader:
        train_set = KvsAll(self.dataset.train_set, entity_idxs=self.dataset.entity_to_idx,
                           relation_idxs=self.dataset.relation_to_idx, form=self.form)

        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_type_class(self.dataset.idx_val_set, target_dim=self.target_dim),
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_type_class(self.dataset.idx_test_set, target_dim=self.target_dim),
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def setup(self, *args, **kwargs):
        pass

    def transfer_batch_to_device(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        # Nothing to be prepared for now.
        pass

class KvsAll(torch.utils.data.Dataset):
    def __init__(self, triples, entity_idxs, relation_idxs, form):
        super().__init__()

        self.train_data = None
        self.train_target = None

        store = dict()
        if form == 'RelationPrediction':
            self.target_dim = len(relation_idxs)
            for s, p, o in triples:
                store.setdefault((entity_idxs[s], entity_idxs[o]), list()).append(relation_idxs[p])
        elif form == 'EntityPrediction':
            self.target_dim = len(entity_idxs)
            for s, p, o in triples:
                store.setdefault((entity_idxs[s], relation_idxs[p]), list()).append(entity_idxs[o])
        else:
            raise NotImplementedError
        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.train_data = torch.torch.LongTensor(list(store.keys()))

        if sum([len(i) for i in store.values()]) == len(store):
            # if each s,p pair contains at most 1 entity
            self.train_target = np.array(list(store.values()), dtype=np.int64)
            assert isinstance(self.train_target[0], np.ndarray)
            assert isinstance(self.train_target[0][0], np.int64)
        else:
            # list of lists where each list has different size
            self.train_target = np.array(list(store.values()), dtype=object)
            assert isinstance(self.train_target[0], list)
            assert isinstance(self.train_target[0][0], int)
        del store

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1
        return self.train_data[idx, 0], self.train_data[idx, 1], y_vec


class RelationPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, idx_triples, target_dim):
        super().__init__()
        assert len(idx_triples) > 0
        self.idx_triples = torch.torch.LongTensor(idx_triples)
        self.target_dim = target_dim

        self.head_entities = self.idx_triples[:, 0]
        self.relations = self.idx_triples[:, 1]
        self.tail_entities = self.idx_triples[:, 2]
        del self.idx_triples

        assert len(self.head_entities) == len(self.relations) == len(self.tail_entities)

    def __len__(self):
        return len(self.head_entities)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.relations[idx]] = 1
        return self.head_entities[idx], self.tail_entities[idx], y_vec


class EntityPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, idx_triples, target_dim):
        super().__init__()
        assert len(idx_triples) > 0
        self.idx_triples = torch.torch.LongTensor(idx_triples)
        self.target_dim = target_dim

        self.head_entities = self.idx_triples[:, 0]
        self.relations = self.idx_triples[:, 1]
        self.tail_entities = self.idx_triples[:, 2]
        del self.idx_triples

        assert len(self.head_entities) == len(self.relations) == len(self.tail_entities)

    def __len__(self):
        return len(self.head_entities)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.tail_entities[idx]] = 1
        return self.head_entities[idx], self.relations[idx], y_vec


class KG:
    def __init__(self, data_dir=None, add_reciprical=False):
        # 1. First pass through data
        self.__train = self.load_data(data_dir + '/train.txt', add_reciprical=add_reciprical)
        self.__valid = self.load_data(data_dir + '/valid.txt', add_reciprical=add_reciprical)
        self.__test = self.load_data(data_dir + '/test.txt', add_reciprical=add_reciprical)

        self.__data = self.__train + self.__valid + self.__test
        self.__entities = self.get_entities(self.__data)
        self.__relations = self.get_relations(self.__data)
        self.__entity_idxs = {self.__entities[i]: i for i in range(len(self.__entities))}
        self.__relation_idxs = {self.__relations[i]: i for i in range(len(self.__relations))}
        s = '------------------- Description of Dataset {data_dir}----------------------------'
        print(f'\n{s}')
        print(f'Number of triples {len(self.__data)}')
        print(f'Number of entities {len(self.__entities)}')
        print(f'Number of relations {len(self.__relations)}')

        print(f'Number of triples on train set{len(self.__train)}')
        print(f'Number of triples on valid set {len(self.__valid)}')
        print(f'Number of triples on test set {len(self.__test)}')
        s = len(s) * '-'
        print(f'{s}\n')

        if self.is_valid_test_available():
            # We can store them in numpy.
            self.idx_val_set = [[self.entity_to_idx[s], self.relation_to_idx[p], self.entity_to_idx[o]]
                                for s, p, o in self.val_set]
            self.idx_test_set = [[self.entity_to_idx[s], self.relation_to_idx[p], self.entity_to_idx[o]]
                                 for s, p, o in self.test_set]

    @property
    def entities(self):
        return self.__entities

    @property
    def relations(self):
        return self.__relations

    @property
    def num_entities(self):
        return len(self.__entities)

    @property
    def entity_to_idx(self):
        return self.__entity_idxs

    @property
    def relation_to_idx(self):
        return self.__relation_idxs

    @property
    def num_relations(self):
        return len(self.__relations)

    def load_data(self, data_path, add_reciprical=True):
        # line can be 1 or 2
        # a) <...> <...> <...> .
        # b) <...> <...> "..." .
        # c) ... ... ...
        # (a) and (b) correspond to the N-Triples format
        # (c) corresponds to the format of current link prediction benchmark datasets.

        try:
            data = []
            with open(data_path, "r") as f:

                for line in f.readlines():

                    # 1. Ignore lines with *** " *** or does only contain 2 or less characters.
                    if '"' in line or len(line) < 3:
                        continue
                    # 2. Tokenize(<...> <...> <...> .) => ['<...>', '<...>','<...>','.']
                    # Tokenize(... ... ...) => ['...', '...', '...',]
                    decomposed_list_of_strings = line.split()

                    # 3. Sanity checking.
                    assert len(decomposed_list_of_strings) == 3 or len(decomposed_list_of_strings) == 4
                    # 4. Storing
                    if len(decomposed_list_of_strings) == 4:
                        assert decomposed_list_of_strings[-1] == '.'
                        data.append(decomposed_list_of_strings[:-1])
                    if len(decomposed_list_of_strings) == 3:
                        data.append(decomposed_list_of_strings)
        except FileNotFoundError:
            print(f'{data_path} is not found')
            return []
        if add_reciprical:
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def is_valid_test_available(self):
        if len(self.__valid) > 0 and len(self.__test) > 0:
            return True
        return False

    @property
    def train_set(self):
        return self.__train

    @property
    def val_set(self):
        return self.__valid

    @property
    def test_set(self):
        return self.__test


"""


class FoldKvsAllDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, target_dim):
        super().__init__()
        self.data, self.target, self.target_dim = data, target, target_dim

    def __len__(self):
        assert len(self.data) == len(self.target)
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        # 2. Set 1's to crrecponding indexes.
        y_vec[self.target[idx]] = 1
        return self.data[idx, 0], self.data[idx, 1], y_vec

"""
"""
class TrainKvsAllDataset(torch.utils.data.Dataset):
    def __init__(self, triples, target_dim, labelling_from='RelationPrediction'):
        super().__init__()
        self.triples = triples
        self.labelling_from = labelling_from
        self.target_dim = target_dim
        self.__labelling()

    def __labelling(self):
        store = dict()
        if self.labelling_from == 'RelationPrediction':
            for s, p, o in self.train:
                store.setdefault((self.kg.__entity_idxs[s], self.kg.__entity_idxs[o]), list()).append(
                    self.kg.__relation_idxs[p])
        elif form == 'EntityPrediction':
            for s, p, o in self.train:
                store.setdefault((self.kg.__entity_idxs[s], self.kg.__relation_idxs[p]), list()).append(
                    self.kg.__entity_idxs[o])
        else:
            raise NotImplementedError

        self.train_data = torch.torch.LongTensor(list(store.keys()))
        # To be able to obtain targets by using a list of indexes.
        self.train_target = np.array(list(store.values()), dtype=object)
        # self.target is still contains list of integers.
        assert isinstance(self.train_target, np.ndarray)
        assert isinstance(self.train_target[0], list)
        assert isinstance(self.train_target[0][0], int)







"""
