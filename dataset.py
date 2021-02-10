from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class StandardDataModule(pl.LightningDataModule):
    """
    train, valid and test sets are available.
    """

    def __init__(self, dataset, batch_size, form='RelationPrediction', num_workers=4):
        super().__init__()
        self.dataset = dataset
        self.form = form
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self) -> DataLoader:
        train_set = KvsAll(self.dataset.train_set, entity_idxs=self.dataset.entity_to_idx,
                           relation_idxs=self.dataset.relation_to_idx, form=self.form)

        return DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        idx_val_set = [[self.dataset.entity_to_idx[s], self.dataset.relation_to_idx[p], self.dataset.entity_to_idx[o]]
                       for
                       s, p, o in
                       self.dataset.val_set]
        return DataLoader(RelationPredictionDataset(idx_val_set, target_dim=self.dataset.num_relations),
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        idx_test_set = [[self.dataset.entity_to_idx[s], self.dataset.relation_to_idx[p], self.dataset.entity_to_idx[o]]
                        for
                        s, p, o in self.dataset.test_set]
        return DataLoader(RelationPredictionDataset(idx_test_set, target_dim=self.dataset.num_relations),
                          batch_size=self.batch_size, num_workers=self.num_workers)

    """
    def setup(self, stage: Optional[str] = None):
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    def prepare_data(self, *args, **kwargs):
        # Nothing to be prepared for now.
        pass
    """


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
                store.setdefault((kg.__entity_idxs[s], relation_idxs[p]), list()).append(entity_idxs[o])
        else:
            raise NotImplementedError

        self.train_data = torch.torch.LongTensor(list(store.keys()))
        self.train_target = np.array(list(store.values()), dtype=object)
        assert isinstance(self.train_target, np.ndarray)
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

    """
    def create_fold(self, idx: np.ndarray):
        # self.target is a list of lists where each item contains index of output.
        # Python does not allow us to use a list/numpy array to obtain items by using list of indexes.
        # For instance, idx is a numpy array a one dimensional array assert idx.ndim == 1
        return FoldKvsAllDataset(data=self.train_data[idx], target=self.train_target[idx], target_dim=self.target_dim)
    """

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
        print(f'\n------------------- Description of Dataset {data_dir}----------------------------')
        print(f'Number of triples {len(self.__data)}')
        print(f'Number of entities {len(self.__entities)}')
        print(f'Number of relations {len(self.__relations)}')

        print(f'Number of triples on train set{len(self.__train)}')
        print(f'Number of triples on valid set {len(self.__valid)}')
        print(f'Number of triples on test set {len(self.__test)}')
        print('----------------------------------------------------------------------\n')

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

    @staticmethod
    def load_data(data_path, add_reciprical=True):
        try:
            with open(data_path, "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split() for i in data]
                if add_reciprical:
                    data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        except FileNotFoundError:
            print(f'{data_path} is not found')
            return []
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