from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch


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
                store.setdefault((self.kg.entity_idxs[s], self.kg.entity_idxs[o]), list()).append(
                    self.kg.relation_idxs[p])
        elif form == 'EntityPrediction':
            for s, p, o in self.train:
                store.setdefault((self.kg.entity_idxs[s], self.kg.relation_idxs[p]), list()).append(
                    self.kg.entity_idxs[o])
        else:
            raise NotImplementedError

        self.train_data = torch.torch.LongTensor(list(store.keys()))
        # To be able to obtain targets by using a list of indexes.
        self.train_target = np.array(list(store.values()), dtype=object)
        # self.target is still contains list of integers.
        assert isinstance(self.train_target, np.ndarray)
        assert isinstance(self.train_target[0], list)
        assert isinstance(self.train_target[0][0], int)


class RelationPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, idx_triples, target_dim):
        super().__init__()
        self.idx_triples = torch.torch.LongTensor(idx_triples)
        self.target_dim = target_dim

        self.head_entities=self.idx_triples[:,0]
        self.relations = self.idx_triples[:, 1]
        self.tail_entities=self.idx_triples[:,2]
        del self.idx_triples

        assert len(self.head_entities)==len(self.relations)==len(self.tail_entities)

    def __len__(self):
        return len(self.head_entities)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.relations[idx]] = 1
        return self.head_entities[idx], self.tail_entities[idx], y_vec


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
                store.setdefault((kg.entity_idxs[s], relation_idxs[p]), list()).append(entity_idxs[o])
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


class KG:
    def __init__(self, data_dir=None, add_reciprical=False):
        # 1. First pass through data
        self.train = self.load_data(data_dir + '/train.txt', add_reciprical=add_reciprical)
        self.valid = self.load_data(data_dir + '/valid.txt', add_reciprical=add_reciprical)
        self.test = self.load_data(data_dir + '/test.txt', add_reciprical=add_reciprical)

        self.__data = self.train + self.valid + self.test
        self.entities = self.get_entities(self.__data)
        self.relations = self.get_relations(self.__data)
        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}
        print('\n------------------- Description of Dataset----------------------------')
        print(f'Number of triples {len(self.__data)}')
        print(f'Number of entities {len(self.entities)}')
        print(f'Number of relations {len(self.relations)}')

        print(f'Number of triples on train set{len(self.train)}')
        print(f'Number of triples on valid set {len(self.valid)}')
        print(f'Number of triples on test set {len(self.test)}')
        print('----------------------------------------------------------------------\n')

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
