from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch


class FoldKvsAllDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, target_dim):
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


class KvsAllDataset(torch.utils.data.Dataset):
    def __init__(self, triples):
        # 1. First pass through data
        self.kg = triples
        self.data, self.target = None, None
        self.target_dim = None

    @property
    def entities(self):
        return self.kg.entities

    @property
    def entity_idx(self):
        return self.kg.entity_idxs

    @property
    def relations(self):
        return self.kg.relations

    @property
    def relation_idx(self):
        return self.kg.relation_idx

    def labelling(self, form='RelationPrediction'):
        """
        Given a kg (s,p,o), we construct input and output for learning problem.

        RelationPrediction => given entities predict a missing relation
                s,o => p
        EntityPrediction => given a subject and a relation predict missing entity, i.e. tail entity prediction.
                s,p => o

        :param form:
        :return:
        """
        store = dict()
        if form == 'RelationPrediction':
            self.target_dim = len(self.relations)
            for s, p, o in self.kg.data:
                store.setdefault((self.kg.entity_idxs[s], self.kg.entity_idxs[o]), list()).append(
                    self.kg.relation_idxs[p])
        elif form == 'EntityPrediction':
            self.target_dim = len(self.entities)
            for s, p, o in self.kg.data:
                store.setdefault((self.kg.entity_idxs[s], self.kg.relation_idxs[p]), list()).append(
                    self.kg.entity_idxs[o])
        else:
            raise NotImplementedError

        self.data = torch.torch.LongTensor(list(store.keys()))
        # To be able to obtain targets by using a list of indexes.
        self.target = np.array(list(store.values()),dtype=object)
        # self.target is still contains list of integers.
        assert isinstance(self.target,np.ndarray)
        assert isinstance(self.target[0], list)
        assert isinstance(self.target[0][0], int)

    def create_fold(self, idx: np.ndarray):
        """
        Create a fold for training
        :param idx:
        :return:
        """
        # self.target is a list of lists where each item contains index of output.
        # Python does not allow us to use a list/numpy array to obtain items by using list of indexes.
        # For instance, idx is a numpy array a one dimensional array
        assert idx.ndim == 1
        return FoldKvsAllDataset(data=self.data[idx], target=self.target[idx],target_dim=self.target_dim)

    def __len__(self):
        assert len(self.data) == len(self.target)
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.target[idx]] = 1
        return self.data[idx, 0], self.data[idx, 1], y_vec


class KG:
    def __init__(self, data_dir=None, add_reciprical=False):
        # 1. First pass through data
        self.data = self.load_data(data_dir, add_reciprical=add_reciprical)
        self.entities = self.get_entities(self.data)
        self.relations = self.get_relations(self.data)

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}
        print(f'Number of triples {len(self.data)}')
        print(f'Number of entities {len(self.entities)}')
        print(f'Number of relations {len(self.relations)}')

        # Store a mapping from inputs to outputs.
        # An input => a tuple of indexes of two inputs, e.g. indexes of two entities or an entitiy and a relation
        self.input_to_output = dict()
        self.length_of_an_output = None
        self.labels = None
        self.first_idx = None
        self.second_idx = None

    def labelling(self, form='RelationPrediction'):
        self.input_to_output = dict()
        self.length_of_an_output = None
        if form == 'RelationPrediction':
            self.length_of_an_output = len(self.relations)
            for s, p, o in self.data:
                self.input_to_output.setdefault((self.entity_idxs[s], self.entity_idxs[o]), list()).append(
                    self.relation_idxs[p])
        elif form == 'EntityPrediction':
            self.length_of_an_output = len(self.entities)

            for s, p, o in self.data:
                self.input_to_output.setdefault((self.entity_idxs[s], self.relation_idxs[p]), list()).append(
                    self.entity_idxs[o])
        else:
            raise NotImplementedError

        unique_input_pairs = torch.torch.LongTensor(list(self.input_to_output.keys()))
        self.labels = list(self.input_to_output.values())
        self.first_idx = unique_input_pairs[:, 0]
        self.second_idx = unique_input_pairs[:, 1]

    @staticmethod
    def load_data(data_dir, add_reciprical=True):
        try:
            with open(data_dir, "r") as f:
                data = f.read().strip().split("\n")
                data = [i.split() for i in data]
                if add_reciprical:
                    data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            raise ValueError
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def __len__(self):
        return len(self.labels)  # dataset is the number of unique tuples.

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.length_of_an_output)
        # 2. Label (1)
        try:
            y_vec[self.labels[idx]] = 1
        except:
            print(self.labels)
            print(idx)
            print(y_vec)
            raise ValueError
        return self.first_idx[idx], self.second_idx[idx], y_vec
