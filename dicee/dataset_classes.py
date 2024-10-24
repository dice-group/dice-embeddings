from torch.utils.data import DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from typing import List, Tuple, Union
from .static_preprocess_funcs import mapping_from_first_two_cols_to_third
from .static_funcs import timeit, load_term_mapping


@timeit
def reload_dataset(path: str, form_of_labelling, scoring_technique, neg_ratio, label_smoothing_rate):
    """ Reload the files from disk to construct the Pytorch dataset """
    return construct_dataset(train_set=np.load(path + '/train_set.npy'),
                             valid_set=None,
                             test_set=None,
                             entity_to_idx=load_term_mapping(file_path=path + '/entity_to_idx'),
                             relation_to_idx=load_term_mapping(file_path=path + '/relation_to_idx'),
                             form_of_labelling=form_of_labelling,
                             scoring_technique=scoring_technique, neg_ratio=neg_ratio,
                             label_smoothing_rate=label_smoothing_rate)


@timeit
def construct_dataset(*,
                      train_set: Union[np.ndarray, list],
                      valid_set=None,
                      test_set=None,
                      ordered_bpe_entities=None,
                      train_target_indices=None,
                      target_dim: int = None,
                      entity_to_idx: dict,
                      relation_to_idx: dict,
                      form_of_labelling: str,
                      scoring_technique: str,
                      neg_ratio: int,
                      label_smoothing_rate: float,
                      byte_pair_encoding=None,
                      block_size: int = None
                      ) -> torch.utils.data.Dataset:
    if ordered_bpe_entities and byte_pair_encoding and scoring_technique == 'NegSample':
        train_set = BPE_NegativeSamplingDataset(
            train_set=torch.tensor(train_set, dtype=torch.long),
            ordered_shaped_bpe_entities=torch.tensor(
                [shaped_bpe_ent for (str_ent, bpe_ent, shaped_bpe_ent) in ordered_bpe_entities]),
            neg_ratio=neg_ratio)
    elif ordered_bpe_entities and byte_pair_encoding and scoring_technique in ['KvsAll', "AllvsAll"]:
        train_set = MultiLabelDataset(train_set=torch.tensor(train_set, dtype=torch.long),
                                      train_indices_target=train_target_indices, target_dim=target_dim,
                                      torch_ordered_shaped_bpe_entities=torch.tensor(
                                          [shaped_bpe_ent for (str_ent, bpe_ent, shaped_bpe_ent) in
                                           ordered_bpe_entities]))
    elif byte_pair_encoding:
        # Multi-class classification based on transformer model's training.
        train_set = MultiClassClassificationDataset(train_set, block_size=block_size)
    elif scoring_technique == 'NegSample':
        # Binary-class.
        train_set = TriplePredictionDataset(train_set=train_set,
                                            num_entities=len(entity_to_idx),
                                            num_relations=len(relation_to_idx),
                                            neg_sample_ratio=neg_ratio,
                                            label_smoothing_rate=label_smoothing_rate)
    elif form_of_labelling == 'EntityPrediction':
        if scoring_technique == '1vsAll':
            # Multi-class.
            train_set = OnevsAllDataset(train_set, entity_idxs=entity_to_idx)
        elif scoring_technique == '1vsSample':
            # Dynamic Multi-class
            train_set = OnevsSample(train_set=train_set,
                                         num_entities=len(entity_to_idx),
                                         num_relations=len(relation_to_idx),
                                         neg_sample_ratio=neg_ratio,
                                         label_smoothing_rate=label_smoothing_rate)
        elif scoring_technique == 'KvsAll':
            # Multi-label.
            train_set = KvsAll(train_set,
                               entity_idxs=entity_to_idx,
                               relation_idxs=relation_to_idx,
                               form=form_of_labelling,
                               label_smoothing_rate=label_smoothing_rate)
        elif scoring_technique == 'AllvsAll':
            # Multi-label imbalanced.
            train_set = AllvsAll(train_set,
                                 entity_idxs=entity_to_idx,
                                 relation_idxs=relation_to_idx,
                                 label_smoothing_rate=label_smoothing_rate)
        elif scoring_technique == 'KvsSample':
            # Dynamic Multi-label.
            train_set = KvsSampleDataset(train_set,
                               entity_idxs=entity_to_idx,
                               relation_idxs=relation_to_idx,
                               form=form_of_labelling,
                                         neg_ratio=neg_ratio,
                               label_smoothing_rate=label_smoothing_rate)
        else:
            raise ValueError(f'Invalid scoring technique : {scoring_technique}')
    elif form_of_labelling == 'RelationPrediction':
        # Multi-label.
        train_set = KvsAll(train_set, entity_idxs=entity_to_idx, relation_idxs=relation_to_idx,
                           form=form_of_labelling, label_smoothing_rate=label_smoothing_rate)
    else:
        raise KeyError('Illegal input.')
    print(f"Number of datapoints: {len(train_set)}")
    return train_set


class BPE_NegativeSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, train_set: torch.LongTensor, ordered_shaped_bpe_entities: torch.LongTensor, neg_ratio: int):
        super().__init__()
        assert isinstance(train_set, torch.LongTensor)
        assert train_set.shape[1] == 3
        self.train_set = train_set
        self.ordered_bpe_entities = ordered_shaped_bpe_entities
        self.num_bpe_entities = len(self.ordered_bpe_entities)
        self.neg_ratio = neg_ratio
        self.num_datapoints = len(self.train_set)

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        return self.train_set[idx]

    def collate_fn(self, batch_shaped_bpe_triples: List[Tuple[torch.Tensor, torch.Tensor]]):
        batch_of_bpe_triples = torch.stack(batch_shaped_bpe_triples, dim=0)

        size_of_batch, _, token_length = batch_of_bpe_triples.shape

        bpe_h, bpe_r, bpe_t = batch_of_bpe_triples[:, 0, :], batch_of_bpe_triples[:, 1, :], batch_of_bpe_triples[:, 2,
                                                                                            :]

        label = torch.ones((size_of_batch,))
        num_of_corruption = size_of_batch * self.neg_ratio
        # Select bpe entities
        corr_bpe_entities = self.ordered_bpe_entities[
            torch.randint(0, high=self.num_bpe_entities, size=(num_of_corruption,))]

        if torch.rand(1) >= 0.5:
            bpe_h = torch.cat((bpe_h, corr_bpe_entities), 0)
            bpe_r = torch.cat((bpe_r, torch.repeat_interleave(input=bpe_r, repeats=self.neg_ratio, dim=0)), 0)
            bpe_t = torch.cat((bpe_t, torch.repeat_interleave(input=bpe_t, repeats=self.neg_ratio, dim=0)), 0)
        else:
            bpe_h = torch.cat((bpe_h, torch.repeat_interleave(input=bpe_h, repeats=self.neg_ratio, dim=0)), 0)
            bpe_r = torch.cat((bpe_r, torch.repeat_interleave(input=bpe_r, repeats=self.neg_ratio, dim=0)), 0)
            bpe_t = torch.cat((bpe_t, corr_bpe_entities), 0)

        bpe_triple = torch.stack((bpe_h, bpe_r, bpe_t), dim=1)
        label = torch.cat((label, torch.zeros(num_of_corruption)), 0)
        return bpe_triple, label


class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, train_set: torch.LongTensor, train_indices_target: torch.LongTensor, target_dim: int,
                 torch_ordered_shaped_bpe_entities: torch.LongTensor):
        super().__init__()
        assert len(train_set) == len(train_indices_target)
        assert target_dim > 0
        self.train_set = train_set
        self.train_indices_target = train_indices_target
        self.target_dim = target_dim
        self.num_datapoints = len(self.train_set)
        # why needed ?!
        self.torch_ordered_shaped_bpe_entities = torch_ordered_shaped_bpe_entities
        self.collate_fn = None

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        # (1) Initialize as all zeros.
        y_vec = torch.zeros(self.target_dim)
        # (2) Indices of labels.
        indices = self.train_indices_target[idx]
        # (3) Add 1s if holds.
        if len(indices) > 0:
            y_vec[indices] = 1.0
        return self.train_set[idx], y_vec


class MultiClassClassificationDataset(torch.utils.data.Dataset):
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

    def __init__(self, subword_units: np.ndarray, block_size: int = 8):
        super().__init__()
        assert isinstance(subword_units, np.ndarray)
        assert len(subword_units) > 0
        self.train_data = torch.LongTensor(subword_units)
        self.block_size = block_size
        self.num_of_data_points = len(self.train_data) - block_size
        self.collate_fn = None

    def __len__(self):
        return self.num_of_data_points

    def __getitem__(self, idx):
        x = self.train_data[idx:idx + self.block_size]
        y = self.train_data[idx + 1: idx + 1 + self.block_size]
        return x, y


class OnevsAllDataset(torch.utils.data.Dataset):
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

    def __init__(self, train_set_idx: np.ndarray, entity_idxs):
        super().__init__()
        assert isinstance(train_set_idx, np.memmap) or isinstance(train_set_idx, np.ndarray)
        assert len(train_set_idx) > 0
        self.train_data = train_set_idx
        self.target_dim = len(entity_idxs)
        self.collate_fn = None
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        triple= torch.from_numpy(self.train_data[idx].copy()).long()
        y_vec[triple[2]] = 1
        return triple[:2], y_vec
class KvsAll(torch.utils.data.Dataset):
    """ Creates a dataset for KvsAll training by inheriting from torch.utils.data.Dataset.
    Let D denote a dataset for KvsAll training and be defined as D:= {(x,y)_i}_i ^N, where
    x: (h,r) is an unique tuple of an entity h \in E and a relation r \in R that has been seed in the input graph.
    y: denotes a multi-label vector \in [0,1]^{|E|} is a binary label. \forall y_i =1 s.t. (h r E_i) \in KG

    .. note::
        TODO

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        n by 3 array representing n triples

    entity_idxs : dictonary
        string representation of an entity to its integer id

    relation_idxs : dictonary
        string representation of a relation to its integer id

    Returns
    -------
    self : torch.utils.data.Dataset

    See Also
    --------

    Notes
    -----

    Examples
    --------
    >>> a = KvsAll()
    >>> a
    ? array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs, form, store=None,
                 label_smoothing_rate: float = 0.0):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, np.memmap) or isinstance(train_set_idx, np.ndarray)
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
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
        self.train_data = torch.LongTensor(list(store.keys()))

        if sum([len(i) for i in store.values()]) == len(store):
            # if each s,p pair contains at most 1 entity
            self.train_target = np.array(list(store.values()))
            try:
                assert isinstance(self.train_target[0], np.ndarray)
            except IndexError or AssertionError:
                print(self.train_target)
                # TODO: Add info
                exit(1)
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
        y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class AllvsAll(torch.utils.data.Dataset):
    """ Creates a dataset for AllvsAll training by inheriting from torch.utils.data.Dataset.
    Let D denote a dataset for AllvsAll training and be defined as D:= {(x,y)_i}_i ^N, where
    x: (h,r) is a possible unique tuple of an entity h \in E and a relation r \in R. Hence N = |E| x |R|
    y: denotes a multi-label vector \in [0,1]^{|E|} is a binary label. \forall y_i =1 s.t. (h r E_i) \in KG

    .. note::
        AllvsAll extends KvsAll via none existing (h,r). Hence, it adds data points that are labelled without 1s,
         only with 0s.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        n by 3 array representing n triples

    entity_idxs : dictonary
        string representation of an entity to its integer id

    relation_idxs : dictonary
        string representation of a relation to its integer id

    Returns
    -------
    self : torch.utils.data.Dataset

    See Also
    --------

    Notes
    -----

    Examples
    --------
    >>> a = AllvsAll()
    >>> a
    ? array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs,
                 label_smoothing_rate=0.0):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, np.memmap)
        assert isinstance(train_set_idx, np.ndarray)
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None
        # (1) Create a dictionary of training data pints
        # Either from tuple of entities or tuple of an entity and a relation
        self.target_dim = len(entity_idxs)
        # (h,r) => [t]
        store = mapping_from_first_two_cols_to_third(train_set_idx)
        print("Number of unique pairs:", len(store))
        for i in range(len(entity_idxs)):
            for j in range(len(relation_idxs)):
                if store.get((i, j), None) is None:
                    store[(i, j)] = list()
        print("Number of unique augmented pairs:", len(store))
        assert len(store) > 0
        self.train_data = torch.LongTensor(list(store.keys()))

        if sum([len(i) for i in store.values()]) == len(store):
            self.train_target = np.array(list(store.values()))
            assert isinstance(self.train_target[0], np.ndarray)
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
        existing_indices = self.train_target[idx]
        if len(existing_indices) > 0:
            y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class OnevsSample(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class for knowledge graph embeddings, which includes
    both positive and negative sampling for a given dataset for multi-class classification problem..

    Args:
        train_set (np.ndarray): A numpy array containing triples of knowledge graph data.
            Each triple consists of (head_entity, relation, tail_entity).
        num_entities (int): The number of unique entities in the knowledge graph.
        num_relations (int): The number of unique relations in the knowledge graph.
        neg_sample_ratio (int, optional): The number of negative samples to be generated
            per positive sample. Must be a positive integer and less than num_entities.
        label_smoothing_rate (float, optional): A label smoothing rate to apply to the
            positive and negative labels. Defaults to 0.0.

    Attributes:
        train_data (torch.Tensor): The input data converted into a PyTorch tensor.
        num_entities (int): Number of entities in the dataset.
        num_relations (int): Number of relations in the dataset.
        neg_sample_ratio (int): Ratio of negative samples to be drawn for each positive sample.
        label_smoothing_rate (torch.Tensor): The smoothing factor applied to the labels.
        collate_fn (function, optional): A function that can be used to collate data samples into
            batches (set to None by default).
    """
    def __init__(self, train_set: np.ndarray, num_entities, num_relations, neg_sample_ratio: int = None,
                 label_smoothing_rate: float = 0.0):
        super().__init__()
        # Input validation
        assert isinstance(train_set, np.ndarray), "train_set must be a numpy array."
        assert isinstance(neg_sample_ratio, int), "neg_sample_ratio must be an integer."
        assert isinstance(num_entities, int) and num_entities > 0, "num_entities must be a positive integer."
        assert isinstance(num_relations, int) and num_relations > 0, "num_relations must be a positive integer."
        assert neg_sample_ratio < num_entities, (
            f"Negative sample ratio {neg_sample_ratio} cannot be larger than the number of entities ({num_entities})."
        )
        assert neg_sample_ratio > 0, f"Negative sample ratio {neg_sample_ratio} must be greater than 0."

        # Converting the input numpy array to a PyTorch tensor
        self.train_data = torch.from_numpy(train_set).long()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_sample_ratio = neg_sample_ratio
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.train_data)

    def __getitem__(self, idx):
        """
         Retrieves a single data sample from the dataset at the given index.

         Args:
             idx (int): The index of the sample to retrieve.

         Returns:
             tuple: A tuple consisting of:
                 - x (torch.Tensor): The head and relation part of the triple.
                 - y_idx (torch.Tensor): The concatenated indices of the true object (tail entity)
                   and the indices of the negative samples.
                 - y_vec (torch.Tensor): A vector containing the labels for the positive and
                   negative samples, with label smoothing applied.
         """
        # Retrieve the triple (head, relation, tail) from the training data
        triple = self.train_data[idx]
        # Separate the head and relation (x) and the tail entity (y)
        x = triple[:2]
        y = triple[-1].unsqueeze(0)  # Tail entity

        # Initialize weights for negative sampling
        weights = torch.ones(self.num_entities)
        # Set the weight of the true tail entity to zero
        weights[y] = 0.0

        # Sample negative examples from the entity set
        negative_idx = torch.multinomial(weights, num_samples=self.neg_sample_ratio, replacement=False)
        # Concatenate the true tail entity with the negative samples
        y_idx = torch.cat((y, negative_idx), 0).long()

        # Create a label vector with smoothing for the true and negative examples
        y_vec = torch.cat(
            (torch.ones(1) - self.label_smoothing_rate,  # Positive label with smoothing
             torch.zeros(self.neg_sample_ratio) + self.label_smoothing_rate),  # Negative labels with smoothing
            0)
        return x, y_idx, y_vec


class KvsSampleDataset(torch.utils.data.Dataset):
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
                 neg_ratio=None,label_smoothing_rate: float = 0.0):
        super().__init__()

        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, np.ndarray)
        assert neg_ratio is not None
        self.train_data = None
        self.train_target = None
        self.neg_ratio = neg_ratio
        self.num_entities = len(entity_idxs)
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None
        store = mapping_from_first_two_cols_to_third(train_set_idx)
        assert len(store) > 0
        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.train_data = torch.LongTensor(list(store.keys()))
        self.train_target = list(store.values())
        self.max_num_of_classes = max([ len(i) for i in self.train_target]) + self.neg_ratio

        del store
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # (1) Get i.th unique (head,relation) pair.
        x = self.train_data[idx]
        # (2) Get tail entities given (1).
        y = self.train_target[idx]
        num_positive_class =len(y)
        num_negative_class = self.max_num_of_classes - num_positive_class
        # Sample negatives
        weights = torch.ones(self.num_entities)
        weights[y] = 0.0
        negative_idx = torch.multinomial(weights, num_samples=num_negative_class, replacement=True)

        y_idx = torch.cat((torch.LongTensor(y), negative_idx), 0)
        y_vec = torch.cat((torch.ones(num_positive_class), torch.zeros(num_negative_class)),0)
        return x, y_idx, y_vec




class NegSampleDataset(torch.utils.data.Dataset):
    def __init__(self, train_set: np.ndarray, num_entities: int, num_relations: int, neg_sample_ratio: int = 1):
        assert isinstance(train_set, np.ndarray)
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # TLDL; replace Python objects with non-refcounted representations such as Pandas, Numpy or PyArrow objects
        self.neg_sample_ratio = torch.tensor(
            neg_sample_ratio)
        print("from numpy to torch")
        self.train_set = torch.from_numpy(train_set).unsqueeze(1)
        self.length = len(self.train_set)
        self.num_entities = torch.tensor(num_entities)
        self.num_relations = torch.tensor(num_relations)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # (1) Get a triple.
        triple = self.train_set[idx]
        # (2) Sample an entity.
        corr_entities = torch.randint(0, high=self.num_entities, size=(1,))
        # (3) Flip a coin
        if torch.rand(1) >= 0.5:
            # (3.1) Corrupt (1) via tai.
            negative_triple = torch.cat((triple[:, 0], triple[:, 1], corr_entities), dim=0).unsqueeze(0)
        else:
            # (3.1) Corrupt (1) via head.
            negative_triple = torch.cat((corr_entities, triple[:, 1], triple[:, 2]), dim=0).unsqueeze(0)
        # (4) Concat positive and negative triples.
        x = torch.cat((triple, negative_triple), dim=0)
        # (5) Concat labels of (4).
        y = torch.tensor([1.0, 0.0])
        return x, y


class TriplePredictionDataset(torch.utils.data.Dataset):
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

    def __init__(self, train_set: np.ndarray, num_entities: int, num_relations: int, neg_sample_ratio: int = 1,
                 label_smoothing_rate: float = 0.0):
        assert isinstance(train_set, np.ndarray)
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # TLDL; replace Python objects with non-refcounted representations such as Pandas, Numpy or PyArrow objects
        
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.neg_sample_ratio = torch.tensor(
            neg_sample_ratio)  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        #self.train_set = torch.from_numpy(train_set)
        self.train_set = train_set
        assert num_entities >= max(self.train_set[:, 0]) and num_entities >= max(self.train_set[:, 2])
        self.length = len(self.train_set)
        self.num_entities = torch.tensor(num_entities)
        self.num_relations = torch.tensor(num_relations)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.from_numpy(self.train_set[idx])

    def collate_fn(self, batch: List[torch.Tensor]):
        batch = torch.stack(batch, dim=0)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        size_of_batch, _ = batch.shape
        assert size_of_batch > 0
        label = torch.ones((size_of_batch,)) - self.label_smoothing_rate
        corr_entities = torch.randint(0, high=self.num_entities, size=(size_of_batch * self.neg_sample_ratio,))
        if torch.rand(1) >= 0.5:
            # corrupt head
            r_head_corr = r.repeat(self.neg_sample_ratio, )
            t_head_corr = t.repeat(self.neg_sample_ratio, )
            label_head_corr = torch.zeros(len(t_head_corr)) + self.label_smoothing_rate

            h = torch.cat((h, corr_entities), 0)
            r = torch.cat((r, r_head_corr), 0)
            t = torch.cat((t, t_head_corr), 0)
            x = torch.stack((h, r, t), dim=1)
            label = torch.cat((label, label_head_corr), 0)
        else:
            # corrupt tail
            h_tail_corr = h.repeat(self.neg_sample_ratio, )
            r_tail_corr = r.repeat(self.neg_sample_ratio, )
            label_tail_corr = torch.zeros(len(r_tail_corr)) + self.label_smoothing_rate

            h = torch.cat((h, h_tail_corr), 0)
            r = torch.cat((r, r_tail_corr), 0)
            t = torch.cat((t, corr_entities), 0)
            x = torch.stack((h, r, t), dim=1)
            label = torch.cat((label, label_tail_corr), 0)

        """        
        # corrupt head, tail or rel ?!
        # (1) Corrupted Entities:
        corr = torch.randint(0, high=self.num_entities, size=(size_of_batch * self.neg_sample_ratio, 2))
        # (2) Head Corrupt:
        h_head_corr = corr[:, 0]
        r_head_corr = r.repeat(self.neg_sample_ratio, )
        t_head_corr = t.repeat(self.neg_sample_ratio, )
        label_head_corr = torch.zeros(len(t_head_corr)) + self.label_smoothing_rate
        # (3) Tail Corrupt:
        h_tail_corr = h.repeat(self.neg_sample_ratio, )
        r_tail_corr = r.repeat(self.neg_sample_ratio, )
        t_tail_corr = corr[:, 1]
        label_tail_corr = torch.zeros(len(t_tail_corr)) + self.label_smoothing_rate
        # (4) Relations Corrupt:
        h_rel_corr = h.repeat(self.neg_sample_ratio, )
        r_rel_corr = torch.randint(0, self.num_relations, (size_of_batch * self.neg_sample_ratio, 1))[:, 0]
        t_rel_corr = t.repeat(self.neg_sample_ratio, )
        label_rel_corr = torch.zeros(len(t_rel_corr)) + self.label_smoothing_rate
        # (5) Stack True and Corrupted Triples
        h = torch.cat((h, h_head_corr, h_tail_corr, h_rel_corr), 0)
        r = torch.cat((r, r_head_corr, r_tail_corr, r_rel_corr), 0)
        t = torch.cat((t, t_head_corr, t_tail_corr, t_rel_corr), 0)
        x = torch.stack((h, r, t), dim=1)
        label = torch.cat((label, label_head_corr, label_tail_corr, label_rel_corr), 0)
        """
        return x, label


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
