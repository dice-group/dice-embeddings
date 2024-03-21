from torch.utils.data import DataLoader
import numpy as np
import torch
import pytorch_lightning as pl
from typing import List, Tuple, Union
from .static_preprocess_funcs import mapping_from_first_two_cols_to_third
from .static_funcs import timeit, load_pickle


@timeit
def reload_dataset(path: str, form_of_labelling: str, scoring_technique: str, 
                   neg_ratio: float, label_smoothing_rate: float) -> torch.utils.data.Dataset:
    """
    Reloads the dataset from disk and constructs a PyTorch dataset for training.

    Parameters
    ----------
    path : str
        The path to the directory where the dataset is stored.
    form_of_labelling : str
        The form of labelling used in the dataset. Determines how data points are represented.
    scoring_technique : str
        The scoring technique used for evaluating the embeddings.
    neg_ratio : float
        The ratio of negative samples to positive samples in the dataset.
    label_smoothing_rate : float
        The rate of label smoothing applied to the dataset.

    Returns
    -------
    torch.utils.data.Dataset
        A PyTorch dataset object ready for training.
    """
    return construct_dataset(train_set=np.load(path + '/train_set.npy'),
                             valid_set=None,
                             test_set=None,
                             entity_to_idx=load_pickle(file_path=path + '/entity_to_idx.p'),
                             relation_to_idx=load_pickle(file_path=path + '/relation_to_idx.p'),
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
    """
    Constructs a dataset based on the specified parameters and returns a PyTorch Dataset object.

    Parameters
    ----------
    train_set : Union[np.ndarray, list]
        The training set consisting of triples or tokens.
    valid_set : Optional
        The validation set. Not currently used in dataset construction.
    test_set : Optional
        The test set. Not currently used in dataset construction.
    ordered_bpe_entities : Optional
        Ordered byte pair encoding entities for the dataset.
    train_target_indices : Optional
        Indices of target entities or relations for training.
    target_dim : int, optional
        The dimension of target entities or relations.
    entity_to_idx : dict
        A dictionary mapping entity strings to indices.
    relation_to_idx : dict
        A dictionary mapping relation strings to indices.
    form_of_labelling : str
        Specifies the form of labelling, such as 'EntityPrediction' or 'RelationPrediction'.
    scoring_technique : str
        The scoring technique used for generating negative samples or evaluating the model.
    neg_ratio : int
        The ratio of negative samples to positive samples.
    label_smoothing_rate : float
        The rate of label smoothing applied to labels.
    byte_pair_encoding : Optional
        Indicates if byte pair encoding is used.
    block_size : int, optional
        The block size for transformer-based models.

    Returns
    -------
    torch.utils.data.Dataset
        A PyTorch dataset object ready for model training.
    """
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
        elif scoring_technique == 'KvsSample':
            # Multi-label.
            train_set = KvsSampleDataset(train_set=train_set,
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
        else:
            raise ValueError(f'Invalid scoring technique : {scoring_technique}')
    elif form_of_labelling == 'RelationPrediction':
        # Multi-label.
        train_set = KvsAll(train_set, entity_idxs=entity_to_idx, relation_idxs=relation_to_idx,
                           form=form_of_labelling, label_smoothing_rate=label_smoothing_rate)
    else:
        raise KeyError('Illegal input.')
    return train_set


class BPE_NegativeSamplingDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for handling negative sampling with Byte Pair Encoding (BPE) entities.

    This dataset extends the PyTorch Dataset class to provide functionality for negative sampling
    in the context of knowledge graph embeddings. It uses byte pair encoding for entities
    to handle large vocabularies efficiently.

    Parameters
    ----------
    train_set : torch.LongTensor
        A tensor containing the training set triples with byte pair encoded entities and relations.
        The shape of the tensor is [N, 3], where N is the number of triples.
    ordered_shaped_bpe_entities : torch.LongTensor
        A tensor containing the ordered and shaped byte pair encoded entities.
    neg_ratio : int
        The ratio of negative samples to generate per positive sample.

    Attributes
    ----------
    num_bpe_entities : int
        The number of byte pair encoded entities.
    num_datapoints : int
        The number of data points (triples) in the training set.
    """
    def __init__(self, train_set: torch.LongTensor, ordered_shaped_bpe_entities: torch.LongTensor, neg_ratio: int):
        super().__init__()
        assert isinstance(train_set, torch.LongTensor)
        assert train_set.shape[1] == 3
        self.train_set = train_set
        self.ordered_bpe_entities = ordered_shaped_bpe_entities
        self.num_bpe_entities = len(self.ordered_bpe_entities)
        self.neg_ratio = neg_ratio
        self.num_datapoints = len(self.train_set)

    def __len__(self) -> int:
        """
        Returns the total number of data points in the dataset.

        Returns
        -------
        int
            The number of data points.
        """
        return self.num_datapoints

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the BPE-encoded triple and its corresponding label at the specified index.

        Parameters
        ----------
        idx : int
            Index of the triple to retrieve.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - The BPE-encoded triple as a torch.Tensor of shape (3,).
            - The label for the triple, where positive examples have a label of 1 and negative examples have a label
              of 0, as a torch.Tensor.
        """
        return self.train_set[idx]

    def collate_fn(self, batch_shaped_bpe_triples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function for the BPE_NegativeSamplingDataset. It processes a batch of byte pair encoded triples, 
        performs negative sampling, and returns the batch along with corresponding labels.

        This function is designed to be used with a PyTorch DataLoader. It takes a list of byte pair encoded triples
        as input and generates negative samples according to the specified negative sampling ratio. The function
        ensures that the negative samples are combined with the original triples to form a single batch, which is
        suitable for training a knowledge graph embedding model.

        Parameters
        ----------
        batch_shaped_bpe_triples : List[Tuple[torch.Tensor, torch.Tensor]]
            A list of tuples, where each tuple contains byte pair encoded representations of head entities, relations,
            and tail entities for a batch of triples.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two elements:
            - The first element is a torch.Tensor of shape [N * (1 + neg_ratio), 3] that contains both the original
            byte pair encoded triples and the generated negative samples. N is the original number of triples in the
            batch, and neg_ratio is the negative sampling ratio.
            - The second element is a torch.Tensor of shape [N * (1 + neg_ratio)] that contains the labels for each
            triple in the batch. Positive samples are labeled as 1, and negative samples are labeled as 0.
        """
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
    """
    A dataset class for multi-label knowledge graph embedding tasks. This dataset is designed for models where
    the output involves predicting multiple labels (entities or relations) for a given input (e.g., predicting all
    possible tail entities given a head entity and a relation).

    Parameters
    ----------
    train_set : torch.LongTensor
        A tensor containing the training set triples with byte pair encoding, shaped as [num_triples, 3], 
        where each triple is [head, relation, tail].
    
    train_indices_target : torch.LongTensor
        A tensor where each row corresponds to the indices of the target labels for each training example. 
        The length of this tensor must match the number of triples in `train_set`.
    
    target_dim : int
        The dimensionality of the target space, typically the total number of possible labels (entities or relations).
    
    torch_ordered_shaped_bpe_entities : torch.LongTensor
        A tensor containing ordered byte pair encoded entities used for creating embeddings. 
        This tensor is not directly used in generating targets but may be utilized for additional processing 
        or embedding lookup.

    Attributes
    ----------
    num_datapoints : int
        The number of data points (triples) in the dataset.
    
    collate_fn : None or callable
        Optional custom collate function to be used with a PyTorch DataLoader. 
        It's set to None by default and can be specified after initializing the dataset if needed.
        
    Note
    ----
    This dataset is particularly suited for KvsAll (K entities vs. All entities) and AllvsAll training strategies 
    in knowledge graph embedding, where a model predicts a set of possible tail entities given a head entity 
    and a relation (or vice versa), and where each training example can have multiple correct labels.
    """
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

    def __len__(self) -> int:
        """
        Returns the total number of data points in the dataset.

        Returns
        -------
        int
            The number of data points.
        """
        return self.num_datapoints

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the knowledge graph triple and its corresponding multi-label target vector at the specified index.

        Parameters
        ----------
        idx : int
            Index of the triple to retrieve.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - The triple as a torch.Tensor of shape (3,).
            - The multi-label target vector as a torch.Tensor of shape (`target_dim`,), where each element
              indicates the presence (1) or absence (0) of a label for the given triple.
        """
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
    A dataset class for multi-class classification tasks, specifically designed for the 1vsALL training strategy
    in knowledge graph embedding models. This dataset supports tasks where the model predicts a single correct
    label from all possible labels for a given input.

    Parameters
    ----------
    subword_units : np.ndarray
        An array of subword unit indices representing the training data. Each row in the array corresponds to a
        sequence of subword units (e.g., Byte Pair Encoding tokens) that have been converted to their respective
        numeric indices.
    
    block_size : int, optional
        The size of each sequence of subword units to be used as input to the model. This defines the length of
        the sequences that the model will receive as input, by default 8.

    Attributes
    ----------
    num_of_data_points : int
        The number of sequences or data points available in the dataset, calculated based on the length of the
        `subword_units` array and the `block_size`.
    
    collate_fn : None or callable
        An optional custom collate function to be used with a PyTorch DataLoader. It's set to None by default
        and can be specified after initializing the dataset if needed.
        
    Note
    ----
    This dataset is tailored for training knowledge graph embedding models on tasks where the output is a single
    label out of many possible labels (1vsALL strategy). It is especially suited for models trained with subword
    tokenization methods like Byte Pair Encoding (BPE), where inputs are sequences of subword unit indices.
    """

    def __init__(self, subword_units: np.ndarray, block_size: int = 8):
        super().__init__()
        assert isinstance(subword_units, np.ndarray)
        assert len(subword_units) > 0
        self.train_data = torch.LongTensor(subword_units)
        self.block_size = block_size
        self.num_of_data_points = len(self.train_data) - block_size
        self.collate_fn = None

    def __len__(self) -> int:
        """
        Returns the total number of sequences or data points available in the dataset.

        Returns
        -------
        int
            The number of sequences or data points.
        """
        return self.num_of_data_points

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an input sequence and its subsequent target sequence for next token prediction.

        Parameters
        ----------
        idx : int
            The starting index for the sequence to be retrieved from the dataset.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two elements:
            - `x`: The input sequence as a torch.Tensor of shape (`block_size`,).
            - `y`: The target sequence as a torch.Tensor of shape (`block_size`,), offset by one position
              from the input sequence.
        """
        x = self.train_data[idx:idx + self.block_size]
        y = self.train_data[idx + 1: idx + 1 + self.block_size]
        return x, y


class OnevsAllDataset(torch.utils.data.Dataset):
    """
    A dataset for the One-vs-All (1vsAll) training strategy designed for knowledge graph embedding tasks.
    This dataset structure is particularly suited for models predicting a single correct label (entity) out of
    all possible entities for a given pair of head entity and relation.

    Parameters
    ----------
    train_set_idx : np.ndarray
        An array containing indexed triples from the knowledge graph. Each row represents a triple consisting of
        indices for the head entity, relation, and tail entity, respectively.
    
    entity_idxs : dict
        A dictionary mapping entity names to their corresponding unique integer indices. This is used to determine
        the dimensionality of the target vector in the 1vsAll setting.

    Attributes
    ----------
    train_data : torch.LongTensor
        A tensor version of `train_set_idx`, prepared for use with PyTorch models.
    
    target_dim : int
        The dimensionality of the target vector, equivalent to the total number of unique entities in the dataset.
    
    collate_fn : None or callable
        An optional custom collate function for use with a PyTorch DataLoader. By default, it is set to None and can
        be specified after initializing the dataset.

    Note
    ----
    This dataset is optimized for training knowledge graph embedding models using the 1vsAll strategy, where the
    model aims to correctly predict the tail entity from all possible entities given the head entity and relation.
    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs):
        super().__init__()
        assert isinstance(train_set_idx, np.ndarray)
        assert len(train_set_idx) > 0
        self.train_data = torch.LongTensor(train_set_idx)
        self.target_dim = len(entity_idxs)
        self.collate_fn = None

    def __len__(self):
        """
        Returns the total number of triples in the dataset.

        Returns
        -------
        int
            The total number of triples.
        """
        return len(self.train_data)

    def __getitem__(self, idx):
        """
        Retrieves the input data and target vector for the triple at index `idx`.

        The input data consists of the indices for the head entity and relation, while the target vector is a
        one-hot encoded vector with a `1` at the position corresponding to the tail entity's index and `0`s elsewhere.

        Parameters
        ----------
        idx : int
            The index of the triple to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two elements:
            - The input data as a torch.Tensor of shape (2,), containing the indices of the head entity and relation.
            - The target vector as a torch.Tensor of shape (`target_dim`,), a one-hot encoded vector for the tail entity.
        """
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_data[idx, 2]] = 1
        return self.train_data[idx, :2], y_vec


class KvsAll(torch.utils.data.Dataset):
    """
    Creates a dataset for K-vs-All training strategy, inheriting from torch.utils.data.Dataset.
    This dataset is tailored for training scenarios where a model predicts all valid tail entities
    given a head entity and relation pair or vice versa. The labels are multi-hot encoded to represent
    the presence of multiple valid entities.

    Let \(D\) denote a dataset for KvsAll training and be defined as \(D := \{(x, y)_i\}_{i=1}^{N}\), where:
    \(x: (h, r)\) is a unique tuple of an entity \(h \in E\) and a relation \(r \in R\) that has been seen in the input graph.
    \(y\) denotes a multi-label vector \(\in [0, 1]^{|E|}\) is a binary label. For all \(y_i = 1\) s.t. \((h, r, E_i) \in KG\).

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        A numpy array of shape `(n, 3)` representing `n` triples, where each triple consists of
        integer indices corresponding to a head entity, a relation, and a tail entity.
    entity_idxs : dict
        A dictionary mapping entity names (strings) to their unique integer identifiers.
    relation_idxs : dict
        A dictionary mapping relation names (strings) to their unique integer identifiers.
    form : str
        A string indicating the prediction form, either 'RelationPrediction' or 'EntityPrediction'.
    store : dict, optional
        A precomputed dictionary storing the training data points. If provided, it should map
        tuples of entity and relation indices to lists of entity indices. If `None`, the store
        will be constructed from `train_set_idx`.
    label_smoothing_rate : float, default=0.0
        A float representing the rate of label smoothing to be applied. A value of 0 means no
        label smoothing is applied.

    Attributes
    ----------
    train_data : torch.LongTensor
        Tensor containing the input features for the model, typically consisting of pairs of
        entity and relation indices.
    train_target : torch.LongTensor
        Tensor containing the target labels for the model, multi-hot encoded to indicate the
        presence of multiple valid entities.
    target_dim : int
        The dimensionality of the target labels, corresponding to the number of unique entities
        or relations, depending on the `form`.
    collate_fn : None
        Placeholder for a custom collate function to be used with a PyTorch DataLoader. This is
        typically set to `None` and can be overridden as needed.
    
    Note
    -----
    The K-vs-All training strategy is used in scenarios where the task is to predict multiple
    valid entities given a single entity and relation pair. This dataset supports both predicting
    multiple valid tail entities given a head entity and relation (EntityPrediction) and predicting
    multiple valid relations given a pair of entities (RelationPrediction).

    The label smoothing rate can be adjusted to control the degree of smoothing applied to the
    target labels, which can help with regularization and model generalization.
    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs, form, store=None,
                 label_smoothing_rate: float = 0.0):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, np.ndarray)
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

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns
        -------
        int
            The total number of items.
        """
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the input pair (head entity, relation) and the corresponding multi-label target vector for the
        item at index `idx`.

        The target vector is a binary vector of length `target_dim`, where each element indicates the presence or
        absence of a tail entity for the given input pair.

        Parameters
        ----------
        idx : int
            The index of the item to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two elements:
            - The input pair as a torch.Tensor of shape (2,), containing the indices of the head entity and relation.
            - The multi-label target vector as a torch.Tensor of shape (`target_dim`,), indicating the presence or
              absence of each possible tail entity.
        """
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class AllvsAll(torch.utils.data.Dataset):
    """ 
    A dataset class for the All-versus-All (AllvsAll) training strategy suitable for knowledge graph embedding models.
    This strategy considers all possible pairs of entities and relations, regardless of whether they exist in the
    knowledge graph, to predict the associated tail entities.
    
    Let D denote a dataset for AllvsAll training and be defined as D:= {(x,y)_i}_i ^N, where
    x: (h,r) is a possible unique tuple of an entity h \in E and a relation r \in R. Hence N = |E| x |R|
    y: denotes a multi-label vector \in [0,1]^{|E|} is a binary label. \forall y_i =1 s.t. (h, r, E_i) \in KG.
    This setup extends beyond observed triples to include all possible combinations of entities and relations,
    marking non-existent combinations as negatives. It aims to enrich the training data with hard negatives.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        An array of shape `(n, 3)`, where each row represents a triple (head entity index, relation index,
        tail entity index).
    entity_idxs : dict
        A dictionary mapping entity names to their unique integer indices.
    relation_idxs : dict
        A dictionary mapping relation names to their unique integer indices.
    label_smoothing_rate : float, default=0.0
        A parameter for label smoothing to mitigate overfitting by softening the hard labels.

    Attributes
    ----------
    train_data : torch.LongTensor
        A tensor containing all possible pairs of entities and relations derived from the input triples.
    train_target : Union[np.ndarray, list]
        A target structure (either a Numpy array or a list) indicating the existence of a tail entity for
        each head entity and relation pair. It supports multi-label classification where a pair can have
        multiple correct tail entities.
    target_dim : int
        The dimension of the target vector, equal to the total number of unique entities.
    collate_fn : None or callable
        An optional function to merge a list of samples into a batch for loading. If not provided, the default
        collate function of PyTorch's DataLoader will be used.
    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, relation_idxs,
                 label_smoothing_rate=0.0):
        super().__init__()
        assert len(train_set_idx) > 0
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

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset, including both existing and potential triples.

        Returns
        -------
        int
            The total number of items.
        """
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the input pair (head entity, relation) and the corresponding multi-label target vector for the
        item at index `idx`. The target vector is a binary vector of length `target_dim`, where each element indicates
        the presence or absence of a tail entity for the given input pair, including negative samples.

        Parameters
        ----------
        idx : int
            The index of the item to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing two elements:
            - The input pair as a torch.Tensor of shape (2,), containing the indices of the head entity and relation.
            - The multi-label target vector as a torch.Tensor of shape (`target_dim`,), indicating the presence or
              absence of each possible tail entity, including negative samples.
        """
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        existing_indices = self.train_target[idx]
        if len(existing_indices) > 0:
            y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class KvsSampleDataset(torch.utils.data.Dataset):
    """
    Constructs a dataset for KvsSample training strategy, specifically designed for knowledge graph embedding models.
    This dataset formulation is aimed at handling the imbalance between positive and negative examples for each
    (head, relation) pair by subsampling tail entities. The subsampling ensures a balanced representation of positive
    and negative examples in each training batch, according to the specified negative sampling ratio.

    The dataset is defined as \(D:= \{(x,y)_i\}_{i=1}^{N}\), where:
        - \(x: (h,r)\) is a unique head entity \(h \in E\) and a relation \(r \in R\).
        - \(y \in [0,1]^{|E|}\) is a binary label vector. For all \(y_i = 1\) such that \((h, r, E_i) \in KG\).

    At each mini-batch construction, we subsample \(y\), hence \(|new_y| \ll |E|\).
    The new \(y\) contains all 1's if \(sum(y) <\) neg_sample_ratio, otherwise, it contains a balanced mix of 1's and 0's.

    Parameters
    ----------
    train_set : np.ndarray
        An array of shape \((n, 3)\), where \(n\) is the number of triples in the dataset. Each row in the array
        represents a triple \((h, r, t)\), consisting of head entity index \(h\), relation index \(r\), and
        tail entity index \(t\).
    num_entities : int
        The total number of unique entities in the dataset.
    num_relations : int
        The total number of unique relations in the dataset.
    neg_sample_ratio : int
        The ratio of negative samples to positive samples for each (head, relation) pair. If the number of
        available positive samples is less than this ratio, additional negative samples are generated to meet the ratio.
    label_smoothing_rate : float, default=0.0
        A parameter for label smoothing, aiming to mitigate overfitting by softening the hard labels. The labels
        are adjusted towards a uniform distribution, with the smoothing rate determining the degree of softening.

    Attributes
    ----------
    train_data : torch.IntTensor
        A tensor containing the (head, relation) pairs derived from the input triples, used to index the training set.
    train_target : list of numpy.ndarray
        A list where each element corresponds to the tail entity indices associated with a given (head, relation) pair.
    collate_fn : None or callable
        A function to merge a list of samples to form a batch. If None, PyTorch's default collate function is used.
    """

    def __init__(self, train_set: np.ndarray, num_entities, num_relations, neg_sample_ratio: int = None,
                 label_smoothing_rate: float = 0.0):
        super().__init__()
        assert isinstance(train_set, np.ndarray)
        assert isinstance(neg_sample_ratio, int)
        self.train_data = train_set
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_sample_ratio = neg_sample_ratio
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None

        if self.neg_sample_ratio == 0:
            print(f'neg_sample_ratio is {neg_sample_ratio}. It will be set to 10.')
            self.neg_sample_ratio = 10

        print('Constructing training data...')
        store = mapping_from_first_two_cols_to_third(train_set)
        self.train_data = torch.IntTensor(list(store.keys()))
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # TLDL; replace Python objects with non-refcounted representations such as Pandas, Numpy or PyArrow objects
        # Unsure whether a list of numpy arrays are non-refcounted
        self.train_target = list([np.array(i) for i in store.values()])
        del store
        # @TODO: Investigate reference counts of using list of numpy arrays.
        # import sys
        # import gc
        # print(sys.getrefcount(self.train_target))
        # print(sys.getrefcount(self.train_target[0]))
        # print(gc.get_referrers(self.train_target))
        # print(gc.get_referrers(self.train_target[0]))

    def __len__(self):
        """
        Returns the total number of unique (head, relation) pairs in the dataset.

        Returns
        -------
        int
            The number of unique (head, relation) pairs.
        """
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        """
        Retrieves the data for the given index, including the (head, relation) pair, selected tail entity indices,
        and their labels. Positive examples are sampled from the training set, and negative examples are generated
        by randomly selecting tail entities not associated with the (head, relation) pair.

        Parameters
        ----------
        idx : int
            The index of the (head, relation) pair in the dataset.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - x: The (head, relation) pair as a torch.Tensor.
            - y_idx: The indices of selected tail entities, both positive and negative, as a torch.IntTensor.
            - y_vec: The labels for the selected tail entities, with 1s indicating positive and 0s indicating negative
                     examples, as a torch.Tensor.
        """
        # (1) Get i.th unique (head,relation) pair.
        x = self.train_data[idx]
        # (2) Get tail entities given (1).
        positives_idx = self.train_target[idx]
        num_positives = len(positives_idx)
        # (3) Do we need to subsample (2) to create training data points of same size.
        if num_positives < self.neg_sample_ratio:
            # (3.1) Take all tail entities as positive examples
            positives_idx = torch.IntTensor(positives_idx)
            # (3.2) Generate more negative entities
            negative_idx = torch.randint(low=0,
                                         high=self.num_entities,
                                         size=(self.neg_sample_ratio + self.neg_sample_ratio - num_positives,))
        else:
            # (3.1) Subsample positives without replacement.
            positives_idx = torch.IntTensor(np.random.choice(positives_idx, size=self.neg_sample_ratio, replace=False))
            # (3.2) Generate random entities.
            negative_idx = torch.randint(low=0,
                                         high=self.num_entities,
                                         size=(self.neg_sample_ratio,))
        # (5) Create selected indexes.
        y_idx = torch.cat((positives_idx, negative_idx), 0)
        # (6) Create binary labels.
        y_vec = torch.cat((torch.ones(len(positives_idx)), torch.zeros(len(negative_idx))), 0)
        return x, y_idx, y_vec


class NegSampleDataset(torch.utils.data.Dataset):
    def __init__(self, train_set: np.ndarray, num_entities: int, num_relations: int, neg_sample_ratio: int = 1):
        assert isinstance(train_set, np.ndarray)
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # TLDL; replace Python objects with non-refcounted representations such as Pandas, Numpy or PyArrow objects
        self.neg_sample_ratio = torch.tensor(
            neg_sample_ratio)
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

    @timeit
    def __init__(self, train_set: np.ndarray, num_entities: int, num_relations: int, neg_sample_ratio: int = 1,
                 label_smoothing_rate: float = 0.0):
        assert isinstance(train_set, np.ndarray)
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        # TLDL; replace Python objects with non-refcounted representations such as Pandas, Numpy or PyArrow objects
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.neg_sample_ratio = torch.tensor(
            neg_sample_ratio)  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        self.train_set = torch.from_numpy(train_set)
        assert num_entities >= max(self.train_set[:, 0]) and num_entities >= max(self.train_set[:, 2])
        self.length = len(self.train_set)
        self.num_entities = torch.tensor(num_entities)
        self.num_relations = torch.tensor(num_relations)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.train_set[idx]

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
