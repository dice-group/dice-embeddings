"""Label-based (multi-label / multi-class) dataset classes.

Provides ``KvsAll``, ``AllvsAll``, ``KvsSampleDataset``, and
``OnevsAllDataset`` — datasets where each sample is a ``(head, relation)``
pair and the target is a label vector over all entities (or relations).
"""

import logging

import numpy as np
import torch

from ._storage import PairIndex, RaggedIndices, WorkerDataset

logger = logging.getLogger(__name__)


class OnevsAllDataset(WorkerDataset):
    """Dataset for the 1-vs-All training strategy (multi-class).

    Each sample is a ``(head, relation)`` pair with a one-hot target vector
    whose single active position corresponds to the true tail entity.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    entity_idxs : dict
        Entity-name → index mapping (used to determine the target dimension).
    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs):
        super().__init__()
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        assert len(train_set_idx) > 0
        # Sort by (head, relation, tail) to ensure order-independent training
        # This prevents different input orderings from affecting optimization
        sorted_indices = np.lexsort(
            (train_set_idx[:, 2], train_set_idx[:, 1], train_set_idx[:, 0])
        )
        self.train_data = train_set_idx[sorted_indices]
        self.target_dim = len(entity_idxs)
        self.collate_fn = None

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        triple = torch.from_numpy(self.train_data[idx].copy()).long()
        y_vec[triple[2]] = 1
        return triple[:2], y_vec


class KvsAll(WorkerDataset):
    """Dataset for KvsAll training (multi-label).

    D := {(x, y)_i}_{i=1}^{N} where
        * x = (h, r) is a unique (entity, relation) pair observed in the KG,
        * y ∈ [0, 1]^{|E|} is a multi-label vector with y_j = 1 iff
          (h, r, e_j) ∈ KG.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    entity_idxs : dict
        Entity-name → index mapping.
    relation_idxs : dict
        Relation-name → index mapping.
    form : str
        ``'EntityPrediction'`` or ``'RelationPrediction'``.
    label_smoothing_rate : float, optional
        Label smoothing coefficient (default ``0.0``).
    """

    def __init__(
        self,
        train_set_idx: np.ndarray,
        entity_idxs,
        relation_idxs,
        form,
        store=None,
        label_smoothing_rate: float = 0.0,
    ):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None

        if store is not None:
            raise ValueError("A prebuilt store is not supported")
        if form == "RelationPrediction":
            self.target_dim = len(relation_idxs)
            index = PairIndex.from_triples(train_set_idx, columns=(0, 2, 1))
        elif form == "EntityPrediction":
            self.target_dim = len(entity_idxs)
            index = PairIndex.from_triples(train_set_idx)
        else:
            raise NotImplementedError(form)
        self.train_data = index.keys
        self.train_target = index.targets

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class AllvsAll(WorkerDataset):
    """Dataset for AllvsAll training (multi-label, exhaustive).

    Extends the ``KvsAll`` idea: every *possible* ``(entity, relation)``
    combination is included — not just those observed in the KG.  Pairs
    without any known tail entities receive an all-zeros label vector.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    entity_idxs : dict
        Entity-name → index mapping.
    relation_idxs : dict
        Relation-name → index mapping.
    label_smoothing_rate : float, optional
        Label smoothing coefficient (default ``0.0``).
    """

    def __init__(
        self,
        train_set_idx: np.ndarray,
        entity_idxs,
        relation_idxs,
        label_smoothing_rate=0.0,
    ):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        self.train_data = None
        self.train_target = None
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None

        self.target_dim = len(entity_idxs)
        index = PairIndex.from_triples(train_set_idx)
        num_relations = len(relation_idxs)
        num_pairs = self.target_dim * num_relations
        pair_ids = np.arange(num_pairs, dtype=np.int64)
        self.train_data = torch.from_numpy(np.column_stack((pair_ids // num_relations,
                                                           pair_ids % num_relations)))
        lengths = np.zeros(num_pairs, dtype=np.int64)
        keys = index.keys.numpy()
        lengths[keys[:, 0] * num_relations + keys[:, 1]] = np.diff(index.targets.offsets.numpy())
        offsets = np.r_[0, lengths.cumsum()]
        self.train_target = RaggedIndices(index.targets.values, offsets)
        logger.info("Number of unique augmented pairs: %s", num_pairs)

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        existing_indices = self.train_target[idx]
        if len(existing_indices) > 0:
            y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class KvsSampleDataset(WorkerDataset):
    """Dataset for KvsSample training (dynamic multi-label).

    Like ``KvsAll`` but sub-samples the target vector at each access to keep
    mini-batch sizes manageable when the entity set is large.

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    entity_idxs : dict
        Entity-name → index mapping.
    relation_idxs : dict
        Relation-name → index mapping.
    form : str
        ``'EntityPrediction'``.
    neg_ratio : int
        Number of negative samples per positive target.
    label_smoothing_rate : float, optional
        Label smoothing coefficient (default ``0.0``).
    """

    def __init__(
        self,
        train_set_idx: np.ndarray,
        entity_idxs,
        relation_idxs,
        form,
        store=None,
        neg_ratio=None,
        label_smoothing_rate: float = 0.0,
    ):
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
        index = PairIndex.from_triples(train_set_idx)
        self.train_data = index.keys
        self.train_target = index.targets
        self.max_num_of_classes = self.train_target.max_length + self.neg_ratio

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # (1) Get i-th unique (head, relation) pair.
        x = self.train_data[idx]
        # (2) Get tail entities given (1).
        y = self.train_target[idx]
        num_positive_class = len(y)
        num_negative_class = self.max_num_of_classes - num_positive_class
        # Sample negatives
        weights = torch.ones(self.num_entities)
        weights[y] = 0.0
        negative_idx = torch.multinomial(
            weights, num_samples=num_negative_class, replacement=True
        )

        y_idx = torch.cat((y, negative_idx), 0)
        y_vec = torch.cat(
            (torch.ones(num_positive_class), torch.zeros(num_negative_class)), 0
        )
        return x, y_idx, y_vec


class FSDP1vsSampleDataset(WorkerDataset):
    """Positive-triple dataset for FSDP 1vsSample training with true-negative sampling.

    Each dataset item is a single positive triple (h, r, t).  The collate_fn
    builds the full (source, target_idx, labels) batch in a DataLoader worker,
    sampling true negatives via index remapping so they never coincide with any
    known positive tail for that (h, r) pair.

    Fixed batch width follows KvsSample:
        max_num_of_classes = max_positives_per_pair + neg_ratio
    Each sample contributes 1 positive and (max_num_of_classes - 1) negatives.
    """

    def __init__(
        self,
        train_set_idx: np.ndarray,
        entity_idxs,
        relation_idxs,
        form,
        neg_ratio=None,
        label_smoothing_rate: float = 0.0,
    ):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        assert form == "EntityPrediction"
        assert neg_ratio is not None

        self.train_data = train_set_idx
        self.num_entities = len(entity_idxs)
        self.num_relations = len(relation_idxs)
        self.neg_ratio = neg_ratio
        self.label_smoothing_rate = label_smoothing_rate

        # Index remapping requires sorted, distinct positives. Keep the table
        # ragged so a high-degree query does not pad every other query in the KG.
        self._positive_index = PairIndex.from_triples(train_set_idx, unique=True)
        # Preserve the old batch width even if duplicate input triples exist;
        # exclusion itself uses distinct positives for correct index remapping.
        max_pos = self._positive_index.max_input_count
        self.max_num_of_classes = max_pos + neg_ratio
        self.num_negatives = self.max_num_of_classes - 1

        self.collate_fn = self._collate

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.train_data[idx].copy()).long()

    def _collate(self, batch):
        """Build a fixed-width (source, target_idx, labels) batch.

        Runs in a DataLoader worker, overlapped with GPU compute.
        Negative sampling uses index remapping: sample from [0, N-k) then
        slide each value past the k sorted true positives for that (h, r) pair.
        No rejection needed; output is always exactly num_negatives per row.
        """
        triples = np.stack([t.numpy() for t in batch])   # (B, 3)
        B = triples.shape[0]
        pos_t = triples[:, 2]

        pair_ids = self._positive_index.find_rows(triples[:, :2])
        pos_pad, k_per_item = self._positive_index.targets.padded_rows(pair_ids, self.num_entities)
        if self.num_negatives and np.any(k_per_item >= self.num_entities):
            raise ValueError("No valid negative candidates for a training query")

        # Sample negatives from reduced range [0, num_entities - k_i) per item
        upper = (self.num_entities - k_per_item).astype(np.float64)  # (B,)
        negs = (
            np.random.uniform(size=(B, self.num_negatives)) * upper[:, np.newaxis]
        ).astype(np.int64)   # (B, num_negatives)

        # Index remapping: for each positive p (sorted), increment every neg >= p.
        # Sentinel comparisons (neg >= num_entities) are always False — no-op.
        for j in range(pos_pad.shape[1]):
            negs += negs >= pos_pad[:, j : j + 1]

        source = torch.from_numpy(triples[:, :2].astype(np.int64))
        target_idx = torch.from_numpy(
            np.concatenate([pos_t.reshape(-1, 1), negs], axis=1)   # (B, max_num_of_classes)
        )

        ls = self.label_smoothing_rate
        labels = torch.cat(
            [
                torch.full((B, 1), 1.0 - ls, dtype=torch.float32),
                torch.full((B, self.num_negatives), ls, dtype=torch.float32),
            ],
            dim=1,
        )   # (B, max_num_of_classes)

        return source, target_idx, labels
