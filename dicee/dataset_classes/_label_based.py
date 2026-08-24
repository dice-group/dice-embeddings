"""Label-based (multi-label / multi-class) dataset classes.

Provides ``KvsAll``, ``AllvsAll``, ``KvsSampleDataset``, and
``OnevsAllDataset`` — datasets where each sample is a ``(head, relation)``
pair and the target is a label vector over all entities (or relations).
"""

import logging

import numpy as np
import torch

from ..static_preprocess_funcs import mapping_from_first_two_cols_to_third

logger = logging.getLogger(__name__)


class OnevsAllDataset(torch.utils.data.Dataset):
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


class KvsAll(torch.utils.data.Dataset):
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

        if store is None:
            store = dict()
            if form == "RelationPrediction":
                self.target_dim = len(relation_idxs)
                for s_idx, p_idx, o_idx in train_set_idx:
                    store.setdefault((s_idx, o_idx), list()).append(p_idx)
                # Sort keys to ensure order-independent training
                store = dict(sorted(store.items()))
            elif form == "EntityPrediction":
                self.target_dim = len(entity_idxs)
                # mapping_from_first_two_cols_to_third already returns sorted dict
                store = mapping_from_first_two_cols_to_third(train_set_idx)
            else:
                raise NotImplementedError
        else:
            raise ValueError()
        assert len(store) > 0

        self.train_data = torch.LongTensor(list(store.keys()))

        if sum(len(i) for i in store.values()) == len(store):
            self.train_target = np.array(list(store.values()))
            try:
                assert isinstance(self.train_target[0], np.ndarray)
            except (IndexError, AssertionError):
                logger.error(self.train_target)
                exit(1)
        else:
            self.train_target = list(store.values())
            assert isinstance(self.train_target[0], list)
        del store

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class AllvsAll(torch.utils.data.Dataset):
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
        # mapping_from_first_two_cols_to_third already returns sorted dict
        store = mapping_from_first_two_cols_to_third(train_set_idx)
        logger.info(f"Number of unique pairs: {len(store)}")
        for i in range(len(entity_idxs)):
            for j in range(len(relation_idxs)):
                if store.get((i, j), None) is None:
                    store[(i, j)] = list()
        logger.info(f"Number of unique augmented pairs: {len(store)}")
        # Re-sort after adding new keys to maintain consistent ordering
        store = dict(sorted(store.items()))
        assert len(store) > 0
        self.train_data = torch.LongTensor(list(store.keys()))

        if sum(len(i) for i in store.values()) == len(store):
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
        y_vec = torch.zeros(self.target_dim)
        existing_indices = self.train_target[idx]
        if len(existing_indices) > 0:
            y_vec[self.train_target[idx]] = 1.0

        if self.label_smoothing_rate:
            y_vec = y_vec * (1 - self.label_smoothing_rate) + (1 / y_vec.size(0))
        return self.train_data[idx], y_vec


class KvsSampleDataset(torch.utils.data.Dataset):
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
        store = mapping_from_first_two_cols_to_third(train_set_idx)
        assert len(store) > 0
        self.train_data = torch.LongTensor(list(store.keys()))
        self.train_target = list(store.values())
        self.max_num_of_classes = (
            max(len(i) for i in self.train_target) + self.neg_ratio
        )
        del store

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

        y_idx = torch.cat((torch.LongTensor(y), negative_idx), 0)
        y_vec = torch.cat(
            (torch.ones(num_positive_class), torch.zeros(num_negative_class)), 0
        )
        return x, y_idx, y_vec


class FSDP1vsSampleDataset(torch.utils.data.Dataset):
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

        # Build er_vocab: (h_idx, r_idx) -> sorted list of positive tail indices
        er_vocab = mapping_from_first_two_cols_to_third(train_set_idx)

        # Fixed row width: same formula as KvsSample
        self.max_num_of_classes = max(len(v) for v in er_vocab.values()) + neg_ratio
        self.num_negatives = self.max_num_of_classes - 1  # 1 slot taken by the positive

        # Build padded positive table for vectorised index remapping.
        # Sentinel = num_entities (valid IDs are [0, num_entities-1]).
        # Rows where the pad fires (negs >= num_entities) never exist after remapping,
        # so the sentinel is naturally a no-op in the increment loop.
        max_pos = max(len(v) for v in er_vocab.values())
        self._pos_table = np.full(
            (len(er_vocab), max_pos),
            fill_value=self.num_entities,
            dtype=np.int32,
        )
        self._pair_to_id: dict = {}
        for i, (pair, tails) in enumerate(er_vocab.items()):
            self._pair_to_id[pair] = i
            sorted_tails = sorted(tails)
            self._pos_table[i, :len(sorted_tails)] = sorted_tails
        self._max_pos = max_pos

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
        h, r, pos_t = triples[:, 0], triples[:, 1], triples[:, 2]

        # Python loop only for dict lookups (trivial body, ~100 ns/item)
        pair_ids = np.array(
            [self._pair_to_id.get((int(h[i]), int(r[i])), 0) for i in range(B)],
            dtype=np.int32,
        )

        # Padded positive table for this batch: (B, max_pos)
        pos_pad = self._pos_table[pair_ids].astype(np.int64)   # sentinel = num_entities

        # Per-item actual positive count (non-sentinel entries)
        k_per_item = np.sum(pos_pad < self.num_entities, axis=1)   # (B,)

        # Sample negatives from reduced range [0, num_entities - k_i) per item
        upper = (self.num_entities - k_per_item).astype(np.float64)  # (B,)
        negs = (
            np.random.uniform(size=(B, self.num_negatives)) * upper[:, np.newaxis]
        ).astype(np.int64)   # (B, num_negatives)

        # Index remapping: for each positive p (sorted), increment every neg >= p.
        # Sentinel comparisons (neg >= num_entities) are always False — no-op.
        for j in range(self._max_pos):
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
