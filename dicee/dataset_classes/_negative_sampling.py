"""Negative-sampling based dataset classes.

Provides ``TriplePredictionDataset``, ``FixedNegSampleDataset``, and
``OnevsSample`` — datasets that generate negative triples by corrupting
head or tail entities at training time.
"""

from typing import List, Tuple

import numpy as np
import torch

from ._storage import PairIndex, WorkerDataset


class OnevsSample(WorkerDataset):
    """Dataset for 1-vs-Sample training (dynamic multi-class with negatives).

    For every positive triple ``(h, r, t)`` the dataset draws
    ``neg_sample_ratio`` random entities as negatives and returns a label
    vector that marks the true tail and the negatives.

    Parameters
    ----------
    train_set : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    num_entities : int
        Total number of entities.
    num_relations : int
        Total number of relations.
    neg_sample_ratio : int
        Number of negative samples per positive.
    label_smoothing_rate : float, optional
        Label smoothing coefficient (default ``0.0``).
    """

    def __init__(
        self,
        train_set: np.ndarray,
        num_entities: int,
        num_relations: int,
        neg_sample_ratio: int = None,
        label_smoothing_rate: float = 0.0,
    ):
        super().__init__()
        assert isinstance(train_set, np.ndarray), "train_set must be a numpy array."
        assert isinstance(neg_sample_ratio, int), "neg_sample_ratio must be an integer."
        assert (
            isinstance(num_entities, int) and num_entities > 0
        ), "num_entities must be a positive integer."
        assert (
            isinstance(num_relations, int) and num_relations > 0
        ), "num_relations must be a positive integer."
        assert neg_sample_ratio < num_entities, (
            f"Negative sample ratio {neg_sample_ratio} cannot be larger "
            f"than the number of entities ({num_entities})."
        )
        assert (
            neg_sample_ratio > 0
        ), f"Negative sample ratio {neg_sample_ratio} must be greater than 0."

        # Sort by (head, relation, tail) to ensure order-independent training
        sorted_indices = np.lexsort(
            (train_set[:, 2], train_set[:, 1], train_set[:, 0])
        )
        sorted_train_set = train_set[sorted_indices]

        self.train_data = torch.from_numpy(sorted_train_set).long()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.neg_sample_ratio = neg_sample_ratio
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.collate_fn = None

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        triple = self.train_data[idx]
        x = triple[:2]
        y = triple[-1].unsqueeze(0)

        weights = torch.ones(self.num_entities)
        weights[y] = 0.0

        negative_idx = torch.multinomial(
            weights, num_samples=self.neg_sample_ratio, replacement=False
        )
        y_idx = torch.cat((y, negative_idx), 0).long()

        y_vec = torch.cat(
            (
                torch.ones(1) - self.label_smoothing_rate,
                torch.zeros(self.neg_sample_ratio) + self.label_smoothing_rate,
            ),
            0,
        )
        return x, y_idx, y_vec


class FixedNegSampleDataset(WorkerDataset):
    """Pre-computed (fixed) negative sampling dataset.

    At construction time every positive triple is paired with one random
    negative (head- or tail-corrupted) using vectorized operations for
    efficiency.  The pairs are stored so that ``__getitem__`` is a simple
    lookup.

    This is useful when you want deterministic negatives across epochs
    (e.g., for reproducibility or debugging).

    Parameters
    ----------
    train_set : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    num_entities : int
        Total number of entities.
    num_relations : int
        Total number of relations.
    neg_sample_ratio : int, optional
        Number of negative samples per positive triple (default ``1``).
    label_smoothing_rate : float, optional
        Label smoothing coefficient (default ``0.0``).
    """

    def __init__(
        self,
        train_set: np.ndarray,
        num_entities: int,
        num_relations: int,
        neg_sample_ratio: int = 1,
        label_smoothing_rate: float = 0.0,
        seed: int = None,
    ):
        assert isinstance(train_set, np.ndarray)
        self.neg_sample_ratio = neg_sample_ratio
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.label_smoothing_rate = label_smoothing_rate
        self.collate_fn = None
        self.seed = seed
        # Sort by (head, relation, tail) to ensure order-independent training
        sorted_indices = np.lexsort(
            (train_set[:, 2], train_set[:, 1], train_set[:, 0])
        )
        sorted_train_set = train_set[sorted_indices]
        self.train_triples = torch.from_numpy(sorted_train_set).long()

        self.length = len(self.train_triples)

        # Vectorized negative generation
        self._precompute_negatives()

    def _precompute_negatives(self) -> None:
        """Vectorized pre-computation of negative triples."""
        n = self.length

        # Set the random seed for reproducibility if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        neg_triples_list = []
        for _ in range(self.neg_sample_ratio):
            # Decide which position to corrupt per triple: 0=head, 1=tail
            # Only corrupt entities (head or tail), not relations, for link prediction
            corruption_choice = torch.randint(0, 2, (n,), dtype=torch.long)

            # Random entities to corrupt with
            corr_entities = torch.randint(0, self.num_entities, (n,), dtype=torch.long)

            # Build negative triples vectorized
            neg_triples = self.train_triples.clone()

            # Corrupt head for triples where corruption_choice == 0
            corrupt_head = corruption_choice == 0
            neg_triples[corrupt_head, 0] = corr_entities[corrupt_head]

            # Corrupt tail for triples where corruption_choice == 1
            corrupt_tail = corruption_choice == 1
            neg_triples[corrupt_tail, 2] = corr_entities[corrupt_tail]

            neg_triples_list.append(neg_triples)

        # Concatenate all negative triples: shape (neg_sample_ratio * N, 3)
        all_neg_triples = torch.cat(neg_triples_list, dim=0)

        # Concatenate positives and negatives: shape ((1 + neg_sample_ratio) * N, 3)
        self.train_set = torch.cat([self.train_triples, all_neg_triples], dim=0)

        # Create labels: positives get (1 - smoothing), negatives get smoothing
        num_negatives = n * self.neg_sample_ratio
        pos_labels = torch.ones(n) - self.label_smoothing_rate
        neg_labels = torch.zeros(num_negatives) + self.label_smoothing_rate
        self.labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Shuffle positives and negatives together to ensure mixed batches
        shuffle_indices = torch.randperm(len(self.train_set))
        self.train_set = self.train_set[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Update length to reflect total number of samples
        self.length = len(self.train_set)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.train_set[idx], self.labels[idx]


class TriplePredictionDataset(WorkerDataset):
    """Dataset for triple prediction with on-the-fly negative sampling.

    Each item is a single positive triple; the custom ``collate_fn``
    generates a batch of mixed positive and negative triples.

    Parameters
    ----------
    train_set : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    num_entities : int
        Total number of entities.
    num_relations : int
        Total number of relations.
    neg_sample_ratio : int, optional
        Number of negative samples per positive triple (default ``1``).
    label_smoothing_rate : float, optional
        Label smoothing coefficient (default ``0.0``).
    """

    def __init__(
        self,
        train_set: np.ndarray,
        num_entities: int,
        num_relations: int,
        neg_sample_ratio: int = 1,
        label_smoothing_rate: float = 0.0,
        seed: int = None,
        sort_train_set: bool = True,
    ):
        assert isinstance(train_set, np.ndarray)
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.neg_sample_ratio = torch.tensor(neg_sample_ratio)

        # Set the random seed for reproducibility if provided
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        if sort_train_set:
            # Sort by (head, relation, tail) to ensure order-independent training.
            sorted_indices = np.lexsort(
                (train_set[:, 2], train_set[:, 1], train_set[:, 0])
            )
            self.train_set = train_set[sorted_indices]
        else:
            self.train_set = train_set

        assert num_entities >= max(self.train_set[:, 0]) and num_entities >= max(
            self.train_set[:, 2]
        ), (
            f"num_entities: {num_entities}, "
            f"max(self.train_set[:, 0]): {max(self.train_set[:, 0])}, "
            f"max(self.train_set[:, 2]): {max(self.train_set[:, 2])}"
        )
        self.length = len(self.train_set)
        self.num_entities = torch.tensor(num_entities)
        self.num_relations = torch.tensor(num_relations)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.from_numpy(self.train_set[idx].copy()).long()

    def collate_fn(self, batch: List[torch.Tensor]):
        batch = torch.stack(batch, dim=0)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        size_of_batch, _ = batch.shape
        assert size_of_batch > 0
        label = torch.ones((size_of_batch,)) - self.label_smoothing_rate
        corr_entities = torch.randint(
            0,
            high=self.num_entities,
            size=(size_of_batch * self.neg_sample_ratio,),
            dtype=torch.long,
        )
        if torch.rand(1) >= 0.5:
            # corrupt head
            r_head_corr = r.repeat(self.neg_sample_ratio)
            t_head_corr = t.repeat(self.neg_sample_ratio)
            label_head_corr = (
                torch.zeros(len(t_head_corr)) + self.label_smoothing_rate
            )

            h = torch.cat((h, corr_entities), 0)
            r = torch.cat((r, r_head_corr), 0)
            t = torch.cat((t, t_head_corr), 0)
            x = torch.stack((h, r, t), dim=1)
            label = torch.cat((label, label_head_corr), 0)
        else:
            # corrupt tail
            h_tail_corr = h.repeat(self.neg_sample_ratio)
            r_tail_corr = r.repeat(self.neg_sample_ratio)
            label_tail_corr = (
                torch.zeros(len(r_tail_corr)) + self.label_smoothing_rate
            )

            h = torch.cat((h, h_tail_corr), 0)
            r = torch.cat((r, r_tail_corr), 0)
            t = torch.cat((t, corr_entities), 0)
            x = torch.stack((h, r, t), dim=1)
            label = torch.cat((label, label_tail_corr), 0)

        return x, label


class GroupedNegativeSamplingDataset(TriplePredictionDataset):
    """Positive-first negative groups reusable by any indexed triple scorer.

    Strict filtering uses training facts only. Negative candidates are drawn
    with replacement, so even dense queries can request many negatives.
    """

    def __init__(self, *args, strict_negative_sampling=False, **kwargs):
        super().__init__(*args, **kwargs)
        if int(self.neg_sample_ratio) < 1:
            raise ValueError('Grouped negative sampling requires neg_ratio > 0')
        self.strict_negative_sampling = strict_negative_sampling
        self.true_heads = self.true_tails = None
        if strict_negative_sampling:
            self.true_heads = PairIndex.from_triples(self.train_set, columns=(1, 2, 0), unique=True)
            self.true_tails = PairIndex.from_triples(self.train_set, unique=True)

    def collate_fn(self, batch):
        positive = torch.stack(batch)
        n, k = len(positive), int(self.neg_sample_ratio)
        triples = positive[:, None].repeat(1, k + 1, 1)
        if self.strict_negative_sampling:
            tail_rows = self.true_tails.find_rows(positive[:n // 2, :2].numpy())
            head_rows = self.true_heads.find_rows(positive[n // 2:, 1:].numpy())
        for i, (h, r, t) in enumerate(positive.tolist()):
            position = 2 if i < n // 2 else 0
            if self.strict_negative_sampling:
                forbidden = (self.true_tails.targets[int(tail_rows[i])] if position == 2
                             else self.true_heads.targets[int(head_rows[i - n // 2])])
                candidates = torch.ones(int(self.num_entities), dtype=torch.bool)
                candidates[forbidden] = False
                candidates = candidates.nonzero().flatten()
                if not len(candidates):
                    raise ValueError(f'No valid negative candidates for training query {(h, r, t)}')
                negative = candidates[torch.randint(len(candidates), (k,))]
            else:
                negative = torch.randint(int(self.num_entities), (k,))
            triples[i, 1:, position] = negative
        targets = torch.zeros(n, k + 1)
        targets[:, 0] = 1
        targets = targets * (1 - 2 * self.label_smoothing_rate) + self.label_smoothing_rate
        return triples, targets
