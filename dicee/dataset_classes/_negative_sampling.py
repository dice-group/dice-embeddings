"""Negative-sampling based dataset classes.

Provides ``TriplePredictionDataset``, ``NegSampleDataset``, and
``OnevsSample`` — datasets that generate negative triples by corrupting
head or tail entities at training time.
"""

from typing import List

import numpy as np
import torch


class OnevsSample(torch.utils.data.Dataset):
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
        num_entities,
        num_relations,
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


class NegSampleDataset(torch.utils.data.Dataset):
    """Pre-computed negative sampling dataset.

    At construction time every positive triple is paired with one random
    negative (head- or tail-corrupted).  The pairs are stored so that
    ``__getitem__`` is a simple lookup.

    Parameters
    ----------
    train_set : numpy.ndarray
        ``(N, 3)`` integer-indexed triples.
    num_entities : int
        Total number of entities.
    num_relations : int
        Total number of relations.
    neg_sample_ratio : int, optional
        Currently unused; kept for API compatibility (default ``1``).
    """

    def __init__(
        self,
        train_set: np.ndarray,
        num_entities: int,
        num_relations: int,
        neg_sample_ratio: int = 1,
    ):
        assert isinstance(train_set, np.ndarray)
        self.neg_sample_ratio = torch.tensor(neg_sample_ratio)

        # Sort by (head, relation, tail) to ensure order-independent training
        sorted_indices = np.lexsort(
            (train_set[:, 2], train_set[:, 1], train_set[:, 0])
        )
        sorted_train_set = train_set[sorted_indices]

        self.train_triples = torch.from_numpy(sorted_train_set).unsqueeze(1)
        self.length = len(self.train_triples)
        self.num_entities = torch.tensor(num_entities)
        self.num_relations = torch.tensor(num_relations)
        self.labels = torch.tensor([1.0, 0.0])

        # Precompute negatives and stack with positives
        self.train_set = []
        for triple in self.train_triples:
            corr_entities = torch.randint(0, high=self.num_entities, size=(1,))
            if torch.rand(1) >= 0.5:
                negative_triple = torch.cat(
                    (triple[:, 0], triple[:, 1], corr_entities), dim=0
                ).unsqueeze(0)
            else:
                negative_triple = torch.cat(
                    (corr_entities, triple[:, 1], triple[:, 2]), dim=0
                ).unsqueeze(0)
            self.train_set.append(torch.cat((triple, negative_triple), dim=0))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.train_set[idx], self.labels


class TriplePredictionDataset(torch.utils.data.Dataset):
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
    ):
        assert isinstance(train_set, np.ndarray)
        self.label_smoothing_rate = torch.tensor(label_smoothing_rate)
        self.neg_sample_ratio = torch.tensor(neg_sample_ratio)

        # Sort by (head, relation, tail) to ensure order-independent training
        sorted_indices = np.lexsort(
            (train_set[:, 2], train_set[:, 1], train_set[:, 0])
        )
        self.train_set = train_set[sorted_indices]

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
