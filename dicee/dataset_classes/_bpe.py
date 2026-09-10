"""Byte-pair encoding (BPE) related dataset classes.

Provides datasets for training knowledge graph embedding models that operate
on byte-pair encoded entity and relation representations.
"""

from typing import List, Tuple

import numpy as np
import torch

from ._storage import RaggedIndices, WorkerDataset


class BPE_NegativeSamplingDataset(WorkerDataset):
    """Dataset for negative sampling with byte-pair encoded triples.

    Each sample is a BPE-encoded triple. The custom ``collate_fn`` constructs
    negatives by corrupting head or tail entities with random BPE entities.

    Parameters
    ----------
    train_set : torch.LongTensor
        Integer-encoded triples of shape ``(N, 3)``.
    ordered_shaped_bpe_entities : torch.LongTensor
        All BPE entity representations, ordered by entity index.
    neg_ratio : int
        Number of negative samples per positive triple.
    """

    def __init__(
        self,
        train_set: torch.LongTensor,
        ordered_shaped_bpe_entities: torch.LongTensor,
        neg_ratio: int,
    ):
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

    def collate_fn(
        self, batch_shaped_bpe_triples: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        batch_of_bpe_triples = torch.stack(batch_shaped_bpe_triples, dim=0)

        size_of_batch, _, token_length = batch_of_bpe_triples.shape

        bpe_h = batch_of_bpe_triples[:, 0, :]
        bpe_r = batch_of_bpe_triples[:, 1, :]
        bpe_t = batch_of_bpe_triples[:, 2, :]

        label = torch.ones((size_of_batch,))
        num_of_corruption = size_of_batch * self.neg_ratio
        # Select bpe entities
        corr_bpe_entities = self.ordered_bpe_entities[
            torch.randint(0, high=self.num_bpe_entities, size=(num_of_corruption,))
        ]

        if torch.rand(1) >= 0.5:
            bpe_h = torch.cat((bpe_h, corr_bpe_entities), 0)
            bpe_r = torch.cat(
                (
                    bpe_r,
                    torch.repeat_interleave(
                        input=bpe_r, repeats=self.neg_ratio, dim=0
                    ),
                ),
                0,
            )
            bpe_t = torch.cat(
                (
                    bpe_t,
                    torch.repeat_interleave(
                        input=bpe_t, repeats=self.neg_ratio, dim=0
                    ),
                ),
                0,
            )
        else:
            bpe_h = torch.cat(
                (
                    bpe_h,
                    torch.repeat_interleave(
                        input=bpe_h, repeats=self.neg_ratio, dim=0
                    ),
                ),
                0,
            )
            bpe_r = torch.cat(
                (
                    bpe_r,
                    torch.repeat_interleave(
                        input=bpe_r, repeats=self.neg_ratio, dim=0
                    ),
                ),
                0,
            )
            bpe_t = torch.cat((bpe_t, corr_bpe_entities), 0)

        bpe_triple = torch.stack((bpe_h, bpe_r, bpe_t), dim=1)
        label = torch.cat((label, torch.zeros(num_of_corruption)), 0)
        return bpe_triple, label


class MultiLabelDataset(WorkerDataset):
    """Multi-label dataset for BPE-based KvsAll / AllvsAll training.

    Each sample is a BPE-encoded ``(head, relation)`` pair together with a
    binary multi-label target vector over all entities.

    Parameters
    ----------
    train_set : torch.LongTensor
        BPE-encoded input pairs of shape ``(N, 2, token_length)``.
    train_indices_target : sequence of integer sequences or RaggedIndices
        Per-sample lists of positive target entity indices.
    target_dim : int
        Dimensionality of the target vector (number of entities).
    torch_ordered_shaped_bpe_entities : torch.LongTensor
        Ordered BPE entity representations.
    """

    def __init__(
        self,
        train_set: torch.LongTensor,
        train_indices_target,
        target_dim: int,
        torch_ordered_shaped_bpe_entities: torch.LongTensor,
    ):
        super().__init__()
        assert len(train_set) == len(train_indices_target)
        assert target_dim > 0
        self.train_set = train_set
        self.train_indices_target = RaggedIndices.from_rows(train_indices_target)
        self.target_dim = target_dim
        self.num_datapoints = len(self.train_set)
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


class MultiClassClassificationDataset(WorkerDataset):
    """Dataset for autoregressive multi-class classification on sub-word units.

    Splits a flat sequence of sub-word token ids into overlapping windows of
    size ``block_size`` for next-token prediction.

    Parameters
    ----------
    subword_units : numpy.ndarray
        1-D array of sub-word token ids.
    block_size : int, optional
        Context window length (default ``8``).
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
        x = self.train_data[idx : idx + self.block_size]
        y = self.train_data[idx + 1 : idx + 1 + self.block_size]
        return x, y
