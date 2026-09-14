import numpy as np
import pytest
import torch

from dicee.dataset_classes import KvsSampleDataset


@pytest.mark.parametrize('smoothing', [0.0, 0.1, 1.0])
def test_sampled_labels_are_smoothed(smoothing):
    dataset = KvsSampleDataset(
        np.array([[0, 0, 1], [0, 0, 2], [1, 0, 0]]),
        {'a': 0, 'b': 1, 'c': 2}, {'r': 0}, 'EntityPrediction',
        neg_ratio=2, label_smoothing_rate=smoothing,
    )
    for i, positives in enumerate([{1, 2}, {0}]):
        _, indices, labels = dataset[i]
        expected = torch.tensor([1 - smoothing if idx.item() in positives else smoothing
                                 for idx in indices])
        torch.testing.assert_close(labels, expected)
        assert labels.shape == indices.shape == (4,)
