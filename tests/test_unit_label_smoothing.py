import numpy as np
import pytest
import torch

from dicee.dataset_classes import AllvsAll, KvsAll


@pytest.mark.parametrize('dataset_class', [KvsAll, AllvsAll])
@pytest.mark.parametrize('smoothing', [0.0, 0.1, 1.0])
def test_label_smoothing(dataset_class, smoothing):
    kwargs = {'form': 'EntityPrediction'} if dataset_class is KvsAll else {}
    dataset = dataset_class(
        np.array([[0, 0, 1]]), {'a': 0, 'b': 1}, {'r': 0},
        label_smoothing_rate=smoothing, **kwargs,
    )
    torch.testing.assert_close(
        dataset[0][1], torch.tensor([smoothing / 2, 1 - smoothing / 2]),
    )
    if dataset_class is AllvsAll:
        torch.testing.assert_close(dataset[1][1], torch.full((2,), smoothing / 2))
