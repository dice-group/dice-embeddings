import numpy as np
import pytest
import torch

from dicee.dataset_classes import AllvsAll


@pytest.mark.parametrize(('triples', 'expected'), [
    ([[0, 0, 0], [0, 0, 1]], [[1., 1.], [0., 0.]]),
    ([[0, 0, 1], [1, 0, 0]], [[0., 1.], [1., 0.]]),
    ([[0, 0, 1]], [[0., 1.], [0., 0.]]),
])
def test_allvsall_target_lengths(triples, expected):
    dataset = AllvsAll(np.array(triples), {'a': 0, 'b': 1}, {'r': 0})
    labels = torch.stack([dataset[i][1] for i in range(len(dataset))])
    torch.testing.assert_close(labels, torch.tensor(expected))
