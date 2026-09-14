from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from dicee.trainer.auto_batch_finder import find_good_batch_size


@pytest.mark.parametrize('safe_batches', [0, 1, 2, 3])
@pytest.mark.parametrize('failure', ['memory_limit', 'oom'])
def test_returns_last_safe_batch(safe_batches, failure):
    dataset = torch.utils.data.TensorDataset(torch.arange(64))
    dataset.collate_fn = None
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    calls = 0

    def step(batch):
        nonlocal calls
        calls += 1
        if failure == 'oom' and calls > safe_batches:
            raise torch.OutOfMemoryError('test')
        return 0.5

    readings = [(50, 100)] * safe_batches + [(5, 100)]
    with patch('torch.cuda.mem_get_info', side_effect=readings), \
         patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=100)), \
         patch('torch.cuda.memory_allocated', return_value=0):
        if safe_batches == 0:
            with pytest.raises(RuntimeError, match='No batch size fits'):
                find_good_batch_size(loader, step, 'cuda')
        else:
            size, runtime = find_good_batch_size(loader, step, 'cuda')
            assert size == 2 ** safe_batches
            assert runtime >= 0
