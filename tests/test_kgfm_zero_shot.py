"""The benchmark must reject test leakage unless explicitly requested."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / 'benchmarks/kgfm_zero_shot.py'
    spec = importlib.util.spec_from_file_location('kgfm_zero_shot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlap_requires_opt_in_and_preserves_splits(benchmark):
    train = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 2]])
    valid = np.array([[1, 0, 2], [2, 0, 3]])
    test = np.array([[0, 0, 1], [2, 0, 3], [3, 0, 4]])
    originals = [split.copy() for split in (train, valid, test)]
    with pytest.raises(ValueError, match='Training and test facts overlap'):
        benchmark.check_split_overlap(train, valid, test)
    assert benchmark.check_split_overlap(train, valid, test, allow=True) == {
        'train_valid': 1, 'train_test': 1, 'valid_test': 1,
    }
    for split, original in zip((train, valid, test), originals):
        np.testing.assert_array_equal(split, original)


def test_disjoint_splits_need_no_override(benchmark):
    splits = [np.array([triple]) for triple in ((0, 0, 1), (1, 0, 2), (2, 0, 3))]
    assert benchmark.check_split_overlap(*splits) == {
        'train_valid': 0, 'train_test': 0, 'valid_test': 0,
    }
