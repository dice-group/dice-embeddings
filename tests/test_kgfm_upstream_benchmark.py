"""Protect the independent ranking and acceptance checks in upstream timings."""
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def benchmark(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'benchmarks'))
    spec = importlib.util.spec_from_file_location('kgfm_upstream_benchmark', root / 'benchmarks/kgfm_upstream.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filtered_pessimistic_ranks_keep_target_and_remove_known_positives(benchmark, tmp_path):
    for name, vocab in [('er_vocab.p', {(0, 0): [np.uint16(0), np.uint16(1)]}),
                        ('re_vocab.p', {(0, 1): [np.uint16(0), np.uint16(2)]})]:
        with (tmp_path / name).open('wb') as stream:
            pickle.dump(vocab, stream)
    scores = torch.tensor([[[5., 3., 3.], [2., 2., 9.]]])
    original = scores.clone()
    result = benchmark.filtered_metrics(scores, torch.tensor([[0, 0, 1]]), tmp_path)
    assert result == {'MRR': 0.5, 'H@1': 0., 'H@3': 1., 'H@10': 1.}
    assert torch.equal(scores, original)


@pytest.mark.parametrize('stochastic', [False, True])
def test_comparison_rejects_metric_drift_except_independent_walks(benchmark, tmp_path, stochastic):
    fields = ('test_indices', 'query_batch_size', 'walk_num', 'test_samples', 'seed', 'torch', 'cuda', 'dtype',
              'tf32', 'hardware', 'device', 'checkpoint_sha256', 'train_sha256', 'test_sha256')
    for name, seconds, mrr in [('official', 2., 0.5), ('dice', 1., 0.6)]:
        report = dict.fromkeys(fields, 'same')
        report.update(median_forward_seconds=seconds, metrics={'MRR': mrr, 'H@1': 0.})
        (tmp_path / f'{name}.json').write_text(json.dumps(report))
        torch.save(torch.ones(1, 2, 3), tmp_path / f'{name}.pt')
    if stochastic:
        benchmark.compare(tmp_path, stochastic=True)
    else:
        with pytest.raises(RuntimeError, match='validation failed'):
            benchmark.compare(tmp_path)
    comparison = json.loads((tmp_path / 'comparison.json').read_text())
    assert comparison['scores_close'] and not comparison['metrics_close']
    expected = None if stochastic else False
    assert comparison['validated'] is expected
    assert comparison['speedup'] == 2.


@pytest.mark.parametrize('dice_better', [True, False])
def test_float64_check_requires_dice_to_be_close_and_more_accurate(benchmark, tmp_path, dice_better):
    fields = ('test_indices', 'query_batch_size', 'walk_num', 'test_samples', 'seed', 'torch', 'cuda', 'dtype',
              'tf32', 'hardware', 'device', 'checkpoint_sha256', 'train_sha256', 'test_sha256')
    values = {'official': 1.01, 'dice': 1.00001} if dice_better else {'official': 1.00001, 'dice': 1.01}
    for name, value in values.items():
        report = dict.fromkeys(fields, 'same')
        report.update(median_forward_seconds=1., metrics={'MRR': 0.5, 'H@1': 0.})
        (tmp_path / f'{name}.json').write_text(json.dumps(report))
        torch.save(torch.full((1, 2, 3), value), tmp_path / f'{name}.pt')
    gold = tmp_path / 'official-fp64.pt'
    torch.save(torch.ones(1, 2, 3, dtype=torch.float64), gold)
    gold.with_suffix('.json').write_text(json.dumps(dict.fromkeys(fields, 'same')))
    if dice_better:
        result = benchmark.compare(tmp_path, fp64_path=gold)
        assert result['validated'] and not result['scores_close']
    else:
        with pytest.raises(RuntimeError, match='validation failed'):
            benchmark.compare(tmp_path, fp64_path=gold)
