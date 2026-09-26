"""Cache, ranking, resume, and CPU/CUDA training equivalence."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dicee.evaluation._filtering import FilteredRanker
from dicee.query_answering import AdapterQuery, QueryContext, QueryScoreAdapter, fit_query_adapter
from dicee.query_answering._query import ULTRAQUERY_SHAPES
from dicee.query_answering._training_rows import TrainingRows
from dicee.query_answering.training import _bank, _filtered_ranks, _query_logs
from tests.test_query_adapter_training import data, source  # noqa: F401
from tests.test_query_engine import TableModel, programs


@pytest.mark.parametrize('normalization', ['none', 'standard'])
def test_features_survive_eviction_and_reopening(tmp_path, normalization):
    model = TableModel(np.random.default_rng(11).normal(size=(8, 2, 8)))
    context = QueryContext(((0, 0, 1),), 8, 2)
    adapter = QueryScoreAdapter('context_scores_v1', 1., normalization=normalization)
    options = dict(cache_dir=tmp_path, row_batch_size=2, seed=0, samples=None, cache_bytes=0)
    pairs = [(0, 0), (1, 1), (0, 0)]
    bank = TrainingRows(model, context, **options)
    actual = bank.prepared(adapter, pairs)
    raw = model.table.detach()[[p[0] for p in pairs], [p[1] for p in pairs]]
    rows = [adapter.prepare(row[None], *context.features([pair], device='cpu')) for pair, row in zip(pairs, raw)]
    expected = tuple(torch.cat([row[i] for row in rows]) for i in range(3))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    calls = len(model.calls)
    assert bank.stats['feature_preparations'] == 2
    bank.close()
    bank = TrainingRows(model, context, **options)
    for a, b in zip(bank.prepared(adapter, pairs), actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert bank.stats['feature_preparations'] == 0 and len(model.calls) == calls
    bank.connection.execute("UPDATE features SET sha256='corrupt'")
    bank.connection.commit()
    bank.feature_cache.clear()
    with pytest.raises(ValueError, match='Corrupt training features'):
        bank.prepared(adapter, pairs)
    bank.close()


@pytest.mark.parametrize('hard_count', [1, 7, 8, 12])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA'))])
def test_filtered_ranking_preserves_ties_and_infinities(hard_count, device):
    scores = torch.tensor([0., 0., -1., -1., -2., -torch.inf] * 8, dtype=torch.float64, device=device)
    answers, hard = set(range(20)), set(range(hard_count))
    ranker = FilteredRanker('optimistic')
    expected = [b + t / 2 for b, t in (
        ranker.bounds_batch(scores[None], [a], [sorted(answers)])[0] for a in sorted(hard))]
    assert _filtered_ranks(scores, answers, hard) == expected


def test_fit_resumes_optimizer_shuffle_and_best_checkpoint(data, tmp_path):  # noqa: F811
    model = TableModel(np.random.default_rng(20).normal(size=(8, 6, 8)))
    options = dict(epochs=6, validation_every=2, seed=42)
    expected = fit_query_adapter(model, [data], **options)
    path = tmp_path / 'fit.pt'
    def interrupt(record):
        if record['epoch'] == 4:
            raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        fit_query_adapter(model, [data], checkpoint_path=path, on_epoch=interrupt, **options)
    epochs = []
    actual = fit_query_adapter(model, [data], checkpoint_path=path, on_epoch=lambda r: epochs.append(r['epoch']), **options)
    assert epochs == [5, 6]
    torch.testing.assert_close(actual.adapter.weights, expected.adapter.weights, rtol=0, atol=0)
    assert actual.validation == expected.validation
    assert actual.adapter.metadata['training']['selected_epoch'] == expected.adapter.metadata['training']['selected_epoch']
    assert [r['loss'] for r in actual.history] == [r['loss'] for r in expected.history]
    with pytest.raises(ValueError, match='checkpoint inputs/settings changed'):
        fit_query_adapter(model, [data], checkpoint_path=path, beam_size=3, **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
@pytest.mark.parametrize('tnorm', ['prod', 'min'])
def test_all_query_shapes_cpu_cuda_scores_and_gradients(tmp_path, tnorm):
    raw = np.random.default_rng(19).normal(size=(4, 2, 4))
    model = TableModel(raw)
    context = QueryContext(((0, 0, 1), (1, 1, 2)), 4, 2)
    queries = [AdapterQuery(q, {0}) for s, (q, _) in programs(raw, 2).items() if s in ULTRAQUERY_SHAPES]
    fixture = SimpleNamespace(context=context, train=queries, validation=())
    weights = np.random.default_rng(1).normal(0, .1, (2, 8))
    banks, adapters = [], []
    for device in ['cpu', 'cuda']:
        bank = _bank(model, fixture, cache_dir=tmp_path, row_batch_size=2, seed=0, samples=None,
                     device=device, device_cache_bytes=256)
        bank.update(beam_size=2, device=device, tnorm=tnorm)
        banks.append(bank)
        adapters.append(QueryScoreAdapter('context_scores_v1', 1., weights=weights).to(device))
    for query in queries:
        scores, grads = [], []
        for adapter, bank in zip(adapters, banks):
            adapter.zero_grad()
            logs = _query_logs(adapter, bank, query)
            (logs.exp() * torch.arange(1., 5., device=logs.device)).sum().backward()
            scores.append(logs.detach().cpu())
            grads.append(adapter.weights.grad.cpu())
        torch.testing.assert_close(*scores, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(*grads, rtol=1e-10, atol=1e-12)
    provider = banks[1]['provider']
    assert provider.device_cache_used <= provider.device_cache_bytes
    assert all(not v.requires_grad for _, entry in provider.device_cache.values() for v in entry)
    for bank in banks:
        bank['provider'].close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')
@pytest.mark.parametrize('validation_device', ['cpu', 'cuda'])
@pytest.mark.parametrize('tnorm', ['prod', 'min'])
def test_fit_cpu_cuda_checkpoint_selection(data, tmp_path, validation_device, tnorm):  # noqa: F811
    model = TableModel(np.random.default_rng(20).normal(size=(8, 6, 8)))
    options = dict(epochs=6, validation_every=2, cache_dir=tmp_path, tnorm=tnorm)
    cpu = fit_query_adapter(model, [data], **options)
    gpu = fit_query_adapter(model, [data], training_device='cuda', validation_device=validation_device, **options)
    torch.testing.assert_close(cpu.adapter.weights, gpu.adapter.weights.cpu(), rtol=1e-10, atol=1e-12)
    assert cpu.validation == gpu.validation
    assert cpu.adapter.metadata['training']['selected_epoch'] == gpu.adapter.metadata['training']['selected_epoch']


def test_continuation_preserves_completed_fits_and_upgrades_verified_rows(tmp_path):
    import dicee
    from dicee.query_answering.benchmark import _implementation_fingerprint
    from dicee.query_answering.context import fingerprint
    from dicee.scripts.benchmark_query_adapters import continue_fits, sha
    previous, output = tmp_path/'previous', tmp_path/'output'
    shutil.copytree(Path(dicee.__file__).parent, previous/'source'/'dicee',
                    ignore=shutil.ignore_patterns('__pycache__'))
    parent = dict(implementation=_implementation_fingerprint(), backbones=['ultra'], variants={'types_2': {}})
    (previous/'configuration.json').write_text(json.dumps(parent))
    model = TableModel(np.random.default_rng(11).normal(size=(4, 2, 4)))
    context = QueryContext((), 4, 2)
    options = dict(row_batch_size=2, seed=0, samples=None)
    bank = TrainingRows(model, context, cache_dir=previous/'score-banks'/'ultra', **options)
    adapter = QueryScoreAdapter('global')
    expected = bank.prepared(adapter, [(0, 0)])
    settings = json.loads(bank.connection.execute('SELECT settings FROM metadata').fetchone()[0])
    settings.pop('scoring_code')
    settings.update(version=1, implementation=parent['implementation'])
    bank.connection.execute('UPDATE metadata SET identity=?,settings=?', (fingerprint(settings), json.dumps(settings, sort_keys=True)))
    bank.connection.commit()
    bank.close()
    fit = previous/'adapters'/'ultra'/'types_2'/'all_sources'
    fit.mkdir(parents=True)
    adapter.save(fit/'adapter.json')
    (fit/'report.json').write_text(json.dumps(dict(adapter_sha256=sha(fit/'adapter.json'))))
    config = dict(parent, continue_from=str(previous), continued_configuration_sha256=sha(previous/'configuration.json'))
    continue_fits(output, config)
    assert json.loads((output/'continuation.json').read_text())['fits'][0]['fold'] == 'all_sources'
    assert (output/'adapters'/'ultra'/'types_2'/'all_sources'/'adapter.json').read_bytes() == (fit/'adapter.json').read_bytes()
    calls = len(model.calls)
    restored = TrainingRows(model, context, cache_dir=output/'score-banks'/'ultra', **options)
    for a, b in zip(restored.prepared(adapter, [(0, 0)]), expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(model.calls) == calls
    restored.close()
    (previous/'source'/'dicee'/'query_answering'/'engine.py').write_text('changed')
    with pytest.raises(ValueError, match='implementation changed'):
        continue_fits(output, config)
