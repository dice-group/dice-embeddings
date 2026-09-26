"""Resume, batching and cache parity at the query API boundary."""

import importlib
import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from dicee.models import TRIX, Flock
from dicee.query_answering import QueryAnswerer, benchmark_model, evaluate_benchmark, load_benchmark
from dicee.query_answering._checkpoint import BenchmarkCheckpoint
from dicee.query_answering._query import stable_topk
from dicee.query_answering.context import fingerprint
from dicee.query_answering.engine import AtomicScorer
from tests.test_query_benchmark import fixture_dataset
from tests.test_query_engine import TableModel, wrapper


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('size,k', [(1, 1), (20, 1), (20, 7), (20, 20), (14505, 64)])
@pytest.mark.parametrize('ties', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_beam_matches_stable_full_sort(size, k, ties, device):
    values = torch.randn(size, generator=torch.Generator().manual_seed(91), dtype=torch.float64)
    if ties:
        values = values.round()
        values[::3] = -torch.inf
    values = values.to(device)
    assert torch.equal(stable_topk(values, k), values.argsort(descending=True, stable=True)[:k])


def test_precision_tracking_survives_cudnn_rnn_import(monkeypatch):
    from dicee.models._inference import float32_precision_backends, float32_precision_token
    if not float32_precision_backends():
        pytest.skip('Per-backend precision requires newer PyTorch')
    control = type(torch.backends.cudnn).rnn
    monkeypatch.setattr(control, 'fp32_precision', 'ieee')
    before = float32_precision_token()
    importlib.import_module('torch.backends.cudnn.rnn')
    assert float32_precision_token() == before
    control.fp32_precision = 'tf32'
    assert float32_precision_token() != before


def test_import_does_not_change_inference_precision():
    subprocess.run([sys.executable, '-c',
                    'import torch; before = torch.get_float32_matmul_precision(); '
                    'import dicee; assert torch.get_float32_matmul_precision() == before'], check=True)


def test_interrupted_evaluation_resumes_committed_queries(tmp_path):
    fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    data = load_benchmark(tmp_path, 'WikiTopicsQuery:art')
    scores = torch.arange(data.context.num_entities, dtype=torch.float64)
    calls = []

    def predict(query):
        calls.append(query)
        if len(calls) == 8:
            raise RuntimeError('interrupted')
        return scores

    options = dict(checkpoint_dir=tmp_path / 'resume', checkpoint_identity='fixed-predictor', checkpoint_every=3)
    with pytest.raises(RuntimeError, match='interrupted'):
        evaluate_benchmark(data, predict, **options)
    resumed_calls = []
    resumed = evaluate_benchmark(data, lambda query: resumed_calls.append(query) or scores, **options)
    expected = evaluate_benchmark(data, lambda query: scores)
    assert resumed['per_shape'] == expected['per_shape']
    assert resumed_calls == [q.query for q in data.queries[6:]]
    final = evaluate_benchmark(data, lambda _: pytest.fail('Completed queries were rescored'), **options)
    assert final['per_shape'] == expected['per_shape']
    changed = replace(data, queries=(replace(data.queries[0], easy=frozenset(), hard=frozenset({0})), *data.queries[1:]))
    with pytest.raises(ValueError, match='changed'):
        evaluate_benchmark(changed, lambda _: scores, **options)
    with pytest.raises(ValueError, match='changed'):
        evaluate_benchmark(data, lambda _: scores, **dict(options, checkpoint_identity='different-model'))


def test_checkpoint_corruption_and_concurrent_writer(tmp_path):
    with BenchmarkCheckpoint(tmp_path, {'model': 1}) as store:
        store.save({'completed': 4})
        with pytest.raises(RuntimeError, match='already running'):
            with BenchmarkCheckpoint(tmp_path, {'model': 1}):
                pass
    path = tmp_path / 'state.json'
    value = json.loads(path.read_text())
    value['state']['completed'] = 5
    path.write_text(json.dumps(value))
    with BenchmarkCheckpoint(tmp_path, {'model': 1}) as store:
        with pytest.raises(ValueError, match='Corrupt'):
            store.load()


def test_relation_scheduling_preserves_query_selection_and_metrics(tmp_path):
    fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    data = load_benchmark(tmp_path, 'WikiTopicsQuery:art')
    n, nr = data.context.num_entities, data.context.num_relations
    raw = torch.randn(n, nr, n, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    reference = benchmark_model(TableModel(raw).eval(), data, query_order='published', query_batch_size=1, cache_bytes=0)
    optimized = benchmark_model(TableModel(raw).eval(), data, query_order='relation', query_batch_size=8)
    for shape in reference['per_shape']:
        assert optimized['per_shape'][shape] == pytest.approx(reference['per_shape'][shape])
    assert optimized['inference']['statistics']['raw_rows'] < reference['inference']['statistics']['raw_rows']


def test_prefetch_batches_unique_rows_and_does_not_change_scores():
    raw = torch.randn(5, 2, 5, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    engine = QueryAnswerer(TableModel(raw).eval(), row_batch_size=8)
    queries = [(0, (0,)), ((0, (0,)), (1, (1,))), (1, (1,))]
    info = engine.prefetch(queries)
    assert info['raw_rows'] == 2
    for query in queries:
        expected = QueryAnswerer(TableModel(raw).eval(), cache_bytes=0).predict(query)
        torch.testing.assert_close(engine.predict(query), expected, rtol=0, atol=0)
        assert engine.last_info['raw_rows'] == 0


def test_named_api_reuses_and_releases_cache():
    model = TableModel(torch.zeros(4, 2, 4, dtype=torch.float64)).eval()
    kge = wrapper(model)
    query = (next(iter(kge.entity_to_idx)), (next(iter(kge.relation_to_idx)),))
    first = kge.answer_multi_hop_query('1p', query, only_scores=True)
    second = kge.answer_multi_hop_query('1p', query, only_scores=True)
    torch.testing.assert_close(first, second)
    assert kge._query_engine.last_info['raw_rows'] == 0
    with torch.no_grad():
        model.table.add_(1.)
    third = kge.answer_multi_hop_query('1p', query, only_scores=True)
    assert kge._query_engine.last_info['raw_rows'] == 1 and not torch.equal(first, third)
    kge.clear_query_cache()
    kge.answer_multi_hop_query('1p', query, only_scores=True)
    assert kge._query_engine.last_info['raw_rows'] == 1


@pytest.mark.parametrize('samples', [1, 2, 3])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_flock_batching_preserves_seeded_walks_scores_and_rng(samples, device):
    model = Flock(dict(num_entities=4, num_relations=4, flock_dim=8, flock_walk_num=2,
                       flock_walk_len=8, flock_refinements=2, flock_seed=77)).set_graph(
                           torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]]), inverse_relations={0: 2, 1: 3}).to(device).eval()
    conditions = [(0, 0), (1, 2), (3, 3), (0, 0)]
    scorer = AtomicScorer(model, seed=9, samples=samples)
    cpu_rng = torch.random.get_rng_state().clone()
    expected = []
    with torch.no_grad():
        for h, r in conditions:
            model.seed = int(fingerprint([scorer.context.identity, 9, samples, h, r])[:15], 16)
            model.test_samples = samples
            expected.append(model.forward_k_vs_all(torch.tensor([[h, r]])))
    model.seed, model.test_samples = 77, 1
    model.query_batch_size = 4
    actual = scorer.rows(conditions)
    torch.testing.assert_close(actual, torch.cat(expected), atol=2e-6, rtol=2e-6)
    reverse = scorer.rows(list(reversed(conditions)))
    torch.testing.assert_close(actual, reverse.flip(0), atol=2e-6, rtol=2e-6)
    assert model.seed == 77 and model.test_samples == 1
    assert torch.equal(cpu_rng, torch.random.get_rng_state())


def test_trix_effective_batch_and_resume_preserve_caller(tmp_path):
    fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    data = load_benchmark(tmp_path, 'WikiTopicsQuery:art')
    model = TRIX(dict(num_entities=1, num_relations=1, trix_dim=8))
    options = dict(beam_size=2, row_batch_size=4, checkpoint_dir=tmp_path / 'trix', checkpoint_every=3)
    report = benchmark_model(model, data, **options)
    assert report['inference']['backend_batch_size'] == 4
    assert model.query_batch_size == 8 and model.training and model.graph_triples is None

    second = benchmark_model(model, data, **options)
    assert second['per_shape'] == report['per_shape']
    assert second['inference']['statistics'] == report['inference']['statistics']
    with pytest.raises(ValueError, match='changed'):
        benchmark_model(model, data, **dict(options, backend_batch_size=2))
    assert model.query_batch_size == 8 and model.training and model.graph_triples is None


def test_resume_rejects_changed_runtime_sampling(tmp_path):
    fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    data = load_benchmark(tmp_path, 'WikiTopicsQuery:art', query_types=['1p'])
    model = Flock(dict(num_entities=1, num_relations=1, flock_dim=8, flock_walk_num=2,
                       flock_walk_len=8, flock_refinements=2))
    options = dict(checkpoint_dir=tmp_path / 'flock')
    benchmark_model(model, data, **options)
    model.walk_num = 3
    with pytest.raises(ValueError, match='changed'):
        benchmark_model(model, data, **options)
