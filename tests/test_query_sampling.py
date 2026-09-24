"""Fixed stratified subsets and evaluation provenance."""

from collections import Counter
from dataclasses import replace

import pytest
import torch

from dicee.query_answering import BenchmarkQuery, QueryBenchmark, QueryContext, evaluate_benchmark
from dicee.query_answering.benchmark import _query_plan


@pytest.fixture
def data():
    queries = tuple(BenchmarkQuery(shape, (head, relations), frozenset({0}), frozenset({1}))
                    for shape, relations in [('1p', (0,)), ('2p', (0, 1))] for head in range(100))
    return QueryBenchmark('sample', 'transductive', 'test', QueryContext([], 100, 2), queries, tuple(range(100)))


def test_uniform_subsets_are_nested_balanced_and_order_independent(data):
    def select(n, seed=17, source=data, order='published'):
        return _query_plan(source, n, order, sampling='uniform', seed=seed)
    small, medium, large = (select(n) for n in (5, 20, 50))
    assert set(small) < set(medium) < set(large)
    assert Counter(q.shape for q in medium) == {'1p': 20, '2p': 20}
    assert medium == select(20, source=replace(data, queries=tuple(reversed(data.queries))))
    assert set(medium) == set(select(20, order='relation'))
    assert medium != select(20, seed=18)
    assert set(medium) != set(_query_plan(data, 20, 'published'))
    assert set(select(1000)) == set(data.queries)


def test_selection_never_depends_on_answers(data):
    changed = replace(data, queries=tuple(replace(q, easy=frozenset({2}), hard=frozenset({3, 4})) for q in data.queries))
    def selected(source):
        return [q.query for q in _query_plan(source, 20, 'relation', sampling='uniform', seed=4)]
    assert selected(data) == selected(changed)


def test_uniform_reporting_observer_and_resume_identity(data, tmp_path):
    recorded = []
    def observe(query, shape, metrics):
        recorded.append((query, shape, metrics))
        metrics['mrr'] = -1  # An observer cannot change reported metrics.
    options = dict(max_queries_per_shape=7, query_sampling='uniform', sampling_seed=41,
                   checkpoint_dir=tmp_path, checkpoint_identity='fixed', checkpoint_every=3, on_query=observe)
    scores = torch.arange(100, dtype=torch.float64)
    report = evaluate_benchmark(data, lambda query: scores, **options)
    assert report['queries'] == len(recorded) == 14
    assert report['protocol']['query_sampling'] == 'uniform'
    assert report['protocol']['sampling_seed'] == 41
    assert not report['protocol']['full_split']
    assert report['averages']['epfo']['mrr'] > 0
    assert [q for q, _, _ in recorded] == [q.query for q in _query_plan(data, 7, 'published', sampling='uniform', seed=41)]
    again = evaluate_benchmark(data, lambda _: pytest.fail('Rescored completed subset'), **options)
    assert again['per_shape'] == report['per_shape'] and len(recorded) == 14
    with pytest.raises(ValueError, match='changed'):
        evaluate_benchmark(data, lambda query: scores, **dict(options, sampling_seed=42))


@pytest.mark.parametrize('sampling,seed', [('random', 0), ('uniform', '1'), ('prefix', True)])
def test_invalid_sampling(data, sampling, seed):
    with pytest.raises(ValueError, match='sampling'):
        _query_plan(data, 5, 'published', sampling=sampling, seed=seed)
