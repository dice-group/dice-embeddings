"""Exact projections, bounded batches and unchanged query semantics."""

import numpy as np
import pytest
import torch

from dicee.query_answering import QueryAnswerer, QueryContext, benchmark_model, load_benchmark
from dicee.query_answering._query import QUERY_SHAPES
from tests.test_query_benchmark import fixture_dataset
from tests.test_query_engine import TableModel, named, programs, wrapper


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('shape', QUERY_SHAPES)
@pytest.mark.parametrize('tnorm', ['prod', 'min'])
@pytest.mark.parametrize('batch', [1, 3])
def test_qto_matches_independent_enumeration(shape, tnorm, batch):
    raw = np.random.default_rng(71).normal(size=(4, 2, 4))
    query, expected = programs(1 / (1 + np.exp(-raw)), 4, tnorm)[shape]
    model = TableModel(raw).eval()
    engine = QueryAnswerer(model, row_batch_size=batch, cache_bytes=32)
    actual = engine.predict(query, executor='qto', beam_size=1, tnorm=tnorm)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-15, rtol=2e-14)
    assert max(map(len, model.calls)) <= batch
    assert not engine.last_info['pruned'] and not engine.last_info['negated_pruning']
    assert engine.last_info['search'] == 'exact' and engine.last_info['cache_bytes'] <= 32


def test_qto_recovers_path_outside_beam_and_preserves_negation_scope():
    p = np.full((4, 2, 4), .01)
    p[0, 0] = [.9, .8, .1, .1]
    p[0, 1, 3], p[1, 1, 3] = .01, .99
    model = TableModel(np.log(p / (1 - p))).eval()
    engine = QueryAnswerer(model, row_batch_size=1)
    query = (0, (0, 1))
    beam = engine.predict(query, beam_size=1)
    exact = engine.predict(query, executor='qto', beam_size=1)
    assert beam[3] == pytest.approx(.009) and exact[3] == pytest.approx(.792)
    negated = engine.predict((0, (0, 1, -2)), executor='qto')
    torch.testing.assert_close(negated, 1 - exact)
    edge_negated = max(p[0, 0, x] * (1 - p[x, 1, 3]) for x in range(4))
    assert negated[3] != pytest.approx(edge_negated)
    kge = wrapper(model)
    torch.testing.assert_close(kge.answer_multi_hop_query('2p', named(query), k=1,
                               executor='qto', only_scores=True), exact)


def test_qto_bound_skips_only_heads_that_cannot_improve_any_tail():
    p = np.full((4, 2, 4), .01)
    p[0, 0], p[0, 1] = [.9, .1, .05, .01], .9
    engine = QueryAnswerer(TableModel(np.log(p / (1 - p))).eval(), row_batch_size=1)
    actual = engine.predict((0, (0, 1)), executor='qto')
    assert engine.last_info['bound_skipped'] == 3
    assert engine.last_info['raw_rows'] == 2
    np.testing.assert_allclose(actual.numpy(), .81)


@pytest.mark.parametrize('shape', QUERY_SHAPES)
def test_qto_observed_facts_and_zero_prefixes(shape):
    raw = np.full((4, 2, 4), -1000.)
    raw[0, 0, 0] = 1000.
    facts = [(0, 0, 1), (0, 0, 2), (2, 1, 3)]
    context = QueryContext(facts, 4, 2)
    p = np.zeros_like(raw)
    p[0, 0, 0] = 1
    for h, r, t in facts:
        p[h, r, t] = 1
    query, expected = programs(p, 4)[shape]
    engine = QueryAnswerer(TableModel(raw).eval(), context=context, observed_mix=1, row_batch_size=2)
    np.testing.assert_allclose(engine.predict(query, executor='qto').numpy(), expected, atol=1e-14)


@pytest.mark.parametrize('name', ['ULTRA', 'TRIX', 'Flock'])
def test_qto_graph_backends_match_full_beam(name):
    from dicee import models
    settings = dict(num_entities=4, num_relations=2, ultra_dim=8, ultra_num_layers=2,
                    trix_dim=8, flock_dim=8, flock_walk_num=2, flock_walk_len=4, flock_refinements=1)
    model = getattr(models, name)(settings).set_graph(torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]])).eval()
    engine = QueryAnswerer(model, row_batch_size=2, observed_mix=1)
    for query in [(0, (0, 1, 0)), ((0, (0, 1, -2)), (1, (1,)))]:
        expected = engine.predict(query, beam_size=4, return_log_scores=True)
        engine.clear_cache()
        actual = engine.predict(query, executor='qto', return_log_scores=True)
        torch.testing.assert_close(actual, expected)


def test_qto_metadata_resume_and_executor_validation(tmp_path):
    fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    data = load_benchmark(tmp_path, 'WikiTopicsQuery:art')
    n, nr = data.context.num_entities, data.context.num_relations
    model = TableModel(torch.randn(n, nr, n)).eval()
    options = dict(checkpoint_dir=tmp_path / 'qto', checkpoint_every=3, executor='qto')
    report = benchmark_model(model, data, **options)
    assert report['inference']['method'] == 'qto-exact'
    assert report['inference']['executor'] == 'qto' and report['inference']['beam_size'] is None
    assert report['inference']['statistics']['pruned'] == 0
    resumed = benchmark_model(model, data, **dict(options, beam_size=1))
    assert resumed['per_shape'] == report['per_shape']
    with pytest.raises(ValueError, match='changed'):
        benchmark_model(model, data, **dict(options, executor='cqd'))
    with pytest.raises(ValueError, match='executor'):
        QueryAnswerer(model).predict((0, (0,)), executor='unknown')
    with pytest.raises(ValueError, match='memberships'):
        QueryAnswerer(model).predict((0, (0,)), executor='qto', use_logits=True)
