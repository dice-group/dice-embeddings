"""Multi-hop supervision matches inference and preserves frozen score provenance."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dicee.query_answering import AdapterQuery, AdapterTrainingData, QueryAnswerer, QueryContext, QueryScoreAdapter, fit_query_adapter, prepare_adapter_data
from dicee.query_answering._query import ULTRAQUERY_SHAPES, compile_query
from dicee.query_answering.training import _bank, _query_logs
from dicee.scripts.benchmark_query_adapters import study_variants
from tests.test_query_engine import TableModel, programs


@pytest.mark.parametrize('mix', [0., 1.])
def test_all_shapes_match_inference_and_reuse_disk_rows(tmp_path, mix):
    raw = np.random.default_rng(17).normal(size=(4, 2, 4))
    context = QueryContext(((0, 0, 1), (1, 1, 2), (2, 0, 3)), 4, 2)
    queries = {s: AdapterQuery(q, {0}) for s, (q, _) in programs(raw, 2).items() if s in ULTRAQUERY_SHAPES}
    assert all(q.shape == s for s, q in queries.items())
    model = TableModel(raw)
    data = SimpleNamespace(context=context, train=tuple(queries.values()), validation=())
    adapter = QueryScoreAdapter('context_scores_v1', mix, weights=np.random.default_rng(1).normal(0, .1, (2, 8)))
    bank = _bank(model, data, cache_dir=tmp_path, row_batch_size=2, seed=0, samples=None)
    bank['beam_size'] = 2
    engine = QueryAnswerer(model, context=context, adapter=adapter, row_batch_size=2)
    for query in queries.values():
        logs = _query_logs(adapter, bank, query)
        expected = engine.predict(query.query, beam_size=2, return_log_scores=True)
        torch.testing.assert_close(logs, expected)
        adapter.zero_grad()
        (logs.exp() * torch.arange(1., 5.)).sum().backward()
        assert torch.isfinite(adapter.weights.grad).all()
    bank['provider'].close()
    calls = len(model.calls)
    restored = _bank(model, data, cache_dir=tmp_path, row_batch_size=2, seed=0, samples=None)
    restored['beam_size'] = 2
    for query in queries.values():
        _query_logs(adapter, restored, query)
    assert len(model.calls) == calls
    assert restored['provider'].cache_used <= restored['provider'].cache_bytes
    restored['provider'].connection.execute("UPDATE rows SET sha256='corrupt'")
    restored['provider'].connection.commit()
    restored['provider'].cache.clear()
    with pytest.raises(ValueError, match='Corrupt'):
        _query_logs(adapter, restored, queries['1p'])
    restored['provider'].close()


@pytest.mark.parametrize('shape', ['3p', 'up', 'pni'])
def test_multihop_gradient_matches_finite_differences(tmp_path, shape):
    model = TableModel(np.random.default_rng(22).normal(size=(4, 2, 4)))
    context = QueryContext((), 4, 2)
    query = AdapterQuery(programs(model.table.detach().numpy(), 2)[shape][0], {0})
    data = SimpleNamespace(context=context, train=(query,), validation=())
    bank = _bank(model, data, cache_dir=tmp_path, row_batch_size=2, seed=0, samples=None)
    bank['beam_size'] = 2
    adapter = QueryScoreAdapter('global', weights=[[.2], [-.1]])
    def objective():
        return (_query_logs(adapter, bank, query).exp() * torch.arange(1., 5.)).sum()
    objective().backward()
    grad = adapter.weights.grad.clone()
    for i in range(2):
        with torch.no_grad():
            adapter.weights[i, 0] += 1e-6
            plus = objective().item()
            adapter.weights[i, 0] -= 2e-6
            minus = objective().item()
            adapter.weights[i, 0] += 1e-6
        assert grad[i, 0].item() == pytest.approx((plus - minus) / 2e-6, rel=1e-4, abs=1e-6)
    bank['provider'].close()


def test_all_shape_generation_and_fitting(tmp_path):
    rng = np.random.default_rng(12)
    context = QueryContext(tuple((h, r, t) for h in range(20) for r in range(3) for t in range(20)
                                 if h != t and rng.uniform() < .13), 20, 6, ((0, 3), (1, 4), (2, 5)))
    data = prepare_adapter_data(context, shapes=ULTRAQUERY_SHAPES, train_per_shape=2,
                                validation_per_shape=1, max_attempts=100000, seed=9)
    assert {q.shape for q in data.train} == set(ULTRAQUERY_SHAPES)
    for q in (*data.train, *data.validation):
        assert q.answers == context.answers(compile_query(q.query))
    data.save(tmp_path/'data.json')
    assert AdapterTrainingData.load(tmp_path/'data.json').to_dict() == data.to_dict()
    model = TableModel(rng.normal(size=(20, 6, 20)))
    result = fit_query_adapter(model, [data], epochs=2, validation_every=1, beam_size=4,
                               cache_dir=tmp_path/'banks')
    assert result.adapter.metadata['training']['training_queries'] == 28
    assert len(result.validation['fitted']) == 14
    assert torch.isfinite(result.adapter.weights).all()


@pytest.mark.parametrize('bound', [2., 4., 8., None])
def test_scale_bounds_identity_gradient_and_roundtrip(tmp_path, bound):
    adapter = QueryScoreAdapter('global', scale_bound=bound)
    raw = torch.tensor([[1., -2.]], dtype=torch.float64)
    values = adapter(raw)
    torch.testing.assert_close(values, torch.nn.functional.logsigmoid(raw))
    values[0, 0].backward()
    assert adapter.weights.grad[0, 0].item() == pytest.approx(np.log(2) * (1 - torch.sigmoid(raw[0, 0]).item()))
    with torch.no_grad():
        adapter.weights[0, 0] = 5.
    logs = adapter(raw)
    effective = torch.logit(logs.exp())[0, 0]
    if bound is not None:
        assert .5 < effective < bound
    else:
        assert effective > 8
    model = TableModel(np.zeros((2, 1, 2)))
    from dicee.query_answering.context import state_fingerprint
    adapter.metadata['backbone_state_sha256'] = state_fingerprint(model)
    adapter.save(tmp_path/'adapter.json')
    loaded = QueryScoreAdapter.load(tmp_path/'adapter.json', model=model)
    assert loaded.scale_bound == bound
    torch.testing.assert_close(loaded(raw), logs)


def test_studies_match_query_budgets_and_scale_control():
    coverage, scale = study_variants('query-types'), study_variants('scale')
    assert [len(v['shapes']) for v in coverage.values()] == [2, 4, 10, 14]
    assert all(len(v['shapes']) * v['train_per_shape'] * 3 == 840 for v in coverage.values())
    assert coverage['types_2'] == scale['types_2']
    assert [v['scale_bound'] for v in scale.values()] == [2., 4., 8., None]


def test_row_cache_eviction_and_context_isolation(tmp_path):
    from dicee.query_answering._training_rows import TrainingRows
    model = TableModel(np.random.default_rng(31).normal(size=(4, 2, 4)))
    a, b = QueryContext((), 4, 2), QueryContext(((0, 0, 1),), 4, 2)
    options = dict(cache_dir=tmp_path, row_batch_size=1, seed=0, samples=None, cache_bytes=40, disk_bytes=40)
    first, second = TrainingRows(model, a, **options), TrainingRows(model, b, **options)
    assert first.identity != second.identity
    adapter = QueryScoreAdapter('global')
    for _ in range(2):
        raw, _, _ = first.prepared(adapter, [(0, 0), (1, 1), (2, 0)])
        torch.testing.assert_close(raw, model.table.detach()[[0, 1, 2], [0, 1, 0]])
        assert first.cache_used <= first.cache_bytes and first.disk_used <= first.disk_limit
    first.close()
    second.close()


@pytest.mark.parametrize('name', ['ULTRA', 'TRIX'])
def test_multihop_fit_preserves_graph_backbones(tmp_path, name):
    from dicee.models import TRIX, ULTRA
    from dicee.query_answering.context import state_fingerprint
    cls = {'ULTRA': ULTRA, 'TRIX': TRIX}[name]
    model = cls(dict(num_entities=3, num_relations=1, ultra_dim=8, ultra_num_layers=2, trix_dim=8))
    model.set_graph(torch.tensor([[0, 0, 1], [1, 0, 2]])).train()
    before, weights = QueryContext.from_model(model), state_fingerprint(model)
    data = AdapterTrainingData('toy', QueryContext(((0, 0, 1),), 3, 1),
                               (AdapterQuery((0, (0, 0)), {2}),))
    result = fit_query_adapter(model, [data], epochs=2, beam_size=2, cache_dir=tmp_path)
    assert model.training and QueryContext.from_model(model) == before
    assert state_fingerprint(model) == weights and all(p.grad is None for p in model.parameters())
    assert torch.isfinite(result.adapter.weights).all()
