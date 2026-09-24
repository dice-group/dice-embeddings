"""Negation supervision, adapter variants, and source-only model selection."""

import numpy as np
import pytest
import torch

from dicee.query_answering import AdapterQuery, AdapterTrainingData, QueryAnswerer, QueryContext, QueryScoreAdapter, fit_query_adapter, prepare_adapter_data
from dicee.query_answering._query import compile_query
from dicee.query_answering.context import state_fingerprint
from dicee.query_answering.training import _bank, _prepared_bank, _query_logs, filtered_softmax_loss
from tests.test_query_engine import TableModel


@pytest.fixture
def negation_data():
    source = QueryContext([(0, 0, 2), (0, 0, 3), (0, 0, 4), (1, 1, 2)], 6, 2)
    context = QueryContext([(0, 0, 2), (0, 0, 3)], 6, 2)
    query = ((0, (0,)), (1, (1, -2)))
    item = AdapterQuery(query, source.answers(compile_query(query)))
    return AdapterTrainingData('source', context, (item,))


def test_negation_masking_keeps_false_positives_as_negatives(negation_data):
    item = negation_data.train[0]
    assert item.answers == {3, 4}
    assert negation_data.context.answers(compile_query(item.query)) == {2, 3}
    model = TableModel(np.random.default_rng(12).normal(size=(6, 2, 6))).eval()
    adapter = QueryScoreAdapter('context_scores_v1', 1.)
    bank = _prepared_bank(adapter, _bank(model, negation_data, cache_dir=None, row_batch_size=2, seed=0, samples=None))
    logs = _query_logs(adapter, bank, item)
    actual = QueryAnswerer(model, context=negation_data.context, adapter=adapter).predict(item.query, return_log_scores=True)
    torch.testing.assert_close(logs, actual)
    loss = filtered_softmax_loss(logs, item.answers, {4})
    loss.backward()
    assert torch.isfinite(adapter.weights.grad).all()
    easier = logs.detach().clone()
    easier[2] -= 10
    assert filtered_softmax_loss(easier, item.answers, {4}) < loss
    result = fit_query_adapter(model, [negation_data], epochs=2)
    assert all(np.isfinite(row['loss']) for row in result.history)
    assert AdapterQuery(tuple(reversed(item.query)), item.answers).query == item.query


def test_observed_negation_has_finite_training_gradients():
    context = QueryContext([(0, 0, 2), (1, 1, 2)], 5, 2)
    item = AdapterQuery(((0, (0,)), (1, (1, -2))), {3})
    data = AdapterTrainingData('source', context, (item,))
    model = TableModel(np.zeros((5, 2, 5)))
    result = fit_query_adapter(model, [data], epochs=2)
    assert all(torch.isfinite(p).all() for p in result.adapter.parameters())
    scores = QueryAnswerer(model.eval(), context=context, adapter=result.adapter).predict(item.query, return_log_scores=True)
    assert scores[2] == -torch.inf


def test_negation_without_negative_mass_has_finite_training_gradients():
    context = QueryContext([(0, 0, 0), (1, 1, 0), (1, 1, 1)], 3, 2)
    item = AdapterQuery(((0, (0,)), (1, (1, -2))), {2})
    data = AdapterTrainingData('source', context, (item,))
    model = TableModel(np.zeros((3, 2, 3)))
    adapter = QueryScoreAdapter('context', 1.)
    bank = _prepared_bank(adapter, _bank(model, data, cache_dir=None, row_batch_size=2, seed=0, samples=None))
    logs = _query_logs(adapter, bank, item)
    assert torch.isneginf(logs[:2]).all() and torch.isfinite(logs[2])
    loss = filtered_softmax_loss(logs, item.answers, {2})
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(adapter.weights.grad, torch.zeros_like(adapter.weights))
    result = fit_query_adapter(model, [data], epochs=2)
    assert all(np.isfinite(row['loss']) for row in result.history)
    assert all(torch.isfinite(p).all() for p in result.adapter.parameters())


def test_generate_all_flat_shapes_and_reload(tmp_path):
    rng = np.random.default_rng(81)
    context = QueryContext([(h, r, t) for h in range(12) for r in range(3) for t in range(12)
                            if h != t and rng.uniform() < .3], 12, 6, ((0, 3), (1, 4), (2, 5)))
    data = prepare_adapter_data(context, shapes=('2i', '3i', '2in', '3in'),
                                train_per_shape=3, validation_per_shape=2, seed=123)
    assert {q.shape for q in data.train} == {'2i', '3i', '2in', '3in'}
    for q in (*data.train, *data.validation):
        assert q.answers == context.answers(compile_query(q.query))
        if 'n' in q.shape:
            assert data.context.answers(compile_query(q.query)) - q.answers
    data.save(tmp_path/'data.json')
    assert AdapterTrainingData.load(tmp_path/'data.json').to_dict() == data.to_dict()


def test_extend_training_preserves_mask_validation_and_prefixes():
    rng = np.random.default_rng(81)
    source = QueryContext([(h, r, t) for h in range(12) for r in range(3) for t in range(12)
                           if h != t and rng.uniform() < .3], 12, 6, ((0, 3), (1, 4), (2, 5)))
    args = dict(shapes=('2i', '3i', '2in', '3in'), validation_per_shape=2, seed=123)
    original = prepare_adapter_data(source, train_per_shape=3, **args)
    larger = prepare_adapter_data(source, train_per_shape=12, extend=original, **args)
    assert larger.context.identity == original.context.identity
    assert larger.validation == original.validation
    assert larger.train[:len(original.train)] == original.train
    assert len(larger.train) == 48
    assert not {q.query for q in larger.train} & {q.query for q in original.validation}
    for shape in args['shapes']:
        assert sum(q.shape == shape for q in larger.train) == 12
    for q in larger.train:
        assert q.answers == source.answers(compile_query(q.query))
    again = prepare_adapter_data(source, train_per_shape=12, extend=original, **args)
    assert again.to_dict() == larger.to_dict()
    with pytest.raises(ValueError, match='same source'):
        prepare_adapter_data(source, train_per_shape=12, extend=original, **dict(args, seed=124))
    with pytest.raises(ValueError, match='retain all'):
        prepare_adapter_data(source, train_per_shape=2, extend=original, **args)
    with pytest.raises(ValueError, match='retain all'):
        prepare_adapter_data(source, train_per_shape=12, extend=original, **dict(args, validation_per_shape=3))


@pytest.mark.parametrize('normalization', ['none', 'standard'])
@pytest.mark.parametrize('hidden_dim', [0, 8])
def test_variant_roundtrip_ordering_rng_and_gradients(normalization, hidden_dim, tmp_path):
    model = TableModel(np.zeros((5, 2, 5)))
    rng = torch.random.get_rng_state().clone()
    adapter = QueryScoreAdapter('context_scores_v1', 1., normalization=normalization, hidden_dim=hidden_dim,
                                bias_bound=8., metadata={'backbone_state_sha256': state_fingerprint(model)})
    assert torch.equal(torch.random.get_rng_state(), rng)
    with torch.no_grad():
        adapter.weights.fill_(.15)
    raw = torch.tensor([[1., -2., 4., 0., 2.], [0., 0., 0., 0., 0.]], dtype=torch.float64)
    observed, base = QueryContext([(0, 0, 4)], 5, 2).features([(0, 0), (1, 1)], device='cpu')
    scores = adapter(raw, observed, base)
    assert torch.isfinite(scores).all() and scores[0, 4] == 0
    assert torch.equal(scores[0, :4].argsort(), raw[0, :4].argsort())
    scores.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    adapter.save(tmp_path/'adapter.json')
    loaded = QueryScoreAdapter.load(tmp_path/'adapter.json', model=model)
    torch.testing.assert_close(scores, loaded(raw, observed, base), rtol=0, atol=0)
    assert loaded.configuration == adapter.configuration


def test_checkpoint_selection_and_heldout_source_isolation():
    rng = np.random.default_rng(81)
    source = QueryContext([(h, r, t) for h in range(8) for r in range(3) for t in range(8)
                           if h != t and rng.uniform() < .3], 8, 3)
    a = prepare_adapter_data(source, name='a', train_per_shape=3, validation_per_shape=2, seed=9)
    b = prepare_adapter_data(source, name='b', train_per_shape=3, validation_per_shape=2, seed=19)
    changed_queries = []
    for query in b.train:
        extra = min(set(range(8)) - query.answers)
        changed_queries.append(AdapterQuery(query.query, query.answers | {extra}))
    changed = AdapterTrainingData('b', b.context, tuple(changed_queries), b.validation)
    model = TableModel(rng.normal(size=(8, 3, 8)))
    args = dict(epochs=4, training_sources=['a'], validation_sources=['b'], validation_every=1, hidden_dim=8,
                early_stopping_patience=2)
    first = fit_query_adapter(model, [a, b], **args)
    second = fit_query_adapter(model, [a, changed], **args)
    assert first.adapter.metadata['training']['selected_epoch'] == max(first.history, key=lambda r: r['validation_mrr'])['epoch']
    for key, value in first.adapter.state_dict().items():
        torch.testing.assert_close(value, second.adapter.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('option,value', [('bias_bound', 8.), ('normalization', 'standard')])
def test_changed_adapter_configuration_invalidates_inference_cache(option, value):
    model = TableModel(np.random.default_rng(14).normal(size=(5, 2, 5))).eval()
    adapter = QueryScoreAdapter('global', weights=[[.2], [.3]])
    engine = QueryAnswerer(model, adapter=adapter)
    before = engine.predict((0, (0,)))
    setattr(adapter, option, value)
    after = engine.predict((0, (0,)))
    assert engine.last_info['raw_rows'] == 1 and not torch.equal(before, after)
