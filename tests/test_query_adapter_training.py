"""Source isolation, differentiable transforms, frozen fitting, and reproducibility."""

import json
import random

import numpy as np
import pytest
import torch

from dicee.query_answering import (
    AdapterQuery,
    AdapterTrainingData,
    QueryAnswerer,
    QueryContext,
    QueryScoreAdapter,
    fit_query_adapter,
    prepare_adapter_data,
)
from dicee.query_answering._query import combine, compile_query
from dicee.query_answering.context import state_fingerprint
from dicee.query_answering.training import filtered_softmax_loss
from tests.test_query_engine import TableModel, save_experiment


@pytest.fixture
def source():
    rng = np.random.default_rng(61)
    triples = [(h, r, t) for h in range(8) for r in (0, 2, 5) for t in range(8) if h != t and rng.uniform() < .3]
    return QueryContext(triples, 8, 6, ((0, 3), (2, 4), (5, 1)))


@pytest.fixture
def data(source):
    return prepare_adapter_data(source, name='fixture', train_per_shape=4, validation_per_shape=2, seed=9)


def test_masking_grounding_splits_and_json(source, data, tmp_path):
    graph = set(data.context.triples)
    inverse = {a: b for h, t in source.inverse_relations for a, b in ((h, t), (t, h))}
    assert graph < set(source.triples)
    for h, r, t in source.triples:
        assert ((h, r, t) in graph) == ((t, inverse[r], h) in graph)
    assert len(data.train) == 8 and len(data.validation) == 4
    assert not {q.query for q in data.train} & {q.query for q in data.validation}
    for query in (*data.train, *data.validation):
        assert query.answers == source.answers(compile_query(query.query))
        assert query.answers - data.context.answers(compile_query(query.query))
    state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    again = prepare_adapter_data(source, name='fixture', train_per_shape=4, validation_per_shape=2, seed=9)
    assert again.to_dict() == data.to_dict()
    assert state == random.getstate() and torch.equal(torch_state, torch.random.get_rng_state())
    path = tmp_path/'data.json'
    data.save(path)
    assert AdapterTrainingData.load(path).to_dict() == data.to_dict()
    with pytest.raises(ValueError, match='distinct'):
        duplicate = AdapterQuery(tuple(reversed(data.train[0].query)), data.train[0].answers)
        AdapterTrainingData('bad', data.context, data.train, (duplicate,))
    with pytest.raises(ValueError, match='Only generated'):
        prepare_adapter_data(QueryContext([(0, 0, 1), (1, 0, 2)], 3, 1), train_per_shape=1, validation_per_shape=0, max_attempts=20)


@pytest.mark.parametrize('mode', ['global', 'context', 'context_scores_v1'])
@pytest.mark.parametrize('gamma', [0., .5, 1.])
def test_transform_and_filtered_loss_gradients(mode, gamma):
    from torch.func import functional_call
    adapter = QueryScoreAdapter(mode, gamma)
    raw = torch.tensor([[1., -2., .5, .1], [-.4, .6, 1.2, -.2]], dtype=torch.float64)
    context = QueryContext([(0, 0, 0), (1, 1, 3)], 4, 2)
    observed, base = context.features([(0, 0), (1, 1)], device='cpu')
    weights = torch.full_like(adapter.weights, .15, requires_grad=True)

    def objective(w):
        logs = functional_call(adapter, {'weights': w}, (raw, observed, base))
        return filtered_softmax_loss(combine(list(logs), 'and'), {0, 1}, {1})

    assert torch.autograd.gradcheck(objective, (weights,), eps=1e-6, atol=2e-5)
    logs = torch.tensor([-.1, -.3, -1., -2.], dtype=torch.float64)
    expected = np.log(1 + np.exp(-1. + .3) + np.exp(-2. + .3))
    assert filtered_softmax_loss(logs, {0, 1}, {1}).item() == pytest.approx(expected)


def test_fit_frozen_reproducible_cache_and_roundtrip(data, tmp_path):
    raw = np.random.default_rng(20).normal(size=(8, 6, 8))
    model = TableModel(raw)
    before = state_fingerprint(model)
    existing_grad = torch.ones_like(model.table)
    model.table.grad = existing_grad.clone()
    rng_state = torch.random.get_rng_state().clone()
    kwargs = dict(epochs=8, batch_size=4, seed=123, cache_dir=tmp_path/'banks')
    result = fit_query_adapter(model, [data], **kwargs)
    assert state_fingerprint(model) == before and model.training and model.table.requires_grad
    torch.testing.assert_close(model.table.grad, existing_grad)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert not torch.equal(result.adapter.weights, torch.zeros_like(result.adapter.weights))
    assert result.history[-1]['loss'] < result.history[0]['loss']
    calls = len(model.calls)
    second = fit_query_adapter(model, [data], **kwargs)
    assert len(model.calls) == calls
    torch.testing.assert_close(result.adapter.weights, second.adapter.weights, rtol=0, atol=0)
    assert result.validation == second.validation
    global_result = fit_query_adapter(model, [data], feature_mode='global', **kwargs)
    assert global_result.adapter.weights.shape == (2, 1) and len(model.calls) == calls
    assert set(result.validation) == {'sigmoid', 'observed', 'fitted'}
    assert set(result.validation['fitted']) == {'fixture/2i', 'fixture/3i'}
    path = tmp_path/'adapter.json'
    result.adapter.save(path)
    loaded = QueryScoreAdapter.load(path, model=model)
    query = data.validation[0].query
    a = QueryAnswerer(model, context=data.context, adapter=result.adapter).predict(query)
    b = QueryAnswerer(model, context=data.context, adapter=loaded).predict(query)
    torch.testing.assert_close(a, b, rtol=0, atol=0)

    cache = next((tmp_path/'banks').glob('*.pt'))
    bank = torch.load(cache, weights_only=True)
    bank['rows'][0, 0] += 1
    torch.save(bank, cache)
    with pytest.raises(ValueError, match='Corrupt'):
        fit_query_adapter(model, [data], **kwargs)


def test_validation_answers_do_not_change_fitted_parameters(data):
    raw = np.random.default_rng(21).normal(size=(8, 6, 8))
    model = TableModel(raw)
    changed = []
    for query in data.validation:
        extra = min(set(range(8))-query.answers)
        changed.append(AdapterQuery(query.query, query.answers | {extra}))
    other = AdapterTrainingData(data.name, data.context, data.train, tuple(changed))
    first = fit_query_adapter(model, [data], epochs=2, seed=8)
    second = fit_query_adapter(model, [other], epochs=2, seed=8)
    torch.testing.assert_close(first.adapter.weights, second.adapter.weights, rtol=0, atol=0)


@pytest.mark.parametrize('name', ['ULTRA', 'TRIX', 'Flock'])
@pytest.mark.parametrize('attached', [True, False])
def test_fit_graph_models_restore_weights_graph_and_mode(name, attached, data, tmp_path):
    from dicee.models import TRIX, ULTRA, Flock
    cls = {'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[name]
    model = cls(dict(num_entities=3, num_relations=1, ultra_dim=8, ultra_num_layers=2, trix_dim=8,
                     flock_dim=8, flock_walk_num=2, flock_walk_len=8, flock_refinements=2, flock_seed=17))
    if attached:
        model.set_graph(torch.tensor([[0, 0, 1], [1, 0, 2]]))
    model.train()
    before, weights = QueryContext.from_model(model) if attached else None, state_fingerprint(model)
    buffers = dict(model._buffers)
    result = fit_query_adapter(model, [data], epochs=1, cache_dir=tmp_path, row_batch_size=4)
    assert model.training
    if attached:
        assert QueryContext.from_model(model) == before
    else:
        assert all(value is buffers[key] for key, value in model._buffers.items())
        if name == 'Flock':
            assert model._walk_graph is None
    assert state_fingerprint(model) == weights
    assert all(p.grad is None for p in model.parameters())
    assert result.adapter.feature_mode == ('context_scores_v1' if name == 'ULTRA' else 'context')


def test_graph_restored_after_scoring_failure(data, monkeypatch):
    from dicee.models import ULTRA
    model = ULTRA(dict(num_entities=3, num_relations=1, ultra_dim=8, ultra_num_layers=2))
    model.set_graph(torch.tensor([[0, 0, 1]])).eval()
    before = QueryContext.from_model(model)
    def fail(_):
        raise RuntimeError('test scorer failure')
    monkeypatch.setattr(model, 'forward_k_vs_all', fail)
    with pytest.raises(RuntimeError, match='scorer failure'):
        fit_query_adapter(model, [data], epochs=1)
    assert QueryContext.from_model(model) == before and not model.training


def test_training_cli(tmp_path, source, monkeypatch, capsys):
    from dicee.query_answering.__main__ import main
    path, output = tmp_path/'source.json', tmp_path/'prepared.json'
    path.write_text(json.dumps(source.to_dict()))
    monkeypatch.setattr('sys.argv', ['query_answering', 'prepare', '--source', str(path), '--output', str(output),
                                  '--train-per-shape', '2', '--validation-per-shape', '1'])
    main()
    prepared = AdapterTrainingData.load(output)
    assert len(prepared.train) == 4 and len(prepared.validation) == 2
    assert 'Saved' in capsys.readouterr().out


def test_fit_cli_loads_experiment_and_compares_global(data, tmp_path, monkeypatch):
    from dicee.models import DistMult
    from dicee.query_answering.__main__ import main
    settings = dict(model='DistMult', num_entities=8, num_relations=6, embedding_dim=8)
    model = DistMult(settings)
    experiment, output, prepared = tmp_path/'experiment', tmp_path/'fit', tmp_path/'data.json'
    save_experiment(model, settings, experiment)
    data.save(prepared)
    monkeypatch.setattr('sys.argv', ['query_answering', 'fit', '--experiment', str(experiment), '--output', str(output),
                                   '--data', str(prepared), '--epochs', '2', '--compare-global'])
    main()
    report = json.loads((output/'report.json').read_text())
    assert set(report) == {'context', 'global'}
    assert report['context']['metadata']['score_banks'] == report['global']['metadata']['score_banks']
    for mode in ('context', 'global'):
        QueryScoreAdapter.load(output/f'{mode}.json', model=model)
