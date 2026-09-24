"""Independent fuzzy-logic examples and common KGE/KGFM query contracts."""

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from dicee import KGE
from dicee.query_answering import QueryAnswerer, QueryContext, QueryScoreAdapter
from dicee.query_answering._query import QUERY_SHAPES, compile_query, index_query
from dicee.query_answering.adapter import score_features
from dicee.query_answering.context import state_fingerprint


class TableModel(nn.Module):
    name = 'Table'

    def __init__(self, raw):
        super().__init__()
        self.table = nn.Parameter(torch.as_tensor(raw, dtype=torch.float64).clone())
        self.num_entities, self.num_relations = self.table.shape[:2]
        self.calls = []

    def forward_k_vs_all(self, pairs):
        self.calls.append(pairs.detach().cpu().tolist())
        return self.table[pairs[:, 0], pairs[:, 1]]


def programs(m, beam, tnorm='prod'):
    """Literal formulas, independent of the parser and recursive evaluator."""
    n = len(m)
    a, b, c = (0, (0,)), (1, (1,)), (2, (0,))
    pa, pb, pc = m[0, 0], m[1, 1], m[2, 0]
    conjunction = np.multiply if tnorm == 'prod' else np.minimum
    union = (lambda x, y: x + y - x*y) if tnorm == 'prod' else np.maximum

    def project(prefix, r):
        heads = sorted(range(n), key=lambda h: (-prefix[h], h))[:beam]
        return np.array([max(conjunction(prefix[h], m[h, r, t]) for h in heads) for t in range(n)])

    two = project(pa, 1)
    intersection, disjunction = (a, b), (a, b, (-1,))
    return {
        '1p': (a, pa), '2p': ((0, (0, 1)), two), '3p': ((0, (0, 1, 0)), project(two, 0)),
        '2i': (intersection, conjunction(pa, pb)), '3i': ((a, b, c), conjunction(conjunction(pa, pb), pc)),
        'pi': (((0, (0, 1)), b), conjunction(two, pb)),
        'ip': ((intersection, (1,)), project(conjunction(pa, pb), 1)),
        '2u': (disjunction, union(pa, pb)), 'up': ((disjunction, (1,)), project(union(pa, pb), 1)),
        '2in': ((a, (1, (1, -2))), conjunction(pa, 1-pb)),
        '3in': ((a, c, (1, (1, -2))), conjunction(conjunction(pa, pc), 1-pb)),
        'inp': (((a, (1, (1, -2))), (1,)), project(conjunction(pa, 1-pb), 1)),
        'pin': (((0, (0, 1)), (1, (1, -2))), conjunction(two, 1-pb)),
        'pni': (((0, (0, 1, -2)), b), conjunction(1-two, pb)),
        '4p': ((0, (0, 1, 0, 1)), project(project(two, 0), 1)),
        '4i': ((a, b, c, (3, (1,))), conjunction(conjunction(conjunction(pa, pb), pc), m[3, 1])),
    }


@pytest.fixture
def raw():
    return np.random.default_rng(17).normal(size=(4, 2, 4))


@pytest.mark.parametrize('shape', QUERY_SHAPES)
@pytest.mark.parametrize('beam', [1, 2, 4])
@pytest.mark.parametrize('tnorm', ['prod', 'min'])
def test_all_shapes_against_literal_formulas(raw, shape, beam, tnorm):
    query, expected = programs(1/(1+np.exp(-raw)), beam, tnorm)[shape]
    model = TableModel(raw)
    result = QueryAnswerer(model).predict(query, beam_size=beam, tnorm=tnorm)
    np.testing.assert_allclose(result.numpy(), expected, atol=2e-15, rtol=2e-14)
    assert model.training  # prediction restores its caller's mode
    assert model.table.grad is None


def named(query):
    if isinstance(query, tuple):
        return tuple(named(x) for x in query)
    return 'not' if query == -2 else 'union' if query == -1 else str(query)


def wrapper(model):
    kge = object.__new__(KGE)
    kge.model = model
    # Insertion order deliberately differs from ID order.
    kge.entity_to_idx = {str(i): i for i in reversed(range(model.num_entities))}
    kge.relation_to_idx = {str(i): i for i in range(model.num_relations)}
    return kge


@pytest.mark.parametrize('shape', QUERY_SHAPES)
def test_named_api_limits_batches_and_numeric_ids(raw, shape):
    query, expected = programs(1/(1+np.exp(-raw)), 2)[shape]
    kge = wrapper(TableModel(raw))
    kwargs = dict(query_type=shape, beam_size=2, k=1)
    result = kge.answer_multi_hop_query(query=named(query), **kwargs)
    assert len(result) == 1 and result[0][0] == str(int(np.argmax(expected)))
    scores = kge.answer_multi_hop_query(query=named(query), only_scores=True, **kwargs)
    np.testing.assert_allclose(scores.numpy(), expected, atol=2e-15, rtol=2e-14)
    batch = kge.answer_multi_hop_query(queries=[named(query)] * 2, **kwargs)
    assert batch == [result, result]
    assert kge.answer_multi_hop_query(query=named(query), query_type=shape, k=0) == []


def test_singleton_ties_invalid_queries_and_logits(raw):
    kge = wrapper(TableModel(np.zeros((1, 1, 1))))
    assert kge.answer_multi_hop_query('2p', ('0', ('0', '0')))[0][0] == '0'
    engine = QueryAnswerer(TableModel(np.zeros_like(raw)))
    scores = engine.predict((0, (0, 1)), beam_size=1)
    assert torch.equal(scores, torch.full_like(scores, .25))
    assert engine.scorer.model.calls[1] == [[0, 1]]
    with pytest.raises(ValueError, match='Anchor'):
        engine.predict((4, (0,)))
    with pytest.raises(ValueError, match='Relation'):
        engine.predict((0, (2,)))
    with pytest.raises(ValueError, match='lambda'):
        engine.predict((0, (0, -2)), neg_norm='yager')
    with pytest.raises(ValueError, match='beam_size'):
        engine.predict((0, (0,)), beam_size=0)
    with pytest.raises(ValueError, match='marker'):
        index_query('2in', (('0', ('0',)), ('1', ('1', 'wrong'))), {'0': 0, '1': 1}, {'0': 0, '1': 1})
    raw_engine = QueryAnswerer(TableModel(raw))
    actual = raw_engine.predict(((0, (0,)), (1, (1,))), use_logits=True)
    np.testing.assert_allclose(actual.numpy(), raw[0, 0] * raw[1, 1])
    with pytest.raises(ValueError, match='Legacy'):
        QueryAnswerer(TableModel(raw), adapter=QueryScoreAdapter('global')).predict((0, (0,)), use_logits=True)


@pytest.mark.parametrize('shape', ['2u', 'up'])
def test_named_unions_accept_historical_marker_omission(raw, shape):
    query = (('0', ('0',)), ('1', ('1',)))
    marked = (*query, ('union',))
    if shape == 'up':
        query, marked = (query, ('0',)), (marked, ('0',))
    kge = wrapper(TableModel(raw))
    torch.testing.assert_close(kge.answer_multi_hop_query(shape, query, only_scores=True),
                               kge.answer_multi_hop_query(shape, marked, only_scores=True))


@pytest.mark.parametrize('norm,parameter', [('standard', 0.), ('sugeno', .5), ('yager', 2.)])
def test_negation_norms(raw, norm, parameter):
    p = 1 / (1 + np.exp(-raw[0, 0]))
    expected = 1-p if norm == 'standard' else (1-p)/(1+parameter*p) if norm == 'sugeno' else (1-p**parameter)**(1/parameter)
    actual = QueryAnswerer(TableModel(raw)).predict((0, (0, -2)), neg_norm=norm, lambda_=parameter)
    np.testing.assert_allclose(actual.numpy(), expected)


def test_observed_proofs_survive_pruning_and_do_not_shortcut_negation():
    raw = np.full((4, 2, 4), -8.)
    raw[0, 0, 0] = 8
    context = QueryContext([(0, 0, 1), (0, 0, 2), (2, 1, 3)], 4, 2)
    engine = QueryAnswerer(TableModel(raw), context=context, observed_mix=1)
    result = engine.predict((0, (0, 1)), beam_size=1)
    # Stable top-1 visits entity 1; the proof via entity 2 is restored separately.
    assert result[3] == 1 and engine.last_info['pruned']
    negated = engine.predict((0, (0, 1, -2)), beam_size=1)
    assert negated[3] > .99 and engine.last_info['negated_pruning']
    exact = engine.predict((0, (0, 1, -2)), beam_size=4)
    assert exact[3] == 0


def test_cache_bounded_reused_and_invalidated(raw):
    model = TableModel(raw)
    engine = QueryAnswerer(model, cache_bytes=4*8)
    a = (0, (0,))
    before = engine.predict((a, a))
    assert engine.last_info['raw_rows'] == 1 and engine.last_info['cache_hits'] == 1
    assert engine.last_info['cache_bytes'] <= 32
    with torch.no_grad():
        model.table.add_(1)
    after = engine.predict((a, a))
    assert (after > before).all()
    engine.predict((0, (0, 1)), beam_size=4)
    assert engine.last_info['cache_bytes'] <= 32


@pytest.mark.parametrize('name', ['DistMult', 'ULTRA', 'TRIX', 'Flock'])
def test_saved_experiment_query_reload(name, tmp_path):
    from dicee.models import TRIX, ULTRA, DistMult, Flock
    cls = {'DistMult': DistMult, 'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[name]
    settings = dict(model=name, num_entities=4, num_relations=2, embedding_dim=8,
                    ultra_dim=8, ultra_num_layers=2, trix_dim=8, flock_dim=8,
                    flock_walk_num=2, flock_walk_len=8, flock_refinements=2, flock_seed=17)
    model = cls(settings)
    if name != 'DistMult':
        model.set_graph(torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]]))
    save_experiment(model, settings, tmp_path)
    restored = KGE(path=str(tmp_path))
    actual = restored.answer_multi_hop_query('3p', ('0', ('0', '1', '0')), k=2, only_scores=True)
    expected = QueryAnswerer(model).predict((0, (0, 1, 0)), beam_size=2)
    torch.testing.assert_close(actual, expected)


def save_experiment(model, settings, path):
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, 'save_graph'):
        model.save_graph(path/model.graph_filename)
    torch.save(model.state_dict(), path/'model.pt')
    (path/'configuration.json').write_text(json.dumps(settings))
    (path/'report.json').write_text(json.dumps(dict(num_entities=model.num_entities, num_relations=model.num_relations)))
    for name, count in [('entity', model.num_entities), ('relation', model.num_relations)]:
        (path/f'{name}_to_idx.csv').write_text(f',{name}\n' + ''.join(f'{i},{i}\n' for i in range(count)))


def test_generator_and_context_truth_against_boolean_oracle(raw):
    from dicee.query_generator import QueryGenerator
    facts = [(h, r, t) for h, r, t in np.ndindex(raw.shape) if raw[h, r, t] > 0]
    context = QueryContext(facts, 4, 2)
    generator = QueryGenerator.from_context(context)
    for query, expected in programs((raw > 0).astype(float), 4).values():
        answers = set(np.flatnonzero(expected).tolist())
        assert context.answers(compile_query(query)) == answers
        assert generator.achieve_answer(generator.tuple2list(query), generator.ent_in, generator.ent_out) == answers


def test_mixed_module_modes_and_scoring_failure_are_restored(raw, monkeypatch):
    model = TableModel(raw)
    model.add_module('dropout', nn.Dropout())
    model.train()
    model.dropout.eval()
    engine = QueryAnswerer(model)
    engine.predict((0, (0,)))
    assert model.training and not model.dropout.training

    def fail(_):
        raise RuntimeError('scoring failed')
    monkeypatch.setattr(model, 'forward_k_vs_all', fail)
    with pytest.raises(RuntimeError, match='scoring failed'):
        engine.predict((0, (0,)))
    assert model.training and not model.dropout.training


@pytest.mark.parametrize('mode', ['global', 'context', 'context_scores_v1'])
@pytest.mark.parametrize('gamma', [0., .4, 1.])
def test_adapter_against_independent_numpy_formulas(raw, mode, gamma):
    context = QueryContext([(0, 0, 1), (0, 0, 2), (1, 1, 3)], 4, 2)
    conditions = [(0, 0), (1, 1)]
    z = np.stack([raw[h, r] for h, r in conditions])
    observed = np.zeros_like(z, dtype=bool)
    observed[0, [1, 2]], observed[1, 3] = True, True
    base = np.array([[1, np.log1p(2)/np.log1p(4), np.log1p(2)/np.log1p(8), np.log1p(2)/np.log1p(16)],
                     [1, np.log1p(1)/np.log1p(4), np.log1p(1)/np.log1p(8), np.log1p(1)/np.log1p(16)]])
    features = base[:, :1] if mode == 'global' else base
    if mode == 'context_scores_v1':
        prob = np.exp(z - z.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        top = np.sort(z, axis=1)[:, -2:]
        features = np.column_stack((base, np.tanh(z.mean(1)/4), -(prob*np.log(prob)).sum(1)/np.log(4),
                                    np.tanh((top[:, 1]-top[:, 0])/4),
                                    np.tanh(((z*observed).sum(1)/observed.sum(1)-z.mean(1))/4)))
    weights = np.random.default_rng(33).normal(size=(2, features.shape[1]))
    logits = 2**np.tanh(features @ weights[0])[:, None]*z + 4*np.tanh(features @ weights[1])[:, None]
    expected = 1/(1+np.exp(-logits))
    expected[observed] = gamma + (1-gamma)*expected[observed]
    obs, actual_base = context.features(conditions, device='cpu')
    np.testing.assert_allclose(actual_base.numpy(), base)
    np.testing.assert_allclose(score_features(torch.tensor(z), obs, actual_base, mode).numpy(), features, atol=2e-15)
    actual = QueryScoreAdapter(mode, gamma, weights=weights)(torch.tensor(z), obs, actual_base)
    np.testing.assert_allclose(actual.detach().exp().numpy(), expected, atol=2e-15)


def test_adapter_artifacts_bind_actual_weights(raw, tmp_path):
    model = TableModel(raw)
    adapter = QueryScoreAdapter('global', metadata={'backbone_state_sha256': state_fingerprint(model)})
    path = tmp_path/'adapter.json'
    adapter.save(path)
    loaded = QueryScoreAdapter.load(path, model=model)
    torch.testing.assert_close(adapter.weights, loaded.weights)
    checkpoint = tmp_path/'model.pt'
    torch.save({'model': model.state_dict()}, checkpoint)
    legacy = {'all_sources': dict(weights=[[0.]*4]*2, feature_mode='context', observed_mix=1.,
                                 backbone_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())}
    path.write_text(json.dumps(legacy))
    QueryScoreAdapter.load(path, model=model, key='all_sources', checkpoint=checkpoint)
    with pytest.raises(ValueError, match='original checkpoint'):
        QueryScoreAdapter.load(path, model=model, key='all_sources')
    with torch.no_grad():
        model.table[0, 0, 0] += 1
    with pytest.raises(ValueError, match='modified backbone'):
        QueryAnswerer(model, adapter=loaded).predict((0, (0,)))
    with pytest.raises(ValueError, match='modified backbone'):
        QueryScoreAdapter.load(path, model=model, key='all_sources', checkpoint=checkpoint)


@pytest.mark.parametrize('name', ['ULTRA', 'TRIX', 'Flock'])
def test_actual_kgfms_share_the_api_and_restore_context(name):
    from dicee.models import TRIX, ULTRA, Flock
    cls = {'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[name]
    settings = dict(num_entities=4, num_relations=4, ultra_dim=8, ultra_num_layers=2, trix_dim=8,
                    flock_dim=8, flock_walk_num=2, flock_walk_len=8, flock_refinements=2, flock_seed=77)
    facts = torch.tensor([[0, 2, 1], [1, 0, 2], [2, 2, 3]])
    model = cls(settings).set_graph(facts, inverse_relations={2: 1, 0: 3}).eval()
    context = QueryContext.from_model(model)
    assert context.inverse_relations == ((0, 3), (2, 1))
    assert (1, 1, 0) in context.triples and (2, 3, 1) in context.triples
    engine = QueryAnswerer(model, observed_mix=1, seed=19)
    first = engine.predict((0, (2, 0)), beam_size=2)
    assert first[2] == 1 and first.shape == (4,)
    kge = wrapper(model)
    for shape in QUERY_SHAPES:
        query = programs(np.full((4, 2, 4), .5), 2)[shape][0]
        result = kge.answer_multi_hop_query(shape, named(query), only_scores=True, k=2)
        assert result.shape == (4,) and torch.isfinite(result).all()
    if name == 'Flock':
        from dicee.query_answering.engine import AtomicScorer
        state = torch.random.get_rng_state().clone()
        scorer = AtomicScorer(model, seed=9)
        batched = scorer.rows([(0, 2), (1, 0)])
        serial = torch.cat([scorer.rows([(h, r)]) for h, r in [(1, 0), (0, 2)]])
        torch.testing.assert_close(batched, serial.flip(0), rtol=0, atol=0)
        assert torch.equal(state, torch.random.get_rng_state()) and model.seed == 77
    model.set_graph(torch.tensor([[0, 2, 3]]), inverse_relations={2: 1, 0: 3})
    with pytest.raises(ValueError, match='must match'):
        QueryAnswerer(model, context=context)
    after = engine.predict((0, (2,)), beam_size=2)
    assert after[3] == 1 and engine.scorer.context != context


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_scores_stay_on_device(raw):
    model = TableModel(raw).cuda()
    engine = QueryAnswerer(model)
    result = engine.predict((0, (0, 1)))
    assert result.is_cuda
    expected = QueryAnswerer(TableModel(raw)).predict((0, (0, 1)))
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize('shape', QUERY_SHAPES)
@pytest.mark.parametrize('tnorm', ['prod', 'min'])
def test_persistent_cache_matches_formulas_and_owns_returned_scores(raw, shape, tnorm):
    query, expected = programs(1/(1+np.exp(-raw)), 2, tnorm)[shape]
    model = TableModel(raw).eval()
    engine = QueryAnswerer(model, row_batch_size=2)
    first = engine.predict(query, beam_size=2, tnorm=tnorm, return_log_scores=True)
    np.testing.assert_allclose(first.exp().numpy(), expected, atol=2e-15, rtol=2e-14)
    calls = len(model.calls)
    first.fill_(123.)
    second = engine.predict(query, beam_size=2, tnorm=tnorm)
    np.testing.assert_allclose(second.numpy(), expected, atol=2e-15, rtol=2e-14)
    assert len(model.calls) == calls and engine.last_info['raw_rows'] == 0
    assert engine.last_info['cache_hits'] > 0


def test_persistent_lru_eviction_resize_and_disable(raw):
    model = TableModel(raw).eval()
    engine = QueryAnswerer(model, cache_bytes=64)
    for head in (0, 1, 0, 2, 0):
        engine.predict((head, (0,)))
    assert len(model.calls) == 3
    engine.predict((1, (0,)))
    assert engine.last_info['raw_rows'] == 1
    engine.cache_bytes = 32
    engine.predict((1, (0,)))
    assert engine.last_info['raw_rows'] == 0 and engine.last_info['cache_bytes'] == 32
    engine.cache_bytes = 0
    engine.predict((1, (0,)))
    assert engine.last_info['raw_rows'] == 1 and engine.last_info['cache_bytes'] == 0
    # Eviction during a beam batch must not invalidate its in-flight rows.
    engine.cache_bytes = 32
    actual = engine.predict((0, (0, 1)), beam_size=4)
    torch.testing.assert_close(actual, QueryAnswerer(TableModel(raw), cache_bytes=0).predict((0, (0, 1)), beam_size=4))
    engine.clear_cache()
    assert not engine._cache and engine._cache_used == 0


@pytest.mark.parametrize('change', ['inplace', 'parameter', 'buffer', 'load', 'dtype', 'adapter', 'observed', 'context', 'precision'])
def test_persistent_cache_invalidates_changed_inputs(raw, change, monkeypatch):
    model = TableModel(raw).eval()
    model.register_buffer('extra', torch.tensor(0.))
    context = QueryContext([(0, 0, 1)], 4, 2)
    adapter = QueryScoreAdapter('context', observed_mix=.5)
    engine = QueryAnswerer(model, adapter=adapter, context=context)
    query = (0, (0,))
    engine.predict(query)
    engine.predict(query)
    assert engine.last_info['raw_rows'] == 0
    with torch.no_grad():
        if change == 'inplace':
            model.table.add_(1)
        elif change == 'parameter':
            model.table = nn.Parameter(model.table.clone() + 1)
        elif change == 'buffer':
            model.extra.add_(1)
        elif change == 'load':
            model.load_state_dict(dict(table=model.table + 1, extra=model.extra))
        elif change == 'dtype':
            model.float()
        elif change == 'adapter':
            adapter.weights.add_(.1)
        elif change == 'observed':
            adapter.observed_mix = 1.
        elif change == 'context':
            engine.scorer.explicit_context = QueryContext([(0, 0, 2)], 4, 2)
        elif change == 'precision':
            monkeypatch.setattr('dicee.models._inference.float32_precision_token', lambda: ('changed',))
    actual = engine.predict(query)
    assert engine.last_info['raw_rows'] == 1
    expected = QueryAnswerer(model, adapter=adapter, context=engine.scorer.context, cache_bytes=0).predict(query)
    torch.testing.assert_close(actual, expected)


def test_cached_adapter_verification_still_rejects_mutations(raw, monkeypatch):
    import dicee.query_answering.adapter as module
    model = TableModel(raw).eval()
    digest = state_fingerprint(model)
    adapter = QueryScoreAdapter('global', metadata={'backbone_state_sha256': digest})
    real_hash, calls = module.state_fingerprint, []
    def counted(value):
        calls.append(value)
        return real_hash(value)
    monkeypatch.setattr(module, 'state_fingerprint', counted)
    engine = QueryAnswerer(model, adapter=adapter)
    engine.predict((0, (0,)))
    engine.predict((0, (0,)))
    assert len(calls) == 1
    with torch.no_grad():
        model.table.add_(1)
    with pytest.raises(ValueError, match='modified backbone'):
        engine.predict((0, (0,)))
    assert len(calls) == 2


def test_inference_tensors_and_training_callers_do_not_persist_rows(raw):
    for inference in (False, True):
        with torch.inference_mode(inference):
            model = TableModel(raw)
        if inference:
            model.eval()
        engine = QueryAnswerer(model)
        engine.predict((0, (0,)))
        engine.predict((0, (0,)))
        assert engine.last_info['raw_rows'] == 1


def test_graph_context_validation_uses_identity_and_detects_changes(monkeypatch):
    from dicee.models import ULTRA
    model = ULTRA(dict(num_entities=3, num_relations=2, ultra_dim=8, ultra_num_layers=2))
    model.set_graph([(0, 0, 1)], inverse_relations={0: 1}).eval()
    context = QueryContext.from_model(model)
    def no_deep_comparison(*args):
        raise AssertionError('must not walk the triples for equality')
    monkeypatch.setattr(QueryContext, '__eq__', no_deep_comparison)
    engine = QueryAnswerer(model, context=context)
    engine.predict((0, (0,)))
    engine.predict((0, (0,)))
    assert engine.last_info['raw_rows'] == 0
    model.set_graph([(0, 0, 2)], inverse_relations={0: 1})
    with pytest.raises(ValueError, match='must match'):
        engine.predict((0, (0,)))


def test_flock_persistent_cache_tracks_sampling():
    from dicee.models import Flock
    model = Flock(dict(num_entities=4, num_relations=2, flock_dim=8, flock_walk_num=2,
                       flock_walk_len=8, flock_refinements=2)).set_graph([(0, 0, 1), (1, 1, 2)]).eval()
    engine = QueryAnswerer(model, seed=13)
    query = (0, (0,))
    engine.predict(query)
    engine.predict(query)
    assert engine.last_info['raw_rows'] == 0
    for attr, value in [('seed', 14), ('samples', 2)]:
        setattr(engine.scorer, attr, value)
        actual = engine.predict(query)
        assert engine.last_info['raw_rows'] == 1
        expected = QueryAnswerer(model, seed=engine.scorer.seed, samples=engine.scorer.samples, cache_bytes=0).predict(query)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_adapter_device_copy_refresh_and_training_gradients(raw):
    adapter = QueryScoreAdapter('global')
    logits = torch.tensor(raw[0, 0][None], device='cuda', dtype=torch.float64)
    with torch.no_grad():
        before = adapter(logits)
        adapter.weights[1, 0].add_(.2)
        after = adapter(logits)
        assert (after > before).all()
        torch.testing.assert_close(after, QueryScoreAdapter('global', weights=adapter.weights).cuda()(logits))
    adapter(logits).sum().backward()
    assert adapter.weights.grad is not None and torch.isfinite(adapter.weights.grad).all()


def test_eval_failure_restores_modes_and_does_not_poison_cache(raw, monkeypatch):
    model = TableModel(raw).eval()
    engine = QueryAnswerer(model)
    original = model.forward_k_vs_all
    def fail(_):
        raise RuntimeError('scoring failed')
    monkeypatch.setattr(model, 'forward_k_vs_all', fail)
    with pytest.raises(RuntimeError, match='scoring failed'):
        engine.predict((0, (0,)))
    assert not model.training and not engine.scorer._evaluating
    monkeypatch.setattr(model, 'forward_k_vs_all', original)
    actual = engine.predict((0, (0,)))
    torch.testing.assert_close(actual, model.table[0, 0].sigmoid())


def test_cache_separates_legacy_logits_and_bypasses_autocast(raw):
    model = TableModel(raw).eval()
    engine = QueryAnswerer(model)
    query = (0, (0,))
    engine.predict(query)
    logits = engine.predict(query, use_logits=True)
    torch.testing.assert_close(logits, model.table[0, 0])
    assert engine.last_info['raw_rows'] == 1
    scores = engine.predict(query)
    torch.testing.assert_close(scores, model.table[0, 0].sigmoid())
    assert engine.last_info['raw_rows'] == 1
    with torch.autocast('cpu', dtype=torch.bfloat16):
        for _ in range(2):
            engine.predict(query)
            assert engine.last_info['raw_rows'] == 1
