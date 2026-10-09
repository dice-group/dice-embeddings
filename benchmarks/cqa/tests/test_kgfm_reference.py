"""Independent integration oracle checks, including ties and observed proofs."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dicee.query_answering import AdapterQuery, AdapterTrainingData, QueryAnswerer, QueryContext, QueryScoreAdapter, fit_query_adapter
from dicee.query_answering._query import QUERY_SHAPES
from dicee.query_answering.context import state_fingerprint

spec = importlib.util.spec_from_file_location('reference_kgfm', Path(__file__).parents[1] / 'verification' / 'reference_kgfm.py')
reference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_module)


class Rows(torch.nn.Module):
    num_entities, num_relations = 7, 4

    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.randn(7, 4, 7, generator=torch.Generator().manual_seed(8)))

    def forward_k_vs_all(self, conditions):
        return self.values[conditions[:, 0], conditions[:, 1]]


@pytest.mark.parametrize('name', ['DistMult', 'ULTRA', 'TRIX', 'Flock'])
def test_default_adapter_fits_and_reloads_for_every_backbone(name, tmp_path):
    from dicee.models import TRIX, ULTRA, DistMult, Flock

    model = {'DistMult': DistMult, 'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[name](dict(
        num_entities=7, num_relations=4, embedding_dim=8, ultra_dim=8, ultra_num_layers=2,
        trix_dim=8, flock_dim=8, flock_walk_num=2, flock_walk_len=8, flock_refinements=2, flock_seed=17,
    )).eval()
    context = QueryContext([(0, 0, 2), (1, 1, 2)], 7, 4)
    data = AdapterTrainingData('source', context, (AdapterQuery(((0, (0,)), (1, (1,))), {2, 3}),))
    before = state_fingerprint(model)
    result = fit_query_adapter(model, [data], epochs=2)
    assert result.adapter.feature_mode == QueryScoreAdapter().feature_mode == 'context_scores'
    assert result.adapter.hidden_dim == 0 and result.adapter.weights.numel() == 16
    assert result.adapter.weights.count_nonzero() > 0
    assert state_fingerprint(model) == before
    result.adapter.save(tmp_path / 'adapter.json')
    restored = QueryScoreAdapter.load(tmp_path / 'adapter.json', model=model)
    assert restored.feature_mode == 'context_scores'
    torch.testing.assert_close(restored.weights, result.adapter.weights, atol=0, rtol=0)


@pytest.mark.parametrize('hidden_dim', [0, 5])
def test_adapter_feature_alias_preserves_scores_and_saves_canonical_name(hidden_dim, tmp_path):
    model = Rows()
    context = QueryContext([(0, 0, 2), (1, 1, 2)], 7, 4)
    expected = QueryScoreAdapter('context_scores', 1., hidden_dim=hidden_dim,
                                weights=torch.full((2, hidden_dim or 8), .1),
                                metadata={'backbone_state_sha256': state_fingerprint(model)})
    payload = dict(expected.to_dict(), feature_mode='context_scores_v1')
    path = tmp_path / 'adapter.json'
    path.write_text(json.dumps(payload))
    actual = QueryScoreAdapter.load(path, model=model)
    raw = model.forward_k_vs_all(torch.tensor([[0, 0], [1, 1]]))
    observed, base = context.features([(0, 0), (1, 1)], device='cpu')
    torch.testing.assert_close(actual(raw, observed, base), expected(raw, observed, base), atol=0, rtol=0)
    actual.save(tmp_path / 'canonical.json')
    assert json.loads((tmp_path / 'canonical.json').read_text())['feature_mode'] == 'context_scores'


def instantiate(shape, counters=None):
    counters = [0, 0] if counters is None else counters
    if shape == 'e':
        counters[0] += 1
        return (counters[0] - 1) % 7
    if shape == 'r':
        counters[1] += 1
        return (counters[1] - 1) % 4
    if shape in ('n', 'u'):
        return -2 if shape == 'n' else -1
    return tuple(instantiate(child, counters) for child in shape)


@pytest.mark.parametrize('operator', ['product', 'min'])
@pytest.mark.parametrize('cache_bytes', [0, 4096])
@pytest.mark.parametrize('facts', ['none', 'atomic', 'both'])
def test_independent_reference_covers_all_shapes(operator, cache_bytes, facts):
    model = Rows().eval()
    context = QueryContext([(0, 0, 1), (0, 0, 2), (2, 1, 6), (1, 1, 3), (3, 2, 4), (4, 3, 5)], 7, 4)
    adapter = QueryScoreAdapter('context_scores', float(facts != 'none'), bias_bound=8.,
                                weights=torch.randn(2, 8, generator=torch.Generator().manual_seed(9)).double() / 5)
    reference = reference_module.ReferenceKGFM(context.triples, 7, 4, adapter.to_dict(), beam_size=1, device='cpu',
                                               restore_observed=facts == 'both')
    conditions = [(h, r) for h in range(7) for r in range(4)]
    with torch.no_grad():
        for start in range(0, len(conditions), 2):
            batch = conditions[start:start + 2]
            reference.capture(batch, model.forward_k_vs_all(torch.tensor(batch)))
    engine = QueryAnswerer(model, context=context, adapter=adapter, row_batch_size=2, cache_bytes=cache_bytes,
                           restore_observed=facts == 'both')
    queries = [instantiate(shape) for shape in QUERY_SHAPES.values()]
    engine.prefetch(queries)
    for query in queries:
        expected = reference.predict(query, operator)
        actual = engine.predict(query, beam_size=1, tnorm='prod' if operator == 'product' else 'min', return_log_scores=True)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        assert torch.equal(actual.argsort(descending=True), expected.argsort(descending=True))
    # Both observed first hops tie at one; ID 2 is pruned, but its proof survives.
    assert (reference.predict((0, (0, 1)), operator)[6] == 0).item() == (facts == 'both')
    assert (engine.predict((0, (0,)), return_log_scores=True)[1] == 0).item() == (facts != 'none')
    with torch.no_grad():
        adapter.weights[1].add_(1)
    assert not torch.allclose(engine.predict((0, (0,)), return_log_scores=True), reference.predict((0, (0,))))


@pytest.mark.parametrize('operator', ['product', 'min'])
@pytest.mark.parametrize('facts', ['none', 'both'])
@pytest.mark.parametrize('mode', ['global', 'context_scores'])
def test_reference_applies_a_fixed_membership_threshold(operator, facts, mode):
    model = Rows().eval()
    context = QueryContext([(0, 0, 1), (0, 0, 2), (2, 1, 6), (1, 1, 3), (3, 2, 4), (4, 3, 5)], 7, 4)
    weights = None if mode == 'global' else torch.randn(2, 8, generator=torch.Generator().manual_seed(9)).double() / 5
    conditions = [(h, r) for h in range(7) for r in range(4)]
    with torch.no_grad():
        # Threshold at the median unobserved membership, so it zeroes about half of them.
        memberships = QueryScoreAdapter(mode, 0., bias_bound=8., weights=weights)(
            model.forward_k_vs_all(torch.tensor(conditions)), *context.features(conditions, device='cpu')).exp()
    adapter = QueryScoreAdapter(mode, float(facts != 'none'), bias_bound=8., weights=weights,
                                membership_threshold=memberships.median().item())
    reference = reference_module.ReferenceKGFM(context.triples, 7, 4, adapter.to_dict(), beam_size=3, device='cpu',
                                               restore_observed=facts == 'both')
    with torch.no_grad():
        for start in range(0, len(conditions), 2):
            batch = conditions[start:start + 2]
            reference.capture(batch, model.forward_k_vs_all(torch.tensor(batch)))
    # The threshold zeroes part of some rows, so ties at zero membership reach the beams.
    assert any(torch.isneginf(row).any() and torch.isfinite(row).any() for row in reference.rows.values())
    engine = QueryAnswerer(model, context=context, adapter=adapter, row_batch_size=2, restore_observed=facts == 'both')
    queries = [instantiate(shape) for shape in QUERY_SHAPES.values()]
    engine.prefetch(queries)
    for query in queries:
        expected = reference.predict(query, operator)
        actual = engine.predict(query, beam_size=3, tnorm='prod' if operator == 'product' else 'min', return_log_scores=True)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        assert torch.equal(actual.argsort(descending=True), expected.argsort(descending=True))
    unthresholded = reference_module.ReferenceKGFM(context.triples, 7, 4, dict(adapter.to_dict(), membership_threshold=0.),
                                                   beam_size=3, device='cpu')
    unthresholded.capture([(0, 0)], model.forward_k_vs_all(torch.tensor([[0, 0]])))
    assert not torch.equal(unthresholded.rows[0, 0], reference.rows[0, 0])
    with pytest.raises(ValueError):
        reference_module.ReferenceKGFM(context.triples, 7, 4, dict(adapter.to_dict(), membership_threshold=1.),
                                       beam_size=3, device='cpu')


@pytest.mark.parametrize('operator', ['product', 'min'])
@pytest.mark.parametrize('facts', ['none', 'atomic', 'both'])
@pytest.mark.parametrize('calibration', ['minmax', 'softmax-degree', 'softmax', 'softmax-degree-ties', 'softmax-degree-ties-masked'])
def test_reference_applies_training_free_calibrations(operator, facts, calibration):
    model = Rows().eval()
    context = QueryContext([(0, 0, 1), (0, 0, 2), (2, 1, 6), (1, 1, 3), (3, 2, 4), (4, 3, 5)], 7, 4)
    masked = calibration.endswith('-masked')
    adapter = QueryScoreAdapter('global', float(facts != 'none'), bias_bound=8., fixed_calibration=calibration.removesuffix('-masked'),
                                mask_known_logits=masked)
    reference = reference_module.ReferenceKGFM(context.triples, 7, 4, adapter.to_dict(), beam_size=2, device='cpu',
                                               restore_observed=facts == 'both')
    conditions = [(h, r) for h in range(7) for r in range(4)]
    with torch.no_grad():
        for start in range(0, len(conditions), 2):
            batch = conditions[start:start + 2]
            reference.capture(batch, model.forward_k_vs_all(torch.tensor(batch)))
    engine = QueryAnswerer(model, context=context, adapter=adapter, row_batch_size=2, restore_observed=facts == 'both')
    queries = [instantiate(shape) for shape in QUERY_SHAPES.values()]
    engine.prefetch(queries)
    for query in queries:
        expected = reference.predict(query, operator)
        actual = engine.predict(query, beam_size=2, tnorm='prod' if operator == 'product' else 'min', return_log_scores=True)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        assert torch.equal(actual.argsort(descending=True), expected.argsort(descending=True))
    with pytest.raises(ValueError):
        reference_module.ReferenceKGFM(context.triples, 7, 4, dict(adapter.to_dict(), weights=[[1.], [0.]]),
                                       beam_size=2, device='cpu')


@pytest.mark.parametrize('operator', ['product', 'min'])
@pytest.mark.parametrize('calibration', [None, 'softmax-degree'])
def test_reference_orders_known_tails_by_the_backbone_when_asked(operator, calibration):
    model = Rows().eval()
    # (0, 0) has three known tails, more than the beam of two holds.
    context = QueryContext([(0, 0, 1), (0, 0, 2), (0, 0, 5), (2, 1, 6), (1, 1, 3), (5, 1, 4), (3, 2, 4), (4, 3, 5)], 7, 4)
    adapter = QueryScoreAdapter('global', 1., bias_bound=8., fixed_calibration=calibration)
    adapter.observed_tie_break = True
    reference = reference_module.ReferenceKGFM(context.triples, 7, 4, adapter.to_dict(), beam_size=2, device='cpu',
                                               observed_ties='model')
    conditions = [(h, r) for h in range(7) for r in range(4)]
    with torch.no_grad():
        for start in range(0, len(conditions), 2):
            batch = conditions[start:start + 2]
            reference.capture(batch, model.forward_k_vs_all(torch.tensor(batch)))
    engine = QueryAnswerer(model, context=context, adapter=adapter, row_batch_size=2)
    plain = QueryAnswerer(model, context=context, adapter=QueryScoreAdapter('global', 1., bias_bound=8., fixed_calibration=calibration),
                          row_batch_size=2)
    queries = [instantiate(shape) for shape in QUERY_SHAPES.values()]
    engine.prefetch(queries)
    plain.prefetch(queries)
    changed = False
    for query in queries:
        expected = reference.predict(query, operator)
        tnorm = 'prod' if operator == 'product' else 'min'
        actual = engine.predict(query, beam_size=2, tnorm=tnorm, return_log_scores=True)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        assert torch.equal(actual.argsort(descending=True), expected.argsort(descending=True))
        changed |= not torch.equal(actual, plain.predict(query, beam_size=2, tnorm=tnorm, return_log_scores=True))
    assert changed


@pytest.mark.parametrize('operator', ['product', 'min'])
@pytest.mark.parametrize('calibration', [None, 'softmax-degree'])
def test_reference_negates_from_known_facts_when_asked(operator, calibration):
    model = Rows().eval()
    context = QueryContext([(0, 0, 1), (0, 0, 2), (0, 0, 5), (2, 1, 6), (1, 1, 3), (5, 1, 4), (3, 2, 4), (4, 3, 5)], 7, 4)
    adapter = QueryScoreAdapter('global', 1., bias_bound=8., fixed_calibration=calibration)
    reference = reference_module.ReferenceKGFM(context.triples, 7, 4, adapter.to_dict(), beam_size=2, device='cpu', negation='observed')
    conditions = [(h, r) for h in range(7) for r in range(4)]
    with torch.no_grad():
        for start in range(0, len(conditions), 2):
            batch = conditions[start:start + 2]
            reference.capture(batch, model.forward_k_vs_all(torch.tensor(batch)))
    engine = QueryAnswerer(model, context=context, adapter=adapter, row_batch_size=2)
    queries = [instantiate(shape) for shape in QUERY_SHAPES.values()]
    engine.prefetch(queries)
    tnorm = 'prod' if operator == 'product' else 'min'
    changed = False
    for query in queries:
        expected = reference.predict(query, operator)
        actual = engine.predict(query, beam_size=2, tnorm=tnorm, return_log_scores=True, negation='observed')
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
        assert torch.equal(actual.argsort(descending=True), expected.argsort(descending=True))
        changed |= not torch.equal(actual, engine.predict(query, beam_size=2, tnorm=tnorm, return_log_scores=True))
    assert changed


@pytest.mark.parametrize('operator', ['product', 'min'])
def test_reference_needs_no_rows_for_zero_prefix_but_requires_live_rows(operator):
    adapter = QueryScoreAdapter('context_scores', 1.)
    reference = reference_module.ReferenceKGFM([(0, 0, t) for t in range(7)], 7, 4,
                                                adapter.to_dict(), beam_size=7, device='cpu')
    reference.capture([(0, 0)], torch.zeros(1, 7))
    contradiction = (((0, (0,)), (0, (0, -2))), (1,))
    assert torch.isneginf(reference.predict(contradiction, operator)).all()
    with pytest.raises(KeyError):
        reference.predict((0, (0, 1)), operator)


def test_recipe_rules_of_the_review_backbone_options():
    from dicee.query_answering.catalog import KGFM_ADAPTERS, KGFM_BACKBONES, check_recipe
    shapes = ['1p', '2i']
    base = dict(operators={'1p': 'product', '2i': 'product'}, adapters={'product': 'a.json'}, selection_protocol='source-validation')
    assert 'flock-adapter' in KGFM_ADAPTERS and KGFM_BACKBONES['flock-adapter'][0] == 'jw9730/flock'
    check_recipe(dict(base, method='flock-adapter', options=dict(test_samples=1, walk_num=128)), shapes)
    check_recipe(dict(base, method='ultra-adapter', options=dict(relation_conditioning='query')), shapes)
    for method, options in (('ultra-adapter', dict(test_samples=1)), ('flock-adapter', dict(walk_num=0)),
                            ('trix-adapter', dict(relation_conditioning='query')), ('ultra-adapter', dict(relation_conditioning='inverse'))):
        with pytest.raises(ValueError):
            check_recipe(dict(base, method=method, options=options), shapes)
