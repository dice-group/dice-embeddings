"""Score adapters, paired identity controls, shared backbone rows and crash-safe comparison traces."""

import json

import pytest
import torch

from benchmarks.cqa import cli
from benchmarks.cqa.tests.test_protocol import fixture_manifest, read, small_data, trace_rows, use_ultraquery_fixture
from dicee.query_answering import AdapterQuery, AdapterTrainingData, QueryContext, QueryScoreAdapter, fit_query_adapter
from dicee.query_answering._checkpoint import write_json
from dicee.query_answering.benchmark import evaluate_benchmark
from dicee.query_answering.context import state_fingerprint
from dicee.query_answering.engine import AtomicBatchCache, AtomicScorer


class BatchRows(torch.nn.Module):
    num_entities, num_relations = 6, 1

    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.arange(36.).reshape(6, 6))
        self.calls = 0

    def forward_k_vs_all(self, conditions):
        self.calls += 1
        return self.values[conditions[:, 0]] + len(conditions) / 100


def test_shared_atomic_cache_preserves_batches_and_invalidates_weights():
    model = BatchRows().eval()
    cache = AtomicBatchCache(48)
    first = AtomicScorer(model, row_batch_size=2, raw_cache=cache)
    second = AtomicScorer(model, row_batch_size=2, raw_cache=cache)
    pair = [(0, 0), (1, 0)]
    expected = first.rows(pair)
    reused = second.rows(pair)
    assert torch.equal(expected, reused) and model.calls == 1 and cache.hits == 2
    reused.zero_()
    assert torch.equal(second.rows(pair), expected)
    # A singleton has different floating-point batch semantics in this fixture.
    assert not torch.equal(second.rows(pair[:1])[0], expected[0])
    assert model.calls == 2 and cache.used <= 48
    first.rows(pair)
    assert model.calls == 3  # The complete pair was evicted.
    with torch.no_grad():
        model.values.add_(1)
    assert torch.equal(second.rows(pair), expected + 1)
    assert model.calls == 4


def test_row_cache_reuses_rows_across_batches_and_invalidates_weights():
    model = BatchRows().eval()
    cache = AtomicBatchCache(10_000, granularity='row')
    scorer = AtomicScorer(model, row_batch_size=2, raw_cache=cache)
    first = scorer.rows([(0, 0), (1, 0)])
    assert model.calls == 1 and cache.computed == 2
    mixed = scorer.rows([(1, 0), (2, 0), (1, 0)])
    # Only the missing row is computed, in a batch of its own; the fixture marks batch sizes in the values.
    assert model.calls == 2 and cache.computed == 3 and cache.hits == 2
    assert torch.equal(mixed[0], first[1]) and torch.equal(mixed[2], first[1])
    assert torch.equal(mixed[1], model.values[2].detach() + 1 / 100)
    mixed.zero_()
    assert torch.equal(scorer.rows([(0, 0)])[0], first[0]) and model.calls == 2
    with torch.no_grad():
        model.values.add_(1)
    assert torch.equal(scorer.rows([(0, 0), (1, 0)]), first + 1) and model.calls == 3


def test_cache_token_is_computed_once_per_evaluation(monkeypatch):
    from dicee.query_answering import engine
    calls = []
    original = engine.state_token
    monkeypatch.setattr(engine, 'state_token', lambda model: calls.append(model) or original(model))
    scorer = AtomicScorer(BatchRows().eval(), row_batch_size=1, raw_cache=AtomicBatchCache(10_000, granularity='row'))
    with scorer.evaluation():
        for head in range(3):
            scorer.rows([(head, 0)])
    assert len(calls) == 1
    scorer.rows([(0, 0)])
    assert len(calls) == 2


def test_paired_resume_replays_both_variants_and_preserves_metrics(tmp_path):
    data = small_data()

    def learned(query):
        return torch.tensor([10., 8., 7., 6., 5., 4.])

    def control(query):
        return torch.tensor([10., 1., 2., 3., 4., 5.])

    settings = dict(checkpoint_dir=tmp_path / 'progress', checkpoint_identity={'paired': True},
                    checkpoint_every=2, rank_trace_path=tmp_path / 'ranks.sqlite3', additional_tie_policies=('expected',))

    def fail(query):
        if query[0] == 3:
            raise RuntimeError('interrupted between variants')
        return control(query)

    with pytest.raises(RuntimeError, match='between variants'):
        evaluate_benchmark(data, learned, comparison_predictors={'without-adapter': fail}, **settings)
    seen = []

    def resume(query):
        seen.append(query[0])
        return learned(query)

    result = evaluate_benchmark(data, resume, comparison_predictors={'without-adapter': control}, **settings)
    assert seen == [2, 3, 4, 5]
    assert result['per_shape'] == evaluate_benchmark(data, learned)['per_shape']
    assert result['comparisons']['without-adapter']['per_shape'] == evaluate_benchmark(data, control)['per_shape']
    left, right = trace_rows(tmp_path / 'ranks.sqlite3'), trace_rows(tmp_path / 'without-adapter/ranks.sqlite3')
    # Both traces hold every query once, in the same order.
    assert len(left) == 6 and [row[:2] for row in left] == [row[:2] for row in right]


def test_filter_controls_share_predictions_and_resume_all_traces(tmp_path, monkeypatch):
    from dataclasses import replace

    from dicee.query_answering._checkpoint import BenchmarkCheckpoint
    from dicee.query_answering.method_evaluation import save_results
    calls = []
    data = replace(small_data(), metadata={'expected_query_types': ['1p']})
    def predict(query):
        calls.append(query)
        return torch.tensor([10., 8., 7., 6., 5., 4.])
    options = dict(checkpoint_dir=tmp_path / 'progress', checkpoint_identity={'fixed': True},
                   checkpoint_every=2, rank_trace_path=tmp_path / 'ranks.sqlite3',
                   additional_tie_policies=('expected',), comparison_predictors={'without-adapter': predict},
                   filter_corrections={q.query: frozenset() for q in data.queries})
    original = BenchmarkCheckpoint.save
    def fail_after_commit(self, state):
        if state['completed'] == 4:
            raise RuntimeError('interrupted filter control')
        return original(self, state)
    monkeypatch.setattr(BenchmarkCheckpoint, 'save', fail_after_commit)
    with pytest.raises(RuntimeError, match='interrupted filter control'):
        evaluate_benchmark(data, predict, **options)
    monkeypatch.setattr(BenchmarkCheckpoint, 'save', original)
    calls.clear()
    report = evaluate_benchmark(data, predict, **options)
    assert len(calls) == 8  # Four remaining queries, two predictors, both filters each.
    assert report['averages']['all']['mrr'] == .5
    assert report['comparisons']['released-filters']['averages']['all']['mrr'] == 1.
    for name, result in [(None, report), *report['comparisons'].items()]:
        result['inference'] = dict(method='ultra-adapter', calibration='without-adapter' if name and name.startswith('without-adapter') else 'learned')
    save_results(report, tmp_path, run={'entry': 'paired', 'phase': 'pilot'}, metadata={'reference': {}})
    # One result and one complete rank trace per calibration and answer filter.
    for name in ('', 'without-adapter', 'released-filters', 'without-adapter-released-filters'):
        assert read(tmp_path / name / 'result.json')['queries'] == 6
        assert len(trace_rows(tmp_path / name / 'ranks.sqlite3')) == 6
    options['filter_corrections'] = {}
    with pytest.raises(ValueError, match='inputs/settings changed'):
        evaluate_benchmark(data, predict, **options)


def kgfm_entry(tmp_path, manifest, backbone, **options):
    """Turn the fixture's entry into a paired adapter recipe of a random ``backbone`` checkpoint."""
    from dicee.models import KGICL, TRIX, ULTRA
    model = {'ultra': ULTRA, 'trix': TRIX, 'kgicl': KGICL}[backbone](dict(num_entities=1, num_relations=1))
    # KG-ICL checkpoints keep their weights under 'state_dict'.
    torch.save({'state_dict' if backbone == 'kgicl' else 'model': model.state_dict()}, tmp_path / 'kgfm.pt')
    state = state_fingerprint(model)
    QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=torch.randn(2, 8, generator=torch.Generator().manual_seed(4)).double() / 5,
                      metadata={'backbone_state_sha256': state}).save(tmp_path / 'product.json')
    QueryScoreAdapter('global', 1., metadata={'backbone_state_sha256': state, 'training': {'tnorm': 'min'}}).save(tmp_path / 'min.json')
    entry = manifest['entries'][0]
    entry.update(id=f'{backbone}-paired', method=f'{backbone}-adapter', checkpoint='kgfm.pt',
                 options={'beam_size': 2, 'cache_bytes': 4096, **options}, adapters={'product': 'product.json', 'min': 'min.json'},
                 operators={shape: 'min' if 'n' in shape else 'product' for shape in entry['query_types']},
                 selection_protocol='source-validation', adapter_ablation=True, query_batch_size=4, query_order='relation')
    return entry


@pytest.mark.parametrize('backbone', ['ultra', 'trix', 'kgicl'])
@pytest.mark.parametrize('facts', [None, 'none', 'atomic', 'both'])
def test_paired_kgfm_evaluation_shares_backbone_rows_and_applies_observed_fact_modes(tmp_path, monkeypatch, backbone, facts):
    from dicee.query_answering import QueryAnswerer
    manifest = fixture_manifest(tmp_path)
    entry = kgfm_entry(tmp_path, manifest, backbone, row_batch_size=1, backend_batch_size=1, raw_cache_bytes=4096)
    write_json(tmp_path / 'manifest.json', manifest)
    args = ['plus_h', 'evaluate', '--manifests', str(tmp_path / 'manifest.json'), '--answer-filter', 'released', '--input-root', str(tmp_path),
            '--output', str(tmp_path / 'results'), '--split', 'test', '--device', 'cpu']
    name = entry['id']
    if facts is not None:
        args += ['--observed-facts', facts]
        name += f'-facts-{facts}'
    cli.main(args)
    result, control = read(tmp_path / 'results' / name / 'result.json'), read(tmp_path / 'results' / name / 'without-adapter/result.json')
    assert result['queries'] == control['queries'] == 16
    assert result['inference']['selection_protocol'] == 'source-validation'
    assert result['inference']['cache_statistics']['backbone_reused_rows'] > 0
    assert control['paired_with'] == name and control['inference']['calibration'] == 'without-adapter'
    settings = result['inference']['observed_facts']
    assert settings == control['inference']['observed_facts']
    assert settings['atomic_mix'] == dict.fromkeys(('product', 'min'), float(facts != 'none'))
    assert settings['restore_answers'] == (facts in (None, 'both'))
    # The mode applies at inference; the adapter file keeps its own observed-fact weight.
    assert read(tmp_path / 'product.json')['observed_mix'] == 1
    monkeypatch.setattr(QueryAnswerer, 'predict', lambda *args, **kwargs: pytest.fail('Completed queries were rescored'))
    cli.main(args)
    assert read(tmp_path / 'results' / name / 'result.json')['per_shape'] == result['per_shape']


@pytest.mark.parametrize('backbone', ['ultra', 'trix', 'kgicl'])
def test_row_granular_raw_cache_reaches_the_engine_and_keeps_scores(tmp_path, backbone):
    manifest = fixture_manifest(tmp_path)
    use_ultraquery_fixture(manifest, tmp_path)
    entry = kgfm_entry(tmp_path, manifest, backbone, row_batch_size=2, backend_batch_size=2, raw_cache_bytes=1 << 20,
                       raw_cache_device='model', relation_cache_mb=1, projection_cache_mb=1)
    results = {}
    for granularity in ('batch', 'row'):
        entry['options']['raw_cache_granularity'] = granularity
        write_json(tmp_path / 'manifest.json', manifest)
        cli.main(['ultraquery', 'evaluate', '--manifests', str(tmp_path / 'manifest.json'), '--input-root', str(tmp_path),
                  '--output', str(tmp_path / granularity), '--entries', entry['id'], '--split', 'test', '--device', 'cpu'])
        results[granularity] = read(tmp_path / granularity / entry['id'] / 'result.json')
    row, batch = results['row'], results['batch']
    assert row['inference']['options']['raw_cache_granularity'] == 'row'
    for shape, values in batch['per_shape'].items():
        assert row['per_shape'][shape]['mrr'] == pytest.approx(values['mrr'], abs=1e-6)
    statistics = row['inference']['cache_statistics'], batch['inference']['cache_statistics']
    assert statistics[0]['backbone_computed_rows'] <= statistics[1]['backbone_computed_rows']


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
