"""Paired calibration controls, shared logits, and crash-safe comparison traces."""

import subprocess
import sys

import pytest
import torch

from benchmarks.cqa import cli
from benchmarks.cqa.manifests import REPO, suite_directory
from benchmarks.cqa.reports import adapter_effects, export_reports, trace_queries
from benchmarks.cqa.study import freeze, integration_passed, read, run_job, verify_bundle, write_json
from benchmarks.cqa.tests.test_protocol import paper_fixture, small_data, use_ultraquery_fixture
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
    left, right = tmp_path / 'ranks.sqlite3', tmp_path / 'without-adapter/ranks.sqlite3'
    assert len(list(trace_queries(left))) == len(list(trace_queries(right))) == 6
    effects = adapter_effects(left, right, bootstrap_samples=10)
    assert effects['macro']['expected']['mrr'] == pytest.approx(.75)
    assert effects['macro']['expected']['mrr_ci95'] == pytest.approx([.75, .75])
    assert adapter_effects(left, left, bootstrap_samples=0)['macro']['sort']['mrr'] == 0


@pytest.mark.parametrize('backbone', ['ultra', 'trix', 'kgicl'])
@pytest.mark.parametrize('facts,ultraquery', [(None, False), ('none', False), ('atomic', False), ('both', False), ('both', True)])
def test_paired_kgfm_verification_full_run_and_reports(tmp_path, backbone, facts, ultraquery):
    from dicee.models import KGICL, TRIX, ULTRA
    from dicee.query_answering.score_adapter import QueryScoreAdapter
    manifest = paper_fixture(tmp_path)
    if ultraquery:
        use_ultraquery_fixture(manifest, tmp_path)
    count = len(manifest['entries'][0]['query_types'])
    model = {'ultra': ULTRA, 'trix': TRIX, 'kgicl': KGICL}[backbone](dict(num_entities=1, num_relations=1))
    # KG-ICL checkpoints keep their weights under 'state_dict'.
    torch.save({'state_dict' if backbone == 'kgicl' else 'model': model.state_dict()}, tmp_path / 'kgfm.pt')
    QueryScoreAdapter('context_scores', 1., bias_bound=8.,
                      weights=torch.randn(2, 8, generator=torch.Generator().manual_seed(4)).double() / 5,
                      metadata={'backbone_state_sha256': state_fingerprint(model)}).save(tmp_path / 'adapter.json')
    entry = manifest['entries'][0]
    entry.update(id=f'{backbone}-paired', method=f'{backbone}-adapter', checkpoint='kgfm.pt',
                 options={'beam_size': 2, 'row_batch_size': 1, 'backend_batch_size': 1, 'cache_bytes': 4096, 'raw_cache_bytes': 4096},
                 adapters={'product': 'adapter.json'}, operators={shape: 'product' for shape in entry['query_types']},
                 selection_protocol='source-validation', finalized=True, adapter_ablation=True,
                 query_batch_size=4, query_order='relation')
    if facts is not None:
        entry['options']['observed_facts'] = facts
    if ultraquery:
        entry['options'].update(raw_cache_device='model', relation_cache_mb=1, projection_cache_mb=1)
    freeze(manifest, tmp_path / 'bundle', tmp_path)
    subprocess.run([sys.executable, '-m', 'benchmarks.cqa', 'plus_h', 'integration', '--bundle', str(tmp_path / 'bundle'),
                    '--input-root', str(tmp_path), '--entry', entry['id'], '--output', str(tmp_path / 'verification'),
                    '--device', 'cpu', '--probes', '1'], check=True, cwd=REPO)
    evidence = read(tmp_path / 'verification/integration.json')
    assert evidence['passed'] and len(evidence['queries']) == count * 2
    entry['verification'] = 'verification/integration.json'
    freeze(manifest, tmp_path / 'verified-bundle', tmp_path)
    result = run_job(tmp_path / 'verified-bundle', entry['id'], tmp_path, tmp_path / 'test', phase='test')
    control = read(tmp_path / 'test/without-adapter/result.json')
    assert result['queries'] == control['queries'] == count
    assert result['inference']['cache_statistics']['backbone_reused_rows'] > 0
    assert control['paired_with'] == entry['id']
    assert control['inference']['calibration'] == 'without-adapter'
    settings = result['inference']['observed_facts']
    assert settings == control['inference']['observed_facts']
    assert settings['atomic_mix'] == {'product': float(facts != 'none')}
    assert settings['restore_answers'] == (facts in (None, 'both'))
    assert read(tmp_path / 'adapter.json')['observed_mix'] == 1
    assert run_job(tmp_path / 'verified-bundle', entry['id'], tmp_path, tmp_path / 'test', phase='test') == result
    rows = export_reports(tmp_path / 'test', suite_directory('plus_h') / 'published_results.json', tmp_path / 'reports', bootstrap_samples=10)
    assert len(rows) == 2 and all(row['complete_test'] for row in rows)
    assert all(row['complete_16_type_test'] == (not ultraquery) for row in rows)
    assert all(row['observed_facts'] == settings for row in rows)
    assert len(read(tmp_path / 'reports/adapter-effects.json')) == 1
    if ultraquery:
        write_json(tmp_path / 'manifest.json', manifest)
        args = ['ultraquery', 'evaluate', '--manifests', str(tmp_path / 'manifest.json'), '--input-root', str(tmp_path),
                '--output', str(tmp_path / 'cli'), '--entries', entry['id'], '--split', 'test', '--device', 'cpu']
        cli.main(args)
        direct = read(tmp_path / 'cli' / entry['id'] / 'result.json')
        direct_control = read(tmp_path / 'cli' / entry['id'] / 'without-adapter/result.json')
        assert direct['per_shape'] == result['per_shape']
        assert direct_control['per_shape'] == control['per_shape']
        cli.main(args)
        assert read(tmp_path / 'cli' / entry['id'] / 'result.json')['per_shape'] == direct['per_shape']
    (tmp_path / 'test/without-adapter/result.json').write_text('{}')
    with pytest.raises(ValueError, match='comparison checksum'):
        run_job(tmp_path / 'verified-bundle', entry['id'], tmp_path, tmp_path / 'test', phase='test')


def test_verify_reuses_passing_integration_checks_and_rejects_partial_ones(tmp_path, monkeypatch, capsys):
    from dicee.models import ULTRA
    from dicee.query_answering.score_adapter import QueryScoreAdapter
    manifest = paper_fixture(tmp_path)
    model = ULTRA(dict(num_entities=1, num_relations=1))
    torch.save({'model': model.state_dict()}, tmp_path / 'kgfm.pt')
    QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=torch.full((2, 8), .1, dtype=torch.float64),
                      metadata={'backbone_state_sha256': state_fingerprint(model)}).save(tmp_path / 'adapter.json')
    entry = manifest['entries'][0]
    entry.update(id='ultra-paired', method='ultra-adapter', checkpoint='kgfm.pt',
                 options={'beam_size': 2, 'row_batch_size': 1, 'backend_batch_size': 1, 'cache_bytes': 4096, 'raw_cache_bytes': 4096},
                 adapters={'product': 'adapter.json'}, operators={shape: 'product' for shape in entry['query_types']},
                 selection_protocol='source-validation', finalized=True, adapter_ablation=True,
                 query_batch_size=4, query_order='relation')
    manifest['entries'] = [entry]
    study = tmp_path / 'study'
    freeze(manifest, study / 'bundle', tmp_path)
    target = study / 'verification' / entry['id']
    # The check verify would run, launched as a separate job.
    subprocess.run([sys.executable, '-m', 'benchmarks.cqa', 'plus_h', 'integration', '--bundle', str(study / 'bundle'),
                    '--input-root', str(tmp_path), '--entry', entry['id'], '--output', str(target), '--device', 'cpu'],
                   check=True, cwd=REPO)
    bundle = verify_bundle(study / 'bundle', tmp_path)
    assert integration_passed(target / 'integration.json', bundle, entry)
    assert not integration_passed(target / 'integration.json', bundle, dict(entry, options=dict(entry['options'], beam_size=3)))
    args = ['plus_h', 'verify', '--input-root', str(tmp_path), '--output', str(study), '--device', 'cpu']
    (target / 'integration.json').rename(tmp_path / 'integration.json')
    with pytest.raises(SystemExit):
        cli.main(args)
    assert 'KGFM verification requires a fresh directory' in capsys.readouterr().err
    (tmp_path / 'integration.json').rename(target / 'integration.json')
    monkeypatch.setattr(cli.subprocess, 'run', lambda command: pytest.fail('A passing check was run again'))
    cli.main(args)
    assert read(study / 'verified-manifest.json')['entries'][0]['verification'] == 'study/evidence/ultra-paired.json'
    assert read(study / 'evidence/ultra-paired.json') == read(target / 'integration.json')


def test_graph_effects_pair_calibrations_and_reject_changed_recipes(tmp_path):
    from dataclasses import replace

    from dicee.query_answering.method_evaluation import save_results
    results = tmp_path / 'results'
    for graph, scores in [('train', [10., 1., 2., 3., 4., 5.]), ('train+valid', [10., 8., 7., 6., 5., 4.])]:
        data = replace(small_data(), name='FB15k237+H', split='test',
                       metadata=dict(inference_graph=graph, expected_query_types=['1p']))
        destination = results / graph
        report = evaluate_benchmark(data, lambda query: torch.tensor(scores),
                    comparison_predictors={'without-adapter': lambda query: torch.zeros(6)},
                    checkpoint_dir=destination / 'progress', checkpoint_identity={'fixture': graph},
                    rank_trace_path=destination / 'ranks.sqlite3', additional_tie_policies=('expected',))
        report['inference'] = dict(method='ultra-adapter', calibration='learned')
        report['comparisons']['without-adapter']['inference'] = dict(method='ultra-adapter', calibration='without-adapter')
        save_results(report, destination, run=dict(entry=graph, phase='test'),
                     metadata=dict(reference={}, graph_ablation='paired', graph_recipe_sha256='same'))
    export_reports(results, None, tmp_path / 'reports', bootstrap_samples=10)
    effects = {row['calibration']: row for row in read(tmp_path / 'reports/graph-effects.json')}
    assert len(read(tmp_path / 'reports/adapter-effects.json')) == 2
    assert effects['learned']['macro']['expected']['mrr'] == pytest.approx(.75)
    assert effects['without-adapter']['macro']['expected']['mrr'] == 0
    target = results / 'train+valid/result.json'
    value = read(target)
    value['graph_recipe_sha256'] = 'changed'
    write_json(target, value)
    with pytest.raises(ValueError, match='Graph ablation changed'):
        export_reports(results, None, tmp_path / 'invalid', bootstrap_samples=0)


def test_parallel_reports_are_byte_identical_to_sequential_reports(tmp_path):
    from dataclasses import replace

    from dicee.query_answering.method_evaluation import save_results

    def scores(offset):
        return lambda query: torch.rand(6, generator=torch.Generator().manual_seed(17 * query[0] + offset))

    results = tmp_path / 'results'
    for offset, graph in enumerate(('train', 'train+valid')):
        data = replace(small_data(), name='FB15k237LogicalQuery', split='test',
                       metadata=dict(inference_graph=graph, expected_query_types=['1p']))
        destination = results / graph
        report = evaluate_benchmark(data, scores(offset), comparison_predictors={'without-adapter': scores(5)},
                                    checkpoint_dir=destination / 'progress', checkpoint_identity={'fixture': graph},
                                    rank_trace_path=destination / 'ranks.sqlite3', additional_tie_policies=('expected',))
        report['inference'] = dict(method='ultra-adapter', calibration='learned')
        report['comparisons']['without-adapter']['inference'] = dict(method='ultra-adapter', calibration='without-adapter')
        save_results(report, destination, run=dict(entry=graph, phase='test'),
                     metadata=dict(reference={}, graph_ablation='paired', graph_recipe_sha256='same'))
    export_reports(results, None, tmp_path / 'sequential', bootstrap_samples=50)
    export_reports(results, None, tmp_path / 'parallel', bootstrap_samples=50, workers=3)
    names = sorted(path.name for path in (tmp_path / 'sequential').iterdir())
    assert names == sorted(path.name for path in (tmp_path / 'parallel').iterdir())
    assert all((tmp_path / 'sequential' / name).read_bytes() == (tmp_path / 'parallel' / name).read_bytes() for name in names)
    suite = read(tmp_path / 'parallel/suite-effects.json')
    assert len(read(tmp_path / 'parallel/adapter-effects.json')) == 2 and len(suite) == 2
    low, high = suite[0]['macro']['sort']['all']['ci95']
    assert low < high


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
    rows = export_reports(tmp_path, None, tmp_path / 'reports', bootstrap_samples=10)
    assert len(rows) == 4
    assert len(read(tmp_path / 'reports/adapter-effects.json')) == 2
    filters = read(tmp_path / 'reports/filter-effects.json')
    assert len(filters) == 2
    assert all(row['macro']['expected']['mrr'] == -.5 for row in filters)
    for name in ('', 'without-adapter', 'released-filters', 'without-adapter-released-filters'):
        assert len(list(trace_queries(tmp_path / name / 'ranks.sqlite3'))) == 6
    options['filter_corrections'] = {}
    with pytest.raises(ValueError, match='inputs/settings changed'):
        evaluate_benchmark(data, predict, **options)


@pytest.mark.parametrize('backbone', ['ultra', 'trix', 'kgicl'])
def test_row_granular_raw_cache_reaches_the_engine_and_keeps_scores(tmp_path, backbone):
    from dicee.models import KGICL, TRIX, ULTRA
    from dicee.query_answering.score_adapter import QueryScoreAdapter
    manifest = paper_fixture(tmp_path)
    use_ultraquery_fixture(manifest, tmp_path)
    model = {'ultra': ULTRA, 'trix': TRIX, 'kgicl': KGICL}[backbone](dict(num_entities=1, num_relations=1))
    # KG-ICL checkpoints keep their weights under 'state_dict'.
    torch.save({'state_dict' if backbone == 'kgicl' else 'model': model.state_dict()}, tmp_path / 'kgfm.pt')
    QueryScoreAdapter('context_scores', 1., bias_bound=8.,
                      weights=torch.randn(2, 8, generator=torch.Generator().manual_seed(4)).double() / 5,
                      metadata={'backbone_state_sha256': state_fingerprint(model)}).save(tmp_path / 'adapter.json')
    entry = manifest['entries'][0]
    entry.update(id=f'{backbone}-paired', method=f'{backbone}-adapter', checkpoint='kgfm.pt',
                 options={'beam_size': 2, 'row_batch_size': 2, 'backend_batch_size': 2, 'cache_bytes': 4096,
                          'raw_cache_bytes': 1 << 20, 'raw_cache_device': 'model', 'relation_cache_mb': 1, 'projection_cache_mb': 1},
                 adapters={'product': 'adapter.json'}, operators={shape: 'product' for shape in entry['query_types']},
                 selection_protocol='source-validation', finalized=True, adapter_ablation=True,
                 query_batch_size=4, query_order='relation')
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
