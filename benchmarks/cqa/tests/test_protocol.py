"""Paper protocol integrity, batch replay, and inference-free tie analysis."""

import itertools
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cqa.oracles import verify_predictions
from benchmarks.cqa.reports import export_reports, summarize_trace, trace_queries
from benchmarks.cqa.study import freeze, read, run_job, verify_bundle
from dicee.query_answering import BenchmarkQuery, QueryBenchmark, QueryContext
from dicee.query_answering._checkpoint import BenchmarkCheckpoint, write_json
from dicee.query_answering._query import PLUS_H_SHAPES, QUERY_SHAPES
from dicee.query_answering.benchmark import QueryMetrics, evaluate_benchmark
from dicee.query_answering.context import state_fingerprint
from dicee.query_answering.datasets import dataset_spec
from dicee.query_answering.methods import ConE
from dicee.query_answering.score_adapter import QueryScoreAdapter


def plus_h_fixture(root, name):
    folder = root / dataset_spec(name)[1]
    folder.mkdir(parents=True)
    graph = folder / 'KG_splits' if name == 'ICEWS18+H' else folder
    graph.mkdir(exist_ok=True)
    for split, triples in [('train', [(0, 0, 1), (1, 2, 2)]), ('valid', [(0, 0, 3), (3, 2, 4)]),
                            ('test', [(0, 0, 5), (5, 2, 6)])]:
        inverse = [(t, r + 1, h) for h, r, t in triples]
        (graph / f'{split}.txt').write_text(''.join(f'{h}\t{r}\t{t}\n' for h, r, t in triples + inverse))
    def dump(name, value):
        (folder / name).write_bytes(pickle.dumps(value))
    dump('id2ent.pkl', {i: str(i) for i in range(8)})
    dump('id2rel.pkl', {i: str(i) for i in range(4)})
    def instantiate(shape, counts):
        if shape in ('n', 'u'):
            return -2 if shape == 'n' else -1
        if shape in ('e', 'r'):
            index = 0 if shape == 'e' else 1
            value = counts[index]
            counts[index] += 1
            return value % (8 if index == 0 else 4)
        return tuple(instantiate(part, counts) for part in shape)
    queries = {QUERY_SHAPES[name]: {instantiate(QUERY_SHAPES[name], [0, 0])} for name in PLUS_H_SHAPES}
    for split, hard in [('valid', {3, 4}), ('test', {5, 6})]:
        dump(f'{split}-queries.pkl', queries)
        dump(f'{split}-easy-answers.pkl', {q: {1, 7} for group in queries.values() for q in group})
        dump(f'{split}-hard-answers.pkl', {q: hard for group in queries.values() for q in group})


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_data():
    queries = tuple(BenchmarkQuery('1p', (i, (0,)), frozenset({0}), frozenset({1, 2})) for i in range(6))
    return QueryBenchmark('test', 'transductive', 'valid', QueryContext([(0, 0, 1)], 6, 1), queries, tuple(range(6)))


def test_resume_preserves_completed_cache_report_and_resets_live_cache(tmp_path):
    data, stats = small_data(), {}
    options = dict(checkpoint_dir=tmp_path / 'resume', checkpoint_identity='fixed', checkpoint_every=3)
    calls = 0

    def predict(query):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError('interrupted')
        stats['raw_rows'] = stats.get('raw_rows', 0) + 1
        stats.update(cache_bytes=256, cache_peak_bytes=256)
        return torch.arange(6).float()

    with pytest.raises(RuntimeError, match='interrupted'):
        evaluate_benchmark(data, predict, statistics=stats, **options)
    stats, observations = {}, []

    def resumed_predict(query):
        stats['raw_rows'] += 1
        stats['cache_bytes'] = 128
        return torch.arange(6).float()

    report = evaluate_benchmark(data, resumed_predict, statistics=stats,
                                progress=lambda *_: observations.append(stats['cache_bytes']), **options)
    assert observations[0] == 0
    assert stats == dict(raw_rows=6, cache_bytes=128, cache_peak_bytes=256)
    replayed_stats = {}
    replayed = evaluate_benchmark(data, lambda _: pytest.fail('Completed queries were rescored'),
                                  statistics=replayed_stats, **options)
    assert replayed['per_shape'] == report['per_shape']
    assert replayed_stats == stats


def test_trix_completed_resume_preserves_cache_statistics(tmp_path, monkeypatch):
    from dicee.models import TRIX
    from dicee.query_answering import QueryAnswerer, benchmark_model

    model = TRIX(dict(num_entities=1, num_relations=1, trix_dim=8))
    options = dict(beam_size=2, row_batch_size=4, checkpoint_dir=tmp_path / 'trix', checkpoint_every=3)
    report = benchmark_model(model, small_data(), **options)
    assert report['inference']['statistics']['cache_bytes'] > 0
    monkeypatch.setattr(QueryAnswerer, 'predict', lambda *a, **k: pytest.fail('Completed queries were rescored'))
    resumed = benchmark_model(model, small_data(), **options)
    assert resumed['per_shape'] == report['per_shape']
    assert resumed['inference']['statistics'] == report['inference']['statistics']
    assert model.query_batch_size == 8 and model.training and model.graph_triples is None


def test_kgfm_determinism_is_strict_and_restores_caller_state():
    from dicee.query_answering.method_evaluation import deterministic_kgfm
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        with pytest.raises(RuntimeError, match='probe'):
            with deterministic_kgfm():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                raise RuntimeError('probe')
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warning)


def test_one_sort_and_exact_uniform_tie_expectation(monkeypatch):
    calls = 0
    original = torch.Tensor.argsort

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'argsort', counted)
    scores = torch.tensor([.8, .5, .5, .5, .2, .5])
    metrics, records = QueryMetrics(6, candidates=[0, 1, 2, 3, 4]).evaluate(scores, {0}, {1, 2}, records=True)
    assert calls == 1
    assert [row[2:] for row in records] == [[0, 2], [0, 2]]
    permutations = list(itertools.permutations([1, 2, 3]))
    rr = []
    for order in permutations:
        rr.append(np.mean([1 / (1 + sum(x == 3 for x in order[:order.index(target)])) for target in (1, 2)]))
    assert metrics['expected']['mrr'] == pytest.approx(np.mean(rr))
    assert metrics['expected']['mrr'] == .75


def test_trace_resume_replays_whole_batch_and_truncates_uncheckpointed_rows(tmp_path, monkeypatch):
    data, prepared, seen = small_data(), {}, []

    def prepare(queries):
        seen.append(tuple(queries))
        values = torch.arange(6).float().remainder(1 + len(queries))
        prepared.update({query: values for query in queries})

    options = dict(query_plan=data.queries, batch_ends=[3, 6], prepare=prepare, checkpoint_every=2,
                   checkpoint_dir=tmp_path / 'progress', checkpoint_identity={'model': 'batch-dependent'},
                   rank_trace_path=tmp_path / 'ranks.sqlite3', additional_tie_policies=('expected',))
    original = BenchmarkCheckpoint.save

    def fail_after_trace_commit(self, state):
        if state['completed'] == 6:
            raise RuntimeError('power loss')
        original(self, state)

    monkeypatch.setattr(BenchmarkCheckpoint, 'save', fail_after_trace_commit)
    with pytest.raises(RuntimeError, match='power loss'):
        evaluate_benchmark(data, lambda query: prepared[query], **options)
    saved = read(tmp_path / 'progress' / 'state.json')['state']
    assert saved['completed'] == 3
    with sqlite3.connect(tmp_path / 'ranks.sqlite3') as connection:
        assert connection.execute('SELECT COUNT(*) FROM queries').fetchone()[0] == 6
    monkeypatch.setattr(BenchmarkCheckpoint, 'save', original)
    resumed = evaluate_benchmark(data, lambda query: prepared[query], **options)
    assert [len(batch) for batch in seen] == [3, 3, 3]
    assert seen[1] == seen[2]
    rows = list(trace_queries(tmp_path / 'ranks.sqlite3'))
    assert len(rows) == 6 and len({row['query_id'] for row in rows}) == 6
    summary = summarize_trace(tmp_path / 'ranks.sqlite3', bootstrap_samples=20)
    assert summary['averages']['sort']['all']['mrr'] == resumed['averages']['all']['mrr']
    assert summary['averages']['expected']['all']['mrr'] == resumed['additional_tie_metrics']['expected']['averages']['all']['mrr']


def paper_fixture(tmp_path):
    data_root = tmp_path / 'data'
    plus_h_fixture(data_root, 'FB15k237+H')
    context = QueryContext([(0, 0, 1)], 8, 4)
    model = ConE(context, dim=8, gamma=12, center_reg=.02, projection_dim=16)
    torch.save(model.state_dict(), tmp_path / 'cone.pt')
    entry = dict(id='cone-fixture', method='cone', dataset='FB15k237+H', checkpoint='cone.pt',
                 options={'center_reg': .02}, query_types=list(PLUS_H_SHAPES), query_batch_size=1,
                 query_order='upstream-pickle', reference={'status': 'test fixture'}, blockers=[])
    manifest = dict(version=1, data_root='data', seed=0, threads=2, archives={}, entries=[entry])
    return manifest


@pytest.mark.parametrize('name', ('FB15k237+H', 'NELL995+H', 'ICEWS18+H'))
def test_graph_conditions_keep_labels_fixed_and_never_read_test_edges(tmp_path, name):
    from dicee.query_answering.datasets import load_benchmark
    plus_h_fixture(tmp_path, name)
    train = load_benchmark(tmp_path, name, inference_graph='train')
    extended = load_benchmark(tmp_path, name, inference_graph='train+valid')
    assert load_benchmark(tmp_path, name).context == extended.context
    assert extended.metadata['inference_graph'] == 'train+valid'
    assert set(train.context.triples) < set(extended.context.triples)
    assert (0, 0, 3) not in train.context.triples and (0, 0, 3) in extended.context.triples
    assert train.queries == extended.queries
    assert [q.identity for q in train.queries] == [q.identity for q in extended.queries]
    assert 'valid.txt' not in {Path(p).name for p in train.metadata['files']}
    path = tmp_path / dataset_spec(name)[1]
    graph = path / 'KG_splits' if name == 'ICEWS18+H' else path
    (graph / 'test.txt').write_text('unreadable target triples')
    assert load_benchmark(tmp_path, name, inference_graph='train+valid').context == extended.context
    assert load_benchmark(tmp_path, name, split='valid', inference_graph='train+valid').context == train.context
    assert load_benchmark(tmp_path, name, split='valid').context == train.context


def test_graph_conditions_pin_distinct_contexts_and_identical_query_plans(tmp_path):
    from benchmarks.cqa.study import dataset_key
    manifest = paper_fixture(tmp_path)
    entry = manifest['entries'][0]
    entry.update(inference_graph='train', graph_ablation='paired')
    manifest['entries'].append(dict(entry, id='extended', inference_graph='train+valid'))
    bundle = freeze(manifest, tmp_path / 'bundle', tmp_path)
    other = manifest['entries'][1]
    assert bundle['datasets'][dataset_key(entry, 'test')]['context'] != bundle['datasets'][dataset_key(other, 'test')]['context']
    assert dataset_key(entry, 'valid') == dataset_key(other, 'valid')
    for split in ('valid', 'test'):
        assert bundle['plans'][f'{entry["id"]}/{split}'] == bundle['plans'][f'{other["id"]}/{split}']
    assert len(bundle['datasets']) == 3


def test_implicit_test_graph_matches_explicit_train_valid_bundle(tmp_path):
    from benchmarks.cqa.study import dataset_key
    manifest = paper_fixture(tmp_path)
    entry = manifest['entries'][0]
    explicit = dict(entry, id='explicit', inference_graph='train+valid')
    manifest['entries'].append(explicit)
    bundle = freeze(manifest, tmp_path / 'bundle', tmp_path)
    assert dataset_key(entry, 'test') == dataset_key(explicit, 'test') == 'FB15k237+H/test/train+valid'
    assert len(bundle['datasets']) == 2
    assert bundle['datasets'][dataset_key(entry, 'test')]['metadata']['inference_graph'] == 'train+valid'
    assert bundle['plans'][f'{entry["id"]}/test'] == bundle['plans'][f'{explicit["id"]}/test']


def test_direct_evaluation_rejects_supplied_wrong_graph(tmp_path):
    from dicee.query_answering.datasets import load_benchmark
    from dicee.query_answering.method_evaluation import evaluate_method
    paper_fixture(tmp_path)
    data = load_benchmark(tmp_path / 'data', 'FB15k237+H', inference_graph='train+valid')
    with pytest.raises(ValueError, match='requested inference graph'):
        evaluate_method(dict(method='cone', dataset='FB15k237+H', inference_graph='train'),
                        output=tmp_path / 'result', data=data)


def test_baseline_filter_control_keeps_inference_and_graph_metadata(tmp_path):
    from dicee.query_answering.datasets import load_benchmark
    from dicee.query_answering.method_evaluation import evaluate_method
    manifest = paper_fixture(tmp_path)
    data = load_benchmark(tmp_path / 'data', 'FB15k237+H', query_types=['2in'], inference_graph='train')
    entry = manifest['entries'][0]
    recipe = dict(method='cone', dataset=entry['dataset'], vocabulary_dataset=entry['dataset'],
                  root=str(tmp_path / 'data'), checkpoint=str(tmp_path / entry['checkpoint']),
                  options=entry['options'], query_types=['2in'], inference_graph='train')
    report = evaluate_method(recipe, output=tmp_path / 'evaluation', data=data, filter_corrections={})
    control = read(tmp_path / 'evaluation/released-filters/result.json')
    assert report['inference'] == control['inference']
    assert report['per_shape'] == control['per_shape']
    assert report['protocol']['answer_filter'] == 'corrected'
    assert control['protocol']['answer_filter'] == 'released'
    assert 'paired_with' not in control
    assert control['filter_paired_with'] == report['benchmark_run']['entry']


def test_filter_audit_preserves_targets_and_keeps_test_facts_out_of_context(tmp_path):
    from dataclasses import replace

    from dicee.query_answering.datasets import audit_plus_h_filters
    plus_h_fixture(tmp_path, 'FB15k237+H')
    query = BenchmarkQuery('2in', ((0, (0,)), (1, (2, -2))), frozenset({1, 2, 3}), frozenset({5}))
    other = BenchmarkQuery('3in', ((0, (0,)), (0, (0,)), (1, (2, -2))), frozenset({1, 3}), frozenset({5}))
    folder = tmp_path / dataset_spec('FB15k237+H')[1] / 'test-query-reduction/2in/all'
    folder.mkdir(parents=True)
    for kind, value in [('queries', {QUERY_SHAPES['2in']: {query.query}}),
                        ('easy-answers', {query.query: {1, 3}}), ('hard-answers', {query.query: {5}})]:
        (folder / f'test-{kind}.pkl').write_bytes(pickle.dumps(value))
    data = QueryBenchmark('FB15k237+H', 'transductive', 'test', QueryContext([(0, 0, 1)], 8, 4),
                          (query, other), tuple(range(8)))
    audit = audit_plus_h_filters(tmp_path, data)
    assert audit['changes'] == {query.identity: {'remove': [2], 'add': []}}
    assert audit['per_shape']['3in']['changed_queries'] == 0
    assert audit['per_shape']['3in']['hard_answers'] == 1
    assert data.context.triples == ((0, 0, 1),)
    invalid = replace(data, queries=(replace(query, hard=frozenset({6})),))
    with pytest.raises(ValueError, match='hard targets'):
        audit_plus_h_filters(tmp_path, invalid)
    (folder / 'test-easy-answers.pkl').write_bytes(pickle.dumps({query.query: {1, 2, 3}}))
    with pytest.raises(ValueError, match='filters disagree with full-graph truth'):
        audit_plus_h_filters(tmp_path, data)


def test_frozen_filter_corrections_reject_tampering(tmp_path, monkeypatch):
    import dicee.query_answering.datasets as datasets
    manifest = paper_fixture(tmp_path)
    manifest['answer_filter'] = 'corrected'
    monkeypatch.setattr(datasets, 'audit_plus_h_filters', lambda *args: dict(changes={}, per_shape={}, source_files={}))
    bundle = freeze(manifest, tmp_path / 'bundle', tmp_path)
    verify_bundle(tmp_path / 'bundle', tmp_path)
    correction = bundle['filter_corrections']['FB15k237+H']
    (tmp_path / 'bundle' / correction['path']).write_text('{}')
    with pytest.raises(ValueError, match='Answer-filter correction checksum'):
        verify_bundle(tmp_path / 'bundle', tmp_path)


def use_ultraquery_fixture(manifest, root):
    from dicee.query_answering._query import ULTRAQUERY_SHAPES
    data = root / manifest['data_root']
    source = data / dataset_spec('FB15k237+H')[1]
    destination = data / dataset_spec('FB15k237LogicalQuery')[1]
    shutil.copytree(source, destination)
    for split in ('valid', 'test'):
        path = destination / f'{split}-queries.pkl'
        queries = pickle.loads(path.read_bytes())
        path.write_bytes(pickle.dumps({QUERY_SHAPES[shape]: queries[QUERY_SHAPES[shape]] for shape in ULTRAQUERY_SHAPES}))
    for entry in manifest['entries']:
        entry.update(dataset='FB15k237LogicalQuery', query_types=list(ULTRAQUERY_SHAPES))


def test_shared_study_and_reports_use_dataset_coverage(tmp_path):
    from benchmarks.cqa.cli import main
    from dicee.query_answering.method_evaluation import evaluate_method
    manifest = paper_fixture(tmp_path)
    use_ultraquery_fixture(manifest, tmp_path)
    frozen = freeze(manifest, tmp_path / 'bundle', tmp_path)
    assert len(frozen['plans']) == 2
    entry = manifest['entries'][0]
    root = str(tmp_path / manifest['data_root'])
    recipe = dict(method='cone', dataset=entry['dataset'], vocabulary_dataset=entry['dataset'], root=root,
                  checkpoint=str(tmp_path / entry['checkpoint']), options=entry['options'], query_types=entry['query_types'])
    # A direct evaluation can be reported without requiring a +H recipe.
    result = evaluate_method(recipe, output=tmp_path / 'direct', split='test')
    assert result['queries'] == 14 and result['protocol']['all_benchmark_shapes']
    pilot = run_job(tmp_path / 'bundle', entry['id'], tmp_path, tmp_path / 'pilot')
    assert pilot['queries'] == 14 and pilot['coverage']['complete_benchmark_types']
    assert not pilot['coverage']['complete_16_types']
    assert frozen['datasets']['FB15k237LogicalQuery/valid']['context'] == frozen['datasets']['FB15k237LogicalQuery/test']['context']
    assert result['benchmark_run']['mode'] == 'direct'
    published = tmp_path / 'published.json'
    write_json(published, {'source': 'fixture', 'datasets': {entry['dataset']: {'cone': dict.fromkeys(entry['query_types'], 10.)}}})
    rows = export_reports(tmp_path / 'direct', published, tmp_path / 'reports', bootstrap_samples=0)
    assert rows[0]['complete_test'] and not rows[0]['complete_16_type_test']
    assert rows[0]['types'] == 14 and rows[0]['published_mrr'] == 10.
    write_json(tmp_path / 'manifest.json', manifest)
    main(['ultraquery', 'evaluate', '--split', 'test', '--manifests', str(tmp_path / 'manifest.json'), '--input-root', str(tmp_path),
          '--output', str(tmp_path / 'cli'), '--entries', entry['id'], '--device', 'cpu'])
    cli = read(tmp_path / 'cli' / entry['id'] / 'result.json')
    assert cli['per_shape'] == result['per_shape']
    assert cli['additional_tie_metrics'] == result['additional_tie_metrics']


@pytest.mark.parametrize('query_order', ('canonical', 'relation', 'upstream-pickle'))
def test_corrected_filter_cli_preserves_query_type_selection(tmp_path, query_order):
    from benchmarks.cqa.cli import main
    manifest = paper_fixture(tmp_path)
    manifest['answer_filter'] = 'corrected'
    entry = manifest['entries'][0]
    entry.update(query_types=['1p'], query_order=query_order)
    write_json(tmp_path / 'manifest.json', manifest)
    main(['plus_h', 'evaluate', '--split', 'test', '--manifests', str(tmp_path / 'manifest.json'), '--input-root', str(tmp_path),
          '--output', str(tmp_path / 'cli'), '--entries', entry['id'], '--device', 'cpu'])
    destination = tmp_path / 'cli' / entry['id']
    result = read(destination / 'result.json')
    control = read(destination / 'released-filters/result.json')
    assert result['queries'] == control['queries'] == 1
    assert set(result['per_shape']) == set(control['per_shape']) == {'1p'}
    assert result['per_shape'] == control['per_shape']
    assert result['protocol']['answer_filter'] == 'corrected'
    assert control['protocol']['answer_filter'] == 'released'


def test_frozen_inputs_pipeline_pilot_and_offline_report(tmp_path):
    manifest = paper_fixture(tmp_path)
    bundle = tmp_path / 'bundle'
    frozen = freeze(manifest, bundle, tmp_path)
    assert len(frozen['datasets']) == 2
    assert len(verify_bundle(bundle, tmp_path)['files']) == 11
    output = tmp_path / 'results' / 'cone-fixture'
    report = run_job(bundle, 'cone-fixture', tmp_path, output)
    assert report['queries'] == 16 and report['split'] == 'valid'
    assert report['protocol']['all_benchmark_shapes']
    assert run_job(bundle, 'cone-fixture', tmp_path, output) == report
    published = tmp_path / 'published.json'
    write_json(published, dict(source='fixture', datasets={'FB15k237+H': {'cone': dict.fromkeys(PLUS_H_SHAPES, 10.)}}))
    rows = export_reports(tmp_path / 'results', published, tmp_path / 'tables', bootstrap_samples=10)
    assert rows[0]['published_mrr'] is None
    assert not rows[0]['complete_16_type_test']
    with pytest.raises(ValueError, match='parity evidence'):
        run_job(bundle, 'cone-fixture', tmp_path, tmp_path / 'test', phase='test')
    (tmp_path / 'cone.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_bundle(bundle, tmp_path)


def test_plan_tampering_and_no_overwrite(tmp_path):
    manifest = paper_fixture(tmp_path)
    bundle = tmp_path / 'bundle'
    frozen = freeze(manifest, bundle, tmp_path)
    with pytest.raises(FileExistsError, match='new bundle'):
        freeze(manifest, bundle, tmp_path)
    plan = bundle / frozen['plans']['cone-fixture/test']['path']
    value = read(plan)
    value['queries'].reverse()
    write_json(plan, value)
    with pytest.raises(ValueError, match='plan checksum'):
        verify_bundle(bundle, tmp_path)


def test_probe_selection_preserves_full_encoding_batches(tmp_path, monkeypatch):
    manifest = paper_fixture(tmp_path)
    manifest['entries'][0]['query_batch_size'] = 4
    path = next((tmp_path / 'data').rglob('valid-queries.pkl')).parent
    query = (4, (0,))
    for filename, update in (
        ('queries', lambda value: next(group for shape, group in value.items() if shape == ('e', ('r',))).add(query)),
        ('easy-answers', lambda value: value.update({query: {1, 7}})),
        ('hard-answers', lambda value: value.update({query: {3, 4}})),
    ):
        target = path / f'valid-{filename}.pkl'
        value = pickle.loads(target.read_bytes())
        update(value)
        target.write_bytes(pickle.dumps(value))
    calls = []
    original = ConE.encode_trees

    def coupled(self, trees):
        calls.append(tuple(trees))
        return [branch + branch.mean(0, keepdim=True) for branch in original(self, trees)]

    monkeypatch.setattr(ConE, 'encode_trees', coupled)
    bundle = tmp_path / 'bundle'
    freeze(manifest, bundle, tmp_path)
    full, probes = {}, {}
    run_job(bundle, 'cone-fixture', tmp_path, tmp_path / 'full-pilot', pilot_queries=1,
            on_prediction=lambda query, scores: full.update({query.query: scores.clone()}))
    full_calls = calls.copy()
    calls.clear()
    report = run_job(bundle, 'cone-fixture', tmp_path, tmp_path / 'probe', pilot_queries=1, probe_only=True,
                     on_prediction=lambda query, scores: probes.update({query.query: scores.clone()}))
    assert len(full) == 17 and len(probes) == report['queries'] == 16
    assert calls == full_calls
    assert max(map(len, calls)) == 2
    assert all(torch.equal(score, full[query]) for query, score in probes.items())
    from dicee.query_answering.method_evaluation import evaluate_method
    calls.clear()
    direct = {}
    entry = manifest['entries'][0]
    recipe = dict(method='cone', checkpoint=str(tmp_path / entry['checkpoint']),
                  root=str(tmp_path / 'data'), dataset=entry['dataset'], vocabulary_dataset=entry['dataset'],
                  options=entry['options'], query_types=entry['query_types'], query_batch_size=4)
    report = evaluate_method(recipe, output=tmp_path / 'direct', split='valid', limit=1,
                             query_order='upstream-pickle',
                             on_prediction=lambda query, scores: direct.update({query: scores.clone()}))
    assert calls == full_calls and len(direct) == report['queries'] == 16
    assert all(torch.equal(score, full[query]) for query, score in direct.items())
    with pytest.raises(ValueError, match='validation pilots'):
        run_job(bundle, 'cone-fixture', tmp_path, tmp_path / 'test', phase='test', probe_only=True)


def test_checked_in_protocol_settings():
    from benchmarks.cqa.manifests import read_manifest, suite_directory
    root = suite_directory('plus_h')
    manifest = read_manifest(root / 'baselines.json')
    assert len(manifest['entries']) == 21
    by_id = {entry['id']: entry for entry in manifest['entries']}
    assert by_id['cqd-FB15k237+H']['options']['per_shape']['2p']['beam_size'] == 512
    assert by_id['cqd-FB15k237+H']['options']['per_shape']['pi']['beam_size'] == 256
    assert by_id['cqd-hybrid-ICEWS18+H']['options']['per_shape']['4p']['max_k'] == 64
    assert by_id['cqd-FB15k237+H']['options']['reference_batching']
    for method in ('cqd', 'cqd-hybrid'):
        assert by_id[f'{method}-FB15k237+H']['options']['per_shape']['2u']['tnorm'] == 'prod'
    assert by_id['cone-ICEWS18+H']['query_batch_size'] == 4
    assert by_id['gnnqe-NELL995+H']['query_batch_size'] == 8
    assert by_id['ultraquery-FB15k237+H']['query_batch_size'] == 32
    assert by_id['clmpt-FB15k237+H']['query_batch_size'] == 5000
    assert by_id['clmpt-NELL995+H']['query_batch_size'] == 8
    for method in ('cqd', 'cqd-hybrid'):
        for dataset in ('FB15k237+H', 'ICEWS18+H', 'NELL995+H'):
            entry = by_id[f'{method}-{dataset}']
            assert len(entry['query_types']) == 16
            assert entry['options']['atomic_negation']
    published = read(root / 'published_results.json')
    assert sum(published['datasets']['FB15k237+H']['ultraquery'].values()) / 16 == pytest.approx(9.6)


def test_rank_trace_rejects_missing_committed_answers(tmp_path):
    data = small_data()
    options = dict(checkpoint_dir=tmp_path / 'progress', checkpoint_identity={'a': 1}, rank_trace_path=tmp_path / 'ranks.sqlite3')
    evaluate_benchmark(data, lambda query: torch.zeros(6), **options)
    with sqlite3.connect(tmp_path / 'ranks.sqlite3') as connection:
        connection.execute('DELETE FROM queries WHERE position=2')
    with pytest.raises(ValueError, match='missing checkpointed'):
        evaluate_benchmark(data, lambda query: torch.zeros(6), **options)


def test_invalid_batch_boundaries(tmp_path):
    data = small_data()
    for ends in ([2, 4], [3, 3, 6], [0, 6], [6, 3]):
        with pytest.raises(ValueError, match='batch boundaries'):
            evaluate_benchmark(data, lambda query: torch.zeros(6), query_plan=data.queries, batch_ends=ends)


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('DICEE_CONE_REFERENCE'), reason='Set DICEE_CONE_REFERENCE to the pinned upstream checkout')
def test_independent_exporter_parity_and_full_run(tmp_path):
    manifest = paper_fixture(tmp_path)
    bundle = tmp_path / 'bundle'
    freeze(manifest, bundle, tmp_path)
    reference = tmp_path / 'reference.pt'
    subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'verification' / 'export_reference.py'),
                    '--bundle', str(bundle), '--input-root', str(tmp_path), '--entry', 'cone-fixture',
                    '--upstream', os.environ['DICEE_CONE_REFERENCE'], '--output', str(reference)], check=True)
    evidence = verify_predictions(bundle, 'cone-fixture', tmp_path, reference, tmp_path / 'parity.json')
    assert evidence['passed'] and len(evidence['queries']) == 16
    manifest['entries'][0]['verification'] = 'parity.json'
    freeze(manifest, tmp_path / 'verified-bundle', tmp_path)
    report = run_job(tmp_path / 'verified-bundle', 'cone-fixture', tmp_path, tmp_path / 'full', phase='test')
    assert report['protocol']['full_split'] and report['coverage']['complete_16_types']
    assert report['split'] == 'test'


@pytest.mark.parametrize('backbone', ['ultra', 'trix'])
def test_provisional_kgfm_pipeline_and_finalization_guard(tmp_path, backbone):
    from dicee.models import TRIX, ULTRA
    manifest = paper_fixture(tmp_path)
    model = {'ultra': ULTRA, 'trix': TRIX}[backbone](dict(num_entities=1, num_relations=1))
    torch.save({'model': model.state_dict()}, tmp_path / 'kgfm.pt')
    for operator in ('product', 'min'):
        adapter = QueryScoreAdapter('global', observed_mix=1, metadata=dict(
            backbone_state_sha256=state_fingerprint(model), training={} if operator == 'product' else {'tnorm': 'min'}))
        adapter.save(tmp_path / f'{operator}.json')
    entry = manifest['entries'][0]
    entry.update(id=f'{backbone}-fixture', method=f'{backbone}-adapter', checkpoint='kgfm.pt', options={'beam_size': 2, 'cache_bytes': 4096},
                 adapters={operator: f'{operator}.json' for operator in ('product', 'min')},
                 operators={shape: 'min' if 'n' in shape else 'product' for shape in PLUS_H_SHAPES},
                 selection_protocol='source-validation', finalized=False, query_batch_size=4, query_order='relation')
    freeze(manifest, tmp_path / 'bundle', tmp_path)
    result = run_job(tmp_path / 'bundle', entry['id'], tmp_path, tmp_path / 'pilot')
    assert result['queries'] == 16
    assert result['inference']['selection_protocol'] == 'source-validation'
    assert len(list(trace_queries(tmp_path / 'pilot' / 'ranks.sqlite3'))) == 16
    with pytest.raises(ValueError, match='not ready'):
        run_job(tmp_path / 'bundle', entry['id'], tmp_path, tmp_path / 'test', phase='test')


def test_comparison_does_not_relax_reference_verification(tmp_path, monkeypatch):
    import benchmarks.cqa.oracles as verification
    from benchmarks.cqa.study import entry_inputs_identity
    from dicee.query_answering.context import fingerprint
    from dicee.query_answering.datasets import load_benchmark
    from dicee.query_answering.methods import REFERENCES
    manifest = paper_fixture(tmp_path)
    entry = manifest['entries'][0]
    entry['query_types'] = ['1p']
    entry['inference_graph'] = 'train'
    bundle = freeze(manifest, tmp_path / 'bundle', tmp_path)
    query = next(q for q in load_benchmark(tmp_path / 'data', 'FB15k237+H', split='valid').queries if q.shape == '1p')
    key = query.identity
    scores = torch.arange(8).float()
    oracle = dict(version=1, entry_sha256=fingerprint(entry), inputs_sha256=entry_inputs_identity(bundle, entry),
                  validation_plan_sha256=bundle['plans'][entry['id'] + '/valid']['sha256'],
                  reference_commit=REFERENCES['cone'][1], environment=dict.fromkeys(
                      ('python', 'torch', 'cuda', 'gpu', 'driver', 'precision', 'threads'), 'fixture'),
                  graphs={split: bundle['datasets'][f'{entry["dataset"]}/{split}']['context'] for split in ('valid', 'test')},
                  pilot_queries=1, probe_only=True, scores={key: scores}, orders={key: scores.argsort(descending=True)})
    torch.save(oracle, tmp_path / 'declared.pt')
    previous = scores.clone()
    previous[3] = 20
    control = dict(oracle, scores={key: previous}, orders={key: previous.argsort(descending=True)})
    torch.save(control, tmp_path / 'dense.pt')

    def run(*args, on_prediction, **kwargs):
        on_prediction(query, scores)
        return {'benchmark_run': {'environment': oracle['environment']}}

    monkeypatch.setattr(verification, 'run_job', run)
    result = verify_predictions(tmp_path / 'bundle', entry['id'], tmp_path, tmp_path / 'declared.pt',
                                tmp_path / 'parity.json', comparison_reference=tmp_path / 'dense.pt')
    assert result['passed']
    comparison = result['upstream_comparison']
    assert not comparison['queries'][key]['ranks_match']
    assert comparison['delta_macro']['sort']['mrr'] != 0
    assert comparison['delta_macro']['expected']['mrr'] != 0
    scores = previous
    failed = verify_predictions(tmp_path / 'bundle', entry['id'], tmp_path, tmp_path / 'declared.pt',
                                tmp_path / 'failed.json', comparison_reference=tmp_path / 'dense.pt')
    assert not failed['passed']
    control['inputs_sha256'] = 'different inputs'
    torch.save(control, tmp_path / 'dense.pt')
    with pytest.raises(ValueError, match='Comparison reference uses different'):
        verify_predictions(tmp_path / 'bundle', entry['id'], tmp_path, tmp_path / 'declared.pt',
                           tmp_path / 'invalid.json', comparison_reference=tmp_path / 'dense.pt')
    oracle['graphs']['test'] = 'wrong graph'
    torch.save(oracle, tmp_path / 'wrong-graph.pt')
    with pytest.raises(ValueError, match='Reference inference graphs differ'):
        verify_predictions(tmp_path / 'bundle', entry['id'], tmp_path, tmp_path / 'wrong-graph.pt',
                           tmp_path / 'wrong-graph.json')
