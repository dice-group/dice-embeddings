"""Evaluation protocol: ties, resumable progress, rank traces, inference graphs and answer filters."""

import itertools
import json
import pickle
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from dicee.query_answering import BenchmarkQuery, QueryBenchmark, QueryContext
from dicee.query_answering._checkpoint import BenchmarkCheckpoint, write_json
from dicee.query_answering._query import PLUS_H_SHAPES, QUERY_SHAPES
from dicee.query_answering.benchmark import QueryMetrics, evaluate_benchmark
from dicee.query_answering.datasets import dataset_spec
from dicee.query_answering.methods import ConE


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


def read(path):
    return json.loads(Path(path).read_text())


def trace_rows(path):
    """(query ID, query type, [entity, rank, greater, tie size] per hard answer) of a rank trace, in query order."""
    with sqlite3.connect(path) as connection:
        return [(query_id, shape, json.loads(answers)) for query_id, shape, answers in
                connection.execute('SELECT query_id, shape, answers FROM queries ORDER BY position')]


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
    rows = trace_rows(tmp_path / 'ranks.sqlite3')
    assert len(rows) == 6 and len({query_id for query_id, _, _ in rows}) == 6
    # The trace holds the sort rank of every hard answer: it reproduces the reported metric.
    traced = np.mean([np.mean([1 / rank for _, rank, _, _ in answers]) for _, _, answers in rows])
    assert traced == pytest.approx(resumed['per_shape']['1p']['mrr'])


def fixture_manifest(tmp_path):
    """A one-entry +H manifest below ``tmp_path``: a random ConE checkpoint on a tiny FB15k237+H."""
    data_root = tmp_path / 'data'
    plus_h_fixture(data_root, 'FB15k237+H')
    context = QueryContext([(0, 0, 1)], 8, 4)
    model = ConE(context, dim=8, gamma=12, center_reg=.02, projection_dim=16)
    torch.save(model.state_dict(), tmp_path / 'cone.pt')
    entry = dict(id='cone-fixture', method='cone', dataset='FB15k237+H', checkpoint='cone.pt',
                 options={'center_reg': .02}, query_types=list(PLUS_H_SHAPES), query_batch_size=1,
                 query_order='upstream-pickle', reference={'status': 'test fixture'})
    return dict(version=1, data_root='data', seed=0, threads=2, archives={}, entries=[entry])


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


def test_direct_evaluation_rejects_supplied_wrong_graph(tmp_path):
    from dicee.query_answering.datasets import load_benchmark
    from dicee.query_answering.method_evaluation import evaluate_method
    fixture_manifest(tmp_path)
    data = load_benchmark(tmp_path / 'data', 'FB15k237+H', inference_graph='train+valid')
    with pytest.raises(ValueError, match='requested inference graph'):
        evaluate_method(dict(method='cone', dataset='FB15k237+H', inference_graph='train'),
                        output=tmp_path / 'result', data=data)


def test_baseline_filter_control_keeps_inference_and_graph_metadata(tmp_path):
    from dicee.query_answering.datasets import load_benchmark
    from dicee.query_answering.method_evaluation import evaluate_method
    manifest = fixture_manifest(tmp_path)
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


def test_ultraquery_recipes_cover_their_14_types_directly_and_through_the_cli(tmp_path):
    from benchmarks.cqa.cli import main
    from dicee.query_answering.method_evaluation import evaluate_method
    manifest = fixture_manifest(tmp_path)
    use_ultraquery_fixture(manifest, tmp_path)
    entry = manifest['entries'][0]
    recipe = dict(method='cone', dataset=entry['dataset'], vocabulary_dataset=entry['dataset'], root=str(tmp_path / manifest['data_root']),
                  checkpoint=str(tmp_path / entry['checkpoint']), options=entry['options'], query_types=entry['query_types'])
    result = evaluate_method(recipe, output=tmp_path / 'direct', split='test')
    assert result['queries'] == 14 and result['protocol']['all_benchmark_shapes']
    assert result['benchmark_run']['mode'] == 'direct'
    write_json(tmp_path / 'manifest.json', manifest)
    main(['ultraquery', 'evaluate', '--split', 'test', '--manifests', str(tmp_path / 'manifest.json'), '--input-root', str(tmp_path),
          '--output', str(tmp_path / 'cli'), '--entries', entry['id'], '--device', 'cpu'])
    cli = read(tmp_path / 'cli' / entry['id'] / 'result.json')
    assert cli['per_shape'] == result['per_shape']
    assert cli['additional_tie_metrics'] == result['additional_tie_metrics']


@pytest.mark.parametrize('query_order', ('canonical', 'relation', 'upstream-pickle'))
def test_corrected_filter_cli_preserves_query_type_selection(tmp_path, query_order):
    from benchmarks.cqa.cli import main
    manifest = fixture_manifest(tmp_path)
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


def test_sampled_queries_keep_their_complete_encoding_batches(tmp_path, monkeypatch):
    """A method whose scores depend on the batch scores a sampled query as in the full run."""
    from dicee.query_answering.method_evaluation import evaluate_method
    manifest = fixture_manifest(tmp_path)
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
    entry = manifest['entries'][0]
    recipe = dict(method='cone', checkpoint=str(tmp_path / entry['checkpoint']), root=str(tmp_path / 'data'), dataset=entry['dataset'],
                  vocabulary_dataset=entry['dataset'], options=entry['options'], query_types=entry['query_types'], query_batch_size=4)
    full, sampled = {}, {}
    evaluate_method(recipe, output=tmp_path / 'full', split='valid', query_order='upstream-pickle',
                    on_prediction=lambda query, scores: full.update({query: scores.clone()}))
    full_calls = calls.copy()
    calls.clear()
    report = evaluate_method(recipe, output=tmp_path / 'sampled', split='valid', limit=1, query_order='upstream-pickle',
                             on_prediction=lambda query, scores: sampled.update({query: scores.clone()}))
    assert len(full) == 17 and len(sampled) == report['queries'] == 16
    # The two 1p queries share an encoding batch in both runs, although only one of them is sampled.
    assert calls == full_calls and max(map(len, calls)) == 2
    assert all(torch.equal(scores, full[query]) for query, scores in sampled.items())


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
