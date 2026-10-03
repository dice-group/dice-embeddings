"""Answer partition, graph provenance and inference-free difficulty regressions."""
import json
import pickle
import sqlite3
from collections import defaultdict

import numpy as np
import pytest

from benchmarks.cqa.difficulty import (
    DifficultyClassifier,
    difficulty_name,
    export_difficulty_reports,
    positive_edges,
    released_labels,
    root_queries,
    summarize_difficulty,
)
from dicee.query_answering._checkpoint import checksum, write_json
from dicee.query_answering._query import QUERY_SHAPES, compile_query


def outgoing(edges):
    result = defaultdict(lambda: defaultdict(set))
    for h, r, t in edges:
        result[h][r].add(t)
    return result


def test_minimum_witness_and_graph_change_with_bounded_cache():
    full = outgoing([(0, 0, 1), (0, 0, 2), (1, 1, 3), (2, 1, 3), (2, 1, 4)])
    train = outgoing([(0, 0, 1)])
    extended = outgoing([(0, 0, 1), (0, 0, 2)])
    q = (0, (0, 1))
    classifier = DifficultyClassifier(full, train, 5, cache_answers=2)
    costs, maximum = classifier.classify('2p', q, {3, 4})
    assert costs == {3: 1, 4: 2} and maximum == 2
    assert classifier.cached_answers <= 2
    for relation in range(10, 20):
        classifier.costs(('project', relation, ('anchor', 0)))
    assert len(classifier.cache) <= 2
    assert DifficultyClassifier(full, extended, 5).classify('2p', q, {3, 4})[0] == {3: 1, 4: 1}
    assert [difficulty_name(k, maximum) for k in range(3)] == ['observed', 'partial', 'full']
    with pytest.raises(ValueError, match='not true'):
        classifier.classify('2p', q, {0})


def test_union_chooses_branch_and_negation_constrains_intermediate_witness():
    full = outgoing([(0, 0, 2), (0, 0, 3), (1, 1, 2), (2, 2, 4), (3, 2, 4)])
    train = outgoing([(0, 0, 2), (2, 2, 4)])
    classifier = DifficultyClassifier(full, train, 5)
    union = (((0, (0,)), (1, (1,)), (-1,)), (2,))
    assert classifier.classify('up', union, {4}) == ({4: 0}, 2)
    # Witness 2 is cheap but violates the negated branch; witness 3 costs two.
    negated = (((0, (0,)), (1, (1, -2))), (2,))
    assert classifier.classify('inp', negated, {4}) == ({4: 2}, 2)
    pni = ((1, (1, 2, -2)), (0, (0,)))
    assert classifier.classify('pni', pni, {2, 3}) == ({2: 0, 3: 1}, 1)
    assert positive_edges(compile_query(((0, (0,)), (1, (1,)), (-1,)))) == 1


def fixture_records(tmp_path):
    queries = [(0, (0, 1)), (1, (0, 1))]
    folder = tmp_path / 'data'
    folder.mkdir()
    for name, value in {
        'queries': {QUERY_SHAPES['2p']: set(queries)},
        'easy-answers': {q: {9} for q in queries},
        'hard-answers': {queries[0]: {3, 4}, queries[1]: {3}},
    }.items():
        (folder / f'test-{name}.pkl').write_bytes(pickle.dumps(value))
    records = root_queries(folder, ['2p'])
    trace = tmp_path / 'ranks.sqlite3'
    with sqlite3.connect(trace) as db:
        db.execute('CREATE TABLE queries(position INTEGER, query_id TEXT, shape TEXT, answers TEXT)')
        for i, q in enumerate(queries):
            key = next(k for k, r in records.items() if r['query'] == q)
            rows = [[3, 1, 0, 2], [4, 4, 3, 1]] if i == 0 else [[3, 2, 1, 1]]
            db.execute('INSERT INTO queries VALUES (?, ?, ?, ?)', (i, key, '2p', json.dumps(rows)))
    return folder, queries, records, trace


class ToyClassifier:
    def classify(self, shape, query, hard):
        return ({3: 1, 4: 2} if query[0] == 0 else {3: 1}), 2


def test_mixed_answers_ties_conditional_means_and_additive_weights(tmp_path):
    _, _, records, trace = fixture_records(tmp_path)
    summary = summarize_difficulty(trace, records, ToyClassifier())
    rows = {(r['grouping'], r['label']): r for r in summary['rows']}
    partial, full = rows['difficulty', 'partial'], rows['difficulty', 'full']
    assert partial['queries'] == 2 and full['queries'] == 1
    assert partial['sort']['mrr'] == .75
    assert partial['expected']['mrr'] == .625  # E[1/rank] for uniform ranks 1,2.
    assert partial['expected']['hits1'] == .25
    assert full['sort']['mrr'] == .25
    overall = rows['overall', 'all']
    assert overall['sort']['mrr'] == .5625
    assert partial['query_weight'] == .75 and full['query_weight'] == .25
    for policy in ('sort', 'expected'):
        for metric in ('mrr', 'hits1', 'hits3', 'hits10'):
            assert np.isclose(sum(r['original_weight_contribution'][policy][metric] for r in (partial, full)), overall[policy][metric])
    assert rows['released_reduction', 'unavailable']['hard_answers'] == 3


@pytest.mark.parametrize('mutation,match', [
    ("UPDATE queries SET query_id='wrong' WHERE position=0", 'identity'),
    ("UPDATE queries SET answers='[[5,1,0,1],[4,4,3,1]]' WHERE position=0", 'exact root'),
    ("UPDATE queries SET answers='[[3,1,0,1],[3,4,3,1]]' WHERE position=0", 'exact root'),
    ("UPDATE queries SET answers='[[3,3,0,2],[4,4,3,1]]' WHERE position=0", 'tie block'),
])
def test_identity_answer_and_rank_invariants(tmp_path, mutation, match):
    _, _, records, trace = fixture_records(tmp_path)
    with sqlite3.connect(trace) as db:
        db.execute(mutation)
    with pytest.raises(ValueError, match=match):
        summarize_difficulty(trace, records, ToyClassifier())


def test_released_partitions_missing_overlap_extra_and_provenance(tmp_path):
    folder, queries, records, _ = fixture_records(tmp_path)
    unavailable, info = released_labels(folder, records, dataset='FB15k237+H')
    assert not unavailable and info['2p']['status'] == 'unavailable'
    first = folder / 'test-query-reduction/2p/1p/test-hard-answers.pkl'
    second = folder / 'test-query-reduction/2p/2p/test-hard-answers.pkl'
    first.parent.mkdir(parents=True)
    first.write_bytes(pickle.dumps({queries[0]: {3, 8}, queries[1]: {3}}))
    with pytest.raises(FileNotFoundError):
        released_labels(folder, records, dataset='FB15k237+H')
    second.parent.mkdir(parents=True)
    second.write_bytes(pickle.dumps({queries[0]: {4}}))
    labels, info = released_labels(folder, records, dataset='FB15k237+H')
    assert len(labels) == 2 and info['2p']['label_reference_graph'] == 'train+valid'
    assert info['2p']['excluded_extra_answers'] == 1
    second.write_bytes(pickle.dumps({queries[0]: {3, 4}}))
    with pytest.raises(ValueError, match='Overlapping'):
        released_labels(folder, records, dataset='FB15k237+H')
    second.write_bytes(pickle.dumps({}))
    with pytest.raises(ValueError, match='Missing released'):
        released_labels(folder, records, dataset='FB15k237+H')


def test_partial_trace_does_not_claim_complete_root_coverage(tmp_path):
    _, _, records, trace = fixture_records(tmp_path)
    with sqlite3.connect(trace) as db:
        db.execute('DELETE FROM queries WHERE position=1')
    summary = summarize_difficulty(trace, records, ToyClassifier())
    assert summary['counts']['2p']['queries'] == 1 and len(records) == 2


@pytest.mark.parametrize('method', ['cone', 'qto', 'cqd-hybrid'])
@pytest.mark.parametrize('inference_graph', ['train', 'train+valid'])
def test_export_fixed_test_reference_corrected_filters_and_pending_jobs(tmp_path, method, inference_graph):
    from dicee.query_answering.datasets import dataset_spec
    folder, queries, records, trace = fixture_records(tmp_path)
    actual = tmp_path / 'inputs' / 'data' / dataset_spec('FB15k237+H')[1]
    actual.parent.mkdir(parents=True)
    folder.rename(actual)
    # Distinct relations to keep reciprocal facts from creating shortcuts.
    (actual / 'train.txt').write_text('0 0 2\n')
    (actual / 'valid.txt').write_text('1 0 2\n')
    (actual / 'test.txt').write_text('2 1 3\n2 1 4\n')
    (actual / 'id2ent.pkl').write_bytes(pickle.dumps({i: str(i) for i in range(10)}))
    bundle, results = tmp_path / 'bundle', tmp_path / 'results'
    correction = bundle / 'filters.json'
    write_json(correction, {'changes': {}})
    pins = {str(p.relative_to(tmp_path / 'inputs')): checksum(p) for p in actual.glob('*')}
    frozen = {'manifest': {'data_root': 'data', 'entries': [{'id': 'done'}, {'id': 'excluded'}, {'id': 'pending'}]},
              'files': pins, 'source_sha256': 'old-code',
              'filter_corrections': {'FB15k237+H': {'path': 'filters.json', 'sha256': checksum(correction)}}}
    write_json(bundle / 'bundle.json', frozen)
    results.mkdir()
    trace.rename(results / 'ranks.sqlite3')
    overall = next(r for r in summarize_difficulty(results / 'ranks.sqlite3', records, ToyClassifier())['rows'] if r['grouping'] == 'overall')
    report = {'benchmark_run': {'entry': 'done', 'bundle': checksum(bundle / 'bundle.json')},
              'dataset': 'FB15k237+H', 'split': 'test', 'inference': {'method': method},
              'dataset_metadata': {'inference_graph': inference_graph},
              'protocol': {'answer_filter': 'corrected', 'full_split': True},
              'rank_trace_sha256': checksum(results / 'ranks.sqlite3'),
              'per_shape': {'2p': overall['sort'] | {'queries': 2, 'hard_answers': 3}},
              'additional_tie_metrics': {'expected': {'per_shape': {'2p': overall['expected']}}}}
    write_json(results / 'result.json', report)
    write_json(results / 'excluded' / 'result.json', report | {
        'benchmark_run': report['benchmark_run'] | {'entry': 'excluded'}})
    # An active job is represented by an unusable trace and no result; never read.
    (results / 'active').mkdir()
    (results / 'active' / 'ranks.sqlite3').write_text('do not open')
    payload = export_difficulty_reports(results, bundle, tmp_path / 'inputs', tmp_path / 'out', entries=['done'], query_types=['2p'])
    assert payload['pending_main_entries'] == ['pending']
    assert payload['hardness_reference_graph'] == 'train+valid'
    assert {r['comparison_graph'] for r in payload['rows']} == {'train+valid'}
    assert {r['label_reference_graph'] for r in payload['rows']} == {'train+valid'}
    assert {r['inference_graph'] for r in payload['rows']} == {inference_graph}
    assert {r['label'] for r in payload['rows'] if r['grouping'] == 'inferred_positive_edges'} == {'1'}
    assert len(payload['details']) == 1
    assert payload['frozen_inference_source_sha256'] == 'old-code'
    assert payload['postprocessor_files']
    assert all(r['answer_filter'] == 'corrected' for r in payload['rows'])
    with pytest.raises(ValueError, match='new report directory'):
        export_difficulty_reports(results, bundle, tmp_path / 'inputs', results / 'out')
