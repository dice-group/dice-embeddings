"""+H release layouts, inference splits, and reference ranking conformance.

Reference: april-tools/is-cqa-complex at
 d1ce74164936a7c09d9147e83190da047cb39429, models.py:test_step and
 create_queries.py:generate_queries. Fixtures mirror benchs-1.0, including
 ICEWS18's nested KG_splits and the extra filtered inference answers.
"""

import io
import json
import os
import zipfile
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

from dicee.query_answering import PLUS_H_DATASETS, evaluate_benchmark, load_benchmark, summarize_benchmarks
from dicee.query_answering._query import PLUS_H_SHAPES, QUERY_SHAPES
from dicee.query_answering.datasets import PLUS_H_PREFIX, PLUS_H_REFERENCE_COMMIT, dataset_spec, download_benchmark
from tests.test_query_benchmark import dump, fixture_dataset, graph
from tests.test_query_engine import programs


def plus_h_fixture(root, name):
    path = root / dataset_spec(name)[1]
    path.mkdir(parents=True)
    graph_path = path / 'KG_splits' if name == 'ICEWS18+H' else path
    graph_path.mkdir(exist_ok=True)
    pairs = ((0, 1), (2, 3))
    train = graph(graph_path, 'train', [(0, 0, 1), (1, 2, 2)], pairs)
    valid = graph(graph_path, 'valid', [(0, 0, 3), (3, 2, 4)], pairs)
    test = graph(graph_path, 'test', [(0, 0, 5), (5, 2, 6)], pairs)
    dump(path, 'id2ent.pkl', {i: str(i) for i in range(8)})
    dump(path, 'id2rel.pkl', {i: str(i) for i in range(4)})
    queries = {QUERY_SHAPES[shape]: {value[0]} for shape, value in programs(np.full((4, 2, 4), .5), 4).items()}
    # Published DNF is used; De Morgan duplicates must never count as new shapes.
    queries[(('e', ('r', 'n')), ('e', ('r', 'n')), ('n',))] = {((0, (0, -2)), (1, (1, -2)), (-2,))}
    for split, hard in [('valid', {3, 4}), ('test', {5, 6})]:
        dump(path, f'{split}-queries.pkl', queries)
        # Entity 7 is not reachable in the inference graph. +H filter files
        # can include unselected inference answers as well as observed answers.
        dump(path, f'{split}-easy-answers.pkl', {q: {1, 7} for qs in queries.values() for q in qs})
        dump(path, f'{split}-hard-answers.pkl', {q: hard for qs in queries.values() for q in qs})
    return path, train, valid, test


@pytest.mark.parametrize('name', PLUS_H_DATASETS)
@pytest.mark.parametrize('split', ['valid', 'test'])
def test_plus_h_release_graphs_ids_labels_and_all_sixteen_shapes(tmp_path, name, split):
    path, train, valid, test = plus_h_fixture(tmp_path, name)
    data = load_benchmark(tmp_path, name, split=split)
    assert set(data.context.triples) == (train | valid if split == 'test' else train)
    assert not set(data.context.triples) & test
    assert data.context.inverse_relations == ((0, 1), (2, 3))
    assert data.candidates == tuple(range(8))  # Includes the unobserved entity 7.
    assert data.entity_to_idx == {str(i): i for i in range(8)}
    assert data.relation_to_idx == {str(i): i for i in range(4)}
    assert len(data.queries) == 16
    assert {q.shape for q in data.queries} == set(PLUS_H_SHAPES)
    assert next(q.query for q in data.queries if q.shape == '4p') == (0, (0, 1, 0, 1))
    assert next(q.query for q in data.queries if q.shape == '4i') == ((0, (0,)), (1, (1,)), (2, (0,)), (3, (1,)))
    assert all(q.easy == frozenset({1, 7}) for q in data.queries)
    assert all(q.hard == frozenset({5, 6} if split == 'test' else {3, 4}) for q in data.queries)
    metadata = data.metadata
    assert metadata['suite'] == 'plus-h' and metadata['release'] == 'benchs-1.0'
    assert metadata['reference_commit'] == PLUS_H_REFERENCE_COMMIT
    assert metadata['inference_graph'] == ('train+valid' if split == 'test' else 'train')
    assert len(metadata['files']) == (7 if split == 'test' else 6)
    assert all(not key.endswith('test.txt') for key in metadata['files'])
    assert all(len(value) == 64 for value in metadata['files'].values())
    # Target triples are not even read; modifying them cannot change inference.
    (path / ('KG_splits/' if name == 'ICEWS18+H' else '') / 'test.txt').write_text('invalid target data')
    assert load_benchmark(tmp_path, name, split=split).context == data.context


def reference_metrics(scores, easy, hard, candidates):
    """Positional answer-rank subtraction used by both pinned upstream evaluators.

    Independently reconstruct the public protocol, without QueryMetrics,
    FilteredRanker, query parsing, or benchmark aggregation helpers.
    """
    masked = scores.clone()
    masked[sorted(set(range(len(scores))) - set(candidates))] = -torch.inf
    order = torch.argsort(masked, descending=True)
    positions = torch.empty(len(scores), dtype=torch.long)
    positions[order] = torch.arange(len(scores))
    answers = sorted(easy) + sorted(hard)
    sorted_positions, permutation = positions[answers].sort()
    filtered = sorted_positions - torch.arange(len(answers)) + 1
    ranks = filtered[permutation >= len(easy)].double()
    return dict(mrr=ranks.reciprocal().mean().item(),
                **{f'hits{k}': (ranks <= k).double().mean().item() for k in (1, 3, 10)})


@pytest.mark.parametrize('name', [*PLUS_H_DATASETS, 'FB15k237LogicalQuery', 'InductiveFB15k237Query:106', 'WikiTopicsQuery:art'])
@pytest.mark.parametrize('tied', [False, True])
def test_both_published_benchmarks_match_independent_reference_end_to_end(tmp_path, name, tied):
    """Exercise the loaders through per-query, per-type and suite reports."""
    (plus_h_fixture if name in PLUS_H_DATASETS else fixture_dataset)(tmp_path, name)
    data = load_benchmark(tmp_path, name)
    # Unequal numbers of queries and hard answers expose incorrect micro-averages.
    extra = replace(data.queries[0], query=(2, (0,)), easy=frozenset(), hard=frozenset({2}))
    data = replace(data, queries=data.queries + (extra,))
    rng = np.random.default_rng(83)
    scores = {q.query: torch.tensor(rng.integers(-2, 2, data.context.num_entities) if tied else
                                    rng.normal(size=data.context.num_entities), dtype=torch.float64) for q in data.queries}
    expected = {}
    for q in data.queries:
        expected.setdefault(q.shape, []).append(reference_metrics(scores[q.query], q.easy, q.hard, data.candidates))
    report = evaluate_benchmark(data, scores.__getitem__)
    for shape, values in expected.items():
        for metric in ('mrr', 'hits1', 'hits3', 'hits10'):
            assert report['per_shape'][shape][metric] == pytest.approx(np.mean([v[metric] for v in values]))
    for category, selected in [('all', list(expected)), ('epfo', [s for s in expected if 'n' not in s]),
                               ('negation', [s for s in expected if 'n' in s])]:
        for metric in ('mrr', 'hits1', 'hits3', 'hits10'):
            value = np.mean([np.mean([v[metric] for v in expected[s]]) for s in selected])
            assert report['averages'][category][metric] == pytest.approx(value)
            assert summarize_benchmarks([report])['groups']['all'][category][metric] == pytest.approx(value)
    assert report['protocol']['all_benchmark_shapes']
    assert report['protocol']['all_14_shapes'] == (name not in PLUS_H_DATASETS)


def test_plus_h_completeness_partial_selection_and_suite_separation(tmp_path):
    reports = []
    for name in PLUS_H_DATASETS:
        plus_h_fixture(tmp_path, name)
        data = load_benchmark(tmp_path, name)
        reports.append(evaluate_benchmark(data, lambda q: torch.arange(8).float()))
    summary = summarize_benchmarks(reports)
    assert summary['complete_plus_h_test_suite']
    assert not summary['complete_23_dataset_test_suite']
    assert reports[0]['averages']['epfo']['shapes'] == 11
    assert reports[0]['averages']['negation']['shapes'] == 5
    assert not summarize_benchmarks(reports[:-1])['complete_plus_h_test_suite']
    data = load_benchmark(tmp_path, PLUS_H_DATASETS[0], query_types=['4p', '4i'])
    partial = evaluate_benchmark(data, lambda q: torch.arange(8).float())
    assert partial['protocol']['full_split'] and not partial['protocol']['all_benchmark_shapes']
    assert not summarize_benchmarks([partial, *reports[1:]])['complete_plus_h_test_suite']
    fixture_dataset(tmp_path, 'FB15k237LogicalQuery')
    old = evaluate_benchmark(load_benchmark(tmp_path, 'FB15k237LogicalQuery'), lambda q: torch.arange(8).float())
    with pytest.raises(ValueError, match='suites'):
        summarize_benchmarks([old, reports[0]])
    with pytest.raises(ValueError, match='supported'):
        load_benchmark(tmp_path, 'FB15k237LogicalQuery', query_types=['4p'])


def test_plus_h_cannot_silently_omit_four_hop_queries(tmp_path):
    path, *_ = plus_h_fixture(tmp_path, PLUS_H_DATASETS[0])
    dump(path, 'test-queries.pkl', {('e', ('r',)): {(0, (0,))}})
    with pytest.raises(ValueError, match='Missing requested query types'):
        load_benchmark(tmp_path, PLUS_H_DATASETS[0])


def test_download_plus_h_extracts_evaluation_inputs_without_old_data(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        for name in PLUS_H_DATASETS:
            folder = dataset_spec(name)[1]
            z.writestr(f'{folder}/id2ent.pkl', b'published')
            z.writestr(f'{folder}/test-hard-answers.pkl', b'published')
            z.writestr(f'{folder}/train-queries.pkl', b'large unused training data')
        z.writestr(f'{PLUS_H_PREFIX}/ICEWS18+H/KG_splits/train.txt', b'0\t0\t1\n')
        z.writestr('FB15k-237-betae/train.txt', b'old benchmark')
    content = archive.getvalue()
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(content))
    old = tmp_path / 'FB15k-237-betae'
    old.mkdir()
    (old / 'train.txt').write_text('keep existing benchmark')
    download_benchmark(PLUS_H_DATASETS[0], tmp_path)
    assert (old / 'train.txt').read_text() == 'keep existing benchmark'
    for name in PLUS_H_DATASETS:
        path = tmp_path / dataset_spec(name)[1]
        assert (path / 'test-hard-answers.pkl').read_bytes() == b'published'
        assert not (path / 'train-queries.pkl').exists()
    assert (tmp_path / PLUS_H_PREFIX / 'ICEWS18+H/KG_splits/train.txt').is_file()


@pytest.mark.parametrize('executor', ['cqd', 'qto'])
def test_plus_h_cli_sixteen_shapes_and_resume(tmp_path, monkeypatch, executor):
    from dicee.models import ULTRA
    from dicee.query_answering.__main__ import main

    for name in PLUS_H_DATASETS:
        plus_h_fixture(tmp_path, name)
    config = dict(ultra_dim=8, ultra_num_layers=2)
    model = ULTRA(dict(config, num_entities=1, num_relations=1))
    checkpoint, config_path, output = tmp_path / 'model.pt', tmp_path / 'config.json', tmp_path / 'report.json'
    torch.save({'model': model.state_dict()}, checkpoint)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr('sys.argv', ['query_answering', 'benchmark', '--data-root', str(tmp_path), '--datasets', '+h',
                                    '--checkpoint', str(checkpoint), '--model-config', str(config_path), '--output', str(output),
                                    '--beam-size', '2', '--threads', '1', '--executor', executor])
    main()
    report = json.loads(output.read_text())
    assert report['state'] == 'complete' and report['summary']['complete_plus_h_test_suite']
    assert all(r['queries'] == 16 for r in report['results'])
    main()
    resumed = json.loads(output.read_text())
    assert [r['per_shape'] for r in resumed['results']] == [r['per_shape'] for r in report['results']]


# Counts inspected directly from the official benchs-1.0 archive. Ordering is
# explicit here so changing the production shape registry cannot change them.
RELEASE_SHAPES = ('1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up', '4p', '4i')
RELEASE_COUNTS = {
    ('FB15k237+H', 'valid'): (20094, 1416, 2262, 7565, 11078, 2113, 776, 5000, 3820, 2406, 5000, 2745, 7610, 507, 3755, 23150),
    ('FB15k237+H', 'test'): (22804, 936, 1461, 6692, 9615, 1586, 604, 5000, 3057, 2093, 5000, 2663, 6695, 473, 2670, 20189),
    ('NELL995+H', 'valid'): (16910, 944, 1953, 8286, 14108, 2114, 1264, 4000, 5100, 2214, 4000, 2459, 8169, 995, 2070, 24479),
    ('NELL995+H', 'test'): (17021, 1054, 1583, 8278, 13118, 1957, 1260, 4000, 5236, 1995, 4000, 2708, 8168, 1073, 2322, 22326),
    ('ICEWS18+H', 'valid'): (30803, 2237, 3077, 11531, 26614, 13480, 4452, 1347, 7487, 2716, 1440, 2642, 7400, 3242, 4187, 38881),
    ('ICEWS18+H', 'test'): (30605, 2125, 3582, 11918, 27252, 14746, 5140, 1548, 7811, 2919, 1508, 2875, 7516, 3855, 4338, 38767),
}


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('DICEE_PLUS_H_DATA_ROOT'), reason='Set DICEE_PLUS_H_DATA_ROOT to the official extracted release')
@pytest.mark.parametrize('name,split', RELEASE_COUNTS)
def test_official_plus_h_release_conformance(name, split):
    from pathlib import Path

    root = Path(os.environ['DICEE_PLUS_H_DATA_ROOT'])
    data = load_benchmark(root, name, split=split)
    assert Counter(q.shape for q in data.queries) == dict(zip(RELEASE_SHAPES, RELEASE_COUNTS[name, split]))
    n, nr = {'FB15k237+H': (14505, 474), 'NELL995+H': (63361, 400), 'ICEWS18+H': (20840, 500)}[name]
    assert (data.context.num_entities, data.context.num_relations) == (n, nr)
    assert data.candidates == tuple(range(n))
    path = root / dataset_spec(name)[1]
    if name == 'ICEWS18+H':
        path /= 'KG_splits'

    def triples(stem):
        return {tuple(map(int, row.split())) for row in (path / f'{stem}.txt').read_text().splitlines() if row.strip()}

    observed = triples('train')
    if split == 'test':
        observed |= triples('valid')
    assert set(data.context.triples) == observed
    assert not observed & triples(split)
    # Only label-free, deterministic scores reach the predictor. Full domains
    # and published answer filters are retained despite the query sample.
    scores = torch.sin(torch.arange(n, dtype=torch.float64))
    report = evaluate_benchmark(data, lambda q: scores, max_queries_per_shape=1)
    first = {}
    for query in data.queries:
        first.setdefault(query.shape, query)
    for shape, query in first.items():
        expected = reference_metrics(scores, query.easy, query.hard, data.candidates)
        assert {key: report['per_shape'][shape][key] for key in expected} == pytest.approx(expected)
    assert report['queries'] == 16 and not report['protocol']['full_split']
