"""Published graph layouts, hard-answer ranking, and shared benchmark execution."""

import io
import json
import pickle
import zipfile
from dataclasses import replace

import numpy as np
import pytest
import torch

from dicee.query_answering import (
    BENCHMARK_DATASETS,
    BenchmarkQuery,
    QueryBenchmark,
    QueryContext,
    benchmark_model,
    evaluate_benchmark,
    load_benchmark,
    summarize_benchmarks,
)
from dicee.query_answering._query import QUERY_SHAPES
from dicee.query_answering.benchmark import filtered_query_metrics
from dicee.query_answering.datasets import dataset_spec, download_benchmark
from tests.test_query_engine import TableModel, programs


def dump(path, name, value):
    with (path / name).open('wb') as stream:
        pickle.dump(value, stream)


def graph(path, name, direct, pairs, binary=False):
    triples = list(QueryContext(direct, 8, 4, pairs).triples)
    if binary:
        torch.save(torch.tensor(triples), path / f'{name}.pt')
    else:
        (path / f'{name}.txt').write_text(''.join(f'{h}\t{r}\t{t}\n' for h, r, t in triples))
    return set(triples)


def fixture_dataset(root, name, binary=False):
    group, folder, _ = dataset_spec(name)
    path = root / folder
    path.mkdir(parents=True)
    pairs = ((0, 1), (2, 3)) if group == 'transductive' else ((0, 2), (1, 3))
    train = graph(path, 'train' if group == 'transductive' else 'train_graph', [(0, 0, 1), (1, 1 if group != 'transductive' else 2, 2)], pairs, binary)
    test = None
    if group == 'transductive':
        dump(path, 'id2ent.pkl', {i: str(i) for i in range(8)})
        dump(path, 'id2rel.pkl', {i: str(i) for i in range(4)})
        (path / 'valid.txt').write_text('0\t0\t6\n')
        (path / 'test.txt').write_text('0\t0\t7\n')
    else:
        graph(path, 'val_inference', [(2, 1, 5)], pairs, binary)
        direct = [(2, 1, 3)] if group == 'inductive-e' else [(0, 1, 3), (1, 0, 2)]
        test = graph(path, 'test_inference', direct, pairs, binary)
    structured = {QUERY_SHAPES[shape]: [value[0]] for shape, value in programs(np.ones((4, 2, 4)) * .5, 2).items()}
    for split in ('valid', 'test'):
        hard_id = (5 if split == 'valid' else 3) if group == 'inductive-e' else (2 if split == 'valid' and group == 'inductive-er' else 3)
        if group == 'transductive':
            dump(path, f'{split}-queries.pkl', structured)
            dump(path, f'{split}-easy-answers.pkl', {q: {1} for qs in structured.values() for q in qs})
            dump(path, f'{split}-hard-answers.pkl', {q: {hard_id} for qs in structured.values() for q in qs})
        else:
            dump(path, f'{split}_queries.pkl', structured)
            dump(path, f'{split}_answers_easy.pkl', {s: {q: {1} for q in qs} for s, qs in structured.items()})
            dump(path, f'{split}_answers_hard.pkl', {s: {q: {hard_id} for q in qs} for s, qs in structured.items()})
    return path, train, test


@pytest.mark.parametrize('name', ['FB15k237LogicalQuery', 'InductiveFB15k237Query:106', 'WikiTopicsQuery:art'])
@pytest.mark.parametrize('split', ['valid', 'test'])
@pytest.mark.parametrize('binary', [False, True])
def test_official_layouts_no_target_leakage(tmp_path, name, split, binary):
    path, train, test = fixture_dataset(tmp_path, name, binary)
    data = load_benchmark(tmp_path, name, split=split)
    assert len(data.queries) == 14
    assert {q.shape for q in data.queries} == set(QUERY_SHAPES)
    assert len(data.metadata['files']) >= 4
    edges = set(data.context.triples)
    if data.group == 'transductive':
        assert edges == train and data.candidates == tuple(range(8))
        assert not {'valid.txt', 'test.txt'} & data.metadata['files'].keys()
    elif data.group == 'inductive-e':
        assert data.context.num_entities == 6
        assert data.candidates == ((0, 1, 2, 5) if split == 'valid' else (0, 1, 2, 3))
        assert edges == (train | test) if split == 'test' else all(3 not in (h, t) for h, _, t in edges)
    else:
        assert edges == (train if split == 'valid' else test)
    assert all(q.hard <= set(data.candidates) for q in data.queries)
    selected = load_benchmark(tmp_path, name, split=split, query_types=['2u', 'pni'])
    assert {q.shape for q in selected.queries} == {'2u', 'pni'}


def test_all_23_names_and_invalid_data(tmp_path):
    assert len(BENCHMARK_DATASETS) == 23
    for name in BENCHMARK_DATASETS:
        assert dataset_spec(name)[2].startswith('https://')
    with pytest.raises(ValueError, match='Unknown benchmark'):
        dataset_spec('WikiTopicsQuery:missing')
    with pytest.raises(FileNotFoundError, match='extract'):
        load_benchmark(tmp_path, 'FB15kLogicalQuery')
    path, _, _ = fixture_dataset(tmp_path, 'FB15kLogicalQuery')
    with pytest.raises(ValueError, match='distinct supported'):
        load_benchmark(tmp_path, 'FB15kLogicalQuery', query_types=['2u-DNF'])
    (path / 'train.txt').write_text('0\t0\t1\n')
    with pytest.raises(ValueError, match='reciprocal'):
        load_benchmark(tmp_path, 'FB15kLogicalQuery')


def test_extended_answers_keep_original_query_order(tmp_path):
    path, _, _ = fixture_dataset(tmp_path, 'InductiveFB15k237Query:106')
    shape = QUERY_SHAPES['1p']
    queries = [(2, (0,)), (0, (0,))]
    dump(path, 'train_queries.pkl', {shape: queries})
    dump(path, 'train_answers_test.pkl', {shape: [{3}, {2}]})
    data = load_benchmark(tmp_path, 'InductiveFB15k237QueryExtendedEval:106', query_types=['1p'])
    assert [q.hard for q in data.queries] == [frozenset({2}), frozenset({3})]
    assert all(not q.easy for q in data.queries)
    assert data.metadata['evaluation'] == 'faithfulness'


@pytest.mark.parametrize('tied', [False, True])
def test_reference_sort_filter_protocol(tied):
    # Independent implementation of the published positional rank adjustment.
    scores = torch.tensor([.9, .8, .7, .6, .5, .4], dtype=torch.float64)
    if tied:
        scores[:] = .5
    easy, hard, candidates = {0}, {2, 4}, {0, 1, 2, 4, 5}
    masked = scores.clone()
    masked[3] = -torch.inf
    order = masked.argsort(descending=True).tolist()
    answer_order = [i for i in order if i in easy | hard]
    ranks = [order.index(h) - answer_order.index(h) + 1 for h in sorted(hard)]
    result = filtered_query_metrics(scores, easy, hard, candidates=candidates)
    assert result['mrr'] == pytest.approx(np.mean([1 / r for r in ranks]))
    assert result['hits1'] == pytest.approx(np.mean([r == 1 for r in ranks]))


@pytest.mark.parametrize('policy,rank', [('average', 2.), ('optimistic', 1.), ('pessimistic', 3.)])
def test_ties_and_negative_infinity_exclude_other_answers(policy, rank):
    scores = torch.full((6,), -torch.inf)
    result = filtered_query_metrics(scores, {0}, {1, 2}, candidates=[0, 1, 2, 3, 4], tie_policy=policy)
    assert result['mrr'] == pytest.approx(1 / rank)
    with pytest.raises(ValueError, match='disjoint'):
        filtered_query_metrics(scores, {0}, {0})
    with pytest.raises(ValueError, match='score vector'):
        filtered_query_metrics(torch.tensor([float('nan')]), set(), {0})


def test_query_then_shape_then_dataset_averaging_and_partial_flags():
    context = QueryContext([(0, 0, 1)], 4, 1)
    queries = (BenchmarkQuery('1p', (0, (0,)), frozenset({0}), frozenset({1, 2})),
               BenchmarkQuery('1p', (1, (0,)), frozenset(), frozenset({3})),
               BenchmarkQuery('2p', (0, (0, 0)), frozenset(), frozenset({0})))
    data = QueryBenchmark('fixture', 'transductive', 'test', context, queries, tuple(range(4)))
    scores = torch.tensor([4., 3., 2., 1.])
    report = evaluate_benchmark(data, lambda q: scores)
    assert report['per_shape']['1p']['mrr'] == pytest.approx((1 + .25) / 2)
    assert report['averages']['epfo']['mrr'] == pytest.approx(((1 + .25) / 2 + 1) / 2)
    subset = evaluate_benchmark(data, lambda q: scores, max_queries_per_shape=1)
    assert subset['queries'] == 2 and not subset['protocol']['full_split']
    assert not subset['protocol']['all_14_shapes']
    other = dict(subset, dataset='other')
    summary = summarize_benchmarks([report, other])
    assert summary['groups']['all']['epfo']['mrr'] == pytest.approx((report['averages']['epfo']['mrr'] + 1) / 2)
    assert not summary['complete_23_dataset_test_suite']
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_benchmarks([report, report])
    with pytest.raises(ValueError, match='complete public'):
        evaluate_benchmark(data, lambda q: scores[:2])


@pytest.mark.parametrize('name', ['ULTRA', 'TRIX', 'Flock'])
@pytest.mark.parametrize('attached', [False, True])
def test_real_models_restore_graph_modes_and_failure(tmp_path, name, attached):
    from dicee.models import TRIX, ULTRA, Flock
    fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    data = load_benchmark(tmp_path, 'WikiTopicsQuery:art')
    settings = dict(num_entities=3, num_relations=2, ultra_dim=8, ultra_num_layers=2, trix_dim=8,
                    flock_dim=8, flock_walk_num=2, flock_walk_len=8, flock_refinements=2, flock_seed=77)
    model = {'ULTRA': ULTRA, 'TRIX': TRIX, 'Flock': Flock}[name](settings)
    if attached:
        model.set_graph(torch.tensor([[0, 0, 1]]), inverse_relations={0: 1})
    original = QueryContext.from_model(model) if attached else None
    states = {k: v.clone() for k, v in model.state_dict().items()}
    report = benchmark_model(model, data, beam_size=2, seed=42)
    assert report['queries'] == 14 and report['protocol']['all_14_shapes']
    assert report['inference']['method'] == 'cqd-global-prefix'
    assert model.training and (model.num_entities, model.num_relations) == (3, 2)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[key], rtol=0, atol=0)
    assert QueryContext.from_model(model) == original if attached else model.graph_triples is None
    with pytest.raises(ValueError, match='positive integer'):
        benchmark_model(model, data, beam_size=0)
    assert QueryContext.from_model(model) == original if attached else model.graph_triples is None
    second = benchmark_model(model, data, beam_size=2, seed=42)
    assert second['per_shape'] == report['per_shape']


@pytest.mark.parametrize('executor', ['cqd', 'qto'])
def test_cli_checkpoint_reports_and_preserves_commit_work(tmp_path, monkeypatch, executor):
    from dicee.models import ULTRA
    from dicee.query_answering.__main__ import main
    fixture_dataset(tmp_path, 'FB15k237LogicalQuery')
    fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    config = dict(ultra_dim=8, ultra_num_layers=2)
    model = ULTRA(dict(config, num_entities=1, num_relations=1))
    checkpoint, config_path, output = tmp_path / 'model.pt', tmp_path / 'config.json', tmp_path / 'report.json'
    torch.save({'model': model.state_dict()}, checkpoint)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr('sys.argv', ['query_answering', 'benchmark', '--data-root', str(tmp_path),
                                    '--datasets', 'FB15k237LogicalQuery', 'WikiTopicsQuery:art',
                                    '--checkpoint', str(checkpoint), '--model-config', str(config_path),
                                    '--output', str(output), '--beam-size', '2', '--max-queries-per-shape', '1',
                                    '--query-sampling', 'uniform', '--sampling-seed', '83', '--executor', executor])
    main()
    report = json.loads(output.read_text())
    assert len(report['results']) == 2
    assert all(r['inference']['executor'] == executor for r in report['results'])
    assert all(r['protocol']['query_sampling'] == 'uniform' and r['protocol']['sampling_seed'] == 83 for r in report['results'])
    assert report['summary']['groups']['all']['datasets'] == 2
    assert not report['summary']['complete_23_dataset_test_suite']
    assert report['results'][0]['inference']['backbone_state_sha256'] == report['results'][1]['inference']['backbone_state_sha256']
    monkeypatch.setattr('dicee.query_answering.engine.QueryAnswerer.predict', lambda *a, **k: pytest.fail('Completed run was rescored'))
    main()
    resumed = json.loads(output.read_text())
    assert resumed['state'] == 'complete'
    assert [r['per_shape'] for r in resumed['results']] == [r['per_shape'] for r in report['results']]


def test_download_archive_paths(tmp_path, monkeypatch):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, 'w') as archive:
        archive.writestr('../escape.txt', 'bad')
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(memory.getvalue()))
    with pytest.raises(ValueError, match='Unsafe path'):
        download_benchmark('WikiTopicsQuery:art', tmp_path)
    assert not (tmp_path.parent / 'escape.txt').exists()


def test_prediction_receives_only_query_and_candidate_filtering_is_final(tmp_path):
    fixture_dataset(tmp_path, 'InductiveFB15k237Query:106')
    data = load_benchmark(tmp_path, 'InductiveFB15k237Query:106', query_types=['2p'])
    # Entity 5 is outside the final candidate domain but remains a valid
    # intermediate entity in the complete public vocabulary.
    raw = torch.full((6, 4, 6), -20., dtype=torch.float64)
    raw[0, 0, 5] = 10
    raw[5, 1, 3] = 10
    query = replace(data.queries[0], query=(0, (0, 1)), easy=frozenset(), hard=frozenset({3}))
    data.queries = (query,)
    result = benchmark_model(TableModel(raw), data, beam_size=1)
    assert result['per_shape']['2p']['mrr'] == 1.


def test_wikitopics_unused_relations_use_original_mapping(tmp_path):
    path, _, _ = fixture_dataset(tmp_path, 'WikiTopicsQuery:art')
    # Full vocabulary has six relation IDs, but the last inverse never appears.
    edges = QueryContext([(0, 0, 1), (2, 1, 3)], 4, 5, ((0, 3), (1, 4))).triples
    (path / 'test_inference.txt').write_text(''.join(f'{h}\t{r}\t{t}\n' for h, r, t in edges))
    dump(path, 'og_mappings.pkl', {'r2id': {'a': 0, 'b': 1, 'unused': 2, 'a_inv': 3, 'b_inv': 4, 'unused_inv': 5}})
    data = load_benchmark(tmp_path, 'WikiTopicsQuery:art')
    assert data.context.num_relations == 5
    assert data.context.inverse_relations == ((0, 3), (1, 4))
    assert data.context.triples == edges
    assert 'og_mappings.pkl' in data.metadata['files']


def test_cli_rejects_mismatched_transductive_vocabulary(tmp_path, monkeypatch):
    from dicee.models import DistMult
    from dicee.query_answering.__main__ import main
    from tests.test_query_engine import save_experiment
    path, _, _ = fixture_dataset(tmp_path, 'FB15k237LogicalQuery')
    settings = dict(model='DistMult', num_entities=8, num_relations=4, embedding_dim=8)
    model = DistMult(settings)
    experiment = tmp_path / 'experiment'
    save_experiment(model, settings, experiment)
    output = tmp_path / 'results.json'
    monkeypatch.setattr('sys.argv', ['query_answering', 'benchmark', '--data-root', str(tmp_path),
                                    '--experiment', str(experiment), '--output', str(output), '--query-types', '1p'])
    main()
    assert output.is_file()
    dump(path, 'id2ent.pkl', {i: str((i + 1) % 8) for i in range(8)})
    with pytest.raises(ValueError, match='exactly match'):
        main()


def test_suite_complete_only_with_all_datasets_shapes_queries():
    base = dict(split='test', protocol=dict(full_split=True, all_14_shapes=True),
                dataset_metadata={'evaluation': 'hard-answers'},
                averages={category: dict.fromkeys(('mrr', 'hits1', 'hits3', 'hits10'), .5)
                          for category in ('all', 'epfo', 'negation')})
    reports = [dict(base, dataset=name, group=dataset_spec(name)[0]) for name in BENCHMARK_DATASETS]
    assert summarize_benchmarks(reports)['complete_23_dataset_test_suite']
    assert not summarize_benchmarks(reports[:-1])['complete_23_dataset_test_suite']
    reports[0] = dict(reports[0], dataset_metadata={'evaluation': 'faithfulness'})
    with pytest.raises(ValueError, match='faithfulness'):
        summarize_benchmarks(reports)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('restricted', [False, True])
def test_reused_ranking_masks_match_independent_sort_oracle(device, restricted):
    from dicee.query_answering.benchmark import QueryMetrics
    n = 29
    candidates = set(range(n)) - ({0, 11, 24} if restricted else set())
    evaluator = QueryMetrics(n, candidates=candidates)
    rng = np.random.default_rng(13)
    for tied in (False, True):
        for _ in range(20):
            scores = torch.tensor(rng.integers(0, 4, n) if tied else rng.normal(size=n), device=device, dtype=torch.float64)
            if tied:
                scores[scores == 0] = -torch.inf
            unchanged = scores.clone()
            labels = rng.choice(sorted(candidates), 6, replace=False).tolist()
            easy, hard = set(labels[:3]), set(labels[3:])
            masked = scores.clone()
            masked[list(set(range(n)) - candidates)] = -torch.inf
            order = masked.argsort(descending=True).tolist()
            ranks = [1 + sum(e in candidates and e not in easy | hard for e in order[:order.index(t)]) for t in hard]
            expected = dict(mrr=np.mean([1/r for r in ranks]), hits1=np.mean([r <= 1 for r in ranks]),
                            hits3=np.mean([r <= 3 for r in ranks]), hits10=np.mean([r <= 10 for r in ranks]))
            assert evaluator(scores, easy, hard) == pytest.approx(expected)
            assert torch.equal(scores, unchanged)
