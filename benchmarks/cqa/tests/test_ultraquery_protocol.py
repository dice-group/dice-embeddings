"""Optional conformance checks against the released UltraQuery suite and code."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from dicee.query_answering import BENCHMARK_DATASETS, evaluate_benchmark, load_benchmark
from dicee.query_answering._query import ULTRAQUERY_SHAPES
from dicee.query_answering.benchmark import QueryMetrics
from dicee.query_answering.datasets import REFERENCE_COMMIT, dataset_spec


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def upstream_reference(monkeypatch):
    if not os.environ.get('DICEE_ULTRAQUERY_REFERENCE_ROOT'):
        pytest.skip('Set DICEE_ULTRAQUERY_REFERENCE_ROOT, with upstream dependencies installed')
    reference = Path(os.environ['DICEE_ULTRAQUERY_REFERENCE_ROOT'])
    # Verify the actual source files, including when a Docker image lacks git.
    hashes = {
        'query_utils.py': '9b41775a5b54a0e40b9bd3acc24e0f20e54ecf0150f30beae382aa95c36191a0',
        'datasets_query.py': 'c82a04f477724cf3b6d69ef9e2bcdc2c03b7da0b856c47c915d8b3608168a510',
    }
    for name, checksum in hashes.items():
        assert hashlib.sha256((reference / 'ultra' / name).read_bytes()).hexdigest() == checksum, REFERENCE_COMMIT
    monkeypatch.syspath_prepend(str(reference))
    from ultra import datasets_query, query_utils
    return datasets_query, query_utils


def test_published_negative_infinity_candidate_edge_case(upstream_reference):
    _, reference = upstream_reference
    scores = torch.full((1, 6), -torch.inf)
    easy = torch.zeros_like(scores, dtype=torch.bool)
    hard = easy.clone()
    hard[0, 2] = True
    candidates = torch.tensor([0, 2, 4])
    published, _ = reference.batch_evaluate(scores.clone(), (torch.tensor([0]), easy, hard), candidates)
    _, records = QueryMetrics(6, candidates=candidates.tolist()).evaluate(scores[0], set(), {2}, records=True)
    assert published.tolist() == [3]
    assert records[0][1] == 2


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('DICEE_ULTRAQUERY_DATA_ROOT'), reason='Set DICEE_ULTRAQUERY_DATA_ROOT to the official extracted data')
@pytest.mark.parametrize('name', BENCHMARK_DATASETS)
@pytest.mark.parametrize('split', ['valid', 'test'])
def test_official_ultraquery_data_graphs_and_reference_ranks(name, split):
    """Real-data conformance for all 23 datasets in the published suite."""
    root = Path(os.environ['DICEE_ULTRAQUERY_DATA_ROOT'])
    data = load_benchmark(root, name, split=split)
    path = root / dataset_spec(name)[1]

    def triples(stem):
        file = path / f'{stem}.txt'
        if file.is_file():
            return {tuple(map(int, row.split())) for row in file.read_text().splitlines() if row.strip()}
        return {tuple(row) for row in torch.load(path / f'{stem}.pt', weights_only=True).tolist()}

    if data.group == 'transductive':
        observed = triples('train')
        assert data.candidates == tuple(range(data.context.num_entities))
    elif data.group == 'inductive-e':
        observed = triples('train_graph') | triples('val_inference' if split == 'valid' else 'test_inference')
    else:
        observed = triples('train_graph' if split == 'valid' else 'test_inference')
    assert set(data.context.triples) == observed
    if data.group != 'transductive':
        assert set(data.candidates) == {entity for h, _, t in observed for entity in (h, t)}
    scores = torch.sin(torch.arange(data.context.num_entities, dtype=torch.float64))
    report = evaluate_benchmark(data, lambda q: scores, max_queries_per_shape=1)
    first = {}
    for query in data.queries:
        first.setdefault(query.shape, query)
    masked = scores.clone()
    masked[list(set(range(len(scores))) - set(data.candidates))] = -torch.inf
    order = masked.argsort(descending=True).tolist()
    positions = {entity: i for i, entity in enumerate(order)}
    for shape, query in first.items():
        true_answers = sorted(query.easy | query.hard, key=positions.__getitem__)
        ranks = [positions[entity] - offset + 1 for offset, entity in enumerate(true_answers) if entity in query.hard]
        expected = dict(mrr=np.mean([1 / r for r in ranks]),
                        **{f'hits{k}': np.mean([r <= k for r in ranks]) for k in (1, 3, 10)})
        assert {key: report['per_shape'][shape][key] for key in expected} == pytest.approx(expected)
    assert report['queries'] == 14 and report['protocol']['all_14_shapes']
    assert not report['protocol']['full_split']


@pytest.mark.integration
@pytest.mark.skipif(not (os.environ.get('DICEE_ULTRAQUERY_DATA_ROOT') and os.environ.get('DICEE_ULTRAQUERY_REFERENCE_ROOT')),
                    reason='Set official data and pinned upstream roots, with upstream dependencies installed')
@pytest.mark.parametrize('name', ['FB15k237LogicalQuery', 'InductiveFB15k237Query:106', 'WikiTopicsQuery:art'])
@pytest.mark.parametrize('split', ['valid', 'test'])
@pytest.mark.parametrize('tied', [False, True])
def test_official_ultraquery_evaluator_parity(upstream_reference, name, split, tied):
    """Execute the authors' unchanged evaluator on real labels from all 14 types."""
    datasets_query, reference = upstream_reference

    data = load_benchmark(os.environ['DICEE_ULTRAQUERY_DATA_ROOT'], name, split=split)
    assert data.metadata['reference_commit'] == REFERENCE_COMMIT
    if data.group != 'transductive':
        cls = datasets_query.InductiveFB15k237Query if data.group == 'inductive-e' else datasets_query.WikiTopicsQuery
        upstream_data = object.__new__(cls)
        upstream_data.root = os.environ['DICEE_ULTRAQUERY_DATA_ROOT']
        upstream_data.version = name.split(':')[1]
        upstream_data.query_types, upstream_data.union_type = None, 'DNF'
        upstream_data.pre_transform = None
        if data.group == 'inductive-er':
            # DICE's selective extraction omits this unused WikiTopics file;
            # the authors' build_vocab ignores validation triples entirely.
            upstream_data.load_file = lambda path: ([] if Path(path).stem == 'val_inference'
                                                    else cls.load_file(upstream_data, path))
        # Run graph construction without loading training queries or building
        # auxiliary relation graphs. This does not modify source data or code.
        upstream_data.load_queries = lambda path: None
        upstream_data.process()
        graph = getattr(upstream_data, f'{split}_graph')
        edges = torch.stack((graph.edge_index[0], graph.edge_type, graph.edge_index[1]), dim=1)
        assert set(map(tuple, edges.tolist())) == set(data.context.triples)
        assert graph.num_nodes == data.context.num_entities
        assert graph.num_relations == data.context.num_relations
        nodes = getattr(graph, 'restrict_nodes', torch.arange(graph.num_nodes))
        assert tuple(sorted(nodes.tolist())) == data.candidates
    grouped = {shape: [] for shape in sorted(ULTRAQUERY_SHAPES)}
    for query in data.queries:
        grouped[query.shape].append(query)
    # Unequal type sizes distinguish macro averaging from query/answer pooling.
    selected = tuple(query for i, queries in enumerate(grouped.values()) for query in queries[:1 + i % 3])
    data = replace(data, queries=selected)
    n = data.context.num_entities
    scores = torch.sin(torch.arange(n, dtype=torch.float64))
    if tied:
        scores = scores.round()
    report = evaluate_benchmark(data, lambda _: scores)
    id2type = [shape + '-DNF' if shape in ('2u', 'up') else shape for shape in grouped]
    types = torch.tensor([id2type.index(q.shape + '-DNF' if q.shape in ('2u', 'up') else q.shape) for q in selected])
    easy = torch.zeros((len(selected), n), dtype=torch.bool)
    hard = easy.clone()
    for i, query in enumerate(selected):
        easy[i, sorted(query.easy)] = True
        hard[i, sorted(query.hard)] = True
    ranks, answer_ranks = reference.batch_evaluate(scores.expand(len(selected), -1).clone(), (types, easy, hard),
                                                   torch.tensor(data.candidates))
    expected = reference.evaluate((ranks, torch.zeros(len(selected))), (types, answer_ranks, easy.sum(-1), hard.sum(-1)),
                                  ['mrr', 'hits@1', 'hits@3', 'hits@10'], id2type)
    for metric in ('mrr', 'hits1', 'hits3', 'hits10'):
        upstream_metric = 'mrr' if metric == 'mrr' else metric.replace('hits', 'hits@')
        for shape, upstream_shape in zip(grouped, id2type):
            assert report['per_shape'][shape][metric] == pytest.approx(expected[f'[{upstream_shape}] {upstream_metric}'])
        for group, label in (('epfo', 'EPFO'), ('negation', 'negation')):
            assert report['averages'][group][metric] == pytest.approx(expected[f'[{label}] {upstream_metric}'])
        assert report['averages']['all'][metric] == pytest.approx(expected[upstream_metric])
