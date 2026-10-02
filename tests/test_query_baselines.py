"""Native baselines against independently generated, pinned upstream outputs."""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dicee.query_answering import BenchmarkQuery, QueryBenchmark, QueryContext, evaluate_benchmark
from dicee.query_answering._query import compile_query
from dicee.query_answering.benchmark import QueryMetrics
from dicee.query_answering.methods import REFERENCES, load_method
from dicee.query_answering.methods._common import union_branches

ROOT = Path(__file__).parent / 'fixtures' / 'query_baselines'


def cases():
    result = []
    for method in ('cone', 'cqd', 'qto', 'clmpt', 'ultra', 'gnnqe'):
        path = ROOT / f'{method}.pt'
        if not path.exists():
            continue
        fixture = torch.load(path, weights_only=True, map_location='cpu')
        for index, case in enumerate(fixture['cases']):
            result.append(pytest.param(fixture, case, id=f'{method}-{index}'))
    return result


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('fixture,case', cases())
def test_checkpoint_scores_ranks_ties_and_metrics(tmp_path, fixture, case):
    context = QueryContext(fixture['triples'], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'checkpoint.pt'
    torch.save({'model_state_dict': case['state']}, path)
    model = load_method(case['name'], path, context, **case['config'])
    assert fixture['upstream_commit'] == REFERENCES[case['name']][1]
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    for shape, expected in case['scores'].items():
        actual = model.predict(fixture['queries'][shape])
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5, msg=shape)
        if case['name'] == 'cone' and shape in ('2u', 'up'):
            # The reference embeds all DNF branches in one projection batch. Batch composition changes the
            # float bits, so rebuild that joint batch on this CPU instead of comparing bits recorded on another.
            branches = union_branches(compile_query(fixture['queries'][shape]))
            joint = [values.reshape(1, len(branches), -1) for values in model.embed_batch(branches)]
            torch.testing.assert_close(actual, model.score_embeddings(joint)[0], atol=0, rtol=0,
                                       msg='ConE union branches must use the reference joint projection batch')
        assert torch.equal(actual.argsort(descending=True), expected.argsort(descending=True)), shape
        assert torch.equal(actual[:, None] == actual[None], expected[:, None] == expected[None]), shape
        for policy in ('sort', 'expected'):
            evaluator = QueryMetrics(context.num_entities, candidates=range(context.num_entities - 1), tie_policy=policy)
            assert evaluator(actual, {1}, {0, 3}) == evaluator(expected, {1}, {0, 3}), (shape, policy)
    if 'adjacency' in case:
        with torch.no_grad():
            for relation, expected in enumerate(case['adjacency']):
                torch.testing.assert_close(model.rows(torch.arange(context.num_entities), relation), expected)
    for batch in case.get('batches', {}).values():
        actual = model.predict_batch(batch['queries'])
        torch.testing.assert_close(actual, batch['scores'], atol=3e-5, rtol=3e-5)
        assert torch.equal(actual.argsort(-1), batch['scores'].argsort(-1))
        if not model.pre_norm:
            assert not torch.allclose(actual[0], model.predict(batch['queries'][0]))


def test_all_reference_methods_present():
    for method in ('cone', 'cqd', 'qto', 'clmpt', 'ultra', 'gnnqe'):
        assert (ROOT / f'{method}.pt').exists(), f'Missing upstream fixture for {method}'


def test_expected_ties_match_all_permutations():
    scores = torch.tensor([1., 1., .4, .4, .4, 0.])
    evaluator = QueryMetrics(len(scores), candidates=[0, 1, 2, 3, 4], tie_policy='expected')
    result = evaluator(scores, {1}, {2, 4})
    # After filtering: one higher negative, one tied negative, and the target.
    ranks = [2, 3]
    assert result['mrr'] == pytest.approx(sum(1 / rank for rank in ranks) / 2)
    assert result['hits1'] == 0
    assert result['hits3'] == 1
    scores = torch.ones(4)
    values = []
    for order in itertools.permutations(range(4)):
        values.append(1 / (order.index(0) + 1))
    result = QueryMetrics(4, tie_policy='expected')(scores, set(), {0})
    assert result['mrr'] == pytest.approx(sum(values) / len(values))
    assert result['hits1'] == .25
    assert result['hits3'] == .75
    assert result['mrr'] != QueryMetrics(4, tie_policy='average')(scores, set(), {0})['mrr']


@pytest.mark.parametrize('policy', ['expected', 'average', 'optimistic', 'pessimistic'])
def test_tie_bounds_match_filtered_candidate_enumeration(policy):
    generator = torch.Generator().manual_seed(42)
    for _ in range(30):
        scores = torch.randint(0, 5, (20,), generator=generator).float()
        scores[0] = -torch.inf
        candidates = set(torch.randperm(20, generator=generator)[:15].tolist())
        easy, hard = set(sorted(candidates)[:3]), set(sorted(candidates)[3:8])
        expected = []
        for target in hard:
            negatives = candidates - easy - hard
            start = 1 + sum(bool(scores[node] > scores[target]) for node in negatives)
            ties = sum(bool(scores[node] == scores[target]) for node in negatives)
            ranks = (range(start, start + ties + 1) if policy == 'expected' else
                     [start + {'average': .5, 'optimistic': 0., 'pessimistic': 1.}[policy] * ties])
            expected.append([sum(1 / r for r in ranks) / len(ranks),
                             *(sum(r <= k for r in ranks) / len(ranks) for k in (1, 3, 10))])
        result = QueryMetrics(20, candidates=candidates, tie_policy=policy)(scores, easy, hard)
        assert list(result.values()) == pytest.approx(torch.tensor(expected, dtype=torch.float64).mean(0).tolist())


def test_strict_checkpoint_and_vocabulary_validation(tmp_path):
    fixture = torch.load(ROOT / 'qto.pt', weights_only=True)
    case = fixture['cases'][0]
    context = QueryContext(fixture['triples'], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'weights.pt'
    torch.save(case['state'] | {'unexpected': torch.tensor(0.)}, path)
    with pytest.raises(ValueError, match='keys'):
        load_method('qto', path, context, **case['config'])
    torch.save(case['state'], path)
    wrong = QueryContext([], context.num_entities + 1, context.num_relations)
    with pytest.raises(ValueError, match='vocabulary'):
        load_method('qto', path, wrong, **case['config'])


@pytest.mark.parametrize('reference_batching', [False, True])
def test_qto_cache_invalidation_and_chunking(tmp_path, reference_batching):
    fixture = torch.load(ROOT / 'qto.pt', weights_only=True)
    case = fixture['cases'][0]
    context = QueryContext(fixture['triples'], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'weights.pt'
    torch.save(case['state'], path)
    model = load_method('qto', path, context, row_batch_size=2, cache_bytes=160,
                          reference_batching=reference_batching, **case['config'])
    query = fixture['queries']['3p']
    torch.testing.assert_close(model.predict(query), case['scores']['3p'])
    assert model.cache.bytes <= 160
    with torch.no_grad():
        model.model.entity_embeddings.weight.mul_(2)
    cached = model.predict(query)
    model.cache.capacity = 0
    model.cache.values.clear()
    model.cache.bytes = 0
    torch.testing.assert_close(cached, model.predict(query), rtol=0, atol=0)


def test_cqd_negation_is_not_silently_reinterpreted(tmp_path):
    fixture = torch.load(ROOT / 'cqd.pt', weights_only=True)
    case = fixture['cases'][0]
    context = QueryContext(fixture['triples'], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'weights.pt'
    torch.save(case['state'], path)
    model = load_method('cqd', path, context, **case['config'])
    with pytest.raises(ValueError, match='negated'):
        model.predict(fixture['queries']['2in'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocation guard')
def test_cqd_memory_guard_precedes_dense_scoring(tmp_path, monkeypatch):
    from dicee.query_answering.methods import cqd
    fixture = torch.load(ROOT / 'cqd.pt', weights_only=True)
    case = fixture['cases'][0]
    context = QueryContext(fixture['triples'], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'weights.pt'
    torch.save(case['state'], path)
    model = load_method('cqd', path, context, device='cuda', **case['config'])
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (0, 2**30))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(cqd, 'complex_rows', lambda *args: pytest.fail('Attempted dense allocation'))
    with pytest.raises(MemoryError, match='reference stage.*scratch'):
        model.predict(fixture['queries']['1p'])


@pytest.mark.parametrize('chunk', [1, 100])
def test_opt_in_cqd_negation_matches_cqda(tmp_path, chunk):
    fixture = torch.load(ROOT / 'cqd-negation.pt', weights_only=True)
    assert fixture['upstream_commit'] == '642ce042708be6247087c09c780b9deb47e941d3'
    context = QueryContext(fixture['triples'], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'weights.pt'
    for case in fixture['cases']:
        torch.save(case['state'], path)
        model = load_method(case['name'], path, context, row_batch_size=chunk, **case['config'])
        for shape, expected in case['scores'].items():
            actual = model.predict(fixture['queries'][shape])
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6, msg=shape)
            assert torch.equal(actual.argsort(), expected.argsort()), shape
            assert torch.equal(actual[:, None] == actual[None], expected[:, None] == expected[None]), shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA reference attention')
def test_bounded_attention_preserves_full_transformer_sequence():
    from dicee.query_answering.methods._bounded_attention import BoundedAttention
    torch.manual_seed(5)
    model = torch.nn.TransformerEncoderLayer(2000, 8, 256, .1).cuda().eval()
    inputs = torch.randn(1100, 2, 2000, device='cuda')
    with torch.no_grad():
        expected = model(inputs)
        with BoundedAttention():
            actual = model(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA ordered reduction')
@pytest.mark.parametrize('reduction', ['sum', 'max', 'min'])
def test_ordered_sparse_kernel_preserves_sequential_rounding(reduction):
    from dicee.query_answering.methods._ordered_aggregation import aggregate, layout
    torch.manual_seed(7)
    edges = torch.tensor([[0, 0, 0, 2, 2], [1, 2, 3, 1, 4]])
    types = torch.tensor([2, 1, 0, 0, 2])
    states = torch.randn(5, 3, 17)
    relations = torch.randn(4, 3, 17)
    states[:3, 0, 0] = torch.tensor([1e20, -1e20, 1.])
    indices = layout(edges, types, len(states))
    expected = aggregate(states, relations, indices, reduction)
    actual = aggregate(states.cuda(), relations.cuda(), tuple(x.cuda() for x in indices), reduction)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


def test_qto_empty_intermediate_stays_empty(tmp_path):
    fixture = torch.load(ROOT / 'qto.pt', weights_only=True)
    case = fixture['cases'][0]
    context = QueryContext([], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'weights.pt'
    torch.save(case['state'], path)
    model = load_method('qto', path, context, threshold=1., negation_scale=1.)
    assert torch.equal(model.predict(fixture['queries']['2p']), torch.zeros(context.num_entities))
    assert torch.equal(model.projection(torch.zeros(context.num_entities), 0, negate=True), torch.zeros(context.num_entities))


def test_dual_tie_protocol_resume_uses_one_prediction(tmp_path):
    data = QueryBenchmark('toy', 'transductive', 'test', QueryContext([], 5, 2),
                          tuple(BenchmarkQuery('1p', (head, (0,)), frozenset({1}), frozenset({3})) for head in range(3)),
                          tuple(range(5)))
    scores = torch.tensor([.5, .5, .4, .5, .1])
    calls = []

    def predict(query):
        calls.append(query)
        return scores

    def interrupt(done, total):
        if done == 2:
            raise InterruptedError

    options = dict(checkpoint_dir=tmp_path, checkpoint_identity={'model': 'toy'}, checkpoint_every=1,
                   additional_tie_policies=('expected',))
    with pytest.raises(InterruptedError):
        evaluate_benchmark(data, predict, progress=interrupt, **options)
    report = evaluate_benchmark(data, predict, **options)
    assert len(calls) == len(set(calls)) == 3
    for policy, values in [('sort', report), ('expected', report['additional_tie_metrics']['expected'])]:
        assert values['per_shape']['1p']['mrr'] == QueryMetrics(5, tie_policy=policy)(scores, {1}, {3})['mrr']


def test_baseline_runner_end_to_end_and_resume(tmp_path, monkeypatch):
    from dicee.query_answering.methods.cone import ConE
    from dicee.scripts.evaluate_query_methods import main
    from tests.test_query_benchmark import fixture_dataset
    fixture_dataset(tmp_path / 'data', 'FB15k237LogicalQuery')
    fixture = torch.load(ROOT / 'cone.pt', weights_only=True)
    state = fixture['cases'][0]['state']
    state['entity_embedding'] = torch.cat((state['entity_embedding'], state['entity_embedding'][:1]))
    path = tmp_path / 'weights.pt'
    torch.save(state, path)
    entry = dict(method='cone', checkpoint=str(path), dataset='FB15k237LogicalQuery', root=str(tmp_path / 'data'),
                 vocabulary_dataset='FB15k237LogicalQuery', options={'center_reg': .02})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(entry))
    args = ['--manifest', str(manifest), '--output', str(tmp_path / 'results'), '--max-queries-per-shape', '1']
    main(args)
    path = tmp_path / 'results' / 'cone-FB15k237LogicalQuery' / 'result.json'
    report = json.loads(path.read_text())
    assert report['queries'] == 14
    assert report['additional_tie_metrics']['expected']['averages']['all']['shapes'] == 14
    monkeypatch.setattr(ConE, 'predict', lambda *args: pytest.fail('Resumed evaluation recomputed a query'))
    main(args)
    assert json.loads(path.read_text())['per_shape'] == report['per_shape']


@pytest.mark.parametrize('chunk,reference_batching,final_batch', [(1, False, None), (3, False, None), (100, False, None),
                                                               (3, True, None), (3, True, 1)])
def test_cqd_chunking_preserves_stage_normalization(tmp_path, chunk, reference_batching, final_batch):
    fixture = torch.load(ROOT / 'cqd.pt', weights_only=True)
    context = QueryContext(fixture['triples'], fixture['num_entities'], fixture['num_relations'])
    path = tmp_path / 'weights.pt'
    for case in fixture['cases']:
        torch.save(case['state'], path)
        model = load_method(case['name'], path, context, row_batch_size=chunk, cache_bytes=160,
                              reference_batching=reference_batching, final_batch_size=final_batch, **case['config'])
        for shape, expected in case['scores'].items():
            actual = model.predict(fixture['queries'][shape])
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            assert torch.equal(actual.argsort(), expected.argsort())
        assert model.cache.bytes <= 160


def test_checkpoint_graphs_validate_ids_without_using_held_out_edges():
    from dicee.query_answering.methods.checkpoints import _validate_graph_buffer
    context = QueryContext([(0, 0, 1), (1, 1, 2)], 3, 2)
    graph = SimpleNamespace(_edge_list=torch.tensor([[0, 1, 0]]), _edge_weight=torch.ones(1), num_node=3, num_relation=2)
    metadata = {}
    _validate_graph_buffer('fact_graph', graph, context, metadata)
    assert not metadata['fact_graph']['used_for_inference']
    graph._edge_list = torch.tensor([[0, 2, 0]])
    with pytest.raises(ValueError, match='training facts'):
        _validate_graph_buffer('fact_graph', graph, context, metadata)
    _validate_graph_buffer('graph', graph, context, metadata)
    assert not metadata['graph']['used_for_inference']
    assert context.triples == ((0, 0, 1), (1, 1, 2))
    graph._edge_list = torch.tensor([[0, 2, 2]])
    with pytest.raises(ValueError, match='out-of-vocabulary'):
        _validate_graph_buffer('graph', graph, context, metadata)
