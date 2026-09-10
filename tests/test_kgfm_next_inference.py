"""Regression tests for inference reuse, compact walks, and fused rank bounds."""
from pathlib import Path

import pytest
import torch

from dicee.evaluation._filtering import FilteredRanker
from dicee.models import TRIX, ULTRA, Flock, FlockRelation
from dicee.models._fused_message import fused_distmult_sum
from dicee.models.flock_walks import WalkGraph

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
FACTS = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0]])


@pytest.fixture(autouse=True)
def small_thread_pool(monkeypatch):
    previous = torch.get_num_threads()
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        for backend in (torch.backends.cuda.matmul, torch.backends.mkldnn.matmul,
                        torch.backends.cudnn.conv, torch.backends.cudnn.rnn):
            monkeypatch.setattr(backend, 'fp32_precision', 'ieee')
    else:
        monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
        monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    try:
        torch.set_num_threads(2)
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('cls,cache_name', [(ULTRA, '_projection_cache'), (TRIX, '_initial_cache')])
def test_cached_stages_invalidate_and_preserve_autograd(cls, cache_name):
    prefix = cls.name.lower()
    model = cls(dict(num_entities=5, num_relations=2, **{prefix + '_dim': 8},
                     ultra_num_layers=2, graph_projection_cache_mb=0.001, graph_relation_cache_mb=0.001)).set_graph(FACTS).eval()
    queries = torch.tensor([[0, 0], [1, 0], [2, 1]])
    keys = set(model.state_dict())
    stage = model.entity_model.layers[0].relation_projection if cls is ULTRA else model.relation_model.layers_hh[0]
    calls = []
    handle = stage.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        scores = model(queries)
        assert getattr(model, cache_name)
        calls.clear()
        torch.testing.assert_close(model(queries), scores)
        assert not calls
        next(stage.parameters()).add_(0.1)
        model(queries)
        assert calls
        calls.clear()
        model.load_state_dict(model.state_dict())
        model(queries)
        assert calls
        model.set_graph(FACTS[:3])
        assert not getattr(model, cache_name)
        model(queries)
        model.double()
        assert not getattr(model, cache_name)
        model(queries)
        if cls is ULTRA:
            model.projection_cache_mb = 0
        else:
            model.relation_cache_mb = 0
        model(queries)
        assert not getattr(model, cache_name)
    assert set(model.state_dict()) == keys
    model.train().eval()
    model(queries).sum().backward()
    assert all(p.grad is not None for name, p in model.named_parameters() if not name.startswith('entity_model_1.mlp.'))
    handle.remove()


@pytest.mark.parametrize('cls', [ULTRA, TRIX])
def test_cache_eviction_and_inference_tensor_bypass(cls):
    prefix = cls.name.lower()
    settings = dict(num_entities=5, num_relations=2, **{prefix + '_dim': 8}, ultra_num_layers=2,
                    graph_relation_cache_mb=128 / 2**20, graph_projection_cache_mb=256 / 2**20)
    model = cls(settings).set_graph(FACTS).eval().requires_grad_(False)
    query = FACTS[:, :2]
    with torch.no_grad():
        expected = model(query)
        cache = model._projection_cache if cls is ULTRA else model._initial_cache
        assert len(cache) <= 1
        torch.testing.assert_close(model(query), expected)
    with torch.inference_mode():
        other = cls(settings).set_graph(FACTS).eval().requires_grad_(False)
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(other(query), expected)
        assert other.inference_token() is None


@pytest.mark.parametrize('cls', [ULTRA, TRIX])
def test_all_candidate_gather_matches_reordered_subset(cls):
    prefix = cls.name.lower()
    model = cls(dict(num_entities=5, num_relations=2, **{prefix + '_dim': 8}, ultra_num_layers=2,
                     **{prefix + '_query_batch_size': 2})).set_graph(FACTS).eval()
    candidates = torch.tensor([4, 1, 1, 0])
    with torch.no_grad():
        tails = model.forward_k_vs_all(FACTS[:, :2])
        heads = model.forward_k_vs_all_heads(FACTS[:, 1:])
        torch.testing.assert_close(model.forward_k_vs_sample(FACTS[:, :2], candidates), tails[:, candidates])
        torch.testing.assert_close(model.forward_k_vs_all_heads(FACTS[:, 1:], candidates), heads[:, candidates])
        assert model.forward_k_vs_sample(FACTS[:, :2], candidates[:0]).shape == (5, 0)


@pytest.mark.parametrize('cls', [Flock, FlockRelation])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
def test_compact_flock_matches_fixed_walks_including_unvisited_queries(cls, device):
    model = cls(dict(num_entities=100, num_relations=2, flock_dim=8, flock_refinements=2,
                     flock_walk_num=2, flock_walk_len=8, flock_seed=42)).set_graph(FACTS).eval().to(device)
    heads, queries = torch.tensor([0, 1], device=device), torch.tensor([1, 0], device=device)
    # Sample without query endpoints so the relation predictor also exercises
    # default-state lookup for an endpoint absent from every walk.
    records = model.sample_walks(torch.tensor([0, 1]))
    if cls is FlockRelation:
        heads = torch.tensor([99, 98], device=device)
        candidates = torch.tensor([[0, 1, 0], [1, 0, 1]], device=device)
    else:
        candidates = torch.arange(100, device=device).expand(2, -1)
    with torch.no_grad():
        model.compact_state = False
        expected = model.score_walks(heads, queries, candidates, records)
        model.compact_state = True
        torch.testing.assert_close(model.score_walks(heads, queries, candidates, records), expected, atol=2e-5, rtol=2e-5)
        subset = candidates[:, [2, 0, 2]]
        torch.testing.assert_close(model.score_walks(heads, queries, subset, records), expected[:, [2, 0, 2]], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('tails', [None, torch.tensor([1, 4])])
@pytest.mark.parametrize('empty', [False, True])
def test_compiled_sampler_preserves_all_records_and_rng(tails, empty):
    model = Flock(dict(num_entities=5, num_relations=2)).set_graph(FACTS)
    edges, types = model.edge_index, model.edge_type
    if empty:
        edges, types = edges[:, :0], types[:0]
    graph = WalkGraph(edges, types, 5, 4)
    eager = torch.Generator().manual_seed(912)
    compiled = torch.Generator().manual_seed(912)
    graph.compile_sampler = False
    expected = graph.sample(torch.tensor([0, 4]), tails, 3, 8, 2, eager)
    graph.compile_sampler = True
    actual = graph.sample(torch.tensor([0, 4]), tails, 3, 8, 2, compiled)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(eager.get_state(), compiled.get_state())


@CUDA
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('dense', [False, True])
@pytest.mark.parametrize('deterministic', [False, True])
def test_low_precision_fused_messages_accumulate_in_float32(dtype, dense, deterministic):
    pytest.importorskip('triton')
    torch.manual_seed(7)
    edges = torch.randint(19, (2, 2000 if dense else 40), device='cuda')
    types = torch.randint(4, (edges.shape[1],), device='cuda')
    states = torch.randn(2, 20, 32, device='cuda').to(dtype)
    relations = torch.randn(2, 4, 32, device='cuda').to(dtype)
    boundary = torch.randn_like(states)
    expected = boundary.float().index_add(1, edges[0], states.float()[:, edges[1]] * relations.float()[:, types]).to(dtype)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        with torch.no_grad():
            actual = fused_distmult_sum(states, boundary, edges, types, relations)
        assert actual is not None and actual.dtype == dtype
        torch.testing.assert_close(actual, expected, atol=0.004 if dtype == torch.float16 else 0.04, rtol=0.005)
    finally:
        torch.use_deterministic_algorithms(previous)


@CUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_fused_rank_bounds_repeated_rows_filters_infinities_and_nans(dtype):
    pytest.importorskip('triton')
    torch.manual_seed(7)
    scores = torch.randint(-5, 6, (3, 5001)).to(dtype)
    scores[0, 17] = float('nan')
    scores[1, 99] = -float('inf')
    scores[2, 101] = float('inf')
    rows, targets = [2, 0, 1, 0], [101, 28, 99, 28]
    filters = [[101, 102, 102], [17, 17, 28, 10], [1, 99], [17]]
    ranker = FilteredRanker('random', 42)
    expected = ranker.bounds_batch(scores, targets, filters, row_indices=rows)
    # Preserve noncontiguous columns and rows.
    backing = torch.empty(3, 10002, dtype=dtype, device='cuda')
    backing[:, ::2] = scores.cuda()
    actual = ranker.bounds_batch(backing[:, ::2], targets, filters, row_indices=rows)
    assert actual == expected
    with pytest.raises(ValueError, match='NaN'):
        ranker.bounds_batch(backing[:, ::2], [28], [[]], row_indices=[0])
    with pytest.raises(ValueError, match='NaN'):
        ranker.bounds_batch(backing[:, ::2], [17], [[17]], row_indices=[0])


@CUDA
@pytest.mark.parametrize('cls,fixture_name', [(ULTRA, 'ultra/tiny.pt'), (TRIX, 'trix/entity_tiny.pt')])
def test_compiled_updates_keep_checkpoint_and_fixture_scores(cls, fixture_name):
    pytest.importorskip('triton')
    fixture = torch.load(Path(__file__).parent / 'fixtures' / fixture_name, weights_only=True)
    prefix = cls.name.lower()
    settings = dict(num_entities=fixture['num_entities'], num_relations=fixture['num_relations'],
                    **{prefix + '_dim': fixture['dim']}, graph_inference_compile=True)
    if cls is ULTRA:
        settings['ultra_num_layers'] = fixture['num_layers']
    model = cls(settings).eval().requires_grad_(False).cuda()
    model.load_state_dict(fixture['state_dict'])
    model.set_graph(fixture['triples'])
    keys = set(model.state_dict())
    with torch.no_grad():
        for _ in range(3):
            actual = model(fixture['queries'][:, :2]).cpu()
            torch.testing.assert_close(actual, fixture['tails'], atol=1e-5, rtol=1e-4)
    assert set(model.state_dict()) == keys


@CUDA
@pytest.mark.parametrize('cls,task', [(Flock, 'entity'), (FlockRelation, 'relation')])
def test_flock_fused_fixed_walks_match_official_checkpoint(cls, task):
    pytest.importorskip('triton')
    fixture = torch.load(Path(__file__).parent / 'fixtures' / 'flock' / (task + '_pretrained.pt'), weights_only=True)
    checkpoint = Path(__file__).parents[1] / 'checkpoints' / 'flock' / ('flock_' + task + '.pth')
    if not checkpoint.exists():
        pytest.skip('Official checkpoint unavailable')
    model = cls(dict(num_entities=7, num_relations=3, flock_query_batch_size=8)).load_pretrained(checkpoint).set_graph(fixture['triples']).eval().requires_grad_(False).cuda()
    case = next(case for name, case in fixture['cases'].items() if name != 'training')
    model._draw_walks = lambda *args: case['records']
    with torch.no_grad():
        actual = model(case['groups']).cpu()
    torch.testing.assert_close(actual, case['scores'], atol=2e-5, rtol=1e-4)


@CUDA
@pytest.mark.parametrize('nodes,relations,length', [(100, 2, 8), (32769, 200, 257)])
def test_packed_walk_transfers_preserve_integer_values(nodes, relations, length):
    model = Flock(dict(num_entities=nodes, num_relations=relations, flock_walk_len=length)).set_graph(FACTS).eval().cuda()
    maxima = (nodes - 1, length, 1, 1, 2 * relations, length + 1, 3)
    records = tuple(torch.tensor([0, limit], dtype=torch.long).view(1, 1, 1, 2) for limit in maxima)
    packed = model._transfer_records(records)
    assert all(r.is_pinned() for r in packed)
    assert sum(r.numel() * r.element_size() for r in packed) < sum(r.numel() * r.element_size() for r in records)
    restored = tuple(r.to(device='cuda', dtype=torch.long, non_blocking=True).cpu() for r in packed)
    assert all(torch.equal(a, b) for a, b in zip(records, restored))


@CUDA
def test_frozen_convolution_keeps_input_gradients():
    from dicee.models.ultra import RelationalConv

    conv = RelationalConv(8).eval().requires_grad_(False).cuda()
    states = torch.randn(2, 5, 8, device='cuda', requires_grad=True)
    boundary = torch.randn_like(states, requires_grad=True)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], device='cuda')
    types = torch.tensor([0, 1, 2], device='cuda')
    conv(states, boundary, edges, types, residual=True).sum().backward()
    assert states.grad is not None and boundary.grad is not None
