"""Inference kernel parity, cache lifecycle, query reuse, and sampling order."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dicee.evaluation._filtering import TIE_POLICIES, FilteredRanker
from dicee.evaluation._graph_inference import GraphRankPlan
from dicee.evaluation.link_prediction import evaluate_lp
from dicee.models import TRIX, ULTRA, Flock, TRIXRelation
from dicee.models._fused_message import csr_layout, fused_distmult_sum

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@CUDA
@pytest.mark.parametrize('dim', [7, 32, 64])
@pytest.mark.parametrize('dense', [False, True])
def test_fused_messages_match_torch_with_strides_and_isolated_nodes(dim, dense):
    pytest.importorskip('triton')
    generator = torch.Generator().manual_seed(45)
    nodes, types, batch = 23, 5, 3
    edges = torch.randint(nodes - 1, (2, 1500 if dense else 40), generator=generator).cuda()
    edge_types = torch.randint(types, (edges.shape[1],), generator=generator).cuda()
    # Non-contiguous features and broadcast relation rows exercise actual
    # relation/entity reasoner layouts, including the last isolated node.
    states = torch.randn(batch, nodes, dim * 2, generator=generator).cuda()[..., ::2]
    relations = torch.randn(1, types, dim, generator=generator).cuda().expand(batch, -1, -1)
    boundary = torch.randn(batch, nodes, dim, generator=generator).cuda()
    with torch.no_grad():
        expected = boundary.index_add(1, edges[0], states[:, edges[1]] * relations[:, edge_types])
        actual = fused_distmult_sum(states, boundary, edges, edge_types, relations)
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(actual[:, -1], boundary[:, -1], atol=0, rtol=0)
        # CSR output row/input column must not accidentally follow PyG direction.
        reverse = fused_distmult_sum(states, boundary, edges.flip(0), edge_types, relations)
        assert not torch.allclose(actual, reverse)
        empty = fused_distmult_sum(states, boundary, edges[:, :0], edge_types[:0], relations)
        torch.testing.assert_close(empty, boundary, atol=0, rtol=0)


def test_layout_invalidates_after_in_place_graph_change():
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    types = torch.tensor([0, 1, 0])
    first = csr_layout(edges, types, 3)
    assert csr_layout(edges, types, 3) is first
    types[0] = 1
    second = csr_layout(edges, types, 3)
    assert second is not first
    assert second[2].tolist() == [1, 1, 0]
    edges[0, 0] = 2
    assert csr_layout(edges, types, 3) is not second


@CUDA
def test_deterministic_fused_reduction_and_inference_mode():
    pytest.importorskip('triton')
    edges = torch.tensor([[0, 0, 1, 1], [1, 2, 0, 2]], device='cuda').repeat(1, 200)
    types = torch.tensor([0, 1, 0, 1], device='cuda').repeat(200)
    states = torch.randn(2, 3, 32, device='cuda')
    relations = torch.randn(2, 2, 32, device='cuda')
    boundary = torch.zeros_like(states)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.inference_mode():
            first = fused_distmult_sum(states, boundary, edges, types, relations)
            second = fused_distmult_sum(states, boundary, edges, types, relations)
        assert torch.equal(first, second)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
    assert fused_distmult_sum(states.requires_grad_(), boundary, edges, types, relations) is None


@CUDA
@pytest.mark.parametrize('cls,fixture_name', [(ULTRA, 'ultra/tiny.pt'), (TRIX, 'trix/entity_tiny.pt'),
                                           (TRIXRelation, 'trix/relation_tiny.pt')])
def test_fused_model_matches_upstream_fixture(cls, fixture_name):
    pytest.importorskip('triton')
    fixture = torch.load(Path(__file__).parent/'fixtures'/fixture_name, weights_only=True)
    prefix = 'ultra' if cls is ULTRA else 'trix'
    settings = dict(num_entities=fixture['num_entities'], num_relations=fixture['num_relations'],
                    **{prefix + '_dim': fixture['dim']})
    if cls is ULTRA:
        settings['ultra_num_layers'] = fixture['num_layers']
    model = cls(settings)
    model.load_state_dict(fixture['state_dict'])
    model.set_graph(fixture['triples']).eval().cuda()
    query = fixture['queries']
    with torch.inference_mode():
        if cls is TRIXRelation:
            torch.testing.assert_close(model(query[:, [0, 2]]).cpu(), fixture['scores'], atol=1e-5, rtol=1e-4)
        else:
            torch.testing.assert_close(model(query[:, :2]).cpu(), fixture['tails'], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(model.forward_k_vs_all_heads(query[:, 1:]).cpu(), fixture['heads'], atol=1e-5, rtol=1e-4)


def test_flock_cached_sampler_matches_original_records_and_generator_state():
    # Golden digests generated with the untouched 91244d3e sampler, including
    # isolated node 4 and both relation directions. No neural computation.
    facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0]])
    model = Flock(dict(num_entities=5, num_relations=2, flock_dim=8, flock_walk_num=2,
                       flock_walk_len=8, flock_refinements=2)).set_graph(facts)
    generator = torch.Generator().manual_seed(42)
    records = model._draw_walks(model._walk_graph, torch.tensor([0, 4]), None, generator)
    digest = hashlib.sha256(b''.join(r.numpy().tobytes() for r in records)).hexdigest()
    assert digest == '41a67a27b5b49d52f6800b5875ed63e9899074a3133a9b9995b329277b132975'
    assert hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest() == '818ae92ebf4dc736aba8613ada3835d63c92af4737229b6ccb5f7198bba4e336'


def test_ultra_relation_cache_reuses_and_invalidates_without_changing_checkpoint():
    facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]])
    model = ULTRA(dict(num_entities=4, num_relations=2, ultra_dim=8, ultra_num_layers=2)).set_graph(facts).eval()
    keys = set(model.state_dict())
    query = torch.tensor([[0, 0], [1, 0], [2, 1]])
    calls = []
    handle = model.relation_model.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        expected = model(query)
        first_calls = len(calls)
        torch.testing.assert_close(model(query), expected)
        assert len(calls) == first_calls
        assert len(model._relation_cache) == 2
        parameter = next(model.relation_model.parameters())
        parameter.add_(0.01)
        model(query)
        assert len(calls) > first_calls
        calls.clear()
        model.set_graph(facts[:2])
        assert not model._relation_cache
        model(query)
        assert calls
        model.double()
        assert not model._relation_cache
        model(query)
        assert all(value.dtype == torch.float64 for value in model._relation_cache.values())
    assert set(model.state_dict()) == keys
    model.train()
    assert not model._relation_cache
    model.eval()
    model(query).sum().backward()  # eval with autograd must not use detached cache.
    assert all(p.grad is not None for p in model.parameters())
    handle.remove()


def test_relation_cache_respects_memory_limit_and_weight_reload():
    facts = torch.tensor([[0, 0, 1], [1, 1, 2]])
    # One representation is 4 * 8 * 4 = 128 bytes; retain just one.
    model = ULTRA(dict(num_entities=3, num_relations=2, ultra_dim=8, ultra_num_layers=1,
                       graph_relation_cache_mb=128 / 2**20)).set_graph(facts).eval()
    with torch.no_grad():
        model(facts[:, :2])
        assert len(model._relation_cache) == 1
        state = {key: value.clone() for key, value in model.state_dict().items()}
        model.load_state_dict(state)
        actual = model(facts[:, :2])
        model.relation_cache_mb = 0
        expected = model(facts[:, :2])
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('cls', [ULTRA, TRIX])
@pytest.mark.parametrize('backend_name', ['legacy', '', 'cuda.matmul', 'mkldnn.matmul', 'cudnn.conv', 'cudnn.rnn'])
def test_relation_cache_tracks_backend_precision(cls, backend_name, monkeypatch):
    if backend_name == 'legacy':
        backend = torch.backends.cuda.matmul
        if hasattr(backend, 'fp32_precision'):
            pytest.skip('Legacy precision fallback is only used on older PyTorch')
        attribute, initial, changed = 'allow_tf32', False, True
    else:
        backend = torch.backends
        for name in backend_name.split('.') if backend_name else []:
            backend = getattr(backend, name, None)
        if not hasattr(backend, 'fp32_precision'):
            pytest.skip('Backend-specific precision settings require newer PyTorch')

        # New precision settings can make the legacy getters raise, even on CPU.
        def legacy_getter():
            raise AssertionError('Cache validation must use backend-specific precision settings')

        monkeypatch.setattr(torch, 'get_float32_matmul_precision', legacy_getter)
        attribute, initial, changed = 'fp32_precision', 'bf16' if backend_name == 'mkldnn.matmul' else 'ieee', 'tf32'
    monkeypatch.setattr(backend, attribute, initial)
    facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]])
    model = cls(dict(num_entities=4, num_relations=2, ultra_dim=8, trix_dim=8,
                     ultra_num_layers=2)).set_graph(facts).eval()
    stage = model.relation_model if cls is ULTRA else model.relation_model.layers_hh[0]
    calls = []
    handle = stage.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        expected = model(facts[:, :2])
        assert calls
        calls.clear()
        torch.testing.assert_close(model(facts[:, :2]), expected)
        assert not calls
        # The cache must be recomputed when either backend changes precision.
        setattr(backend, attribute, changed)
        actual = model(facts[:, :2])
        assert calls
        model.clear_inference_cache()
        torch.testing.assert_close(actual, model(facts[:, :2]))
    handle.remove()


@CUDA
def test_loaded_frozen_inference_uses_kernel_without_disabling_global_grad(monkeypatch):
    pytest.importorskip('triton')
    from dicee.knowledge_graph_embeddings import KGE
    from dicee.models import ultra

    facts = torch.tensor([[0, 0, 1], [1, 0, 2]])
    model = ULTRA(dict(num_entities=3, num_relations=1, ultra_dim=8, ultra_num_layers=2)).set_graph(facts).eval().requires_grad_(False).cuda()
    wrapper = SimpleNamespace(model=model, entity_to_idx={'e0': 0, 'e1': 1, 'e2': 2},
                              relation_to_idx={'r': 0}, all_have_inverse=False)
    original = ultra.fused_distmult_sum
    fused = []

    def observe(*args, **kwargs):
        update = original(*args, **kwargs)
        fused.append(update is not None)
        return update

    monkeypatch.setattr(ultra, 'fused_distmult_sum', observe)
    assert torch.is_grad_enabled()
    scores, indices = KGE.predict_missing_head_entity(wrapper, 'r', 'e1', within=['e2', 'e0'], topk=2, return_indices=True)
    assert fused and all(fused)
    assert model._relation_cache
    assert not scores.requires_grad and not scores.is_cuda
    candidates = torch.tensor([2, 0])
    expected = model.forward_k_vs_all_heads(torch.tensor([[0, 1]]), candidates).cpu().flatten()
    sorted_scores, order = expected.topk(2)
    torch.testing.assert_close(scores, sorted_scores)
    assert torch.equal(indices, candidates[order])


class CountingGraph(torch.nn.Module):
    deterministic_inference = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def inference_token(self):
        return self.weight._version

    def forward_k_vs_all(self, queries):
        self.calls.extend((False, *q) for q in queries.tolist())
        return torch.tensor([1., 2., 2., 0., 2., -torch.inf]).expand(len(queries), -1)

    def forward_k_vs_all_heads(self, queries):
        self.calls.extend((True, *q) for q in queries.tolist())
        return torch.tensor([2., 2., 0., 2., 1., -torch.inf]).expand(len(queries), -1)


@pytest.mark.parametrize('policy', TIE_POLICIES)
def test_whole_split_query_reuse_keeps_targets_ties_and_resume_order(policy):
    triples = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 2], [0, 0, 1], [1, 0, 2], [0, 0, 2]])
    er, re = {(0, 0): [1, 2], (1, 0): [1, 2]}, {(0, 1): [0, 1], (0, 2): [0, 1]}
    model = CountingGraph()
    expected = evaluate_lp(model, triples, 6, er, re, batch_size=2, tie_policy=policy, tie_seed=7, reuse_graph_queries=False)
    model.calls.clear()
    plan = GraphRankPlan(triples)
    generator = torch.Generator().manual_seed(7) if policy == 'random' else None
    parts = []
    for part in np.split(triples, [2, 4]):
        parts.append(evaluate_lp(model, part, 6, er, re, batch_size=2, tie_policy=policy,
                                 tie_seed=7, tie_generator=generator, graph_rank_plan=plan))
    assert {key: sum(part[key] for part in parts) / 3 for key in expected} == pytest.approx(expected)
    assert len(model.calls) == len(set(model.calls)) == plan.scored_queries == 4
    # A changed checkpoint cannot inherit ranks from the previous model state.
    with torch.no_grad():
        model.weight.add_(1)
    evaluate_lp(model, triples, 6, er, re, graph_rank_plan=plan, tie_policy=policy)
    assert plan.scored_queries == 8


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('policy', ['optimistic', 'pessimistic', 'random'])
def test_vectorized_ranking_matches_scalar_with_infinities_and_filters(device, policy):
    scores = torch.tensor([[2., 2., 1., float('nan'), -torch.inf],
                           [-torch.inf, -torch.inf, 2., 3., 4.]], device=device)
    targets, filters = [0, 0], [[0, 3, 3], [0, 1, 3]]
    scalar = FilteredRanker(policy, 91)
    expected = [scalar.rank(row, target, filt) for row, target, filt in zip(scores, targets, filters)]
    assert FilteredRanker(policy, 91).rank_batch(scores, targets, filters) == expected
    with pytest.raises(ValueError, match='NaN'):
        FilteredRanker(policy).rank_batch(scores, targets, [[], filters[1]])


@CUDA
@pytest.mark.parametrize('seed', [None, 42])
def test_flock_prefetch_preserves_walk_draws_and_rng(seed):
    facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0]])
    model = Flock(dict(num_entities=5, num_relations=2, flock_dim=8, flock_walk_num=2,
                       flock_walk_len=8, flock_refinements=2, flock_query_batch_size=1,
                       flock_seed=seed)).set_graph(facts).eval().cuda()
    records = []
    original = model._draw_walks

    def record(*args):
        value = original(*args)
        records.append(tuple(x.clone() for x in value))
        return value

    model._draw_walks = record
    with torch.no_grad():
        torch.manual_seed(17)
        model.prefetch_walks = False
        expected = model(facts[:, :2])
        state = torch.random.get_rng_state()
        before = list(records)
        records.clear()
        torch.manual_seed(17)
        model.prefetch_walks = True
        actual = model(facts[:, :2])
    assert torch.equal(torch.random.get_rng_state(), state)
    assert len(before) == len(records)
    assert all(torch.equal(a, b) for left, right in zip(before, records) for a, b in zip(left, right))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
