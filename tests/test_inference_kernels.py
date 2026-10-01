"""Device-cache isolation, exact kernel scheduling and host-synchronization checks for H100 settings."""

import importlib

import pytest
import torch

from dicee.models import ULTRA
from dicee.models._inference import cached_relation_batch, float32_precision_backends
from dicee.query_answering.engine import AtomicBatchCache, AtomicScorer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
DEVICES = ['cpu', pytest.param('cuda', marks=CUDA)]


class TableModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.table = torch.nn.Parameter(values.clone())
        self.num_entities, self.num_relations = values.shape[:2]

    def forward_k_vs_all(self, pairs):
        return self.table[pairs[:, 0], pairs[:, 1]]


@pytest.fixture(autouse=True)
def execution_settings(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    for backend in float32_precision_backends():
        monkeypatch.setattr(backend, 'fp32_precision', 'ieee')
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('storage', ['cpu', 'model'])
def test_atomic_cache_placement_isolation_eviction_and_invalidation(device, storage):
    values = torch.arange(50., dtype=torch.float64).reshape(5, 2, 5).to(device)
    model = TableModel(values).eval()
    cache = AtomicBatchCache(80, device=storage)
    scorer = AtomicScorer(model, row_batch_size=2, raw_cache=cache)
    pair = [(0, 0), (1, 1)]
    expected = scorer.rows(pair)
    saved = next(iter(cache.rows.values()))
    assert saved.device.type == (device if storage == 'model' else 'cpu')
    assert saved.data_ptr() != expected.data_ptr()
    expected.zero_()
    reused = scorer.rows(pair)
    assert torch.equal(reused, values[[0, 1], [0, 1]])
    reused.zero_()
    assert torch.equal(scorer.rows(pair), saved.to(device))
    scorer.rows([(2, 0)])
    assert tuple(pair) not in cache.rows and cache.used == 40 and cache.peak == 80
    with torch.no_grad():
        model.table.add_(1)
    assert torch.equal(scorer.rows(pair), values[[0, 1], [0, 1]] + 1)
    assert list(cache.rows) == [tuple(pair)]


def test_zero_budget_and_invalid_cache_device():
    cache = AtomicBatchCache(0, device='model')
    cache.get('token', [(0, 0)])
    cache.put([(0, 0)], torch.ones(1, 5))
    assert cache.used == 0 and not cache.rows
    with pytest.raises(ValueError, match='device'):
        AtomicBatchCache(100, device='cuda')


def test_same_relation_batch_shares_storage_and_mixed_batch_preserves_order():
    values = {0: torch.randn(6, 8), 1: torch.randn(6, 8)}
    batch = cached_relation_batch(values, [1] * 32)
    assert batch.stride(0) == 0 and batch.data_ptr() == values[1].data_ptr()
    assert batch.untyped_storage().nbytes() == values[1].numel() * values[1].element_size()
    ids = [1, 0, 1, 0]
    assert torch.equal(cached_relation_batch(values, ids), torch.stack([values[key] for key in ids]))


@pytest.mark.parametrize('device', DEVICES)
def test_broadcast_cached_relations_preserve_scores_order_and_cache(device, monkeypatch):
    cls = ULTRA
    torch.manual_seed(11)
    facts = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0]])
    prefix = cls.name.lower()
    model = cls(dict(num_entities=6, num_relations=2, **{prefix + '_dim': 8},
                     ultra_num_layers=2, **{prefix + '_query_batch_size': 4})).set_graph(facts).to(device).eval().requires_grad_(False)
    queries = torch.tensor([[0, 0], [1, 0], [2, 0], [4, 0], [1, 1], [3, 0]], device=device)
    module = importlib.import_module('dicee.models.' + prefix)
    with torch.no_grad(), monkeypatch.context() as patch:
        patch.setattr(module, 'cached_relation_batch', lambda values, ids: torch.stack([values[key] for key in ids]))
        expected = model(queries)
    cached = model._relation_cache if cls is ULTRA else model._initial_cache
    before = {key: value.clone() for key, value in cached.items()}
    with torch.no_grad():
        actual = model(queries)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(actual.argsort(-1), expected.argsort(-1))
    assert all(torch.equal(cached[key], value) for key, value in before.items())


@CUDA
@pytest.mark.parametrize('dim', [7, 64, 128])
@pytest.mark.parametrize('reduction', ['sum', 'min', 'max'])
def test_one_warp_ordered_aggregation_is_bitwise_equal_to_four_warps(dim, reduction):
    pytest.importorskip('triton')
    from dicee.query_answering.methods._ordered_aggregation import layout
    from dicee.query_answering.methods._ordered_aggregation_cuda import ordered_reduce

    generator = torch.Generator().manual_seed(17)
    # Hub, empty row, repeated edges, non-contiguous states and broadcast relations.
    edges = torch.randint(10, (2, 157), generator=generator)
    edges[0, :100] = 0
    types = torch.randint(3, (157,), generator=generator)
    csr = layout(edges.cuda(), types.cuda(), 11)
    states = torch.randn(5, 11, dim * 2, generator=generator).cuda()[..., ::2].transpose(0, 1)
    relations = torch.randn(3, 1, dim, generator=generator).cuda().expand(-1, 5, -1)
    expected = ordered_reduce(states, relations, csr, reduction, num_warps=4)
    actual = ordered_reduce(states, relations, csr, reduction, num_warps=1)
    assert torch.equal(actual, expected)

def _single_pass_rows():
    """The previous one-program-per-row kernel, retained as the bitwise oracle."""
    import triton
    import triton.language as tl

    @triton.jit
    def rows(S, R, B, O, PTR, SRC, TYPE,
             SB: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr,
             RB: tl.constexpr, RN: tl.constexpr, RD: tl.constexpr,
             BB: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
             N: tl.constexpr, D: tl.constexpr, FEATURES: tl.constexpr, EDGES: tl.constexpr):
        row, batch = tl.program_id(0), tl.program_id(1)
        feature = tl.arange(0, FEATURES)
        start, end = tl.load(PTR + row), tl.load(PTR + row + 1)
        total = tl.full((FEATURES,), 0, tl.float32)
        for first in range(start, end, EDGES):
            edge = first + tl.arange(0, EDGES)
            source = tl.load(SRC + edge, edge < end, other=0)
            relation = tl.load(TYPE + edge, edge < end, other=0)
            mask = (edge[:, None] < end) & (feature[None, :] < D)
            state = tl.load(S + batch * SB + source[:, None] * SN + feature[None, :] * SD, mask, other=0)
            rel = tl.load(R + batch * RB + relation[:, None] * RN + feature[None, :] * RD, mask, other=0)
            total += tl.sum(state.to(tl.float32) * rel.to(tl.float32), axis=0)
        boundary = tl.load(B + batch * BB + row * BN + feature * BD, feature < D, other=0)
        tl.store(O + (batch * N + row) * D + feature, total + boundary, feature < D)

    def launch(states, boundary, relations, layout):
        batch, nodes, dim = states.shape
        output = torch.empty_like(states, memory_format=torch.contiguous_format)
        rows[(nodes, batch)](states, relations, boundary, output, layout.offsets, layout.sources, layout.types,
                             *states.stride(), *relations.stride(), *boundary.stride(),
                             nodes, dim, triton.next_power_of_2(dim), 32, num_warps=4, enable_fp_fusion=False)
        return output
    return launch


def _hub_graph(nodes, edges, types, generator, device):
    index = torch.randint(nodes, (2, edges), generator=generator)
    index[0, :edges // 3] = 1                      # one hub with hundreds of chunks
    index[0, edges // 3:edges // 2] = nodes - 1    # a second hub at the last row
    index[0, index[0] == 2] = 3                    # an empty row
    return index.to(device), torch.randint(types, (edges,), generator=generator).to(device)


@CUDA
@pytest.mark.parametrize('batch', [1, 3, 5, 16])
@pytest.mark.parametrize('dim', [7, 32, 64])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_hub_split_rows_are_bitwise_equal_to_single_pass_rows(batch, dim, dtype):
    pytest.importorskip('triton')
    from dicee.models._fused_message import HUB_TILES, csr_layout
    from dicee.models._triton_message import distmult_sum
    generator = torch.Generator().manual_seed(batch * 1000 + dim)
    nodes, types = 97, 11
    edges, edge_types = _hub_graph(nodes, 20000, types, generator, 'cuda')
    layout = csr_layout(edges, edge_types, nodes)
    assert len(layout.hub_rows) and (layout.hub_first >= 0).sum() == 2 and 32 * HUB_TILES < 20000 // 6
    # Strided states and a batch-broadcast relation table exercise stride handling.
    states = torch.randn(batch, nodes, 2 * dim, generator=generator)[..., ::2].to('cuda', dtype)
    relations = torch.randn(1, types, dim, generator=generator).to('cuda', dtype).expand(batch, -1, -1)
    boundary = torch.randn(batch, nodes, dim, generator=generator).to('cuda', dtype)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual = distmult_sum(states, boundary, relations, layout)
    finally:
        torch.use_deterministic_algorithms(previous)
    expected = _single_pass_rows()(states, boundary, relations, layout)
    assert torch.equal(actual.view(torch.int16 if dtype != torch.float32 else torch.int32),
                       expected.view(torch.int16 if dtype != torch.float32 else torch.int32))


def _sequential_ordered():
    """The previous one-edge-per-iteration ordered kernel, retained as the bitwise oracle."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def reduce(S, R, O, PTR, SRC, REL,
               SN: tl.constexpr, SB: tl.constexpr, SD: tl.constexpr,
               RN: tl.constexpr, RB: tl.constexpr, RD: tl.constexpr,
               B: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr, KIND: tl.constexpr):
        row, batch = tl.program_id(0), tl.program_id(1)
        feature = tl.arange(0, WIDTH)
        start, end = tl.load(PTR + row), tl.load(PTR + row + 1)
        value = tl.full((WIDTH,), 0. if KIND == 'sum' else -3.4028234663852886e38 if KIND == 'max' else 3.4028234663852886e38, tl.float32)
        for edge in range(start, end):
            source, relation = tl.load(SRC + edge), tl.load(REL + edge)
            state = tl.load(S + source * SN + batch * SB + feature * SD, feature < D, other=0)
            rel = tl.load(R + relation * RN + batch * RB + feature * RD, feature < D, other=0)
            message = libdevice.mul_rn(state, rel)
            if KIND == 'sum':
                value = libdevice.add_rn(value, message)
            elif KIND == 'max':
                value = tl.maximum(value, message)
            else:
                value = tl.minimum(value, message)
        tl.store(O + (row * B + batch) * D + feature, value, feature < D)

    def launch(states, relations, edges, kind):
        nodes, batch, dim = states.shape
        output = torch.empty((nodes, batch, dim), device=states.device)
        reduce[(nodes, batch)](states, relations, output, *edges, *states.stride(), *relations.stride(),
                               batch, dim, triton.next_power_of_2(dim), kind, num_warps=4, enable_fp_fusion=False)
        return output
    return launch


@CUDA
@pytest.mark.parametrize('batch', [1, 6])
@pytest.mark.parametrize('dim', [7, 32, 64])
@pytest.mark.parametrize('warps', [1, 4])
def test_scheduled_unrolled_ordered_reductions_are_bitwise_equal(batch, dim, warps):
    pytest.importorskip('triton')
    from dicee.query_answering.methods._ordered_aggregation import aggregate, layout
    from dicee.query_answering.methods._ordered_aggregation_cuda import _schedule
    generator = torch.Generator().manual_seed(batch * 100 + dim)
    nodes, types = 61, 5
    edges, edge_types = _hub_graph(nodes, 3001, types, generator, 'cuda')
    csr = layout(edges, edge_types, nodes, relation_order=dim == 32)
    order, hubs = _schedule(csr[0])
    assert hubs == 2 and order[:2].tolist() == [1, nodes - 1] and sorted(order.tolist()) == list(range(nodes))
    # (N, B, D) views of batch-major storage, as used by UltraQuery, and broadcast relations.
    states = torch.randn(batch, nodes, dim, generator=generator).cuda().transpose(0, 1)
    states[3, 0, 0] = 1e30  # cancellation is order sensitive
    relations = torch.randn(types, 1, dim, generator=generator).cuda().expand(-1, batch, -1)
    oracle = _sequential_ordered()
    for kind in ('sum', 'max', 'min'):
        assert torch.equal(aggregate(states, relations, csr, kind, num_warps=warps), oracle(states, relations, csr, kind))
    fused = aggregate(states, relations, csr, 'pna', num_warps=warps)
    separate = (oracle(states, relations, csr, 'sum'), oracle(states.square(), relations.square(), csr, 'sum'),
                oracle(states, relations, csr, 'max'), oracle(states, relations, csr, 'min'))
    assert all(torch.equal(a, b) for a, b in zip(fused, separate))
    cpu = aggregate(states.cpu(), relations.cpu(), tuple(x.cpu() for x in csr), 'pna')
    for actual, expected in zip(fused, cpu):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@CUDA
def test_hub_split_handles_graphs_without_hubs_and_empty_batches():
    pytest.importorskip('triton')
    from dicee.models._fused_message import csr_layout
    from dicee.models._triton_message import distmult_sum
    edges = torch.tensor([[0, 0, 1, 3], [1, 2, 0, 1]], device='cuda')
    types = torch.tensor([0, 1, 0, 1], device='cuda')
    layout = csr_layout(edges, types, 4)
    assert not len(layout.hub_rows) and (layout.hub_first < 0).all()
    states, boundary = torch.randn(2, 4, 5, device='cuda'), torch.randn(2, 4, 5, device='cuda')
    relations = torch.randn(2, 2, 5, device='cuda')
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual = distmult_sum(states, boundary, relations, layout)
        assert distmult_sum(states[:0], boundary[:0], relations[:0], layout).shape == (0, 4, 5)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert torch.equal(actual, _single_pass_rows()(states, boundary, relations, layout))


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
@CUDA
def test_deferred_device_checks_raise_and_drop_rejected_rows(bad):
    from dicee.query_answering import QueryAnswerer
    values = torch.randn(4, 2, 4, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    model = TableModel(values).cuda().eval()
    with torch.no_grad():
        model.table[1, 1, 2] = bad
    engine = QueryAnswerer(model)
    with pytest.raises(ValueError, match='finite complete'):
        engine.predict((0, (0, 1)))
    assert not engine._cache and not model.training
    with torch.no_grad():
        model.table[1, 1, 2] = 0.
    # A fresh engine shows that no rejected row survived in either cache.
    expected = QueryAnswerer(TableModel(model.table.detach()).cuda().eval()).predict((0, (0, 1)))
    assert torch.equal(engine.predict((0, (0, 1))), expected)


def test_scorers_share_model_context_until_the_graph_changes(monkeypatch):
    from dicee.query_answering import QueryAnswerer, QueryContext
    model = ULTRA(dict(num_entities=3, num_relations=2, ultra_dim=8, ultra_num_layers=2))
    model.set_graph([(0, 0, 1)], inverse_relations={0: 1}).eval()
    calls, original = [], QueryContext.from_model
    monkeypatch.setattr(QueryContext, 'from_model', classmethod(lambda cls, m: calls.append(m) or original(m)))
    first, second = QueryAnswerer(model), QueryAnswerer(model)
    assert len(calls) == 1 and first.scorer.context is second.scorer.context
    model.set_graph([(0, 0, 2)], inverse_relations={0: 1})
    first.predict((0, (0,)))
    second.predict((0, (0,)))
    assert len(calls) == 2 and first.scorer.context is second.scorer.context
    assert (0, 0, 2) in first.scorer.context.triples
