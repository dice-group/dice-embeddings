"""ULTRA's relation-graph options: ultra_relation_weighting and ultra_pair_channels.

The defaults must stay official ULTRA. The weights are checked by hand on a small graph and, when the ProofPFN golden
folder is available (environment variable PROOFPFN_GOLDEN), bit for bit against the relation graphs that the research
worker prepared for 24 task graphs (golden G2, golden/large/g2_worlds_k0_seed42.pt.gz).
"""
import gzip
import io
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from dicee.config import Namespace
from dicee.models._relation_weights import incidence_counts, pair_channel_weights, pair_counts, relation_count_weights
from dicee.models.ultra import ULTRA, build_relation_graph, weighted_relation_graph

# Direct triples of 3 entities and 2 relations; (0, 1) carries both relations.
DIRECT = torch.tensor([[0, 0, 1], [0, 1, 1], [0, 1, 2], [1, 1, 2]])
FACTS = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0], [3, 2, 3], [1, 2, 0]])


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small(**options) -> ULTRA:
    return ULTRA(dict(num_entities=5, num_relations=3, ultra_dim=8, ultra_num_layers=2, **options))


def tables(model: ULTRA) -> list[str]:
    return [k for k in model.state_dict() if k.startswith('relation_model.layers.') and k.endswith('.relation.weight')]


def test_defaults_are_official_ultra():
    assert (Namespace().ultra_relation_weighting, Namespace().ultra_pair_channels) == ('binary', 'off')
    model = ULTRA({})
    assert (model.relation_weighting, model.pair_channels) == ('binary', 'off')
    assert len(model.state_dict()) == 82 and sum(p.numel() for p in model.parameters()) == 168705
    assert all(model.state_dict()[k].shape == (4, 64) for k in tables(model))
    model = small().set_graph(FACTS)
    assert model.rel_edge_weight is None
    index, types = build_relation_graph(model.edge_index, model.edge_type, model.num_entities, 2 * model.num_direct_relations)
    assert torch.equal(model.rel_edge_index, index) and torch.equal(model.rel_edge_type, types)


def test_option_validation():
    with pytest.raises(ValueError, match='ultra_relation_weighting'):
        small(ultra_relation_weighting='counts')
    with pytest.raises(ValueError, match='ultra_pair_channels must'):
        small(ultra_relation_weighting='normalized', ultra_pair_channels='yes')
    with pytest.raises(ValueError, match='requires'):
        small(ultra_pair_channels='on')


def test_initialization_keeps_ultra_draws():
    """The count option draws nothing; the pair option draws one 6-row table per relation layer after all of ULTRA."""
    torch.manual_seed(7)
    plain = small().state_dict()
    torch.manual_seed(7)
    count = small(ultra_relation_weighting='normalized').state_dict()
    torch.manual_seed(7)
    pair = small(ultra_relation_weighting='normalized', ultra_pair_channels='on')
    torch.manual_seed(7)
    small()
    later = [nn.Embedding(6, 8).weight.detach() for _ in range(2)]
    assert list(count) == list(plain) and all(torch.equal(count[k], plain[k]) for k in plain)
    state = pair.state_dict()
    assert list(state) == list(plain) and len(tables(pair)) == 2
    for key in plain:
        if key in tables(pair):
            assert state[key].shape == (6, 8)
            assert torch.equal(state[key][:4], plain[key])
            assert torch.equal(state[key][4:], later[int(key.split('.')[2])][4:])
        else:
            assert torch.equal(state[key], plain[key])


def test_incidence_counts_by_hand():
    counts, degrees = incidence_counts(DIRECT, 3, 2)
    # Relations 0, 1 and their technical inverses 2, 3. Heads: 0 {0}, 1 {0, 1}, 2 {1}, 3 {1, 2}; tails: 0 {1},
    # 1 {1, 2}, 2 {0}, 3 {0, 1}. Types: hh, tt, ht, th.
    assert degrees.tolist() == [[1, 2, 1, 2], [1, 2, 1, 2]]
    assert counts[0, 0, 1] == 1 and counts[0, 1, 1] == 2 and counts[0, 2, 3] == 1
    assert counts[2, 1, 3] == 2 and counts[2, 0, 2] == 1 and counts[3, 2, 0] == 1 and counts[3, 0, 0] == 0
    weights = relation_count_weights(DIRECT, 3, 2)
    half = torch.tensor(0.5, dtype=torch.float64).sqrt()
    assert weights.dtype == torch.float64 and weights.shape == (4, 4, 4)
    assert weights[0, 0, 1] == half and weights[0, 1, 1] == 1 and weights[2, 1, 3] == 1 and weights[3, 0, 0] == 0


def test_pair_counts_by_hand():
    counts, degree = pair_counts(DIRECT, 3, 2)
    assert degree.tolist() == [1, 3, 1, 3]
    same = {(a, b) for a, b in counts[0].nonzero().tolist()}
    reverse = {(a, b) for a, b in counts[1].nonzero().tolist()}
    # (0, 1) carries relations 0 and 1, so (1, 0) carries their inverses 2 and 3; a relation never matches itself or,
    # reversed, its own technical inverse.
    assert same == {(0, 1), (1, 0), (2, 3), (3, 2)} and reverse == {(0, 3), (1, 2), (2, 1), (3, 0)}
    assert int(counts.sum()) == 8
    weights = pair_channel_weights(DIRECT, 3, 2)
    one = torch.tensor(1.0, dtype=torch.float64)
    assert weights[0, 0, 1] == one / torch.tensor(3.0, dtype=torch.float64).sqrt() and weights[1, 0, 2] == 0


def test_weighted_relation_graph_layout():
    model = small(ultra_relation_weighting='normalized', ultra_pair_channels='on').set_graph(FACTS)
    index, types = build_relation_graph(model.edge_index, model.edge_type, model.num_entities, 2 * model.num_direct_relations)
    count = len(types)
    assert torch.equal(model.rel_edge_index[:, :count], index) and torch.equal(model.rel_edge_type[:count], types)
    assert set(model.rel_edge_type[count:].tolist()) <= {4, 5} and len(model.rel_edge_type) > count
    direct = model.graph_triples[model.graph_triples[:, 1] < model.num_direct_relations]
    dense = relation_count_weights(direct, 5, 3)
    pairs = pair_channel_weights(direct, 5, 3)
    expected = torch.cat((dense[types, index[0], index[1]],
                          pairs[model.rel_edge_type[count:] - 4, model.rel_edge_index[0, count:], model.rel_edge_index[1, count:]]))
    assert model.rel_edge_weight.dtype == torch.float32 and torch.equal(model.rel_edge_weight, expected.float())
    without = small(ultra_relation_weighting='normalized').set_graph(FACTS)
    assert torch.equal(without.rel_edge_index, index) and torch.equal(without.rel_edge_weight, expected[:count].float())


@pytest.mark.parametrize('training', [True, False])
def test_unit_weights_give_ultra_relation_reasoning(training):
    model = small().set_graph(FACTS).train(training)
    queries = torch.tensor([0, 1, 2, 4])
    with torch.set_grad_enabled(training):
        plain = model.relation_model(model.rel_edge_index, model.rel_edge_type, 6, queries)
        weighted = model.relation_model(model.rel_edge_index, model.rel_edge_type, 6, queries,
                                        edge_weight=torch.ones(len(model.rel_edge_type)))
    assert torch.equal(plain, weighted)
    with pytest.raises(ValueError, match='match'):
        model.relation_model(model.rel_edge_index, model.rel_edge_type, 6, queries, edge_weight=torch.ones(3))


@pytest.mark.parametrize('pair', ['off', 'on'])
def test_weighted_model_scores(pair):
    model = small(ultra_relation_weighting='normalized', ultra_pair_channels=pair, ultra_query_batch_size=2).set_graph(FACTS)
    queries = FACTS[:, :2]
    train_scores = model.train()(queries)
    assert train_scores.shape == (len(FACTS), 5) and torch.isfinite(train_scores).all()
    train_scores.sum().backward()
    assert all(p.grad is not None for p in model.relation_model.parameters())
    model.eval().requires_grad_(False)
    scores = model(queries)
    assert torch.isfinite(scores).all()
    model.query_batch_size = 1
    model.clear_inference_cache()
    torch.testing.assert_close(model(queries), scores)
    assert model.forward_k_vs_all_heads(FACTS[:, 1:]).shape == (len(FACTS), 5)


def test_checkpoint_options_must_match(tmp_path):
    model = small(ultra_relation_weighting='normalized', ultra_pair_channels='on')
    config = dict(ultra_dim=8, ultra_num_layers=2, ultra_relation_weighting='normalized', ultra_pair_channels='on')
    torch.save(dict(model=model.state_dict(), model_config=config), tmp_path / 'pair.pt')
    loaded = small(ultra_relation_weighting='normalized', ultra_pair_channels='on').load_pretrained(tmp_path / 'pair.pt')
    assert all(torch.equal(loaded.state_dict()[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match='saved with'):
        small().load_pretrained(tmp_path / 'pair.pt')
    torch.save(model.state_dict(), tmp_path / 'pair-tensors.pt')   # without model_config the shapes still guard
    with pytest.raises(ValueError, match='architecture mismatch'):
        small(ultra_relation_weighting='normalized').load_pretrained(tmp_path / 'pair-tensors.pt')
    count = small(ultra_relation_weighting='normalized')
    torch.save(dict(model=count.state_dict(), model_config=dict(config, ultra_pair_channels='off')), tmp_path / 'count.pt')
    with pytest.raises(ValueError, match='saved with'):
        small().load_pretrained(tmp_path / 'count.pt')
    small(ultra_relation_weighting='normalized').load_pretrained(tmp_path / 'count.pt')
    torch.save(small().state_dict(), tmp_path / 'official.pt')   # no model_config: the official layout
    small().load_pretrained(tmp_path / 'official.pt')


def golden_large(name: str) -> Path:
    root = os.environ.get('PROOFPFN_GOLDEN')
    for folder in ((Path(root) / 'golden' / 'large', Path(root) / 'large') if root else ()):
        if (folder / name).is_file():
            return folder / name
    pytest.skip(f'golden file {name} not found (set PROOFPFN_GOLDEN to the proofpfn-golden folder)')


def test_relation_graphs_equal_the_research_worker():
    """G2: the research worker's relation graph (types 0-3, then pair types 4 and 5) and float64 weights, per task."""
    raw = gzip.decompress(golden_large('g2_worlds_k0_seed42.pt.gz').read_bytes())
    worlds = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    checked = 0
    for name, world in worlds.items():
        prepared, content = world['prepared'], world['content']
        for task in sorted({k.split('/')[0] for k in content}):
            graph = {k.split('/', 2)[2]: v for k, v in prepared.items() if k.startswith(f'{task}/graph/')}
            n, nr = int(content[f'{task}/num_entities']), int(content[f'{task}/num_relations'])
            assert int(graph['num_direct']) == nr
            index, types, weight = weighted_relation_graph(graph['graph_triples'], n, nr, pair_channels=True)
            assert torch.equal(index, graph['rel_edge_index']), (name, task)
            assert torch.equal(types, graph['rel_edge_type']), (name, task)
            assert weight.dtype == torch.float64 and torch.equal(weight, graph['edge_weight']), (name, task)
            model = ULTRA(dict(ultra_dim=8, ultra_num_layers=1, ultra_relation_weighting='normalized', ultra_pair_channels='on'))
            model.set_graph(content[f'{task}/context'], n, nr)
            assert torch.equal(model.graph_triples, graph['graph_triples']) and torch.equal(model.relation_id_map, graph['relation_id_map'])
            assert torch.equal(model.rel_edge_index, index) and torch.equal(model.rel_edge_weight, weight.float())
            checked += 1
    assert checked == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_relation_graph_is_device_independent():
    cpu = small(ultra_relation_weighting='normalized', ultra_pair_channels='on').set_graph(FACTS)
    cuda = small(ultra_relation_weighting='normalized', ultra_pair_channels='on').cuda().set_graph(FACTS)
    assert cuda.rel_edge_index.is_cuda and cuda.rel_edge_weight.is_cuda
    assert torch.equal(cuda.rel_edge_index.cpu(), cpu.rel_edge_index) and torch.equal(cuda.rel_edge_type.cpu(), cpu.rel_edge_type)
    assert torch.equal(cuda.rel_edge_weight.cpu(), cpu.rel_edge_weight)
