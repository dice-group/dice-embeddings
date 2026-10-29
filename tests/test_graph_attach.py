"""Graphs prepared outside the model (GraphKGE.internal_graph, ultra.relation_graph, GraphKGE.attach_graph) and the
scores of query facts on the whole attached graph (GraphKGE.forward_query_facts)."""
import pytest
import torch

from dicee.models import TRIX
from dicee.models.graph_model import GraphKGE
from dicee.models.ultra import ULTRA, RelationGraph, relation_graph

FACTS = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 2], [2, 0, 3], [3, 1, 0], [3, 2, 3], [1, 2, 0], [4, 1, 2], [2, 2, 4], [0, 0, 1]])
OPTIONS = [dict(), dict(ultra_relation_weighting='normalized'), dict(ultra_relation_weighting='normalized', ultra_pair_channels='on')]
BUFFERS = ('graph_triples', 'relation_id_map', 'edge_index', 'edge_type', 'rel_edge_index', 'rel_edge_type', 'rel_edge_weight')


def ultra(**options) -> ULTRA:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return ULTRA(dict(num_entities=5, num_relations=3, ultra_dim=8, ultra_num_layers=2, **options))


def same_buffers(a: GraphKGE, b: GraphKGE) -> None:
    assert (a.num_entities, a.num_relations, a.num_direct_relations) == (b.num_entities, b.num_relations, b.num_direct_relations)
    for name in BUFFERS:
        x, y = getattr(a, name, None), getattr(b, name, None)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype and x.device == y.device and torch.equal(x, y), name


@pytest.mark.parametrize('options', OPTIONS, ids=['binary', 'normalized', 'pairs'])
def test_a_prepared_graph_attaches_like_set_graph(options):
    expected = ultra(**options).set_graph(FACTS)
    graph, mapping, num_direct = ULTRA.internal_graph(FACTS, 5, 3)
    assert num_direct == 3 and torch.equal(mapping, torch.arange(3))
    assert torch.equal(graph, torch.cat((FACTS, FACTS[:, [2, 1, 0]] + torch.tensor([0, 3, 0]))).unique(dim=0))
    weighting, pairs = options.get('ultra_relation_weighting', 'binary'), options.get('ultra_pair_channels', 'off')
    prepared = relation_graph(graph, 5, num_direct, weighting, pairs)
    assert isinstance(prepared, RelationGraph) and (prepared.edge_weight is None) == (weighting == 'binary')
    assert prepared.edge_weight is None or prepared.edge_weight.dtype == torch.float64
    same_buffers(ultra(**options).attach_graph(graph, mapping, 5, 3, num_direct, prepared), expected)
    same_buffers(ultra(**options).attach_graph(graph, mapping, 5, 3, num_direct), expected)   # derived in the model


def test_attach_checks_its_inputs():
    graph, mapping, num_direct = ULTRA.internal_graph(FACTS, 5, 3)
    with pytest.raises(ValueError, match='relation map'):
        ultra().attach_graph(graph, mapping[:2], 5, 3, num_direct)
    with pytest.raises(ValueError, match='ultra_relation_weighting'):
        ultra().attach_graph(graph, mapping, 5, 3, num_direct, relation_graph(graph, 5, 3, 'normalized'))
    with pytest.raises(ValueError, match='ultra_relation_weighting'):
        ultra(ultra_relation_weighting='normalized').attach_graph(graph, mapping, 5, 3, num_direct, relation_graph(graph, 5, 3))
    with pytest.raises(ValueError, match='requires'):
        relation_graph(graph, 5, 3, 'binary', 'on')
    with pytest.raises(ValueError, match='outside'):
        ULTRA.internal_graph(FACTS, 4, 3)
    trix = TRIX(dict(num_entities=5, num_relations=3, trix_dim=8, trix_num_layers=1))
    with pytest.raises(NotImplementedError):
        trix.attach_graph(graph, mapping, 5, 3, num_direct, relation_graph(graph, 5, 3))
    expected = TRIX(dict(num_entities=5, num_relations=3, trix_dim=8, trix_num_layers=1)).set_graph(FACTS)
    trix.attach_graph(graph, mapping, 5, 3, num_direct)   # TRIX derives its own graph tensors
    assert torch.equal(trix.graph_triples, expected.graph_triples) and torch.equal(trix.edge_index, expected.edge_index)


def test_internal_graph_with_inverse_relations():
    model = ultra().set_graph(FACTS, inverse_relations={0: 2})
    graph, mapping, num_direct = ULTRA.internal_graph(FACTS, 5, 3, {0: 2})
    assert num_direct == 2 and mapping.tolist() == [0, 1, 2]
    assert torch.equal(model.graph_triples, graph) and torch.equal(model.relation_id_map, mapping)


@pytest.mark.parametrize('options', OPTIONS, ids=['binary', 'normalized', 'pairs'])
def test_query_facts_score_like_the_kvsall_queries_in_evaluation(options):
    model = ultra(**options).set_graph(FACTS).eval()
    facts = FACTS[[0, 2, 5]]
    with torch.no_grad():
        assert torch.equal(model.forward_query_facts(facts, False), model.forward_k_vs_all(facts[:, :2]))
        assert torch.equal(model.forward_query_facts(facts, True), model.forward_k_vs_all_heads(facts[:, 1:]))


@pytest.mark.parametrize('options', OPTIONS, ids=['binary', 'normalized', 'pairs'])
def test_query_facts_keep_their_edges_in_training(options):
    model = ultra(**options).set_graph(FACTS)
    facts = FACTS[[0, 2]]
    with torch.no_grad():
        reference = model.eval().forward_query_facts(facts, True)
    model.train()
    scores = model.forward_query_facts(facts, True)
    assert scores.shape == (2, 5) and scores.requires_grad
    torch.testing.assert_close(scores.detach(), reference)
    with torch.no_grad():   # forward_k_vs_all hides the queried facts' edges in training mode
        assert not torch.allclose(model.forward_k_vs_all(facts[:, :2]), model.forward_query_facts(facts, False))
    scores.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
