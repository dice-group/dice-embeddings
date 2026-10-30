"""The worker preparation (``dicee.synthetic_kg.prepare``) against golden G2.

G2 holds per world the per-key digests of the research worker's prepared tensors (``fast_pipeline.prepare_world`` with
mode ``'rbce'``): per task the graph, the relation graph (types 0 to 3, then the pair types 4 and 5) and its float64
weights, the counts, the RB-W rows ``soft`` and the context masks. The research digest also hashes ``degree`` (the
in-degree of every entity), which fed only the role adapter at zero output: ``prepare`` does not compute it, so the
comparison is per key without it. The worlds are prepared for PCUltra's options (pair channels on, which give every
key); CountULTRA and ULTRA take the block of types 0 to 3 of the same relation graph.
"""
import os

import pytest
import torch

from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.arms import ARMS
from dicee.synthetic_kg.batch import World, WorldBatch
from dicee.synthetic_kg.prepare import PreparedWorld, context_answers, prepare_slot, prepare_task, prepare_world, rbw_answers
from dicee.synthetic_kg.priors import Slot, real, world
from tests.proofpfn_fixtures import per_key_digests
from tests.synthetic_worlds import WORLDS

PCU = ARMS['S'].model_config()
SMALL = ('42/S/plain/0', '42/S/list1/4', '42/S/statistical/2', '42/N/attr2/4', '43/S/plain/1')
DEGREE = '/graph/degree'


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def prepared_tensors(prepared: PreparedWorld) -> dict[str, torch.Tensor]:
    """The prepared tensors by golden name: ``'<task>/graph/<name>'``, ``'<task>/soft'`` and ``'<task>/context_mask'``.

    The graph names are ``graph_triples``, ``relation_id_map``, ``rel_edge_index``, ``rel_edge_type``, ``edge_weight``
    (only with count weighting) and the int64 scalars ``num_entities``, ``num_relations`` and ``num_direct``.
    """
    out = {}
    for name, task in prepared.tasks.items():
        graph = task.graph
        values = dict(graph_triples=graph.graph_triples, relation_id_map=graph.relation_id_map, rel_edge_index=graph.relation_graph.edge_index,
                      rel_edge_type=graph.relation_graph.edge_type, edge_weight=graph.relation_graph.edge_weight,
                      num_entities=torch.tensor(graph.num_entities), num_relations=torch.tensor(graph.num_relations),
                      num_direct=torch.tensor(graph.num_direct))
        out.update({f'{name}/graph/{key}': value for key, value in values.items() if value is not None})
        out.update({f'{name}/soft': task.soft, f'{name}/context_mask': task.context_mask})
    return out


def golden_world(name: str) -> tuple[Slot, World]:
    golden = WORLDS[name]
    slot = Slot(golden['family'], golden['source'], golden['index'])
    return slot, world(golden['seed'], slot, ARMS[golden['arm']].prior)


def check_prepared(name: str) -> None:
    """Prepare a golden world for PCUltra and compare every research key except ``degree``."""
    slot, generated = golden_world(name)
    prepared = prepare_world(generated, slot, PCU['ultra_relation_weighting'], PCU['ultra_pair_channels'])
    expected = {k: v for k, v in WORLDS[name]['prepared_per_key'].items() if not k.endswith(DEGREE)}
    found = per_key_digests(prepared_tensors(prepared))
    assert list(found) == list(expected), name
    differing = [k for k in expected if found[k] != expected[k]]
    assert differing == [], f'{name}: the prepared tensors differ in {differing}'
    assert list(prepared.tasks) == list(generated.batch) and prepared.num_queries == 30
    for task, golden in WORLDS[name]['tasks'].items():
        types = prepared.tasks[task].graph.relation_graph.edge_type
        assert (len(types), int((types >= 4).sum())) == (golden['relation_graph_edges'], golden['pair_edges']), (name, task)


@pytest.mark.parametrize('name', SMALL)
def test_prepared_world_equals_the_research_worker(name):
    check_prepared(name)


@pytest.mark.slow
def test_every_golden_world_prepares_like_the_research_worker():
    names = [name for name, w in WORLDS.items() if w['family'] != 'real']
    assert len(names) == 113
    for name in names:
        check_prepared(name)


@pytest.mark.slow
def test_every_golden_real_world_prepares_like_the_research_worker():
    if not os.environ.get(real.SOURCES_ENV):
        pytest.skip(f'real worlds need their source graphs (set {real.SOURCES_ENV})')
    names = [name for name, w in WORLDS.items() if w['family'] == 'real']
    assert len(names) == 10
    for name in names:
        check_prepared(name)


@pytest.mark.parametrize('arm', ['CU', 'UL'])
def test_the_relation_graph_without_pairs_is_the_block_of_types_0_to_3(arm):
    slot, generated = golden_world('42/S/plain/0')
    options = ARMS[arm].model_config()
    full = prepare_world(generated, slot, PCU['ultra_relation_weighting'], PCU['ultra_pair_channels'])
    part = prepare_world(generated, slot, options['ultra_relation_weighting'], options['ultra_pair_channels'])
    for task in full.tasks:
        index, types, weight = full.tasks[task].graph.relation_graph
        assert weight is not None
        block = int((types < 4).sum())
        assert bool((types[:block] < 4).all()) and bool((types[block:] >= 4).all())
        cut = part.tasks[task].graph.relation_graph
        assert torch.equal(cut.edge_index, index[:, :block]) and torch.equal(cut.edge_type, types[:block])
        assert (cut.edge_weight is None) == (arm == 'UL')
        if cut.edge_weight is not None:
            assert torch.equal(cut.edge_weight, weight[:block])
        for key in ('soft', 'context_mask', 'queries', 'head_prediction'):
            assert torch.equal(getattr(part.tasks[task], key), getattr(full.tasks[task], key))


@pytest.mark.parametrize('arm', ['S', 'CU', 'UL'])
def test_attaching_a_prepared_graph_equals_set_graph(arm):
    slot, generated = golden_world('42/S/list1/4')
    prepared = prepare_slot(42, slot, ARMS['S'].prior, ARMS[arm].model_config())
    for task, batch in generated.batch.items():
        with torch.random.fork_rng(devices=[]):
            expected = ULTRA(ARMS[arm].model_config()).set_graph(sorted(batch.context), batch.num_entities, batch.num_relations)
            model = ULTRA(ARMS[arm].model_config())
        prepared.tasks[task].graph.attach(model)
        for name in ('graph_triples', 'relation_id_map', 'edge_index', 'edge_type', 'rel_edge_index', 'rel_edge_type', 'rel_edge_weight'):
            a, b = getattr(model, name), getattr(expected, name)
            assert (a is None) == (b is None) and (a is None or (a.dtype == b.dtype and torch.equal(a, b))), (task, name)
        assert (model.num_entities, model.num_relations, model.num_direct_relations) == (expected.num_entities, expected.num_relations,
                                                                                       expected.num_direct_relations)


def test_targets_by_hand():
    groups = torch.tensor([[[0, 0, 1], [0, 0, 3]], [[2, 0, 1], [3, 0, 1]], [[1, 1, 2], [1, 1, 0]]])
    truth = frozenset({(0, 0, 1), (0, 0, 2), (0, 0, 4), (2, 0, 1), (4, 0, 1), (1, 1, 2)})
    batch = WorldBatch(groups, torch.tensor([False, True, False]), frozenset({(0, 0, 2), (4, 0, 1)}), truth, 5, 2)
    assert rbw_answers(batch) == [[1, 4], [0, 2], [2]] and context_answers(batch) == [[2], [4], []]
    task = prepare_task(batch, None)   # type: ignore[arg-type]  # the targets do not read the graph
    assert task.soft.dtype == torch.float32 and task.soft[0].tolist() == [0, 0.5, 0, 0, 0.5]
    assert torch.equal(task.soft[1], torch.tensor([0.5, 0, 0.5, 0, 0])) and task.soft[2].tolist() == [0, 0, 1, 0, 0]
    assert task.context_mask.tolist() == [[False, False, True, False, False], [False, False, False, False, True], [False] * 5]
    observed = WorldBatch(groups, batch.head_prediction, batch.context | {(1, 1, 2)}, truth, 5, 2)
    with pytest.raises(ValueError, match='hidden true answer'):
        rbw_answers(observed)
