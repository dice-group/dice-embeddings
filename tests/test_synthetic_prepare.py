"""The worker preparation (``dicee.synthetic_kg.prepare``) against the pinned worlds.

The pins (``tests/proofpfn_fixtures.py``) hold per world the digest of its prepared tensors for PCUltra: per task the
graph, the relation graph (types 0 to 3, then the pair types 4 and 5) and its float64 weights, the counts, the RB-W
rows ``soft`` and the context masks. CountULTRA and ULTRA take the block of types 0 to 3 of the same relation graph.
"""
import pytest
import torch

from dicee.models.ultra import ULTRA
from dicee.synthetic_kg.batch import WorldBatch
from dicee.synthetic_kg.determinism import THREADS, torch_threads
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.prepare import context_answers, prepare_slot, prepare_task, prepare_world, rbw_answers
from dicee.synthetic_kg.variants import VARIANTS
from tests.proofpfn_fixtures import PRIORS, pinned_world, prepared_tensors, require_real_sources
from tests.synthetic_worlds import WORLDS

PCU = VARIANTS['PCUltra-S'].model_config()
SMALL = ('42/synthetic/plain/0', '42/synthetic/list1/4', '42/synthetic/statistical/2', '42/null/attr2/4', '43/synthetic/plain/1')


@pytest.fixture(autouse=True)
def two_threads():
    with torch_threads(THREADS):
        yield


def check_prepared(name: str) -> None:
    """Prepare a pinned world for PCUltra and compare the digest of its prepared tensors."""
    slot, generated = pinned_world(name)
    prepared = prepare_world(generated, slot, PCU['ultra_relation_weighting'], PCU['ultra_pair_channels'])
    assert tensor_digest(prepared_tensors(prepared)) == WORLDS[name]['prepared'], f'{name}: the prepared tensors differ'
    assert list(prepared.tasks) == list(generated.batch) and prepared.num_queries == 30


@pytest.mark.parametrize('name', SMALL)
def test_prepared_world_equals_its_pin(name):
    check_prepared(name)


@pytest.mark.slow
def test_every_pinned_world_prepares_like_its_pin():
    names = [name for name in WORLDS if name.split('/')[2] != 'real']
    assert len(names) == 113
    for name in names:
        check_prepared(name)


@pytest.mark.slow
def test_every_pinned_real_world_prepares_like_its_pin():
    require_real_sources()
    names = [name for name in WORLDS if name.split('/')[2] == 'real']
    assert len(names) == 10
    for name in names:
        check_prepared(name)


@pytest.mark.parametrize('variant', ['CountULTRA-S', 'ULTRA-S'])
def test_the_relation_graph_without_pairs_is_the_block_of_types_0_to_3(variant):
    slot, generated = pinned_world('42/synthetic/plain/0')
    options = VARIANTS[variant].model_config()
    full = prepare_world(generated, slot, PCU['ultra_relation_weighting'], PCU['ultra_pair_channels'])
    part = prepare_world(generated, slot, options['ultra_relation_weighting'], options['ultra_pair_channels'])
    for task in full.tasks:
        index, types, weight, _ = full.tasks[task].graph.relation_graph
        assert weight is not None
        block = int((types < 4).sum())
        assert bool((types[:block] < 4).all()) and bool((types[block:] >= 4).all())
        cut = part.tasks[task].graph.relation_graph
        assert torch.equal(cut.edge_index, index[:, :block]) and torch.equal(cut.edge_type, types[:block])
        assert (cut.edge_weight is None) == (variant == 'ULTRA-S')
        if cut.edge_weight is not None:
            assert torch.equal(cut.edge_weight, weight[:block])
        for key in ('soft', 'context_mask', 'queries', 'head_prediction'):
            assert torch.equal(getattr(part.tasks[task], key), getattr(full.tasks[task], key))


@pytest.mark.parametrize('variant', ['PCUltra-S', 'CountULTRA-S', 'ULTRA-S'])
def test_attaching_a_prepared_graph_equals_set_graph(variant):
    slot, generated = pinned_world('42/synthetic/list1/4')
    options = VARIANTS[variant].model_config()
    prepared = prepare_slot(42, slot, PRIORS['synthetic'], options)
    for task, batch in generated.batch.items():
        with torch.random.fork_rng(devices=[]):
            expected = ULTRA(options).set_graph(sorted(batch.context), batch.num_entities, batch.num_relations)
            model = ULTRA(options)
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
