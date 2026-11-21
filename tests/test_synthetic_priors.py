"""The prior's streams, recipe, world batch and world order.

The pins (``tests/proofpfn_fixtures.py``) hold the shares and cumulative bounds of the three mixtures and the digest
of the first 400 slots of each world order; the presets pin the digests of the full orders.
"""
import dataclasses
import hashlib

import pytest
import torch

from dicee.synthetic_kg import recipe, streams
from dicee.synthetic_kg.batch import WorldBatch, check_batch, world_sizes
from dicee.synthetic_kg.priors import RESERVED_HOOK, RESERVED_STAGE2, FamilySpec, PriorConfig, Slot, build_order, hook_indices, mixture_bounds, order_sha256, parse_mix, stage2_indices, world
from dicee.synthetic_kg.variants import SEEDS
from tests.proofpfn_fixtures import PRIORS, content_sha256, content_tensors, pins

LABELS = ('synthetic', 'real', 'mixed')


@pytest.mark.parametrize('label', LABELS)
def test_mixture_shares_and_bounds(label):
    pinned, prior = pins()['mixtures'][label], PRIORS[label]
    assert {k: repr(v) for k, v in prior.shares.items()} == pinned['shares'] and list(prior.shares) == sorted(pinned['shares'])
    assert [[name, repr(bound)] for name, bound in mixture_bounds(prior)] == pinned['bounds']


@pytest.mark.parametrize('label', LABELS)
@pytest.mark.parametrize('seed', SEEDS)
def test_orders_equal_their_pins(label, seed):
    order = build_order(seed, PRIORS[label])
    assert len(order) == recipe.TOTAL_UPDATES and all(isinstance(slot, Slot) for slot in order)
    assert order_sha256(order[:400]) == pins()['orders'][label][str(seed)]
    if label == 'synthetic':   # the structural null keeps the order
        assert order == build_order(seed, PRIORS['null'])


def test_order_is_prefix_stable_and_skips_reserved_indices():
    prior = PRIORS['mixed']
    assert build_order(43, prior, 2000)[:700] == build_order(43, prior, 700)
    assert stage2_indices(42, 'structured', 3) == [i for i in range(20_000, 20_100) if streams.stage2_family(42, i) == 'structured'][:3]
    assert hook_indices(129_998, 4) == [129_998, 129_999, 160_000, 160_001]
    assert RESERVED_STAGE2 == ((90_000, 90_048), (120_000, 160_000)) and RESERVED_HOOK == ((130_000, 160_000), (500_000, 501_000))
    statistical = stage2_indices(42, 'statistical', 25_000)
    assert statistical[-1] > 90_048 and not any(90_000 <= i < 90_048 for i in statistical)
    assert any(90_000 <= i < 90_048 for i in range(89_000, 91_000) if streams.is_statistical(42, i))


def test_streams():
    assert streams.sha256_int('abc') == int.from_bytes(bytes.fromhex('ba7816bf8f01cfea'), 'big')
    assert streams.sha256_int('abc', 'little') == int.from_bytes(bytes.fromhex('ba7816bf8f01cfea'), 'little')
    assert streams.mixture_draw(42, 'REC10A-XLE', 0) == streams.sha256_int('prior-fix-v1:mix:42:REC10A-XLE:0') / 2 ** 64
    assert streams.initial_seed(42) == streams.sha256_int('capacity-prior-v1:42:initial:d64') % (2 ** 63 - 1)
    assert streams.hard_target_seed(43, 7) == int.from_bytes(hashlib.sha256(b'pt_hard:43:7').digest()[:8], 'little')
    for family, source, index in build_order(42, PRIORS['synthetic'], 400):
        if source == 's2':
            assert streams.stage2_family(42, index) == family


def test_recipe_values():
    """The values of the recipe that the generators and the schedule read."""
    assert dataclasses.asdict(recipe.WORLD) == dict(min_core_entities=96, max_core_entities=192, min_relations=5, max_relations=10,
                                                    min_density=0.001, max_density=0.003, min_observed=0.4, max_observed=0.9,
                                                    blocks_per_rule=6, grid_size=3, max_copies=2)
    assert (recipe.QUERIES, recipe.HEAD_QUERIES, recipe.NEGATIVES) == (30, 15, 512)
    assert (recipe.STATISTICAL_SHARE, recipe.STRUCTURED_SHARE) == (0.35, 0.5)
    assert (recipe.STAGE1_WORLDS, recipe.STAGE2_OFFSET, recipe.LEGACY_OFFSET) == (20_000, 20_000, 300_000)
    assert (recipe.CONSTANT_UPDATES, recipe.DECAY_UPDATES, recipe.TOTAL_UPDATES) == (20_000, 4_000, 24_000)
    assert list(recipe.RECORD_PARAMS) == ['categorical2', 'attr2', 'rec1', 'list1']


def test_record_params_overrides():
    params = recipe.record_params('rec1', {'max_entities': 30000, 'rows': [2500, 15000]})
    assert params['rows'] == (2500, 15000) and params['max_entities'] == 30000 and list(params) == list(recipe.REC1_PARAMS)
    assert recipe.REC1_PARAMS['rows'] == (300, 2500)
    with pytest.raises(ValueError, match='no parameter'):
        recipe.record_params('rec1', {'swap_budget': 3})
    with pytest.raises(TypeError):
        recipe.REC1_PARAMS['rows'] = (1, 2)   # type: ignore[index]  # the parameters are read-only


def test_prior_validation():
    with pytest.raises(ValueError, match='without a spec'):
        PriorConfig('plain=0.5,rec1=0.5', 'x')
    with pytest.raises(ValueError, match='outside the mixture'):
        PriorConfig('plain=1.0', 'x', (FamilySpec('rec1', 'rec1', 8_100_000),))
    with pytest.raises(ValueError, match='repeated'):
        parse_mix('plain=0.5,plain=0.5')
    with pytest.raises(ValueError, match='unknown generator'):
        FamilySpec('x', 'peer1', 0)
    with pytest.raises(ValueError, match='unknown parameters'):
        FamilySpec('rec1', 'rec1', 0, (('swap_budget', 3),))
    with pytest.raises(ValueError, match='other than'):
        FamilySpec('plain', 'rec1', 0)
    assert parse_mix('a=0.25, b=0.75,c=0') == {'a': 0.25, 'b': 0.75}


def test_world_refuses_slots_of_other_streams():
    prior = PRIORS['mixed']
    with pytest.raises(ValueError, match='does not belong'):
        world(42, Slot('rec1', 's2', 20_000), prior)
    with pytest.raises(ValueError, match='does not belong'):
        world(42, Slot('structured', 's1', 3), prior)


def batch(n: int = 50) -> dict[str, WorldBatch]:
    """A two-task world with 15 + 15 query groups of 4 negatives on a ring graph."""
    truth = frozenset((i, r, (i + 1 + r) % n) for i in range(n) for r in range(2))
    out = {}
    for t, task in enumerate(('rule', 'base')):
        queries = sorted(truth)[t * 15:(t + 1) * 15]
        groups = torch.tensor(queries)[:, None].repeat(1, 5, 1)
        heads = torch.arange(15) >= 8 - t
        for q, (h, r, tail) in enumerate(queries):
            column = 0 if heads[q] else 2
            groups[q, 1:, column] = torch.tensor([e for e in range(n) if ((e, r, tail) if heads[q] else (h, r, e)) not in truth][:4])
        out[task] = WorldBatch(groups, heads, truth - frozenset(queries), truth, n, 2)
    return out


def test_check_batch():
    world_batch = batch()
    check_batch(world_batch)
    assert world_sizes(world_batch) == {'n_entities': 50, 'n_context': 85}
    rule = world_batch['rule']
    leaked = dataclasses.replace(rule, context=rule.context | {tuple(rule.groups[0, 0].tolist())})
    with pytest.raises(ValueError, match='observed'):
        check_batch({**world_batch, 'rule': leaked})
    groups = rule.groups.clone()
    groups[0, 1] = groups[0, 0]
    with pytest.raises(ValueError, match='negative is a true fact'):
        check_batch({**world_batch, 'rule': dataclasses.replace(rule, groups=groups)})
    with pytest.raises(ValueError, match='head queries'):
        check_batch({**world_batch, 'rule': dataclasses.replace(rule, head_prediction=~rule.head_prediction)})
    with pytest.raises(ValueError, match='query groups'):
        check_batch({'rule': rule})
    content = content_tensors(world_batch)
    assert sorted(content)[:6] == ['base/context', 'base/groups', 'base/head_prediction', 'base/num_entities', 'base/num_relations', 'base/truth']
    assert content_sha256(world_batch) == content_sha256(dict(reversed(list(world_batch.items()))))   # the task order does not enter
