"""The prior's streams, recipe, world batch and world order, against the research code's golden references.

G1 (tests/fixtures/proofpfn/g1_orders.json.gz): the 24,000-slot world orders of the arms S, CU, UL, HT, N, R and M for
seeds 42, 43 and 44 (21 order digests, all equal to the research pins) and the first 400 slots of each distinct order.
The priors are built here from the research run options (the hook records of the opts files), independently of the
arm presets. The research configuration chain: tests/fixtures/proofpfn/recipe_research.json.gz.
"""
import dataclasses
import gzip
import hashlib
import io

import pytest
import torch

from dicee.synthetic_kg import recipe, streams
from dicee.synthetic_kg.batch import WorldBatch, check_batch, world_sizes
from dicee.synthetic_kg.priors import RESERVED_HOOK, RESERVED_STAGE2, FamilySpec, PriorConfig, Slot, build_order, hook_indices, mixture_bounds, order_sha256, parse_mix, stage2_indices, world
from tests.proofpfn_fixtures import fixture, golden_large, typed
from tests.proofpfn_fixtures import research_prior as prior_of_entry
from tests.synthetic_worlds import content_sha256, content_tensors

G1 = fixture('g1_orders.json.gz')
RECIPE = fixture('recipe_research.json.gz')
SEEDS = (42, 43, 44)


def research_prior(arm: str) -> PriorConfig:
    """The prior of a research arm of G1, from its --mix string, order name and hook records."""
    return prior_of_entry(G1['arms'][arm])


@pytest.mark.parametrize('arm', sorted(G1['arms']))
def test_mixture_shares_and_bounds(arm):
    entry = G1['arms'][arm]
    prior = research_prior(arm)
    assert {k: repr(v) for k, v in prior.shares.items()} == entry['shares_repr']
    assert list(prior.shares) == sorted(entry['shares_repr'])
    assert [[name, repr(bound)] for name, bound in mixture_bounds(prior)] == entry['cumulative_bounds_repr']


@pytest.mark.parametrize('arm', sorted(G1['arms']))
@pytest.mark.parametrize('seed', SEEDS)
def test_orders_equal_the_research_orders(arm, seed):
    golden = G1['arms'][arm]['seeds'][str(seed)]
    assert golden['equal_to_pins']
    order = build_order(seed, research_prior(arm))
    assert len(order) == recipe.TOTAL_UPDATES and all(isinstance(slot, Slot) for slot in order)
    assert order_sha256(order) == golden['order_sha256']
    assert order_sha256(order[:400]) == golden['first400_sha256']
    assert [list(slot) for slot in order[:400]] == G1['distinct_orders'][golden['order_sha256']]['first400']


def test_orders_cover_the_21_research_digests():
    digests = {(arm, int(seed)): v['order_sha256'] for arm, e in G1['arms'].items() for seed, v in e['seeds'].items()}
    assert len(digests) == 21 and len(set(digests.values())) == len(G1['distinct_orders']) == 7


def test_order_is_prefix_stable_and_skips_reserved_indices():
    prior = research_prior('M')
    assert build_order(43, prior, 2000)[:700] == build_order(43, prior, 700)
    assert stage2_indices(42, 'structured', 3) == [i for i in range(20_000, 20_100) if streams.stage2_family(42, i) == 'structured'][:3]
    assert hook_indices(129_998, 4) == [129_998, 129_999, 160_000, 160_001]
    assert RESERVED_STAGE2 == tuple(map(tuple, RECIPE['constants']['pt_common']['RESERVED_S2']))
    assert RESERVED_HOOK == tuple(map(tuple, RECIPE['constants']['pt_common']['RESERVED_HOOK']))
    statistical = stage2_indices(42, 'statistical', 25_000)
    assert statistical[-1] > 90_048 and not any(90_000 <= i < 90_048 for i in statistical)
    assert any(90_000 <= i < 90_048 for i in range(89_000, 91_000) if streams.is_statistical(42, i))


def test_streams():
    assert streams.sha256_int('abc') == int.from_bytes(bytes.fromhex('ba7816bf8f01cfea'), 'big')
    assert streams.sha256_int('abc', 'little') == int.from_bytes(bytes.fromhex('ba7816bf8f01cfea'), 'little')
    assert streams.mixture_draw(42, 'REC10A-XLE', 0) == streams.sha256_int('prior-fix-v1:mix:42:REC10A-XLE:0') / 2 ** 64
    assert streams.initial_seed(42) == streams.sha256_int('capacity-prior-v1:42:initial:d64') % (2 ** 63 - 1)
    assert streams.hard_target_seed(43, 7) == int.from_bytes(hashlib.sha256(b'pt_hard:43:7').digest()[:8], 'little')   # pt_hard.generator
    versions = RECIPE['constants']
    assert streams.MIXTURE_VERSION == versions['pt_common']['VERSION'] and streams.CAPACITY_VERSION == versions['cp_prior_VERSION']
    assert streams.STATISTICAL_VERSION == versions['st_prior_VERSION'] and streams.STRUCTURED_VERSION == versions['sw_prior_VERSION']
    first = G1['distinct_orders'][G1['arms']['S']['seeds']['42']['order_sha256']]['first400']
    for family, source, index in first:
        if source == 's2':
            assert streams.stage2_family(42, index) == family


def test_recipe_equals_the_research_chain():
    """The values of the research recipe that the generators read: the seed, the world sizes, the queries and negatives."""
    world = dataclasses.asdict(recipe.WORLD)

    def read(config: dict) -> dict:
        return dict(world={k: config['world'][k] for k in world}, batch_size=config['batch_size'], num_negative=config['num_negative'])

    expected = dict(world=world, batch_size=recipe.QUERIES, num_negative=recipe.NEGATIVES)
    for seed in SEEDS:
        final = RECIPE['final'][str(seed)]
        assert read(final) == typed(expected) and final['seed'] == typed(seed)
    assert len(RECIPE['levels']) == 12 and all(read(level['config']) == expected for level in RECIPE['levels'].values())
    for name, params in recipe.RECORD_PARAMS.items():
        assert typed(dict(params)) == RECIPE['params'][name] and list(params) == RECIPE['params_key_order'][name]
    c = RECIPE['constants']
    assert recipe.NEGATIVES == c['pf_prior_NEG'] == c['st_prior_NEG'] == c['sw_prior_NEG'] == c['pf_rec1_NEG'] == c['pf_list1_NEG']
    assert (recipe.STATISTICAL_SHARE, recipe.STRUCTURED_SHARE) == (c['st_prior_P_STAT'], c['sw_prior_V2_SHARE'])
    assert (recipe.STAGE1_WORLDS, recipe.STAGE2_OFFSET, recipe.LEGACY_OFFSET) == (c['pt_common']['N1'], c['op_common']['S2_OFFSET'], c['cp_prior_OFFSET'])
    assert (recipe.CONSTANT_UPDATES, recipe.DECAY_UPDATES, recipe.TOTAL_UPDATES) == (c['op_common']['N1'], c['op_common']['N2'], c['pt_common']['TOTAL'])
    assert 2 * recipe.HEAD_QUERIES == recipe.QUERIES


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
    prior = research_prior('M')
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


def test_content_digest_of_research_worlds():
    """G2: the research worlds of seed 42, position 0 (12 worlds) pass the batch check and give the golden digest."""
    raw = gzip.decompress(golden_large('g2_worlds_k0_seed42.pt.gz').read_bytes())
    worlds = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    golden = {f"{w['arm']}/{w['family']}/{w['source']}/{w['index']}": w for w in fixture('g2_worlds.json.gz')['worlds']
              if w['seed'] == 42 and w['k'] == 0}
    assert set(worlds) == set(golden) and len(worlds) == 12
    for name, value in worlds.items():
        content = value['content']
        world_batch = {task: WorldBatch(content[f'{task}/groups'], content[f'{task}/head_prediction'],
                                        frozenset(map(tuple, content[f'{task}/context'].tolist())),
                                        frozenset(map(tuple, content[f'{task}/truth'].tolist())),
                                        int(content[f'{task}/num_entities']), int(content[f'{task}/num_relations']))
                       for task in golden[name]['research_fingerprint']}
        check_batch(world_batch)
        assert content_sha256(world_batch) == golden[name]['content_sha256'], name
        assert world_sizes(world_batch) == {k: golden[name]['meta'][k] for k in ('n_entities', 'n_context')}
