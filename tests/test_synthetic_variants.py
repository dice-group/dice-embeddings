"""The model variants (``dicee.synthetic_kg.variants``): their options, their world orders and their initial weights.

The pins (``tests/proofpfn_fixtures.py``) hold the digest of the first 400 slots of each world order and the digest of
the initial weights of each model (PCUltra, CountULTRA, ULTRA) for seeds 42, 43 and 44; the presets pin the digests of
the full orders.
"""
import pytest
import torch

from dicee.synthetic_kg.determinism import THREADS, torch_threads
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.priors import FAMILY_TITLES, build_order, order_sha256
from dicee.synthetic_kg.variants import MIXED, REAL_ONLY, SEEDS, SYNTHETIC, SYNTHETIC_NULL, VARIANTS, initial_model, variant_of
from tests.proofpfn_fixtures import pins

PRIOR_OF = {'S': ('synthetic', SYNTHETIC), 'N': ('synthetic', SYNTHETIC_NULL), 'HT': ('synthetic', SYNTHETIC), 'R': ('real', REAL_ONLY),
            'M': ('mixed', MIXED)}
"""The order label of the pins and the prior of each name suffix."""
OPTIONS_OF = {'PCUltra': ('normalized', 'on'), 'CountULTRA': ('normalized', 'off'), 'ULTRA': ('binary', 'off')}


@pytest.fixture(autouse=True)
def two_threads():
    with torch_threads(THREADS):
        yield


def test_the_eleven_variants():
    assert list(VARIANTS) == ['PCUltra-S', 'PCUltra-R', 'PCUltra-M', 'PCUltra-N', 'PCUltra-HT', 'CountULTRA-S', 'CountULTRA-R', 'CountULTRA-M',
                              'ULTRA-S', 'ULTRA-R', 'ULTRA-M']
    for name, preset in VARIANTS.items():
        model, suffix = name.split('-')
        assert preset.name == name and (preset.relation_weighting, preset.pair_channels) == OPTIONS_OF[model]
        assert preset.prior == PRIOR_OF[suffix][1] and preset.target == ('hard' if suffix == 'HT' else 'rbw')
        assert preset.model_config() == dict(ultra_dim=64, ultra_num_layers=6, ultra_relation_weighting=OPTIONS_OF[model][0],
                                             ultra_pair_channels=OPTIONS_OF[model][1])
    assert variant_of('CountULTRA-S') is VARIANTS['CountULTRA-S'] and variant_of(VARIANTS['ULTRA-M']) is VARIANTS['ULTRA-M']
    with pytest.raises(KeyError, match='unknown variant'):
        variant_of('S')
    assert VARIANTS['PCUltra-S'].pinned_order(45) is None
    assert len({digest for preset in VARIANTS.values() for _, digest in preset.order_sha256}) == 7   # 3 synthetic, 1 real, 3 mixed


@pytest.mark.parametrize('name', ['PCUltra-S', 'PCUltra-N', 'PCUltra-R', 'PCUltra-M', 'ULTRA-R', 'CountULTRA-M'])
@pytest.mark.parametrize('seed', SEEDS)
def test_world_orders_equal_their_pins(name, seed):
    preset = VARIANTS[name]
    order = build_order(seed, preset.prior)
    assert order_sha256(order) == preset.pinned_order(seed)
    assert order_sha256(order[:400]) == pins()['orders'][PRIOR_OF[name.split('-')[1]][0]][str(seed)]


def test_variants_of_one_prior_share_their_world_order():
    for suffix in ('S', 'R', 'M'):
        group = [preset for name, preset in VARIANTS.items() if name.endswith(f'-{suffix}')]
        assert len(group) == 3 and len({preset.order_sha256 for preset in group}) == 1
    assert VARIANTS['PCUltra-N'].order_sha256 == VARIANTS['PCUltra-HT'].order_sha256 == VARIANTS['PCUltra-S'].order_sha256
    assert len({digest for _, digest in VARIANTS['PCUltra-R'].order_sha256}) == 1   # the real order is the same for every seed


@pytest.mark.parametrize('name', sorted(VARIANTS))
@pytest.mark.parametrize('seed', SEEDS)
def test_initial_weights_equal_their_pins(name, seed):
    state = initial_model(name, seed).state_dict()
    rows = 6 if VARIANTS[name].pair_channels == 'on' else 4
    tables = [key for key in state if key.startswith('relation_model.layers.') and key.endswith('.relation.weight')]
    assert len(state) == 82 and len(tables) == 6 and all(state[key].shape[0] == rows for key in tables)
    assert tensor_digest(state) == pins()['initial_weights'][name.split('-')[0]][str(seed)]


def test_initialization_leaves_the_random_state_unchanged():
    torch.manual_seed(123)
    before = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    model = initial_model('PCUltra-S', 42)
    assert torch.equal(torch.get_rng_state(), before)
    if cuda is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda)
    assert model.training and model.relation_weighting == 'normalized' and model.pair_channels == 'on'
    assert tensor_digest(initial_model('PCUltra-S', 42).state_dict()) == tensor_digest(model.state_dict())
    assert torch.equal(torch.get_rng_state(), before)


def test_every_world_family_has_a_title():
    families = {family for preset in VARIANTS.values() for family in preset.prior.shares}
    assert families == set(FAMILY_TITLES) and len(set(FAMILY_TITLES.values())) == 8
    assert (FAMILY_TITLES['plain'], FAMILY_TITLES['attr2'], FAMILY_TITLES['real']) == ('Rule', 'Attribute-dependency', 'Real subgraph')
