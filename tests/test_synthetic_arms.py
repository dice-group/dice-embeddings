"""The arm presets and their initial weights against the research runs.

G1 (g1_orders.json.gz): each preset's mixture, order name, hooks and options equal the research run options of its arm,
and its pinned order digests equal the research orders. G3 (g3_init.json.gz): the initial weights of S (= HT, N, R, M),
CU and UL for seeds 42, 43 and 44, as digests over ULTRA's 82 keys (6-row relation tables with pair channels).
"""
import pytest
import torch

from dicee.synthetic_kg.arms import ARMS, SEEDS, arm_preset, initial_model, research_record
from dicee.synthetic_kg.digests import per_key_digests, tensor_digest
from dicee.synthetic_kg.priors import build_order, order_sha256
from tests.proofpfn_fixtures import fixture, research_prior

G1 = fixture('g1_orders.json.gz')
G3 = fixture('g3_init.json.gz')
INIT_OF = {'S': 'S', 'R': 'S', 'M': 'S', 'N': 'S', 'HT': 'S', 'CU': 'CU', 'UL': 'UL'}   # the research initialization of each arm


@pytest.fixture(autouse=True)
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('name', sorted(ARMS))
def test_presets_equal_the_research_arms(name):
    preset, entry = ARMS[name], G1['arms'][name]
    assert preset.name == name and preset.research_arm == entry['arm_name']
    assert preset.prior.mix == entry['mix_string'] and preset.prior.order_name == entry['order_name']
    expected = research_prior(entry)
    assert {s.name: s for s in preset.prior.families} == {s.name: s for s in expected.families}
    assert research_record(preset) == entry['record']
    assert preset.order_sha256 == tuple((int(seed), v['order_sha256']) for seed, v in sorted(entry['seeds'].items()))
    assert all(v['equal_to_pins'] for v in entry['seeds'].values())


@pytest.mark.parametrize('name', sorted(ARMS))
@pytest.mark.parametrize('seed', SEEDS)
def test_preset_orders(name, seed):
    preset = ARMS[name]
    order = build_order(seed, preset.prior)
    assert order_sha256(order) == preset.pinned_order(seed) == G1['arms'][name]['seeds'][str(seed)]['order_sha256']
    assert order_sha256(order[:400]) == G1['arms'][name]['seeds'][str(seed)]['first400_sha256']


def test_model_options():
    assert ARMS['S'].model_config() == dict(ultra_dim=64, ultra_num_layers=6, ultra_relation_weighting='normalized', ultra_pair_channels='on')
    assert {name: (p.relation_weighting, p.pair_channels, p.target) for name, p in ARMS.items()} == {
        'S': ('normalized', 'on', 'rbw'), 'R': ('normalized', 'on', 'rbw'), 'M': ('normalized', 'on', 'rbw'),
        'N': ('normalized', 'on', 'rbw'), 'HT': ('normalized', 'on', 'hard'), 'CU': ('normalized', 'off', 'rbw'),
        'UL': ('binary', 'off', 'rbw')}
    assert arm_preset('CU') is ARMS['CU'] and arm_preset(ARMS['CU']) is ARMS['CU']
    with pytest.raises(KeyError, match='unknown arm'):
        arm_preset('S0')
    assert ARMS['S'].pinned_order(45) is None


@pytest.mark.parametrize('name', sorted(ARMS))
@pytest.mark.parametrize('seed', SEEDS)
def test_initial_weights_equal_the_research_initialization(name, seed):
    golden = G3['arms'][INIT_OF[name]][str(seed)]
    assert golden['equal_to_pins']
    state = initial_model(name, seed).state_dict()
    assert list(state) == G3['ultra_keys']
    rows = 6 if ARMS[name].pair_channels == 'on' else 4
    assert golden['clean']['table_rows'] == rows and all(state[k].shape[0] == rows for k in G3['relation_tables'])
    assert per_key_digests(state) == golden['clean']['per_key']
    assert tensor_digest(state) == golden['clean']['digest']


def test_initialization_leaves_the_random_state_unchanged():
    torch.manual_seed(123)
    before = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    model = initial_model('S', 42)
    assert torch.equal(torch.get_rng_state(), before)
    if cuda is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda)
    assert model.training and model.relation_weighting == 'normalized' and model.pair_channels == 'on'
    assert tensor_digest(initial_model('S', 42).state_dict()) == tensor_digest(model.state_dict())
    assert torch.equal(torch.get_rng_state(), before)
