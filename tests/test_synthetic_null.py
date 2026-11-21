"""The structural null of PCUltra-N (``priors.null`` through ``FamilySpec.null_swap_budget``) against its pinned worlds."""
import numpy as np
import pytest

from dicee.synthetic_kg.priors import FamilySpec, null
from dicee.synthetic_kg.streams import null_stream, sha256_int
from tests.proofpfn_fixtures import PRIORS
from tests.synthetic_worlds import WORLDS, check_world, pinned_names

FAST = ('42/null/categorical2/0', '42/null/attr2/3', '42/null/list1/4', '44/null/rec1/0')


@pytest.mark.parametrize('name', FAST)
def test_null_world_equals_its_pin(name):
    check_world(name)


@pytest.mark.slow
def test_every_pinned_null_world():
    names = pinned_names({'categorical2', 'attr2', 'rec1', 'list1'}, {'null'})
    assert len(names) == 36
    for name in names:
        check_world(name)


def test_the_group_tests_cover_every_pinned_world():
    groups = [pinned_names({'plain'}), pinned_names({'structured', 'statistical'}), pinned_names({'categorical2', 'attr2', 'rec1', 'list1'}, {'synthetic'}),
              pinned_names({'real'}), pinned_names({'categorical2', 'attr2', 'rec1', 'list1'}, {'null'})]
    assert sorted(name for names in groups for name in names) == sorted(WORLDS) and len(WORLDS) == 123
    assert {tuple(name.split('/')[1:3]) for name in WORLDS} == {('synthetic', f) for f in PRIORS['synthetic'].shares} | \
        {('null', s.name) for s in PRIORS['null'].families} | {('real', 'real')}


def test_null_streams_and_spec():
    assert null_stream('rec1', 42, 8, True) == sha256_int('step5-null-v1:rec1:42:8')
    assert null_stream('rec1', 42, 8, False) == sha256_int('step5-null-v2:rec1:42:8')
    with pytest.raises(ValueError, match='record families'):
        FamilySpec('real', 'real', 400_000, null_swap_budget=3)


def test_null_swaps_keep_degrees_types_and_symmetry():
    rng = np.random.default_rng(0)
    directed = {(int(h), 0, int(t)) for h, t in rng.integers(0, 40, (300, 2)) if h != t}
    pairs = {tuple(sorted(map(int, p))) for p in rng.integers(40, 80, (200, 2)) if p[0] != p[1]}
    symmetric = {(a, 1, b) for a, b in pairs} | {(b, 1, a) for a, b in pairs} | {(45, 1, 45)}
    facts = np.array(sorted(directed | symmetric), dtype=np.int64)
    domains = [np.arange(40), np.arange(40, 80)]
    new, summary = null.swap_truth(facts, domains, domains, 80, 42, 7, 'attr2', 3.0)
    again, _ = null.swap_truth(facts, domains, domains, 80, 42, 7, 'attr2', 3.0)
    assert np.array_equal(new, again) and len(new) == len(facts) and not np.array_equal(new, facts)
    for column in (0, 2):
        for r in (0, 1):
            before = np.unique(facts[facts[:, 1] == r][:, column], return_counts=True)
            after = np.unique(new[new[:, 1] == r][:, column], return_counts=True)
            assert all(np.array_equal(x, y) for x, y in zip(before, after))
    sym = {tuple(map(int, q)) for q in new[new[:, 1] == 1]}
    assert sym == {(t, r, h) for h, r, t in sym} and (45, 1, 45) in sym
    assert set(summary) == {'swap_version', 'swap_budget', 'swap_moved_share', 'swap_accepted_per_fact', 'swap_reached_share'}
    assert summary['swap_version'] == 'step5-null-v2' and summary['swap_budget'] == 3.0 and 0 < summary['swap_moved_share'] <= 1
