"""The legacy plain generator (``priors.legacy`` with ``priors.factorial``) against golden G2, and its parts.

G2's plain worlds (``tests/synthetic_worlds.py``) are rule worlds, rule worlds with new rule programs, with pair-affinity
rewiring, and control worlds; the templates of the other families add more of each kind.
"""
import hashlib
import random

import numpy as np
import pytest
import torch

from dicee.synthetic_kg.arms import ARMS
from dicee.synthetic_kg.priors import Slot, factorial, legacy, world
from dicee.synthetic_kg.streams import blake2b_int, joint_world_stream, pair_affinity_stream, rule_program_stream, sha256_int, world_first_stream
from tests.synthetic_worlds import check_world, golden_names, require_slow

FAST = ('42/S/plain/0',    # a rule world
        '42/S/plain/1',    # with new rule programs
        '42/S/plain/4',    # with pair-affinity rewiring and new rule programs
        '42/S/plain/2',    # a control world: family current, core topology
        '42/S/plain/50')   # a control world: split composition, attached topology


@pytest.mark.parametrize('name', FAST)
def test_plain_world_equals_the_research_world(name):
    check_world(name)


@pytest.mark.slow
def test_every_golden_plain_world(request):
    require_slow(request)
    names = golden_names({'plain'})
    assert len(names) == 11
    for name in names:
        check_world(name)


def test_world_leaves_global_random_state_unchanged():
    states = random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state()
    check_world('42/S/plain/50')
    assert random.getstate() == states[0] and np.array_equal(np.random.get_state()[1], states[1])
    assert torch.equal(torch.get_rng_state(), states[2])


def test_slots_outside_their_streams_are_refused():
    prior = ARMS['S'].prior
    for slot in (Slot('plain', 's1', 20_000), Slot('plain', 's2', 19_999), Slot('structured', 's2', 7), Slot('rec1', 's1', 0)):
        with pytest.raises(ValueError, match='does not belong'):
            world(42, slot, prior)


def test_streams():
    assert blake2b_int('k') == int.from_bytes(hashlib.blake2b(b'k', digest_size=8).digest(), 'big')
    assert world_first_stream(7, 'world', ('train', 3)) == blake2b_int("world-first-v1:7:world:('train', 3)")
    assert joint_world_stream(7, 'sample:rule') == blake2b_int('joint-worlds-v1:7:sample:rule')
    assert rule_program_stream(7, 'program') == sha256_int('rule-program-v1:7:program')
    assert pair_affinity_stream(7, 'latent') == sha256_int('pair-affinity-v1:7:latent')


def test_derive_gives_the_closure_and_the_first_witness():
    rules = (factorial.Rule((0, 1), 2), factorial.Rule((2,), 3, True))
    base = {(0, 0, 1), (1, 1, 2), (0, 0, 3), (3, 1, 2)}
    truth, proofs = factorial.derive(base, rules)
    assert truth == frozenset(base | {(0, 2, 2), (2, 3, 0)})
    assert proofs == {(0, 2, 2): (0, ((0, 0, 1), (1, 1, 2))), (2, 3, 0): (1, ((0, 2, 2),))}
    assert factorial.proof_leaves((2, 3, 0), frozenset(base), proofs) == {(0, 0, 1), (1, 1, 2)}
    with pytest.raises(ValueError, match='budget'):
        factorial.derive(base, rules, limit=5)
    with pytest.raises(ValueError, match='only unary'):
        factorial.derive(base, (factorial.Rule((0, 1), 2, True),))


def test_legacy_source_selection():
    for block in range(362_000, 362_400, 10):
        assert [legacy.source_arm(42, i) for i in range(block, block + 10)].count('control') == 2
        assert sum(legacy.pair_selected(42, i) for i in range(block, block + 10)) == 2
        assert sum(legacy.program_selected(42, i - legacy.SAMPLER_OFFSET) for i in range(block, block + 10)) == 4
    assert factorial.task_quotas(362_001, 30) == {'rule': (15, 8), 'base': (15, 7)}
    with pytest.raises(ValueError, match='starts at'):
        legacy.source_arm(42, 249)
