"""The evaluation on held-out worlds of the prior (``benchmarks.proofpfn.prior``).

The pins (``tests/fixtures/proofpfn/pins.json``, section ``prior``) hold the digest of the 100 plain held-out slots.
"""
import numpy as np
import pytest
import torch

from benchmarks.proofpfn import prior as pr
from benchmarks.proofpfn import systems as sy
from benchmarks.proofpfn.data import rank_of
from dicee.synthetic_kg.priors import Slot
from tests.proofpfn_fixtures import clean_checkpoints

from .helpers import int_digest, pins

PINS = pins()['prior']


def test_the_answer_stays_a_candidate_also_as_its_own_anchor():
    assert pr.banned_candidates({3, 5, 7}, anchor=1, target=5) == {1, 3, 7}
    assert pr.banned_candidates({4, 6}, anchor=4, target=4) == {6}   # a self-loop fact: the anchor is the answer
    scores = np.array([0.1, 0.9, 0.5, 0.7, 2.0])
    assert rank_of(scores, 4, pr.banned_candidates({4, 1}, anchor=4, target=4)) == 1.0
    assert rank_of(scores, 2, pr.banned_candidates({2, 3}, anchor=4, target=2)) == 2.0   # entity 1 scores higher; 3 and 4 leave


def test_heldout_slots():
    assert pr.heldout_slots('rec1', 3) == [Slot('rec1', 'hook', 8_810_000 + k) for k in range(3)]
    assert pr.heldout_slots('categorical2', 1) == [Slot('categorical2', 'hook', 7_700_000)]
    assert [s.stream_index for s in pr.heldout_slots('plain', 2)] == [120_000, 120_002]
    plain = pr.heldout_slots('plain', 100)
    assert len(plain) == 100 and int_digest([s.stream_index for s in plain]) == PINS['plain_slots']
    for bad in (('plain', 101), ('rec1', 0), ('structured', 1)):
        with pytest.raises(ValueError):
            pr.heldout_slots(*bad)


def test_wrong_input_and_nondeterministic_cuda_are_refused(monkeypatch):
    systems = sy.checkpoints('gs=x-{seed}.pt')
    for bad in (dict(groups={}), dict(groups={'gs': []}), dict(groups={'gs': systems}, families=['structured']),
                dict(groups={'gs': systems}, per_family=101)):
        with pytest.raises(ValueError):
            pr.evaluate(**bad)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(RuntimeError, match='CUBLAS_WORKSPACE_CONFIG'):
        pr.evaluate({'gs': systems}, device='cuda')


def test_two_models_score_one_world_of_a_family(tmp_path):
    """Two models on one held-out world: 30 queries and an MRR in (0, 1] per model."""
    groups = {name: sy.checkpoints(f'{name}={clean_checkpoints(tmp_path, variant)}', (42,))
              for name, variant in (('ts', 'PCUltra-S'), ('tu', 'ULTRA-S'))}
    record = pr.evaluate(groups, ['attr2'], 1)
    assert record['queries'] == {'attr2': [30]} and list(record['models']) == ['ts', 'tu']
    for name in groups:
        values = record['per_world'][name][f'{name}.seed42']['attr2']
        assert len(values) == 1 and 0 < values[0] <= 1
        assert record['results'][name]['attr2'] == dict(worlds=1, MRR=values[0], seeds=values)
    assert record['provenance']['threads'] == 2 and record['world_seed'] == 42


def test_a_score_that_is_not_finite_is_refused(tmp_path):
    from dicee.synthetic_kg.prepare import prepare_world
    from dicee.synthetic_kg.priors import world
    from dicee.synthetic_kg.variants import VARIANTS, initial_model
    slot = pr.heldout_slots('attr2', 1)[0]
    generated = world(pr.WORLD_SEED, slot, VARIANTS['PCUltra-S'].prior)
    model = initial_model('PCUltra-S', 42)
    with torch.no_grad():
        next(model.parameters()).fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='not finite'):
        pr.reciprocal_ranks(model, generated, prepare_world(generated, slot, 'normalized', 'on'))
