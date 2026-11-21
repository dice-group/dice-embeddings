"""The evaluation on held-out worlds of the prior (``benchmarks.proofpfn.prior``).

The pins (``tests/fixtures/proofpfn/pins.json``, section ``prior``) hold per-world MRRs of two groups of three models on
2 held-out worlds per family, and the statistics that ``compare`` gives for them. The values belong to models with 20
updates and are references for equality only. The pins also hold the digest of the 100 plain held-out slots.
"""
import pytest

from benchmarks.proofpfn import prior as pr
from benchmarks.proofpfn import systems as sy
from dicee.synthetic_kg.priors import Slot
from tests.proofpfn_fixtures import clean_checkpoints

from .helpers import int_digest, pins

PINS = pins()['prior']


def test_compare_reproduces_the_pinned_statistics():
    assert pr.compare(PINS['per_world'], PINS['families']) == PINS['results']
    assert PINS['families'] == list(pr.FAMILIES)


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
    """Two models on one held-out world: 30 queries, an MRR in (0, 1] per model, and the difference of the two groups."""
    groups = {name: sy.checkpoints(f'{name}={clean_checkpoints(tmp_path, variant)}', (42,))
              for name, variant in (('ts', 'PCUltra-S'), ('tu', 'ULTRA-S'))}
    record = pr.evaluate(groups, ['attr2'], 1)
    assert record['queries'] == {'attr2': [30]} and list(record['models']) == ['ts', 'tu']
    for name in groups:
        values = record['per_world'][name][f'{name}.seed42']['attr2']
        assert len(values) == 1 and 0 < values[0] <= 1
        assert record['results'][name]['attr2'] == dict(worlds=1, MRR=values[0], seeds=values)
    difference = record['difference']
    assert (difference['a'], difference['b']) == ('ts', 'tu') and difference['families']['attr2']['worlds'] == 1
    assert record['provenance']['threads'] == 2 and record['world_seed'] == 42


def test_one_model_gives_no_difference(tmp_path, monkeypatch):
    systems = sy.checkpoints('gs=x-{seed}.pt', (42,))
    monkeypatch.setattr(pr, 'score', lambda models, families, per_family: dict(mrr={'gs': {'gs.seed42': {'plain': [0.5, 0.25]}}}, queries={'plain': [30, 30]}))
    monkeypatch.setattr(sy.Checkpoint, 'build_model', lambda self, *args: __import__('dicee').models.ultra.ULTRA({}))
    monkeypatch.setattr(sy.Checkpoint, 'sha256', lambda self: '0' * 64)
    record = pr.evaluate({'gs': systems}, ['plain'], 2)
    assert 'difference' not in record and record['results']['gs']['plain'] == dict(worlds=2, MRR=0.375, seeds=[0.375])
