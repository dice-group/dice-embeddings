"""The N validity gate (``benchmarks.proofpfn.gate``) against golden G9.

G9 (``golden/g9_gate.json``) is the research gate (``step5/s5_manip.py``) on the G4 exports of S and N at update 20,
EMA weights, seeds 42 to 44, with 2 held-out worlds per family on the CPU. Its values belong to models with 20 updates
and are references for equality only. The model checks need the large goldens (``PROOFPFN_GOLDEN``).
"""
import json
from pathlib import Path

import pytest

from benchmarks.proofpfn import cli
from benchmarks.proofpfn import gate as gt
from benchmarks.proofpfn import systems as sy
from benchmarks.proofpfn.manifests import file_sha256
from benchmarks.proofpfn.tests.golden import fixture, golden_root
from dicee.synthetic_kg.arms import SEEDS
from dicee.synthetic_kg.priors import Slot

G9 = fixture('g9_gate.json')
RUNS = {'S': 'REC10A-XLE-PCU-noA', 'N': 'PCU-N'}   # the research --arm of the G4 exports of S and N
INDEX = Path('inputs') / 'kiowl' / 'prior-fix-v1' / 'attr-v1' / 'heldout' / 'index.json'


def g4_family(arm: str) -> sy.ProofPFNFamily:
    """The G4 exports of an arm at update 20 as a test family (skips without the large goldens)."""
    folder = golden_root() / 'golden' / 'large' / 'g4' / 'golden-runs' / RUNS[arm] / 'training'
    if not all((folder / str(seed) / 'export-20.pt').is_file() for seed in SEEDS):
        pytest.skip(f'the G4 exports of {arm} are not available')
    return sy.ProofPFNFamily(key=f'g{arm.lower()}', arm=arm, checkpoints={seed: folder / str(seed) / 'export-20.pt' for seed in SEEDS})


def golden_per_world(arm: str) -> list[dict]:
    """G9's per-world MRRs of arm A (S) or B (N), in seed order."""
    return list(G9['per_world']['a' if arm == 'S' else 'b'].values())


def test_compare_reproduces_the_research_statistics():
    assert gt.compare(G9['per_world'], G9['families']) == G9['results']


def test_reading_of_amendment_a1():
    results = {family: dict(D=0.01, se=0.004) for family in gt.NULLED} | {'plain': dict(D=-0.003, se=0.002)}
    assert gt.reading(results) == dict(n_validity='pass', nulled_below=dict.fromkeys(gt.NULLED, True), plain_within=True)
    assert gt.reading(results | {'attr2': dict(D=0.008, se=0.004)})['n_validity'] == 'fail'   # not more than 2 SE
    assert gt.reading(results | {'plain': dict(D=0.0041, se=0.002)})['plain_within'] is False
    assert gt.reading(G9['results'])['n_validity'] == 'fail'   # the 20-update golden models: every family within 2 SE
    with pytest.raises(ValueError, match='every family'):
        gt.reading({'plain': dict(D=0.0, se=1.0)})


def test_heldout_slots():
    assert gt.heldout_slots('rec1', 3) == [Slot('rec1', 'hook', 8_810_000 + k) for k in range(3)]
    assert gt.heldout_slots('categorical2', 1) == [Slot('categorical2', 'hook', 7_700_000)]
    assert [s.index for s in gt.heldout_slots('plain', 2)] == [120_000, 120_002]   # G9's plain worlds
    for bad in (('plain', 101), ('rec1', 0), ('structured', 1)):
        with pytest.raises(ValueError):
            gt.heldout_slots(*bad)


def test_plain_slots_equal_the_heldout_index():
    path = golden_root() / INDEX
    if not path.is_file():
        pytest.skip(f'{INDEX} is not available')
    index = json.loads(path.read_text())
    plain = [tuple(slot) for slot, file in index if slot[0] == 'plain' and 'lockbox3' not in file]
    assert len(plain) == 100 and plain == [tuple(s) for s in gt.heldout_slots('plain', 100)]


def test_gate_refuses_other_arms():
    family = sy.ProofPFNFamily(key='gs', arm='S', checkpoints={seed: Path(f'{seed}.pt') for seed in SEEDS})
    with pytest.raises(ValueError, match='compares S with N'):
        gt.gate(family, family)


def test_one_world_scores_like_the_research_gate():
    s, n = g4_family('S'), g4_family('N')
    models = {}
    for arm, family in (('a', s), ('b', n)):
        family.check()
        models[arm] = {f'{family.key}.seed42': sy.ultra_from_checkpoint(family.members[42].checkpoint).train()}
    scored = gt.score(models, ['attr2'], 1)
    assert scored['queries'] == {'attr2': [30]}
    assert scored['mrr']['a']['gs.seed42']['attr2'] == golden_per_world('S')[0]['attr2'][:1]
    assert scored['mrr']['b']['gn.seed42']['attr2'] == golden_per_world('N')[0]['attr2'][:1]


@pytest.mark.slow
def test_gate_equals_the_research_gate(tmp_path):
    s, n = g4_family('S'), g4_family('N')
    out = tmp_path / 'gate.json'
    assert cli.main(['gate', '--family', f'gs=S:{s.checkpoints[42]}'.replace('/42/', '/{seed}/'),
                     '--family', f'gn=N:{n.checkpoints[42]}'.replace('/42/', '/{seed}/'), '--per-family', '2', '--device', 'cpu',
                     '--out', str(out)]) == 0
    record = json.loads(out.read_text())
    assert record['queries'] == G9['queries'] and record['results'] == G9['results']
    for arm, side in (('S', 'a'), ('N', 'b')):
        assert list(record['per_world'][side].values()) == golden_per_world(arm)
    assert record['reading'] == gt.reading(G9['results'])
    for side, family in (('a', s), ('b', n)):   # a research export's model hash is its file's SHA-256
        assert record['systems'][side]['model_sha256'] == {str(seed): file_sha256(family.checkpoints[seed]) for seed in SEEDS}



def test_gate_command_needs_one_family_of_s_and_one_of_n(tmp_path, capsys):
    out = tmp_path / 'gate.json'
    assert cli.main(['gate', '--family', 'gs=S:x{seed}.pt', '--out', str(out)]) == cli.EXIT_INPUT
    assert cli.main(['gate', '--family', 'gs=S:x{seed}.pt', '--family', 'gc=CU:y{seed}.pt', '--out', str(out)]) == cli.EXIT_INPUT
    assert 'one --family of arm S and one of arm N' in capsys.readouterr().err and not out.exists()
