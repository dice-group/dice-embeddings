"""The evaluation on held-out worlds of the prior (``benchmarks.proofpfn.prior``) against golden G9.

G9 (``golden/g9_gate.json``) is the research run (``step5/s5_manip.py``) on the G4 exports of S and N at update 20,
EMA weights, seeds 42 to 44, with 2 held-out worlds per family on the CPU. Its values belong to models with 20 updates
and are references for equality only. The model checks need the large goldens (``PROOFPFN_GOLDEN``).
"""
import json
from pathlib import Path

import pytest

from benchmarks.proofpfn import cli
from benchmarks.proofpfn import prior as pr
from benchmarks.proofpfn import systems as sy
from benchmarks.proofpfn.tests.golden import fixture, golden_root
from dicee.synthetic_kg.arms import SEEDS
from dicee.synthetic_kg.checkpoint import from_research_export, save_checkpoint
from dicee.synthetic_kg.priors import Slot

G9 = fixture('g9_gate.json')
RUNS = {'S': 'REC10A-XLE-PCU-noA', 'N': 'PCU-N'}   # the research --arm of the G4 exports of S and N
INDEX = Path('inputs') / 'kiowl' / 'prior-fix-v1' / 'attr-v1' / 'heldout' / 'index.json'


def g4_template(arm: str, folder: Path) -> str:
    """The G4 exports of an arm at update 20, converted to checkpoints in ``folder``; the template (skips without the
    large goldens)."""
    source = golden_root() / 'golden' / 'large' / 'g4' / 'golden-runs' / RUNS[arm] / 'training'
    if not all((source / str(seed) / 'export-20.pt').is_file() for seed in SEEDS):
        pytest.skip(f'the G4 exports of {arm} are not available')
    for seed in SEEDS:
        save_checkpoint(from_research_export(source / str(seed) / 'export-20.pt', arm), folder / f'{arm}-{seed}.pt')
    return str(folder / f'{arm}-{{seed}}.pt')


def golden_per_world(arm: str) -> list[dict]:
    """G9's per-world MRRs of group A (S) or B (N), in seed order."""
    return list(G9['per_world']['a' if arm == 'S' else 'b'].values())


def test_compare_reproduces_the_research_statistics():
    assert pr.compare(G9['per_world'], G9['families']) == G9['results']


def test_heldout_slots():
    assert pr.heldout_slots('rec1', 3) == [Slot('rec1', 'hook', 8_810_000 + k) for k in range(3)]
    assert pr.heldout_slots('categorical2', 1) == [Slot('categorical2', 'hook', 7_700_000)]
    assert [s.stream_index for s in pr.heldout_slots('plain', 2)] == [120_000, 120_002]   # G9's plain worlds
    for bad in (('plain', 101), ('rec1', 0), ('structured', 1)):
        with pytest.raises(ValueError):
            pr.heldout_slots(*bad)


def test_plain_slots_equal_the_heldout_index():
    path = golden_root() / INDEX
    if not path.is_file():
        pytest.skip(f'{INDEX} is not available')
    index = json.loads(path.read_text())
    plain = [tuple(slot) for slot, file in index if slot[0] == 'plain' and 'lockbox3' not in file]
    assert len(plain) == 100 and plain == [tuple(s) for s in pr.heldout_slots('plain', 100)]


def test_wrong_input_and_nondeterministic_cuda_are_refused(monkeypatch):
    systems = sy.checkpoints('gs=x-{seed}.pt')
    for bad in (dict(groups={}), dict(groups={'gs': []}), dict(groups={'gs': systems}, families=['structured']),
                dict(groups={'gs': systems}, per_family=101)):
        with pytest.raises(ValueError):
            pr.evaluate(**bad)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(RuntimeError, match='CUBLAS_WORKSPACE_CONFIG'):
        pr.evaluate({'gs': systems}, device='cuda')


def test_one_world_scores_like_the_research_run(tmp_path):
    models = {}
    for group, arm in (('a', 'S'), ('b', 'N')):
        system = sy.checkpoints(f'g{arm.lower()}={g4_template(arm, tmp_path)}', (42,))[0]
        models[group] = {system.label: system.build_model(0, 0).train()}
    scored = pr.score(models, ['attr2'], 1)
    assert scored['queries'] == {'attr2': [30]}
    assert scored['mrr']['a']['gs.seed42']['attr2'] == golden_per_world('S')[0]['attr2'][:1]
    assert scored['mrr']['b']['gn.seed42']['attr2'] == golden_per_world('N')[0]['attr2'][:1]


@pytest.mark.slow
def test_the_command_equals_the_research_run(tmp_path, capsys):
    out = tmp_path / 'prior.json'
    assert cli.main(['prior', '--model', f"gs={g4_template('S', tmp_path)}", '--model', f"gn={g4_template('N', tmp_path)}", '--per-family', '2',
                     '--device', 'cpu', '--out', str(out)]) == 0
    record = json.loads(out.read_text())
    assert record['queries'] == G9['queries'] and record['difference'] == dict(a='gs', b='gn', families=G9['results'])
    for arm, name in (('S', 'gs'), ('N', 'gn')):
        assert list(record['per_world'][name].values()) == golden_per_world(arm)
        assert list(record['models'][name]) == [f'{name}.seed{seed}' for seed in SEEDS]
        for family in G9['families']:
            side = 'a' if arm == 'S' else 'b'
            assert record['results'][name][family]['seeds'] == G9['results'][family][f'seeds_{side}']
            assert record['results'][name][family]['MRR'] == pytest.approx(G9['results'][family][f'mrr_{side}'], abs=1e-15)
    assert record['provenance']['threads'] == 2 and '| D = gs - gn |' in capsys.readouterr().out


def test_one_model_gives_no_difference(tmp_path, monkeypatch):
    systems = sy.checkpoints('gs=x-{seed}.pt', (42,))
    monkeypatch.setattr(pr, 'score', lambda models, families, per_family: dict(mrr={'gs': {'gs.seed42': {'plain': [0.5, 0.25]}}}, queries={'plain': [30, 30]}))
    monkeypatch.setattr(sy.Checkpoint, 'build_model', lambda self, *args: __import__('dicee').models.ultra.ULTRA({}))
    monkeypatch.setattr(sy.Checkpoint, 'sha256', lambda self: '0' * 64)
    record = pr.evaluate({'gs': systems}, ['plain'], 2)
    assert 'difference' not in record and record['results']['gs']['plain'] == dict(worlds=2, MRR=0.375, seeds=[0.375])
