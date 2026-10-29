"""The systems: family pins and keys, the converter path, and Flock's official setting."""
import json
from pathlib import Path

import pytest
import torch

from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn import systems as sy
from dicee.synthetic_kg.arms import ARMS, initial_model
from dicee.synthetic_kg.checkpoint import make_checkpoint, save_checkpoint

from .golden import fixture, g4_export


def clean_checkpoints(folder, arm: str) -> str:
    """Checkpoints of an arm's initial weights for the three seeds; returns the template."""
    for seed in (42, 43, 44):
        order = ARMS[arm].pinned_order(seed)
        assert order is not None
        checkpoint = make_checkpoint(initial_model(arm, seed).state_dict(), arm=arm, seed=seed, updates=0, weights='raw',
                                     order_sha256=order, provenance={'source': 'test'})
        save_checkpoint(checkpoint, folder / f'{arm}-{seed}.pt')
    return str(folder / f'{arm}-{{seed}}.pt')


def test_family_keys_and_pins(tmp_path):
    pins = mf.read_models()
    template = clean_checkpoints(tmp_path, 'S')
    test_family = sy.family('ts', 'S', template, pins)   # an unregistered key: no pins
    test_family.check()
    assert test_family.pins is None and set(test_family.model_sha256()) == {'42', '43', '44'}
    assert test_family.model_sha256()['42'] == mf.file_sha256(tmp_path / 'S-42.pt')   # no research export: the file's hash
    registered = sy.family('rec10axlepcunoa', 'S', template, pins)
    assert registered.pins == pins.families['S'].export_sha256
    with pytest.raises(sy.SystemCheckError, match='pinned research export'):
        registered.check()
    assert sy.family('pcuht', 'HT', template, pins).pins == pins.families['HT'].export_sha256
    for key, arm, message in (('rec10axlepcunoa', 'CU', 'registered key of arm S'), ('rec10axlenoa', 'S', 'registered key of arm S0'),
                              ('trix', 'S', 'reference key'), ('Ts', 'S', 'lower-case')):
        with pytest.raises(sy.SystemCheckError, match=message):
            sy.family(key, arm, template, pins)
    doc = json.loads((mf.MANIFESTS / 'models.json').read_text())
    doc['families']['HT']['export_sha256'] = None   # a registered key whose pins are pending is refused
    (tmp_path / 'pending.json').write_text(json.dumps(doc))
    with pytest.raises(sy.SystemCheckError, match='no pins'):
        sy.family('pcuht', 'HT', template, mf.read_models(tmp_path / 'pending.json'))
    with pytest.raises(sy.SystemCheckError, match='seed'):
        sy.family('ts', 'S', str(tmp_path / 'S.pt'), pins)
    with pytest.raises(Exception, match='arm S, not UL'):
        sy.family('tu', 'UL', template, pins).check()


def test_built_models_are_frozen_and_leave_the_random_state(tmp_path):
    family = sy.family('ts', 'S', clean_checkpoints(tmp_path, 'S'), mf.read_models())
    before = torch.get_rng_state()
    models = family.build()
    assert torch.equal(torch.get_rng_state(), before)
    assert list(models) == [('ts', 42), ('ts', 43), ('ts', 44)]
    for (_, seed), model in models.items():
        assert seed is not None and not model.training and not any(p.requires_grad for p in model.parameters())
        assert (model.relation_weighting, model.pair_channels) == ('normalized', 'on')
        reference = initial_model('S', seed).state_dict()
        assert all(torch.equal(v, reference[k]) for k, v in model.state_dict().items())


def test_research_exports_convert_under_their_pins():
    """A G4 research export loads through the converter; its file hash is the model hash and matches given pins."""
    template = g4_export('gs')
    pins = {s: mf.file_sha256(template.format(seed=s)) for s in (42, 43, 44)}
    family = sy.ProofPFNFamily(key='gs', arm='S', checkpoints={s: Path(template.format(seed=s)) for s in (42, 43, 44)}, pins=pins)
    family.check()
    assert family.model_sha256() == {str(s): v for s, v in pins.items()}
    assert family.members[42].checkpoint['provenance']['research_arm'] == 'REC10A-XLE-PCU-noA'
    wrong = sy.ProofPFNFamily(key='gs', arm='S', checkpoints=family.checkpoints, pins={**pins, 44: pins[42]})
    with pytest.raises(sy.SystemCheckError, match='seed 44'):
        wrong.check()


def test_flock_setting_of_the_gate_graphs():
    gate = mf.read_manifest('gate')
    for name, graph in fixture('g6_ranks.json.gz')['graphs'].items():
        recorded = graph['record']['flock_official']
        entry = gate.entry(name)
        setting = sy.flock_setting(entry.size['entities'], entry.size['inference_edges'], entry.suite_id)
        assert setting == {k: recorded[k] for k in setting}, name
        assert recorded['rule'] == sy.FLOCK_RULE and recorded['keys'] == {'flockoff': dict(walks=16, passes=16)}


def test_the_script_table_covers_t1():
    assert {e.suite_id for e in mf.read_manifest('t1').entries} == set(sy.FLOCK_SCRIPT)


def test_flock_rule():
    raw, walks = sy.rule_walks(24038, 181511)
    assert raw == pytest.approx(128, rel=1e-5) and walks == 128
    assert [sy.rule_passes(w) for w in (16, 128, 256, 512)] == [16, 16, 4, 2]
    assert sy.flock_setting(10, 10)['walks'] == 16 and sy.flock_setting(10 ** 7, 10 ** 8)['walks'] == 512
    assert sy.flock_setting(10, 10, 'YAGO310')[('walks')] == 512 and sy.flock_setting(10, 10, 'YAGO310')['passes'] == 8
    with pytest.raises(ValueError):
        sy.rule_walks(0, 5)
