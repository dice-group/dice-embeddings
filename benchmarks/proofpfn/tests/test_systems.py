"""The systems: model options and labels, built models, the reference pins, and Flock's official setting."""
import pytest
import torch

from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn import systems as sy
from dicee.synthetic_kg.variants import SEEDS, initial_model
from tests.proofpfn_fixtures import clean_checkpoints

from .helpers import GATE, pins


def test_model_options_give_one_system_per_seed(tmp_path):
    template = clean_checkpoints(tmp_path, 'PCUltra-S')
    systems = sy.checkpoints(f'ts={template}')
    assert [s.label for s in systems] == ['ts.seed42', 'ts.seed43', 'ts.seed44']
    assert [s.path for s in systems] == [tmp_path / f'PCUltra-S-{seed}.pt' for seed in SEEDS]
    assert [s.label for s in sy.checkpoints(f'ts={template}', (7,))] == ['ts.seed7']
    one = sy.checkpoints(f"single={tmp_path / 'PCUltra-S-42.pt'}")
    assert [s.label for s in one] == ['single'] and one[0].sha256() == mf.file_sha256(tmp_path / 'PCUltra-S-42.pt')
    spec = one[0].spec()
    assert spec['kind'] == 'checkpoint' and (spec['variant'], spec['seed'], spec['updates'], spec['weights']) == ('PCUltra-S', 42, 0, 'raw')
    assert spec['model_config']['ultra_pair_channels'] == 'on'
    for bad, message in (('ts', 'NAME=TEMPLATE'), ('=x.pt', 'NAME=TEMPLATE'), ('ts=', 'NAME=TEMPLATE'), ('4s=x.pt', 'starting with a letter'),
                         ('t s=x.pt', 'letters, digits'), ('trix=x.pt', 'label of a reference'), ('ts=x{seed}-{other}.pt', 'no other field'),
                         ('ts=x{}.pt', 'no other field'), ('ts=x{seed.pt', 'template')):
        with pytest.raises(sy.SystemCheckError, match=message):
            sy.checkpoints(bad)


def test_a_missing_or_invalid_file_is_refused(tmp_path):
    (tmp_path / 'text.pt').write_text('not a checkpoint')
    with pytest.raises(sy.SystemCheckError, match='ts: text.pt does not load'):
        sy.checkpoints(f"ts={tmp_path / 'text.pt'}")[0].check()
    with pytest.raises(sy.SystemCheckError, match='ts: missing'):
        sy.checkpoints(f"ts={tmp_path / 'none.pt'}")[0].check()


def test_built_models_are_frozen_and_leave_the_random_state(tmp_path):
    systems = sy.checkpoints(f"ts={clean_checkpoints(tmp_path, 'PCUltra-S')}")
    before = torch.get_rng_state()
    models = [system.build_model(0, 0) for system in systems]
    assert torch.equal(torch.get_rng_state(), before)
    for seed, model in zip(SEEDS, models):
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert (model.relation_weighting, model.pair_channels) == ('normalized', 'on')
        reference = initial_model('PCUltra-S', seed).state_dict()
        assert all(torch.equal(v, reference[k]) for k, v in model.state_dict().items())


def test_references(tmp_path):
    assert set(sy.REFERENCES) == {'trix', 'ultra3g', 'flock'}
    flock = sy.reference('flock', tmp_path)
    assert isinstance(flock, sy.OfficialFlock) and flock.path == tmp_path / 'flock' / 'flock_entity.pth'
    assert flock.sha256() == sy.REFERENCES['flock'][1] and flock.spec()['model'] == 'Flock'
    assert not isinstance(sy.reference('trix', tmp_path), sy.OfficialFlock) and sy.reference('trix', tmp_path).spec()['model'] == 'TRIX'
    with pytest.raises(sy.SystemCheckError, match='ultra3g: missing'):
        sy.reference('ultra3g', tmp_path).check()
    (tmp_path / 'ultra_3g.pth').write_text('another file')
    with pytest.raises(sy.SystemCheckError, match='differs from the released checkpoint'):
        sy.reference('ultra3g', tmp_path).check()
    with pytest.raises(sy.SystemCheckError, match='unknown reference'):
        sy.reference('flockoff', tmp_path)


def test_flock_setting_of_the_gate_graphs():
    gate = mf.read_manifest(GATE)
    for name, pinned in pins()['gate'].items():
        entry = gate.entry(name)
        setting = sy.flock_setting(entry.size['entities'], entry.size['inference_edges'], entry.suite_id)
        assert setting == pinned['flock'], name
        assert (setting['walks'], setting['passes']) == (16, 16)


def test_the_script_table_covers_the_standard_suite():
    assert {e.suite_id for e in mf.read_manifest('ultra54').entries} == set(sy.FLOCK_SCRIPT)


def test_flock_rule():
    raw, walks = sy.rule_walks(24038, 181511)
    assert raw == pytest.approx(128, rel=1e-5) and walks == 128
    assert [sy.rule_passes(w) for w in (16, 128, 256, 512)] == [16, 16, 4, 2]
    assert sy.flock_setting(10, 10)['walks'] == 16 and sy.flock_setting(10 ** 7, 10 ** 8)['walks'] == 512
    assert sy.flock_setting(10, 10, 'YAGO310')[('walks')] == 512 and sy.flock_setting(10, 10, 'YAGO310')['passes'] == 8
    with pytest.raises(ValueError):
        sy.rule_walks(0, 5)
