"""The command line: family options, list, check, run and cells on a tiny graph set, the refusals, and fv1-models."""
import json
import os

import pytest

from benchmarks.proofpfn import cli
from benchmarks.proofpfn import manifests as mf

from .golden import tiny_graph, write_graph, write_manifest
from .test_systems import clean_checkpoints


def test_family_options():
    pins = mf.read_models()
    assert cli.parse_family('S:/runs/{seed}/export.pt', pins) == ('rec10axlepcunoa', 'S', '/runs/{seed}/export.pt')
    assert cli.parse_family('ts=CU:/a=b/{seed}.pt', pins) == ('ts', 'CU', '/a=b/{seed}.pt')
    for bad in ('S', 'X:/x/{seed}.pt'):
        with pytest.raises(ValueError):
            cli.parse_family(bad, pins)


def test_list(capsys):
    assert cli.main(['list', '--set', 'dev121', '--view', 'apart']) == 0
    assert capsys.readouterr().out.split() == ['UMLS']
    assert cli.main(['list', '--set', 'gate']) == 0
    assert capsys.readouterr().out.split() == list(mf.GATE_GRAPHS)


def test_check_run_and_cells(tmp_path, capsys):
    folder = tiny_graph(tmp_path / 'KGs' / 'g')
    manifest = write_manifest(tmp_path / 'tiny.json', tmp_path, {'g': folder})
    family = f"ts=S:{clean_checkpoints(tmp_path, 'S')}"
    common = ['--set', str(manifest), '--data-root', str(tmp_path), '--family', family]
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    assert cli.main(['check', *common]) == 0 and 'check passed (1 graphs' in capsys.readouterr().out
    assert cli.main(['run', *common, '--device', 'cpu', '--out', str(tmp_path / 'out')]) == 0
    doc = json.loads((tmp_path / 'out' / 'tiny' / 'ts' / 'g.json').read_text())
    assert doc['families'] == ['ts'] and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == workspace
    assert cli.main(['cells', '--set', str(manifest), '--data-root', str(tmp_path), '--out', str(tmp_path / 'queries.csv')]) == 0
    assert (tmp_path / 'queries.csv').read_text().count('\n') == 1 + len(doc['rows'])


def test_refusals(tmp_path, capsys):
    folder = write_graph(tmp_path / 'KGs' / 'lockbox3' / 'g', ['a\tr\tb'], [], ['b\tr\ta'])
    blind = write_manifest(tmp_path / 'blind.json', tmp_path, {'g': folder}, name='blindtest', blind=True)
    assert cli.main(['list', '--set', str(blind)]) == cli.EXIT_INPUT
    assert cli.main(['list', '--set', str(blind), '--release', 'blindtest']) == 0
    assert cli.main(['list', '--set', 'gate', '--graphs', 'FB15k-237']) == cli.EXIT_INPUT
    template = clean_checkpoints(tmp_path, 'S')
    for family in (f'HT:{template}', f'rec10axlepcunoa=CU:{template}'):   # pins pending; the key of another arm
        assert cli.main(['check', '--set', 'gate', '--graphs', 'UMLS', '--family', family]) == cli.EXIT_INPUT
    assert cli.main(['check', '--set', 'gate', '--graphs', 'UMLS', '--family', f'S:{template}']) == cli.EXIT_INPUT   # not the pinned exports
    assert 'error:' in capsys.readouterr().err


def test_fv1_models(tmp_path):
    assert cli.main(['fv1-models', '--out', str(tmp_path / 'models.json')]) == 0
    assert json.loads((tmp_path / 'models.json').read_text()) == mf.read_models().fv1_models()
