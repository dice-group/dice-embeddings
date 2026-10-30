"""The command line: family options, list, check, run and cells on a tiny graph set, the refusals of wrong input with
one error line, the traceback of an internal error, and fv1-models."""
import argparse
import json
import os

import pytest

from benchmarks.proofpfn import cli, evaluate
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


def test_every_command_and_option_has_a_help_text(capsys):
    parser = cli.build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert [a.dest for a in commands._choices_actions] == list(cli.COMMANDS)   # the commands of the top-level --help
    for name, sub in commands.choices.items():
        assert all(action.help for action in sub._actions), name
    with pytest.raises(SystemExit):
        cli.main(['--help'])
    text = capsys.readouterr().out
    assert all(f'    {name} ' in text for name in cli.COMMANDS)


def test_list(capsys):
    assert cli.main(['list', '--set', 'dev121', '--view', 'apart']) == 0
    assert capsys.readouterr().out.split() == ['UMLS']
    assert cli.main(['list', '--set', 'gate']) == 0
    assert capsys.readouterr().out.split() == [e.key for e in mf.read_manifest('gate').entries]


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


def test_check_reports_every_failing_graph(tmp_path, capsys):
    folders = {name: tiny_graph(tmp_path / 'KGs' / name, seed=seed) for seed, name in enumerate('fgh')}
    manifest = write_manifest(tmp_path / 'tiny.json', tmp_path, folders)
    (folders['f'] / 'test.txt').write_text('x\tr0\ty\n')   # another file
    (folders['g'] / 'valid.txt').unlink()                   # a missing file
    common = ['--set', str(manifest), '--data-root', str(tmp_path), '--family', f"ts=S:{clean_checkpoints(tmp_path, 'S')}"]
    assert cli.main(['check', *common]) == cli.EXIT_INPUT
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[-4:-1]] == ['tiny/f', 'tiny/g', 'tiny/h'] and lines[-2].endswith(' ok')
    assert 'FAILED' in lines[-4] and 'FAILED' in lines[-3] and lines[-1] == 'check FAILED: 2 of 3 graphs (tiny/f, tiny/g)'


def test_refusals(tmp_path, capsys):
    folder = write_graph(tmp_path / 'KGs' / 'lockbox3' / 'g', ['a\tr\tb'], [], ['b\tr\ta'])
    blind = write_manifest(tmp_path / 'blind.json', tmp_path, {'g': folder}, name='blindtest', blind=True)
    assert cli.main(['list', '--set', str(blind)]) == cli.EXIT_INPUT
    assert cli.main(['list', '--set', str(blind), '--release', 'blindtest']) == 0
    assert cli.main(['list', '--set', 'gate', '--graphs', 'FB15k-237']) == cli.EXIT_INPUT
    template = clean_checkpoints(tmp_path, 'S')
    for family in (f'HT:{template}', f'rec10axlepcunoa=CU:{template}'):   # S checkpoints under HT's pins; the key of another arm
        assert cli.main(['check', '--set', 'gate', '--graphs', 'UMLS', '--family', family]) == cli.EXIT_INPUT
    assert cli.main(['check', '--set', 'gate', '--graphs', 'UMLS', '--family', f'S:{template}']) == cli.EXIT_INPUT   # not the pinned exports
    capsys.readouterr()
    for command in (['list', '--set', 'nosuch'], ['check', '--set', 'gate', '--references', 'trix', '--models', str(tmp_path / 'none.json')],
                    ['run', '--set', 'gate', '--references', 'trix', '--shard', '1-2', '--out', str(tmp_path / 'rows')],
                    ['gate', '--family', f'gs=S:{template}', '--family', f'gn=N:{template}', '--per-family', '101', '--out', str(tmp_path / 'g.json')]):
        assert cli.main(command) == cli.EXIT_INPUT, command
        error = capsys.readouterr().err
        assert error.startswith('error: ') and error.count('\n') == 1, (command, error)   # one line, no traceback


def test_internal_errors_keep_their_traceback(tmp_path, monkeypatch):
    def broken(*args: object, **kwargs: object) -> str:
        raise ValueError('an internal error')
    monkeypatch.setattr(evaluate, 'evaluate_graph', broken)
    with pytest.raises(ValueError, match='an internal error'):
        cli.main(['run', '--set', 'gate', '--graphs', 'UMLS', '--references', 'trix', '--device', 'cpu', '--out', str(tmp_path)])


def test_fv1_models(tmp_path):
    assert cli.main(['fv1-models', '--out', str(tmp_path / 'models.json')]) == 0
    assert json.loads((tmp_path / 'models.json').read_text()) == mf.read_models().fv1_models()
