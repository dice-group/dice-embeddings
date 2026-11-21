"""The command line: list, check, run and table on a tiny graph set, the refusals of wrong input with one error line, and the
traceback of an internal error."""
import argparse
import os

import pytest

from benchmarks.proofpfn import cli, evaluate
from benchmarks.proofpfn import manifests as mf
from tests.proofpfn_fixtures import clean_checkpoints

from .helpers import GATE, tiny_graph, write_manifest


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
    assert cli.main(['list', '--set', 'ultra54', '--view', 'trans']) == 0
    assert capsys.readouterr().out.split() == [e.name for e in mf.read_manifest('ultra54').select('trans')]
    assert cli.main(['list', '--set', str(GATE)]) == 0
    assert capsys.readouterr().out.split() == ['UMLS', 'NELLInductive_v1', 'grail-WN18RR_v1']


def test_check_and_run(tmp_path, capsys):
    folders = {'g': tiny_graph(tmp_path / 'KGs' / 'g'), 'h': tiny_graph(tmp_path / 'KGs' / 'h', seed=1)}
    manifest = write_manifest(tmp_path / 'tiny.json', tmp_path, folders)
    common = ['--set', str(manifest), '--data-root', str(tmp_path), '--model', f"ts={clean_checkpoints(tmp_path, 'PCUltra-S')}"]
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    assert cli.main(['check', *common]) == 0 and 'check passed (2 graphs, systems ts.seed42,ts.seed43,ts.seed44)' in capsys.readouterr().out
    assert cli.main(['run', *common, '--device', 'cpu', '--out', str(tmp_path / 'out')]) == 0
    assert capsys.readouterr().out.splitlines() == [f'{graph} ts.seed42:done ts.seed43:done ts.seed44:done' for graph in 'gh']
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == workspace
    assert cli.main(['run', *common, '--seeds', '42', '--device', 'cpu', '--out', str(tmp_path / 'out')]) == 0
    assert capsys.readouterr().out.splitlines() == ['g ts.seed42:exists', 'h ts.seed42:exists']
    result = evaluate.read_result(tmp_path / 'out' / 'tiny' / 'ts.seed44' / 'h.json')
    assert result['label'] == 'ts.seed44' and 0 < result['metrics']['both']['MRR'] <= 1


def test_check_reports_every_failing_graph(tmp_path, capsys):
    folders = {name: tiny_graph(tmp_path / 'KGs' / name, seed=seed) for seed, name in enumerate('fgh')}
    manifest = write_manifest(tmp_path / 'tiny.json', tmp_path, folders)
    (folders['f'] / 'test.txt').write_text('x\tr0\ty\n')   # another file
    (folders['g'] / 'valid.txt').unlink()                   # a missing file
    common = ['--set', str(manifest), '--data-root', str(tmp_path), '--model', f"ts={clean_checkpoints(tmp_path, 'PCUltra-S')}"]
    assert cli.main(['check', *common]) == cli.EXIT_INPUT
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[-4:-1]] == ['f', 'g', 'h'] and lines[-2].endswith(' ok')
    assert 'FAILED' in lines[-4] and 'FAILED' in lines[-3] and lines[-1] == 'check FAILED: 2 of 3 graphs (f, g)'


def test_refusals(tmp_path, capsys):
    template, gate = clean_checkpoints(tmp_path, 'PCUltra-S'), str(GATE)
    for command in (['list', '--set', 'nosuch'], ['list', '--set', gate, '--graphs', 'FB15k-237'],
                    ['list', '--set', gate, '--view', 'nosuch', '--graphs', 'UMLS'],
                    ['list', '--set', 'ultra54', '--view', 'trans', '--graphs', 'FBIngram_25'],
                    ['check', '--set', gate], ['check', '--set', gate, '--references', 'flockoff'],
                    ['check', '--set', gate, '--model', f'ts={tmp_path}/{{seed}}-{{other}}.pt'], ['check', '--set', gate, '--model', 'ts={seed'],
                    ['check', '--set', gate, '--model', f'ts={template}', '--model', f'ts={template}'],
                    ['check', '--set', gate, '--model', f'ts={template}', '--seeds', '42,x'],
                    ['check', '--set', gate, '--model', f'ts={template}', '--seeds', '42,42'],
                    ['check', '--set', gate, '--model', f'ts={tmp_path}/none-{{seed}}.pt'],
                    ['run', '--set', gate, '--model', f'ts={template}', '--device', 'gpu', '--out', str(tmp_path / 'rows')],
                    ['run', '--set', gate, '--model', f'ts={template}', '--device', 'cuda:99', '--out', str(tmp_path / 'rows')],
                    ['prior', '--out', str(tmp_path / 'prior.json')],
                    ['prior', '--model', f'ts={template}', '--per-family', '101', '--out', str(tmp_path / 'prior.json')],
                    ['prior', '--model', f'ts={template}', '--families', 'structured', '--out', str(tmp_path / 'prior.json')],
                    ['prior', '--model', f'ts={template}', '--device', 'cuda:99', '--out', str(tmp_path / 'prior.json')]):
        assert cli.main(command) == cli.EXIT_INPUT, command
        error = capsys.readouterr().err
        assert error.startswith('error: ') and error.count('\n') == 1, (command, error)   # one line, no traceback


def test_internal_errors_keep_their_traceback(tmp_path, monkeypatch):
    def broken(*args: object, **kwargs: object) -> str:
        raise ValueError('an internal error')
    monkeypatch.setattr(evaluate, 'evaluate_graph', broken)
    with pytest.raises(ValueError, match='an internal error'):
        cli.main(['run', '--set', str(GATE), '--graphs', 'UMLS', '--model', f"ts={clean_checkpoints(tmp_path, 'PCUltra-S')}", '--device', 'cpu',
                  '--out', str(tmp_path / 'out')])
