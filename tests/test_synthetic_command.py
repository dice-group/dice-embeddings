"""The command ``dicee_pretrain_synthetic`` (``dicee.scripts.pretrain_synthetic``): train, resume, stop and sources."""
import math
import os
import re
import signal
from pathlib import Path

import pytest
import torch

from dicee.scripts import pretrain_synthetic as command
from dicee.synthetic_kg.checkpoint import read_checkpoint
from dicee.synthetic_kg.pretraining import PretrainResult
from tests.proofpfn_fixtures import pins

SETUP = Path(__file__).resolve().parents[1] / 'setup.py'


def test_the_entry_point_is_declared():
    assert re.search(r'"dicee_pretrain_synthetic=dicee\.scripts\.pretrain_synthetic:main"', SETUP.read_text())


def test_train_and_resume(tmp_path, monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    options = ['train', '--variant', 'PCUltra-S', '--seed', '42', '--out', str(tmp_path), '--workers', '0', '--device', 'cpu', '--exports', '1,2']
    assert command.main([*options, '--stop_after', '1']) == 0
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ   # set for the run, then restored
    assert command.main([*options, '--stop_after', '2']) == 0
    history = torch.load(tmp_path / 'last.pt', map_location='cpu', weights_only=True)['history']
    pinned = pins()['training']['updates']['PCUltra-S']
    assert [h['update'] for h in history] == [1, 2]
    for h, p, tolerance in zip(history, pinned, (1e-5, 1e-4), strict=True):   # the resumed run continues the first one
        assert (h['family'], h['index']) == (p['family'], p['index']) and math.isclose(h['loss'], float(p['loss']), rel_tol=tolerance)
    checkpoint = read_checkpoint(tmp_path / 'export-2-ema.pt')
    assert (checkpoint['variant'], checkpoint['seed'], checkpoint['updates'], checkpoint['weights']) == ('PCUltra-S', 42, 2, 'ema')


def test_signals_stop_the_run_after_the_current_update(tmp_path, monkeypatch):
    seen = {}

    def fake_pretrain(config, out, stop):
        seen['before'] = stop()
        os.kill(os.getpid(), signal.SIGTERM)
        seen['after'] = stop()
        return PretrainResult('stopped', 1, Path(out))

    handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(command, 'pretrain', fake_pretrain)
    assert command.main(['train', '--variant', 'ULTRA-S', '--seed', '42', '--out', str(tmp_path), '--device', 'cpu']) == command.EXIT_STOPPED
    assert seen == dict(before=False, after=True)
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == handlers


def test_wrong_input(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*argv: str) -> str:
        assert command.main(list(argv)) == command.EXIT_INPUT
        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == 1 and lines[0].startswith('error: ')   # one line, no traceback
        return lines[0]

    assert 'stop_after' in refused('train', '--variant', 'PCUltra-S', '--seed', '42', '--out', str(tmp_path), '--stop_after', '30000')
    monkeypatch.delenv('PROOFPFN_REAL_SOURCES', raising=False)
    assert 'PROOFPFN_REAL_SOURCES' in refused('train', '--variant', 'PCUltra-R', '--seed', '42', '--out', str(tmp_path / 'real'), '--device', 'cpu')
    assert 'missing' in refused('train', '--variant', 'PCUltra-M', '--seed', '42', '--out', str(tmp_path / 'mixed'), '--device', 'cpu', '--real_sources', str(tmp_path))
    with pytest.raises(SystemExit):
        command.main(['train', '--variant', 'S', '--seed', '42', '--out', str(tmp_path)])
    capsys.readouterr()
    with pytest.raises(SystemExit):   # the converter of research exports is not part of the command
        command.main(['convert', '--export', 'x.pt', '--variant', 'PCUltra-S', '--out', str(tmp_path / 'x.pt')])
    capsys.readouterr()
    command.main(['train', '--variant', 'ULTRA-S', '--seed', '43', '--out', str(tmp_path / 'run'), '--device', 'cpu', '--workers', '0', '--stop_after', '1',
                  '--exports', '1'])
    capsys.readouterr()   # warnings of the run, for example torch's notes on sparse tensors
    assert 'another run' in refused('train', '--variant', 'ULTRA-S', '--seed', '44', '--out', str(tmp_path / 'run'), '--device', 'cpu', '--workers', '0')


def test_sources_refuse_split_files_that_miss_or_differ(tmp_path, capsys):
    graph = tmp_path / 'KGs' / 'WN18RR'
    graph.mkdir(parents=True)
    (graph / 'train.txt').write_text('a r b\n')
    options = ['sources', '--graphs', str(tmp_path / 'KGs'), '--out', str(tmp_path / 'sources'), '--names', 'WN18RR']
    assert command.main(options) == command.EXIT_INPUT and 'has no valid.txt, test.txt' in capsys.readouterr().err
    for split in ('valid', 'test'):
        (graph / f'{split}.txt').write_text('b r a\n')
    assert command.main(options) == command.EXIT_INPUT and 'other triples than the research files' in capsys.readouterr().err
    assert not (tmp_path / 'sources').exists()
