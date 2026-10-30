"""The command ``dicee_pretrain_synthetic`` (``dicee.scripts.pretrain_synthetic``): train, resume, stop and convert."""
import os
import re
import signal
from pathlib import Path

import pytest
import torch

from dicee.scripts import pretrain_synthetic as command
from dicee.synthetic_kg.checkpoint import from_research_export, read_checkpoint
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.pretraining import PretrainResult
from tests.proofpfn_fixtures import fixture, golden_large

SETUP = Path(__file__).resolve().parents[1] / 'setup.py'


def test_the_entry_point_is_declared():
    assert re.search(r'"dicee_pretrain_synthetic=dicee\.scripts\.pretrain_synthetic:main"', SETUP.read_text())


def test_train_and_resume(tmp_path, monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    options = ['train', '--arm', 'S', '--seed', '42', '--out', str(tmp_path), '--workers', '0', '--device', 'cpu', '--exports', '1,2']
    assert command.main([*options, '--stop-after', '1']) == 0
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ   # set for the run, then restored
    assert command.main([*options, '--stop-after', '2']) == 0
    history = torch.load(tmp_path / 'last.pt', map_location='cpu', weights_only=True)['history']
    golden = fixture('g4_s42.json.gz')['arms']['S']['updates']
    assert [(repr(h['loss']), repr(h['grad_norm'])) for h in history] == [(g['loss'], g['grad_norm']) for g in golden[:2]]
    assert tensor_digest(read_checkpoint(tmp_path / 'export-2-ema.pt')['model']) == golden[1]['ema_clean']


def test_signals_stop_the_run_after_the_current_update(tmp_path, monkeypatch):
    seen = {}

    def fake_pretrain(config, out, stop):
        seen['before'] = stop()
        os.kill(os.getpid(), signal.SIGTERM)
        seen['after'] = stop()
        return PretrainResult('stopped', 1, Path(out))

    handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(command, 'pretrain', fake_pretrain)
    assert command.main(['train', '--arm', 'UL', '--seed', '42', '--out', str(tmp_path), '--device', 'cpu']) == command.EXIT_STOPPED
    assert seen == dict(before=False, after=True)
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == handlers


def test_wrong_input(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*argv: str) -> str:
        assert command.main(list(argv)) == command.EXIT_INPUT
        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == 1 and lines[0].startswith('error: ')   # one line, no traceback
        return lines[0]

    assert 'stop_after' in refused('train', '--arm', 'S', '--seed', '42', '--out', str(tmp_path), '--stop-after', '30000')
    monkeypatch.delenv('PROOFPFN_REAL_SOURCES', raising=False)
    assert 'PROOFPFN_REAL_SOURCES' in refused('train', '--arm', 'R', '--seed', '42', '--out', str(tmp_path / 'R'), '--device', 'cpu')
    assert 'missing' in refused('train', '--arm', 'M', '--seed', '42', '--out', str(tmp_path / 'M'), '--device', 'cpu', '--real-sources', str(tmp_path))
    with pytest.raises(SystemExit):
        command.main(['train', '--arm', 'S0', '--seed', '42', '--out', str(tmp_path)])
    capsys.readouterr()
    torch.save({'model': {}}, tmp_path / 'other.pt')
    (tmp_path / 'text.pt').write_text('not an export')
    for export in ('other.pt', 'text.pt', 'missing.pt'):
        assert export in refused('convert', '--export', str(tmp_path / export), '--arm', 'S', '--out', str(tmp_path / 'x.pt'))
    assert not (tmp_path / 'x.pt').exists()
    command.main(['train', '--arm', 'UL', '--seed', '43', '--out', str(tmp_path / 'UL'), '--device', 'cpu', '--workers', '0', '--stop-after', '1',
                  '--exports', '1'])
    assert 'another run' in refused('train', '--arm', 'UL', '--seed', '44', '--out', str(tmp_path / 'UL'), '--device', 'cpu', '--workers', '0')


def test_convert_a_research_export(tmp_path):
    export = golden_large('g4/golden-runs/REC10A-XLE-UL-noA/training/43/export-20.pt')
    assert command.main(['convert', '--export', str(export), '--arm', 'UL', '--out', str(tmp_path / 'UL-43.pt'), '--weights', 'raw']) == 0
    converted = read_checkpoint(tmp_path / 'UL-43.pt')
    assert (converted['arm'], converted['seed'], converted['updates'], converted['weights']) == ('UL', 43, 20, 'raw')
    assert tensor_digest(converted['model']) == tensor_digest(from_research_export(export, 'UL', 'raw')['model'])
