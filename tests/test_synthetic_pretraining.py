"""The pretraining loop (``dicee.synthetic_kg.pretraining``).

The pins (``tests/proofpfn_fixtures.py``) hold the first two CPU updates of five variants at seed 42: per update the
world, the learning rate, the logged loss and the gradient norm (``repr``) and the digests of the raw and EMA weights.
They are bitwise for torch 2.5.1 with AVX-512 CPU kernels and 2 threads; elsewhere these tests skip, and a smoke test
compares the losses of two updates within a relative 1e-5 and 1e-4.
"""
import fcntl
import json
import math
from pathlib import Path

import pytest
import torch

from dicee.models._inference import float32_precision_backends
from dicee.synthetic_kg import pretraining as pt
from dicee.synthetic_kg.checkpoint import load_model, read_checkpoint
from dicee.synthetic_kg.determinism import CUBLAS_HINT, deterministic_execution, device_problem
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.priors import real
from dicee.synthetic_kg.variants import VARIANTS, initial_model
from tests.proofpfn_fixtures import pins, require_training_pin_environment

UPDATES = pins()['training']['updates']
SLOT_KEYS = ('family', 'source', 'index', 'n_entities', 'n_context')


def train(folder: Path, variant: str, seed: int, k: int, device: str = 'cpu', workers: int = 0, **options) -> list[dict]:
    """Pretrain K updates with an export after each; returns the history of ``last.pt``."""
    config = pt.PretrainConfig(variant, seed, stop_after=k, exports=tuple(range(1, k + 1)), workers=workers, device=device, **options)
    result = pt.pretrain(config, folder)
    assert result == pt.PretrainResult('stopped-after', k, folder)
    history: list[dict] = state_of(folder)['history']
    assert [h['update'] for h in history] == list(range(1, k + 1))
    return history


def state_of(folder: Path) -> dict:
    state: dict = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
    return state


def export_digest(folder: Path, update: int, weights: str) -> str:
    checkpoint = read_checkpoint(folder / f'export-{update}-{weights}.pt')
    assert (checkpoint['updates'], checkpoint['weights']) == (update, weights)
    return tensor_digest(checkpoint['model'])


def adamw_by_name(folder: Path, variant: str) -> dict[str, torch.Tensor]:
    """The AdamW state of ``last.pt`` by parameter name (``<name>/step``, ``/exp_avg``, ``/exp_avg_sq``)."""
    names = [name for name, _ in initial_model(variant, 42).named_parameters()]
    return {f'{names[index]}/{key}': value for index, entry in state_of(folder)['optimizer']['state'].items() for key, value in entry.items()}


@pytest.fixture(autouse=True)
def default_threads():
    previous = torch.get_num_threads()
    yield
    assert torch.get_num_threads() == previous   # pretrain restores the thread count


@pytest.mark.parametrize('variant', sorted(UPDATES))
def test_first_updates_equal_their_pins(variant, tmp_path):
    require_training_pin_environment()
    history = train(tmp_path, variant, 42, 2)
    for h, pinned in zip(history, UPDATES[variant], strict=True):
        assert tuple(h[key] for key in SLOT_KEYS) == tuple(pinned[key] for key in SLOT_KEYS), h['update']
        assert (repr(h['lr']), repr(h['loss']), repr(h['grad_norm'])) == (pinned['lr'], pinned['loss'], pinned['grad_norm']), h['update']
        assert export_digest(tmp_path, h['update'], 'raw') == pinned['raw'], h['update']
        assert export_digest(tmp_path, h['update'], 'ema') == pinned['ema'], h['update']


def test_two_updates_are_close_to_their_pins(tmp_path):
    """In any environment, the logged losses of two updates of PCUltra-S are finite and close to the pinned losses.

    The loss of update 1 runs the forward pass only (relative 1e-5); one AdamW step amplifies the last bits of other CPU
    kernels, so update 2 gets a relative 1e-4.
    """
    history = train(tmp_path, 'PCUltra-S', 42, 2)
    for h, pinned, tolerance in zip(history, UPDATES['PCUltra-S'], (1e-5, 1e-4), strict=True):
        assert tuple(h[key] for key in SLOT_KEYS) == tuple(pinned[key] for key in SLOT_KEYS), h['update']
        assert math.isfinite(h['loss']) and math.isclose(h['loss'], float(pinned['loss']), rel_tol=tolerance, abs_tol=0.0), h['update']


def test_the_run_does_not_depend_on_the_workers(tmp_path):
    alone = train(tmp_path / 'inline', 'PCUltra-S', 42, 2, workers=0)
    pooled = train(tmp_path / 'pool', 'PCUltra-S', 42, 2, workers=2, prefetch=1)
    assert [{k: v for k, v in h.items() if k != 'seconds'} for h in alone] == [{k: v for k, v in h.items() if k != 'seconds'} for h in pooled]
    for weights in ('raw', 'ema'):
        assert export_digest(tmp_path / 'inline', 2, weights) == export_digest(tmp_path / 'pool', 2, weights)


def test_resume_equals_the_uninterrupted_run(tmp_path):
    whole = train(tmp_path / 'whole', 'PCUltra-HT', 42, 3)
    config = pt.PretrainConfig('PCUltra-HT', 42, stop_after=1, exports=(1, 2, 3), device='cpu', workers=0)
    assert pt.pretrain(config, tmp_path / 'parts').status == 'stopped-after'
    calls = []

    def stop_now() -> bool:
        calls.append(1)
        return True

    stopped = pt.pretrain(pt.PretrainConfig('PCUltra-HT', 42, stop_after=3, exports=(1, 2, 3), device='cpu', workers=0), tmp_path / 'parts', stop=stop_now)
    assert stopped == pt.PretrainResult('stopped', 2, tmp_path / 'parts') and len(calls) == 1
    assert pt.pretrain(pt.PretrainConfig('PCUltra-HT', 42, stop_after=3, exports=(1, 2, 3), device='cpu', workers=0), tmp_path / 'parts').done == 3
    parts = state_of(tmp_path / 'parts')['history']
    assert [{k: v for k, v in h.items() if k != 'seconds'} for h in parts] == [{k: v for k, v in h.items() if k != 'seconds'} for h in whole]
    for weights in ('raw', 'ema'):
        assert export_digest(tmp_path / 'parts', 3, weights) == export_digest(tmp_path / 'whole', 3, weights)
    assert tensor_digest(adamw_by_name(tmp_path / 'parts', 'PCUltra-HT')) == tensor_digest(adamw_by_name(tmp_path / 'whole', 'PCUltra-HT'))


def test_a_run_that_reaches_its_last_update_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'TOTAL_UPDATES', 2)   # a schedule of two updates
    config = pt.PretrainConfig('ULTRA-S', 42, exports=(1,), device='cpu', workers=0)
    assert pt.pretrain(config, tmp_path) == pt.PretrainResult('complete', 2, tmp_path)
    assert sorted(p.name for p in tmp_path.glob('export-*')) == ['export-1-ema.pt', 'export-1-raw.pt', 'export-2-ema.pt', 'export-2-raw.pt']
    assert state_of(tmp_path)['next'] == 2 and json.loads((tmp_path / 'run.json').read_text())['run']['order_slots'] == 2
    assert pt.pretrain(config, tmp_path) == pt.PretrainResult('complete', 2, tmp_path)   # a complete run does nothing


def test_run_record_and_folder_checks(tmp_path):
    config = pt.PretrainConfig('ULTRA-S', 43, stop_after=1, exports=(1,), device='cpu', workers=0)
    result = pt.pretrain(config, tmp_path)
    assert result == pt.PretrainResult('stopped-after', 1, tmp_path)
    stored = json.loads((tmp_path / 'run.json').read_text())
    run, origin = stored['run'], stored['provenance']
    assert run['order_sha256'] == VARIANTS['ULTRA-S'].pinned_order(43) and run['model_config'] == VARIANTS['ULTRA-S'].model_config()
    assert (run['variant'], run['seed'], run['target'], run['device'], run['threads']) == ('ULTRA-S', 43, 'rbw', 'cpu', 2)
    assert {'synthetic_kg/pretraining.py', 'synthetic_kg/prepare.py', 'models/ultra.py', 'models/_inference.py'} <= set(origin['sources'])
    assert origin['real_sources'] == {} and origin['gpu'] is None
    assert sorted(p.name for p in tmp_path.glob('export-*')) == ['export-1-ema.pt', 'export-1-raw.pt']
    model = load_model(tmp_path / 'export-1-ema.pt')
    assert (model.relation_weighting, model.pair_channels) == ('binary', 'off')
    checkpoint = read_checkpoint(tmp_path / 'export-1-raw.pt')
    assert checkpoint['provenance']['source'] == 'pretraining' and checkpoint['provenance']['run'] == run
    assert pt.pretrain(config, tmp_path) == result   # a run at its stop does nothing
    with pytest.raises(pt.RunInputError, match='another run'):
        pt.pretrain(pt.PretrainConfig('ULTRA-S', 44, stop_after=1, device='cpu', workers=0), tmp_path)
    allowed = pt.PretrainConfig('ULTRA-S', 43, stop_after=1, device='cpu', workers=0, allow_code_change=True)
    for key, value in (('sources', {**origin['sources'], 'synthetic_kg/prepare.py': '0' * 64}), ('torch', '0.0.0'), ('gpu', 'another GPU')):
        (tmp_path / 'run.json').write_text(json.dumps(dict(stored, provenance=dict(origin, **{key: value}))))
        with pytest.raises(pt.RunInputError, match=f"{'prepare.py' if key == 'sources' else key}.*changed since the run started"):
            pt.pretrain(config, tmp_path)
        assert pt.pretrain(allowed, tmp_path) == result
    with (tmp_path / '.lock').open('a') as handle:   # another process holds the folder
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(pt.RunInputError, match='another process'):
            pt.pretrain(allowed, tmp_path)
    assert pt.pretrain(allowed, tmp_path) == result


def test_configuration_checks(monkeypatch, tmp_path):
    assert pt.PretrainConfig('PCUltra-S', 42).export_updates == [4000, 8000, 16000, 20000, 24000]
    assert pt.PretrainConfig('PCUltra-S', 42, exports=(3,), stop_after=10).export_updates == [3, 24000]
    for bad in (dict(stop_after=0), dict(stop_after=24001), dict(exports=(0,)), dict(exports=(24001,)), dict(workers=-1), dict(save_every=0)):
        with pytest.raises(pt.RunInputError):
            pt.PretrainConfig('PCUltra-S', 42, **bad)
    with pytest.raises(KeyError):
        pt.PretrainConfig('S', 42)
    monkeypatch.delenv(real.SOURCES_ENV, raising=False)
    with pytest.raises(pt.RunInputError, match=real.SOURCES_ENV):
        pt.provenance(pt.PretrainConfig('PCUltra-R', 42, device='cpu'), VARIANTS['PCUltra-R'])
    with pytest.raises(pt.RunInputError, match='missing'):
        pt.provenance(pt.PretrainConfig('PCUltra-R', 42, device='cpu', real_sources=tmp_path), VARIANTS['PCUltra-R'])


def test_devices_that_this_machine_lacks_are_refused(tmp_path, monkeypatch):
    for device, message in (('gpu', "device 'gpu'"), ('mps', 'expected cpu or cuda'), ('cuda:99', 'no such CUDA device')):
        with pytest.raises(pt.RunInputError, match=message):
            pt.pretrain(pt.PretrainConfig('PCUltra-S', 42, device=device), tmp_path / 'run')
    assert not (tmp_path / 'run').exists() and device_problem('cpu') is None
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)   # CUDA without a deterministic cuBLAS setting
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':0:0')
    assert device_problem('cuda') == CUBLAS_HINT and device_problem('cuda:0') == CUBLAS_HINT


def test_deterministic_execution_sets_ieee_precision_and_restores_the_settings():
    """IEEE float32 precision inside the block, the previous settings after it. With per-backend settings the block reads
    no legacy flag: torch refuses them once cuDNN's convolution and RNN settings differ, as here."""
    backends = float32_precision_backends()
    if not backends:   # torch 2.5.1: the legacy flags only
        before = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
        with deterministic_execution():
            assert torch.are_deterministic_algorithms_enabled()
            assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32) == ('highest', False)
        assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32) == before
        return
    cudnn = torch.backends.cudnn
    conv = getattr(type(cudnn), 'conv', None)
    rnn = getattr(type(cudnn), 'rnn', None) or getattr(cudnn, 'rnn', None)   # the lookup of float32_precision_backends
    if not (hasattr(conv, 'fp32_precision') and hasattr(rnn, 'fp32_precision')):
        pytest.skip('this torch has no separate settings for the convolutions and RNNs of cuDNN')
    before = [backend.fp32_precision for backend in backends]
    try:
        conv.fp32_precision, rnn.fp32_precision = 'ieee', 'tf32'
        mixed = [backend.fp32_precision for backend in backends]
        with deterministic_execution():
            assert torch.are_deterministic_algorithms_enabled()
            assert all(backend.fp32_precision == 'ieee' for backend in backends)
        assert [backend.fp32_precision for backend in backends] == mixed
    finally:
        for backend, value in reversed(tuple(zip(backends, before, strict=True))):
            backend.fp32_precision = value


def test_learning_rate_schedule():
    assert [pt.learning_rate(i) for i in (0, 19999, 20000)] == [5e-4, 5e-4, 5e-4]
    assert pt.learning_rate(22000) == 5e-4 * (1 - 2000 / 4000) and pt.learning_rate(23999) == 5e-4 * (1 - 3999 / 4000)
    assert repr(pt.learning_rate(0)) == UPDATES['PCUltra-S'][0]['lr'] == '0.0005'


def test_hard_target_draws_one_answer_per_row():
    soft = torch.tensor([[0.0, 0.5, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    first = pt.hard_target(soft, torch.Generator().manual_seed(7))
    assert torch.equal(first, pt.hard_target(soft, torch.Generator().manual_seed(7)))
    assert first.dtype == soft.dtype and torch.equal(first.sum(-1), torch.ones(3)) and bool((first <= (soft > 0)).all())
    assert first[1].tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match='sum to 1'):
        pt.hard_target(soft * 0.5, torch.Generator().manual_seed(7))
