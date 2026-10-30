"""The pretraining loop (``dicee.synthetic_kg.pretraining``) against goldens G4 (CPU), G10 (GPU) and G5 (exports).

G4 (``g4_s42.json.gz``, ``g4_s42rm.json.gz``, ``g4_s43.json.gz``, ``g4_s44.json.gz``): the first K updates of the research
loop on the CPU, bitwise for torch 2.5.1 with AVX-512 CPU kernels and 2 threads (elsewhere the G4 tests skip, and a smoke
test compares the losses of two updates within a relative 1e-5 and 1e-4) (``proofpfn-golden/tools/g4_train.py``, which
equals the genuine trainer on the GPU): per update the
learning rate, the logged loss and the gradient norm (``repr``) and the digests of the clean parameters and EMA; at the
end the clean AdamW state. Seed 42: S with K = 100, CU, UL, HT, N, R and M with K = 20; seeds 43 and 44: S, CU, UL, HT
and N with K = 20. G10 (``g10_gpu.json.gz``): the genuine research trainer on the local GPU (NVIDIA GeForce RTX 4070 Ti
SUPER), deterministic: S, seed 42, 20 updates; UL, CU and HT 2 updates. CPU and GPU values differ from the first
update on. G5: the exports against the research exports after conversion, tensor by tensor (large goldens,
``PROOFPFN_GOLDEN``). The default run checks the first updates of four arms; the full checks are slow.
"""
import fcntl
import json
import math
import os
from pathlib import Path

import pytest
import torch

from dicee.synthetic_kg import pretraining as pt
from dicee.synthetic_kg.arms import ARMS, initial_model
from dicee.synthetic_kg.checkpoint import from_research_export, load_model, read_checkpoint
from dicee.synthetic_kg.determinism import CUBLAS_HINT, device_problem
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.priors import real
from tests.proofpfn_fixtures import fixture, golden_large, require_cpu_training_golden_environment, require_gpu_golden_environment

G4 = {tag: fixture(f'g4_{tag}.json.gz') for tag in ('s42', 's42rm', 's43', 's44')}
G10 = fixture('g10_gpu.json.gz')
TABLE = '.relation.weight'
SLOT_KEYS = ('family', 'source', 'index', 'n_entities', 'n_context')


def g4_runs() -> list[tuple[str, int, int, str]]:
    """(arm, seed, K, fixture tag) of every G4 run but S's 100 updates of seed 42, which ``s_42_100`` trains."""
    return [(arm, golden['seed'], k, tag) for tag, golden in G4.items() for arm, k in golden['k'].items() if (arm, golden['seed'], k) != ('S', 42, 100)]


def train(folder: Path, arm: str, seed: int, k: int, device: str = 'cpu', workers: int = 0, **options) -> list[dict]:
    """Pretrain K updates with an export after each; returns the history of ``last.pt``."""
    config = pt.PretrainConfig(arm, seed, stop_after=k, exports=tuple(range(1, k + 1)), workers=workers, device=device, **options)
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


def adamw_by_name(folder: Path, arm: str, rows: int = 0) -> dict[str, torch.Tensor]:
    """The AdamW state of ``last.pt`` by parameter name (``<name>/step``, ``/exp_avg``, ``/exp_avg_sq``); ``rows`` zero rows
    are appended to the 2-D state of the relation tables (CountULTRA's pair rows in the research layout)."""
    names = [name for name, _ in initial_model(arm, 42).named_parameters()]
    out = {}
    for index, entry in state_of(folder)['optimizer']['state'].items():
        for key, value in entry.items():
            if rows and names[index].endswith(TABLE) and value.ndim == 2:
                value = torch.cat((value, value.new_zeros(rows, value.shape[1])))
            out[f'{names[index]}/{key}'] = value
    return out


def check_g4(folder: Path, history: list[dict], arm: str, k: int, tag: str) -> None:
    golden = G4[tag]['arms'][arm]
    for h, g in zip(history, golden['updates'][:k], strict=True):
        assert tuple(h[key] for key in SLOT_KEYS) == tuple(g[key] for key in SLOT_KEYS), h['update']
        assert (repr(h['lr']), repr(h['loss']), repr(h['grad_norm'])) == (g['lr'], g['loss'], g['grad_norm']), h['update']
        assert export_digest(folder, h['update'], 'raw') == g['params_clean'], h['update']
        assert export_digest(folder, h['update'], 'ema') == g['ema_clean'], h['update']
    if k == len(golden['updates']):
        assert tensor_digest(adamw_by_name(folder, arm)) == golden['final']['adamw_state_clean_sha256']
        groups = [{key: list(v) if isinstance(v, tuple) else v for key, v in group.items() if key != 'params'}
                  for group in state_of(folder)['optimizer']['param_groups']]
        assert groups == golden['final']['adamw_param_groups']   # AdamW's defaults: betas, eps, no amsgrad


@pytest.fixture(autouse=True)
def default_threads():
    previous = torch.get_num_threads()
    yield
    assert torch.get_num_threads() == previous   # pretrain restores the thread count


@pytest.mark.parametrize('arm', ['S', 'HT', 'CU', 'UL'])
def test_first_updates_equal_the_research_loop(arm, tmp_path):
    require_cpu_training_golden_environment()
    history = train(tmp_path, arm, 42, 2)
    check_g4(tmp_path, history, arm, 2, 's42')


def test_two_updates_of_s_are_close_to_the_research_loop(tmp_path):
    """Not a golden: in any environment, the logged losses of two updates of S are finite and close to G4.

    The loss of update 1 runs the forward pass only (relative 1e-5); one AdamW step amplifies the last bits of other CPU
    kernels, so update 2 gets a relative 1e-4 (with torch 2.5.1's AVX2 and default kernels about 3e-6 and 8e-6).
    """
    history = train(tmp_path, 'S', 42, 2)
    for h, g, tolerance in zip(history, G4['s42']['arms']['S']['updates'][:2], (1e-5, 1e-4), strict=True):
        assert math.isfinite(h['loss']) and math.isclose(h['loss'], float(g['loss']), rel_tol=tolerance, abs_tol=0.0), h['update']


@pytest.mark.slow
@pytest.mark.parametrize('arm, seed, k, tag', g4_runs(), ids=[f'{a}-{s}-{k}' for a, s, k, _ in g4_runs()])
def test_every_golden_run_equals_the_research_loop(arm, seed, k, tag, tmp_path):
    require_cpu_training_golden_environment()
    options = {}
    if ARMS[arm].prior.family('real') is not None:
        if not os.environ.get(real.SOURCES_ENV):
            pytest.skip(f'real worlds need their source graphs (set {real.SOURCES_ENV})')
        options['real_sources'] = os.environ[real.SOURCES_ENV]
    history = train(tmp_path, arm, seed, k, workers=2, **options)
    check_g4(tmp_path, history, arm, k, tag)


@pytest.fixture(scope='module')
def s_42_100(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    """S's 100 updates of seed 42 with an export after each, trained once for the G4 and the G5 test."""
    require_cpu_training_golden_environment()
    folder = tmp_path_factory.mktemp('S-42-100')
    return folder, train(folder, 'S', 42, 100, workers=2)


@pytest.mark.slow
def test_the_100_updates_of_s_equal_the_research_loop(s_42_100):
    folder, history = s_42_100
    check_g4(folder, history, 'S', 100, 's42')


@pytest.mark.slow
def test_the_exports_of_the_100_updates_equal_the_converted_research_exports(request):
    """G5: the exports at updates 20 and 100 equal the converted research exports, tensor by tensor."""
    research = {update: golden_large(f'g4/golden-runs/REC10A-XLE-PCU-noA/training/42/export-{update}.pt') for update in (20, 100)}
    folder, _ = request.getfixturevalue('s_42_100')   # after the golden lookup: without the goldens the test skips first
    for update, path in research.items():
        for weights in ('raw', 'ema'):
            expected = from_research_export(path, 'S', weights)['model']
            found = read_checkpoint(folder / f'export-{update}-{weights}.pt')['model']
            assert list(found) == list(expected) and all(torch.equal(found[key], expected[key]) for key in expected)


@pytest.mark.slow
@pytest.mark.parametrize('arm', ['S', 'UL', 'CU', 'HT'])
def test_gpu_training_equals_the_genuine_trainer(arm, tmp_path):
    require_gpu_golden_environment()
    golden = G10['arms'][arm]
    k = len(golden['updates'])
    history = train(tmp_path, arm, 42, k, device='cuda', workers=2)
    for h, g in zip(history, golden['updates'], strict=True):
        assert (h['family'], h['source'], h['index']) == (g['family'], g['source'], g['index']), h['update']
        assert (repr(h['lr']), repr(h['loss']), repr(h['grad_norm'])) == (g['lr'], g['loss'], g['grad_norm']), h['update']
    assert sorted(map(int, golden['exports'])) == (list(range(1, k + 1)) if arm == 'S' else [k])
    for update, export in golden['exports'].items():   # the genuine trainer's exports: S after every update, the others after 2
        assert export_digest(tmp_path, int(update), 'raw') == export['raw_clean'], update
        assert export_digest(tmp_path, int(update), 'ema') == export['ema_clean'], update
    pair_rows = 2 if arm == 'CU' else 0   # the research CountULTRA kept zero pair rows in its tables and their AdamW state
    assert tensor_digest(adamw_by_name(tmp_path, arm, pair_rows)) == golden['adamw_state_sha256']
    research = golden_large(f"g10/golden-gpu/{ARMS[arm].research_arm}/training/42/export-{k}.pt")
    for weights in ('raw', 'ema'):   # G5 on the GPU
        expected = from_research_export(research, arm, weights)['model']
        found = read_checkpoint(tmp_path / f'export-{k}-{weights}.pt')['model']
        assert all(torch.equal(found[key], expected[key]) for key in expected)


def test_the_run_does_not_depend_on_the_workers(tmp_path):
    alone = train(tmp_path / 'inline', 'S', 42, 2, workers=0)
    pooled = train(tmp_path / 'pool', 'S', 42, 2, workers=2, prefetch=1)
    assert [{k: v for k, v in h.items() if k != 'seconds'} for h in alone] == [{k: v for k, v in h.items() if k != 'seconds'} for h in pooled]
    for weights in ('raw', 'ema'):
        assert export_digest(tmp_path / 'inline', 2, weights) == export_digest(tmp_path / 'pool', 2, weights)


def test_resume_equals_the_uninterrupted_run(tmp_path):
    whole = train(tmp_path / 'whole', 'HT', 42, 3)
    config = pt.PretrainConfig('HT', 42, stop_after=1, exports=(1, 2, 3), device='cpu', workers=0)
    assert pt.pretrain(config, tmp_path / 'parts').status == 'stopped-after'
    calls = []

    def stop_now() -> bool:
        calls.append(1)
        return True

    stopped = pt.pretrain(pt.PretrainConfig('HT', 42, stop_after=3, exports=(1, 2, 3), device='cpu', workers=0), tmp_path / 'parts', stop=stop_now)
    assert stopped == pt.PretrainResult('stopped', 2, tmp_path / 'parts') and len(calls) == 1
    assert pt.pretrain(pt.PretrainConfig('HT', 42, stop_after=3, exports=(1, 2, 3), device='cpu', workers=0), tmp_path / 'parts').done == 3
    parts = state_of(tmp_path / 'parts')['history']
    assert [{k: v for k, v in h.items() if k != 'seconds'} for h in parts] == [{k: v for k, v in h.items() if k != 'seconds'} for h in whole]
    for weights in ('raw', 'ema'):
        assert export_digest(tmp_path / 'parts', 3, weights) == export_digest(tmp_path / 'whole', 3, weights)
    assert tensor_digest(adamw_by_name(tmp_path / 'parts', 'HT')) == tensor_digest(adamw_by_name(tmp_path / 'whole', 'HT'))


def test_a_run_that_reaches_its_last_update_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, 'TOTAL_UPDATES', 2)   # a schedule of two updates
    config = pt.PretrainConfig('UL', 42, exports=(1,), device='cpu', workers=0)
    assert pt.pretrain(config, tmp_path) == pt.PretrainResult('complete', 2, tmp_path)
    assert sorted(p.name for p in tmp_path.glob('export-*')) == ['export-1-ema.pt', 'export-1-raw.pt', 'export-2-ema.pt', 'export-2-raw.pt']
    assert state_of(tmp_path)['next'] == 2 and json.loads((tmp_path / 'run.json').read_text())['run']['order_slots'] == 2
    assert pt.pretrain(config, tmp_path) == pt.PretrainResult('complete', 2, tmp_path)   # a complete run does nothing


def test_run_record_and_folder_checks(tmp_path):
    config = pt.PretrainConfig('UL', 43, stop_after=1, exports=(1,), device='cpu', workers=0)
    result = pt.pretrain(config, tmp_path)
    assert result == pt.PretrainResult('stopped-after', 1, tmp_path)
    stored = json.loads((tmp_path / 'run.json').read_text())
    run, origin = stored['run'], stored['provenance']
    assert run['order_sha256'] == ARMS['UL'].pinned_order(43) and run['model_config'] == ARMS['UL'].model_config()
    assert (run['arm'], run['research_arm'], run['seed'], run['target'], run['device'], run['threads']) == ('UL', 'REC10A-XLE-UL-noA', 43, 'rbw', 'cpu', 2)
    assert {'synthetic_kg/pretraining.py', 'synthetic_kg/prepare.py', 'models/ultra.py', 'models/_inference.py'} <= set(origin['sources'])
    assert origin['real_sources'] == {} and origin['gpu'] is None
    assert sorted(p.name for p in tmp_path.glob('export-*')) == ['export-1-ema.pt', 'export-1-raw.pt']
    model = load_model(tmp_path / 'export-1-ema.pt')
    assert (model.relation_weighting, model.pair_channels) == ('binary', 'off')
    checkpoint = read_checkpoint(tmp_path / 'export-1-raw.pt')
    assert checkpoint['provenance']['source'] == 'pretraining' and checkpoint['provenance']['run'] == run
    assert pt.pretrain(config, tmp_path) == result   # a run at its stop does nothing
    with pytest.raises(pt.RunInputError, match='another run'):
        pt.pretrain(pt.PretrainConfig('UL', 44, stop_after=1, device='cpu', workers=0), tmp_path)
    allowed = pt.PretrainConfig('UL', 43, stop_after=1, device='cpu', workers=0, allow_code_change=True)
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
    assert pt.PretrainConfig('S', 42).export_updates == [4000, 8000, 16000, 20000, 24000]
    assert pt.PretrainConfig('S', 42, exports=(3,), stop_after=10).export_updates == [3, 24000]
    for bad in (dict(stop_after=0), dict(stop_after=24001), dict(exports=(0,)), dict(exports=(24001,)), dict(workers=-1), dict(save_every=0)):
        with pytest.raises(pt.RunInputError):
            pt.PretrainConfig('S', 42, **bad)
    with pytest.raises(KeyError):
        pt.PretrainConfig('S0', 42)
    monkeypatch.delenv(real.SOURCES_ENV, raising=False)
    with pytest.raises(pt.RunInputError, match=real.SOURCES_ENV):
        pt.provenance(pt.PretrainConfig('R', 42, device='cpu'), ARMS['R'])
    with pytest.raises(pt.RunInputError, match='missing'):
        pt.provenance(pt.PretrainConfig('R', 42, device='cpu', real_sources=tmp_path), ARMS['R'])


def test_devices_that_this_machine_lacks_are_refused(tmp_path, monkeypatch):
    for device, message in (('gpu', "device 'gpu'"), ('mps', 'expected cpu or cuda'), ('cuda:99', 'no such CUDA device')):
        with pytest.raises(pt.RunInputError, match=message):
            pt.pretrain(pt.PretrainConfig('S', 42, device=device), tmp_path / 'run')
    assert not (tmp_path / 'run').exists() and device_problem('cpu') is None
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)   # CUDA without a deterministic cuBLAS setting
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':0:0')
    assert device_problem('cuda') == CUBLAS_HINT and device_problem('cuda:0') == CUBLAS_HINT


def test_learning_rate_schedule():
    assert [pt.learning_rate(i) for i in (0, 19999, 20000)] == [5e-4, 5e-4, 5e-4]
    assert pt.learning_rate(22000) == 5e-4 * (1 - 2000 / 4000) and pt.learning_rate(23999) == 5e-4 * (1 - 3999 / 4000)
    assert repr(pt.learning_rate(0)) == G4['s42']['arms']['S']['updates'][0]['lr']


def test_hard_target_draws_one_answer_per_row():
    soft = torch.tensor([[0.0, 0.5, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    first = pt.hard_target(soft, torch.Generator().manual_seed(7))
    assert torch.equal(first, pt.hard_target(soft, torch.Generator().manual_seed(7)))
    assert first.dtype == soft.dtype and torch.equal(first.sum(-1), torch.ones(3)) and bool((first <= (soft > 0)).all())
    assert first[1].tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match='sum to 1'):
        pt.hard_target(soft * 0.5, torch.Generator().manual_seed(7))
