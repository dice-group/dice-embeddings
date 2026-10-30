"""The MERIT adapter: the model's per-query noise seeds, the systems and their row file through the benchmark's runner, the
compiler policy, the setup checks and the child process; with MERIT's environment on an H100 NVL, the check ranks.

MERIT's own code imports only in MERIT's environment, so most tests give the model a small stand-in relation model and
motif tensor of the same interfaces (``FAKE``).
"""
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.proofpfn import cli
from benchmarks.proofpfn import evaluate as ev
from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn import systems as sy
from benchmarks.proofpfn.adapters import AdapterSetupError, merit

from .golden import GOLDEN, int_digest, tiny_graph, write_manifest

REFERENCE_GPU = 'NVIDIA H100 NVL'   # the GPU of kiowl, where merit-ref-v1/check ran (torch 2.5.1+cu124)


class FakeRelationModel(nn.Module):
    """A stand-in of ``RelTransformerMERIT``: token states from a table, the motif row sums and Gaussian noise (RNI)."""

    def __init__(self, input_dim: int, hidden_dims: list, attn_impl: str = 'mha', **_: object) -> None:
        super().__init__()
        self.table = nn.Embedding(64, input_dim)
        self.mix = nn.Linear(input_dim, input_dim)
        self.attn_impl = attn_impl
        self._prof_cache = None

    def forward(self, struct, query: torch.Tensor, motif: torch.Tensor) -> torch.Tensor:
        n = struct.num_nodes
        tokens = self.table.weight[:n] + motif.sum((1, 2))[:, None]
        x = tokens[None].expand(len(query), n, -1).clone()
        x[torch.arange(len(query)), query] += 1.0
        states: torch.Tensor = self.mix(x + torch.randn(len(query), n, x.shape[-1], device=x.device))
        return states


def fake_motif(graph) -> torch.Tensor:
    """A stand-in of ``build_motif_tensor``: ``log1p`` of the relation co-occurrence counts at shared entities."""
    r = graph.num_relations
    incidence = torch.zeros(graph.num_nodes, r)
    incidence.index_put_((graph.edge_index[0], graph.edge_type), torch.ones(len(graph.edge_type)), accumulate=True)
    return torch.log1p(incidence.T @ incidence)[..., None]


FAKE = merit.Upstream(relation_model=FakeRelationModel, motif_tensor=fake_motif, files={'fake': '0' * 64})


def fake_checkpoints(folder: Path) -> dict[str, Path]:
    """One checkpoint per MERIT system: the state of a stand-in MERIT under the system's seed."""
    files = {}
    for key, seed in zip(merit.KEYS, merit.SEEDS):
        torch.manual_seed(seed)
        model = merit.MERIT(dict(num_entities=4, num_relations=2), FAKE)
        files[key] = folder / f'merit_s{seed}.pth'
        torch.save({'model': model.state_dict()}, files[key])
    return files


def fake_systems(folder: Path) -> list[merit.MeritReference]:
    files = fake_checkpoints(folder)
    return [merit.MeritReference(key=k, path=files[k], sha256=mf.file_sha256(files[k]), upstream=FAKE) for k in merit.KEYS]


def tiny(tmp_path: Path):
    folder = tiny_graph(tmp_path / 'KGs' / 'g')
    manifest = mf.read_manifest(write_manifest(tmp_path / 'tiny.json', tmp_path, {'g': folder}))
    return manifest, manifest.entry('g')


def test_noise_seed():
    assert merit.merit_seed(0, 5, 3) == 5 * 4_099 + 3 and merit.merit_seed(2, 0, 0) == 2_000_006
    assert merit.merit_seed(1, 2**62, 0) < 2**63 - 1


def test_scores_do_not_depend_on_the_query_order_and_leave_the_random_state(tmp_path):
    system = fake_systems(tmp_path)[0]
    model = system.build_model(16, 3)
    graph = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 2, 3], [3, 0, 0], [4, 1, 5], [5, 2, 6], [6, 0, 7], [7, 1, 4]])
    model.set_graph(graph, 16, 3)
    queries = torch.tensor([[0, 0], [5, 2], [3, 1], [0, 0]])
    state = torch.get_rng_state()
    with torch.no_grad():
        together = model.forward_k_vs_all(queries)
        alone = torch.cat([model.forward_k_vs_all(q[None]) for q in queries.flip(0)]).flip(0)
        heads = model.forward_k_vs_all_heads(torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(together, alone) and torch.equal(together[0], together[3]) and not torch.equal(together[0], together[1])
    assert heads.shape == (2, 16) and torch.isfinite(heads).all()
    assert model.motif.shape == (6, 6, 1) and model.relation_model.attn_impl == 'mha'   # the CPU path


def test_systems_write_the_reference_rows(tmp_path):
    manifest, entry = tiny(tmp_path)
    systems = fake_systems(tmp_path)
    assert ev.evaluate_graph(manifest, entry, systems, tmp_path / 'rows', root=tmp_path, device='cpu') == 'done'
    path = ev.row_file_path(tmp_path / 'rows', entry, merit.KEYS)
    assert path.parent.name == 'merit1024-merit2025-merit2026-merit4000-merit8000'
    doc = json.loads(path.read_text())
    keys = [k for s in merit.KEYS for k in (s, f'{s}:gt2', f'{s}:raw', f'{s}:raw:gt2')]
    assert all(list(r) == ['index', 'head', *keys] for r in doc['rows'])
    assert doc['allgraph']['model_sha256'] == {s.key: {'model': s.sha256} for s in systems}
    assert doc['allgraph']['families']['merit1024']['model'] == 'MERIT' and doc['allgraph']['families']['merit1024']['noise_seed'] == 0
    with pytest.raises(sy.SystemCheckError, match="unknown reference 'merit7'"):
        merit.MeritReference(key='merit7', path=systems[0].path, sha256=systems[0].sha256)
    with pytest.raises(AdapterSetupError, match='not loaded'):
        merit.MeritReference(key='merit1024', path=systems[0].path, sha256=systems[0].sha256).build_model(4, 2)


def test_fresh_compiler_policy():
    from torch._dynamo import config
    with merit.fresh_compiler():
        inside = config.error_on_recompile
    assert inside and not config.error_on_recompile
    with config.patch(cache_size_limit=0):
        with pytest.raises(AdapterSetupError, match='compiler cache'):
            with merit.fresh_compiler():
                pass


def fake_home(tmp_path: Path, commit: str) -> Path:
    home = tmp_path / 'MERIT'
    (home / '.git').mkdir(parents=True)
    (home / '.git' / 'HEAD').write_text(commit + '\n')
    for name in merit.UPSTREAM:
        (home / name).parent.mkdir(parents=True, exist_ok=True)
        (home / name).write_text('# not MERIT\n')
    return home


def test_setup_refusals(tmp_path):
    pins = merit.external_pins('merit')
    with pytest.raises(AdapterSetupError, match='commit'):
        merit.setup(fake_home(tmp_path / 'a', '0' * 40))
    with pytest.raises(AdapterSetupError, match='SHA-256'):
        merit.setup(fake_home(tmp_path / 'b', pins['code_commit']))


def test_child_process_command(tmp_path):
    home = fake_home(tmp_path, '0' * 40)
    setup = merit.MeritSetup(home=home, pins={}, checkpoints={}, sha256={})
    argv = ['adapter', 'merit', '--set', 'gate', '--out', 'rows']
    command, env = merit.child_command(argv, setup, python='/opt/merit/bin/python -X utf8')
    assert command == ['/opt/merit/bin/python', '-X', 'utf8', '-B', '-m', 'benchmarks.proofpfn', *argv, '--worker']
    assert env['PYTHONPATH'].split(os.pathsep) == [str(mf.REPO), str(home)] and env['MERIT_HOME'] == str(home)
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'
    try:
        import kgfm  # type: ignore[import-not-found]  # noqa: F401
        pytest.skip("MERIT's package imports here")
    except ImportError:
        with pytest.raises(AdapterSetupError, match="MERIT's environment"):
            merit.import_upstream(home, {})


def test_checkpoint_pins_are_the_registered_references():
    references = mf.read_models().references
    assert all(references[k].path is None for k in merit.KEYS)
    record = json.loads((GOLDEN / 'merit_check_ranks.json').read_text())
    assert record['checkpoint_sha256'] == {k: references[k].sha256 for k in merit.KEYS}
    assert record['upstream_sha256'] == {k: merit.external_pins('merit')[k] for k in merit.UPSTREAM}


# ------------------------------------------------------------------------------------------------ check graphs

@pytest.mark.slow
def test_check_graph_ranks_equal_the_research_ranks(tmp_path):
    """The raw ranks of the five systems on the graphs of merit-ref-v1/check (kiowl, torch 2.5.1+cu124).

    Needs MERIT's environment (``$MERIT_PYTHON``) and the MERIT folder (``$MERIT_HOME``) on the GPU type of the research run.
    """
    if not (os.environ.get(merit.VARIABLE) and os.environ.get(merit.PYTHON_VARIABLE)) or not torch.cuda.is_available():
        pytest.skip("set $MERIT_HOME and $MERIT_PYTHON (MERIT's environment) on a CUDA machine")
    if torch.cuda.get_device_name(0) != REFERENCE_GPU:
        pytest.skip(f'the MERIT check ranks hold for {REFERENCE_GPU}')
    record = json.loads((GOLDEN / 'merit_check_ranks.json').read_text())
    for name, info in record['graphs'].items():
        manifest = mf.read_manifest(info['manifest'])
        entry = manifest.entry(info['key'])
        assert cli.main(['adapter', 'merit', '--set', info['manifest'], '--graphs', info['key'], '--out', str(tmp_path / 'rows')]) == 0
        rows = json.loads((tmp_path / 'rows' / entry.collection / '-'.join(merit.KEYS) / f'{entry.name}.json').read_text())['rows']
        assert len(rows) == 2 * info['test_rows']
        for key in merit.KEYS:
            ranks = {(r['index'], r['head']): r[f'{key}:raw'] for r in rows}
            got = {side: int_digest([ranks[i, side == 'head'] for i in range(info['test_rows'])]) for side in ('tail', 'head')}
            assert got == record['digests'][name][key], (name, key)
