"""Golden G6: the raw ranks of every model on the three gate graphs equal the rows of the research harness, bit for bit.

The runner ranks UMLS, NELLInductive_v1 and grail-WN18RR_v1 with S, CU, UL and HT (the G4 research exports at update
20 of seeds 42, 43, 44, converted to checkpoints) and the references TRIX, ULTRA-3g and official Flock, on the GPU of
the golden run in deterministic mode. The rank digest of every model and the row identity must equal the research rows
(their keys ``<key>.seed<S>:raw`` and ``<reference>:raw``). The test skips without CUDA, on another GPU, without
``CUBLAS_WORKSPACE_CONFIG`` or without the large goldens (``PROOFPFN_GOLDEN``).

The CPU test compares the ranks of S, CU and UL on UMLS with the research models' CPU ranks.
"""
from pathlib import Path

import pytest

from benchmarks.proofpfn import evaluate as ev
from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn import systems as sy
from dicee.synthetic_kg.arms import SEEDS
from dicee.synthetic_kg.checkpoint import from_research_export, save_checkpoint
from tests.proofpfn_fixtures import require_gpu_golden_environment

from .golden import G6_KEYS, GATE, fixture, g4_export, int_digest

GRAPHS = ('UMLS', 'NELLInductive_v1', 'grail-WN18RR_v1')
RESEARCH_REFERENCE = {'trix': 'trix', 'ultra3g': 'ultra3g', 'flock': 'flockoff'}   # label to the key of the research rows


def g6_systems(folder: Path, keys: tuple[str, ...] = tuple(G6_KEYS), references: tuple[str, ...] = tuple(RESEARCH_REFERENCE)) -> list:
    """The systems of G6: the G4 exports of the test keys, converted to checkpoints in ``folder``, then the references."""
    systems: list = []
    for key in keys:
        template, arm = g4_export(key), G6_KEYS[key][0]
        for seed in SEEDS:
            save_checkpoint(from_research_export(template.format(seed=seed), arm), folder / f'{key}-{seed}.pt')
        systems += sy.checkpoints(f"{key}={folder / (key + '-{seed}.pt')}")
    return systems + [sy.reference(label) for label in references]


def research_key(label: str) -> str:
    """The key of the research rows that holds the raw ranks of a label."""
    return f'{RESEARCH_REFERENCE.get(label, label)}:raw'


@pytest.mark.slow
@pytest.mark.parametrize('name', GRAPHS)
def test_raw_ranks_equal_the_research_rows(name, tmp_path):
    require_gpu_golden_environment()
    gate = mf.read_manifest(GATE)
    entry = gate.entry(name)
    if not (mf.REPO / entry.path).is_dir():
        pytest.skip(f'{entry.path} is not available')
    systems = g6_systems(tmp_path)
    assert set(ev.evaluate_graph(gate, entry, systems, tmp_path / 'out', device='cuda').values()) == {'done'}
    reference = fixture('g6_ranks.json.gz')['graphs'][name]
    for system in systems:
        result = ev.read_result(ev.result_path(tmp_path / 'out', gate, system.label, entry))
        assert len(result['rank']) == reference['rows']
        assert int_digest([[i, int(head)] for i, head in zip(result['index'], result['head'])]) == reference['rows_identity']
        assert int_digest(result['rank']) == reference['per_key'][research_key(system.label)], system.label
        assert (result['entities'], result['relations']) == (reference['header']['entities'], reference['header']['relations'])


@pytest.mark.slow
def test_ranks_on_the_cpu_equal_the_research_cpu_ranks(tmp_path):
    """The ranks of S, CU and UL on UMLS on the CPU against the ranks of the research models."""
    reference = fixture('umls_research_scores.json')['devices']['cpu']
    gate = mf.read_manifest(GATE)
    entry = gate.entry('UMLS')
    if not (mf.REPO / entry.path).is_dir():
        pytest.skip('KGs/UMLS is not available')
    systems = g6_systems(tmp_path, ('gs', 'gcu', 'gul'), references=())
    assert set(ev.evaluate_graph(gate, entry, systems, tmp_path / 'out', device='cpu').values()) == {'done'}
    for system in systems:
        result = ev.read_result(ev.result_path(tmp_path / 'out', gate, system.label, entry))
        assert int_digest([[i, int(head)] for i, head in zip(result['index'], result['head'])]) == reference['rows_identity']
        assert int_digest(result['rank']) == reference['models'][system.label]['raw_rank_sha256'], system.label
