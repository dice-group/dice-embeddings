"""Golden G6: every row key of the research harness on the three gate graphs, bit for bit.

The clean runner ranks UMLS, NELLInductive_v1 and grail-WN18RR_v1 with the families S, CU, UL and HT (the G4 research
exports at update 20 of seeds 42, 43, 44, converted on load, under the golden's test keys gs, gcu, gul, ght) and the
references TRIX, ULTRA-3g and official Flock, on the GPU of the golden run in deterministic mode (about 7 minutes). Every
key's int64 digest, the row identity, the key order and the metrics must equal the research rows. The test skips without
CUDA, on another GPU, without ``CUBLAS_WORKSPACE_CONFIG`` or without the large goldens (``PROOFPFN_GOLDEN``).

The seam test feeds the clean rows and the research rows (``large/g6/ranks``), under the registered keys, to the
registered ``fv1_tables.py``. The CPU test compares the per-seed raw ranks of S, CU and UL on UMLS with the research
models' CPU ranks (recorded from the research models).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.proofpfn import evaluate as ev
from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn import systems as sy
from dicee.synthetic_kg.digests import json_sha256
from tests.proofpfn_fixtures import GOLDEN_GPU, require_gpu_golden_environment

from .golden import G6_KEYS, G6_REGISTERED, fixture, g4_export, golden_large, int_digest

GRAPHS = ('UMLS', 'NELLInductive_v1', 'grail-WN18RR_v1')
REGISTERED = mf.REPO / 'benchmarks' / 'proofpfn' / 'registered'


def g6_systems(keys: tuple[str, ...] = tuple(G6_KEYS), references: tuple[str, ...] = ('trix', 'ultra3g', 'flockoff')) -> list:
    """The systems of G6: the G4 exports under the test keys (pinned to the research harness's model hashes), then the references."""
    recorded = fixture('g6_ranks.json.gz')['graphs']['UMLS']['record']['model_sha256']
    systems: list = []
    for key in keys:
        template, arm = g4_export(key), G6_KEYS[key][0]
        exports = mf.FamilyPins(arm=arm, key=key, export=20, weights='decay/ema', export_sha256={s: recorded[key][str(s)] for s in (42, 43, 44)},
                                loadable=True)
        systems.append(sy.ProofPFNFamily(key=key, arm=arm, checkpoints={s: Path(template.format(seed=s)) for s in (42, 43, 44)}, pins=exports))
    pins = mf.read_models()
    return systems + [sy.reference(key, pins) for key in references]


@pytest.fixture(scope='module')
def clean_rows(tmp_path_factory):
    """The clean row files of the three gate graphs on the GPU: graph name -> path."""
    require_gpu_golden_environment()
    gate = mf.read_manifest('gate')
    systems = g6_systems()
    out = tmp_path_factory.mktemp('g6')
    paths = {}
    for name in GRAPHS:
        entry = gate.entry(name)
        if not gate.folder(entry, mf.REPO).is_dir():
            pytest.skip(f'{entry.path} is not available')
        assert ev.evaluate_graph(gate, entry, systems, out, device='cuda') == 'done'
        paths[name] = ev.row_file_path(out, entry, [s.key for s in systems])
    return paths


@pytest.mark.slow
@pytest.mark.parametrize('name', GRAPHS)
def test_every_key_equals_the_research_rows(clean_rows, name):
    reference = fixture('g6_ranks.json.gz')['graphs'][name]
    doc = json.loads(clean_rows[name].read_text())
    rows = doc['rows']
    assert list(rows[0]) == reference['keys'] and len(rows) == reference['rows']
    assert int_digest([[r['index'], int(r['head'])] for r in rows]) == reference['rows_identity']
    differing = [k for k, digest in reference['per_key'].items() if int_digest([int(r[k]) for r in rows]) != digest]
    assert differing == [], f'{name}: {len(differing)} keys differ from the research rows: {differing[:8]}'
    assert {k: json_sha256(v) for k, v in doc['metrics'].items()} == reference['metrics']
    for position, row in reference['examples'].items():
        assert rows[int(position)] == row
    header = {k: doc[k] for k in ('dataset', 'families', 'entities', 'relations', 'reflexive')}
    assert header == {k: v for k, v in reference['header'].items() if k != 'protocol'}
    record = doc['allgraph']
    assert record['model_sha256'] == reference['record']['model_sha256'] and record['gpu'] == GOLDEN_GPU
    assert {k: v for k, v in record['flock_official'].items()} == {k: v for k, v in reference['record']['flock_official'].items() if k != 'code_sha256'}
    for key in ('key', 'path', 'collection', 'test_rows', 'ranked_facts', 'sampled', 'positions', 'sample_rule', 'seeds'):
        assert record[key] == reference['record'][key], key
    assert {k: v for k, v in record['sha256'].items() if k.endswith('.txt')} == \
        {k: v for k, v in reference['record']['sha256'].items() if k.endswith('.txt')}


def registered(doc: dict) -> dict:
    """A row file with the test keys of G6 renamed to the keys of the registered systems."""
    def rename(key: str) -> str:
        for old, new in G6_REGISTERED.items():
            if key == old or key.startswith((f'{old}.', f'{old}:')):
                return new + key[len(old):]
        return key
    allgraph = dict(doc['allgraph'], model_sha256={rename(k): v for k, v in doc['allgraph']['model_sha256'].items()})
    return dict(doc, rows=[{rename(k): v for k, v in r.items()} for r in doc['rows']], allgraph=allgraph)


def tables(rows_dir: Path, out: Path, gate: mf.Manifest) -> bytes:
    """``mrr.csv`` of the registered ``fv1_tables.py`` on a folder of row files (the gate graphs as a block-3 list)."""
    lists = out.with_name(out.name + '-lists')
    lists.mkdir()
    (lists / 'manifest.json').write_text(json.dumps(dict(graphs={})))
    block3 = {e.name: dict(cluster='gate', files={f'{s}.txt': e.sha256[s] for s in mf.SPLITS}) for e in gate.entries}
    (lists / 'block3.json').write_text(json.dumps(dict(graphs=block3)))
    subprocess.run([sys.executable, '-B', str(REGISTERED / 'fv1_tables.py'), '--rows', str(rows_dir), '--manifest', str(lists / 'manifest.json'),
                    '--block3', str(lists / 'block3.json'), '--out-dir', str(out)], check=True, capture_output=True)
    return (out / 'mrr.csv').read_bytes()


@pytest.mark.slow
def test_registered_tables_read_clean_and_research_rows_alike(clean_rows, tmp_path):
    gate = mf.read_manifest('gate')
    for kind in ('clean', 'research'):
        for name, path in clean_rows.items():
            source = path if kind == 'clean' else golden_large(f'g6/ranks/{path.relative_to(path.parents[2])}')
            target = tmp_path / kind / 'rows' / f'{name}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(registered(json.loads(source.read_text()))))
    clean, research = (tables(tmp_path / kind / 'rows', tmp_path / kind / 'tables', gate) for kind in ('clean', 'research'))
    assert clean == research and clean.count(b'\n') == 1 + 3 * (4 * (3 + 3 + 1) + 3 * 2)   # per graph: 4 families, 3 references


@pytest.mark.slow
def test_seed_ranks_on_the_cpu_equal_the_research_cpu_ranks(tmp_path):
    """Per-seed raw ranks of S, CU and UL on UMLS on the CPU against the ranks of the research models."""
    reference = fixture('umls_research_scores.json')['devices']['cpu']
    gate = mf.read_manifest('gate')
    entry = gate.entry('UMLS')
    if not gate.folder(entry, mf.REPO).is_dir():
        pytest.skip('KGs/UMLS is not available')
    systems = g6_systems(('gs', 'gcu', 'gul'), references=())
    assert ev.evaluate_graph(gate, entry, systems, tmp_path, device='cpu') == 'done'
    rows = json.loads(ev.row_file_path(tmp_path, entry, [s.key for s in systems]).read_text())['rows']
    assert int_digest([[r['index'], int(r['head'])] for r in rows]) == reference['rows_identity']
    for key in ('gs', 'gcu', 'gul'):
        for seed in (42, 43, 44):
            assert int_digest([r[f'{key}.seed{seed}:raw'] for r in rows]) == reference['models'][f'{key}.seed{seed}']['raw_rank_sha256'], (key, seed)
