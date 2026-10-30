"""The evaluation runner on the CPU: ranks and views, the row file, sharding and resume, and the seam with fv1_tables."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.proofpfn import evaluate as ev
from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn import systems as sy
from dicee.evaluation._filtering import FilteredRanker

from .golden import tiny_graph, write_graph, write_manifest
from .test_systems import clean_checkpoints

REGISTERED = mf.REPO / 'benchmarks' / 'proofpfn' / 'registered'


def research_rank(v: torch.Tensor, target: int, mask: torch.Tensor) -> int:
    """``lockbox-v5-runtime/lb5_eval.rank``: the research pessimistic rank."""
    return int(((v >= v[target]) & mask).sum())


def test_pessimistic_ranker_equals_the_research_rank():
    rng = np.random.default_rng(0)
    ranker = FilteredRanker(tie_policy='pessimistic')
    for _ in range(400):
        n = int(rng.integers(2, 40))
        v = torch.tensor(np.round(rng.normal(size=n), 1))        # rounding makes ties
        anchor, target = (int(x) for x in rng.integers(0, n, 2))
        if rng.random() < 0.5:
            v[anchor] = -float('inf')                            # the rule view, also when the target is the anchor
        others = sorted(set(rng.integers(0, n, int(rng.integers(0, n))).tolist()) - {target})
        mask = torch.ones(n, dtype=torch.bool)
        mask[torch.tensor(others, dtype=torch.long)] = False
        assert ranker.rank(v, target, others, exclude_target=False) == research_rank(v, target, mask)


def test_gt2_rank():
    v = torch.log(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64))
    allowed, unseen = np.ones(3, bool), np.array([False, False, True])
    # p = (0.5, 0.3, 0.2); the unseen mass 0.2 moves half-way towards g = 0.5: q = (0.395, 0.237, 0.316)
    assert [ev.gt2_rank(v, allowed, t, unseen, 0.5) for t in range(3)] == [1, 3, 2]
    allowed[0] = False                                           # a filtered candidate leaves the distribution
    assert [ev.gt2_rank(v, allowed, t, unseen, 0.5) for t in (1, 2)] == [1, 2]   # p = (0, 0.6, 0.4)


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """A tiny graph set, the clean checkpoints of two arms, the systems and one run of all of them.

    The S family has the registered key of S (built directly, without pins), so that the registered tables read its rows.
    TRIX and ULTRA-3g join when their checkpoints are available. Official Flock takes 1.5 s per query group on the CPU,
    so it has its own test on a smaller graph.
    """
    root = tmp_path_factory.mktemp('tiny')
    manifest = mf.read_manifest(write_manifest(root / 'tiny.json', root, {'g': tiny_graph(root / 'KGs' / 'g')}))
    pins = mf.read_models()
    template = clean_checkpoints(root, 'S')
    systems: list = [sy.ProofPFNFamily(key='rec10axlepcunoa', arm='S', checkpoints={s: Path(template.format(seed=s)) for s in (42, 43, 44)}),
                     sy.family('tu', 'UL', clean_checkpoints(root, 'UL'), pins)]
    for key in ('trix', 'ultra3g'):
        reference = sy.reference(key, pins)
        if reference.path.is_file():
            systems.append(reference)
    status, whole = run(root, manifest, systems, root / 'whole')
    assert status == 'done'
    return root, manifest, systems, whole


def run(root, manifest, systems, out, **options):
    status = ev.evaluate_graph(manifest, manifest.entry('g'), systems, out, root=root, device='cpu', **options)
    return status, ev.row_file_path(out, manifest.entry('g'), systems)


def test_row_file(tiny):
    root, manifest, systems, path = tiny
    assert run(root, manifest, systems, root / 'whole')[0] == 'exists'
    doc = json.loads(path.read_text())
    assert list(doc) == ['dataset', 'families', 'entities', 'relations', 'reflexive', 'protocol', 'metrics', 'rows', 'seconds', 'allgraph',
                         'metrics_views', 'provenance']
    rows, keys = doc['rows'], ev.row_keys(systems)
    facts = len((root / 'KGs' / 'g' / 'test.txt').read_text().splitlines())
    assert len(rows) == 2 * facts and all(list(r) == ['index', 'head', *keys] for r in rows)
    assert keys[:7] == ['rec10axlepcunoa', 'rec10axlepcunoa:gt2', 'rec10axlepcunoa:raw', 'rec10axlepcunoa:raw:gt2',
                        'rec10axlepcunoa.seed42:raw', 'rec10axlepcunoa.seed42', 'rec10axlepcunoa.seed42:gt2']
    assert doc['reflexive'] == [0] and doc['metrics'] == {k: float(np.mean([1 / r[k] for r in rows])) for k in keys}
    seed_raw = [k for k in keys if '.seed' in k and k.endswith(':raw')]
    for r in rows:   # the rule view moves a seed's rank by at most one, except where the target is the anchor
        assert all(r[k] - r[k.removesuffix(':raw')] in (0, 1) or r[k.removesuffix(':raw')] > r[k] for k in seed_raw)
    planted = [r for r in rows if r['index'] == facts - 1]   # the fact whose answer is its anchor, in a closed relation
    rule_keys = [k.removesuffix(':raw') for k in seed_raw] + [s.key for s in systems]
    assert len(planted) == 2 and all(len({r[k] for k in rule_keys}) == 1 for r in planted)   # the same last rank in every view
    record = doc['allgraph']
    assert record['key'] == 'tiny/g' and record['ranked_facts'] == facts and record['sampled'] is False
    assert record['sha256']['train.txt'] == mf.file_sha256(root / 'KGs' / 'g' / 'train.txt')
    assert record['model_sha256']['rec10axlepcunoa'] == {str(s): mf.file_sha256(root / f'S-{s}.pt') for s in (42, 43, 44)}
    assert 'flock_official' not in record and doc['provenance']['deterministic'] and doc['provenance']['threads'] == 2
    assert doc['provenance']['code_sha256'] == ev.code_sha256() and 'dicee/models/_inference.py' in ev.code_sha256()


def test_shards_and_resumed_runs_equal_one_run(tiny):
    root, manifest, systems, whole = tiny
    rows = json.loads(whole.read_text())['rows']
    statuses = [run(root, manifest, systems, root / 'sharded', shard=(i, 3))[0] for i in (2, 3, 1)]
    assert statuses == ['shard-done', 'shard-done', 'done']
    sharded = json.loads(ev.row_file_path(root / 'sharded', manifest.entry('g'), systems).read_text())
    assert sharded['rows'] == rows and sharded['metrics'] == json.loads(whole.read_text())['metrics'] and sharded['provenance']['shards'] == 3
    status, stopped = run(root, manifest, systems, root / 'resumed', parts=True, max_minutes=1e-9)   # stops after the first group
    assert status == 'stopped' and not stopped.exists()
    part = json.loads(ev.part_path(stopped, (1, 1)).read_text())
    assert part['done'] == 1 and not part['complete']
    assert run(root, manifest, systems, root / 'resumed', parts=True)[0] == 'done'
    resumed = json.loads(stopped.read_text())
    assert resumed['rows'] == rows and resumed['provenance']['runs'] == [2]
    bad = ev.part_path(ev.row_file_path(root / 'resumed2', manifest.entry('g'), systems), (1, 1))
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps(dict(part, identity='0' * 64)))
    with pytest.raises(mf.InputError, match='another run'):   # a part of other systems or graph files is refused
        run(root, manifest, systems, root / 'resumed2', parts=True)
    assert run(root, manifest, systems, root / 'stale', shard=(1, 2))[0] == 'shard-done'
    other = ev.part_path(ev.row_file_path(root / 'stale', manifest.entry('g'), systems), (2, 2))
    other.write_text(json.dumps(dict(complete=True, identity='0' * 64)))   # the stale part of another run blocks the merge
    with pytest.raises(mf.InputError, match='shard-2-of-2.json is a part of another run'):
        run(root, manifest, systems, root / 'stale', shard=(1, 2))


def test_official_flock_rows_do_not_depend_on_the_sharding(tmp_path):
    """Official Flock draws the walks of each query from a generator seeded per call: two shards give the rows of one run."""
    flock = sy.reference('flockoff', mf.read_models())
    if not flock.path.is_file():
        pytest.skip(f'{flock.path} is not available')
    folder = write_graph(tmp_path / 'KGs' / 'g', ['a\tr0\tb', 'b\tr0\tc', 'c\tr1\ta', 'a\tr1\tc', 'b\tr1\td', 'd\tr0\ta'], [],
                         ['a\tr0\tc', 'd\tr1\tb'])
    manifest = mf.read_manifest(write_manifest(tmp_path / 'm.json', tmp_path, {'g': folder}))
    status, whole = run(tmp_path, manifest, [flock], tmp_path / 'whole')
    assert status == 'done' and [run(tmp_path, manifest, [flock], tmp_path / 'sharded', shard=(i, 2))[0] for i in (2, 1)] == ['shard-done', 'done']
    doc = json.loads(whole.read_text())
    assert json.loads(ev.row_file_path(tmp_path / 'sharded', manifest.entry('g'), [flock]).read_text())['rows'] == doc['rows']
    assert doc['allgraph']['flock_official']['keys'] == {'flockoff': dict(walks=16, passes=16)}


def test_global_settings_are_restored(tiny):
    root, manifest, systems, _ = tiny
    threads, deterministic, rng = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), torch.get_rng_state()
    run(root, manifest, systems[:1], root / 'settings')
    assert (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()) == (threads, deterministic)
    assert torch.equal(torch.get_rng_state(), rng)


def test_registered_tables_read_the_rows(tiny, tmp_path):
    """The seam: the frozen fv1_tables.py reads a clean row file (keys of the registered systems, a block-3 list)."""
    root, manifest, systems, path = tiny
    entry = manifest.entry('g')
    doc = json.loads(path.read_text())
    lists = tmp_path / 'lists'
    lists.mkdir()
    (lists / 'manifest.json').write_text(json.dumps(dict(graphs={})))
    (lists / 'block3.json').write_text(json.dumps(dict(graphs={'g': dict(cluster='tiny', files={f'{s}.txt': entry.sha256[s] for s in mf.SPLITS})})))
    (lists / 'models.json').write_text(json.dumps(dict(models=doc['allgraph']['model_sha256'])))
    subprocess.run([sys.executable, '-B', str(REGISTERED / 'fv1_tables.py'), '--rows', str(root / 'whole'), '--manifest',
                    str(lists / 'manifest.json'), '--block3', str(lists / 'block3.json'), '--models', str(lists / 'models.json'),
                    '--out-dir', str(tmp_path / 'tables')], check=True, capture_output=True)
    with (tmp_path / 'tables' / 'mrr.csv').open() as handle:
        table = {(r['system'], int(r['seed'])): r['mrr'] for r in csv.DictReader(handle)}
    rows = doc['rows']

    def mrr(key: str) -> str:
        return repr(math.fsum(1.0 / r[key] for r in rows) / len(rows))
    for seed in (42, 43, 44):
        assert table['rec10axlepcunoa', seed] == mrr(f'rec10axlepcunoa.seed{seed}:raw')
        assert table['rec10axlepcunoa+rule', seed] == mrr(f'rec10axlepcunoa.seed{seed}')
    assert table['rec10axlepcunoa+ens', 0] == mrr('rec10axlepcunoa:raw')
    for reference in [s for s in systems if not isinstance(s, sy.ProofPFNFamily)]:
        assert table[reference.key, 0] == mrr(f'{reference.key}:raw') and table[f'{reference.key}+rule', 0] == mrr(reference.key)
    meta = json.loads((tmp_path / 'tables' / 'tables.meta.json').read_text())
    assert meta['models_checked'] == sorted(s.key for s in systems if s.key != 'tu') and meta['counts']['row_files'] == 1   # tu is no registered system
