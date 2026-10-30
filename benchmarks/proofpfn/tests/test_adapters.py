"""The shared pieces of the adapters (setup checks, ranks, row files), their command line, and the seam with the registered
tables: ``fv1_tables.py`` reads the row files of all three adapters under the registered keys.
"""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.proofpfn import adapters, cli
from benchmarks.proofpfn import evaluate as ev
from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn.adapters import AdapterSetupError, anyburl, kgpfn
from benchmarks.proofpfn.data import load_graph, prepare_queries
from dicee.evaluation._filtering import FilteredRanker

from .golden import tiny_graph, write_manifest
from .test_adapter_merit import fake_systems

REGISTERED = mf.REPO / 'benchmarks' / 'proofpfn' / 'registered'


def test_external_folder_and_pins(tmp_path):
    assert adapters.external_folder(tmp_path, 'PROOFPFN_TEST_UNSET_VARIABLE', 'X') == tmp_path.resolve()
    with pytest.raises(AdapterSetupError, match='PROOFPFN_TEST_UNSET_VARIABLE'):
        adapters.external_folder(None, 'PROOFPFN_TEST_UNSET_VARIABLE', 'X')
    with pytest.raises(AdapterSetupError, match='not a folder'):
        adapters.external_folder(tmp_path / 'missing', 'PROOFPFN_TEST_UNSET_VARIABLE', 'X')
    pins = {name: adapters.external_pins(name) for name in cli.ADAPTERS}
    assert set(pins['kgpfn']) >= {'code_commit', kgpfn.CHECKPOINT, 'tabicl'} and set(pins['anyburl']) >= {anyburl.JAR, anyburl.SOURCES}
    assert set(pins['merit']) >= {'code_commit', 'kgfm/models/merit.py', 'kgfm/tasks.py'}
    (tmp_path / 'models.json').write_text(json.dumps(dict(json.loads((mf.MANIFESTS / 'models.json').read_text()), external={})))
    with pytest.raises(AdapterSetupError, match='pins no files'):
        adapters.external_pins('kgpfn', tmp_path / 'models.json')
    with pytest.raises(mf.ManifestError, match='cannot read'):
        adapters.external_pins('kgpfn', tmp_path / 'missing.json')


def test_file_and_commit_checks(tmp_path):
    (tmp_path / 'f').write_text('x')
    digest = mf.file_sha256(tmp_path / 'f')
    assert adapters.check_file(tmp_path / 'f', digest, 'X') == digest
    for path, pin in ((tmp_path / 'f', '0' * 64), (tmp_path / 'g', digest)):
        with pytest.raises(AdapterSetupError):
            adapters.check_file(path, pin, 'X')
    git = tmp_path / 'code' / '.git'
    (git / 'refs' / 'heads').mkdir(parents=True)
    (git / 'HEAD').write_text('ref: refs/heads/main\n')
    (git / 'refs' / 'heads' / 'main').write_text('a' * 40 + '\n')
    assert adapters.check_commit(tmp_path / 'code', 'a' * 40, 'X') == 'a' * 40
    (git / 'refs' / 'heads' / 'main').unlink()
    (git / 'packed-refs').write_text('# pack-refs with: peeled\n' + 'b' * 40 + ' refs/heads/main\n')
    assert adapters.git_head(tmp_path / 'code') == 'b' * 40
    (git / 'HEAD').write_text('c' * 40)
    with pytest.raises(AdapterSetupError, match='commit'):
        adapters.check_commit(tmp_path / 'code', 'b' * 40, 'X')
    with pytest.raises(AdapterSetupError, match='no git checkout'):
        adapters.git_head(tmp_path)


def test_view_ranks_equal_the_research_rank():
    rng = np.random.default_rng(1)
    ranker = FilteredRanker(tie_policy='pessimistic')
    for _ in range(300):
        n = int(rng.integers(2, 30))
        v = torch.tensor(np.round(rng.normal(size=n), 1))
        v[rng.random(n) < 0.3] = -float('inf')                   # candidates without a score (AnyBURL: not listed)
        anchor, target = (int(x) for x in rng.integers(0, n, 2))
        others = sorted(set(rng.integers(0, n, int(rng.integers(0, n))).tolist()) - {target})
        mask = torch.ones(n, dtype=torch.bool)
        mask[torch.tensor(others, dtype=torch.long)] = False
        gated = v.clone()
        gated[anchor] = -float('inf')
        research = (int(((gated >= gated[target]) & mask).sum()), int(((v >= v[target]) & mask).sum()))   # lb5_eval.rank
        assert adapters.view_ranks(v, target, others, anchor, False, ranker) == research
        assert adapters.view_ranks(v, target, others, anchor, True, ranker) == (research[1], research[1])


@pytest.fixture(scope='module')
def adapter_rows(tmp_path_factory):
    """Row files of the three adapters on one tiny graph: AnyBURL and KGPFN from lists of random confidences, MERIT from
    stand-in models through the benchmark's runner."""
    root = tmp_path_factory.mktemp('adapters')
    manifest = mf.read_manifest(write_manifest(root / 'tiny.json', root, {'g': tiny_graph(root / 'KGs' / 'g')}))
    entry = manifest.entry('g')
    graph = load_graph(manifest, entry, root)
    queries = prepare_queries(graph, manifest.sample_rule)
    rng = np.random.default_rng(2)

    def listed() -> str:
        entities = rng.permutation(graph.num_entities)[:int(rng.integers(0, graph.num_entities))]
        return ''.join(f'e{e}\t{c!r}\t' for e, c in zip(entities.tolist(), np.round(rng.random(len(entities)), 1).tolist()))

    preds = root / 'preds'
    preds.write_text(''.join(f'e{h} r{r} e{t}\nHeads: {listed()}\nTails: {listed()}\n' for h, r, t in queries.facts))
    rows, _, _ = anyburl.rebuild(preds, graph, queries, graph.num_entities)
    out = root / 'rows'
    common = dict(manifest=manifest, entry=entry, graph=graph, queries=queries, seconds=1.5, code=anyburl.CODE)
    adapters.write_rows(adapters.row_path(out, entry, ['anyburl']), key='anyburl', spec=dict(kind='external'), protocol=anyburl.PROTOCOL,
                        rows=rows, meta_name='anyburl_meta', meta={}, **common)
    raw = {f'{side}_rank{view}': [0] * len(queries.facts) for side in ('tail', 'head') for view in ('', '_rule')}
    for r in rows:
        side = 'head' if r['head'] else 'tail'
        raw[f'{side}_rank'][r['index']], raw[f'{side}_rank_rule'][r['index']] = r['anyburl:raw'], r['anyburl']
    adapters.write_rows(adapters.row_path(out, entry, ['kgpfn']), key='kgpfn', spec=dict(kind='external'), protocol=kgpfn.PROTOCOL,
                        rows=kgpfn.rows_of(raw, queries), meta_name='kgpfn_meta', meta={}, **common)
    systems = fake_systems(root)
    assert ev.evaluate_graph(manifest, entry, systems, out, root=root, device='cpu') == 'done'
    return root, out, entry, systems


def test_adapter_row_files(adapter_rows):
    _, out, entry, _ = adapter_rows
    for key in ('anyburl', 'kgpfn'):
        doc = json.loads(adapters.row_path(out, entry, [key]).read_text())
        assert list(doc) == ['dataset', 'families', 'entities', 'relations', 'reflexive', 'protocol', 'metrics', 'rows', 'seconds', 'allgraph',
                             'metrics_views', f'{key}_meta', 'provenance']
        assert all(list(r) == ['index', 'head', key, f'{key}:raw'] for r in doc['rows'])
        assert doc['allgraph']['model_sha256'] == {} and doc['allgraph']['key'] == entry.key and doc['allgraph']['ranked_facts'] * 2 == len(doc['rows'])
        assert doc['metrics'] == {k: float(np.mean([1 / r[k] for r in doc['rows']])) for k in (key, f'{key}:raw')}


def test_row_writer_refusals(adapter_rows, tmp_path):
    root, out, entry, _ = adapter_rows
    manifest = mf.read_manifest(root / 'tiny.json')
    graph = load_graph(manifest, entry, root)
    queries = prepare_queries(graph, manifest.sample_rule)
    rows = json.loads(adapters.row_path(out, entry, ['kgpfn']).read_text())['rows']
    common = dict(manifest=manifest, entry=entry, graph=graph, queries=queries, key='kgpfn', spec={}, protocol='', seconds=1.0,
                  meta_name='kgpfn_meta', meta={}, code=())
    for bad, message in ((rows[::-1], 'query groups in order'), ([dict(r, kgpfn=0) for r in rows], 'rank outside'),
                         ([{**r, 'extra': 1} for r in rows], 'other keys')):
        with pytest.raises(ValueError, match=message):
            adapters.write_rows(tmp_path / 'x.json', rows=bad, **common)


def test_registered_tables_read_the_adapter_rows(adapter_rows, tmp_path):
    """The seam: the frozen fv1_tables.py reads the adapters' row files under the registered keys (kgpfn, anyburl, merit<S>)."""
    root, out, entry, systems = adapter_rows
    lists = tmp_path / 'lists'
    lists.mkdir()
    (lists / 'manifest.json').write_text(json.dumps(dict(graphs={})))
    (lists / 'block3.json').write_text(json.dumps(dict(graphs={'g': dict(cluster='tiny', files={f'{s}.txt': entry.sha256[s] for s in mf.SPLITS})})))
    (lists / 'models.json').write_text(json.dumps(dict(models={s.key: {'model': s.sha256} for s in systems})))
    subprocess.run([sys.executable, '-B', str(REGISTERED / 'fv1_tables.py'), '--rows', str(out), '--manifest', str(lists / 'manifest.json'),
                    '--block3', str(lists / 'block3.json'), '--models', str(lists / 'models.json'), '--out-dir', str(tmp_path / 'tables')],
                   check=True, capture_output=True)
    with (tmp_path / 'tables' / 'mrr.csv').open() as handle:
        table = {(r['system'], int(r['seed'])): r['mrr'] for r in csv.DictReader(handle)}

    def mrr(key: str, path: Path) -> str:
        rows = json.loads(path.read_text())['rows']
        return repr(math.fsum(1.0 / r[key] for r in rows) / len(rows))
    for key in ('anyburl', 'kgpfn'):
        path = adapters.row_path(out, entry, [key])
        assert table[key, 0] == mrr(f'{key}:raw', path) and table[f'{key}+rule', 0] == mrr(key, path)
    path = ev.row_file_path(out, entry, systems)
    for system, seed in zip(systems, (1024, 2025, 2026, 4000, 8000)):
        assert table['merit', seed] == mrr(f'{system.key}:raw', path) and table['merit+rule', seed] == mrr(system.key, path)
    meta = json.loads((tmp_path / 'tables' / 'tables.meta.json').read_text())
    assert meta['models_checked'] == sorted(s.key for s in systems) and {'anyburl', 'kgpfn'} <= set(meta['models_unchecked'])
    assert meta['counts']['row_files'] == 3   # the work folder of an adapter is a dot folder, which the tables skip


def test_command_line_refusals(tmp_path, capsys):
    common = ['--set', 'gate', '--graphs', 'UMLS', '--out', str(tmp_path / 'rows')]
    assert cli.main(['adapter', 'anyburl', *common, '--code', str(tmp_path / 'missing')]) == cli.EXIT_INPUT
    assert 'not a folder' in capsys.readouterr().err
    (tmp_path / 'KGPFN' / '.git').mkdir(parents=True)
    (tmp_path / 'KGPFN' / '.git' / 'HEAD').write_text('0' * 40)
    assert cli.main(['adapter', 'kgpfn', *common, '--code', str(tmp_path / 'KGPFN'), '--checkpoints', str(tmp_path)]) == cli.EXIT_INPUT
    assert 'commit' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(['adapter', 'other', *common])
    if not os.environ.get(anyburl.VARIABLE):
        assert cli.main(['adapter', 'anyburl', *common]) == cli.EXIT_INPUT and 'ANYBURL_HOME' in capsys.readouterr().err
