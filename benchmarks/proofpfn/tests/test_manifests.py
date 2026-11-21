"""The frozen manifests: graph sets, views, hash checks, malformed files, and the builders that derive the manifests from
the research inputs named in their ``sources``.
"""
import gzip
import json
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pytest

from benchmarks.proofpfn import manifests as mf

from .golden import GATE, fixture, golden_root, write_graph, write_manifest

GATE_GRAPHS = ('UMLS', 'ultra-suite-v1/NELLInductive_v1', 'inductive/grail-WN18RR_v1')   # their keys in the graph audit


def test_manifests_resolve_without_torch():
    code = 'import sys; import benchmarks.proofpfn.manifests as m; m.read_manifest("ultra54"); assert "torch" not in sys.modules, "torch was imported"'
    subprocess.run([sys.executable, '-B', '-c', code], cwd=mf.REPO, check=True)


def test_the_standard_suite():
    suite = mf.read_manifest('ultra54')
    assert {k: len(v) for k, v in suite.views.items()} == {'all': 54, 'trans': 13, 'ind_e': 18, 'ind_er': 23}
    assert suite.sample_rule == 'all' and len({e.suite_id for e in suite.entries}) == 54
    assert all(e.name == str(e.suite_id).replace(':', '_') and all(e.sha256.values()) and e.counts is not None and tuple(e.counts) == mf.COUNTS
               for e in suite.entries)
    assert [e.name for e in suite.select('trans', ['FBIngram_25', 'CoDExSmall'])] == ['CoDExSmall']
    assert [e.name for e in suite.select(graphs=['NELL995', 'CoDExSmall', 'NELL995'])] == ['NELL995', 'CoDExSmall']
    for bad in (dict(view='nosuch'), dict(graphs=['UMLS'])):
        with pytest.raises(KeyError):
            suite.select(**bad)
    with pytest.raises(ValueError):
        suite.select()


def test_gate_graphs_match_the_golden_record():
    gate = mf.read_manifest(GATE)
    assert [e.name for e in gate.entries] == ['UMLS', 'NELLInductive_v1', 'grail-WN18RR_v1'] and gate.sample_rule == 'first-1000'
    for name, recorded in fixture('g6_ranks.json.gz')['graphs'].items():
        entry = gate.entry(name)
        assert {f'{s}.txt': entry.sha256[s] for s in mf.SPLITS} == {k: v for k, v in recorded['record']['sha256'].items() if k != 'manifest'}
        folder = mf.REPO / entry.path
        if folder.is_dir():
            problems, hashes = mf.graph_problems(entry, folder)
            assert problems == [] and hashes['train_content'] == entry.content_sha256['train']


def test_hash_checks(tmp_path):
    folder = write_graph(tmp_path / 'g', ['a\tr\tb', 'b\tr\tc', 'a\tr\tb'], ['c\tr\ta'], ['b\tr\ta'])
    manifest = mf.read_manifest(write_manifest(tmp_path / 'm.json', tmp_path, {'g': folder}))
    entry = manifest.entry('g')
    assert mf.graph_problems(entry, folder)[0] == []
    (folder / 'valid.txt').write_text('c\tr\ta\n\n')   # another file, the same facts: only the file hash differs
    assert mf.graph_problems(entry, folder)[0] == ['g: the SHA-256 of valid.txt differs from the manifest']
    (folder / 'test.txt').write_text('b r a\nb\tr\ta\nx y\n')   # the same content: spaces and tabs, a duplicate, a 2-field line
    assert mf.content_sha256(folder / 'test.txt') == entry.content_sha256['test']
    (folder / 'train.txt').unlink()
    assert any('missing train.txt' in p for p in mf.graph_problems(entry, folder)[0])


def test_malformed_manifests_are_refused(tmp_path):
    folder = write_graph(tmp_path / 'g', ['a\tr\tb'], [], ['b\tr\ta'])
    good = json.loads(write_manifest(tmp_path / 'm.json', tmp_path, {'g': folder}).read_text())

    def refused(message: str, **changes: Any) -> None:
        doc = {k: v for k, v in dict(good, **changes).items() if v is not None}
        (tmp_path / 'bad.json').write_text(json.dumps(doc))
        with pytest.raises(mf.ManifestError, match=message):
            mf.read_manifest(tmp_path / 'bad.json')

    graph = good['graphs'][0]
    refused('is not a proofpfn-graph-manifest-v2 file', format='proofpfn-graph-manifest-v1')
    refused('unknown sample rule', sample_rule='first-10')
    refused('graph names must be unique', graphs=[graph, graph])
    refused('letters, digits', name='tiny set')
    refused('malformed manifest', graphs=None)
    refused('relative to the data root', graphs=[dict(graph, path=str(folder))])
    refused('malformed graph entry', graphs=[{k: v for k, v in graph.items() if k != 'sha256'}])
    refused('train and test only', graphs=[dict(graph, content_sha256=dict(valid='0' * 64))])
    (tmp_path / 'text.json').write_text('no JSON')
    for source, message in ((tmp_path / 'text.json', 'cannot read'), ('nosuch', 'neither a manifest')):
        with pytest.raises(mf.ManifestError, match=message):
            mf.read_manifest(source)


# ---------------------------------------------------------------------------------------------- the builders

def dumps(doc: Mapping[str, Any]) -> str:
    """The canonical text of a manifest file."""
    return json.dumps(doc, indent=1) + '\n'


def build_ultra54(suite: Mapping[str, Any], audit: Mapping[str, Any], sources: Mapping[str, str]) -> dict:
    """``ultra54``: the 54 zero-shot graphs of ``p0-suite-v1/manifest.json``, named by suite id with ``:`` as ``_``; ``size``
    from the audit entry of the same folder."""
    by_path = {g['path']: g for g in audit['graphs']}
    graphs = []
    for sid, g in suite['graphs'].items():
        if g['group'] == 'pretrain':
            continue
        audited = by_path[g['folder']]
        assert audited['suite_id'] == sid, f'{sid}: no audit entry with this suite id for {g["folder"]}'
        graphs.append(dict(name=sid.replace(':', '_'), path=g['folder'], view=g['group'], suite_id=sid,
                           size=dict(entities=audited['size']['entities'], inference_edges=audited['size']['inference_edges']),
                           sha256={s: g['files'][s]['sha256'] for s in mf.SPLITS}, content_sha256={},
                           counts={k: g['counts'][k] for k in mf.COUNTS}))
    assert len(graphs) == 54
    about = ('The standard zero-shot suite of 54 graphs, built under the official ULTRA conventions, in the groups trans, ind_e and ind_er. '
             'Every test row is ranked in both directions. File SHA-256 and counts from the suite manifest p0-suite-v1; the suite id is the '
             'key of the official Flock walk table; size from the audit entry of the same folder.')
    return dict(format=mf.FORMAT, name='ultra54', about=about, sample_rule='all', sources=dict(sources), graphs=graphs)


def build_gate(audit: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], recorded: Mapping[str, Mapping[str, str]],
               sources: Mapping[str, str]) -> dict:
    """``gate``: the graphs of ``GATE_GRAPHS`` with the content hashes of the audit inventory and the file hashes that the
    golden runs recorded (``recorded``: audit key to the SHA-256 per split)."""
    inv = {e['key']: e for e in inventory}
    by_key = {g['key']: g for g in audit['graphs']}
    graphs = []
    for key in GATE_GRAPHS:
        assert set(recorded[key]) == set(mf.SPLITS), f'{key}: no recorded file hashes'
        g = by_key[key]
        graphs.append(dict(name=g['name'], path=g['path'], view='gate', suite_id=g['suite_id'],
                           size=dict(entities=g['size']['entities'], inference_edges=g['size']['inference_edges']),
                           sha256={s: recorded[key][s] for s in mf.SPLITS},
                           content_sha256={s: inv[key][f'sha_{s}'] for s in mf.CONTENT_SPLITS}, counts=None))
    about = ('The three graphs of the exact-rank golden G6: UMLS, NELLInductive_v1 and grail-WN18RR_v1. File SHA-256 as the golden runs '
             'hashed the files; content hashes from the audit inventory.')
    return dict(format=mf.FORMAT, name='gate', about=about, sample_rule='first-1000', sources=dict(sources), graphs=graphs)


def test_manifests_rebuild_from_their_inputs():
    root = golden_root()
    kiowl = root / 'inputs' / 'kiowl'
    audit, inventory, suite = (json.loads((kiowl / p).read_text()) for p in
                               ('graph-audit-v1/graphs.json', 'graph-audit-v1/stats/inventory.json', 'p0-suite-v1/manifest.json'))
    for name, path in (('ultra54', mf.MANIFESTS / 'ultra54.json'), ('gate', GATE)):
        committed = path.read_text()
        sources = json.loads(committed)['sources']
        for source, digest in sources.items():   # 'K/<file>' (a kiowl input) or 'golden/small/<file> (what it holds)'
            file = kiowl / source.removeprefix('K/') if source.startswith('K/') else root / source.split(' ')[0]
            assert mf.file_sha256(file) == digest, source
        if name == 'ultra54':
            doc = build_ultra54(suite, audit, sources)
        else:
            g6 = json.loads(gzip.decompress((root / 'golden' / 'small' / 'g6_ranks.json.gz').read_bytes()))
            recorded = {g['record']['key']: {s: g['record']['sha256'][f'{s}.txt'] for s in mf.SPLITS} for g in g6['graphs'].values()}
            doc = build_gate(audit, inventory, recorded, sources)
        assert dumps(doc) == committed, name
