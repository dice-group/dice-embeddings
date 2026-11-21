"""The frozen manifests: graph sets, views, hash checks, malformed files, and the builders that derive the manifests from
the input files named in their ``sources``.

The builders run when ``PROOFPFN_MANIFEST_INPUTS`` names a folder that holds the input files (``inputs/kiowl/...`` for
``ultra54`` and ``inputs/fresh-v1/MANIFEST.json`` for ``heldout44``); without it the rebuild test skips.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from benchmarks.proofpfn import manifests as mf

from .helpers import GATE, pins, write_graph, write_manifest

INPUTS_ENV = 'PROOFPFN_MANIFEST_INPUTS'


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


def test_the_held_out_collection():
    held = mf.read_manifest('heldout44')
    assert {k: len(v) for k, v in held.views.items()} == {'all': 44, 'published': 16, 'converted': 28}
    assert held.sample_rule == 'first-1000' and all(e.suite_id is None and all(e.sha256.values()) for e in held.entries)
    assert all(e.counts is not None and tuple(e.counts) == mf.COUNTS and e.path == f'KGs/fresh-v1/{e.view}/{e.name}' for e in held.entries)
    assert not {e.name for e in held.entries} & {e.name for e in mf.read_manifest('ultra54').entries}


def test_gate_graphs_have_hashes_and_pins():
    gate = mf.read_manifest(GATE)
    assert [e.name for e in gate.entries] == ['UMLS', 'NELLInductive_v1', 'grail-WN18RR_v1'] and gate.sample_rule == 'first-1000'
    assert set(pins()['gate']) == {e.name for e in gate.entries}
    for entry in gate.entries:
        assert all(entry.sha256.values()) and set(entry.content_sha256) == set(mf.CONTENT_SPLITS)
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


def build_heldout44(collection: Mapping[str, Any], counts: Mapping[str, Mapping[str, int]], sources: Mapping[str, str]) -> dict:
    """``heldout44``: the 44 graphs of the collection manifest ``KGs/fresh-v1/MANIFEST.json`` with its file hashes; ``size``
    is its count of the entities and distinct triples of train.txt. ``counts`` (graph name to ``COUNTS``) comes from the
    graph files, as the benchmark's reader counts them."""
    graphs = []
    for name, g in sorted(collection['graphs'].items()):
        view = g['folder'].split('/')[2]
        assert view in ('published', 'converted') and g['folder'] == f'KGs/fresh-v1/{view}/{name}', name
        graphs.append(dict(name=name, path=g['folder'], view=view, suite_id=None,
                           size=dict(entities=g['counts']['entities'], inference_edges=g['counts']['inference_edges']),
                           sha256={s: g['sha256'][s] for s in mf.SPLITS}, content_sha256={},
                           counts={k: counts[name][k] for k in mf.COUNTS}))
    assert len(graphs) == 44
    about = ('The held-out collection of the thesis: 44 graphs outside the standard suites, 16 published graphs and 28 graphs converted from '
             'public sources. The test file of each graph holds the 1,000 test facts that the collection sampled, and every row is ranked in '
             'both directions. File SHA-256 and size from the manifest of the collection; counts from the graph files.')
    return dict(format=mf.FORMAT, name='heldout44', about=about, sample_rule='first-1000', sources=dict(sources), graphs=graphs)


def test_manifests_rebuild_from_their_inputs():
    if not os.environ.get(INPUTS_ENV) or not Path(os.environ[INPUTS_ENV]).is_dir():
        pytest.skip(f'set {INPUTS_ENV} to the folder with the input files of the manifests')
    root = Path(os.environ[INPUTS_ENV])
    kiowl = root / 'inputs' / 'kiowl'
    for name in ('ultra54', 'heldout44'):
        committed = (mf.MANIFESTS / f'{name}.json').read_text()
        sources = json.loads(committed)['sources']
        for source, digest in sources.items():   # 'K/<file>' (below inputs/kiowl) or '<path below the folder> (what it holds)'
            file = kiowl / source.removeprefix('K/') if source.startswith('K/') else root / source.split(' ')[0]
            assert mf.file_sha256(file) == digest, source
        if name == 'ultra54':
            audit, suite = (json.loads((kiowl / p).read_text()) for p in ('graph-audit-v1/graphs.json', 'p0-suite-v1/manifest.json'))
            doc = build_ultra54(suite, audit, sources)
        else:   # the counts need the graph files; load_graph checks them when a graph loads
            collection = json.loads((root / 'inputs' / 'fresh-v1' / 'MANIFEST.json').read_text())
            doc = build_heldout44(collection, {g['name']: g['counts'] for g in json.loads(committed)['graphs']}, sources)
        assert dumps(doc) == committed, name
