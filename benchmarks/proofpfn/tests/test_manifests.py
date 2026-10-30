"""The frozen manifests: graph sets, views, hash checks, the blind-set guard, the model pins, and the builders that derive
the graph manifests from the research inputs named in their ``sources``.
"""
import gzip
import json
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import pytest

from benchmarks.proofpfn import manifests as mf

from .golden import fixture, golden_root, write_graph, write_manifest

REGISTERED_KEYS = {'S': 'rec10axlepcunoa', 'R': 'pcur', 'M': 'pcum', 'N': 'pcun', 'HT': 'pcuht', 'CU': 'rec10axlecunoa',
                   'UL': 'rec10axleulnoa', 'S0': 'rec10axlenoa', 'DT': 'rec10axledtnoa', 'PV': 'rec10axlepvnoa'}
GROUP_CODE = {'E': 'E', 'L': 'L', 'B': 'B', 'S': 'S', 'O': 'O', 'R': 'R', 'T (reported apart)': 'T', 'contamination view': 'C'}
VIEW_OF_CODE = {**dict.fromkeys('ELBSOR', 'main'), 'C': 'contamination', 'T': 'apart'}
GATE_GRAPHS = ('UMLS', 'ultra-suite-v1/NELLInductive_v1', 'inductive/grail-WN18RR_v1')


def test_manifests_resolve_without_torch():
    code = ('import sys; import benchmarks.proofpfn.manifests as m; [m.read_manifest(n) for n in ("dev121", "t1", "gate")]; '
            'm.read_models(); assert "torch" not in sys.modules, "torch was imported"')
    subprocess.run([sys.executable, '-B', '-c', code], cwd=mf.REPO, check=True)


def test_dev121_views():
    dev = mf.read_manifest('dev121')
    views = {k: len(v) for k, v in dev.views.items()}
    assert views == {'all': 121, 'main': 91, 'contamination': 29, 'apart': 1} and dev.views['apart'] == ('UMLS',)
    main = dev.select('main')
    assert abs(sum(e.weight for e in main) - 1.0) < 1e-12 and all(e.code is not None and e.code in 'ELBSOR' for e in main)
    assert {e.code for e in dev.select('contamination')} == {'C'} and dev.sample_rule == 'first-1000' and not dev.blind
    assert all(e.content_sha256.keys() == {'train', 'test'} for e in dev.entries)
    assert not any('lockbox3' in e.path or 'fresh-v1' in e.path for e in dev.entries)
    assert 'grail-WN18RR_v1' not in dev.views['all']   # the audit excludes the copy; the gate lists it
    assert [e.name for e in dev.select(graphs=['NELLInductive_v1', 'UMLS'])] == ['NELLInductive_v1', 'UMLS']
    assert [e.name for e in dev.select('main', codes=['T'])] == []


def test_t1_suite():
    t1 = mf.read_manifest('t1')
    assert {k: len(v) for k, v in t1.views.items()} == {'all': 54, 'trans': 13, 'ind_e': 18, 'ind_er': 23}
    assert t1.sample_rule == 'all' and len({e.suite_id for e in t1.entries}) == 54
    assert all(e.name == str(e.suite_id).replace(':', '_') and all(e.sha256.values()) and e.counts is not None for e in t1.entries)
    assert sum(bool(e.tail_only_view) for e in t1.entries) == 5


def test_gate_graphs_match_the_golden_record():
    gate = mf.read_manifest('gate')
    assert [e.key for e in gate.entries] == list(GATE_GRAPHS)
    for name, recorded in fixture('g6_ranks.json.gz')['graphs'].items():
        entry = gate.entry(name)
        assert {f'{s}.txt': entry.sha256[s] for s in mf.SPLITS} == {k: v for k, v in recorded['record']['sha256'].items() if k != 'manifest'}
        folder = gate.folder(entry, mf.REPO)
        if folder.is_dir():
            problems, hashes = mf.graph_problems(entry, folder)
            assert problems == [] and hashes['train_content'] == entry.content_sha256['train']


def test_hash_checks(tmp_path):
    folder = write_graph(tmp_path / 'g', ['a\tr\tb', 'b\tr\tc', 'a\tr\tb'], ['c\tr\ta'], ['b\tr\ta'])
    manifest = mf.read_manifest(write_manifest(tmp_path / 'm.json', tmp_path, {'g': folder}))
    entry = manifest.entry('g')
    assert mf.graph_problems(entry, folder)[0] == []
    (folder / 'valid.txt').write_text('c\tr\ta\n\n')   # another file, the same facts: only the file hash differs
    assert mf.graph_problems(entry, folder)[0] == ['tiny/g: the SHA-256 of valid.txt differs from the manifest']
    (folder / 'test.txt').write_text('b r a\nb\tr\ta\nx y\n')   # the same content: spaces and tabs, a duplicate, a 2-field line
    assert mf.content_sha256(folder / 'test.txt') == entry.content_sha256['test']
    (folder / 'train.txt').unlink()
    assert any('missing train.txt' in p for p in mf.graph_problems(entry, folder)[0])


def test_blind_sets_open_only_with_their_release_flag(tmp_path):
    blind_dir = tmp_path / 'KGs' / 'lockbox3' / 'g'
    folder = write_graph(blind_dir, ['a\tr\tb'], [], ['b\tr\ta'])
    path = write_manifest(tmp_path / 'blind.json', tmp_path, {'g': folder}, name='blindtest', blind=True)
    with pytest.raises(mf.BlindSetError):
        mf.read_manifest(path)
    with pytest.raises(mf.BlindSetError):
        mf.read_manifest(path, release='another')
    released = mf.read_manifest(path, release='blindtest')
    assert released.blind and released.folder(released.entry('g'), tmp_path) == folder
    with pytest.raises(mf.BlindSetError):   # a development manifest may not name a blind folder
        mf.read_manifest(write_manifest(tmp_path / 'dev.json', tmp_path, {'g': folder}, name='devtest'))
    assert mf.in_blind_folder('KGs/fresh-v1/x') and mf.in_blind_folder('KGs/lockbox3/x') and not mf.in_blind_folder('KGs/lockbox4/x')


def test_the_blind_guard_compares_resolved_paths(tmp_path, monkeypatch):
    shifted = write_graph(tmp_path / 'KGs' / 'fresh-v1' / 'g', ['a\tr\tb'], [], ['b\tr\ta'])
    with pytest.raises(mf.BlindSetError):   # the data root KGs and the path fresh-v1/g
        mf.read_manifest(write_manifest(tmp_path / 'shifted.json', tmp_path / 'KGs', {'g': shifted}, name='devtest'))
    blind = write_graph(tmp_path / 'elsewhere' / 'lockbox3' / 'g', ['a\tr\tb'], [], ['b\tr\ta'])
    secret = write_graph(tmp_path / 'secret' / 'g', ['a\tr\tb'], [], ['b\tr\ta'])
    (tmp_path / 'KGs' / 'dev').symlink_to(blind.parent)    # names without a blind part that resolve into a blind folder
    (tmp_path / 'KGs' / 'dev2').symlink_to(secret.parent)
    folders = {'g': tmp_path / 'KGs' / 'dev' / 'g', 'h': tmp_path / 'KGs' / 'dev2' / 'g'}
    manifest = mf.read_manifest(write_manifest(tmp_path / 'linked.json', tmp_path, folders, name='devtest'))
    with pytest.raises(mf.BlindSetError):
        manifest.folder(manifest.entry('g'), tmp_path)
    assert manifest.folder(manifest.entry('h'), tmp_path) == folders['h']
    monkeypatch.setattr(mf, 'BLIND_FOLDERS', (str(tmp_path / 'secret'),))   # a blind folder of the repository by its resolved path
    with pytest.raises(mf.BlindSetError):
        manifest.folder(manifest.entry('h'), tmp_path)


def test_model_pins():
    pins = mf.read_models()
    assert {arm: f.key for arm, f in pins.families.items()} == REGISTERED_KEYS
    assert all(f.export_sha256 is not None for f in pins.families.values())
    assert not any(pins.families[a].loadable for a in ('S0', 'DT', 'PV'))
    assert set(pins.references) == {'trix', 'ultra3g', 'flockoff', 'merit1024', 'merit2025', 'merit2026', 'merit4000', 'merit8000'}
    models = pins.fv1_models()['models']
    ht = pins.families['HT'].export_sha256
    assert ht is not None and models['pcuht'] == {str(s): v for s, v in ht.items()}
    assert models['trix'] == {'model': pins.references['trix'].sha256}


def test_pending_pins_leave_the_fv1_models(tmp_path):
    doc = json.loads((mf.MANIFESTS / 'models.json').read_text())
    doc['families']['HT']['export_sha256'] = None
    path = tmp_path / 'models.json'
    path.write_text(json.dumps(doc))
    pins = mf.read_models(path)
    assert pins.families['HT'].export_sha256 is None and 'pcuht' not in pins.fv1_models()['models']


# ---------------------------------------------------------------------------------------------- the builders

def dumps(doc: Mapping[str, Any]) -> str:
    """The canonical text of a manifest file."""
    return json.dumps(doc, indent=1) + '\n'


def manifest(name: str, about: str, sample_rule: str, sources: Mapping[str, str], graphs: list[dict]) -> dict:
    """A development manifest of graph entries."""
    return dict(format=mf.FORMAT, name=name, about=about, blind=False, sample_rule=sample_rule, sources=dict(sources), graphs=graphs)


def audit_entry(g: Mapping[str, Any], inventory: Mapping[str, Mapping[str, Any]], view: Optional[str] = None) -> dict:
    """The entry of an audited graph: its audit fields, the content hashes of the inventory and no file hashes."""
    code = GROUP_CODE.get(g['group']) if g['group'] is not None else None
    if view is None:
        assert code is not None, f"{g['key']}: the audit gives no group, so the graph has no view"
        view = VIEW_OF_CODE[code]
    inv = inventory[g['key']]
    return dict(key=g['key'], name=g['name'], path=g['path'], collection=g['collection'], view=view,
                code=code, group=g['group'], verdict=g['verdict'], weight=g['weight'], origin_class=g['origin_class'],
                suite_id=g['suite_id'], size=dict(entities=g['size']['entities'], inference_edges=g['size']['inference_edges']),
                sha256=dict.fromkeys(mf.SPLITS), content_sha256={s: inv[f'sha_{s}'] for s in mf.CONTENT_SPLITS}, counts=None, tail_only_view=None)


def build_dev121(audit: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], suite: Mapping[str, Any], sources: Mapping[str, str]) -> dict:
    """``dev121`` from ``graph-audit-v1/graphs.json``, its ``stats/inventory.json`` and ``p0-suite-v1/manifest.json``: the 121
    graphs with an audit group, in audit order; the file hashes from the suite manifest where it records the same folder."""
    inv = {e['key']: e for e in inventory}
    files = {g['folder']: g['files'] for g in suite['graphs'].values()}
    graphs = []
    for g in audit['graphs']:
        if g['group'] is None:
            continue
        assert 'lockbox3' not in g['path'], f"{g['key']}: an audited path names lockbox3"
        entry = audit_entry(g, inv)
        if g['path'] in files:
            entry['sha256'] = {s: files[g['path']][s]['sha256'] for s in mf.SPLITS}
        graphs.append(entry)
    assert len(graphs) == len({g['name'] for g in graphs}) == 121   # as ag_eval asserts
    about = ('The 121 audited development graphs of graph-audit-v1 (16 October 2026): views main (91 graphs of the groups E, L, '
             'B, S, O, R; weights sum to 1), contamination (29) and apart (UMLS). Content hashes of train.txt and test.txt '
             'from the audit inventory (checked before a graph loads); file hashes from the standard suite manifest where it '
             'records the same folder, else null (the research harness checked the content hashes only).')
    return manifest('dev121', about, 'first-1000', sources, graphs)


def build_t1(suite: Mapping[str, Any], audit: Mapping[str, Any], sources: Mapping[str, str]) -> dict:
    """``t1``: the 54 zero-shot graphs of ``p0-suite-v1/manifest.json``, named by suite id with ``:`` as ``_``; ``size`` from
    the audit entry of the same folder."""
    by_path = {g['path']: g for g in audit['graphs']}
    graphs = []
    for sid, g in suite['graphs'].items():
        if g['group'] == 'pretrain':
            continue
        audited = by_path[g['folder']]
        assert audited['suite_id'] == sid, f'{sid}: no audit entry with this suite id for {g["folder"]}'
        name = sid.replace(':', '_')
        graphs.append(dict(key=f't1/{name}', name=name, path=g['folder'], collection='t1', view=g['group'], code=None, group=g['group'],
                           verdict=None, weight=0.0, origin_class=None, suite_id=sid,
                           size=dict(entities=audited['size']['entities'], inference_edges=audited['size']['inference_edges']),
                           sha256={s: g['files'][s]['sha256'] for s in mf.SPLITS}, content_sha256={}, counts=dict(g['counts']),
                           tail_only_view=g['tail_only_view']))
    assert len(graphs) == 54
    about = ('T1, the standard zero-shot suite p0-suite-v1 (built under the official ULTRA conventions): 54 graphs in the groups '
             'trans, ind_e and ind_er, every test row ranked in both directions. File SHA-256 and counts from the suite manifest; '
             'the suite id is the key of the official Flock walk table; size from the audit entry of the same folder.')
    return manifest('t1', about, 'all', sources, graphs)


def build_gate(audit: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]], recorded: Mapping[str, Mapping[str, str]],
               sources: Mapping[str, str]) -> dict:
    """``gate``: the graphs of ``GATE_GRAPHS`` with the audit fields, the content hashes and the file hashes that the golden
    runs recorded (``recorded``: graph key to the SHA-256 per split)."""
    inv = {e['key']: e for e in inventory}
    by_key = {g['key']: g for g in audit['graphs']}
    graphs = []
    for key in GATE_GRAPHS:
        assert set(recorded[key]) == set(mf.SPLITS), f'{key}: no recorded file hashes'
        entry = audit_entry(by_key[key], inv, view='gate')
        entry['sha256'] = {s: recorded[key][s] for s in mf.SPLITS}
        graphs.append(entry)
    about = ('The gate graphs of the exact-rank goldens G6 and G7: UMLS, NELLInductive_v1 and grail-WN18RR_v1. The audit excludes '
             'grail-WN18RR_v1 (X2: a copy of ultra-suite-v1/WN18RRInductive_v1), so it is listed here and not in dev121. File '
             'SHA-256 as the golden runs hashed the files; content hashes from the audit inventory.')
    return manifest('gate', about, 'first-1000', sources, graphs)


def test_manifests_rebuild_from_their_inputs():
    root = golden_root()
    kiowl = root / 'inputs' / 'kiowl'
    audit, inventory, suite = (json.loads((kiowl / p).read_text()) for p in
                               ('graph-audit-v1/graphs.json', 'graph-audit-v1/stats/inventory.json', 'p0-suite-v1/manifest.json'))
    for name in ('dev121', 't1', 'gate'):
        committed = (mf.MANIFESTS / f'{name}.json').read_text()
        sources = json.loads(committed)['sources']
        for source, digest in sources.items():   # 'K/<file>' (a kiowl input) or 'golden/small/<file> (what it holds)'
            path = kiowl / source.removeprefix('K/') if source.startswith('K/') else root / source.split(' ')[0]
            assert mf.file_sha256(path) == digest, source
        if name == 'dev121':
            doc = build_dev121(audit, inventory, suite, sources)
        elif name == 't1':
            doc = build_t1(suite, audit, sources)
        else:
            g6 = json.loads(gzip.decompress((root / 'golden' / 'small' / 'g6_ranks.json.gz').read_bytes()))
            recorded = {g['record']['key']: {s: g['record']['sha256'][f'{s}.txt'] for s in mf.SPLITS} for g in g6['graphs'].values()}
            doc = build_gate(audit, inventory, recorded, sources)
        assert dumps(doc) == committed, name
    research_path = root / 'work' / 'tree' / 'Experiments' / 'synthetic-kg' / 'fresh-v1' / 'models.json'
    ht_path = kiowl / 'fresh-v1' / 'models-pcuht.json'   # written on kiowl after the snapshot, so a kiowl input
    research = json.loads(research_path.read_text())['models']
    research_ht = json.loads(ht_path.read_text())['models']   # the frozen worker merges the two files the same way
    pins = mf.read_models()
    assert pins.fv1_models() == {'models': dict(research, **research_ht)}
    sources = json.loads((mf.MANIFESTS / 'models.json').read_text())['sources']
    assert sources == {'K/fresh-v1/models.json': mf.file_sha256(research_path), 'K/fresh-v1/models-pcuht.json': mf.file_sha256(ht_path)}
