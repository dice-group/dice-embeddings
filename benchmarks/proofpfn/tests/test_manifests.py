"""The frozen manifests: graph sets, views, hash checks, the blind-set guard and the model pins."""
import gzip
import json
import subprocess
import sys

import pytest

from benchmarks.proofpfn import manifests as mf

from .golden import fixture, golden_root, write_graph, write_manifest

REGISTERED_KEYS = {'S': 'rec10axlepcunoa', 'R': 'pcur', 'M': 'pcum', 'N': 'pcun', 'HT': 'pcuht', 'CU': 'rec10axlecunoa',
                   'UL': 'rec10axleulnoa', 'S0': 'rec10axlenoa', 'DT': 'rec10axledtnoa', 'PV': 'rec10axlepvnoa'}


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
    assert [e.key for e in gate.entries] == list(mf.GATE_GRAPHS)
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


def test_model_pins():
    pins = mf.read_models()
    assert {arm: f.key for arm, f in pins.families.items()} == REGISTERED_KEYS
    assert pins.families['HT'].export_sha256 is None   # filled by the lead when HT's training ends
    assert all(f.export_sha256 is not None for arm, f in pins.families.items() if arm != 'HT')
    assert not any(pins.families[a].loadable for a in ('S0', 'DT', 'PV'))
    assert set(pins.references) == {'trix', 'ultra3g', 'flockoff', 'merit1024', 'merit2025', 'merit2026', 'merit4000', 'merit8000'}
    models = pins.fv1_models()['models']
    assert 'pcuht' not in models and models['trix'] == {'model': pins.references['trix'].sha256}


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
            doc = mf.build_dev121(audit, inventory, suite, sources)
        elif name == 't1':
            doc = mf.build_t1(suite, audit, sources)
        else:
            g6 = json.loads(gzip.decompress((root / 'golden' / 'small' / 'g6_ranks.json.gz').read_bytes()))
            recorded = {g['record']['key']: {s: g['record']['sha256'][f'{s}.txt'] for s in mf.SPLITS} for g in g6['graphs'].values()}
            doc = mf.build_gate(audit, inventory, recorded, sources)
        assert mf.dumps(doc) == committed, name
    research = json.loads((root / 'work' / 'tree' / 'Experiments' / 'synthetic-kg' / 'fresh-v1' / 'models.json').read_text())
    pins = mf.read_models()
    assert pins.fv1_models() == {'models': research['models']}
    assert mf.file_sha256(root / 'work' / 'tree' / 'Experiments' / 'synthetic-kg' / 'fresh-v1' / 'models.json') == \
        json.loads((mf.MANIFESTS / 'models.json').read_text())['sources']['K/fresh-v1/models.json']
