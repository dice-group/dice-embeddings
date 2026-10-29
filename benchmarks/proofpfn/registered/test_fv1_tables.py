"""Tests of fv1_tables.py on synthetic row files: no graph file, no model and no real row file. Plain asserts.

    systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 /home/lukef/miniconda3/bin/python -B \\
        test_fv1_tables.py [--tmp DIR]

The synthetic run lives in a new folder under --tmp (default: the system temporary folder) and is removed at the end.
The small cases set fa.QUERIES_PER_GRAPH to 40 (as test_fv1_analysis.py does); the last case uses the registered 2000.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import random
import shutil
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fv1_analysis as fa  # noqa: E402
import fv1_tables as ft  # noqa: E402

Q = 40                       # queries per confirmatory graph in the small cases
GRAPHS = {                   # name: (block, source); SRC-A spans blocks 1a and 1b
    'pub-a1': ('1a', 'SRC-A'), 'pub-b1': ('1a', 'SRC-B'), 'pub-a2': ('1b', 'SRC-A'), 'pub-c1': ('1b', 'SRC-C'),
    'conv-d1': ('2', 'SRC-D'), 'conv-e1': ('2', 'SRC-E')}
BLOCK3 = {'lb3-x': ('dbpedia', 13), 'lb3-y': ('wikidata', 9)}   # name: (cluster, ranked test facts)
EXTRA = {'lb3-x': ['N']}     # a registered system outside BLOCK3_SYSTEMS on a block 3 graph: kept, never required
NO_HASH = ('kgpfn', 'anyburl')   # references whose runners record their hashes elsewhere: no model_sha256 entry
SPLITS = ('train.txt', 'valid.txt', 'test.txt')
S_KEY = fa.SYSTEMS['S'][0]
G1, G3 = 'pub-a1', 'lb3-x'   # the graphs that the stop cases change


# --- the synthetic run ----------------------------------------------------------------------------------------------

def hexdigest(*parts) -> str:
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding='utf-8')


def load(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def edit(path: Path, change) -> None:
    doc = load(path)
    change(doc)
    write_json(path, doc)


def keys_of(label: str) -> tuple:
    """Of one system: its registered rank keys {key: (system, seed)}, its extra views of section 10 {key: (extra
    system, seed)}, the other keys of a harness row (ignored), and the allgraph.model_sha256 entries of its runner
    (the format of models.json)."""
    key, seeds = fa.SYSTEMS[label]
    if seeds == fa.OUR_SEEDS:
        reg = {f'{key}.seed{s}:raw': (key, s) for s in seeds}
        ext = {f'{key}:raw': (f'{key}+ens', 0), **{f'{key}.seed{s}': (f'{key}+rule', s) for s in seeds}}
        other = [key, f'{key}:gt2', f'{key}:raw:gt2'] + [f'{key}.seed{s}{v}' for s in seeds for v in (':gt2', ':raw:gt2')]
        hashes = {key: {str(s): hexdigest('model', key, s) for s in seeds}}
    elif seeds == fa.MERIT_SEEDS:
        reg = {f'{key}{s}:raw': (key, s) for s in seeds}
        ext = {f'{key}{s}': (f'{key}+rule', s) for s in seeds}
        other = [key] + [f'{key}{s}{v}' for s in seeds for v in (':gt2', ':raw:gt2')]
        hashes = {f'{key}{s}': {'model': hexdigest('model', key, s)} for s in seeds}
    else:
        reg = {f'{key}:raw': (key, 0)}
        ext = {} if key in NO_HASH else {key: (f'{key}+rule', 0)}   # here the kgpfn and anyburl runners write no rule view
        other = [f'{key}:gt2', f'{key}:raw:gt2']
        hashes = {} if key in NO_HASH else {key: {'model': hexdigest('model', key, 0)}}
    return reg, ext, other, hashes


def jobs(graph: str, block: str) -> dict:
    """{runner job: system labels}, split as in the run order (registration, section 15)."""
    labels = fa.BLOCK3_SYSTEMS if block == '3' else tuple(fa.SYSTEMS)
    ours = [x for x in labels if fa.SYSTEMS[x][1] == fa.OUR_SEEDS]
    out = {'ours-a': ours[:3], 'ours-b': ours[3:]}
    out.update({x.lower(): [x] for x in labels if x not in ours})
    if graph in EXTRA:
        out['extra'] = list(EXTRA[graph])
    return {j: v for j, v in out.items() if v}


def row_file(graph: str, block: str, facts: int, labels: list, seconds: float) -> tuple:
    """One harness row file and its truth {(system, seed): {(index, head): rank}}, registered models and extras."""
    rng = random.Random(zlib.crc32(graph.encode()))
    pairs = [(i, h) for i in sorted(rng.sample(range(3 * facts), facts)) for h in (True, False)]
    cols, truth, extra, hashes = {}, {}, {}, {}
    for label in labels:
        reg, ext, other, entries = keys_of(label)
        hashes.update(entries)
        for k, target in (*reg.items(), *ext.items()):
            r = random.Random(zlib.crc32(f'{graph}|{k}'.encode()))
            cols[k] = [1 + int(r.expovariate(0.08)) for _ in pairs]
            (truth if k in reg else extra)[target] = dict(zip(pairs, cols[k]))
        for k in other:
            cols[k] = [1] * len(pairs)   # rank 1 everywhere: a confusion with a registered key would show in the MRR
    rows = []
    for n, (i, h) in enumerate(pairs):
        row = {'index': i, 'head': h}
        row.update((k, v[n]) for k, v in cols.items())
        rows.append(row)
    sha256 = {f: hexdigest(graph, f) for f in SPLITS}
    sha256['manifest'] = 'block3' if block == '3' else 'fresh'
    ag = dict(key=f'{"fresh-v1-block3" if block == "3" else "fresh-v1"}/{graph}', collection=f'fresh-v1-{block}',
              path=f'KGs/synthetic/{graph}', sha256=sha256, model_sha256=hashes, test_rows=3 * facts,
              ranked_facts=facts, sampled=False)
    return dict(allgraph=ag, rows=rows, seconds=seconds), truth, extra


def build_tree(root: Path, graphs: dict, block3: dict, q: int) -> dict:
    """The synthetic run: root/rows/<job>/<graph>.json, manifest.json, block3.json, models.json, plus noise that the
    extraction must skip (a hidden folder, a hidden file, a log). Returns the truth per (graph, system, seed)."""
    listed = {g: (b, s, q // 2) for g, (b, s) in graphs.items()}
    listed.update({g: ('3', c, n) for g, (c, n) in block3.items()})
    truth, extra, cost = {}, {}, {}
    for gi, (g, (block, _, facts)) in enumerate(sorted(listed.items())):
        for ji, (job, labels) in enumerate(sorted(jobs(g, block).items())):
            seconds = 30.0 + 7.25 * ji + 0.5 * gi
            doc, t, e = row_file(g, block, facts, labels, seconds)
            write_json(root / 'rows' / job / f'{g}.json', doc)
            for (system, seed), ranks in t.items():
                truth[(g, system, seed)] = ranks
                cost[(g, system, seed)] = seconds / len(t)
            extra.update({(g, system, seed): ranks for (system, seed), ranks in e.items()})
    manifest = dict(version='synthetic', graphs={
        g: dict(block=b, source=s, folder=f'KGs/fresh-v1/_build/{b}/{g}', sha256={f[:-4]: hexdigest(g, f) for f in SPLITS},
                counts=dict(entities=900, inference_edges=12000)) for g, (b, s) in graphs.items()})
    b3 = dict(folder='KGs/lockbox3', graphs={g: dict(family='isdea', cluster=c, files={f: hexdigest(g, f) for f in SPLITS})
                                              for g, (c, _) in block3.items()})
    models = {}
    for label in fa.SYSTEMS:
        models.update(keys_of(label)[3])
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'block3.json', b3)
    write_json(root / 'models.json', dict(models=models, info=dict(note='synthetic hashes')))
    stray = dict(allgraph=dict(key='fresh-v1/not-listed'), rows=[], seconds=1.0)   # would stop the run if read
    write_json(root / 'rows' / '.scratch' / 'not-listed.json', stray)
    write_json(root / 'rows' / 'ours-a' / '.not-listed.json', stray)
    (root / 'rows' / 'logs').mkdir(parents=True, exist_ok=True)
    (root / 'rows' / 'logs' / 'run.log').write_text('exit 0\n', encoding='utf-8')
    return dict(truth=truth, extra=extra, cost=cost, manifest=manifest, listed=listed)


def run(root: Path, out: Path, rows: Optional[list] = None, models: bool = True) -> tuple:
    """fv1_tables.main on the synthetic run. Returns (True, '') or (False, the stop message)."""
    args = []
    for r in rows or [root / 'rows']:
        args += ['--rows', str(r)]
    args += ['--manifest', str(root / 'manifest.json'), '--block3', str(root / 'block3.json'), '--out-dir', str(out)]
    if models:
        args += ['--models', str(root / 'models.json')]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ft.main(args)
    except SystemExit as e:
        return False, str(e.code)
    return True, ''


def outputs(out: Path) -> dict:
    return {name: (out / name).read_bytes() for name in (*ft.HEADERS, ft.META)}


def p(c: Path, job: str, g: str = G1) -> Path:
    return c / 'rows' / job / f'{g}.json'


def write_queries(path: Path, ranks_csv: Path) -> None:
    """A synthetic queries.csv (the cell predictors that fv1_cells.py writes) for the queries of ranks.csv."""
    rng = random.Random(5)
    qs = sorted({(r['graph'], r['query']) for r in fa.read_csv(ranks_csv)})
    lines = ['graph,query,r1_at,rank_peer,r2_at'] + [
        f'{g},{q},{int(rng.random() < 0.2)},{rng.randint(1, 40)},{int(rng.random() < 0.3)}' for g, q in qs]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# --- changes of a copy of the run ---------------------------------------------------------------------------------

def drop_fact(doc: dict) -> None:
    """Remove both queries of the first test fact."""
    first = doc['rows'][0]['index']
    doc['rows'] = [r for r in doc['rows'] if r['index'] != first]


def move_fact(doc: dict, index: int, both: bool = True) -> None:
    """Give the first test fact another index: both of its queries, or the first row only."""
    first = doc['rows'][0]['index']
    for r in doc['rows'][:None if both else 1]:
        if r['index'] == first:
            r['index'] = index


def set_rank(value):
    def change(doc: dict) -> None:
        doc['rows'][2][f'{S_KEY}.seed43:raw'] = value
    return change


def copy_as(src: Path, dst: Path, key: Optional[str] = None) -> None:
    doc = load(src)
    if key is not None:
        doc['allgraph']['key'] = key
    write_json(dst, doc)


def rule_only(c: Path, change=None, keep: bool = False) -> None:
    """Move the rule view of trix on G1 into a file of its own (a runner job that writes only that view).

    change: applied to the new file. keep: the original file keeps its rule view too (one extra in two files).
    """
    doc = load(p(c, 'trix'))
    view = dict(doc, rows=[dict(index=r['index'], head=r['head'], trix=r['trix']) for r in doc['rows']])
    if change is not None:
        change(view)
    write_json(p(c, 'trix-rule'), view)
    if not keep:
        edit(p(c, 'trix'), lambda d: [r.pop('trix') for r in d['rows']])


def broken_link(c: Path) -> None:
    (c / 'rows' / 'links').mkdir()
    (c / 'rows' / 'links' / f'{G1}.json').symlink_to(c / 'rows' / 'nowhere.json')


def shuffle_rows(root: Path, seed: int) -> None:
    """Rewrite every row file with its rows, and the keys of every row, in random order."""
    for path in sorted((root / 'rows').rglob('*.json')):
        doc = load(path)
        if doc.get('rows'):
            rng = random.Random(seed + zlib.crc32(path.name.encode()))
            rows = [dict(rng.sample(list(r.items()), len(r))) for r in doc['rows']]
            rng.shuffle(rows)
            doc['rows'] = rows
            write_json(path, doc)


STOPS = [  # (case, change of a copy of the run, a fragment of the stop message)
    ('a graph in neither list', lambda c: copy_as(p(c, 'trix'), p(c, 'more', 'pub-zz'), 'fresh-v1/pub-zz'),
     "graph 'pub-zz' is in neither the manifest nor the block 3 list"),
    ('allgraph.key against the file name', lambda c: edit(p(c, 'trix'), lambda d: d['allgraph'].update(key='fresh-v1/pub-b1')),
     "allgraph.key 'fresh-v1/pub-b1' does not end in the file name 'pub-a1'"),
    ('train.txt hash', lambda c: edit(p(c, 'trix'), lambda d: d['allgraph']['sha256'].update({'train.txt': '0' * 64})),
     'the SHA-256 of train.txt differs from the manifest'),
    ('block 3: test.txt hash', lambda c: edit(p(c, 'trix', G3), lambda d: d['allgraph']['sha256'].update({'test.txt': 'f' * 64})),
     'the SHA-256 of test.txt differs from the block 3 list'),
    ('valid.txt hash absent', lambda c: edit(p(c, 'merit'), lambda d: d['allgraph']['sha256'].pop('valid.txt')),
     'the SHA-256 of valid.txt differs from the manifest (or is absent)'),
    ('rows per model (blocks 1a, 1b, 2)', lambda c: edit(p(c, 'ours-a'), drop_fact),
     f'{Q - 2} rows per model, blocks 1a, 1b and 2 need {Q}'),
    ('duplicate (index, head)', lambda c: edit(p(c, 'ours-b'), lambda d: d['rows'][1].update(index=d['rows'][0]['index'],
                                                                                            head=d['rows'][0]['head'])),
     '1 duplicate (index, head) pairs'),
    ('a test fact in one direction', lambda c: edit(p(c, 'ours-b'), lambda d: move_fact(d, 10 ** 6, both=False)),
     'test facts ranked in one direction only'),
    ('other (index, head) pairs than another model', lambda c: edit(p(c, 'kgpfn'), lambda d: move_fact(d, 10 ** 6)),
     'ranks other (index, head) pairs than'),
    ('block 3: other pairs', lambda c: edit(p(c, 'merit', G3), lambda d: move_fact(d, 10 ** 6)),
     'ranks other (index, head) pairs than'),
    ('block 3: not 2 x ranked_facts', lambda c: edit(p(c, 'ours-a', G3), lambda d: d['allgraph'].update(
        ranked_facts=d['allgraph']['ranked_facts'] + 1)), 'block 3 needs 2 x allgraph.ranked_facts'),
    ('block 3: a smaller set with a matching ranked_facts', lambda c: edit(p(c, 'trix', G3), lambda d: (
        drop_fact(d), d['allgraph'].update(ranked_facts=d['allgraph']['ranked_facts'] - 1))),
     'ranks other (index, head) pairs than'),
    ('one model in two files (equal values)', lambda c: copy_as(p(c, 'trix'), p(c, 'trix-again')), f'{G1} trix seed 0: in '),
    ('rank 0', lambda c: edit(p(c, 'ours-a'), set_rank(0)), 'ranks are not an int >= 1 (first: 0)'),
    ('rank -3', lambda c: edit(p(c, 'ours-a'), set_rank(-3)), 'ranks are not an int >= 1 (first: -3)'),
    ('rank 2.0', lambda c: edit(p(c, 'ours-a'), set_rank(2.0)), 'ranks are not an int >= 1 (first: 2.0)'),
    ('rank true', lambda c: edit(p(c, 'ours-a'), set_rank(True)), 'ranks are not an int >= 1 (first: True)'),
    ('rank "3"', lambda c: edit(p(c, 'ours-a'), set_rank('3')), "ranks are not an int >= 1 (first: '3')"),
    ('rank null', lambda c: edit(p(c, 'ours-a'), set_rank(None)), 'ranks are not an int >= 1 (first: None)'),
    ('a rank missing on one row', lambda c: edit(p(c, 'ours-a'), lambda d: d['rows'][5].pop(f'{S_KEY}.seed43:raw')),
     f'rank key {S_KEY}.seed43:raw is missing on 1 of {Q} rows'),
    ('extra: a rule-view rank missing on one row', lambda c: edit(p(c, 'ours-a'), lambda d: d['rows'][5].pop(f'{S_KEY}.seed43')),
     f'rank key {S_KEY}.seed43 is missing on 1 of {Q} rows'),
    ('extra: an ensemble rank 0', lambda c: edit(p(c, 'ours-b'), lambda d: d['rows'][7].update({f'{fa.SYSTEMS["N"][0]}:raw': 0})),
     f'rank key {fa.SYSTEMS["N"][0]}:raw: 1 of {Q} ranks are not an int >= 1 (first: 0)'),
    ('extra: a reference rule rank 2.5', lambda c: edit(p(c, 'trix'), lambda d: d['rows'][3].update(trix=2.5)),
     f'rank key trix: 1 of {Q} ranks are not an int >= 1 (first: 2.5)'),
    ('extra: a MERIT rule rank null', lambda c: edit(p(c, 'merit', G3), lambda d: d['rows'][0].update(merit4000=None)),
     f'rank key merit4000: 1 of {2 * BLOCK3[G3][1]} ranks are not an int >= 1 (first: None)'),
    ('extra: an extras-only file with other pairs', lambda c: rule_only(c, lambda v: move_fact(v, 10 ** 6)),
     'ranks other (index, head) pairs than'),
    ('extra: an extras-only file with too few rows', lambda c: rule_only(c, drop_fact),
     f'{Q - 2} rows per model, blocks 1a, 1b and 2 need {Q}'),
    ('extra: one extra in two files', lambda c: rule_only(c, keep=True), f'{G1} trix+rule seed 0: in '),
    ('--models: a seed hash of ours differs', lambda c: edit(c / 'models.json', lambda d: d['models'][S_KEY].update({'43': '0' * 64})),
     f"allgraph.model_sha256['{S_KEY}'] differs from --models (43)"),
    ('--models: a MERIT checkpoint hash differs', lambda c: edit(p(c, 'merit'), lambda d: d['allgraph']['model_sha256']['merit2025']
                                                                .update(model='0' * 64)),
     "allgraph.model_sha256['merit2025'] differs from --models (model)"),
    ('--models: a seed of ours absent from the entry', lambda c: edit(p(c, 'ours-a'), lambda d: d['allgraph']['model_sha256'][S_KEY]
                                                                     .pop('44')),
     f"allgraph.model_sha256['{S_KEY}'] differs from --models (44)"),
    ('--models: the entry of a family of ours absent', lambda c: edit(p(c, 'ours-a'), lambda d: d['allgraph']['model_sha256']
                                                                     .pop(S_KEY)),
     f"allgraph.model_sha256 has no entry '{S_KEY}', whose models the file holds"),
    ('--models: a wrong entry of a model that the file does not hold',
     lambda c: edit(p(c, 'ours-b'), lambda d: d['allgraph']['model_sha256'].update({S_KEY: {'42': '0' * 64}})),
     f"allgraph.model_sha256['{S_KEY}'] differs from --models"),
    ('a JSON file that is not a row file', lambda c: write_json(c / 'rows' / 'notes' / 'summary.json', {'graphs': 8}),
     'not a row file'),
    ('invalid JSON', lambda c: p(c, 'trix').write_text('{"rows": [', encoding='utf-8'), 'not valid JSON'),
    ('a broken link named *.json', broken_link, f'{G1}.json: cannot read'),
    ('seconds 0', lambda c: edit(p(c, 'trix'), lambda d: d.update(seconds=0)), 'seconds 0 is not a positive finite number'),
    ('seconds absent', lambda c: edit(p(c, 'trix'), lambda d: d.pop('seconds')), 'seconds None is not a positive finite number'),
    ('no rows', lambda c: edit(p(c, 'trix'), lambda d: d.update(rows=[])), 'no rows'),
    ('head not a bool', lambda c: edit(p(c, 'trix'), lambda d: d['rows'][0].update(head=1)),
     '1 rows without an int index >= 0 and a bool head'),
    ('index not an int', lambda c: edit(p(c, 'trix'), lambda d: d['rows'][0].update(index=3.0)),
     '1 rows without an int index >= 0 and a bool head'),
]


# --- the tests ------------------------------------------------------------------------------------------------------

def test_rank_keys() -> None:
    keys = ft.registered_models()
    assert keys[f'{S_KEY}.seed42:raw'] == (S_KEY, 42, S_KEY, True) and keys[f'{S_KEY}.seed44:raw'][:2] == (S_KEY, 44)
    assert keys['merit1024:raw'] == ('merit', 1024, 'merit1024', False) and keys['merit8000:raw'][:3] == ('merit', 8000, 'merit8000')
    for k in ('trix', 'ultra3g', 'flockoff', 'kgpfn', 'anyburl'):
        assert keys[f'{k}:raw'] == (k, 0, k, False), k
    for k in (f'{S_KEY}:raw', S_KEY, f'{S_KEY}.seed42', f'{S_KEY}.seed42:gt2', f'{S_KEY}.seed42:raw:gt2', f'{S_KEY}:raw:gt2',
              'merit:raw', 'merit1024', 'merit1024:raw:gt2', 'trix', 'trix:gt2', 'trix:raw:gt2'):
        assert k not in keys, k
    assert len(keys) == sum(len(seeds) for _, seeds in fa.SYSTEMS.values())
    ext = ft.extra_views()
    ours = [key for key, seeds in fa.SYSTEMS.values() if seeds == fa.OUR_SEEDS]
    assert all(ext[f'{k}:raw'] == (f'{k}+ens', 0) for k in ours) and len(ours) == 10, 'the ensemble of every family of ours'
    assert all(ext[f'{k}.seed{s}'] == (f'{k}+rule', s) for k in ours for s in fa.OUR_SEEDS)
    assert all(ext[f'merit{s}'] == ('merit+rule', s) for s in fa.MERIT_SEEDS)
    assert all(ext[k] == (f'{k}+rule', 0) for k in ('trix', 'ultra3g', 'flockoff', 'kgpfn', 'anyburl'))
    for k in (S_KEY, 'merit', f'{S_KEY}:gt2', f'{S_KEY}:raw:gt2', f'{S_KEY}.seed42:gt2', f'{S_KEY}.seed42:raw:gt2',
              'merit1024:gt2', 'trix:gt2', 'trix:raw:gt2'):
        assert k not in ext and k not in keys, k
    assert not set(ext) & set(keys) and len(ext) == sum(len(s) + (s == fa.OUR_SEEDS) for _, s in fa.SYSTEMS.values())
    assert not {system for system, _ in ext.values()} & {key for key, _ in fa.SYSTEMS.values()}
    print('PASS row keys: <key>.seed<S>:raw, merit<S>:raw and <key>:raw are the registered single models; the extra '
          'systems are <key>:raw of ours -> <key>+ens (seed 0), <key>.seed<S> -> <key>+rule (S), merit<S> -> merit+rule '
          '(S), a reference <key> -> <key>+rule (0); the GT2 views and the bare <key> of ours are ignored')


def test_happy_path(base: Path, world: dict, work: Path) -> Path:
    out = work / 'out-base'
    ok, msg = run(base, out)
    assert ok, msg
    truth, extra, listed = world['truth'], world['extra'], world['listed']
    mrr = fa.read_csv(out / 'mrr.csv')
    got = {(r['graph'], r['system'], int(r['seed'])): r for r in mrr}
    assert len(got) == len(mrr) and set(got) == set(truth) | set(extra), 'one mrr row per model and per extra with rows'
    assert list(got) == sorted(got), 'mrr.csv is sorted by graph, system and seed'
    systems = {r['system'] for r in mrr}
    assert {f'{S_KEY}+ens', f'{S_KEY}+rule', f'{fa.SYSTEMS["PV"][0]}+ens', 'merit+rule', 'trix+rule'} <= systems
    assert not {'kgpfn+rule', 'anyburl+rule'} & systems, 'a file without an extra key gives no extra rows'
    for k, ranks in {**truth, **extra}.items():
        vals = list(ranks.values())
        direct = math.fsum(1.0 / x for x in vals) / len(vals)
        exact = sum(Fraction(1, x) for x in vals) / len(vals)
        assert got[k]['mrr'] == repr(direct) and abs(float(got[k]['mrr']) - float(exact)) < 1e-15, k
        assert (got[k]['block'], got[k]['source']) == listed[k[0]][:2], k
    # ranks.csv: the cell systems on blocks 1a, 1b and 2; it joins back to mrr.csv exactly
    cell_keys = {fa.SYSTEMS[x][0] for x in fa.CELL_SYSTEMS}
    rank_rows = fa.read_csv(out / 'ranks.csv')
    by_model, order = defaultdict(dict), []
    for r in rank_rows:
        index, direction = r['query'].split(':')
        k = (r['graph'], r['system'], int(r['seed']))
        assert direction in ('h', 't') and index == str(int(index))
        by_model[k][(int(index), direction == 'h')] = int(r['rank'])
        order.append((*k, int(index), direction == 'h'))
    assert order == sorted(order) and len(set(order)) == len(order), 'ranks.csv is sorted and has no duplicate'
    want = {k for k in truth if k[1] in cell_keys and listed[k[0]][0] != '3'}
    assert set(by_model) == want and len(rank_rows) == len(want) * Q, 'ranks.csv: CELL_SYSTEMS on blocks 1a, 1b, 2'
    for k, ranks in by_model.items():
        vals = list(ranks.values())
        assert ranks == truth[k], k
        assert math.fsum(1.0 / x for x in vals) / len(vals) == float(got[k]['mrr']), k
        assert abs(sum(1.0 / x for x in vals) / len(vals) - float(got[k]['mrr'])) < 1e-12, k
    cost = {(r['graph'], r['system'], int(r['seed'])): float(r['seconds']) for r in fa.read_csv(out / 'cost.csv')}
    assert cost == world['cost'], 'cost: the seconds of the file divided by its registered models'
    meta = load(out / ft.META)
    files = sorted((x for x in (base / 'rows').rglob('*.json')
                    if not any(part.startswith('.') for part in x.relative_to(base / 'rows').parts)), key=str)
    assert [f['path'] for f in meta['inputs']['row_files']] == [str(x) for x in files], 'hidden paths are skipped'
    assert all(f['sha256'] == sha(Path(f['path'])) for f in meta['inputs']['row_files'])
    assert meta['outputs'] == {n: sha(out / n) for n in ft.HEADERS}
    assert meta['script_sha256'] == sha(Path(ft.__file__)) and meta['analysis_sha256'] == sha(Path(fa.__file__))
    assert all(meta['inputs'][k]['sha256'] == sha(base / f'{k}.json') for k in ('manifest', 'block3', 'models'))
    n_extra = sum(len(fa.SYSTEMS[x][1]) for x in EXTRA[G3])   # N on a block 3 graph: kept, not expected
    assert meta['missing'] == [] and meta['counts']['missing'] == 0
    assert meta['counts']['models'] == len(truth) and meta['counts']['expected'] == len(truth) - n_extra
    assert meta['counts']['extra_rows'] == len(extra) and meta['counts']['mrr_rows'] == len(truth) + len(extra)
    assert meta['extra_systems'] == sorted({k[1] for k in extra}) and meta['extra_incomplete'] == []
    assert {S_KEY, f'{S_KEY}:gt2', f'{S_KEY}.seed42:gt2', 'merit', 'merit1024:gt2', 'trix:raw:gt2'} <= set(meta['ignored_row_keys'])
    assert not {'index', 'head', 'trix:raw', 'trix', f'{S_KEY}:raw', f'{S_KEY}.seed42', f'{S_KEY}.seed42:raw',
                'merit1024'} & set(meta['ignored_row_keys'])
    assert meta['models_checked'] == sorted(load(base / 'models.json')['models']) and meta['models_unchecked'] == sorted(NO_HASH)
    print(f'PASS happy path: {len(files)} row files of {len(listed)} graphs (blocks 1a, 1b, 2, 3), our families with '
          f'three seeds, MERIT with five, the references, {len({k[1] for k in extra})} extra systems; mrr = a direct '
          f'computation (fsum and exact rational, < 1e-15); ranks.csv joins back to mrr.csv exactly; cost; extras in '
          f'mrr.csv only (not in ranks, cost, expected, missing); meta hashes; hidden paths and a log skipped')
    return out


def test_allowed(base: Path, work: Path, base_out: Path) -> None:
    """Variations that must not stop the run and must not change the tables."""
    want = outputs(base_out)
    same = lambda out: all((out / n).read_bytes() == want[n] for n in ft.HEADERS)  # noqa: E731
    c = work / 'allowed-ref'
    shutil.copytree(base, c)
    edit(p(c, 'trix'), lambda d: d['allgraph']['model_sha256'].pop('trix'))
    ok, msg = run(c, c / 'out')
    assert ok and same(c / 'out'), msg
    assert 'trix' in load(c / 'out' / ft.META)['models_unchecked'], 'a reference without an entry is listed as unchecked'
    c = work / 'allowed-nomodels'
    shutil.copytree(base, c)
    edit(p(c, 'ours-a'), lambda d: d['allgraph']['model_sha256'][S_KEY].update({'43': '0' * 64}))
    ok, msg = run(c, c / 'out', models=False)
    assert ok and same(c / 'out') and load(c / 'out' / ft.META)['models_checked'] == [], msg
    ok, msg = run(base, work / 'out-overlap', rows=[base / 'rows', base / 'rows' / 'trix', base / 'rows'])
    assert ok and same(work / 'out-overlap'), msg
    hidden = work / '.cache' / 'rows'
    shutil.copytree(base / 'rows', hidden)
    ok, msg = run(base, work / 'out-hidden-root', rows=[hidden])
    assert ok and same(work / 'out-hidden-root'), msg
    linked = work / 'linked' / 'rows'
    linked.mkdir(parents=True)
    (linked / 'all').symlink_to(base / 'rows', target_is_directory=True)
    (linked / 'loop').symlink_to(linked, target_is_directory=True)
    ok, msg = run(base, work / 'out-linked', rows=[linked])
    assert ok and same(work / 'out-linked'), msg
    n = load(base_out / ft.META)['counts']['row_files']
    assert load(work / 'out-linked' / ft.META)['counts']['row_files'] == n == load(work / 'out-overlap' / ft.META)['counts']['row_files']
    c = work / 'allowed-rule-file'
    shutil.copytree(base, c)
    rule_only(c)
    ok, msg = run(c, c / 'out')
    assert ok and same(c / 'out'), msg
    # an incomplete seed set of a multi-seed extra is left out, and listed: fv1_analysis would refuse the whole table
    c = work / 'allowed-incomplete'
    shutil.copytree(base, c)
    rule = f'{S_KEY}+rule'
    edit(p(c, 'ours-a'), lambda d: [r.pop(f'{S_KEY}.seed44') for r in d['rows']])
    ok, msg = run(c, c / 'out')
    assert ok, msg
    meta = load(c / 'out' / ft.META)
    assert meta['extra_incomplete'] == [dict(graph=G1, system=rule, seeds=[42, 43], needs=list(fa.OUR_SEEDS))]
    rows = fa.read_csv(c / 'out' / 'mrr.csv')
    assert not [r for r in rows if r['graph'] == G1 and r['system'] == rule]
    assert [r for r in rows if r['system'] == rule] == [r for r in fa.read_csv(base_out / 'mrr.csv')
                                                       if r['system'] == rule and r['graph'] != G1]
    assert all((c / 'out' / n).read_bytes() == want[n] for n in ('ranks.csv', 'cost.csv'))
    fa.Table(rows, {})
    partial = rows + [dict(graph=G1, block='1a', source='SRC-A', system=rule, seed=str(s), mrr='0.25') for s in (42, 43)]
    try:
        fa.Table(partial, {})
        refused = ''
    except ValueError as e:
        refused = str(e)
    assert f'unregistered system {rule}: its seeds cover different graphs' in refused, refused
    print('PASS allowed: a reference without a model_sha256 entry, no --models, overlapping --rows (each file once), '
          'a --rows folder below a hidden folder, a linked folder, a link loop and an extras-only file give the same '
          'tables; an incomplete extra seed set is left out and listed (fv1_analysis refuses it otherwise)')


def test_stops(base: Path, work: Path) -> None:
    for n, (case, change, fragment) in enumerate(STOPS):
        c = work / f'stop-{n:02d}'
        shutil.copytree(base, c)
        change(c)
        ok, msg = run(c, c / 'out')
        assert not ok and fragment in msg, (case, msg)
        assert msg.startswith('fv1_tables: ') and 'nothing written' in msg, (case, msg)
        assert not (c / 'out').exists(), f'{case}: an output was written'
        shutil.rmtree(c)
    ok, msg = run(base, base / 'rows' / 'tables')
    assert not ok and 'lies inside --rows' in msg and not (base / 'rows' / 'tables').exists(), msg
    print(f'PASS stop conditions ({len(STOPS) + 1} cases, nothing written): unlisted graph, key and file name, file '
          f'hashes, rows per model, duplicate pairs, one-direction facts, pair sets across models (blocks 1-2 and 3), '
          f'2 x ranked_facts, one model in two files, ranks 0, -3, 2.0, true, "3", null, a missing rank; the extra keys '
          f'(a missing rule rank, an ensemble rank 0, a reference rule rank 2.5, a MERIT rule rank null, an extras-only '
          f'file with other pairs or too few rows, one extra in two files); model hashes (ours, MERIT, absent entry or '
          f'seed, unused entry), non-row JSON, invalid JSON, a broken link, seconds, empty or malformed rows, out folder '
          f'inside --rows')


def test_determinism(base: Path, work: Path, base_out: Path) -> None:
    want = outputs(base_out)
    real_row_files = ft.row_files
    for seed in range(3):
        def shuffled(roots, seed=seed):
            paths = real_row_files(roots)
            random.Random(seed).shuffle(paths)
            return paths
        ft.row_files = shuffled
        try:
            ok, msg = run(base, work / f'out-perm{seed}')
        finally:
            ft.row_files = real_row_files
        assert ok and outputs(work / f'out-perm{seed}') == want, f'file order {seed}: {msg}'
    real_walk, rng = ft.os.walk, random.Random(11)

    def walk(top, **kw):
        for dirpath, dirnames, filenames in real_walk(top, **kw):
            rng.shuffle(dirnames)
            rng.shuffle(filenames)
            yield dirpath, dirnames, filenames
    ft.os.walk = walk
    try:
        ok, msg = run(base, work / 'out-walk')
    finally:
        ft.os.walk = real_walk
    assert ok and outputs(work / 'out-walk') == want, f'listing order: {msg}'
    c = work / 'shuffled-rows'
    shutil.copytree(base, c)
    ok, msg = run(c, c / 'out-a')
    assert ok, msg
    shuffle_rows(c, 3)
    ok, msg = run(c, c / 'out-b')
    assert ok, msg
    a, b = outputs(c / 'out-a'), outputs(c / 'out-b')
    assert all(a[n] == b[n] == want[n] for n in ft.HEADERS), 'row order changes a table'
    ma, mb = load(c / 'out-a' / ft.META), load(c / 'out-b' / ft.META)
    assert all(x['sha256'] != y['sha256'] for x, y in zip(ma['inputs']['row_files'], mb['inputs']['row_files']))
    for m in (ma, mb):
        for f in m['inputs']['row_files']:
            f.pop('sha256')
    assert ma == mb, 'row order changes the meta beyond the row-file hashes'
    print('PASS determinism: three permutations of the file list and a shuffled folder listing give byte-identical '
          'outputs (meta included); shuffled rows and keys give byte-identical tables and a meta that differs only in '
          'the row-file hashes')


def test_missing(base: Path, work: Path) -> None:
    c = work / 'missing'
    shutil.copytree(base, c)
    kgpfn, merit = fa.SYSTEMS['KGPFN'][0], fa.SYSTEMS['MERIT'][0]
    p(c, 'kgpfn', 'conv-d1').unlink()
    p(c, 'merit', 'lb3-y').unlink()
    edit(p(c, 'ours-a', 'conv-e1'), lambda d: [r.pop(f'{S_KEY}.seed44:raw') for r in d['rows']])
    ok, msg = run(c, c / 'out')
    assert ok, msg
    meta = load(c / 'out' / ft.META)
    want = ([dict(graph='conv-d1', block='2', system=kgpfn, seed=0), dict(graph='conv-e1', block='2', system=S_KEY, seed=44)]
            + [dict(graph='lb3-y', block='3', system=merit, seed=s) for s in fa.MERIT_SEEDS])
    assert meta['missing'] == sorted(want, key=lambda m: (m['graph'], m['system'], m['seed'])), meta['missing']
    assert meta['counts']['missing'] == len(want) and meta['counts']['graphs_with_missing'] == 3
    rows = fa.read_csv(c / 'out' / 'mrr.csv')
    try:
        fa.Table(rows, {})
        refused = False
    except ValueError:
        refused = True
    assert refused, 'the analysis refuses a table with missing models'
    label_of = {key: label for label, (key, _) in fa.SYSTEMS.items()}
    drops = {}
    for m in meta['missing']:   # option B: {graph: {system label: reason}}
        drops.setdefault(m['graph'], {})[label_of[m['system']]] = 'synthetic: failed on every GPU'
    table = fa.Table(rows, drops)
    table.check_manifest(load(c / 'manifest.json'))
    conf = [g for g in table.graphs if table.block[g] in fa.CONFIRMATORY]
    write_queries(c / 'queries.csv', c / 'out' / 'ranks.csv')
    fa.Cells(fa.read_csv(c / 'queries.csv'), fa.read_csv(c / 'out' / 'ranks.csv'), table, conf)
    print('PASS missing: an absent file (block 2 and block 3) and an absent seed are listed exactly; N on a block 3 '
          'graph is kept and never required; fv1_analysis refuses the table, and accepts it with these graphs dropped')


def test_analysis(world: dict, out: Path, work: Path) -> None:
    table = fa.Table(fa.read_csv(out / 'mrr.csv'), {})
    table.check_manifest(world['manifest'])
    conf = [g for g in table.graphs if table.block[g] in fa.CONFIRMATORY]
    assert conf == sorted(GRAPHS) and sorted(set(table.graphs) - set(conf)) == sorted(BLOCK3)
    assert all((table.block[g], table.source[g]) == world['listed'][g][:2] for g in table.graphs)
    extra_systems = {k[1] for k in world['extra']}
    assert set(table.extra) == extra_systems, 'the extra systems are unregistered systems of the table'
    value = {(r['graph'], r['system'], int(r['seed'])): float(r['mrr']) for r in fa.read_csv(out / 'mrr.csv')}
    assert all(table.values[label][seed][g] == value[(g, key, seed)]
               for label, (key, seeds) in fa.SYSTEMS.items() for seed in seeds for g in conf), 'registered values unchanged'
    levels = fa.levels_table(table, conf)
    assert {k for k in levels if k.startswith('unregistered:')} == {f'unregistered:{s}' for s in extra_systems}
    assert set(levels) - {f'unregistered:{s}' for s in extra_systems} == set(fa.SYSTEMS)
    assert all(levels[f'unregistered:{s}']['note'] == 'descriptive only' for s in extra_systems)
    write_queries(work / 'queries.csv', out / 'ranks.csv')
    cells = fa.Cells(fa.read_csv(work / 'queries.csv'), fa.read_csv(out / 'ranks.csv'), table, conf)
    assert len(cells.rr) == len(conf) * sum(len(fa.SYSTEMS[x][1]) for x in fa.CELL_SYSTEMS)
    assert all(abs(float(v.mean()) - table.values[label][seed][g]) < 1e-12 for (g, label, seed), v in cells.rr.items())
    cost = fa.cost_table(fa.read_csv(out / 'cost.csv'), conf)
    assert cost['per_system'][fa.SYSTEMS['M'][0]]['graphs'] == len(conf) and cost['flock_over_m']['graphs'] == len(conf)
    assert set(cost['per_system']) == {key for key, _ in fa.SYSTEMS.values()}
    print('PASS analysis input: fv1_analysis.Table (with check_manifest), Cells (the ranks reproduce the MRR within '
          '1e-12) and cost_table read mrr.csv, ranks.csv and cost.csv without an input error; the extra systems land '
          'in Table.extra and reach only the descriptive levels (levels_table, "descriptive only")')


def test_full_size(work: Path) -> None:
    """The registered size: QUERIES_PER_GRAPH = 2000 per confirmatory graph, a block 3 graph with 1,000 facts."""
    assert fa.QUERIES_PER_GRAPH == 2000
    root = work / 'full'
    world = build_tree(root, {'pub-a1': ('1a', 'SRC-A'), 'conv-d1': ('2', 'SRC-D')}, {'lb3-x': ('dbpedia', 1000)},
                       fa.QUERIES_PER_GRAPH)
    t0 = time.time()
    ok, msg = run(root, root / 'out')
    seconds = time.time() - t0
    assert ok, msg
    rows = fa.read_csv(root / 'out' / 'mrr.csv')
    every = {**world['truth'], **world['extra']}
    assert len(rows) == len(every) and len(world['extra']) > 0
    for r in rows:
        vals = list(every[(r['graph'], r['system'], int(r['seed']))].values())
        assert len(vals) == fa.QUERIES_PER_GRAPH and r['mrr'] == repr(math.fsum(1.0 / x for x in vals) / len(vals))
    table = fa.Table(rows, {})
    table.check_manifest(world['manifest'])
    conf = [g for g in table.graphs if table.block[g] in fa.CONFIRMATORY]
    write_queries(root / 'queries.csv', root / 'out' / 'ranks.csv')
    fa.Cells(fa.read_csv(root / 'queries.csv'), fa.read_csv(root / 'out' / 'ranks.csv'), table, conf)
    edit(p(root, 'trix', 'conv-d1'), drop_fact)
    ok, msg = run(root, root / 'out-short')
    assert not ok and f'{fa.QUERIES_PER_GRAPH - 2} rows per model, blocks 1a, 1b and 2 need {fa.QUERIES_PER_GRAPH}' in msg, msg
    print(f'PASS registered size: 2000 queries per confirmatory graph and 2000 on block 3; MRR exact (models and '
          f'extras), Table and Cells accept, a model with 1998 rows stops the run ({seconds:.1f} s for {len(rows)} '
          f'mrr rows)')


def main() -> int:
    ap = argparse.ArgumentParser(description='Tests of fv1_tables.py on synthetic row files')
    ap.add_argument('--tmp', type=Path, default=Path(tempfile.gettempdir()), help='the parent of the test folder')
    a = ap.parse_args()
    created = not a.tmp.exists()
    a.tmp.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='fv1-tables-', dir=a.tmp))
    saved, t0 = fa.QUERIES_PER_GRAPH, time.time()
    try:
        fa.QUERIES_PER_GRAPH = Q
        test_rank_keys()
        base = work / 'base'
        world = build_tree(base, GRAPHS, BLOCK3, Q)
        out = test_happy_path(base, world, work)
        test_allowed(base, work, out)
        test_stops(base, work)
        test_determinism(base, work, out)
        test_missing(base, work)
        test_analysis(world, out, work)
        fa.QUERIES_PER_GRAPH = saved
        test_full_size(work)
    finally:
        fa.QUERIES_PER_GRAPH = saved
        shutil.rmtree(work, ignore_errors=True)
        if created:
            with contextlib.suppress(OSError):
                a.tmp.rmdir()
    print(f'PASS all checks ({time.time() - t0:.1f} s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
