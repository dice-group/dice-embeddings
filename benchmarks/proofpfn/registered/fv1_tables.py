"""fresh-v1: the table extraction, harness row files -> the inputs of fv1_analysis.py (registration, sections 5 and 15).

DRAFT, NOT FROZEN. Run order step 6. CPU only, standard library only; SYSTEMS and the protocol constants come from
fv1_analysis.py. The script reads no graph file and loads no model. Its messages print no MRR and no valid rank. The
outputs depend only on the content of the inputs: not on the order in which the file system lists the row files, and
not on the order of the rows inside a file.

Input
  --rows DIR    a folder of row files, searched recursively (links to folders are followed); repeat it for more
                folders. Every *.json below DIR is a row file, except a path with a part below DIR that starts with
                '.' (for example .scratch/), which is skipped. Other files (logs) are ignored. A file reached twice
                (overlapping folders, links) counts once.
  --manifest    KGs/fresh-v1/MANIFEST.json: {"graphs": {name: {"block": "1a" | "1b" | "2", "source": str,
                "sha256": {"train": hex, "valid": hex, "test": hex}, ...}}, ...}
  --block3      the block 3 list: {"graphs": {name: {"cluster": str, "files": {"train.txt": hex, "valid.txt": hex,
                "test.txt": hex}, ...}}, ...}. A block 3 graph gets block 3 and source = its cluster.
  --models      optional (K/fresh-v1/models.json): {"models": {hash key: the dict that the harness writes into
                allgraph.model_sha256 for that key}, ...}; for example {"42": hex, "43": hex, "44": hex} for a family
                of ours and {"model": hex} for a reference. Only "models" is read.
  --out-dir     the output folder. It may not lie inside a --rows folder.

A row file <graph>.json (one per graph and runner job) holds
  rows       [{"index": int, "head": bool, <rank key>: int, ...}]. index: the position of the test fact in test.txt
             (0-based, as in fv1_cells.py). head: True for the head query (the anchor is the tail). A rank is the
             pessimistic filtered rank (>= 1).
  seconds    the wall seconds of the whole job (all models of the file)
  allgraph   {"key": ".../<graph>", "sha256": {"train.txt": hex, "valid.txt": hex, "test.txt": hex, ...},
              "model_sha256": {hash key: {...}}, "ranked_facts": int, ...}
The registered single models follow from SYSTEMS (raw view: fair_eval without the train-only self-loop rule):
  kind                             rank key            system, seed       hash key
  our families (seeds OUR_SEEDS)   <key>.seed<S>:raw   <key>, S           <key> (one entry for the three seeds)
  MERIT (seeds MERIT_SEEDS)        <key><S>:raw        <key>, S           <key><S> (merit1024, ...)
  a single checkpoint (seed 0)     <key>:raw           <key>, 0           <key> (trix, ultra3g, ...)
The descriptive systems of registration section 10 (extra systems, outside SYSTEMS) follow from SYSTEMS too:
  the seed ensemble of a family of ours (the 3-seed score ensemble, raw)   <key>:raw       -> <key>+ens, seed 0
  the rule view (train-only self-loop rule) of a model of ours             <key>.seed<S>   -> <key>+rule, seed S
  the rule view of a MERIT checkpoint                                      <key><S>        -> <key>+rule, seed S
  the rule view of a single checkpoint                                     <key>           -> <key>+rule, seed 0
Every other key of a row is ignored: the GT2 views and the rule view <key> of the seed ensembles of ours.

Output (--out-dir; written only when every check passes)
  mrr.csv           graph, block, source, system, seed, mrr. One row per graph and registered single model with rows,
                    and per graph and extra system (seed) with rows. mrr: the mean of 1/rank over the rows (math.fsum,
                    so independent of the row order), written with repr(). Sorted by graph, system, seed.
                    fv1_analysis uses an extra system only in its descriptive levels, and refuses the whole table if
                    the seeds of an extra system cover different graphs. So an extra system with several seeds (the
                    rule view of ours and of MERIT) is written for a graph only with all its seeds; an incomplete set
                    is left out and listed in the meta (extra_incomplete). A file without extra keys gives no extra rows.
  ranks.csv         graph, query, system, seed, rank: the systems of CELL_SYSTEMS on blocks 1a, 1b and 2. query is
                    <index>:h for a head query and <index>:t for a tail query (as in fv1_cells.py). Sorted by graph,
                    system, seed, index, tail before head.
  cost.csv          graph, system, seed, seconds: the file's seconds divided by the number of registered single models
                    in the file. This is the job's wall time shared equally among its models, not a per-model
                    measurement. Sorted by graph, system, seed.
  tables.meta.json  counts; the path and SHA-256 of every input, with every row file; the SHA-256 of every output, of
                    this script and of fv1_analysis.py; the ignored row keys; the extra systems written and the
                    incomplete ones left out; the hash keys of the models whose hash was compared with --models
                    (models_checked) or not (models_unchecked); and missing: every (graph, system, seed) of the expected
                    grid without rows (every manifest graph x SYSTEMS x seeds, every block 3 graph x BLOCK3_SYSTEMS x
                    seeds). A missing model does not stop the script. The analysis refuses it unless dropped.json covers
                    its graph (registration, section 14). Written last.
Extra systems never enter the expected grid, missing, ranks.csv or cost.csv.

The script stops (SystemExit with the list of problems; nothing is written) on
  - a *.json that is not a row file; a graph in neither list; allgraph.key not ending in the file name; a SHA-256 of
    train.txt, valid.txt or test.txt that differs from the manifest or the block 3 list (or is absent);
  - for the keys of the registered models and of the extra systems: a rank that is not an int >= 1; a key that a file
    holds on some rows and not on others; duplicate (index, head) pairs; a test fact without both directions; on
    blocks 1a, 1b and 2 a key without exactly QUERIES_PER_GRAPH rows; on block 3 a key without exactly
    2 x allgraph.ranked_facts rows; two keys of one graph with different (index, head) sets;
  - seconds that is not a positive number (in a file with a registered model);
  - the same (graph, system, seed) in two files, even with equal values;
  - with --models: an allgraph.model_sha256 entry of a listed hash key that differs from --models, or a model of our
    families (seeds 42 to 44) whose listed hash key is absent from the model_sha256 of its file. A reference without
    an entry is allowed (kgpfn and anyburl have none).

Usage (repository root):
  /home/lukef/miniconda3/bin/python -B Experiments/synthetic-kg/fresh-v1-runtime/fv1_tables.py --rows ROWS \\
      [--rows ROWS ...] --manifest KGs/fresh-v1/MANIFEST.json --block3 Experiments/synthetic-kg/fresh-v1/block3.json \\
      --models Experiments/synthetic-kg/fresh-v1/models.json --out-dir TABLES
Tests: test_fv1_tables.py.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fv1_analysis as fa  # noqa: E402

VERSION = 'fv1-tables draft 2 (29 October 2026, with the extra systems of section 10)'
SPLIT_FILES = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}   # manifest key -> file name
HEADERS = {'mrr.csv': ('graph', 'block', 'source', 'system', 'seed', 'mrr'),
           'ranks.csv': ('graph', 'query', 'system', 'seed', 'rank'),
           'cost.csv': ('graph', 'system', 'seed', 'seconds')}
META = 'tables.meta.json'
SHOW = 40   # problems listed in a stop message


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stop(problems: list, what: str) -> None:
    """Raise SystemExit with the problems, if there are any. Nothing has been written at that point."""
    if problems:
        more = f'\n  ... {len(problems) - SHOW} more' if len(problems) > SHOW else ''
        raise SystemExit(f'fv1_tables: {what}, nothing written ({len(problems)} problems):\n  '
                         + '\n  '.join(problems[:SHOW]) + more)


def is_int(value) -> bool:
    """An int that is not a bool. JSON true is a bool and 2.0 a float in Python: neither counts."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_rank(value) -> bool:
    return is_int(value) and value >= 1


def seed_form(label: str, seeds: tuple) -> str:
    """'ours', 'merit' or 'single': the form of the row keys of a system of SYSTEMS."""
    if seeds == tuple(fa.OUR_SEEDS):
        return 'ours'
    if seeds == tuple(fa.MERIT_SEEDS):
        return 'merit'
    if seeds == (0,):
        return 'single'
    raise SystemExit(f'fv1_tables: SYSTEMS[{label!r}] has seeds {seeds}, for which no row key is defined')


def registered_models() -> dict:
    """{rank key: (system, seed, hash key, ours)} of every registered single model (the raw view).

    hash key: the key of the model in allgraph.model_sha256 and in --models. ours: a family of ours (seeds OUR_SEEDS).
    """
    out = {}
    for label, (key, seeds) in fa.SYSTEMS.items():
        form = seed_form(label, tuple(seeds))
        for seed in seeds:
            rank, hkey = {'ours': (f'{key}.seed{seed}:raw', key), 'merit': (f'{key}{seed}:raw', f'{key}{seed}'),
                          'single': (f'{key}:raw', key)}[form]
            if rank in out:
                raise SystemExit(f'fv1_tables: two models of SYSTEMS share the rank key {rank}')
            out[rank] = (key, seed, hkey, form == 'ours')
    return out


def extra_views() -> dict:
    """{row key: (extra system, seed)} of the descriptive systems of registration section 10 (outside SYSTEMS)."""
    out = {}
    for label, (key, seeds) in fa.SYSTEMS.items():
        form = seed_form(label, tuple(seeds))
        if form == 'ours':
            out[f'{key}:raw'] = (f'{key}+ens', 0)                      # the score ensemble of the three seeds
            out.update({f'{key}.seed{s}': (f'{key}+rule', s) for s in seeds})
        elif form == 'merit':
            out.update({f'{key}{s}': (f'{key}+rule', s) for s in seeds})
        else:
            out[key] = (f'{key}+rule', 0)
    clash = sorted(set(out) & set(registered_models())) + sorted({s for s, _ in out.values()} & {k for k, _ in fa.SYSTEMS.values()})
    if clash or len(set(out.values())) != len(out):
        raise SystemExit(f'fv1_tables: the extra views clash with the registered models or with each other {clash}')
    return out


def graph_list(manifest: dict, block3: dict) -> dict:
    """{graph: dict(block, source, sha256={file name: hex}, list)} of the manifest (blocks 1a, 1b, 2) and block 3."""
    problems, out = [], {}
    for which, doc in (('manifest', manifest), ('block 3 list', block3)):
        listed = doc.get('graphs') if isinstance(doc, dict) else None
        if not isinstance(listed, dict):
            problems.append(f'{which}: no "graphs" object')
            continue
        for g, info in sorted(listed.items()):
            try:
                if which == 'manifest':
                    block, source = str(info['block']), info['source']
                    sha = {f: info['sha256'][k] for k, f in SPLIT_FILES.items()}
                else:
                    block, source = '3', info['cluster']
                    sha = {f: info['files'][f] for f in SPLIT_FILES.values()}
            except (KeyError, TypeError) as e:
                problems.append(f'{which} {g}: unreadable entry ({e!r})')
                continue
            if which == 'manifest' and block not in fa.CONFIRMATORY:
                problems.append(f'manifest {g}: block {block!r} is not one of {fa.CONFIRMATORY}')
            if not isinstance(source, str) or not source or source != source.strip():
                problems.append(f'{which} {g}: source {source!r} is not a non-empty string without outer spaces')
            if not all(isinstance(v, str) and v for v in sha.values()):
                problems.append(f'{which} {g}: a file SHA-256 is not a string')
            if g in out:
                problems.append(f'{g}: in the manifest and in the block 3 list')
            out[g] = dict(block=block, source=source, sha256=sha, list=which)
    stop(problems, 'graph lists refused')
    return out


def models_list(doc) -> dict:
    """The "models" object of --models: {hash key: {str: str}}."""
    models = doc.get('models') if isinstance(doc, dict) else None
    if not isinstance(models, dict) or not all(isinstance(v, dict) and all(isinstance(x, str) for x in v.values())
                                               for v in models.values()):
        raise SystemExit('fv1_tables: --models needs an object "models" of {hash key: {seed or "model": sha256}}')
    return models


def row_files(roots: list) -> list:
    """Every *.json below the folders, sorted by path, without the paths that have a part starting with '.' below their
    folder. A file reached by two paths (overlapping folders, links) appears once, under the path that sorts first."""
    def refuse(err: OSError) -> None:
        raise SystemExit(f'fv1_tables: cannot list {err.filename}: {err.strerror}')

    found = []
    for root in map(Path, roots):
        if not root.is_dir():
            raise SystemExit(f'fv1_tables: --rows {root} is not a folder')
        visited = set()
        for dirpath, dirnames, filenames in os.walk(root, onerror=refuse, followlinks=True):
            real = os.path.realpath(dirpath)
            if real in visited or any(p.startswith('.') for p in Path(dirpath).relative_to(root).parts):
                dirnames[:] = []   # a link loop, or a hidden folder
                continue
            visited.add(real)
            dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
            found += [Path(dirpath) / f for f in filenames if f.endswith('.json') and not f.startswith('.')]
    unique = {}
    for path in sorted(found, key=str):
        unique.setdefault(os.path.realpath(path), path)
    return sorted(unique.values(), key=str)


def hash_problems(path: Path, hashes, held: list, models: Optional[dict]) -> tuple:
    """Compare allgraph.model_sha256 with --models. held: (hash key, ours) of the registered models of the file.

    Returns (problems, checked hash keys, unchecked hash keys); the last two cover the models of the file.
    """
    if hashes is None:
        hashes = {}
    if not isinstance(hashes, dict):
        return [f'{path}: allgraph.model_sha256 is not an object'], set(), set()
    problems, checked, unchecked = [], set(), set()
    for hkey in sorted(models or {}):
        if hkey in hashes and hashes[hkey] != models[hkey]:
            got, want = hashes[hkey], models[hkey]
            parts = sorted(set(got) | set(want), key=str) if isinstance(got, dict) else ['the whole entry']
            parts = [p for p in parts if not isinstance(got, dict) or got.get(p) != want.get(p)]
            problems.append(f'{path}: allgraph.model_sha256[{hkey!r}] differs from --models ({", ".join(map(str, parts))})')
    for hkey, ours in sorted(set(held)):
        if models is not None and hkey in models and hkey in hashes:
            checked.add(hkey)
        elif models is not None and hkey in models and ours:
            problems.append(f'{path}: allgraph.model_sha256 has no entry {hkey!r}, whose models the file holds')
        else:
            unchecked.add(hkey)
    return problems, checked, unchecked


def read_row_file(path: Path, graphs: dict, keys: dict, extras: dict, models: Optional[dict]) -> tuple:
    """Check one row file. Returns (record, problems); record is None when the file has a problem.

    record['models'] maps (system, seed) of every registered single model of the file to dict(mrr, seconds, ranks);
    ranks ({(index, head): rank}) is kept only for CELL_SYSTEMS on blocks 1a, 1b and 2. record['extras'] maps (extra
    system, seed) to dict(mrr).
    """
    try:
        raw = path.read_bytes()
    except OSError as e:
        return None, [f'{path}: cannot read ({e.strerror})']
    try:
        data = json.loads(raw)
    except ValueError as e:
        return None, [f'{path}: not valid JSON ({e})']
    if not isinstance(data, dict) or not isinstance(data.get('rows'), list) or not isinstance(data.get('allgraph'), dict):
        return None, [f'{path}: not a row file (no list "rows" and object "allgraph")']
    ag, rows, g = data['allgraph'], data['rows'], path.stem
    key = ag.get('key')
    if not isinstance(key, str) or key.rsplit('/', 1)[-1] != g:
        return None, [f'{path}: allgraph.key {key!r} does not end in the file name {g!r}']
    info = graphs.get(g)
    if info is None:
        return None, [f'{path}: graph {g!r} is in neither the manifest nor the block 3 list']
    problems = []
    sha = ag.get('sha256') if isinstance(ag.get('sha256'), dict) else {}
    for name, want in sorted(info['sha256'].items()):
        if sha.get(name) != want:
            problems.append(f'{path}: the SHA-256 of {name} differs from the {info["list"]} (or is absent)')
    if not rows:
        return None, problems + [f'{path}: no rows']
    malformed = [i for i, r in enumerate(rows) if not (isinstance(r, dict) and is_int(r.get('index'))
                                                         and r['index'] >= 0 and isinstance(r.get('head'), bool))]
    if malformed:
        return None, problems + [f'{path}: {len(malformed)} rows without an int index >= 0 and a bool head '
                                 f'(first: row {malformed[0]})']
    held, ignored = defaultdict(int), set()   # held: rows that hold each used key (a registered model or an extra view)
    for r in rows:
        for k in r:
            if k in keys or k in extras:
                held[k] += 1
            elif k not in ('index', 'head'):
                ignored.add(k)
    registered = [k for k in held if k in keys]
    probs, checked, unchecked = hash_problems(path, ag.get('model_sha256'), [keys[k][2:] for k in registered], models)
    problems += probs
    rec = dict(path=str(path), sha256=sha256_bytes(raw), graph=g, rows=len(rows), models={}, extras={}, queries=None,
               ignored=ignored, checked=checked, unchecked=unchecked)
    if not held:
        return (None if problems else rec), problems
    for k in sorted(held):
        if held[k] != len(rows):
            problems.append(f'{path}: rank key {k} is missing on {len(rows) - held[k]} of {len(rows)} rows')
        bad = [r[k] for r in rows if k in r and not is_rank(r[k])]
        if bad:
            problems.append(f'{path}: rank key {k}: {len(bad)} of {len(rows)} ranks are not an int >= 1 '
                            f'(first: {bad[0]!r})')
    pairs = [(r['index'], r['head']) for r in rows]
    queries = frozenset(pairs)
    if len(queries) != len(pairs):
        problems.append(f'{path}: {len(pairs) - len(queries)} duplicate (index, head) pairs')
    one_way = sorted(i for i, h in queries if (i, not h) not in queries)
    if one_way:
        problems.append(f'{path}: {len(one_way)} test facts ranked in one direction only (first: index {one_way[0]})')
    if info['block'] == '3':
        facts = ag.get('ranked_facts')
        if not is_int(facts) or len(queries) != 2 * facts:
            problems.append(f'{path}: {len(queries)} queries per model, block 3 needs 2 x allgraph.ranked_facts '
                            f'({facts!r})')
    elif len(rows) != fa.QUERIES_PER_GRAPH:
        problems.append(f'{path}: {len(rows)} rows per model, blocks 1a, 1b and 2 need {fa.QUERIES_PER_GRAPH}')
    seconds = data.get('seconds')
    if registered and (not (is_int(seconds) or isinstance(seconds, float)) or not math.isfinite(seconds) or seconds <= 0):
        problems.append(f'{path}: seconds {seconds!r} is not a positive finite number')
    if problems:
        return None, problems
    cells = {fa.SYSTEMS[label][0] for label in fa.CELL_SYSTEMS} if info['block'] in fa.CONFIRMATORY else set()
    rec['queries'] = queries
    for k in held:
        mrr = math.fsum(1.0 / r[k] for r in rows) / len(rows)
        if k in keys:
            system, seed = keys[k][:2]
            rec['models'][(system, seed)] = dict(mrr=mrr, seconds=seconds / len(registered),
                                                 ranks={(r['index'], r['head']): r[k] for r in rows} if system in cells else None)
        else:
            rec['extras'][extras[k]] = dict(mrr=mrr)
    return rec, problems


def build_tables(paths: list, graphs: dict, models: Optional[dict] = None) -> dict:
    """Read and check every row file, then the files of each graph against each other. Returns the output rows."""
    keys, extras = registered_models(), extra_views()
    seeds_of = defaultdict(set)   # the seeds of each extra system
    for system, seed in extras.values():
        seeds_of[system].add(seed)
    problems, files, first, ignored, checked, unchecked = [], [], {}, set(), set(), set()
    found, found_extra = {}, {}
    for path in paths:
        rec, probs = read_row_file(Path(path), graphs, keys, extras, models)
        problems += probs
        if rec is None:
            continue
        files.append(dict(path=rec['path'], sha256=rec['sha256'], graph=rec['graph'], rows=rec['rows'],
                          models=len(rec['models']), extras=len(rec['extras'])))
        ignored |= rec['ignored']
        checked |= rec['checked']
        unchecked |= rec['unchecked']
        g = rec['graph']
        if rec['models'] or rec['extras']:
            ref_path, ref = first.setdefault(g, (rec['path'], rec['queries']))
            if rec['queries'] != ref:
                problems.append(f'{g}: {rec["path"]} ranks other (index, head) pairs than {ref_path} '
                                f'({len(rec["queries"] ^ ref)} pairs differ)')
        for store, items in ((found, rec['models']), (found_extra, rec['extras'])):
            for (system, seed), m in items.items():
                if (g, system, seed) in store:
                    problems.append(f'{g} {system} seed {seed}: in {store[(g, system, seed)]["path"]} and in {rec["path"]}')
                store[(g, system, seed)] = dict(m, path=rec['path'])
    stop(problems, 'row files refused')

    # an extra system with several seeds is written for a graph only with all its seeds (fv1_analysis.Table)
    have = defaultdict(set)
    for g, system, seed in found_extra:
        have[(g, system)].add(seed)
    incomplete = [dict(graph=g, system=system, seeds=sorted(s), needs=sorted(seeds_of[system]))
                  for (g, system), s in sorted(have.items()) if s != seeds_of[system]]
    extra_rows = {k: m for k, m in found_extra.items() if have[k[:2]] == seeds_of[k[1]]}

    out = {name: [] for name in HEADERS}
    for k in sorted(set(found) | set(extra_rows)):
        g, system, seed = k
        m, info = found.get(k) or extra_rows[k], graphs[g]
        out['mrr.csv'].append([g, info['block'], info['source'], system, seed, repr(m['mrr'])])
        if k in found:
            out['cost.csv'].append([g, system, seed, repr(m['seconds'])])
            if m['ranks'] is not None:
                out['ranks.csv'] += [[g, f'{i}:{"h" if h else "t"}', system, seed, rank]
                                     for (i, h), rank in sorted(m['ranks'].items())]
    expected = sorted((g, key, seed) for g, info in graphs.items()
                      for label in (fa.BLOCK3_SYSTEMS if info['block'] == '3' else fa.SYSTEMS)
                      for key, seeds in (fa.SYSTEMS[label],) for seed in seeds)
    missing = [dict(graph=g, block=graphs[g]['block'], system=system, seed=seed) for g, system, seed in expected
               if (g, system, seed) not in found]
    with_rows = sorted({g for g, _, _ in found} | {g for g, _, _ in extra_rows})
    by_block = defaultdict(int)
    for g in with_rows:
        by_block[graphs[g]['block']] += 1
    out.update(files=sorted(files, key=lambda f: f['path']), missing=missing, ignored=sorted(ignored),
               checked=sorted(checked), unchecked=sorted(unchecked), incomplete=incomplete,
               extra_systems=sorted({system for _, system, _ in extra_rows}),
               counts=dict(row_files=len(files), row_files_without_models=sum(1 for f in files if not f['models']),
                           graphs_listed=len(graphs), graphs_with_rows=len(with_rows), graphs_by_block=dict(by_block),
                           models=len(found), expected=len(expected), missing=len(missing),
                           graphs_with_missing=len({m['graph'] for m in missing}), extra_rows=len(extra_rows),
                           extra_incomplete=len(incomplete),
                           **{name.replace('.csv', '_rows'): len(out[name]) for name in HEADERS}))
    return out


def csv_bytes(header: tuple, rows: list) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode('utf-8')


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def write_tables(out_dir: Path, tables: dict, inputs: dict) -> dict:
    """Write the three CSV files, then tables.meta.json (last: a meta file marks a complete extraction)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / META).unlink(missing_ok=True)
    blobs = {name: csv_bytes(header, tables[name]) for name, header in HEADERS.items()}
    for name, blob in blobs.items():
        write_atomic(out_dir / name, blob)
    meta = dict(version=VERSION, script_sha256=sha256_bytes(Path(__file__).resolve().read_bytes()),
                analysis_sha256=sha256_bytes(Path(fa.__file__).resolve().read_bytes()),
                inputs=dict(inputs, row_files=tables['files']), outputs={n: sha256_bytes(b) for n, b in blobs.items()},
                counts=tables['counts'], missing=tables['missing'], ignored_row_keys=tables['ignored'],
                extra_systems=tables['extra_systems'], extra_incomplete=tables['incomplete'],
                models_checked=tables['checked'], models_unchecked=tables['unchecked'])
    write_atomic(out_dir / META, (json.dumps(meta, indent=1, sort_keys=True) + '\n').encode('utf-8'))
    return meta


def load_json(path: Path, option: str) -> tuple:
    """(content, SHA-256) of a JSON input."""
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw), sha256_bytes(raw)
    except (OSError, ValueError) as e:
        raise SystemExit(f'fv1_tables: {option} {path}: {e}')


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rows', action='append', required=True, type=Path, metavar='DIR',
                    help='a folder of row files, searched recursively (repeat for more folders)')
    ap.add_argument('--manifest', required=True, type=Path, help='KGs/fresh-v1/MANIFEST.json')
    ap.add_argument('--block3', required=True, type=Path, help='the block 3 list')
    ap.add_argument('--models', type=Path, help='the model hashes ({"models": {hash key: {...}}})')
    ap.add_argument('--out-dir', required=True, type=Path)
    a = ap.parse_args(argv)
    out = a.out_dir.resolve()
    for root in a.rows:
        if out == root.resolve() or root.resolve() in out.parents:
            raise SystemExit(f'fv1_tables: --out-dir {a.out_dir} lies inside --rows {root}')
    manifest, manifest_sha = load_json(a.manifest, '--manifest')
    block3, block3_sha = load_json(a.block3, '--block3')
    inputs = dict(rows_dirs=[str(r) for r in a.rows], manifest=dict(path=str(a.manifest), sha256=manifest_sha),
                  block3=dict(path=str(a.block3), sha256=block3_sha), models=None)
    models = None
    if a.models:
        doc, models_sha = load_json(a.models, '--models')
        models = models_list(doc)
        inputs['models'] = dict(path=str(a.models), sha256=models_sha)
    tables = build_tables(row_files(a.rows), graph_list(manifest, block3), models)
    c = write_tables(a.out_dir, tables, inputs)['counts']
    print(f"{c['row_files']} row files, {c['graphs_with_rows']} of {c['graphs_listed']} graphs, {c['models']} models and "
          f"{c['extra_rows']} extra rows (section 10): mrr.csv {c['mrr_rows']} rows, ranks.csv {c['ranks_rows']} rows, "
          f"cost.csv {c['cost_rows']} rows -> {a.out_dir}")
    print(f"missing: {c['missing']} of {c['expected']} expected models, on {c['graphs_with_missing']} graphs; "
          f"{c['extra_incomplete']} incomplete extra seed sets left out (both listed in {META})"
          + ('' if models is not None else '; no --models: no model hash was compared'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
