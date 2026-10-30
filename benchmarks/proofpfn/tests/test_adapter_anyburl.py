"""The AnyBURL adapter: input files, the rank rebuild (hand-made lists, and the research rebuild on a stored ranking file),
the row file, and, with ``$ANYBURL_HOME``, the Java steps.

Learn is time-bounded and threaded, so its rules differ between runs; the tests with Java check the protocol and the
format, and Apply with a stored rule file (whose lists do not depend on the threads).
"""
import fcntl
import gzip
import json
import os
import shutil
import signal
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn.adapters import AdapterSetupError, anyburl
from benchmarks.proofpfn.data import GraphData, Queries, load_graph, prepare_queries
from dicee.synthetic_kg.digests import json_sha256

from .golden import GOLDEN, write_graph, write_manifest

FIXTURE = GOLDEN / 'anyburl'
GATE_GRAPH = 'ultra-suite-v1/NELLInductive_v1'


def ranking_file(path: Path, blocks: dict) -> None:
    """``{(h, r, t): (heads, tails)}`` with lists of (entity, confidence) -> AnyBURL's ranking file."""
    with open(path, 'w') as handle:
        for (h, r, t), (heads, tails) in blocks.items():
            handle.write(f'e{h} r{r} e{t}\n')
            handle.write('Heads: ' + ''.join(f'e{e}\t{c!r}\t' for e, c in heads) + '\n')
            handle.write('Tails: ' + ''.join(f'e{e}\t{c!r}\t' for e, c in tails) + '\n')


def ids(rows: list) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 3)


# relation 0: no self-loop in train (the rule removes the anchor); relation 1: 1 of 2 train facts is a self-loop (open)
TRAIN = [(1, 0, 3), (2, 1, 2), (3, 1, 4), (5, 0, 2)]
RANKED = [(0, 0, 1), (0, 0, 2), (3, 1, 3), (4, 0, 4), (5, 0, 0)]
BLOCKS = {(0, 0, 1): ([], [(2, 0.9), (0, 0.7), (1, 0.5), (3, 0.5)]),    # 2 is a known tail of (0, 0) (test): filtered
          (0, 0, 2): ([(2, 0.25)], [(0, 0.7), (2, 0.6), (3, 0.5)]),     # heads: only the anchor 2 is listed
          (3, 1, 3): ([(3, 0.2)], [(4, 0.9), (3, 0.4), (5, 0.4)]),      # 4: train fact (3, 1, 4), filtered
          (4, 0, 4): ([(1, 0.3)], [(4, 0.8), (1, 0.3)]),                # self-loop target, relation 0: gated
          (5, 0, 0): ([(5, 0.9)], [(1, 0.6), (0, 0.5), (3, 0.5)])}      # full at k = 3, target ties the last entry
EXPECT = {  # (index, head): (anyburl, anyburl:raw, native) for k = 1000 (the research checks of test_anyburl_run.py)
    (0, False): (2, 3, 2), (0, True): (6, 6, 0), (1, False): (1, 2, 2), (1, True): (5, 5, 0), (2, False): (2, 2, 1),
    (2, True): (1, 1, 1), (3, False): (6, 1, 1), (3, True): (6, 6, 0), (4, False): (3, 3, 2), (4, True): (1, 1, 1)}


def handmade() -> tuple[GraphData, Queries]:
    graph = GraphData(tuple(f'e{i}' for i in range(6)), ('r0', 'r1'), ids(TRAIN), ids([]), ids(RANKED), {})
    return graph, prepare_queries(graph, 'all')


def brute(graph: GraphData, queries, k: int) -> dict:
    """Independent pessimistic ranks in plain Python: ``{(index, head): (rule rank, raw rank)}``."""
    known = {tuple(x) for x in graph.train.tolist() + graph.test.tolist() + graph.valid.tolist()}
    train = graph.train.tolist()
    loops = {r for _, r, _ in train}
    reflexive = {r for r in loops if sum(1 for h, q, t in train if q == r and h == t) >= 0.01 * sum(1 for _, q, _ in train if q == r)}
    out = {}
    for i, (h, r, t) in enumerate(queries.facts):
        for head in (False, True):
            target, anchor = (h, t) if head else (t, h)
            listed = BLOCKS[h, r, t][0 if head else 1]
            score = dict(listed)
            allowed = [e for e in range(graph.num_entities) if e == target or ((e, r, t) if head else (h, r, e)) not in known]
            cut = len(listed) == k < len(allowed) and target in score and score[target] == listed[-1][1]
            ranks = []
            for gate in (r not in reflexive, False):
                s = {e: (-np.inf if gate and e == anchor else score.get(e, -np.inf)) for e in allowed}
                ranks.append(len(allowed) if cut else sum(1 for e in allowed if s[e] >= s[target]))
            out[i, head] = tuple(ranks)
    return out


@pytest.mark.parametrize('k', [1000, 3])
def test_rebuild_of_handmade_lists(tmp_path, k):
    graph, queries = handmade()
    path = tmp_path / 'preds'
    ranking_file(path, BLOCKS)
    rows, native, counts = anyburl.rebuild(path, graph, queries, k)
    got = {(r['index'], r['head']): (r[anyburl.KEY], r[anyburl.RAW]) for r in rows}
    want = {q: v[:2] for q, v in EXPECT.items()}
    if k == 3:   # full lists (3 entries < 5 allowed) whose last entry ties with the target: the worst rank
        want[2, False] = want[4, False] = (5, 5)
    assert got == want == brute(graph, queries, k)
    assert {(i, h): x for i, h, x in native} == {q: v[2] for q, v in EXPECT.items()}
    assert queries.reflexive == {1}
    assert counts == dict(queries=10, unlisted=3, cut_ties=2 if k == 3 else 0, empty=1, full=3 if k == 3 else 0, filtered_listed=2,
                          anchor_listed=3, gated_self=2, unsorted=0)
    assert [(r['index'], r['head']) for r in rows] == [(m, head) for (head, _, _), members in queries.groups for m in members]


def test_ranking_file_reading(tmp_path):
    path = tmp_path / 'preds'
    ranking_file(path, BLOCKS)
    with open(path, 'a') as handle:   # a repeated block keeps the last one (AnyBURL's ResultSet); the two lines in either order
        handle.write('e0 r0 e1\nTails: e1\t0.5\t\nHeads: \n')
    lists, blocks = anyburl.read_lists(path)
    assert blocks == 6 and lists[0, 0, 1][1][0].tolist() == [1] and lists[0, 0, 1][0][0].tolist() == []
    with gzip.open(tmp_path / 'preds.gz', 'wt') as handle:
        handle.write(path.read_text())
    gz, _ = anyburl.read_lists(tmp_path / 'preds.gz')
    assert set(gz) == set(lists) and all(np.array_equal(gz[q][1][1], lists[q][1][1]) for q in lists)
    graph, queries = handmade()
    with pytest.raises(ValueError, match='one block per ranked fact'):
        anyburl.rebuild(tmp_path / 'preds', graph, prepare_queries(GraphData(graph.entities, graph.relations, graph.train, graph.valid,
                                                                             ids(RANKED[:2]), {}), 'all'), 1000)
    with pytest.raises(ValueError):
        anyburl.candidates('Tails: x1\t0.5\t', 'Tails: ')


def test_input_files_of_a_sampled_graph(tmp_path):
    rng = np.random.default_rng(0)
    facts = sorted({(f'n{h}', f'p{r}', f'n{t}') for h, r, t in rng.integers(0, [80, 3, 80], (6000, 3)) if h != t})
    lines = ['\t'.join(f) for f in facts]
    folder = write_graph(tmp_path / 'KGs' / 'big', lines[:3000], lines[3000:3300], lines[3300:4600] + lines[3300:3310])
    manifest = mf.read_manifest(write_manifest(tmp_path / 'm.json', tmp_path, {'big': folder}))
    graph = load_graph(manifest, manifest.entry('big'), tmp_path)
    queries = prepare_queries(graph, manifest.sample_rule)
    assert queries.positions is not None and len(queries.facts) == 1000
    work = tmp_path / 'work'
    work.mkdir()
    hashes = anyburl.write_inputs(work, graph, queries)
    files = {name: [tuple(int(x[1:]) for x in line.split('\t')) for line in (work / name).read_text().splitlines()] for name in hashes}
    assert files['train.txt'] == [tuple(x) for x in graph.train.tolist()] and files['test.txt'] == queries.facts
    assert len(files['valid.txt']) == len(graph.valid) + len(graph.test) - 1000   # valid plus the unranked test rows, duplicates kept
    every = {tuple(x) for x in graph.train.tolist() + graph.valid.tolist() + graph.test.tolist()}
    assert set().union(*map(set, files.values())) == every
    assert hashes == {name: mf.file_sha256(work / name) for name in hashes}
    assert (work / 'train.txt').read_text().splitlines()[0] == 'e{}\tr{}\te{}'.format(*graph.train[0].tolist())


def gate_graph() -> tuple[mf.Manifest, mf.GraphEntry, GraphData]:
    manifest = mf.read_manifest('gate')
    entry = manifest.entry(GATE_GRAPH)
    if not (mf.REPO / entry.path / 'train.txt').is_file():
        pytest.skip(f'{entry.path} is not available')
    return manifest, entry, load_graph(manifest, entry, mf.REPO)


def test_rebuild_equals_the_research_rebuild_on_a_stored_ranking_file(tmp_path):
    """The research anyburl_run.rebuild (and write_inputs) on the same ranking file of NELLInductive_v1 (golden README)."""
    record = json.loads((FIXTURE / 'NELLInductive_v1-research.json').read_text())
    _, _, graph = gate_graph()
    queries = prepare_queries(graph, 'first-1000')
    rows, native, counts = anyburl.rebuild(FIXTURE / 'NELLInductive_v1-preds-225.gz', graph, queries, record['k'])
    assert json_sha256(rows) == record['rows_digest'] and json_sha256(native) == record['native_digest']
    assert counts == record['counts'] and sorted(queries.reflexive) == record['reflexive'] and queries.positions == record['positions']
    assert anyburl.write_inputs(tmp_path, graph, queries) == record['inputs'] == record['kiowl_inputs']


def fake_home(tmp_path: Path) -> tuple[Path, Path]:
    """An AnyBURL folder with stand-in files and the KR setting, and a models.json that pins its files."""
    home = tmp_path / 'AnyBURL'
    (home / anyburl.KR_LEARN_FILE).parent.mkdir(parents=True)
    for name in (anyburl.JAR, anyburl.SOURCES):
        (home / name).write_bytes(b'not the release')
    (home / anyburl.KR_LEARN_FILE).write_text(''.join(f'{k} = {v}\n' for k, v in anyburl.KR_LEARN.items()))
    doc = json.loads((mf.MANIFESTS / 'models.json').read_text())
    doc['external']['anyburl'] = {name: mf.file_sha256(home / name) for name in (anyburl.JAR, anyburl.SOURCES)}
    (tmp_path / 'models.json').write_text(json.dumps(doc))
    return home, tmp_path / 'models.json'


def test_setup_refuses_other_files(tmp_path):
    home, models = fake_home(tmp_path)
    with pytest.raises(AdapterSetupError, match='SHA-256'):
        anyburl.setup(home)
    (home / anyburl.KR_LEARN_FILE).write_text('SNAPSHOTS_AT = 10\n')
    with pytest.raises(AdapterSetupError, match='KR setting'):
        anyburl.setup(home, models)


def fake_java(folder: Path, script: str) -> anyburl.AnyBURL:
    """AnyBURL with a stand-in ``java`` (a shell script) that runs without the memory scope."""
    java = folder / 'java'
    java.write_text('#!/bin/sh\n' + script)
    java.chmod(0o755)
    return anyburl.AnyBURL(home=folder, pins={}, java=str(java), scope=(), version='a stand-in')


def test_a_stopped_step_kills_its_java_process(tmp_path):
    """A signal while a Java step runs (the command line exits on SIGTERM) kills and reaps the JVM before the lock is free."""
    system = fake_java(tmp_path, 'echo $$ > java.pid\nexec sleep 60\n')
    config = tmp_path / 'work' / 'config-learn.properties'
    config.parent.mkdir()
    config.touch()
    previous = signal.signal(signal.SIGALRM, lambda *_: sys.exit(143))
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        with pytest.raises(SystemExit):
            system.run('Learn', config, tmp_path / 'learn.log', tmp_path / 'java.lock')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    with pytest.raises(ProcessLookupError):   # killed and reaped
        os.kill(int((config.parent / 'java.pid').read_text()), 0)
    with open(tmp_path / 'java.lock') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)


def installed() -> anyburl.AnyBURL:
    """The AnyBURL folder of ``$ANYBURL_HOME`` (skips without it, or without Java)."""
    if not os.environ.get(anyburl.VARIABLE) or shutil.which('java') is None:
        pytest.skip('set $ANYBURL_HOME to the AnyBURL folder (and put Java 11 on the path)')
    return anyburl.setup()


def test_apply_with_a_stored_rule_file_equals_the_research_rebuild(tmp_path):
    system = installed()
    record = json.loads((FIXTURE / 'NELLInductive_v1-research.json').read_text())
    _, _, graph = gate_graph()
    queries = prepare_queries(graph, 'first-1000')
    anyburl.write_inputs(tmp_path, graph, queries)
    anyburl.apply(system, tmp_path, record['k'], tmp_path / 'java.lock', rules=FIXTURE / 'NELLInductive_v1-rules-30')
    rows, native, counts = anyburl.rebuild(tmp_path / f"preds-{record['k']}", graph, queries, record['k'])
    assert json_sha256(rows) == record['rows_digest'] and json_sha256(native) == record['native_digest'] and counts == record['counts']
    printed = anyburl.native_eval(system, tmp_path, record['k'], tmp_path / 'java.lock')
    assert anyburl.agrees(printed, anyburl.native_metrics(native))


@pytest.mark.slow
def test_learn_apply_and_the_row_file(tmp_path):
    """One graph end to end (30 seconds of Learn): format and protocol of the row file."""
    system = installed()
    _, _, graph = gate_graph()
    manifest = mf.read_manifest('gate')
    entry = manifest.entry(GATE_GRAPH)
    assert anyburl.evaluate_graph(manifest, entry, tmp_path / 'rows', system) == 'done'
    path = tmp_path / 'rows' / entry.collection / 'anyburl' / f'{entry.name}.json'
    doc = json.loads(path.read_text())
    assert list(doc) == ['dataset', 'families', 'entities', 'relations', 'reflexive', 'protocol', 'metrics', 'rows', 'seconds', 'allgraph',
                         'metrics_views', 'anyburl_meta', 'provenance']
    assert all(list(r) == ['index', 'head', 'anyburl', 'anyburl:raw'] for r in doc['rows']) and len(doc['rows']) == 2 * len(graph.test)
    meta = doc['anyburl_meta']
    assert meta['native']['agree'] and meta['rules'] > 0 and meta['kr_learn_setting'] == anyburl.KR_LEARN and meta['threads'] == 4
    assert doc['allgraph']['model_sha256'] == {} and doc['allgraph']['sha256']['train.txt'] == entry.sha256['train']
    work = tmp_path / 'rows' / '.work' / 'anyburl' / entry.collection / entry.name
    assert (work / f"preds-{meta['top_k']}.gz").is_file() and (work / 'rules-30').is_file()
    assert anyburl.evaluate_graph(manifest, entry, tmp_path / 'rows', system) == 'exists'
