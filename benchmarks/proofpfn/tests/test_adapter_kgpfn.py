"""The KGPFN adapter: context size and settings, staging and meta, the worker's ranks against the benchmark's ranker, the
finalize and verify steps, and, with KGPFN's environment and a GPU of the reference, the ranks of the probe graphs.

The worker module imports KGPFN only inside ``main``, so its rank function runs here without KGPFN.
"""
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks.proofpfn import manifests as mf
from benchmarks.proofpfn.adapters import AdapterSetupError, ExternalStepError, kgpfn, kgpfn_worker
from benchmarks.proofpfn.data import as_triples, load_graph, prepare_queries, read_triples

from .golden import GOLDEN, int_digest, tiny_graph, write_manifest


def tiny(tmp_path: Path, **options):
    folder = tiny_graph(tmp_path / 'KGs' / 'g', **options)
    manifest = mf.read_manifest(write_manifest(tmp_path / 'tiny.json', tmp_path, {'g': folder}))
    graph = load_graph(manifest, manifest.entry('g'), tmp_path)
    return manifest, manifest.entry('g'), graph, prepare_queries(graph, manifest.sample_rule), folder


def fake_setup(tmp_path: Path) -> kgpfn.KGPFN:
    return kgpfn.KGPFN(code=tmp_path / 'KGPFN', checkpoints=tmp_path / 'ckpt', python=('python3.12',), pins=kgpfn.external_pins('kgpfn'))


def test_settings_follow_the_entity_count():
    assert [kgpfn.settings(n) for n in (100, 19_000, 19_001, 38_001, 76_001)] == [
        (64, 64, 256), (64, 64, 256), (32, 64, 512), (16, 32, 1024), (8, 16, 2048)]


def test_unfillable_contexts_equal_a_direct_count(tmp_path):
    _, _, graph, queries, _ = tiny(tmp_path, entities=12)
    train = as_triples(graph.train)
    for num_neg in (0, 4, 8, 12):
        bad = 0
        for h, r, t in queries.facts:
            pairs = len({(a, b) for a, q, b in train if q == r})
            need = num_neg + 20 - min(20, pairs)
            bad += (graph.num_entities - len({b for a, q, b in train if (a, q) == (h, r)}) < need)
            bad += (graph.num_entities - len({a for a, q, b in train if (q, b) == (r, t)}) < need)
        assert kgpfn.unfillable(graph, queries, num_neg) == bad
    assert kgpfn.unfillable(graph, queries, 80) == 2 * len(queries.facts)   # 12 entities can never hold 80 negatives


def test_prepare_stages_the_files_and_writes_the_meta(tmp_path):
    _, entry, graph, queries, folder = tiny(tmp_path, entities=200)
    work = tmp_path / 'work'
    meta = kgpfn.prepare(fake_setup(tmp_path), entry, graph, queries, work, folder)
    assert json.loads((work / 'meta.json').read_text()) == meta
    assert meta['num_neg'] in kgpfn.NEG_STEPS and meta['unfillable'][str(meta['num_neg'])] == 0
    assert meta['positions'] is None and meta['reflexive_ids'] == sorted(queries.reflexive)
    assert meta['reflexive'] == sorted(graph.relations[r] for r in queries.reflexive)
    assert (meta['batch_size'], meta['entity_chunk'], meta['eval_chunk']) == kgpfn.settings(graph.num_entities) and meta['dump']
    assert meta['pins'] == kgpfn.external_pins('kgpfn') and meta['sha256']['train.txt'] == entry.sha256['train']
    for s in mf.SPLITS:
        assert (work / 'stage' / f'{s}.txt').read_bytes() == (folder / f'{s}.txt').read_bytes()
        # the worker's parse (a tab split) hashes as the host's
        assert kgpfn_worker.parse_sha(kgpfn_worker.read_split(work / 'stage' / f'{s}.txt')) == meta['parse_sha256'][s]
    small = tiny(tmp_path / 'small', entities=12)
    with pytest.raises(ExternalStepError) as caught:   # no context size fills every query: the graph fails (an error record)
        kgpfn.evaluate_graph(small[0], small[1], tmp_path / 'rows', fake_setup(tmp_path), root=tmp_path / 'small')
    assert caught.value.record['stage'] == 'prepare' and caught.value.record['unfillable'][20] > 0


def worker_ranks(graph, queries, scores_of) -> tuple[dict, torch.Tensor]:
    """The worker's ``our_ranks`` in the worker's entity order (first appearance in train, valid and test):
    ``({(position, head): (rule, raw)}, benchmark id -> worker id)``; ``scores_of`` gives scores in the benchmark's ids."""
    order = {e: i for i, e in enumerate(dict.fromkeys(e for split in (graph.train, graph.valid, graph.test)
                                                      for h, _, t in split.tolist() for e in (h, t)))}
    to_worker = torch.tensor([order[e] for e in range(graph.num_entities)])
    known_t: dict = {}
    known_h: dict = {}
    for split in (graph.train, graph.valid, graph.test):
        for h, r, t in split.tolist():
            known_t.setdefault((order[h], r), set()).add(order[t])
            known_h.setdefault((order[t], r), set()).add(order[h])
    out = {}
    for position, (h, r, t) in enumerate(queries.facts):
        benchmark = torch.stack([scores_of(position, False), scores_of(position, True)])
        pred = torch.empty_like(benchmark)
        pred[:, to_worker] = benchmark
        batch, target = torch.tensor([[order[h], order[t], r]]), torch.tensor([order[t], order[h]])
        gate, anchor = torch.tensor([r not in queries.reflexive] * 2), torch.tensor([order[h], order[t]])
        rank, _, rule, _ = kgpfn_worker.our_ranks(pred, batch, target, known_t, known_h, gate, anchor)
        out[position, False], out[position, True] = (int(rule[0]), int(rank[0])), (int(rule[1]), int(rank[1]))
    return out, to_worker


def test_worker_ranks_equal_the_benchmark_ranker_and_verify(tmp_path):
    """The worker's ranks on raw scores, the host's on log_softmax of the dumped scores (the research verify step)."""
    _, entry, graph, queries, folder = tiny(tmp_path, entities=40)
    rng = np.random.default_rng(0)
    table = {(p, head): torch.tensor(np.round(rng.normal(size=graph.num_entities), 1), dtype=torch.float32)   # rounding makes ties
             for p in range(len(queries.facts)) for head in (False, True)}
    expected, to_worker = worker_ranks(graph, queries, lambda p, head: table[p, head])
    work = tmp_path / 'work'
    kgpfn.prepare(fake_setup(tmp_path), entry, graph, queries, work, folder)
    keys = sorted(table)
    scores = np.stack([table[q].numpy() for q in keys])   # columns in the benchmark's ids: permute to the worker's
    worker_scores = np.zeros_like(scores)
    worker_scores[:, to_worker.numpy()] = scores
    np.savez_compressed(work / 'scores.npz', index=np.array([p for p, _ in keys]), head=np.array([h for _, h in keys]), scores=worker_scores)
    raw = {f'{side}_rank': [None] * len(queries.facts) for side in ('tail', 'head')}
    raw.update({f'{side}_rank_rule': [None] * len(queries.facts) for side in ('tail', 'head')})
    for (p, head), (rule, rank) in expected.items():
        side = 'head' if head else 'tail'
        raw[f'{side}_rank'][p], raw[f'{side}_rank_rule'][p] = rank, rule
    rows = kgpfn.rows_of(raw, queries)
    assert [(r['index'], r['head']) for r in rows] == [(m, head) for (head, _, _), members in queries.groups for m in members]
    assert all(list(r) == ['index', 'head', 'kgpfn', 'kgpfn:raw'] for r in rows)
    assert kgpfn.verify(work / 'scores.npz', work, graph, queries, rows) == 0
    rows[3] = dict(rows[3], **{'kgpfn:raw': rows[3]['kgpfn:raw'] + 1})
    assert kgpfn.verify(work / 'scores.npz', work, graph, queries, rows) == 1


def test_rows_and_outputs_are_checked(tmp_path):
    _, entry, graph, queries, folder = tiny(tmp_path, entities=200)
    work = tmp_path / 'work'
    kgpfn.prepare(fake_setup(tmp_path), entry, graph, queries, work, folder)
    n = len(queries.facts)
    raw = {key: [1] * n for key in ('tail_rank', 'head_rank', 'tail_rank_rule', 'head_rank_rule')}
    assert len(kgpfn.rows_of(raw, queries)) == 2 * n
    raw['tail_rank_rule'][0] = 2   # the rule view moves a rank down only for a target that is its anchor
    with pytest.raises(ValueError, match='position 0'):
        kgpfn.rows_of(raw, queries)
    good: dict[str, Any] = dict(status='complete', context_mode='blind', seed=0, settings=dict(meta_sha256=mf.file_sha256(work / 'meta.json')),
                                strict_load=dict(missing=[], shape_mismatch=[], value_mismatch=[], unexpected=[]), counts=dict(nonfinite_rows=0))
    kgpfn.check_output(good, work / 'meta.json')
    for change, message in ((dict(status='partial'), 'partial'), (dict(context_mode='native'), 'native'),
                            (dict(settings=dict(meta_sha256='0')), 'another meta'),
                            (dict(strict_load=dict(good['strict_load'], missing=['x'])), 'exactly'), (dict(counts=dict(nonfinite_rows=2)), 'non-finite')):
        with pytest.raises(ValueError, match=message):
            kgpfn.check_output(dict(good, **change), work / 'meta.json')


def test_worker_command_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_EXTENSIONS_DIR', str(tmp_path / 'elsewhere'))   # a container would not see this folder
    system = fake_setup(tmp_path)
    folder = tmp_path / 'work' / 'kgpfn' / 'tiny' / 'g'
    command = kgpfn.worker_command(system, mf.GraphEntry.from_json(json.loads(
        write_manifest(tmp_path / 'm.json', tmp_path, {'g': tiny_graph(tmp_path / 'KGs' / 'g')}).read_text())['graphs'][0]),
        folder, 64, 64, 256, 40, True)
    assert command[:4] == ['python3.12', '-B', '-P', str(kgpfn.WORKER)] and command[-2:] == ['--dump-scores', str(folder / 'scores.npz')]
    assert command[command.index('--num-neg') + 1] == '40' and command[command.index('--code') + 1] == str(system.code)
    env = kgpfn.worker_environment(system, folder)
    assert env['PYTHONPATH'] == str(system.code) and env['PYTHONDONTWRITEBYTECODE'] == '1' and env['KGPFN_WORK'] == str(folder.parents[1])
    assert env['KGPFN_WORKER_DIR'] == str(kgpfn.WORKER.parent) and env['TORCH_EXTENSIONS_DIR'] == str(folder.parents[1] / 'torch_extensions')
    worker = kgpfn_worker.parse_args(command[4:])
    assert (worker.batch_size, worker.entity_chunk, worker.eval_chunk, worker.num_neg) == (64, 64, 256, 40)


def test_setup_refusals(tmp_path):
    code, ckpt = tmp_path / 'KGPFN', tmp_path / 'ckpt'
    (code / '.git').mkdir(parents=True)
    ckpt.mkdir()
    (code / '.git' / 'HEAD').write_text('0' * 40 + '\n')
    with pytest.raises(AdapterSetupError, match='commit'):
        kgpfn.setup(code, ckpt, 'python3.12')
    pins = kgpfn.external_pins('kgpfn')
    (code / '.git' / 'HEAD').write_text('ref: refs/heads/main\n')
    (code / '.git' / 'packed-refs').write_text(f"# pack-refs\n{pins['code_commit']} refs/heads/main\n")
    with pytest.raises(AdapterSetupError, match='missing'):
        kgpfn.setup(code, ckpt, 'python3.12')
    for name in (kgpfn.CHECKPOINT, kgpfn.TABICL):
        (ckpt / name).write_bytes(b'x')
    with pytest.raises(AdapterSetupError, match='SHA-256'):
        kgpfn.setup(code, ckpt, 'python3.12')


def test_worker_parse_and_entity_order(tmp_path):
    folder = tiny_graph(tmp_path / 'g')
    for s in mf.SPLITS:
        assert kgpfn_worker.read_split(folder / f'{s}.txt') == read_triples(folder / f'{s}.txt')
    names = kgpfn.worker_entities(folder)
    assert len(names) == len(set(names)) and names[:2] == list(read_triples(folder / 'train.txt')[0][::2])


# ------------------------------------------------------------------------------------------------ probe graphs

def kgpfn_environment() -> kgpfn.KGPFN:
    """KGPFN's setup from ``$KGPFN_CODE``, ``$KGPFN_CHECKPOINT_DIR`` and ``$KGPFN_PYTHON`` (skips without them)."""
    if not all(os.environ.get(v) for v in ('KGPFN_CODE', 'KGPFN_CHECKPOINT_DIR', 'KGPFN_PYTHON')) or not torch.cuda.is_available():
        pytest.skip("set $KGPFN_CODE, $KGPFN_CHECKPOINT_DIR and $KGPFN_PYTHON (KGPFN's environment) on a CUDA machine")
    return kgpfn.setup()


@pytest.mark.slow
def test_probe_graph_ranks_equal_the_reference_of_this_gpu(tmp_path):
    """The worker's ranks of the three probe graphs equal the research ranks on the same GPU type (KGPFN's ranks depend on it)."""
    system = kgpfn_environment()
    record = json.loads((GOLDEN / 'kgpfn_probe_ranks.json').read_text())
    gpu = torch.cuda.get_device_name(0)
    if gpu not in record['references']:
        pytest.skip(f'no KGPFN reference ranks for {gpu}; references: {sorted(record["references"])}')
    want = record['references'][gpu]['digests']
    for name, info in record['graphs'].items():
        manifest = mf.read_manifest(info['manifest'])
        entry = manifest.entry(info['key'])
        if not (mf.REPO / entry.path / 'train.txt').is_file():
            pytest.skip(f'{entry.path} is not available')
        assert kgpfn.evaluate_graph(manifest, entry, tmp_path / 'rows', system) == 'done'
        work = tmp_path / 'rows' / '.work' / 'kgpfn' / entry.collection / entry.name
        raw = json.loads((work / 'raw.json').read_text())
        meta = json.loads((work / 'meta.json').read_text())
        assert (meta['num_neg'], meta['batch_size'], meta['entity_chunk'], meta['eval_chunk']) == \
            (info['num_neg'], info['batch_size'], info['entity_chunk'], info['eval_chunk'])
        assert {k: int_digest(raw[k]) for k in want[name]} == want[name], name
        doc = json.loads((tmp_path / 'rows' / entry.collection / 'kgpfn' / f'{entry.name}.json').read_text())
        assert doc['allgraph']['model_sha256'] == {} and doc['kgpfn_meta']['verified']
