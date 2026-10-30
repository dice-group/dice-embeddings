"""One graph x systems -> one row file, by the protocol of the research evaluation (``fair_eval`` with per-seed raw ranks).

Per graph (``evaluate_graph``):

1. The graph's files are hash-checked against its manifest, then it loads by the research rules (``data``); the ranked
   test facts follow the manifest's sample rule.
2. Every model of the systems is built on the CPU, attaches train.txt as its inference graph, and moves to the device.
3. Query groups (head direction, relation, anchor), in relation order: one call per group and model,
   ``forward_k_vs_all([[h, r]])`` or ``forward_k_vs_all_heads([[r, t]])``, scores asserted finite.
4. Views per group: ``log_softmax`` of each model's float64 scores; a family's view ``F`` is the mean over its seeds and
   ``F.seed<S>`` the seed's own; a reference's view is its model's. The rule view sets the anchor to ``-inf`` unless
   the relation passes the self-loop rule ``share1``.
5. Per test member: the filtered pessimistic rank of the target (``FilteredRanker(tie_policy='pessimistic')``, which
   equals the research ``#{v >= v[target]}`` over the unfiltered candidates, the rule's ``-inf`` anchor included), and
   the GT2 rank (``gt2_rank``, lambda 0.5).

Row keys, per family ``F`` and seed ``S``, and per reference ``F``:

======================  =======================================================
``F``                   rule view, the family mean (references: the model)
``F:gt2``               rule view with GT2
``F:raw``               no rule
``F:raw:gt2``           no rule, with GT2
``F.seedS:raw``         one seed, no rule (computed directly; no frame hook)
``F.seedS``             one seed, rule view
``F.seedS:gt2``         one seed, rule view with GT2
======================  =======================================================

A row file ``<out>/<collection>/<keys joined by '-'>/<graph>.json`` holds ``dataset``, ``families``, ``entities``,
``relations``, ``reflexive``, ``protocol``, ``metrics`` (the mean reciprocal rank per key), ``rows`` (``index`` = the
position in test.txt, ``head``, every key), ``seconds``, ``allgraph`` (the research record: graph, hashes, sample, systems,
model hashes, Flock's setting), ``metrics_views`` (MRR and Hits per direction) and ``provenance``.

Sharding: ``shard=(i, k)`` ranks block i of k contiguous blocks of the query groups and stores
``.parts/<graph>/shard-i-of-k.json``; a part saves every ``save_minutes`` and when the run stops, resumes after its last
stored group, and the run that completes the last part writes the row file. Ranks do not depend on the sharding:
every query is scored on its own (Flock draws its walks from a generator seeded per call).
"""
import json
import logging
import platform
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from dicee.evaluation._filtering import FilteredRanker
from dicee.models.graph_model import GraphKGE
from dicee.synthetic_kg.arms import SEEDS
from dicee.synthetic_kg.checkpoint import atomic_write_bytes
from dicee.synthetic_kg.determinism import THREADS, deterministic_execution, require_cublas_workspace, torch_threads
from dicee.synthetic_kg.digests import json_sha256

from .data import GraphData, Queries, load_graph, prepare_queries
from .manifests import REPO, SAMPLE_RULES, GraphEntry, Manifest, PathLike, file_sha256
from .systems import Label, OfficialFlock, ProofPFNFamily, System, attach, build_models, official_setting

logger = logging.getLogger(__name__)

PROTOCOL = ('proofpfn benchmark v1: inference = train.txt; candidates = every entity of train, valid and test; filter = train, '
            'valid and test; both directions; pessimistic ties; log_softmax per model in float64; family = mean over its seeds; '
            'rule view share1; GT2 lambda 0.5')
PART_FORMAT = 'proofpfn-rows-part-v1'
GT2_LAMBDA = 0.5
CODE_FILES = ('benchmarks/proofpfn/data.py', 'benchmarks/proofpfn/evaluate.py', 'benchmarks/proofpfn/manifests.py',
              'benchmarks/proofpfn/systems.py', 'dicee/evaluation/_filtering.py', 'dicee/models/flock.py',
              'dicee/models/flock_walks.py', 'dicee/models/graph_model.py', 'dicee/models/trix.py', 'dicee/models/ultra.py',
              'dicee/models/_relation_weights.py', 'dicee/synthetic_kg/checkpoint.py')


# --------------------------------------------------------------------------------------------------------- views

def gt2_rank(v: torch.Tensor, allowed: np.ndarray, target: int, unseen: np.ndarray, good_turing: float) -> int:
    """The GT2 rank of a target (``lockbox-v6-runtime/fair_eval.gt2`` with ``lb5_gt.gt2_rank``).

    The probabilities of the allowed candidates move half-way in log space so that the mass of the candidates unseen
    in the query role approaches the Good-Turing estimate; the rank is pessimistic over the allowed candidates.

    Args:
        v: Log-probabilities over all entities (a view, the rule applied or not).
        allowed: The candidates outside the filter (the target included).
        target: The target entity.
        unseen: The candidates unseen in the query role of train.txt.
        good_turing: ``(n1 + 0.5) / (N + 1)`` of that role.

    Returns:
        The rank.
    """
    z = np.where(allowed, v.numpy(), -np.inf)
    p = np.exp(z - z.max())
    p /= p.sum()
    m = float(np.clip(p[unseen].sum(), 1e-12, 1 - 1e-12))
    q = p.copy()
    q[unseen] *= (good_turing / m) ** GT2_LAMBDA
    q[~unseen] *= ((1 - good_turing) / (1 - m)) ** GT2_LAMBDA
    return int(((q >= q[target]) & allowed).sum())


def group_views(systems: Sequence[System], logp: Mapping[Label, torch.Tensor]) -> list[tuple[str, torch.Tensor, bool]]:
    """``[(key, view, seed view?)]`` of a query group: per family the seed mean, then each seed; per reference its model."""
    views = []
    for system in systems:
        if isinstance(system, ProofPFNFamily):
            views.append((system.key, torch.stack([logp[system.key, s] for s in SEEDS]).mean(0), False))
            views += [(f'{system.key}.seed{s}', logp[system.key, s], True) for s in SEEDS]
        else:
            views.append((system.key, torch.stack([logp[system.key, None]]).mean(0), False))
    return views


def query_scores(model: GraphKGE, head: bool, h: int, r: int, t: int) -> torch.Tensor:
    """The model's scores over all entities for one query, on the CPU.

    Raises:
        FloatingPointError: When a score is not finite.
    """
    query = torch.tensor([[r, t] if head else [h, r]], device=model.device)
    with torch.no_grad():
        scores: torch.Tensor = (model.forward_k_vs_all_heads(query) if head else model.forward_k_vs_all(query))[0].cpu()
    if not torch.isfinite(scores).all():
        raise FloatingPointError('a model returned a score that is not finite')
    return scores


def group_rows(systems: Sequence[System], models: Mapping[Label, GraphKGE], queries: Queries, position: int,
               ranker: FilteredRanker) -> list[dict[str, Any]]:
    """The rows of one query group (one per test member), with every key of every system."""
    (head, r, anchor), members = queries.groups[position]
    h0, _, t0 = queries.facts[members[0]]
    logp = {label: torch.log_softmax(query_scores(model, head, h0, r, t0).double(), 0) for label, model in models.items()}
    views = group_views(systems, logp)
    gated = []
    for _, v, _ in views:
        g = v.clone()
        if r not in queries.reflexive:
            g[anchor] = -float('inf')
        gated.append(g)
    n = len(views[0][1])
    unseen, good_turing = queries.gt2.get((head, r), (np.ones(n, bool), 0.5))
    known = queries.known[head][(r, anchor)]
    rows = []
    for member in members:
        h, _, t = queries.facts[member]
        target = h if head else t
        others = sorted(known - {target})
        mask = torch.ones(n, dtype=torch.bool)
        mask[torch.tensor(others, dtype=torch.long)] = False
        allowed = mask.numpy()
        row: dict[str, Any] = dict(index=queries.index(member), head=head)
        for (key, view, seed), rule in zip(views, gated):
            if seed:
                row[f'{key}:raw'] = ranker.rank(view, target, others, exclude_target=False)
            row[key] = ranker.rank(rule, target, others, exclude_target=False)
            row[f'{key}:gt2'] = gt2_rank(rule, allowed, target, unseen, good_turing)
            if not seed:
                row[f'{key}:raw'] = ranker.rank(view, target, others, exclude_target=False)
                row[f'{key}:raw:gt2'] = gt2_rank(view, allowed, target, unseen, good_turing)
        rows.append(row)
    return rows


def row_keys(systems: Sequence[System]) -> list[str]:
    """The rank keys of the systems, in row order."""
    keys = []
    for system in systems:
        keys += [system.key, f'{system.key}:gt2', f'{system.key}:raw', f'{system.key}:raw:gt2']
        if isinstance(system, ProofPFNFamily):
            keys += [k for s in SEEDS for k in (f'{system.key}.seed{s}:raw', f'{system.key}.seed{s}', f'{system.key}.seed{s}:gt2')]
    return keys


def metrics_of(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The mean reciprocal rank per key, as the research rows compute it."""
    return {k: float(np.mean([1 / row[k] for row in rows])) for k in rows[0] if k not in ('index', 'head')}


def view_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """MRR and Hits@1, 3, 10 per key for both directions, tail queries and head queries (the T1 views)."""
    keys = [k for k in rows[0] if k not in ('index', 'head')]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for view, part in (('both', rows), ('tail', [r for r in rows if not r['head']]), ('head', [r for r in rows if r['head']])):
        out[view] = {}
        for k in keys:
            ranks = np.array([r[k] for r in part], dtype=np.float64)
            out[view][k] = dict(MRR=float((1.0 / ranks).mean()), **{f'H@{c}': float((ranks <= c).mean()) for c in (1, 3, 10)})
    return out


# --------------------------------------------------------------------------------------------------------- files

def write_json(path: Path, value: Any) -> None:
    """Write compact JSON atomically (``checkpoint.atomic_write_bytes``), the layout of the research row files."""
    atomic_write_bytes(path, json.dumps(value).encode())


def row_file_path(out: PathLike, entry: GraphEntry, systems: Sequence[System]) -> Path:
    """``<out>/<collection>/<keys joined by '-'>/<graph>.json``."""
    return Path(out) / entry.collection / '-'.join(s.key for s in systems) / f'{entry.name}.json'


def part_path(final: Path, shard: tuple[int, int]) -> Path:
    """The part file of a shard: ``.parts/<graph>/shard-i-of-k.json`` next to the row file (a dot folder, which the
    registered ``fv1_tables.py`` skips)."""
    return final.parent / '.parts' / final.stem / f'shard-{shard[0]}-of-{shard[1]}.json'


def error_path(final: Path) -> Path:
    """The error record of a graph that could not run: ``.errors/<graph>.json`` next to the row file."""
    return final.parent / '.errors' / final.name


def shard_blocks(groups: int, shards: int) -> list[tuple[int, int]]:
    """``shards`` contiguous blocks ``[a, b)`` of the query groups, of sizes equal up to one group."""
    return [(j * groups // shards, (j + 1) * groups // shards) for j in range(shards)]


def code_sha256() -> dict[str, str]:
    """The SHA-256 of the benchmark modules and the dicee files on the scoring path."""
    return {name: file_sha256(REPO / name) for name in CODE_FILES if (REPO / name).is_file()}


def record_of(manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, systems: Sequence[System],
              setting: Optional[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    """The ``allgraph`` record of a row file (the fields of the research record that the tables read, and the rest)."""
    record = dict(entry.record(), sha256=dict(graph.hashes, manifest=manifest.name), test_rows=queries.test_rows,
                  ranked_facts=len(queries.facts), sampled=queries.positions is not None, sample_rule=SAMPLE_RULES[manifest.sample_rule],
                  positions=queries.positions, families={s.key: s.spec() for s in systems},
                  model_sha256={s.key: s.model_sha256() for s in systems}, seeds=list(SEEDS),
                  gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None, torch=torch.__version__)
    if setting is not None:
        record['flock_official'] = dict(setting)
    return record


def identity_of(manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, systems: Sequence[System]) -> str:
    """A digest of what a part file's rows depend on: graph, sample, systems and model hashes."""
    value = dict(manifest=manifest.name, key=entry.key, hashes=graph.hashes, positions=queries.positions, keys=row_keys(systems),
                 models={s.key: s.model_sha256() for s in systems}, specs={s.key: s.spec() for s in systems})
    return json_sha256(value, compact=False)


# --------------------------------------------------------------------------------------------------------- runner

def evaluate_graph(manifest: Manifest, entry: GraphEntry, systems: Sequence[System], out: PathLike, *, root: PathLike = REPO,
                   device: Union[str, torch.device] = 'cuda', shard: tuple[int, int] = (1, 1), parts: bool = False,
                   save_minutes: float = 10.0, max_minutes: float = 0.0, threads: int = THREADS) -> str:
    """Evaluate one graph with the systems and write its row file (or one shard's part file).

    Runs under ``deterministic_execution`` with ``threads`` CPU threads; both are restored afterwards. On CUDA,
    ``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA starts.

    Args:
        manifest: The graph set.
        entry: The graph.
        systems: The systems, checked (``check``) by the caller or here; their keys name the row folder.
        out: The output root.
        root: The data root of the manifest paths.
        device: The scoring device.
        shard: ``(i, k)``: rank block i of k blocks of the query groups.
        parts: Store a part file (and resume from it) also for an unsharded run.
        save_minutes: Interval of the part file.
        max_minutes: Save and stop after this time (0: no limit).
        threads: CPU threads.

    Returns:
        ``'exists'`` (the row file was there), ``'done'`` (written now), ``'shard-done'`` (this shard is complete and
        others are open) or ``'stopped'`` (saved at ``max_minutes``).

    Raises:
        manifests.ManifestError: When the graph fails its hash check.
        systems.SystemCheckError: When a system cannot run on the graph.
    """
    device = torch.device(device)
    final = row_file_path(out, entry, systems)
    if final.exists():
        logger.info('%s: exists', entry.name)
        return 'exists'
    keys = [s.key for s in systems]
    if len(set(keys)) != len(keys):
        raise ValueError(f'system keys must be unique: {keys}')
    i, k = shard
    if device.type == 'cuda':
        require_cublas_workspace()
    began = time.monotonic()
    with deterministic_execution(), torch_threads(threads):
        graph = load_graph(manifest, entry, root)
        queries = prepare_queries(graph, manifest.sample_rule)
        if not 1 <= i <= k <= max(1, len(queries.groups)):
            raise ValueError(f'{entry.name}: shard {i}/{k} for {len(queries.groups)} query groups')
        for system in systems:   # a family reads its files once; a reference checks its file for every graph
            if not isinstance(system, ProofPFNFamily) or set(system.members) != set(SEEDS):
                system.check()
        flock = [s.key for s in systems if isinstance(s, OfficialFlock)]
        setting = official_setting(entry, graph, flock) if flock else None
        identity = identity_of(manifest, entry, graph, queries, systems)
        a, b = shard_blocks(len(queries.groups), k)[i - 1]
        use_parts = parts or k > 1
        part = part_path(final, shard)
        stored = _read_part(part, entry, keys, shard, len(queries.groups), (a, b), identity) if use_parts else None
        rows: list[dict[str, Any]] = list(stored['rows']) if stored else []
        start = a + (stored['done'] if stored else 0)
        seconds = stored['seconds'] if stored else 0.0
        runs = (stored['runs'] if stored else 0) + 1

        def save(done: int) -> None:
            write_json(part, dict(format=PART_FORMAT, dataset=entry.name, keys=keys, shard=list(shard), groups_total=len(queries.groups),
                                  block=[a, b], done=done - a, complete=done == b, identity=identity,
                                  seconds=seconds + time.monotonic() - began, runs=runs, rows=rows))

        if start < b:
            models = build_models(systems, graph, setting)
            sizes = {s.key: s.query_batch_size(setting) for s in systems}
            for (key, _), model in models.items():
                attach(model, graph.train, graph.num_entities, graph.num_relations, device, sizes[key])
            ranker = FilteredRanker(tie_policy='pessimistic')
            saved = time.monotonic()
            done = start
            try:
                for position in range(start, b):
                    rows += group_rows(systems, models, queries, position, ranker)
                    done = position + 1
                    now = time.monotonic()
                    if use_parts and done < b and now - saved >= 60 * save_minutes:
                        save(done)
                        saved = now
                    if max_minutes and done < b and now - began >= 60 * max_minutes:
                        save(done)
                        logger.info('%s shard %d/%d: stopped after %d of %d groups', entry.name, i, k, done - a, b - a)
                        return 'stopped'
            except BaseException:
                if use_parts:
                    del rows[sum(len(m) for _, m in queries.groups[a:done]):]
                    save(done)
                raise
            finally:
                del models
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if use_parts:
                save(b)
        if not use_parts:
            _write_row_file(final, manifest, entry, graph, queries, systems, setting, rows, time.monotonic() - began, device, [runs], k)
            return 'done'
        return _merge(final, manifest, entry, graph, queries, systems, setting, identity, k, device)


def _read_part(path: Path, entry: GraphEntry, keys: list[str], shard: tuple[int, int], groups: int, block: tuple[int, int],
               identity: str) -> Optional[dict[str, Any]]:
    """A stored part of this run, or ``None``; a part of another layout or run is refused."""
    if not path.exists():
        return None
    part: dict[str, Any] = json.loads(path.read_text())
    expected = (PART_FORMAT, entry.name, keys, list(shard), groups, list(block), identity)
    if (part.get('format'), part.get('dataset'), part.get('keys'), part.get('shard'), part.get('groups_total'), part.get('block'),
            part.get('identity')) != expected:
        raise ValueError(f'{path} is a part of another run or shard layout')
    return part


def _merge(final: Path, manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, systems: Sequence[System],
           setting: Optional[Mapping[str, Any]], identity: str, shards: int, device: torch.device) -> str:
    """Write the row file when all parts of the layout are complete; ``'shard-done'`` while one is open."""
    paths = [part_path(final, (j, shards)) for j in range(1, shards + 1)]
    stored = [json.loads(p.read_text()) if p.exists() else None for p in paths]
    if any(p is None or not p['complete'] or p['identity'] != identity for p in stored):
        return 'shard-done'
    rows = [row for p in stored if p is not None for row in p['rows']]
    _write_row_file(final, manifest, entry, graph, queries, systems, setting, rows, sum(p['seconds'] for p in stored if p is not None),
                    device, [p['runs'] for p in stored if p is not None], shards)
    return 'done'


def _write_row_file(final: Path, manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, systems: Sequence[System],
                    setting: Optional[Mapping[str, Any]], rows: list[dict[str, Any]], seconds: float, device: torch.device,
                    runs: list[int], shards: int) -> None:
    """Check the complete rows (order, keys, ranges) and write the row file."""
    order = [(queries.index(m), head) for (head, _, _), members in queries.groups for m in members]
    if [(r['index'], r['head']) for r in rows] != order or len(rows) != 2 * len(queries.facts):
        raise ValueError(f'{entry.name}: the rows are not the query groups in order')
    keys = row_keys(systems)
    n = graph.num_entities
    if any(list(r) != ['index', 'head', *keys] for r in rows) or any(not 1 <= r[q] <= n for r in rows for q in keys):
        raise ValueError(f'{entry.name}: a row has other keys or a rank outside 1..{n}')
    provenance = dict(benchmark='benchmarks.proofpfn', protocol=PROTOCOL, manifest=dict(name=manifest.name, sha256=manifest.sha256),
                      code_sha256=code_sha256(), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      device=str(device), deterministic=True, threads=torch.get_num_threads(), host=socket.gethostname(),
                      shards=shards, runs=runs, checkpoints={s.key: s.provenance() for s in systems if isinstance(s, ProofPFNFamily)})
    write_json(final, dict(dataset=entry.name, families=[s.key for s in systems], entities=n, relations=graph.num_relations,
                           reflexive=sorted(queries.reflexive), protocol=PROTOCOL, metrics=metrics_of(rows), rows=rows, seconds=seconds,
                           allgraph=record_of(manifest, entry, graph, queries, systems, setting, device), metrics_views=view_metrics(rows),
                           provenance=provenance))
    logger.info('%s: %d rows, %.0f s -> %s', entry.name, len(rows), seconds, final)
