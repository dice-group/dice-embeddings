"""KGPFN (arXiv 2605.14907v2), ``kgpfn_icl_3g`` with the answer-blind context and context seed 0, as rows of the benchmark
(README, section KGPFN; PROTOCOL, section External baselines).

Host side of the research runners ``kgpfn-ref-v1-runtime/kgpfn_ag.py`` (prepare, finalize, verify) and
``fresh-v1-runtime/fv1_worker.py`` (the run with its retry); the scoring runs in KGPFN's own Python environment
(``kgpfn_worker.py``). Per graph (``evaluate_graph``):

1. Prepare: the context negatives are KGPFN's 80, else the largest of 40 and 20 that fills the context of every ranked
   query (``unfillable``). Batch, NBFNet head chunk and candidate chunk follow the entity count (``settings``, memory
   only). The split files go to ``stage/`` with ``meta.json`` (positions, open relations, parse hashes, pins).
2. Run: the worker, and after CUDA out of memory (exit 3) once more with half the batch. The batch changes the context
   draws, so the record keeps the attempts.
3. Verify (entities x ranked facts up to 2.5 million): both views of the dumped scores, ranked by the benchmark's ranker,
   must equal the worker's ranks.
4. Finalize: the row file ``<out>/<collection>/kgpfn/<graph>.json`` with ``kgpfn`` (rule view), ``kgpfn:raw`` and
   ``kgpfn_meta``.
"""
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from dicee.evaluation._filtering import FilteredRanker

from ..data import GraphData, Queries, as_triples, load_graph, prepare_queries, read_triples
from ..evaluate import row_file_path
from ..manifests import REPO, SPLITS, GraphEntry, Manifest, PathLike, file_sha256
from . import (
    AdapterSetupError,
    ExternalStepError,
    call,
    check_commit,
    check_file,
    external_folder,
    external_pins,
    group_targets,
    view_ranks,
    work_folder,
    write_rows,
)

logger = logging.getLogger(__name__)

KEY, RAW = 'kgpfn', 'kgpfn:raw'
CHECKPOINT, TABICL = 'kgpfn_icl_3g.pth', 'tabicl-regressor-v2-20260212.ckpt'
NUM_POS, NEG_STEPS = 20, (80, 40, 20)
DUMP_LIMIT = 2_500_000   # entities x ranked facts up to which the scores are dumped and verified
EXIT_OOM = 3
WORKER_HOURS = 24.0
"""The time after which a worker run stops."""
WORKER = Path(__file__).resolve().with_name('kgpfn_worker.py')
IMPORTS = 'import pfn.tasks, pfn.util, script.test_kgpfn'
"""The worker's imports of KGPFN, which import KGPFN's requirements."""
SETTING = 'KGPFN icl_3g (arXiv 2605.14907v2, code 4f1bc61), answer-blind context, context seed 0'
PROTOCOL = ('inference graph = train.txt; candidates = all entities of train + valid + test; filter = train + valid + all '
            'test facts, the target kept; pessimistic ties; tail and head query of every ranked fact; KGPFN context: 20 '
            'positives and 80 negatives of the query relation from the visible graph, drawn without the answer (blind)')
CODE = ('benchmarks/proofpfn/adapters/__init__.py', 'benchmarks/proofpfn/adapters/kgpfn.py', 'benchmarks/proofpfn/adapters/kgpfn_worker.py',
        'benchmarks/proofpfn/data.py', 'benchmarks/proofpfn/evaluate.py', 'benchmarks/proofpfn/manifests.py', 'dicee/evaluation/_filtering.py')


@dataclass(frozen=True)
class KGPFN:
    """A checked KGPFN setup.

    Attributes:
        code: The KGPFN code folder.
        checkpoints: The folder of ``kgpfn_icl_3g.pth`` and the TabICL checkpoint.
        python: The command that starts KGPFN's Python (an interpreter or a wrapper).
        pins: The pins of ``models.json`` (``external.kgpfn``).
        gpu_gb: The cap of the worker's CUDA allocator in GB (0: none).
    """

    code: Path
    checkpoints: Path
    python: tuple[str, ...]
    pins: Mapping[str, Any]
    gpu_gb: float = 0.0

    def record(self) -> dict[str, Any]:
        """The setup for the logs and the row files' record."""
        return dict(code=str(self.code), code_commit=self.pins['code_commit'], checkpoints=str(self.checkpoints), python=list(self.python),
                    checkpoint_sha256={CHECKPOINT: self.pins[CHECKPOINT], TABICL: self.pins['tabicl']}, gpu_gb=self.gpu_gb)


def setup(code: Optional[PathLike] = None, checkpoints: Optional[PathLike] = None, python: Optional[str] = None,
          models: Optional[PathLike] = None, gpu_gb: float = 0.0) -> KGPFN:
    """The checked KGPFN setup (options, else ``$KGPFN_CODE``, ``$KGPFN_CHECKPOINT_DIR`` and ``$KGPFN_PYTHON``).

    Checks the commit, the KGPFN files that the worker loads (``imported``) and the checkpoints against their pins, and
    that the command imports the worker's KGPFN modules in the worker's environment.

    Raises:
        AdapterSetupError: For a missing folder or command, another commit, a file that differs from its pin, or a command
            that does not import KGPFN.
    """
    pins = external_pins(KEY, models)
    folder = external_folder(code, 'KGPFN_CODE', 'KGPFN')
    check_commit(folder, pins['code_commit'], 'KGPFN')
    for name, digest in sorted(pins['imported'].items()):
        check_file(folder / name, digest, 'KGPFN')
    files = external_folder(checkpoints, 'KGPFN_CHECKPOINT_DIR', 'KGPFN checkpoints')
    check_file(files / CHECKPOINT, pins[CHECKPOINT], 'KGPFN')
    check_file(files / TABICL, pins['tabicl'], 'KGPFN')
    command = python if python is not None else os.environ.get('KGPFN_PYTHON')
    if not command:
        raise AdapterSetupError("KGPFN: give the command that starts KGPFN's Python with --python or $KGPFN_PYTHON")
    system = KGPFN(code=folder, checkpoints=files, python=tuple(shlex.split(command)), pins=pins, gpu_gb=gpu_gb)
    with tempfile.TemporaryDirectory(prefix='kgpfn-setup-') as work:
        trial = Path(work) / KEY / 'setup' / 'python'   # the layout <work>/kgpfn/<collection>/<graph> of a worker's folder
        trial.mkdir(parents=True)
        try:
            done = subprocess.run([*system.python, '-B', '-c', IMPORTS], env=worker_environment(system, trial), cwd=trial,
                                  capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AdapterSetupError(f'KGPFN: cannot run {command} ({error})') from None
    if done.returncode:
        raise AdapterSetupError(f"KGPFN: {command} does not import KGPFN's modules ({IMPORTS}): {done.stderr.strip()[-600:]}")
    return system


# ---------------------------------------------------------------------------------------------------------- prepare

def unfillable(graph: GraphData, queries: Queries, num_neg: int) -> int:
    """The ranked queries whose blind context cannot be filled (KGPFN's ``build_context_relation_aware`` would raise).

    A query of relation r needs ``num_neg + 20 - min(20, visible pairs of r)`` entities outside the anchor's visible
    answers in the role of the query.
    """
    tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    heads: dict[tuple[int, int], set[int]] = defaultdict(set)
    pairs: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for h, r, t in as_triples(graph.train):
        tails[h, r].add(t)
        heads[t, r].add(h)
        pairs[r].add((h, t))
    n, bad = graph.num_entities, 0
    for h, r, t in queries.facts:
        need = num_neg + NUM_POS - min(NUM_POS, len(pairs[r]))
        bad += (n - len(tails[h, r]) < need) + (n - len(heads[t, r]) < need)
    return bad


def settings(entities: int) -> tuple[int, int, int]:
    """``(batch, NBFNet head chunk, candidate chunk)`` by the entity count (memory only; 64 at 17k entities peaked at 5.4 GB)."""
    batch = 64 if entities <= 19_000 else 32 if entities <= 38_000 else 16 if entities <= 76_000 else 8
    heads = 64 if entities <= 38_000 else 32 if entities <= 76_000 else 16
    return batch, heads, 256 * 64 // batch


def parse_sha256(folder: Path) -> dict[str, str]:
    """The SHA-256 of the host's parse of each split (the triples in file order), which the worker's parse must equal."""
    return {s: hashlib.sha256('\n'.join('\t'.join(t) for t in read_triples(folder / f'{s}.txt')).encode('utf-8')).hexdigest() for s in SPLITS}


def prepare(system: KGPFN, entry: GraphEntry, graph: GraphData, queries: Queries, folder: Path, source: Path) -> dict[str, Any]:
    """Stage the split files and write ``meta.json`` for the worker; returns the meta.

    Raises:
        ExternalStepError: When no context size of ``NEG_STEPS`` fills every ranked query (the graph cannot run).
        AdapterSetupError: When a staged copy differs from the checked file.
    """
    counts = {k: unfillable(graph, queries, k) for k in NEG_STEPS}
    fill = [k for k in NEG_STEPS if counts[k] == 0]
    if not fill:
        raise ExternalStepError(dict(stage='prepare', reason=f'no context size of {NEG_STEPS} fills every ranked query', unfillable=counts))
    batch, heads, candidates = settings(graph.num_entities)
    stage = folder / 'stage'
    stage.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        shutil.copyfile(source / f'{s}.txt', stage / f'{s}.txt')
        if file_sha256(stage / f'{s}.txt') != graph.hashes[f'{s}.txt']:
            raise AdapterSetupError(f'{entry.name}: the staged {s}.txt differs from the checked file')
    meta = dict(graph=entry.name, key=entry.key, collection=entry.collection, positions=queries.positions,
                reflexive=sorted(graph.relations[r] for r in queries.reflexive), reflexive_ids=sorted(queries.reflexive),
                parse_sha256=parse_sha256(source), num_neg=fill[0], unfillable={str(k): v for k, v in counts.items()}, batch_size=batch,
                entity_chunk=heads, eval_chunk=candidates, dump=graph.num_entities * len(queries.facts) <= DUMP_LIMIT,
                entities=graph.num_entities, relations=graph.num_relations, test_rows=queries.test_rows, ranked_facts=len(queries.facts),
                sha256={f'{s}.txt': graph.hashes[f'{s}.txt'] for s in SPLITS}, pins=dict(system.pins), prepared=time.strftime('%Y-%m-%d %H:%M:%S'))
    (folder / 'meta.json').write_text(json.dumps(meta))
    return meta


# ---------------------------------------------------------------------------------------------------------- run

def worker_command(system: KGPFN, entry: GraphEntry, folder: Path, batch: int, heads: int, candidates: int, num_neg: int,
                   dump: bool) -> list[str]:
    """The command line of the worker for one attempt."""
    command = [*system.python, '-B', '-P', str(WORKER), '--graph-dir', str(folder / 'stage'), '--name', entry.name,
               '--meta', str(folder / 'meta.json'), '--code', str(system.code), '--checkpoints', str(system.checkpoints),
               '--work', str(folder), '--out', str(folder / 'raw.json'), '--batch-size', str(batch), '--entity-chunk', str(heads),
               '--eval-chunk', str(candidates), '--num-neg', str(num_neg), '--gpu-gb', str(system.gpu_gb)]
    return command + (['--dump-scores', str(folder / 'scores.npz')] if dump else [])


def worker_environment(system: KGPFN, folder: Path) -> dict[str, str]:
    """The worker's environment: KGPFN's code folder on ``PYTHONPATH``, no bytecode, the folders that a container wrapper
    mounts, and the cache of the rspmm kernel build in the work root ``KGPFN_WORK``, the one folder that a container writes."""
    work = folder.parents[1]
    return dict(os.environ, PYTHONPATH=str(system.code), PYTHONDONTWRITEBYTECODE='1', TORCH_EXTENSIONS_DIR=str(work / 'torch_extensions'),
                MPLCONFIGDIR=str(folder / 'mpl'), WANDB_MODE='disabled', HF_HUB_OFFLINE='1', KGPFN_CODE=str(system.code),
                KGPFN_CHECKPOINT_DIR=str(system.checkpoints), KGPFN_WORKER_DIR=str(WORKER.parent), KGPFN_WORK=str(work))


def run_worker(system: KGPFN, entry: GraphEntry, folder: Path, meta: Mapping[str, Any]) -> tuple[Optional[int], int]:
    """Run the worker with the meta's settings and, after CUDA out of memory, once more with half the batch.

    Returns:
        ``(exit code, attempts)``; the exit code is ``None`` when the worker ran past ``WORKER_HOURS``.
    """
    batch, heads, candidates = meta['batch_size'], meta['entity_chunk'], meta['eval_chunk']
    env = worker_environment(system, folder)
    for stale in ('raw.json', 'raw.error.json', 'scores.npz'):   # outputs of an earlier run of this graph
        (folder / stale).unlink(missing_ok=True)
    code: Optional[int] = 0
    attempt = 0
    for attempt in (1, 2):
        command = worker_command(system, entry, folder, batch, heads, candidates, meta['num_neg'], meta['dump'])
        logger.info('%s: attempt %d, batch %d, heads %d, candidates %d, negatives %d', entry.name, attempt, batch, heads, candidates,
                    meta['num_neg'])
        with open(folder / f'worker-{attempt}.log', 'w') as log:
            code = call(command, timeout=WORKER_HOURS * 3600, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=folder)
        if code != EXIT_OOM or attempt == 2 or batch <= 1:
            break
        batch, heads, candidates = max(1, batch // 2), max(1, heads // 2), candidates * 2
    return code, attempt


def check_output(raw: Mapping[str, Any], meta_path: Path, imported: Mapping[str, str]) -> None:
    """Refuse a worker output that is incomplete, of another mode or meta, with a failed load or non-finite scores, or of
    KGPFN files other than the pinned ``imported`` ones.

    Raises:
        AdapterSetupError: When the worker loaded other KGPFN files than the pins.
        ValueError: For another such output.
    """
    loaded = raw['provenance']['imported']
    if loaded != imported:
        differ = sorted(name for name in set(loaded) | set(imported) if loaded.get(name) != imported.get(name))
        raise AdapterSetupError(f'KGPFN: the worker loaded other files than the pins (models.json, external.kgpfn.imported): {differ[:8]}')
    if (raw['status'], raw['context_mode'], raw['seed']) != ('complete', 'blind', 0):
        raise ValueError(f"the worker output is {raw['status']}, {raw['context_mode']}, seed {raw['seed']}")
    if raw['settings']['meta_sha256'] != file_sha256(meta_path):
        raise ValueError('the worker used another meta.json')
    if any(raw['strict_load'][k] for k in ('missing', 'shape_mismatch', 'value_mismatch', 'unexpected')):
        raise ValueError('the worker did not load the checkpoint exactly')
    if raw['counts']['nonfinite_rows']:
        raise ValueError('KGPFN returned non-finite scores')


def rows_of(raw: Mapping[str, Any], queries: Queries) -> list[dict[str, Any]]:
    """The rows in the benchmark's order from the worker's per-position ranks.

    Raises:
        ValueError: When a rank is missing, or the rule view is worse than the raw view for a target that is not its anchor.
    """
    rows = []
    for member, head, _, anchor, target, _ in group_targets(queries):
        position = queries.index(member)
        side = 'head' if head else 'tail'
        raw_rank, rule_rank = raw[f'{side}_rank'][position], raw[f'{side}_rank_rule'][position]
        if raw_rank is None or rule_rank is None or not (rule_rank <= raw_rank or target == anchor):
            raise ValueError(f'position {position} ({side}): ranks {raw_rank} and {rule_rank}')
        rows.append({'index': position, 'head': head, KEY: rule_rank, RAW: raw_rank})
    return rows


def worker_entities(folder: Path) -> list[str]:
    """The worker's entity order: names by first appearance (head, then tail) in train, valid and test.

    The worker's parse equals the host's (``parse_sha256`` of the meta), so the host's parse gives its order.
    """
    order: dict[str, int] = {}
    for s in SPLITS:
        for h, _, t in read_triples(folder / f'{s}.txt'):
            order.setdefault(h, len(order))
            order.setdefault(t, len(order))
    return list(order)


def verify(dump: Path, folder: Path, graph: GraphData, queries: Queries, rows: Sequence[Mapping[str, Any]]) -> int:
    """Recompute both views from the dumped scores with the benchmark's ranker; returns the number of rows that differ.

    Raises:
        ValueError: When the dump does not cover the rows.
    """
    data = np.load(dump)
    order = {name: i for i, name in enumerate(worker_entities(folder / 'stage'))}
    permutation = torch.tensor([order[name] for name in graph.entities])   # benchmark id -> worker id
    want = {(r['index'], r['head']): (r[KEY], r[RAW]) for r in rows}
    queries_of = {(queries.index(m), head): (r, anchor, target, others) for m, head, r, anchor, target, others in group_targets(queries)}
    ranker = FilteredRanker(tie_policy='pessimistic')
    got = {}
    for position, head, scores in zip(data['index'].tolist(), data['head'].tolist(), data['scores']):
        r, anchor, target, others = queries_of[position, head]
        view = torch.log_softmax(torch.from_numpy(scores).double()[permutation], 0)
        got[position, head] = view_ranks(view, target, others, anchor, r in queries.reflexive, ranker)
    if set(got) != set(want):
        raise ValueError('the dumped scores do not cover the rows')
    return sum(got[q] != want[q] for q in want)


def evaluate_graph(manifest: Manifest, entry: GraphEntry, out: PathLike, system: KGPFN, *, root: PathLike = REPO,
                   work: Optional[PathLike] = None) -> str:
    """Score one graph with KGPFN and write its row file ``<out>/<collection>/kgpfn/<graph>.json``.

    Args:
        manifest: The graph set.
        entry: The graph.
        out: The output root.
        system: The checked KGPFN setup.
        root: The data root of the manifest paths.
        work: The work root (default ``<out>/.work``).

    Returns:
        ``'exists'`` or ``'done'``.

    Raises:
        manifests.ManifestError: When the graph fails its hash check.
        ExternalStepError: When no context size fills every query, the worker fails, or its ranks do not verify.
        ValueError: When the worker's output is incomplete or of another run.
    """
    final = row_file_path(out, entry, [KEY])
    if final.exists():
        logger.info('%s: exists', entry.name)
        return 'exists'
    graph = load_graph(manifest, entry, root)
    queries = prepare_queries(graph, manifest.sample_rule)
    folder = work_folder(work, out, KEY, entry)
    folder.mkdir(parents=True, exist_ok=True)
    meta = prepare(system, entry, graph, queries, folder, manifest.folder(entry, root))
    code, attempts = run_worker(system, entry, folder, meta)
    if code != 0:
        oom = folder / 'raw.error.json'
        reason = f'stopped after {WORKER_HOURS} h' if code is None else f'the worker exited with code {code}'
        raise ExternalStepError(dict(stage='run', reason=reason, exit_code=code, attempts=attempts,
                                     oom=json.loads(oom.read_text()) if oom.is_file() else None,
                                     log_tail=(folder / f'worker-{attempts}.log').read_text(errors='replace').splitlines()[-30:]))
    raw = json.loads((folder / 'raw.json').read_text())
    check_output(raw, folder / 'meta.json', system.pins['imported'])
    rows = rows_of(raw, queries)
    differ = verify(folder / 'scores.npz', folder, graph, queries, rows) if meta['dump'] else 0
    if differ:
        raise ExternalStepError(dict(stage='verify', reason=f'{differ} rows differ from the ranks of the dumped scores', differing_rows=differ,
                                     attempts=attempts))
    s = raw['settings']
    kgpfn_meta = dict(mode='blind', seed=raw['seed'], batch_size=s['batch_size'], entity_chunk=s['entity_chunk_size'], eval_chunk=s['eval_chunk_size'],
                      num_pos=s['num_pos'], num_neg=s['num_neg'], unfillable_contexts=meta['unfillable'], attempts=attempts,
                      reflexive_names=meta['reflexive'], verified=meta['dump'], setup=system.record(), times=raw['times'], gpu=raw['gpu'],
                      provenance=raw['provenance'], rank_mismatches_vs_native=raw['rank_mismatches_vs_native'],
                      queries_with_target_tie=raw['queries_with_target_tie'], order=s['order'],
                      seeding=s['seeding'], raw_sha256=file_sha256(folder / 'raw.json'), meta_sha256=s['meta_sha256'], work=str(folder))
    spec = dict(kind='external', system='KGPFN', setting=SETTING)
    write_rows(final, manifest=manifest, entry=entry, graph=graph, queries=queries, key=KEY, spec=spec, protocol=PROTOCOL, rows=rows,
               seconds=raw['times']['total_s'], meta_name='kgpfn_meta', meta=kgpfn_meta, code=CODE)
    for s_name in SPLITS:   # the staged copies of the graph files
        (folder / 'stage' / f'{s_name}.txt').unlink(missing_ok=True)
    return 'done'
