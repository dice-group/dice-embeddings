"""AnyBURL, trained per graph, as rows of the benchmark (README, section AnyBURL; PROTOCOL, section External baselines).

Port of the research runner ``anyburl-ref-v1-runtime/anyburl_run.py`` (view ``ag``). AnyBURL learns rules on the
inference graph (train.txt) of each graph in the KR 2026 baseline setting (Meilicke et al., "A Simple Baseline for
Inductive Knowledge Base Completion"), then ranks the test queries:

- Input files (``write_inputs``): entity ``i`` and relation ``j`` become ``e<i>`` and ``r<j>`` (AnyBURL drops names shorter
  than two characters and splits rule atoms at ``(`` and ``,``). ``valid.txt`` also holds the test rows that are not
  ranked, so that Apply's filter is the benchmark's (asserted).
- Learn: ``KR_LEARN``, the setting of ``suppl/exp/fb237/config-learn.properties`` (checked), with 4 threads: one rule
  file after 30 seconds. Apply: maxplus with ``TOP_K_OUTPUT`` = min(1000, entities). Eval: AnyBURL's own metrics.
- Each Java step runs as ``java -Xmx6G`` under ``systemd-run --user --scope -p MemoryMax=8G`` (unless the scope is off),
  one at a time (a lock file of the work root), and stops after ``STEP_HOURS``.
- Ranks (``rebuild``): the benchmark's pessimistic filtered rank (``view_ranks``) on AnyBURL's confidences, ``-inf`` for an
  unlisted candidate. An unlisted target, and a target tied with the last entry of a full list, get the number of
  allowed candidates. The native ranks of the list order must agree with Eval to its four printed decimals.
"""
import fcntl
import gzip
import logging
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from dicee.evaluation._filtering import FilteredRanker

from ..data import GraphData, Queries, as_triples, load_graph, prepare_queries
from ..evaluate import row_file_path, write_json
from ..manifests import REPO, GraphEntry, Manifest, PathLike, file_sha256
from . import AdapterSetupError, ExternalStepError, call, check_file, external_folder, external_pins, group_targets, view_ranks, work_folder, write_rows

logger = logging.getLogger(__name__)

KEY, RAW = 'anyburl', 'anyburl:raw'
VARIABLE = 'ANYBURL_HOME'
JAR, SOURCES = 'AnyBURL-23-1x-java11.jar', 'AnyBURL-23-1x-sources.zip'
KR_LEARN_FILE = Path('suppl') / 'exp' / 'fb237' / 'config-learn.properties'
KR_LEARN = {'THRESHOLD_CORRECT_PREDICTIONS': '2', 'THRESHOLD_CORRECT_PREDICTIONS_ZERO': '2', 'ZERO_RULES_ACTIVE': 'true',
            'MAX_LENGTH_CYCLIC': '1', 'SNAPSHOTS_AT': '30', 'BATCH_TIME': '1000'}
THREADS, TOP_K, EVAL_TOP_K, EVAL_SCAN = 4, 1000, 1000, 100   # EVAL_SCAN: AnyBURL's HitsAtK.ATKMAX
JVM = ('-Xmx6G', '-XX:+ExitOnOutOfMemoryError', f'-XX:ParallelGCThreads={THREADS}')
SCOPE = ('systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=8G', '-p', 'MemorySwapMax=0')
STEP_HOURS = 6.0
LABEL = 'trained per graph'
SETTING = 'AnyBURL-23-1x (KR 2026 baseline setting, 30 s of rule learning on train.txt, 4 threads), trained per graph'
PROTOCOL = ('AnyBURL reference, trained per graph: rules learned on train.txt (KR 2026 setting, 30 s), applied to train.txt; '
            'candidates=all entities of train+valid+test; filter=train+valid+all test facts; both directions; pessimistic '
            'ties; a target that AnyBURL does not list (or that ties at the cut of a full list) = worst rank; keys anyburl '
            '(self-loop rule share1) and anyburl:raw (no rule)')
CODE = ('benchmarks/proofpfn/adapters/__init__.py', 'benchmarks/proofpfn/adapters/anyburl.py', 'benchmarks/proofpfn/data.py',
        'benchmarks/proofpfn/evaluate.py', 'benchmarks/proofpfn/manifests.py', 'dicee/evaluation/_filtering.py')
EVAL_LINE = re.compile(r'\s*(\d\.\d{4}\s+){3}\d\.\d{4}\s*')
Candidates = tuple[np.ndarray, np.ndarray]


def properties(path: PathLike) -> dict[str, str]:
    """The ``KEY = value`` lines of a Java properties file (AnyBURL's files have no continuation lines)."""
    pairs = (x.split('=', 1) for x in Path(path).read_text().splitlines() if '=' in x and not x.lstrip().startswith('#'))
    return {k.strip(): v.strip() for k, v in pairs}


def write_properties(path: Path, values: Mapping[str, Any]) -> dict[str, str]:
    """Write a properties file; returns its values with paths relative to the file's folder."""
    path.write_text(''.join(f'{k} = {v}\n' for k, v in values.items()))
    prefix = f'{path.parent}/'
    return {k: str(v).replace(prefix, '') for k, v in values.items()}


@dataclass(frozen=True)
class AnyBURL:
    """A checked AnyBURL folder and the way its Java steps run (``setup``).

    Attributes:
        home: The AnyBURL folder (jar, sources archive, ``suppl/``).
        pins: The pins of ``models.json`` (``external.anyburl``).
        java: The Java 11 executable.
        scope: The command prefix of every Java step (the memory cap); empty to run Java directly.
        version: ``java -version`` as one line.
    """

    home: Path
    pins: Mapping[str, str]
    java: str = 'java'
    scope: tuple[str, ...] = SCOPE
    version: str = ''

    def record(self) -> dict[str, Any]:
        """The setup for the logs."""
        return dict(home=str(self.home), java=self.version, scope=list(self.scope))

    def run(self, main: str, config: Path, log: Path, lock: Path) -> dict[str, Any]:
        """One AnyBURL step (``Learn``, ``Apply``, ``Eval``) on a configuration file, one Java process at a time.

        The scope runs the JVM as this process's child, so ``call`` gives the CPU seconds and the peak memory of the JVM.
        The JVM stops after ``STEP_HOURS``, and also when this process stops while it runs (a signal, an error), before
        the lock is released.

        Returns:
            The exit code (``None`` when stopped), wall and CPU seconds, peak resident memory and whether the step was
            stopped.
        """
        command = [*self.scope, self.java, *JVM, '-cp', str(self.home / JAR), f'de.unima.ki.anyburl.{main}', str(config)]
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, 'w') as held, open(log, 'w') as output:
            fcntl.flock(held, fcntl.LOCK_EX)
            end = call(command, timeout=STEP_HOURS * 3600, stdout=output, stderr=subprocess.STDOUT, cwd=config.parent)
        usage = end.usage
        return dict(code=end.code, wall=end.seconds, cpu=usage.ru_utime + usage.ru_stime, user=usage.ru_utime, sys=usage.ru_stime,
                    max_rss_mb=usage.ru_maxrss / 1024, timed_out=end.code is None)


def setup(code: Optional[PathLike] = None, models: Optional[PathLike] = None, java: str = 'java', scope: bool = True) -> AnyBURL:
    """The checked AnyBURL folder: ``code``, else ``$ANYBURL_HOME``, with the pins of ``models.json`` (``external.anyburl``).

    Checks the jar and the sources archive against their pins, the KR learning setting, and that Java starts (in the
    memory scope, unless ``scope`` is off).

    Raises:
        AdapterSetupError: For a missing folder, a file that differs from its pin, another learning setting, no
            ``systemd-run`` for the scope, or no Java.
    """
    home, pins = external_folder(code, VARIABLE, 'AnyBURL'), external_pins(KEY, models)
    for name in (JAR, SOURCES):
        check_file(home / name, pins[name], 'AnyBURL')
    if not (home / KR_LEARN_FILE).is_file():
        raise AdapterSetupError(f'AnyBURL: missing {home / KR_LEARN_FILE} (the supplementary material of the KR paper)')
    kr = {k: v for k, v in properties(home / KR_LEARN_FILE).items() if not k.startswith('PATH_') and k != 'WORKER_THREADS'}
    if kr != KR_LEARN:
        raise AdapterSetupError(f'AnyBURL: {KR_LEARN_FILE} holds {kr}, not the KR setting {KR_LEARN}')
    java = str(Path(java).absolute()) if os.sep in java else java   # the Java steps run in their work folder
    prefix = SCOPE if scope else ()
    if prefix and shutil.which(prefix[0]) is None:
        raise AdapterSetupError(f'AnyBURL: {prefix[0]} is missing, which caps the memory of each Java step (--no-scope runs Java without it)')
    try:
        done = subprocess.run([*prefix, java, '-version'], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AdapterSetupError(f'AnyBURL: cannot run {java} ({error})') from None
    lines = done.stderr.strip().splitlines()
    if done.returncode or not lines:
        raise AdapterSetupError(f'AnyBURL: {java} -version failed: {done.stderr.strip()[:300]}')
    return AnyBURL(home=home, pins=pins, java=java, scope=prefix, version=' | '.join(lines))


def step_failed(step: str, times: Mapping[str, Any], log: Path, why: str) -> ExternalStepError:
    """The error of a failed Java step, with the reason read from its log and exit status."""
    text = log.read_text(errors='replace') if log.is_file() else ''
    if 'OutOfMemoryError' in text:
        why = 'out of memory (JVM heap -Xmx6G)'
    elif times['code'] in (-9, 137) and not times['timed_out']:
        why = 'killed (memory cap MemoryMax=8G of the scope)'
    elif times['timed_out']:
        why = f'stopped after {STEP_HOURS} h'
    return ExternalStepError(dict(stage=step, reason=why, **times, log=log.name, log_tail=text.splitlines()[-15:]))


# ---------------------------------------------------------------------------------------------------------- steps

def write_inputs(work: Path, graph: GraphData, queries: Queries) -> dict[str, str]:
    """Write AnyBURL's ``train.txt``, ``valid.txt`` (valid plus the unranked test rows) and ``test.txt`` (the ranked facts)
    as lines ``e<h> r<r> e<t>``; returns their SHA-256.

    Asserted: each file holds exactly its rows, and AnyBURL's filter (train + valid.txt + test.txt) equals the benchmark's
    (train + valid + all test facts).

    Raises:
        ValueError: When a file or the filter differs.
    """
    test = as_triples(graph.test)
    ranked = set(range(len(test))) if queries.positions is None else set(queries.positions)
    files = {'train.txt': as_triples(graph.train), 'valid.txt': as_triples(graph.valid) + [fact for i, fact in enumerate(test) if i not in ranked],
             'test.txt': list(queries.facts)}
    for name, rows in files.items():
        (work / name).write_text(''.join(f'e{h}\tr{r}\te{t}\n' for h, r, t in rows))
        back = [tuple(int(x[1:]) for x in line.split('\t')) for line in (work / name).read_text().splitlines()]
        if back != rows:
            raise ValueError(f'{name}: the written rows differ')
    every = set(as_triples(graph.train)) | set(as_triples(graph.valid)) | set(as_triples(graph.test))
    if set().union(*map(set, files.values())) != every or len(files['test.txt']) + len(files['valid.txt']) != len(graph.test) + len(graph.valid):
        raise ValueError("AnyBURL's filter differs from train + valid + all test facts")
    return {name: file_sha256(work / name) for name in files}


def learn(anyburl: AnyBURL, work: Path, lock: Path) -> dict[str, Any]:
    """Learn in the KR setting: one rule file ``rules-30`` (Learn ends with exit code 1 after its last snapshot).

    Raises:
        ExternalStepError: When Learn fails or its rule file is missing or incomplete.
    """
    config = work / 'config-learn.properties'
    used = write_properties(config, dict(PATH_TRAINING=work / 'train.txt', PATH_OUTPUT=work / 'rules', **KR_LEARN, WORKER_THREADS=THREADS))
    rules = work / f"rules-{KR_LEARN['SNAPSHOTS_AT']}"
    rules.unlink(missing_ok=True)
    log = work / 'learn.log'
    times = anyburl.run('Learn', config, log, lock)
    stored = re.findall(r'>>> stored (\d+) rules', log.read_text(errors='replace'))
    if times['code'] not in (0, 1) or not stored or not rules.is_file():
        raise step_failed('learn', times, log, 'no rule file')
    count = sum(1 for _ in rules.open())
    if count != int(stored[-1]):
        raise step_failed('learn', times, log, f'rule file has {count} lines, Learn stored {stored[-1]}')
    return dict(times, config=used, rules=count, rules_sha256=file_sha256(rules))


def apply(anyburl: AnyBURL, work: Path, k: int, lock: Path, rules: Optional[Path] = None) -> dict[str, Any]:
    """Apply the rule file (default ``rules-30`` of the work folder) with ``TOP_K_OUTPUT`` k: the ranking file ``preds-<k>``.

    Raises:
        ExternalStepError: When Apply fails or writes no ranking file.
    """
    config = work / f'config-apply-{k}.properties'
    predictions = work / f'preds-{k}'
    for stale in (predictions, work / f'preds-{k}.gz'):
        stale.unlink(missing_ok=True)
    rules = rules if rules is not None else work / f"rules-{KR_LEARN['SNAPSHOTS_AT']}"
    used = write_properties(config, dict(PATH_TRAINING=work / 'train.txt', PATH_VALID=work / 'valid.txt', PATH_TEST=work / 'test.txt',
                                         PATH_RULES=rules, PATH_OUTPUT=predictions, WORKER_THREADS=THREADS, TOP_K_OUTPUT=k))
    log = work / f'apply-{k}.log'
    times = anyburl.run('Apply', config, log, lock)
    if times['code'] != 0 or not predictions.is_file() or '* done with rule application' not in log.read_text(errors='replace'):
        raise step_failed('apply', times, log, 'no ranking file')
    return dict(times, config=used)


def native_eval(anyburl: AnyBURL, work: Path, k: int, lock: Path) -> dict[str, Any]:
    """AnyBURL's Eval on the ranking file ``preds-<k>``: Hits@1, Hits@3, Hits@10 and MRR as printed (4 decimals).

    Raises:
        ExternalStepError: When Eval fails or prints no result line.
    """
    config = work / f'config-eval-{k}.properties'
    used = write_properties(config, dict(PATH_TRAINING=work / 'train.txt', PATH_VALID=work / 'valid.txt', PATH_TEST=work / 'test.txt',
                                         PATH_PREDICTIONS=work / f'preds-{k}', TOP_K=EVAL_TOP_K))
    log = work / f'eval-{k}.log'
    times = anyburl.run('Eval', config, log, lock)
    lines = [x.split() for x in log.read_text(errors='replace').splitlines() if EVAL_LINE.fullmatch(x)]
    if times['code'] != 0 or not lines:
        raise step_failed('eval', times, log, 'no result line')
    return dict(times, config=used, **dict(zip(('H@1', 'H@3', 'H@10', 'MRR'), map(float, lines[-1]))))


# ---------------------------------------------------------------------------------------------------------- ranks

def candidates(line: str, tag: str) -> Candidates:
    """One ``Heads: `` or ``Tails: `` line: the tab-separated candidates and confidences in AnyBURL's order, as ids and floats."""
    if not line.startswith(tag):
        raise ValueError(f'expected {tag!r}: {line[:60]!r}')
    tokens = [x for x in line[len(tag):].rstrip('\n').split('\t') if x]
    names, confidence = tokens[0::2], tokens[1::2]
    if len(names) != len(confidence) or not all(x[0] == 'e' for x in names):
        raise ValueError(f'malformed {tag!r} line: {line[:60]!r}')
    return np.array([int(x[1:]) for x in names], dtype=np.int64), np.array([float(x) for x in confidence], dtype=np.float64)


def read_lists(path: PathLike) -> tuple[dict[tuple[int, int, int], tuple[Candidates, Candidates]], int]:
    """AnyBURL's ranking file (``RuleEngine.writeTopKCandidates``; ``eval.ResultSet`` reads it alike).

    Per test triple the line ``e<h> r<r> e<t>``, then ``Heads: `` and ``Tails: `` (in either order); a repeated triple keeps
    its last block, as ``ResultSet`` does. A ``.gz`` file is read decompressed.

    Returns:
        ``({(h, r, t): (head candidates, tail candidates)}, number of blocks)``.
    """
    lists, blocks = {}, 0
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else open(path, encoding='utf-8')) as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if len(line.rstrip('\n')) < 3:
                continue
            h, r, t = line.split()
            if (h[0], r[0], t[0]) != ('e', 'r', 'e'):
                raise ValueError(f'{path.name}: not a triple line: {line[:60]!r}')
            heads, tails = handle.readline(), handle.readline()
            if heads.startswith('Tails:'):
                heads, tails = tails, heads
            lists[int(h[1:]), int(r[1:]), int(t[1:])] = (candidates(heads, 'Heads: '), candidates(tails, 'Tails: '))
            blocks += 1
    return lists, blocks


def rebuild(path: PathLike, graph: GraphData, queries: Queries, k: int) -> tuple[list[dict[str, Any]], list[list[Any]], dict[str, int]]:
    """The rows, AnyBURL's native ranks and the list counts of one ranking file, in the benchmark's row order.

    Args:
        path: The ranking file (or its ``.gz``).
        graph: The graph.
        queries: Its ranked facts, filter and query groups.
        k: The ``TOP_K_OUTPUT`` of the file.

    Returns:
        ``(rows, native, counts)``. A row holds ``index``, ``head``, ``anyburl`` and ``anyburl:raw``; ``native`` holds
        ``[index, head, rank]`` with rank 0 for a target outside the first 100 list positions.

    Raises:
        ValueError: When the file does not hold one block per ranked fact.
    """
    lists, blocks = read_lists(path)
    if blocks != len(queries.facts) or set(lists) != set(queries.facts):
        raise ValueError(f'{Path(path).name}: not one block per ranked fact')
    n, ranker = graph.num_entities, FilteredRanker(tie_policy='pessimistic')
    rows: list[dict[str, Any]] = []
    native: list[list[Any]] = []
    counts: Counter = Counter()
    for member, head, r, anchor, target, others in group_targets(queries):
        h, _, t = queries.facts[member]
        index = queries.index(member)
        entities, confidence = lists[h, r, t][0 if head else 1]
        if len(np.unique(entities)) != len(entities) or (len(entities) and entities.max() >= n):
            raise ValueError(f'{Path(path).name}: a list of ({h}, {r}, {t}) repeats a candidate or names an unknown entity')
        scores = torch.full((n,), -float('inf'), dtype=torch.float64)
        scores[torch.from_numpy(entities)] = torch.from_numpy(confidence)
        worst = n - len(others)
        position = np.flatnonzero(entities == target)
        cut = len(entities) == k < worst and len(position) == 1 and confidence[position[0]] == confidence[-1]
        rule, raw = (worst, worst) if cut else view_ranks(scores, target, others, anchor, r in queries.reflexive, ranker)
        rows.append({'index': index, 'head': head, KEY: rule, RAW: raw})
        p = int(position[0]) if len(position) and position[0] < min(EVAL_SCAN, EVAL_TOP_K) else -1
        known = set(others) | {target}
        native.append([index, head, p - sum(int(e) in known for e in entities[:p]) + 1 if p >= 0 else 0])
        listed_others = np.isin(entities, np.array(others, dtype=np.int64)) if others else np.zeros(len(entities), bool)
        counts.update(queries=1, unlisted=int(len(position) == 0), cut_ties=int(cut), empty=int(len(entities) == 0),
                      full=int(len(entities) == k), filtered_listed=int(listed_others.sum()),
                      anchor_listed=int(anchor != target and anchor in entities), gated_self=int(target == anchor and r not in queries.reflexive),
                      unsorted=int(bool(np.any(np.diff(confidence) > 0))))
    return rows, native, dict(counts)


def native_metrics(native: Sequence[Sequence[Any]]) -> dict[str, float]:
    """MRR and Hits@1, 3, 10 of AnyBURL's native ranks (rank 0 adds nothing)."""
    x = np.array([q[2] for q in native], dtype=np.float64)
    rr = np.divide(1.0, x, out=np.zeros_like(x), where=x > 0)
    return dict(MRR=float(rr.mean()), **{f'H@{c}': float(((x > 0) & (x <= c)).mean()) for c in (1, 3, 10)})


def agrees(printed: Mapping[str, float], rebuilt: Mapping[str, float]) -> bool:
    """Whether the rebuilt native metrics equal AnyBURL's Eval to its 4 printed decimals (half a unit of the last digit)."""
    return all(abs(printed[m] - rebuilt[m]) <= 5e-5 + 1e-9 for m in ('MRR', 'H@1', 'H@3', 'H@10'))


def compress(path: Path) -> Path:
    """Replace a file by its gzip copy; returns the new path."""
    target = path.with_name(path.name + '.gz')
    with open(path, 'rb') as source, gzip.open(target, 'wb', compresslevel=6) as sink:
        shutil.copyfileobj(source, sink)
    path.unlink()
    return target


# ---------------------------------------------------------------------------------------------------------- runner

def evaluate_graph(manifest: Manifest, entry: GraphEntry, out: PathLike, anyburl: AnyBURL, *, root: PathLike = REPO,
                   work: Optional[PathLike] = None) -> str:
    """Learn, apply and rank one graph, and write its row file ``<out>/<collection>/anyburl/<graph>.json``.

    Args:
        manifest: The graph set.
        entry: The graph.
        out: The output root.
        anyburl: The checked AnyBURL folder.
        root: The data root of the manifest paths.
        work: The work root (default ``<out>/.work``).

    Returns:
        ``'exists'`` (the row file was there) or ``'done'``.

    Raises:
        manifests.ManifestError: When the graph fails its hash check.
        ExternalStepError: When a Java step fails (the record of the step and the graph's sizes).
    """
    final = row_file_path(out, entry, [KEY])
    if final.exists():
        logger.info('%s: exists', entry.name)
        return 'exists'
    began = time.monotonic()
    graph = load_graph(manifest, entry, root)
    queries = prepare_queries(graph, manifest.sample_rule)
    folder = work_folder(work, out, KEY, entry)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder.parents[1] / 'java.lock'
    inputs = write_inputs(folder, graph, queries)
    k = min(TOP_K, graph.num_entities)
    steps: dict[str, dict[str, Any]] = {}
    try:
        steps['learn'] = learn(anyburl, folder, lock)
        steps['apply'] = apply(anyburl, folder, k, lock)
        steps['eval'] = native_eval(anyburl, folder, k, lock)
    except ExternalStepError as error:
        error.record.update(entities=graph.num_entities, train=len(graph.train), ranked=len(queries.facts), seconds=time.monotonic() - began)
        raise
    rows, native, counts = rebuild(folder / f'preds-{k}', graph, queries, k)
    rebuilt = native_metrics(native)
    printed = {m: steps['eval'][m] for m in ('MRR', 'H@1', 'H@3', 'H@10')}
    if not agrees(printed, rebuilt):
        logger.warning("%s: AnyBURL's Eval %s differs from the native rebuild %s", entry.name, printed, rebuilt)
    write_json(folder / f'native-{k}.json', dict(top_k=k, scan=EVAL_SCAN, rows=native))
    compress(folder / f'preds-{k}')
    times = {s: {x: steps[s][x] for x in ('wall', 'cpu', 'user', 'sys', 'max_rss_mb', 'code')} for s in steps}
    valid = len(graph.valid)
    meta = dict(label=LABEL, setting=SETTING, jar=JAR, jar_sha256=anyburl.pins[JAR], sources_sha256=anyburl.pins[SOURCES],
                java=anyburl.version, jvm=list(JVM), scope=list(anyburl.scope), threads=THREADS, top_k=k, eval_top_k=EVAL_TOP_K,
                eval_scan=EVAL_SCAN, kr_learn_setting=KR_LEARN, configs={s: steps[s]['config'] for s in steps}, times=times,
                cpu_seconds=steps['learn']['cpu'] + steps['apply']['cpu'], wall_seconds=steps['learn']['wall'] + steps['apply']['wall'],
                rules=steps['learn']['rules'], rules_sha256=steps['learn']['rules_sha256'], inputs=inputs, work=str(folder),
                filter=dict(rule='AnyBURL filter = train.txt + valid.txt + test.txt of the work folder = train + valid + all test facts (asserted)',
                            train=len(graph.train), valid=valid, unranked_test_rows_in_valid_file=len(graph.test) - len(queries.facts),
                            test=len(queries.facts)),
                counts=counts, native=dict(eval=printed, rebuild=rebuilt, agree=agrees(printed, rebuilt)))
    spec = dict(kind='external', system='AnyBURL', label=LABEL, setting=SETTING)
    write_rows(final, manifest=manifest, entry=entry, graph=graph, queries=queries, key=KEY, spec=spec, protocol=PROTOCOL, rows=rows,
               seconds=time.monotonic() - began, meta_name='anyburl_meta', meta=meta, code=CODE)
    return 'done'
