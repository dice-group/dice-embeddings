"""Optional adapters of three external baselines: KGPFN, MERIT and AnyBURL.

Each adapter wraps the logic of its research runner and writes row files that the registered ``fv1_tables.py`` reads
(README, section External baselines; PROTOCOL, section External baselines). The graphs, ranked facts, filter and query
groups come from ``data``, and the row files from ``evaluate.write_row_file``. Work files stay in
``<work>/<adapter>/<collection>/<graph>`` (``work_folder``), and an error record in ``.errors/`` next to the row file:
both are dot folders, which ``fv1_tables.py`` skips.

The external code is never part of this repository. Its folder comes from a command-line option or an environment
variable (``external_folder``), and its files are checked against the pins of ``manifests/models.json``.
"""
import os
import resource
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import torch

from dicee.evaluation._filtering import FilteredRanker

from ..data import GraphData, Queries
from ..evaluate import graph_record, provenance_of, write_row_file
from ..manifests import GraphEntry, InputError, Manifest, PathLike, file_sha256, read_models

STOP_SECONDS = 30.0
"""The time between SIGTERM and SIGKILL when an external process stops (``call``)."""
POLL_SECONDS = 0.2
"""The interval at which ``call`` checks whether an external process has ended."""


class AdapterSetupError(InputError):
    """An adapter that cannot start: external code or runtime missing, or a file that differs from its pin."""


class ExternalStepError(RuntimeError):
    """A step of the external system failed on a graph (out of memory, no output); the graph gets an error record.

    Attributes:
        record: The fields of the error record (``stage``, ``reason`` and the step's details).
    """

    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__(f"{record.get('stage')}: {record.get('reason')}")
        self.record = dict(record)


# ------------------------------------------------------------------------------------------------------ setup

def external_folder(option: Optional[PathLike], variable: str, what: str) -> Path:
    """The folder of an external system: the command-line option, else the environment variable.

    Args:
        option: The value of the command-line option, or ``None``.
        variable: The environment variable that names the folder otherwise.
        what: The system, for the error message.

    Returns:
        The folder.

    Raises:
        AdapterSetupError: When neither names an existing folder.
    """
    value = option if option is not None else os.environ.get(variable)
    if not value:
        raise AdapterSetupError(f'{what}: give the folder of its code with the option or ${variable}')
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        raise AdapterSetupError(f'{what}: {folder} is not a folder')
    return folder


def external_pins(name: str, models: Optional[PathLike] = None) -> dict[str, Any]:
    """The pins of an external adapter in the ``external`` section of ``models.json``.

    Raises:
        AdapterSetupError: When the section has no entry for the adapter.
        manifests.ManifestError: For a missing or malformed file.
    """
    external = read_models(models).external
    if name not in external:
        raise AdapterSetupError(f'{models or "models.json"} pins no files of the {name} adapter (section external)')
    return dict(external[name])


def check_file(path: PathLike, sha256: str, what: str) -> str:
    """The SHA-256 of a file that must equal its pin.

    Raises:
        AdapterSetupError: For a missing file or another SHA-256.
    """
    if not Path(path).is_file():
        raise AdapterSetupError(f'{what}: missing {path}')
    digest = file_sha256(path)
    if digest != sha256:
        raise AdapterSetupError(f'{what}: the SHA-256 of {path} is {digest}, the pin {sha256}')
    return digest


def git_head(folder: PathLike) -> str:
    """The commit checked out in a git folder, read from ``.git`` without running git.

    ``.git`` is the git folder, or the file ``gitdir: <path>`` of a worktree or a submodule; a worktree keeps its branch
    refs in the common git folder that its ``commondir`` names.

    Raises:
        AdapterSetupError: When the folder holds no readable git checkout.
    """
    git = Path(folder) / '.git'
    try:
        if git.is_file():
            git = (Path(folder) / git.read_text().strip().removeprefix('gitdir:').strip()).resolve()
        head = (git / 'HEAD').read_text().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[5:]
        common = (git / (git / 'commondir').read_text().strip()).resolve() if (git / 'commondir').is_file() else git
        for base in (git, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()
        return next(line.split()[0] for line in (common / 'packed-refs').read_text().splitlines() if line.endswith(' ' + ref))
    except (OSError, StopIteration) as error:
        raise AdapterSetupError(f'{folder}: no git checkout ({error!r})') from None


def check_commit(folder: PathLike, commit: str, what: str) -> str:
    """The commit of an external checkout, which must equal its pin.

    Raises:
        AdapterSetupError: For another commit.
    """
    head = git_head(folder)
    if head != commit:
        raise AdapterSetupError(f'{what}: {folder} is at commit {head}, the pin {commit}')
    return head


class Finished(NamedTuple):
    """The end of an external process (``call``).

    Attributes:
        code: The exit code (``-N`` after signal N), or ``None`` when the process ran past its timeout.
        seconds: The wall seconds of the process.
        usage: Its resource usage from ``os.wait4``: CPU seconds and peak resident memory.
    """

    code: Optional[int]
    seconds: float
    usage: resource.struct_rusage


def _reap(process: subprocess.Popen, deadline: float) -> Optional[resource.struct_rusage]:
    """The resource usage of a child process once it has ended, or ``None`` at the deadline.

    ``os.wait4`` reaps the process (``Popen.wait`` would drop its usage) and its exit code goes to ``process.returncode``.
    """
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return usage
        if time.monotonic() >= deadline:
            return None
        time.sleep(POLL_SECONDS)


def _stop(process: subprocess.Popen) -> resource.struct_rusage:
    """Stop a child process: SIGTERM, then SIGKILL after ``STOP_SECONDS`` or at a second interruption; returns its usage."""
    try:
        os.kill(process.pid, signal.SIGTERM)   # not Popen.terminate: its poll could reap the process before os.wait4
        usage = _reap(process, time.monotonic() + STOP_SECONDS)
    except BaseException:   # a second interruption
        if process.returncode is None:
            _kill(process)
        raise
    return usage if usage is not None else _kill(process)


def _kill(process: subprocess.Popen) -> resource.struct_rusage:
    """SIGKILL to a child process, which ends at once; returns the usage of the reaped process."""
    os.kill(process.pid, signal.SIGKILL)
    _, status, usage = os.wait4(process.pid, 0)
    process.returncode = os.waitstatus_to_exitcode(status)
    return usage


def call(command: Sequence[str], *, timeout: float, **options: Any) -> Finished:
    """Run an external process to its end.

    At the timeout, and when this process stops while the other one runs (a signal, an error), the other process gets
    SIGTERM and, after ``STOP_SECONDS`` or at a second interruption, SIGKILL (``_stop``). A container that runs with
    ``docker run --init`` gets the SIGTERM through its client; a SIGKILL stops only the client.

    Args:
        command: The command line.
        timeout: The seconds after which the process stops.
        options: Further options of ``subprocess.Popen``.

    Returns:
        The exit code (``None`` after the timeout), the wall seconds and the resource usage.

    Raises:
        OSError: When the command does not start.
    """
    began = time.monotonic()
    process = subprocess.Popen(command, **options)
    timed_out = False
    try:
        usage = _reap(process, began + timeout)
        if usage is None:
            timed_out, usage = True, _stop(process)
    except BaseException:
        if process.returncode is None:   # an interruption of this process: a signal or an error
            _stop(process)
        raise
    return Finished(None if timed_out else process.returncode, time.monotonic() - began, usage)


# ------------------------------------------------------------------------------------------------------ ranks

def view_ranks(scores: torch.Tensor, target: int, others: Sequence[int], anchor: int, rule_open: bool,
               ranker: FilteredRanker) -> tuple[int, int]:
    """``(rule view rank, raw rank)`` of a target from the scores over all entities.

    Both ranks are the benchmark's pessimistic filtered rank (``FilteredRanker(tie_policy='pessimistic')``, equal to
    the research ``#{v >= v[target]}`` over the unfiltered candidates, the target included); the rule view first sets the
    anchor to ``-inf`` unless the relation is open under the self-loop rule (``evaluate.group_rows``).

    Args:
        scores: Float64 scores over all entities, ``-inf`` for a candidate without a score.
        target: The target entity.
        others: The filtered candidates (every known answer except the target).
        anchor: The query's anchor entity.
        rule_open: Whether the relation passes the self-loop rule.
        ranker: A pessimistic ranker.

    Returns:
        The two ranks.
    """
    raw = ranker.rank(scores, target, others, exclude_target=False)
    if rule_open:
        return raw, raw
    gated = scores.clone()
    gated[anchor] = -float('inf')
    return ranker.rank(gated, target, others, exclude_target=False), raw


def group_targets(queries: Queries) -> list[tuple[int, bool, int, int, int, list[int]]]:
    """Every ranked query in the benchmark's row order: ``(member, head, relation, anchor, target, others)``.

    ``member`` indexes ``queries.facts``; ``others`` holds the filtered candidates of the query (sorted).
    """
    out = []
    for (head, r, anchor), members in queries.groups:
        known = queries.known[head][(r, anchor)]
        for member in members:
            h, _, t = queries.facts[member]
            target = h if head else t
            out.append((member, head, r, anchor, target, sorted(known - {target})))
    return out


# ------------------------------------------------------------------------------------------------------ rows

def work_folder(work: Optional[PathLike], out: PathLike, adapter: str, entry: GraphEntry) -> Path:
    """The absolute work folder of a graph: ``<work>/<adapter>/<collection>/<graph>``; ``work`` defaults to ``<out>/.work``.

    The external steps run in their work folder and a container mounts it, so a relative path would point elsewhere.
    """
    root = (Path(work) if work is not None else Path(out) / '.work').absolute()
    return root / adapter / entry.collection / entry.name


def write_rows(final: Path, *, manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, key: str, spec: Mapping[str, Any],
               protocol: str, rows: list[dict[str, Any]], seconds: float, meta_name: str, meta: Mapping[str, Any], code: Sequence[str]) -> None:
    """Write the row file of an adapter with one system (``evaluate.write_row_file``).

    The rows hold ``key`` (the rule view) and ``key:raw``. The ``allgraph`` record has no model hash: the registered tables
    compare none of KGPFN and AnyBURL.

    Args:
        final: The row file.
        manifest: The graph set.
        entry: The graph.
        graph: The loaded graph.
        queries: Its ranked facts and query groups.
        key: The system key.
        spec: The system's record for ``allgraph.families``.
        protocol: The protocol string of the row file.
        rows: The rows, in the order of ``queries.groups``.
        seconds: The wall seconds of the graph's job.
        meta_name: The name of the adapter's record (``kgpfn_meta``, ``anyburl_meta``).
        meta: The adapter's record.
        code: The repository files whose SHA-256 the provenance records.

    Raises:
        ValueError: When the rows are not the query groups in order, have other keys or a rank outside ``1..|E|``.
    """
    record = dict(graph_record(manifest, entry, graph, queries), families={key: dict(spec)}, model_sha256={}, host=socket.gethostname())
    write_row_file(final, entry, graph, queries, [key, f'{key}:raw'], rows, families=[key], protocol=protocol, seconds=seconds, record=record,
                   provenance=provenance_of(manifest, code, adapter=key), extra={meta_name: dict(meta)})
