"""Optional adapters of three external baselines: KGPFN, MERIT and AnyBURL (decision D4 of the ProofPFN cleanup).

Each adapter wraps the logic of its research runner and writes row files that the registered ``fv1_tables.py`` reads:

=======  ==================================================  ===============================  ==================
adapter  research runner                                     row keys                         model hash entry
=======  ==================================================  ===============================  ==================
kgpfn    ``kgpfn-ref-v1-runtime/kgpfn_ag.py``,               ``kgpfn``, ``kgpfn:raw``         none
         ``kgpfn_eval.py``
merit    ``allgraph-v1/ag_merit.py``,                        ``merit<S>``, ``merit<S>:gt2``,  ``merit<S>``
         ``merit-ref-v1-runtime/merit_ref.py``               ``merit<S>:raw``,
                                                             ``merit<S>:raw:gt2``
anyburl  ``anyburl-ref-v1-runtime/anyburl_run.py``           ``anyburl``, ``anyburl:raw``     none
=======  ==================================================  ===============================  ==================

``F`` is the rule view (the anchor gets ``-inf`` unless the relation passes the self-loop rule ``share1``) and ``F:raw``
the plain view; every rank is the benchmark's pessimistic filtered rank (``PROTOCOL.md``). The graphs load and are
hash-checked by ``data.load_graph``, and the ranked facts, filter and query groups come from ``data.prepare_queries``,
so a row file has the benchmark's row order and record layout (``allgraph``). Work files, part files and error records
live in dot folders below the row folder, which ``fv1_tables.py`` skips.

The external code is never part of this repository. Its folder comes from a command-line option or an environment
variable (``external_folder``), its files are checked against the pins of ``manifests/models.json`` (``external`` and,
for MERIT, ``references``), and code that needs another Python environment runs in a child process whose
``PYTHONPATH`` names the external folder. Tests that need the external code skip without it.
"""
import hashlib
import json
import logging
import os
import platform
import socket
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from dicee.evaluation._filtering import FilteredRanker

from ..data import GraphData, Queries
from ..evaluate import metrics_of, view_metrics, write_json
from ..manifests import MANIFESTS, REPO, SAMPLE_RULES, GraphEntry, Manifest, file_sha256

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ROW_FIELDS = ('index', 'head')


class AdapterSetupError(ValueError):
    """An adapter that cannot start: external code or runtime missing, or a file that differs from its pin."""


class ExternalStepError(RuntimeError):
    """A step of the external system failed on a graph (out of memory, no output); the graph gets an error record."""


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


def external_pins(name: str, models: Optional[PathLike] = None) -> dict[str, str]:
    """The pins of an external adapter in the ``external`` section of ``models.json``.

    Raises:
        AdapterSetupError: When the section has no entry for the adapter.
    """
    path = Path(models) if models is not None else MANIFESTS / 'models.json'
    section = json.loads(path.read_text()).get('external', {})
    if name not in section:
        raise AdapterSetupError(f'{path} pins no files of the {name} adapter (section external)')
    return dict(section[name])


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

    Raises:
        AdapterSetupError: When the folder holds no readable git checkout.
    """
    git = Path(folder) / '.git'
    try:
        head = (git / 'HEAD').read_text().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        return next(line.split()[0] for line in (git / 'packed-refs').read_text().splitlines() if line.endswith(' ' + ref))
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

def row_path(out: PathLike, entry: GraphEntry, keys: Sequence[str]) -> Path:
    """``<out>/<collection>/<keys joined by '-'>/<graph>.json``, the path rule of ``evaluate.row_file_path``."""
    return Path(out) / entry.collection / '-'.join(keys) / f'{entry.name}.json'


def error_record_path(final: Path) -> Path:
    """The error record of a graph: ``.errors/<graph>.json`` next to its row file (as ``evaluate.error_path``)."""
    return final.parent / '.errors' / final.name


def work_folder(work: Optional[PathLike], out: PathLike, adapter: str, entry: GraphEntry) -> Path:
    """The work folder of a graph: ``<work>/<adapter>/<collection>/<graph>``; ``work`` defaults to ``<out>/.work``."""
    root = Path(work) if work is not None else Path(out) / '.work'
    return root / adapter / entry.collection / entry.name


def code_sha256(*names: str) -> dict[str, str]:
    """The SHA-256 of files of this repository (paths relative to its root)."""
    return {name: file_sha256(REPO / name) for name in names if (REPO / name).is_file()}


def record_of(manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, key: str,
              spec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``allgraph`` record of an adapter's row file: the fields of ``evaluate.record_of`` that fit one external system.

    ``model_sha256`` stays empty: the registered tables compare no model hash of KGPFN or AnyBURL.
    """
    return dict(entry.record(), sha256=dict(graph.hashes, manifest=manifest.name), test_rows=queries.test_rows,
                ranked_facts=len(queries.facts), sampled=queries.positions is not None, sample_rule=SAMPLE_RULES[manifest.sample_rule],
                positions=queries.positions, families={key: dict(spec)}, model_sha256={}, host=socket.gethostname())


def write_rows(path: Path, *, manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, key: str,
               spec: Mapping[str, Any], protocol: str, rows: list[dict[str, Any]], seconds: float, meta_name: str,
               meta: Mapping[str, Any], code: Sequence[str]) -> dict[str, Any]:
    """Check an adapter's rows and write its row file (the layout of the benchmark's row files).

    Args:
        path: The row file.
        manifest: The graph set.
        entry: The graph.
        graph: The loaded graph.
        queries: Its ranked facts and query groups.
        key: The system key; the rows hold ``key`` (rule view) and ``key:raw``.
        spec: The system's record for ``allgraph.families``.
        protocol: The protocol string of the row file.
        rows: The rows, in the order of ``queries.groups``.
        seconds: The wall seconds of the graph's job.
        meta_name: The name of the adapter's record (``kgpfn_meta``, ``anyburl_meta``).
        meta: The adapter's record.
        code: The repository files whose SHA-256 the provenance records.

    Returns:
        The written document.

    Raises:
        ValueError: When the rows are not the query groups in order, have other keys or a rank outside ``1..|E|``.
    """
    order = [(queries.index(m), head) for (head, _, _), members in queries.groups for m in members]
    if [(r['index'], r['head']) for r in rows] != order or len(rows) != 2 * len(queries.facts):
        raise ValueError(f'{entry.name}: the rows are not the query groups in order')
    keys = [key, f'{key}:raw']
    n = graph.num_entities
    if any(list(r) != [*ROW_FIELDS, *keys] for r in rows) or any(not 1 <= r[k] <= n for r in rows for k in keys):
        raise ValueError(f'{entry.name}: a row has other keys or a rank outside 1..{n}')
    provenance = dict(benchmark='benchmarks.proofpfn', adapter=key, manifest=dict(name=manifest.name, sha256=manifest.sha256),
                      code_sha256=code_sha256(*code), python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, host=socket.gethostname())
    doc = dict(dataset=entry.name, families=[key], entities=n, relations=graph.num_relations, reflexive=sorted(queries.reflexive),
               protocol=protocol, metrics=metrics_of(rows), rows=rows, seconds=seconds,
               allgraph=record_of(manifest, entry, graph, queries, key, spec), metrics_views=view_metrics(rows))
    doc[meta_name] = dict(meta)
    doc['provenance'] = provenance
    write_json(path, doc)
    logger.info('%s: %d rows -> %s', entry.name, len(rows), path)
    return doc


def write_error(final: Path, record: Mapping[str, Any]) -> Path:
    """Write the error record of a graph (``.errors/<graph>.json`` next to its row file); returns its path."""
    path = error_record_path(final)
    write_json(path, dict(record, host=socket.gethostname()))
    return path


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
