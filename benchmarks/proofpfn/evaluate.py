"""One graph x one system -> one result file with the filtered rank of every ranked test fact, in both directions.

Per graph (``evaluate_graph``):

1. The graph's files are hash-checked against its manifest, then it loads (``data``); the ranked test facts follow the
   manifest's sample rule.
2. Every system without a result file builds its model on the CPU, attaches train.txt as its inference graph, and moves
   to the device.
3. Query groups (head direction, relation, anchor), in relation order: one call per group, ``forward_k_vs_all([[h, r]])``
   or ``forward_k_vs_all_heads([[r, t]])``, scores asserted finite, then ``log_softmax`` of the float64 scores.
4. Per test member: the filtered pessimistic rank of the target, ``#{v >= v[target]}`` over the candidates outside the
   filter (``FilteredRanker(tie_policy='pessimistic')``).

A result file ``<out>/<manifest>/<label>/<graph>.json`` holds the graph, the system's record, ``metrics`` (MRR and
Hits@1, 3, 10 for both directions, tail queries and head queries) and the rows as three lists of equal length:
``index`` (the position in test.txt), ``head`` (a head query) and ``rank``. The rows are in the order of the query groups.
"""
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from dicee.evaluation._filtering import FilteredRanker
from dicee.models.graph_model import GraphKGE
from dicee.synthetic_kg.checkpoint import atomic_write_bytes
from dicee.synthetic_kg.determinism import THREADS, deterministic_execution, require_cublas_workspace, torch_threads

from .data import GraphData, Queries, load_graph, prepare_queries
from .manifests import REPO, GraphEntry, InputError, Manifest, PathLike
from .systems import OfficialFlock, System, attach, official_setting

logger = logging.getLogger(__name__)

FORMAT = 'proofpfn-ranks-v1'
PROTOCOL = ('inference graph = train.txt; candidates = every entity of train, valid and test; filter = train, valid and test; '
            'both directions; pessimistic ties; log_softmax of the float64 scores')
CUTOFFS = (1, 3, 10)
VIEWS = ('both', 'tail', 'head')


def query_scores(model: GraphKGE, head: bool, h: int, r: int, t: int) -> torch.Tensor:
    """The model's scores over all entities for one query, on the CPU.

    A head query calls ``forward_k_vs_all_heads([[r, t]])`` and a tail query ``forward_k_vs_all([[h, r]])``, both
    without gradients.

    Args:
        model: The model, with its inference graph attached (``systems.attach``).
        head: True for a head query ``(?, r, t)``, False for a tail query ``(h, r, ?)``.
        h: The head entity (unused by a head query).
        r: The relation.
        t: The tail entity (unused by a tail query).

    Returns:
        One score per entity, a 1-D CPU tensor.

    Raises:
        FloatingPointError: When a score is not finite.
    """
    query = torch.tensor([[r, t] if head else [h, r]], device=model.device)
    with torch.no_grad():
        scores: torch.Tensor = (model.forward_k_vs_all_heads(query) if head else model.forward_k_vs_all(query))[0].cpu()
    if not torch.isfinite(scores).all():
        raise FloatingPointError('a model returned a score that is not finite')
    return scores


def rank_queries(model: GraphKGE, queries: Queries) -> tuple[list[int], list[bool], list[int]]:
    """The filtered pessimistic rank of every ranked test fact in both directions.

    Args:
        model: The model, with its inference graph attached.
        queries: The ranked facts, their query groups and the filter.

    Returns:
        ``(index, head, rank)``: per row the position in test.txt, whether it is a head query, and the rank. The rows are
        in the order of the query groups.

    Raises:
        FloatingPointError: When the model returns a score that is not finite.
    """
    ranker = FilteredRanker(tie_policy='pessimistic')
    index, heads, ranks = [], [], []
    for (head, r, anchor), members in queries.groups:
        h0, _, t0 = queries.facts[members[0]]
        view = torch.log_softmax(query_scores(model, head, h0, r, t0).double(), 0)
        known = queries.known[head][(r, anchor)]
        for member in members:
            h, _, t = queries.facts[member]
            target = h if head else t
            index.append(queries.index(member))
            heads.append(head)
            ranks.append(int(ranker.rank(view, target, sorted(known - {target}), exclude_target=False)))
    return index, heads, ranks


def metrics_of(heads: Sequence[bool], ranks: Sequence[int]) -> dict[str, dict[str, float]]:
    """MRR and Hits@1, 3, 10 for both directions, tail queries and head queries.

    Args:
        heads: Per row whether it is a head query.
        ranks: Per row the rank.

    Returns:
        View (``both``, ``tail``, ``head``) to ``MRR``, ``H@1``, ``H@3`` and ``H@10``. A view without rows is left out.
    """
    rank, is_head = np.asarray(ranks, dtype=np.float64), np.asarray(heads, dtype=bool)
    out = {}
    for view, part in (('both', rank), ('tail', rank[~is_head]), ('head', rank[is_head])):
        if len(part):
            out[view] = dict(MRR=float((1.0 / part).mean()), **{f'H@{c}': float((part <= c).mean()) for c in CUTOFFS})
    return out


def result_path(out: PathLike, manifest: Manifest, label: str, entry: GraphEntry) -> Path:
    """``<out>/<manifest>/<label>/<graph>.json``.

    Args:
        out: The output root.
        manifest: The graph set.
        label: The system label.
        entry: The graph.

    Returns:
        The path of the result file.
    """
    return Path(out) / manifest.name / label / f'{entry.name}.json'


def _result(manifest: Manifest, entry: GraphEntry, graph: GraphData, queries: Queries, system: System, setting: Optional[Mapping[str, Any]],
            rows: tuple[list[int], list[bool], list[int]], seconds: float, device: torch.device) -> dict[str, Any]:
    """The content of a result file, after a check of the rows (order and rank range)."""
    index, heads, ranks = rows
    order = [(queries.index(m), head) for (head, _, _), members in queries.groups for m in members]
    if list(zip(index, heads)) != order or len(ranks) != 2 * len(queries.facts):
        raise ValueError(f'{entry.name}: the rows are not the query groups in order')
    if any(not 1 <= rank <= graph.num_entities for rank in ranks):
        raise ValueError(f'{entry.name}: a rank outside 1..{graph.num_entities}')
    result: dict[str, Any] = dict(
        format=FORMAT, graph=entry.name, manifest=dict(name=manifest.name, sha256=manifest.sha256), label=system.label, system=system.spec(),
        model_sha256=system.sha256(), protocol=PROTOCOL, entities=graph.num_entities, relations=graph.num_relations, test_rows=queries.test_rows,
        ranked_facts=len(queries.facts), sample_rule=manifest.sample_rule, files=dict(graph.hashes), metrics=metrics_of(heads, ranks),
        seconds=seconds, index=index, head=heads, rank=ranks)
    if setting is not None:
        result['flock_setting'] = dict(setting)
    result['provenance'] = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=str(device),
                                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None, deterministic=True,
                                threads=torch.get_num_threads())
    return result


def _check_stored(path: Path, manifest: Manifest, system: System) -> None:
    """Refuse a stored result file that belongs to another manifest file, sample rule or model file.

    A stored result counts as done, so its label must still mean the same graph set and the same model.

    Raises:
        manifests.InputError: When the stored file differs in one of the three.
    """
    stored = read_result(path)
    differs = [what for what, old, new in (('manifest file', stored.get('manifest', {}).get('sha256'), manifest.sha256),
                                           ('sample rule', stored.get('sample_rule'), manifest.sample_rule),
                                           ('model file', stored.get('model_sha256'), system.sha256())) if old != new]
    if differs:
        raise InputError(f'{path} is a result of another {" and ".join(differs)}; remove it, or choose another --out or model name')


def evaluate_graph(manifest: Manifest, entry: GraphEntry, systems: Sequence[System], out: PathLike, *, root: PathLike = REPO,
                   device: Union[str, torch.device] = 'cuda') -> dict[str, str]:
    """Evaluate one graph with every system that has no result file yet, and write the result files.

    Runs under ``deterministic_execution`` with ``THREADS`` CPU threads; both are restored afterwards. On CUDA,
    ``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA starts.

    Args:
        manifest: The graph set.
        entry: The graph.
        systems: The systems; their labels name the result folders.
        out: The output root.
        root: The data root of the manifest paths.
        device: The scoring device.

    Returns:
        Label to ``'exists'`` (the result file was there), ``'done'`` (written now) or ``'oom'`` (the system ran out of
        CUDA memory on this graph; no file is written).

    Raises:
        manifests.ManifestError: When the graph fails its hash check.
        manifests.InputError: For systems with the same label, for a stored result file of another manifest file, sample
            rule or model file, and for a graph without a test fact.
        systems.SystemCheckError: When a system cannot run on the graph.
        FloatingPointError: When a model returns a score that is not finite.
        RuntimeError: On CUDA, when ``CUBLAS_WORKSPACE_CONFIG`` is not a deterministic setting.
    """
    device = torch.device(device)
    labels = [s.label for s in systems]
    if len(set(labels)) != len(labels):
        raise InputError(f'system labels must be unique: {labels}')
    statuses: dict[str, str] = {}
    for system in systems:
        path = result_path(out, manifest, system.label, entry)
        if path.exists():
            _check_stored(path, manifest, system)
            statuses[system.label] = 'exists'
    todo = [s for s in systems if s.label not in statuses]
    if not todo:
        return statuses
    if device.type == 'cuda':
        require_cublas_workspace()
    with deterministic_execution(), torch_threads(THREADS):
        graph = load_graph(entry, root)
        queries = prepare_queries(graph, manifest.sample_rule)
        if not queries.facts:
            raise InputError(f'{entry.name}: test.txt holds no fact to rank')
        for system in todo:
            began = time.monotonic()
            system.check()
            setting = official_setting(entry, graph) if isinstance(system, OfficialFlock) else None
            model = attach(system.build_model(graph.num_entities, graph.num_relations, setting), graph.train, graph.num_entities,
                           graph.num_relations, device, system.query_batch_size(setting))
            try:
                rows = rank_queries(model, queries)
            except torch.cuda.OutOfMemoryError:
                logger.error('%s: %s ran out of CUDA memory', entry.name, system.label)
                statuses[system.label] = 'oom'
                continue
            finally:
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            result = _result(manifest, entry, graph, queries, system, setting, rows, time.monotonic() - began, device)
            atomic_write_bytes(result_path(out, manifest, system.label, entry), json.dumps(result).encode())
            statuses[system.label] = 'done'
            logger.info('%s: %s MRR %.4f (%d rows, %.0f s)', entry.name, system.label, result['metrics']['both']['MRR'], len(rows[2]),
                        result['seconds'])
    return {label: statuses[label] for label in labels}


def read_result(path: PathLike) -> dict[str, Any]:
    """A result file.

    Args:
        path: The file.

    Returns:
        Its content.

    Raises:
        manifests.InputError: When the file is not a result file of this format.
    """
    try:
        result: dict[str, Any] = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise InputError(f'cannot read {path}: {error}') from None
    if not isinstance(result, dict) or result.get('format') != FORMAT:
        raise InputError(f'{path} is not a {FORMAT} file')
    return result
