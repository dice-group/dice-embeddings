"""The real subgraph worlds of arms R and M (``pt_real.sample`` = ``rw_prior.real_world`` with ``RW_MAXN`` 20,000).

A world draws one of the train splits of FB15k-237, WN18RR and CoDEx-M uniformly, then a connected entity set by
breadth-first expansion from a random entity with at most 50 random neighbours per expanded entity, until
n ~ LogUniform(400, ``max_entities``) entities. Its facts are the train facts among them without self-loops (at least
500, else a new draw; after ten draws the last one stays). Entities and relations are renumbered at random; 70 to 90 %
of the facts are observed; the queries come from the hidden facts in the task layout of the legacy template; the
negatives are false in the world.

The source graphs are not part of the code. Their folder holds ``<name>/data.pt`` for each name of
``recipe.REAL_SOURCE_SHA256`` (the research files, checked against these pins before they load); it is given to
``real_world``, else read from the environment variable ``PROOFPFN_REAL_SOURCES``. Pretraining passes it to its worker
processes explicitly. ``build_source`` rebuilds a file from the graph's split files.
"""
import os
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Union, cast

import numpy as np
import torch

from ...knowledge_graph import KG
from ..batch import WorldBatch
from ..digests import file_sha256
from ..recipe import REAL_SOURCE_SHA256, record_params
from ..streams import REAL_VERSION, world_stream
from .legacy import template
from .stage2 import query_world

if TYPE_CHECKING:
    from . import FamilySpec

SOURCES_ENV = 'PROOFPFN_REAL_SOURCES'
"""The environment variable that names the folder of the real source graphs."""

NAMES = tuple(REAL_SOURCE_SHA256)
"""The source graphs in the order of the research draw (``rw_prior.NAMES``)."""

NEIGHBOURS = 50
"""Most neighbours of one expanded entity that the breadth-first expansion follows."""

MIN_FACTS, DRAWS = 500, 10
"""A subgraph needs this many facts, or the entity set is drawn again, at most ``DRAWS`` times."""

LAYOUTS: Mapping[str, tuple[tuple[str, ...], bool]] = MappingProxyType({
    'FB15k-237': (('splits', 'entities', 'relations'), False),
    'WN18RR': (('splits', 'entities', 'relations'), False),
    'CoDEx-M': (('entities', 'relations', 'splits'), True),
})
"""The layout of each pinned ``data.pt``: its key order, and whether its split tensors are row-major. Two research builders
wrote the files: ``transfer.prepare_target`` (FB15k-237, WN18RR; dicee's column-major arrays) and
``transfer-reserve-v1-runtime/prepare_codex.py`` (CoDEx-M), with the same ids."""

_SOURCES: dict[tuple[str, str], dict[str, Any]] = {}


def source_folder(folder: Union[str, Path, None] = None) -> Path:
    """The folder of the real source graphs: ``folder``, else ``$PROOFPFN_REAL_SOURCES``.

    Args:
        folder: The folder, or ``None``.

    Returns:
        The folder.

    Raises:
        RuntimeError: When neither is given.
    """
    value = folder if folder is not None else os.environ.get(SOURCES_ENV)
    if not value:
        raise RuntimeError(f'the real worlds need the folder of their source graphs: pass it or set {SOURCES_ENV}')
    return Path(value)


def build_source(name: str, graph: Union[str, Path], folder: Union[str, Path]) -> Path:
    """Build the ``data.pt`` of a real source graph from its split files, as the research did, and check it against its pin.

    dicee's ``KG`` (pandas backend, no reciprocal triples) reads ``train.txt``, ``valid.txt`` and ``test.txt`` and numbers
    the entities and relations by sorted name over all splits. The file holds the splits as int64 id tensors in file
    order (``splits``), the entity names (``entities``) and the relation names (``relations``), in the layout of the
    pinned file (``LAYOUTS``). ``torch.save`` names the archive inside the file after the file it writes, so the file is
    written as ``data.pt.tmp``, as the research wrote it, and then renamed.

    Args:
        name: A name of ``NAMES``.
        graph: The folder with the graph's split files.
        folder: The real-source folder; the file is ``<folder>/<name>/data.pt``.

    Returns:
        The file.

    Raises:
        ValueError: For an unknown name, or a file whose SHA-256 differs from the pin (other split files than the
            research's); nothing is written then.
    """
    if name not in LAYOUTS:
        raise ValueError(f'unknown real source graph {name!r}; expected one of {NAMES}')
    keys, row_major = LAYOUTS[name]
    kg = KG(dataset_dir=str(graph), add_reciprocal=False, eval_model='test', training_technique='NegSample', separator=r'\s+', backend='pandas')
    splits = {s: torch.as_tensor(getattr(kg, f'{s}_set').astype('int64')) for s in ('train', 'valid', 'test')}
    # The pandas backend keeps each vocabulary as a DataFrame sorted by name (KG annotates it as a dictionary).
    value = dict(splits={s: t.contiguous() if row_major else t for s, t in splits.items()}, entities=cast(Any, kg.entity_to_idx)['entity'].tolist(),
                 relations=cast(Any, kg.relation_to_idx)['relation'].tolist())
    path = Path(folder) / name / 'data.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('data.pt.tmp')
    torch.save({key: value[key] for key in keys}, temporary)
    digest = file_sha256(temporary)
    if digest != REAL_SOURCE_SHA256[name]:
        temporary.unlink()
        raise ValueError(f'the {name} split files in {graph} give SHA-256 {digest}, not the pin {REAL_SOURCE_SHA256[name]}')
    os.replace(temporary, path)
    return path


def load_source(name: str, folder: Union[str, Path, None] = None) -> dict[str, Any]:
    """The train split of a source graph with its neighbour lists (``rw_prior.source``), cached per process.

    Args:
        name: A name of ``NAMES``.
        folder: The source folder (``source_folder``).

    Returns:
        ``train`` (int64 ``[F, 3]``), ``n`` (entities), and the neighbours ``nbr`` of every entity in CSR form with ``ptr``.

    Raises:
        ValueError: When the file's SHA-256 differs from its pin.
    """
    path = source_folder(folder) / name / 'data.pt'
    key = (str(path.resolve()), name)
    if key not in _SOURCES:
        digest = file_sha256(path)
        if digest != REAL_SOURCE_SHA256[name]:
            raise ValueError(f'{path} has SHA-256 {digest}; the pin of {name} is {REAL_SOURCE_SHA256[name]}')
        data = torch.load(path, map_location='cpu', weights_only=True)
        train = data['splits']['train'].numpy().astype(np.int64)
        n = len(data['entities'])
        order = np.argsort(np.concatenate((train[:, 0], train[:, 2])), kind='stable')
        ends = np.concatenate((train[:, 2], train[:, 0]))[order]
        starts = np.searchsorted(np.concatenate((train[:, 0], train[:, 2]))[order], np.arange(n + 1))
        _SOURCES[key] = dict(train=train, n=n, nbr=ends, ptr=starts)
    return _SOURCES[key]


def _entity_set(src: dict[str, Any], rng: np.random.Generator, target: int) -> np.ndarray:
    """A connected entity set of ``target`` entities by breadth-first expansion (``rw_prior.entity_set``).

    Returns:
        A boolean mask over the source's entities.
    """
    n, nbr, ptr = src['n'], src['nbr'], src['ptr']
    chosen = np.zeros(n, bool)
    start = int(rng.integers(n))
    while ptr[start + 1] == ptr[start]:
        start = int(rng.integers(n))
    queue, head = [start], 0
    chosen[start] = True
    count = 1
    while head < len(queue) and count < target:
        x = queue[head]
        head += 1
        cand = nbr[ptr[x]:ptr[x + 1]]
        if len(cand) > NEIGHBOURS:
            cand = rng.choice(cand, NEIGHBOURS, replace=False)
        for y in cand.tolist():
            if not chosen[y]:
                chosen[y] = True
                queue.append(y)
                count += 1
                if count >= target:
                    break
    return chosen


def real_world(seed: int, index: int, spec: 'FamilySpec', folder: Union[str, Path, None] = None) -> tuple[dict[str, WorldBatch], dict[str, Any]]:
    """The real subgraph world of stream index ``index`` (``rw_prior.real_world``).

    Args:
        seed: The run seed.
        index: The stream index.
        spec: The family; ``params`` may override ``max_entities``.
        folder: The source folder (``source_folder``).

    Returns:
        The batch and its metadata.
    """
    P = record_params('real', spec.overrides)
    layout = template(seed, index)

    def stream(name: str) -> int:
        return world_stream(REAL_VERSION, seed, index, name)

    rng = np.random.default_rng(stream('world'))
    name = NAMES[int(rng.integers(len(NAMES)))]
    src = load_source(name, folder)
    train = src['train']
    for _ in range(DRAWS):
        target = int(np.exp(rng.uniform(np.log(400), np.log(P['max_entities']))))
        chosen = _entity_set(src, rng, target)
        facts_array = train[chosen[train[:, 0]] & chosen[train[:, 2]]]
        facts_array = facts_array[facts_array[:, 0] != facts_array[:, 2]]
        if len(facts_array) >= MIN_FACTS:
            break
    ents = np.flatnonzero(chosen)
    rels = np.unique(facts_array[:, 1])
    emap = np.full(src['n'], -1)
    emap[ents] = rng.permutation(len(ents))
    rmap = {int(r): i for i, r in enumerate(rng.permutation(rels).tolist())}
    facts = sorted({(int(emap[h]), rmap[int(r)], int(emap[t])) for h, r, t in facts_array.tolist()})
    n, nr = len(ents), len(rels)
    domains: list[set[int]] = [set() for _ in range(nr)]
    ranges: list[set[int]] = [set() for _ in range(nr)]
    for h, r, t in facts:   # the research iteration order of these sets picks the negatives
        domains[r].add(h)
        ranges[r].add(t)
    batch, observed = query_world(layout, facts, domains, ranges, n, nr, rng, stream, (0.7, 0.9), fill=True, permute=False)
    return batch, dict(family='real', source=name, program_slot=layout.program_slot, entities=n, relations_total=nr, context=observed,
                       facts=len(facts), recipe_batch=layout.recipe_batch)
