"""Load published UltraQuery and Is-CQA-Complex +H files without PyG.

Graph splits and public inverse IDs follow DeepGraphLearning/ULTRA's
ultra/datasets_query.py at 427966ad8ed60420eef034063d44f3153addff90.
"""

import hashlib
import pickle
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ._query import QUERY_SHAPES, compile_query, index_query, nested
from .catalog import (  # noqa: F401
    BENCHMARK_DATASETS,
    INDUCTIVE_VERSIONS,
    IS_CQA,
    PLUS_H_ARCHIVE,
    PLUS_H_DATASETS,
    PLUS_H_FOLDERS,
    PLUS_H_PREFIX,
    TRANSDUCTIVE,
    ULTRA,
    WIKITOPICS,
    dataset_spec,
    inference_graph_for_split,
    query_types_for_dataset,
)
from .context import QueryContext, fingerprint

REFERENCE_COMMIT = ULTRA[1]
PLUS_H_REFERENCE_COMMIT = IS_CQA[1]


def audit_plus_h_filters(root, data):
    """Verify the authors' corrected filters; retain every root hard target.

    The complete graph is label-generation input only. It never leaves this
    function or becomes the benchmark's inference context.
    """
    if data.name not in PLUS_H_DATASETS or data.split != 'test':
        raise ValueError('Filter correction is only defined for +H test labels')
    folder = Path(root) / dataset_spec(data.name)[1]
    graph = folder / 'KG_splits' if data.name == 'ICEWS18+H' else folder
    files, triples = {}, []
    for split in ('train', 'valid', 'test'):
        path = graph / f'{split}.txt'
        payload = path.read_bytes()
        files[path.relative_to(folder).as_posix()] = hashlib.sha256(payload).hexdigest()
        triples.extend(tuple(map(int, line.split())) for line in payload.splitlines() if line.strip())
    full = QueryContext(triples, data.context.num_entities, data.context.num_relations, data.context.inverse_relations)
    author_filters = {}
    if data.name != 'ICEWS18+H':
        for shape in ('2in', 'pni'):
            selected = {q.query: q for q in data.queries if q.shape == shape}
            if not selected:
                continue
            source = {}
            for kind in ('queries', 'easy-answers', 'hard-answers'):
                path = folder / 'test-query-reduction' / shape / 'all' / f'test-{kind}.pkl'
                payload = path.read_bytes()
                files[path.relative_to(folder).as_posix()] = hashlib.sha256(payload).hexdigest()
                source[kind] = pickle.loads(payload)
            queries = {q for group in source['queries'].values() for q in group}
            if queries != selected.keys() or set(source['easy-answers']) != queries or set(source['hard-answers']) != queries:
                raise ValueError('Author all/ files must contain exactly the selected query type')
            for query, record in selected.items():
                if source['hard-answers'][query] != record.hard:
                    raise ValueError('Author all/ files changed the released hard targets')
                author_filters[query] = frozenset(source['easy-answers'][query])
    changes, counts = {}, defaultdict(Counter)
    for query in data.queries:
        if 'n' not in query.shape:
            continue
        truth = full.answers(compile_query(query.query))
        if query.hard - truth:
            raise ValueError(f'Released hard targets are not true in the complete graph: {data.name} {query.query}')
        corrected = author_filters.get(query.query, query.easy)
        if corrected != truth - query.hard:
            raise ValueError(f'Author filters disagree with full-graph truth: {data.name} {query.query}')
        remove, add = query.easy - corrected, corrected - query.easy
        counts[query.shape].update(queries=1, hard_answers=len(query.hard), removed=len(remove), added=len(add),
                                   changed_queries=int(bool(remove or add)))
        if remove or add:
            key = query.identity
            changes[key] = dict(remove=sorted(remove), add=sorted(add))
    return dict(version=1, dataset=data.name, source_files=files, per_shape=dict(counts), changes=changes,
                policy='Use author 2in/all and pni/all filters for FB/NELL; retain all root hard targets and other filters.')


def corrected_answer_filters(data, audit):
    lookup = {q.identity: q for q in data.queries}
    if set(audit['changes']) - lookup.keys():
        raise ValueError('Answer-filter audit does not match the benchmark queries')
    return {lookup[key].query: (lookup[key].easy - set(change['remove'])) | set(change['add'])
            for key, change in audit['changes'].items()}


@dataclass(frozen=True)
class BenchmarkQuery:
    shape: str
    query: tuple
    easy: frozenset
    hard: frozenset

    @property
    def identity(self) -> str:
        """Stable ID of the query and its released answers, used by plans, traces and audits."""
        return fingerprint((self.shape, self.query, sorted(self.easy), sorted(self.hard)))


@dataclass
class QueryBenchmark:
    name: str
    group: str
    split: str
    context: QueryContext
    queries: tuple
    candidates: tuple
    metadata: dict = field(default_factory=dict)
    entity_to_idx: dict | None = None
    relation_to_idx: dict | None = None

    def __post_init__(self):
        n, nr = self.context.num_entities, self.context.num_relations
        candidates = set(self.candidates)
        if not candidates or len(candidates) != len(self.candidates) or any(type(i) is not int or not 0 <= i < n for i in candidates):
            raise ValueError('Benchmark candidates must be distinct valid entity IDs')
        if not self.queries:
            raise ValueError('Benchmark has no queries for the selected shapes')
        seen = set()
        for q in self.queries:
            index_query(q.shape, q.query, range(n), range(nr))
            if q.query in seen:
                raise ValueError('Duplicate benchmark query')
            seen.add(q.query)
            if not q.hard or q.easy & q.hard or not (q.easy | q.hard) <= candidates:
                raise ValueError('Answers must be disjoint, include hard answers, and belong to the candidate domain')


def source_query_groups(data, root, shapes):
    """Recover original per-type order for methods whose batches affect scores."""
    files = [name for name in data.metadata['files'] if name.endswith('queries.pkl')]
    if len(files) != 1:
        raise ValueError('Expected one source query file for upstream ordering')
    with (Path(root) / dataset_spec(data.name)[1] / files[0]).open('rb') as stream:
        structured = pickle.load(stream)
    lookup = {query.query: query for query in data.queries}
    return [[lookup[nested(query)] for query in structured[QUERY_SHAPES[shape]]] for shape in shapes]


def download_benchmark(name, root):
    """Download an official archive once and extract its data under root."""
    _, folder, url = dataset_spec(name)
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix='.download-') as temporary:
        archive = Path(temporary) / 'data.zip'
        with urllib.request.urlopen(url, timeout=120) as response, archive.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        with zipfile.ZipFile(archive) as zipped:
            # Validate the complete archive before writing any member.
            for member in zipped.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe path in dataset archive')
            members = zipped.infolist()
            if name in PLUS_H_DATASETS:
                # This release also contains old benchmarks, training queries,
                # and large stratified analyses. Extract only evaluation inputs
                # for all three +H datasets, so the suite downloads only once.
                prefixes = [f'{PLUS_H_PREFIX}/{folder}/' for folder in PLUS_H_FOLDERS.values()]
                inputs = {'id2ent.pkl', 'id2rel.pkl', 'train.txt', 'valid.txt', 'test.txt', 'stats.txt',
                          *(f'{split}-{kind}.pkl' for split in ('valid', 'test')
                            for kind in ('queries', 'easy-answers', 'hard-answers')),
                          *(f'test-query-reduction/{shape}/all/test-{kind}.pkl' for shape in ('2in', 'pni')
                            for kind in ('queries', 'easy-answers', 'hard-answers'))}
                members = [m for m in members for prefix in prefixes if m.filename.startswith(prefix)
                           and (m.filename[len(prefix):] in inputs or
                                m.filename[len(prefix):] in {'KG_splits/train.txt', 'KG_splits/valid.txt', 'KG_splits/test.txt'})]
            zipped.extractall(root, members=members)
    return root / folder


def load_benchmark(root: str | Path, name: str, *, split: str = 'test', query_types: Sequence[str] | None = None,
                   download: bool = False, inference_graph: str | None = None) -> QueryBenchmark:
    """Load a released benchmark split from its official, trusted pickle files.

    Entity and relation IDs and answer labels are kept unchanged. Edges of the
    evaluated split never enter the inference graph: +H test queries use
    train+valid facts unless ``inference_graph='train'``, and validation queries
    always use training facts.

    Args:
        root: Directory into which the official archives are extracted.
        name: A name from ``BENCHMARK_DATASETS`` or ``PLUS_H_DATASETS``.
        split: ``'valid'`` or ``'test'``.
        query_types: Query types to load (default: every type of the dataset).
        download: Download and extract a missing archive into ``root``.
        inference_graph: Facts for +H test queries, ``'train'`` or ``'train+valid'``.

    Returns:
        The queries with their easy and hard answers, inference graph and candidates.

    Raises:
        FileNotFoundError: If the dataset is missing and ``download`` is false.
        ValueError: For an invalid split or query types, or inconsistent released files.
    """
    if split not in ('valid', 'test'):
        raise ValueError('Benchmark split must be valid or test')
    group, folder, url = dataset_spec(name)
    plus_h = name in PLUS_H_DATASETS
    inference_graph = inference_graph_for_split(name, split, inference_graph)
    expected_shapes = query_types_for_dataset(name)
    shapes = tuple(expected_shapes if query_types is None else query_types)
    if not shapes or len(set(shapes)) != len(shapes) or set(shapes) - set(expected_shapes):
        raise ValueError('Select distinct supported query types for this benchmark (use 2u/up for DNF unions)')
    path = Path(root).expanduser() / folder
    if not path.is_dir():
        if not download:
            raise FileNotFoundError(f'{path}: extract {url} under {root}, or pass download=True')
        download_benchmark(name, root)
    checksums = {}

    def source(filename):
        file = path / filename
        with file.open('rb') as stream:
            checksums[filename] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return file

    def unpickle(filename):
        with source(filename).open('rb') as stream:
            return pickle.load(stream)

    def triples(stem):
        if (path / f'{stem}.txt').is_file():
            with source(f'{stem}.txt').open() as stream:
                return [tuple(map(int, line.split())) for line in stream if line.strip()]
        return [tuple(row) for row in torch.load(source(f'{stem}.pt'), map_location='cpu', weights_only=True).tolist()]

    def nodes(edges):
        return sorted({v for h, _, t in edges for v in (h, t)})

    entities = relations = None
    extended = name.startswith('InductiveFB15k237QueryExtendedEval:')
    if group == 'transductive':
        entities, relations = unpickle('id2ent.pkl'), unpickle('id2rel.pkl')
        n, nr = len(entities), len(relations)
        if set(entities) != set(range(n)) or set(relations) != set(range(nr)):
            raise ValueError('Expected contiguous public vocabulary IDs')
        graph_prefix = 'KG_splits/' if name == 'ICEWS18+H' else ''
        edges = triples(graph_prefix + 'train')
        if plus_h and split == 'test' and inference_graph == 'train+valid':
            edges += triples(graph_prefix + 'valid')
        candidates = list(range(n))
        pairs = tuple((r, r + 1) for r in range(0, nr, 2))
        structured = unpickle(f'{split}-queries.pkl')
        easy, hard = unpickle(f'{split}-easy-answers.pkl'), unpickle(f'{split}-hard-answers.pkl')
    else:
        train = triples('train_graph')
        if group == 'inductive-e':
            valid, test = triples('val_inference'), triples('test_inference')
            vocabulary = train + valid + test
            n = max(nodes(vocabulary)) + 1
            nr = max(r for _, r, _ in vocabulary) + 1
            edges = train + (valid if split == 'valid' else test)
        else:
            edges = train if split == 'valid' else triples('test_inference')
            n = max(nodes(edges)) + 1
            nr = max(r for _, r, _ in edges) + 1
            if nodes(edges) != list(range(n)):
                raise ValueError('WikiTopics requires contiguous entity IDs')
        candidates = nodes(edges)
        pairs = tuple((r, r + nr // 2) for r in range(nr // 2))
        if group == 'inductive-er' and (path / 'og_mappings.pkl').is_file():
            # The full relation vocabulary can contain trailing unused IDs;
            # max(observed ID) + 1 need not be twice the inverse offset.
            mapping = unpickle('og_mappings.pkl')['r2id']
            pairs = tuple((r, mapping[label + '_inv']) for label, r in mapping.items()
                          if label + '_inv' in mapping and r < nr and mapping[label + '_inv'] < nr)
        elif nr % 2:
            raise ValueError('A relation mapping is required for an incomplete reciprocal vocabulary')
        structured = unpickle('train_queries.pkl' if extended else f'{split}_queries.pkl')
        easy = None if extended else unpickle(f'{split}_answers_easy.pkl')
        hard = unpickle(f'train_answers_{split}.pkl' if extended else f'{split}_answers_hard.pkl')
    if group == 'transductive' and nr % 2:
        raise ValueError('Official benchmarks require paired direct/inverse relations')
    # These archives already contain reciprocals. Fail rather than silently
    # changing the inference graph if their documented convention does not hold.
    context = QueryContext(edges, n, nr, pairs)
    if set(context.triples) != set(edges):
        raise ValueError('Dataset graph is missing reciprocal edges or uses unexpected inverse IDs')
    records, available = [], set()
    for structure, queries in structured.items():
        shape = next((key for key, value in QUERY_SHAPES.items() if value == nested(structure)), None)
        if shape not in shapes:
            continue
        available.add(shape)
        for i, query in enumerate(queries):
            if group == 'transductive':
                e, h = easy[query], hard[query]
            elif extended:
                e, h = (), hard[structure][i]
            else:
                e, h = easy[structure][query], hard[structure][query]
            records.append(BenchmarkQuery(shape, nested(query), frozenset(e), frozenset(h)))
    if set(shapes) - available:
        raise ValueError(f'Missing requested query types: {sorted(set(shapes) - available)}')
    records.sort(key=lambda q: (q.shape, q.query))
    metadata = dict(reference_commit=PLUS_H_REFERENCE_COMMIT if plus_h else REFERENCE_COMMIT,
                    suite='plus-h' if plus_h else 'ultraquery', files=checksums, url=url,
                    query_types=list(shapes), expected_query_types=list(expected_shapes),
                    evaluation='hardness-balanced-answers' if plus_h else 'faithfulness' if extended else 'hard-answers')
    if plus_h:
        metadata.update(paper='https://arxiv.org/abs/2410.12537', release='benchs-1.0',
                        inference_graph='train+valid' if split == 'test' and inference_graph == 'train+valid' else 'train',
                        answer_filter='published easy answers, including non-selected inference answers')
    return QueryBenchmark(name, group, split, context, tuple(records), tuple(candidates), metadata,
                          {v: k for k, v in entities.items()} if entities else None,
                          {v: k for k, v in relations.items()} if relations else None)
