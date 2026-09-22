"""Load the published UltraQuery benchmark files without a PyG dependency.

Graph splits and public inverse IDs follow DeepGraphLearning/ULTRA's
ultra/datasets_query.py at 427966ad8ed60420eef034063d44f3153addff90.
"""

import hashlib
import pickle
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ._query import QUERY_SHAPES, index_query, nested
from .context import QueryContext

REFERENCE_COMMIT = '427966ad8ed60420eef034063d44f3153addff90'
TRANSDUCTIVE = {'FB15k237LogicalQuery': 'FB15k-237-betae', 'FB15kLogicalQuery': 'FB15k-betae', 'NELL995LogicalQuery': 'NELL-betae'}
INDUCTIVE_VERSIONS = ('106', '113', '122', '134', '150', '175', '217', '300', '550')
WIKITOPICS = ('art', 'award', 'edu', 'health', 'infra', 'loc', 'org', 'people', 'sci', 'sport', 'tax')
BENCHMARK_DATASETS = (*TRANSDUCTIVE, *(f'InductiveFB15k237Query:{v}' for v in INDUCTIVE_VERSIONS),
                      *(f'WikiTopicsQuery:{v}' for v in WIKITOPICS))


@dataclass(frozen=True)
class BenchmarkQuery:
    shape: str
    query: tuple
    easy: frozenset
    hard: frozenset


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


def dataset_spec(name):
    if name in TRANSDUCTIVE:
        return 'transductive', TRANSDUCTIVE[name], 'https://snap.stanford.edu/betae/KG_data.zip'
    family, _, version = name.partition(':')
    if family in ('InductiveFB15k237Query', 'InductiveFB15k237QueryExtendedEval') and version in INDUCTIVE_VERSIONS:
        return 'inductive-e', version, f'https://zenodo.org/records/7306046/files/{version}.zip'
    if family == 'WikiTopicsQuery' and version in WIKITOPICS:
        return 'inductive-er', f'WikiTopics_QE/{version}', 'https://reltrans.s3.us-east-2.amazonaws.com/WikiTopics_QE.zip'
    raise ValueError(f'Unknown benchmark {name!r}; choose from BENCHMARK_DATASETS')


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
            zipped.extractall(root)
    return root / folder


def load_benchmark(root, name, *, split='test', query_types=None, download=False):
    """Read official trusted pickle files; preserve their IDs and answer labels.

    ``root`` is the shared archive extraction directory, not a dataset's raw
    directory. Validation/test target edges are never attached to the scorer.
    """
    if split not in ('valid', 'test'):
        raise ValueError('Benchmark split must be valid or test')
    shapes = tuple(QUERY_SHAPES if query_types is None else query_types)
    if not shapes or len(set(shapes)) != len(shapes) or set(shapes) - QUERY_SHAPES.keys():
        raise ValueError('Select distinct supported query types (use 2u/up for DNF unions)')
    group, folder, url = dataset_spec(name)
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
        edges = triples('train')
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
    metadata = dict(reference_commit=REFERENCE_COMMIT, files=checksums, url=url,
                    query_types=list(shapes), evaluation='faithfulness' if extended else 'hard-answers')
    return QueryBenchmark(name, group, split, context, tuple(records), tuple(candidates), metadata,
                          {v: k for k, v in entities.items()} if entities else None,
                          {v: k for k, v in relations.items()} if relations else None)
