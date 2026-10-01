"""CPU-only answer-level difficulty analysis of immutable +H rank traces.

Truth edges are label-generation input, never inputs to a predictor. Released
reduction names and minimum missing-positive-edge counts are distinct analyses:
the authors' test reduction files use train+valid (despite the paper's shorthand
"train"); in particular their union reductions are not min-plus edge counts.
"""

import csv
import json
import math
import pickle
import sqlite3
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

import dicee.query_answering
from dicee.query_answering._checkpoint import checksum, read, write_json
from dicee.query_answering._query import PLUS_H_SHAPES, QUERY_SHAPES, compile_query, exact_answers
from dicee.query_answering.benchmark import METRICS
from dicee.query_answering.datasets import PLUS_H_REFERENCE_COMMIT, BenchmarkQuery, dataset_spec

# This postprocessor and the library code that loads queries and computes metrics.
POSTPROCESSOR = (Path(__file__), *(Path(dicee.query_answering.__file__).with_name(name) for name in (
    '_query.py', 'context.py', '_checkpoint.py', 'datasets.py', 'benchmark.py', 'method_evaluation.py')))


def postprocessor_files():
    return {f'{path.parent.name}/{path.name}': checksum(path) for path in POSTPROCESSOR}


# Explicit allowlist from the pinned author release; never treat all/ as a bin.
REDUCTIONS = {
    '2p': ('1p', '2p'), '3p': ('1p', '2p', '3p'), '4p': ('1p', '2p', '3p', '4p'),
    '2i': ('1p', '2i'), '3i': ('1p', '2i', '3i'), '4i': ('1p', '2i', '3i', '4i'),
    'ip': ('1p', '2i', '2p', 'ip'), 'pi': ('1p', '2i', '2p', 'pi'),
    'up': ('1p', '2u', 'up'),
    **{s: ('pos-exist', 'pos-only-miss') for s in ('3in', 'pin', 'inp')},
}


class DifficultyClassifier:
    """Min-plus evaluation on valid full-truth witnesses of anchored trees.

    Cache size is bounded by the total number of entity records, not just keys.
    Negative branches restrict truth and contribute no positive-edge cost.
    """

    def __init__(self, full_outgoing, observed_outgoing, num_entities, *, cache_answers=100000):
        self.full = full_outgoing
        self.observed = observed_outgoing
        self.num_entities = num_entities
        self.cache_answers = cache_answers
        self.cache = OrderedDict()
        self.cached_answers = 0

    def costs(self, node):
        if node in self.cache:
            self.cache.move_to_end(node)
            return self.cache[node]
        op = node[0]
        if op == 'anchor':
            result = {node[1]: 0}
        elif op == 'project':
            result = {}
            for head, cost in self.costs(node[2]).items():
                seen = self.observed.get(head, {}).get(node[1], ())
                for tail in self.full.get(head, {}).get(node[1], ()):
                    value = cost + int(tail not in seen)
                    if tail not in result or value < result[tail]:
                        result[tail] = value
        elif op == 'and':
            positives = [self.costs(c) for c in node[1:] if c[0] != 'not']
            if not positives:
                raise ValueError('Difficulty requires a positive anchored branch')
            smallest = min(positives, key=len)
            result = {a: sum(part[a] for part in positives) for a in smallest if all(a in part for part in positives)}
            for child in node[1:]:
                if child[0] == 'not':
                    # Includes negated paths (pni), and restricts intermediate
                    # witnesses before the final projection in inp.
                    for answer in exact_answers(child[1], self.full, self.num_entities):
                        result.pop(answer, None)
        elif op == 'or':
            result = {}
            for child in node[1:]:
                for answer, value in self.costs(child).items():
                    result[answer] = min(result.get(answer, value), value)
        else:
            raise ValueError(f'Unsupported difficulty operator: {op}')
        size = max(1, len(result))
        if size <= self.cache_answers:
            while self.cache and self.cached_answers + size > self.cache_answers:
                _, evicted = self.cache.popitem(last=False)
                self.cached_answers -= max(1, len(evicted))
            self.cache[node] = result
            self.cached_answers += size
        return result

    def classify(self, shape, query, hard):
        if shape not in PLUS_H_SHAPES:
            raise ValueError(f'Unsupported difficulty shape: {shape}')
        tree = compile_query(query)
        maximum = positive_edges(tree)
        costs = self.costs(tree)
        if set(hard) - costs.keys():
            raise ValueError(f'Hard targets are not true on the full graph: {shape} {query}')
        result = {a: costs[a] for a in hard}
        if any(not 0 <= cost <= maximum for cost in result.values()):
            raise ValueError('Missing-positive-edge count outside tree bounds')
        return result, maximum


def positive_edges(node):
    if node[0] in ('anchor', 'not'):
        return 0
    if node[0] == 'project':
        return 1 + positive_edges(node[2])
    counts = [positive_edges(child) for child in node[1:]]
    if node[0] == 'or':
        if len(set(counts)) != 1:
            raise ValueError('Unequal-size union branches have no single full-inference threshold')
        return counts[0]
    if node[0] == 'and':
        return sum(counts)
    raise ValueError('Unsupported difficulty tree')


def difficulty_name(cost, maximum):
    return 'observed' if cost == 0 else 'full' if cost == maximum else 'partial'


def load_graphs(folder):
    """Read public-ID truth/observed adjacencies once per dataset, CPU only."""
    folder = Path(folder)
    graph = folder / 'KG_splits' if (folder / 'KG_splits').exists() else folder
    full, train, extended = (defaultdict(lambda: defaultdict(set)) for _ in range(3))
    files = {}
    for split in ('train', 'valid', 'test'):
        path = graph / f'{split}.txt'
        files[path.relative_to(folder).as_posix()] = checksum(path)
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                h, r, t = map(int, line.split())
                # Same public reciprocal pairing as load_benchmark.
                for head, relation, tail in ((h, r, t), (t, r ^ 1, h)):
                    full[head][relation].add(tail)
                    if split == 'train':
                        train[head][relation].add(tail)
                    if split != 'test':
                        extended[head][relation].add(tail)
    with (folder / 'id2ent.pkl').open('rb') as stream:
        n = len(pickle.load(stream))
    return full, {'train': train, 'train+valid': extended}, n, files


def root_queries(folder, shapes):
    """Recover trace identity from root labels, including original easy filters.

    Corrected filters do not change RankTrace's identity or its hard targets.
    Loading the raw pickles separately avoids retaining their many unused keys.
    """
    folder = Path(folder)
    with (folder / 'test-queries.pkl').open('rb') as stream:
        structured = pickle.load(stream)
    selected = {q: shape for shape in shapes for q in structured[QUERY_SHAPES[shape]]}
    values = {}
    for kind in ('easy', 'hard'):
        with (folder / f'test-{kind}-answers.pkl').open('rb') as stream:
            raw = pickle.load(stream)
        if selected.keys() - raw.keys():
            raise ValueError(f'Missing root {kind} answers')
        values[kind] = {q: frozenset(raw[q]) for q in selected}
        del raw
    records = {}
    for q, shape in selected.items():
        easy, hard = values['easy'][q], values['hard'][q]
        if not hard or easy & hard:
            raise ValueError('Root hard answers must be nonempty and disjoint from easy answers')
        key = BenchmarkQuery(shape, q, easy, hard).identity
        records[key] = dict(shape=shape, query=q, hard=hard)
    if len(records) != len(selected):
        raise ValueError('Duplicate root query identities')
    return records


def released_labels(folder, records, *, dataset, required=False):
    """Join released hard-answer partitions to the retained root hard subset.

    A wholly absent family is explicitly unavailable unless required. A present
    family must be complete, disjoint and cover every retained hard answer.
    Extra released answers are counted and excluded, never added to evaluation.
    """
    folder = Path(folder)
    by_shape = defaultdict(dict)
    for key, record in records.items():
        by_shape[record['shape']][record['query']] = (key, record['hard'])
    labels, provenance = {}, {}
    for shape, selected in by_shape.items():
        bins = REDUCTIONS.get(shape)
        if bins is None or dataset == 'ICEWS18+H':
            provenance[shape] = dict(status='unavailable', reason='No released reduction partition for this dataset/type')
            continue
        paths = {label: folder / 'test-query-reduction' / shape / label / 'test-hard-answers.pkl' for label in bins}
        present = [p.exists() for p in paths.values()]
        if not any(present) and not required:
            provenance[shape] = dict(status='unavailable', reason='Released reduction files not supplied')
            continue
        if not all(present):
            raise FileNotFoundError(f'Missing released reduction files: {[str(p) for p in paths.values() if not p.exists()]}')
        assignments = {q: {} for q in selected}
        extra = 0
        for label, path in paths.items():
            with path.open('rb') as stream:
                raw = pickle.load(stream)
            for q, answers in raw.items():
                if q not in selected:
                    extra += len(answers)
                    continue
                _, hard = selected[q]
                extra += len(set(answers) - hard)
                for answer in set(answers) & hard:
                    if answer in assignments[q]:
                        raise ValueError(f'Overlapping released reduction labels: {shape} {q} {answer}')
                    assignments[q][answer] = label
        for q, (key, hard) in selected.items():
            if assignments[q].keys() != hard:
                raise ValueError(f'Missing released reduction labels for root hard targets: {shape} {q}')
            labels[key] = assignments[q]
        provenance[shape] = dict(status='available', label_reference_graph='train+valid', excluded_extra_answers=extra,
                                 files={str(p): checksum(p) for p in paths.values()})
    return labels, provenance


def summarize_difficulty(path, records, classifier, *, author_labels=None, shapes=None):
    """Conditional answer→query means plus additive original-query contributions."""
    totals = defaultdict(lambda: dict(queries=0, hard_answers=0, sums=np.zeros((2, 4)),
                                     contribution=np.zeros((2, 4)), query_mass=0.))
    counts = defaultdict(Counter)
    seen = set()
    harmonic = np.zeros(1)
    with sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro', uri=True) as connection:
        for key, shape, payload in connection.execute('SELECT query_id, shape, answers FROM queries ORDER BY position'):
            if shapes is not None and shape not in shapes:
                continue
            if key in seen or key not in records or records[key]['shape'] != shape:
                raise ValueError('Trace query identity/shape does not match root labels')
            seen.add(key)
            record = records[key]
            rows = json.loads(payload)
            if not rows or any(len(row) != 4 or any(type(x) is not int for x in row) for row in rows):
                raise ValueError('Malformed answer-rank records')
            arr = np.asarray(rows, dtype=np.int64)
            answers, ranks, greater, sizes = arr.T
            if len(set(answers)) != len(answers) or set(answers) != record['hard']:
                raise ValueError('Trace answer identities differ from the exact root hard targets')
            if (greater < 0).any() or (sizes < 1).any() or (ranks <= greater).any() or (ranks > greater + sizes).any():
                raise ValueError('Inconsistent filtered rank / tie block')
            last = greater + sizes
            if last.max() >= len(harmonic):
                harmonic = np.concatenate(([0.], np.cumsum(1. / np.arange(1, last.max() + 1))))
            values = np.stack((np.stack((1. / ranks, *(ranks <= k for k in (1, 3, 10))), axis=1),
                               np.stack(((harmonic[last] - harmonic[greater]) / sizes,
                                         *(np.clip(k - greater, 0, sizes) / sizes for k in (1, 3, 10))), axis=1)))
            costs, maximum = classifier.classify(shape, record['query'], record['hard'])
            groups = defaultdict(list)
            released = (author_labels or {}).get(key)
            for index, answer in enumerate(answers):
                cost = costs[answer]
                groups['difficulty', difficulty_name(cost, maximum)].append(index)
                groups['inferred_positive_edges', str(cost)].append(index)
                groups['released_reduction', released[answer] if released else 'unavailable'].append(index)
            groups['overall', 'all'] = list(range(len(answers)))
            counts[shape].update(queries=1, hard_answers=len(answers),
                                 labeled_author_answers=len(answers) if released else 0)
            for (level, label), indices in groups.items():
                row = totals[shape, level, label, maximum]
                selected = values[:, indices, :]
                row['queries'] += 1
                row['hard_answers'] += len(indices)
                row['sums'] += selected.mean(axis=1)
                row['contribution'] += selected.sum(axis=1) / len(answers)
                row['query_mass'] += len(indices) / len(answers)
    if not seen:
        raise ValueError('No completed trace queries for the requested types')
    result = []
    for (shape, level, label, maximum), values in sorted(totals.items()):
        count = counts[shape]['queries']
        result.append(dict(shape=shape, grouping=level, label=label, positive_edges=maximum,
                           one_positive_edge=maximum == 1, queries=values['queries'], hard_answers=values['hard_answers'],
                           parent_queries=count, parent_hard_answers=counts[shape]['hard_answers'],
                           query_weight=values['query_mass'] / count,
                           **{policy: dict(zip(METRICS, values['sums'][i] / values['queries'])) for i, policy in enumerate(('sort', 'expected'))},
                           original_weight_contribution={policy: dict(zip(METRICS, values['contribution'][i] / count))
                                                         for i, policy in enumerate(('sort', 'expected'))}))
    return dict(rows=result, counts=dict(counts), seen=seen)


def export_difficulty_reports(results, bundle, input_root, output, *, labels_root=None, entries=None, query_types=None):
    """Postprocess completed jobs from a frozen inference bundle.

    No frozen-source verification, model loading, checkpoint locks or writes to
    inference artifacts. A new output directory is required. In-progress traces
    without result.json are never opened.
    """
    from dicee.query_answering.method_evaluation import GRAPH_INDEPENDENT_METHODS

    results, bundle, input_root, output = map(lambda p: Path(p).resolve(), (results, bundle, input_root, output))
    if output.exists() or any(output.is_relative_to(p) for p in (results, bundle)):
        raise ValueError('Use a new report directory outside the frozen bundle and inference results')
    source_files = postprocessor_files()
    frozen = read(bundle / 'bundle.json')
    bundle_hash = checksum(bundle / 'bundle.json')
    shapes = set(PLUS_H_SHAPES if query_types is None else query_types)
    if not shapes or shapes - set(PLUS_H_SHAPES):
        raise ValueError('Select supported +H query types')
    selected_entries = set(entries) if entries is not None else None
    jobs = []
    completed = set()
    for path in sorted(results.rglob('result.json')):
        report = read(path)
        run = report.get('benchmark_run', report.get('paper_run'))
        if run is None:
            continue
        if run.get('bundle') == bundle_hash:
            completed.add(run['entry'])
        if selected_entries is not None and run['entry'] not in selected_entries:
            continue
        if run['bundle'] != bundle_hash:
            raise ValueError(f'Result was not produced by this frozen bundle: {path}')
        if report['split'] != 'test':
            raise ValueError('Inference difficulty currently supports test traces only')
        jobs.append((path, report, run))
    if not jobs:
        raise ValueError('No completed matching benchmark results')
    if selected_entries is not None and selected_entries - completed:
        raise ValueError(f'Requested entries are not completed: {sorted(selected_entries - completed)}')
    expected_jobs = {entry['id'] for entry in frozen['manifest']['entries']}
    dataset_jobs = defaultdict(list)
    for job in jobs:
        dataset_jobs[job[1]['dataset']].append(job)
    all_rows, details, sources = [], [], {}
    for dataset, jobs in sorted(dataset_jobs.items()):
        folder = input_root / frozen['manifest']['data_root'] / dataset_spec(dataset)[1]
        available_shapes = shapes & {s for _, r, _ in jobs for s in r['per_shape']}
        if not available_shapes:
            continue
        # Check consumed root labels and all label-generation graph files against
        # the bundle's pins. This intentionally does not compare current code or
        # load/checksum frozen model weights.
        graph_prefix = 'KG_splits/' if dataset == 'ICEWS18+H' else ''
        names = ['test-queries.pkl', 'test-easy-answers.pkl', 'test-hard-answers.pkl', 'id2ent.pkl',
                 *(graph_prefix + s + '.txt' for s in ('train', 'valid', 'test'))]
        pins = {}
        for name in names:
            path = folder / name
            key = str(path.relative_to(input_root))
            digest = checksum(path)
            if frozen['files'].get(key) != digest:
                raise ValueError(f'Consumed label-generation input is not pinned by bundle: {path}')
            pins[name] = digest
        records = root_queries(folder, available_shapes)
        label_folder = (Path(labels_root).resolve() / dataset_spec(dataset)[1]) if labels_root else folder
        author, author_source = released_labels(label_folder, records, dataset=dataset, required=labels_root is not None)
        full, observed, n, graph_files = load_graphs(folder)
        graph = 'train+valid'
        classifier = DifficultyClassifier(full, observed[graph], n)
        # Retain only classifications of root hard targets (not all reachable
        # entities) across model/filter/adapter rows, bounded by this dataset.
        target_labels = {}

        class SharedClassifier:
            def __init__(self, classifier, cache):
                self.classifier = classifier
                self.cache = cache

            def classify(self, shape, query, hard):
                key = (shape, query)
                if key not in self.cache:
                    self.cache[key] = self.classifier.classify(shape, query, hard)
                return self.cache[key]

        sources[dataset] = dict(root_files=pins, truth_graph_files=graph_files, released=author_source)
        available = Counter(record['shape'] for record in records.values())
        for path, report, run in jobs:
            if not (available_shapes & report['per_shape'].keys()):
                continue
            trace = path.parent / 'ranks.sqlite3'
            if checksum(trace) != report['rank_trace_sha256']:
                raise ValueError(f'Rank trace checksum mismatch: {trace}')
            filter_policy = report['protocol'].get('answer_filter', 'released')
            if filter_policy == 'corrected':
                correction = frozen.get('filter_corrections', {}).get(dataset)
                if not correction or checksum(bundle / correction['path']) != correction['sha256']:
                    raise ValueError('Corrected-filter result lacks a pinned correction audit')
            method = report['inference']['method']
            inference_graph = report['dataset_metadata']['inference_graph']
            summary = summarize_difficulty(trace, records, SharedClassifier(classifier, target_labels),
                                           author_labels=author, shapes=available_shapes)
            for row in summary['rows']:
                shape = row['shape']
                if row['grouping'] == 'overall':
                    online_counts = report['per_shape'][shape]
                    if (row['queries'] != online_counts['queries'] or row['hard_answers'] != online_counts['hard_answers']
                            or (report['protocol']['full_split'] and row['queries'] != available[shape])):
                        raise ValueError('Completed trace coverage differs from evaluation/root query coverage')
                    for policy in ('sort', 'expected'):
                        online = (report['per_shape'][shape] if policy == 'sort' else
                                  report['additional_tie_metrics'][policy]['per_shape'][shape])
                        if any(not math.isclose(row[policy][m], online[m], abs_tol=1e-11) for m in METRICS):
                            raise ValueError(f'Offline rank metrics differ from completed evaluation: {path}')
                row.update(entry=run['entry'], dataset=dataset, method=method,
                           calibration=report['inference'].get('calibration'), answer_filter=filter_policy,
                           inference_graph=inference_graph, comparison_graph=graph,
                           graph_independent=method in GRAPH_INDEPENDENT_METHODS,
                           label_reference_graph=graph, label_graph_matches_comparison=True,
                           available_queries=available[shape],
                           complete_shape=row['parent_queries'] == available[shape])
                all_rows.append(row)
            details.append(dict(entry=run['entry'], dataset=dataset, comparison_graph=graph,
                                result=str(path), result_sha256=checksum(path), rank_trace_sha256=report['rank_trace_sha256'],
                                coverage=summary['counts'], available_queries=dict(available),
                                full_selected_types=all(c['queries'] == available[s] for s, c in summary['counts'].items()),
                                complete_benchmark_types=set(summary['counts']) == set(PLUS_H_SHAPES),
                                protocol_full_split=report['protocol']['full_split']))
        del records, full, observed, classifier, target_labels, author
    if not all_rows:
        raise ValueError('No completed results contain selected query types')
    if postprocessor_files() != source_files:
        raise ValueError('Postprocessor source changed during reporting; rerun with stable source')
    payload = dict(version=1, bundle_sha256=bundle_hash, frozen_inference_source_sha256=frozen['source_sha256'],
                   postprocessor_files=source_files,
                   author_commit=PLUS_H_REFERENCE_COMMIT, author_release='benchs-1.0',
                   author_label_reference_graph='train+valid',
                   hardness_reference_graph='train+valid',
                   semantics='Minimum missing positive atoms on full-truth witnesses; negation restricts truth at zero cost; union selects a branch.',
                   averaging='Within each group: answers per query, then participating queries per parent type. No cross-type difficulty macro.',
                   contribution='original_weight_contribution sums to the parent metric within each grouping; query_weight is its original answer-per-query mass.',
                   pending_main_entries=sorted(expected_jobs - completed), selected_types=sorted(shapes),
                   sources=sources, details=details, rows=all_rows)
    write_json(output / 'inference-difficulty.json', payload)
    fields = ['entry', 'dataset', 'method', 'calibration', 'answer_filter', 'inference_graph', 'comparison_graph',
              'graph_independent', 'label_reference_graph', 'label_graph_matches_comparison', 'shape', 'grouping', 'label',
              'positive_edges', 'one_positive_edge', 'queries', 'hard_answers', 'parent_queries', 'parent_hard_answers',
              'available_queries', 'complete_shape', 'query_weight']
    with (output / 'inference-difficulty.csv').open('w', newline='') as stream:
        metric_fields = [f'{p}_{m}' for p in ('sort', 'expected') for m in METRICS]
        writer = csv.DictWriter(stream, fieldnames=fields + metric_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fields} | {f'{p}_{m}': row[p][m] for p in ('sort', 'expected') for m in METRICS})
    lines = ['# Answer-level inference difficulty', '',
             'Metrics in the CSV/JSON are fractions. The table reports MRR ×100. Each row retains its original query type.', '',
             'All hardness categories use train+valid facts as their fixed reference, independently of the predictor\'s inference graph. '
             'Full means all positive atoms in a chosen proof are missing. One-positive-edge types '
             '(including 1p and 2u) are identified explicitly and are not evidence of multi-hop reasoning.', '',
             'Only released hard targets are scored. Negated branches constrain full-graph truth but add no positive-edge cost. '
             'Author union reductions need not equal minimum-edge groups.', '',
             'Group scores average answers within each participating query, then queries. JSON includes additive contributions '
             'under the original query weights. No difficulty macro averages mix parent types.', '',
             f'Pending main jobs: {len(payload["pending_main_entries"])}. Only completed result files were read.', '',
             '| Entry | Filters | Type | Difficulty | Positive atoms | Queries | Answers | Sort | Expected |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in all_rows:
        if row['grouping'] == 'difficulty':
            lines.append(f'| {row["entry"]} | {row["answer_filter"]} | {row["shape"]} | {row["label"]} | '
                         f'{row["positive_edges"]} | {row["queries"]} | {row["hard_answers"]} | '
                         f'{100 * row["sort"]["mrr"]:.2f} | {100 * row["expected"]["mrr"]:.2f} |')
    (output / 'inference-difficulty.md').write_text('\n'.join(lines) + '\n')
    return payload
