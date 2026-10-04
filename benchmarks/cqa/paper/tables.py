"""Render saved +H/UltraQuery JSON reports as LaTeX paper tables: three main floats and appendix A1--A11.

Table generation uses only the standard library; --figures uses matplotlib.
It never loads models, datasets,
rank traces, or files referenced inside a report. Run without inputs for the
full default evaluation layout with missing scores, or pass result.json,
comparison.json, inference-difficulty.json, and
*-effects.json files together. Use --help for examples and table descriptions.
"""

import argparse
import functools
import json
import math
import re
import sys
from collections import defaultdict
from copy import copy
from pathlib import Path

from ..manifests import SUITES, catalog, read_manifest, suite_directory

ULTRA_TYPES = catalog().ULTRAQUERY_SHAPES
PLUS_H_TYPES = catalog().PLUS_H_SHAPES
MAIN_HARDNESS_TYPES = ('2p', '3p', '4p', '2i', '3i', '4i', '3in', 'pin', 'inp')
PLUS_H_DATASETS = catalog().PLUS_H_DATASETS
GRAPH_INDEPENDENT_METHODS = catalog().GRAPH_INDEPENDENT_METHODS
FAMILIES = {'transductive': ('Transductive', len(catalog().TRANSDUCTIVE)),
            'inductive-e': ('Inductive (e)', len(catalog().INDUCTIVE_VERSIONS)),
            'inductive-er': ('Inductive (e,r)', len(catalog().WIKITOPICS)),
            'all': ('All datasets', len(catalog().BENCHMARK_DATASETS))}
METRICS = ('mrr', 'hits1', 'hits3', 'hits10')
MISSING = '-'
# Positive-atom bounds from the answer-level classifier.
POSITIVE_EDGES = dict(zip(PLUS_H_TYPES, (1, 2, 3, 2, 3, 3, 3, 1, 2, 1, 2, 2, 2, 1, 4, 4)))
AUTHOR_REDUCTION_TYPES = ('1p', '2p', '3p', '4p', '2i', '3i', '4i', 'pi', 'ip', '2u', 'up')
AUTHOR_QUERY_NAMES = {'pi': '1p2i', 'ip': '2i1p', 'up': '2u1p',
                      'pin': '2pi1pn', 'pni': '2nu1p', 'inp': '2in1p'}
METHOD_NAMES = {'cone': 'ConE', 'gnnqe': 'GNN-QE', 'ultraquery': 'UltraQuery',
                'clmpt': 'CLMPT', 'cqd': 'CQD', 'cqd-hybrid': 'CQD-Hybrid', 'qto': 'QTO',
                'ultra-adapter': 'ULTRA + adapter', 'trix-adapter': 'TRIX + adapter',
                'ultraquery-lp': 'UltraQuery-LP', 'incoming-relation': 'Incoming relation',
                'inductive-gnnqe': 'Inductive GNN-QE'}
# Appendix tables A1--A11; the main paper's compact tables are in summary.py.
TABLE_TITLES = (
    'UltraQuery Results by Dataset Family', 'UltraQuery Results by Freebase Derivation', 'Full UltraQuery Results',
    '+H Per-Query-Type Performance', '+H Performance by Hardness',
    'Full +H Hardness Breakdowns', '+H Performance by Author Query Reduction',
    'Learned vs Identity Adapter', 'Adapter Training Seeds',
    'Protocol Sensitivity', 'Data, Training, and Selection Details',
)
LATEX_ESCAPES = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                 '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                 '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def is_missing(value):
    return (value is None or isinstance(value, str) and not value.strip()
            or isinstance(value, float) and not math.isfinite(value))


def escape(value):
    """Escape input text, including IDs and paths, without interpreting LaTeX."""
    if is_missing(value):
        return MISSING
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return ''.join(LATEX_ESCAPES.get(c, c) for c in ' '.join(str(value).split()))


def number(value, *, scale=100, signed=False):
    if is_missing(value):
        return MISSING
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected a finite numeric metric, got {value!r}')
    if scale == 100 and (abs(value) > 1 or not signed and value < 0):
        raise ValueError(f'Metric outside its fractional range: {value!r}')
    return f'{value * scale:+.2f}' if signed else f'{value * scale:.2f}'


def ci(value):
    if is_missing(value):
        return MISSING
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'Invalid confidence interval: {value!r}')
    if any(is_missing(endpoint) for endpoint in value):
        return MISSING
    if value[0] > value[1]:
        raise ValueError(f'Invalid confidence interval: {value!r}')
    return f'[{number(value[0], signed=True)}, {number(value[1], signed=True)}]'


def family(dataset):
    """UltraQuery dataset family; None for +H and unknown datasets."""
    return catalog().dataset_spec(dataset)[0] if dataset in catalog().BENCHMARK_DATASETS else None


def scope(record):
    return 'full test' if record['complete'] else f"{record['split'] or 'unknown split'} / partial"


def condition(record):
    """Keep alternative graphs, filters, adapters and execution recipes separate."""
    fields = (record['graph'], record['filter'], record['profile'], record['facts'])
    return ', '.join(str(v) for v in fields if v is not None and v != '')


def averages(shapes):
    return {group: {m: sum(values[m] for values in selected) / len(selected)
                    for m in METRICS if all(not is_missing(values.get(m)) for values in selected)} if selected else {}
            for group, selected in (
                ('all', list(shapes.values())),
                ('epfo', [v for s, v in shapes.items() if 'n' not in s]),
                ('negation', [v for s, v in shapes.items() if 'n' in s]))}


def execution_settings(record):
    """Recorded recipe evidence; absent settings stay absent, never become defaults."""
    inference = record.get('raw', {}).get('inference') or {}
    manifest = inference.get('manifest') or {}
    settings = {key: inference.get(key) if inference.get(key) is not None else manifest.get(key) for key in
                ('checkpoint', 'checkpoint_sha256', 'options', 'operators', 'selection_protocol')}
    options = dict(settings.get('options') or {})
    if manifest.get('per_shape'):
        options['per_shape'] = {**manifest['per_shape'], **(options.get('per_shape') or {})}
    if options:
        settings['options'] = options
    settings['query_batch_size'] = inference.get('query_batch_size', manifest.get('query_batch_size'))
    settings['candidates'] = record.get('raw', {}).get('num_candidates')
    for key in ('candidate_sha256', 'context_sha256'):
        value = record.get('raw', {}).get(key, record.get('detail', {}).get(key))
        settings[key] = None if is_missing(value) else value
    protocol = record.get('raw', {}).get('protocol') or {}
    for key in ('query_order', 'query_sampling', 'sampling_seed', 'max_queries_per_shape'):
        settings[key] = protocol.get(key)
    settings['graph_recipe_sha256'] = record.get('detail', {}).get('graph_recipe_sha256',
                                      record.get('raw', {}).get('graph_recipe_sha256'))
    return {key: value for key, value in settings.items() if value is not None}


def transfer_settings(record):
    """Comparable recipe cohorts exclude dataset sizes, weights and plan hashes.

    Dataset-normalized entry IDs declare the cohort. Dataset-trained checkpoints
    may differ within it; A11 retains their identities and strict paired checks
    still require matching supplied weights and candidate domains.
    """
    settings = execution_settings(record)
    return {key: settings[key] for key in ('options', 'operators', 'selection_protocol', 'query_batch_size')
            if key in settings}


def result_record(report, *, row=None, comparison=False):
    row = row or {}
    inference = report.get('inference') or {}
    metadata = report.get('dataset_metadata') or {}
    protocol = report.get('protocol') or {}
    run = report.get('benchmark_run', report.get('paper_run')) or {}
    dataset = report.get('dataset', row.get('dataset'))
    if not isinstance(dataset, str):
        raise ValueError('A result must have a dataset name')
    method = row.get('method', inference.get('method', report.get('method', 'unspecified')))
    entry = report.get('id', row.get('id', run.get('entry', f'{method}-{dataset}')))
    if not isinstance(entry, str):
        raise ValueError('A result entry ID must be text')
    per_shape = {s: v or {} for s, v in (report.get('per_shape') or {}).items()}
    if comparison:
        policies = {p: {s: v[p] for s, v in per_shape.items() if p in v} for p in ('sort', 'expected')}
        groups = report.get('averages') or {}
        policies = {p: {'per_shape': shapes, 'averages': groups.get(p) or averages(shapes)}
                    for p, shapes in policies.items()}
    else:
        primary = protocol.get('tie_policy', 'sort')
        policies = {primary: {'per_shape': per_shape, 'averages': report.get('averages') or averages(per_shape)}}
        policies.update(report.get('additional_tie_metrics') or {})
    policies = {p: {'per_shape': {s: v or {} for s, v in ((values or {}).get('per_shape') or {}).items()},
                    'averages': {g: v or {} for g, v in ((values or {}).get('averages') or {}).items()}}
                for p, values in policies.items()}
    for values in policies.values():
        if not values['averages'] and values['per_shape']:
            values['averages'] = averages(values['per_shape'])
    # Comparison summary scores are already x100. Its details and raw results
    # are fractions. Never silently treat a summary mean as per-type evidence.
    if not per_shape and row:
        for p, name in (('sort', 'sort_mrr'), ('expected', 'expected_random_mrr')):
            if row.get(name) is not None:
                policies[p] = {'per_shape': {}, 'averages': {'all': {'mrr': row[name] / 100}}}
    split = report.get('split', row.get('phase', 'test' if row.get('complete_test') else None))
    expected = PLUS_H_TYPES if dataset in PLUS_H_DATASETS else ULTRA_TYPES
    complete = row.get('complete_test')
    if complete is None:
        complete = (split == 'test' and protocol.get('full_split') is True and set(per_shape) == set(expected))
    facts = row.get('observed_facts', inference.get('observed_facts'))
    if per_shape and (set(per_shape) != set(expected) or any(
            v.get('available_queries') is not None and v.get('queries') != v['available_queries']
            for v in per_shape.values())):
        complete = False
    for p, values in policies.items():
        computed = averages(values['per_shape'])
        for group, metrics in values['averages'].items():
            for metric, value in metrics.items():
                actual = computed.get(group, {}).get(metric)
                if metric in METRICS and not is_missing(value) and not is_missing(actual) and not math.isclose(value, actual, abs_tol=1e-11):
                    raise ValueError(f'Inconsistent {p} {group} {metric} average for {entry}')
    return dict(id=entry, dataset=dataset, method=method, split=split, complete=bool(complete),
                graph=row.get('inference_graph', report.get('inference_graph', metadata.get('inference_graph'))),
                filter=row.get('answer_filter', report.get('answer_filter', protocol.get('answer_filter'))),
                profile=row.get('execution_profile', report.get('execution_profile')),
                facts=json.dumps(facts, sort_keys=True) if isinstance(facts, dict) else facts,
                calibration=row.get('calibration', report.get('calibration', inference.get('calibration'))),
                paired_with=report.get('paired_with'), metrics=policies,
                counts={s: {k: v[k] for k in ('queries', 'hard_answers') if k in v} for s, v in per_shape.items()},
                tie_ci=report.get('delta_macro_mrr_ci95'), detail=report if comparison else {},
                raw=report if not comparison else {}, reference=row.get('reference', report.get('reference')) or {})


class Reports:
    def __init__(self):
        self.results = {}
        self.difficulty = []
        self.effects = {'adapter': [], 'filter': [], 'graph': []}
        self.template = False
        self.primary_recipes = ()

    def add_result(self, record):
        key = record['id']
        previous = self.results.get(key)
        if previous is not None:
            for name in ('dataset', 'method', 'graph', 'filter', 'profile', 'facts', 'calibration'):
                if previous[name] is not None and record[name] is not None and previous[name] != record[name]:
                    raise ValueError(f'Conflicting entry {key!r}: {name}; use distinct entry IDs for different runs')
            a, b = execution_settings(previous), execution_settings(record)
            for name in a.keys() & b.keys():
                if a[name] != b[name]:
                    raise ValueError(f'Conflicting entry {key!r}: {name}; use distinct entry IDs for different runs')
            if previous['tie_ci'] is not None and record['tie_ci'] is not None and previous['tie_ci'] != record['tie_ci']:
                raise ValueError(f'Conflicting entry {key!r}: tie confidence interval')
            record['tie_ci'] = record['tie_ci'] if record['tie_ci'] is not None else previous['tie_ci']
            record['detail'] = previous['detail'] or record['detail']
            for policy in previous['metrics'].keys() & record['metrics'].keys():
                left, right = previous['metrics'][policy]['per_shape'], record['metrics'][policy]['per_shape']
                if left and right and set(left) != set(right):
                    raise ValueError(f'Conflicting entry {key!r}: {policy} query-type coverage')
                for shape in left.keys() & right.keys():
                    for metric in METRICS:
                        a, b = left[shape].get(metric), right[shape].get(metric)
                        if not is_missing(a) and not is_missing(b) and not math.isclose(a, b, abs_tol=1e-11):
                            raise ValueError(f'Conflicting entry {key!r}: {policy} {shape} {metric}')
                left_means, right_means = previous['metrics'][policy]['averages'], record['metrics'][policy]['averages']
                for group in left_means.keys() & right_means.keys():
                    for metric in METRICS:
                        a, b = (left_means[group] or {}).get(metric), (right_means[group] or {}).get(metric)
                        if not is_missing(a) and not is_missing(b) and not math.isclose(a, b, abs_tol=1e-11):
                            raise ValueError(f'Conflicting entry {key!r}: {policy} {group} {metric}')
                if left and not right:
                    record['metrics'][policy] = previous['metrics'][policy]
            record['raw'] = previous['raw'] or record['raw']
            record['metrics'] = {**previous['metrics'], **record['metrics']}
            record['counts'] = record['counts'] or previous['counts']
            for name in ('graph', 'filter', 'profile', 'facts', 'calibration', 'paired_with'):
                if record[name] is None:
                    record[name] = previous[name]
        self.results[key] = record

    def consume(self, payload):
        if payload is None or payload == {} or payload == []:
            return
        if isinstance(payload, list):
            for item in payload:
                self.consume(item)
            return
        if not isinstance(payload, dict):
            raise ValueError('Reports must be JSON objects or lists of report objects')
        if 'macro' in payload:
            kind = ('adapter' if 'learned' in payload and 'control' in payload else
                    'filter' if 'corrected' in payload and 'released' in payload else
                    'graph' if 'train' in payload and 'train_valid' in payload else None)
            if kind is None:
                raise ValueError('Unknown paired-effect report')
            fields = {'adapter': ('learned', 'control'), 'filter': ('released', 'corrected'),
                      'graph': ('train', 'train_valid')}[kind]
            for previous in self.effects[kind]:
                if all(previous[key] == payload[key] for key in fields) and previous != payload:
                    raise ValueError(f'Conflicting {kind} effects for the same pair')
            if payload not in self.effects[kind]:
                self.effects[kind].append(payload)
        elif 'grouping' in payload and 'shape' in payload:
            if payload not in self.difficulty:
                self.difficulty.append(payload)
        elif 'dataset' in payload and ('per_shape' in payload or 'averages' in payload):
            self.add_result(result_record(payload))
        elif 'rows' in payload and 'details' in payload and all('id' in r for r in payload['rows']):
            details = {d['id']: d for d in payload['details']}
            for row in payload['rows']:
                self.add_result(result_record(details.get(row['id'], {}), row=row, comparison=True))
        elif 'rows' in payload and ('semantics' in payload or all('grouping' in r for r in payload['rows'])):
            self.consume(payload['rows'])
        else:
            keys = ('results', 'comparison', 'difficulty', 'adapter_effects', 'filter_effects', 'graph_effects')
            found = [key for key in keys if key in payload]
            if not found:
                raise ValueError('Unrecognized report; use result, comparison, difficulty, or paired-effect JSON')
            for key in found:
                self.consume(payload[key])


def load_reports(paths):
    reports = Reports()
    for path in paths:
        try:
            reports.consume(json.loads(Path(path).read_text(encoding='utf-8')))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'{path}: {error}') from error
    return reports


def policy_values(record, policy):
    return record['metrics'].get(policy, {'per_shape': {}, 'averages': {}})


def transfer_rows(reports, policy):
    groups = defaultdict(dict)
    for record in reports.results.values():
        group = family(record['dataset'])
        shapes = policy_values(record, policy)['per_shape']
        if group is None or not record['complete'] or set(shapes) != set(ULTRA_TYPES):
            continue
        # Retain recipe names while removing only the dataset component.
        recipe = record['id'].replace(record['dataset'], '*')
        configuration = transfer_settings(record)
        details = condition(record)
        if configuration:
            details += '; ' + '; '.join(key.replace('_', ' ') + ': ' + setting_value(value)
                                        for key, value in sorted(configuration.items()))
        key = (recipe, details, json.dumps(configuration, sort_keys=True))
        if record['dataset'] in groups[key]:
            raise ValueError(f'Duplicate dataset in transfer condition: {recipe}')
        groups[key][record['dataset']] = record
    rows = []
    for (recipe, details, _), records in sorted(groups.items()):
        for group, (label, total) in FAMILIES.items():
            selected = [r for r in records.values() if group == 'all' or family(r['dataset']) == group]
            if not selected:
                continue
            scores = []
            for category in ('epfo', 'negation'):
                values = [averages(policy_values(r, policy)['per_shape'])[category] for r in selected]
                for metric in ('mrr', 'hits10'):
                    scores.append(number(sum(v[metric] for v in values) / len(values))
                                  if all(not is_missing(v.get(metric)) for v in values) else MISSING)
            rows.append([escape(recipe), escape(details), escape(label), f'{len(selected)}/{total}', *scores])
    return rows






def validate_pair(left, right, *, kind):
    """Only the ablated factor may change in the supplied recipe evidence."""
    fields = ('dataset', 'method', 'split', 'profile', 'facts', 'complete')
    fields += (() if kind == 'graph' else ('graph',))
    fields += (() if kind == 'filter' else ('filter',))
    fields += (() if kind == 'adapter' else ('calibration',))
    for field in fields:
        a, b = left.get(field), right.get(field)
        if a is not None and b is not None and a != b:
            raise ValueError(f'{kind.title()} pair changes {field}: {left["id"]}, {right["id"]}')
    if kind == 'adapter':
        if any(record.get('method') and not record['method'].endswith('-adapter') for record in (left, right)):
            raise ValueError('Adapter pairs require a matching adapter backbone')
        if left.get('calibration') not in (None, 'learned') or right.get('calibration') not in (None, 'without-adapter', 'identity'):
            raise ValueError('Adapter pair does not compare learned and identity calibration')
    a, b = execution_settings(left), execution_settings(right)
    for field in a.keys() & b.keys():
        if kind == 'graph' and field == 'context_sha256':
            continue  # Changing the inference graph changes its context identity.
        if a[field] != b[field]:
            raise ValueError(f'{kind.title()} pair changes {field}: {left["id"]}, {right["id"]}')
    if left['counts'] != right['counts']:
        raise ValueError(f'{kind.title()} pair has different query/answer counts: {left["id"]}, {right["id"]}')
    for policy in left['metrics'].keys() & right['metrics'].keys():
        if set(policy_values(left, policy)['per_shape']) != set(policy_values(right, policy)['per_shape']):
            raise ValueError(f'{kind.title()} pair has different query-type coverage: {left["id"]}, {right["id"]}')


def effect_metrics(effect, left, right, policy, *, kind, direction=1):
    """Check supplied B-minus-A effects and preserve CIs without estimating them."""
    supplied = dict((effect.get('macro') or {}).get(policy) or {})
    if left and right:
        validate_pair(left, right, kind=kind)
        if effect.get('dataset') is not None and effect['dataset'] != left['dataset']:
            raise ValueError(f'{kind.title()} effect dataset differs from its paired results')
        a = policy_values(left, policy)['averages'].get('all', {})
        b = policy_values(right, policy)['averages'].get('all', {})
        for metric in METRICS:
            if not is_missing(a.get(metric)) and not is_missing(b.get(metric)):
                delta = direction * (b[metric] - a[metric])
                if not is_missing(supplied.get(metric)) and not math.isclose(supplied[metric], delta, abs_tol=1e-11):
                    raise ValueError(f'Inconsistent {kind} {policy} {metric} delta for {left["id"]}, {right["id"]}')
                supplied[metric] = delta
        shapes = effect.get('per_shape') or {}
        if shapes and set(shapes) != set(policy_values(left, policy)['per_shape']):
            raise ValueError(f'{kind.title()} effect has different query-type coverage from its results')
        for shape, values in shapes.items():
            if values.get('queries') is not None and left['counts'].get(shape, {}).get('queries') is not None and values['queries'] != left['counts'][shape]['queries']:
                raise ValueError(f'{kind.title()} effect has different query counts from its results')
            for metric in METRICS:
                a_value = policy_values(left, policy)['per_shape'].get(shape, {}).get(metric)
                b_value = policy_values(right, policy)['per_shape'].get(shape, {}).get(metric)
                delta = (values.get(policy) or {}).get(metric)
                if all(not is_missing(v) for v in (a_value, b_value, delta)) and not math.isclose(delta, direction * (b_value - a_value), abs_tol=1e-11):
                    raise ValueError(f'Inconsistent {kind} {policy} {shape} {metric} delta')
    return supplied


def adapter_rows(reports, policy):
    pairs = {(p['learned'], p['control']): p for p in reports.effects['adapter']}
    for r in reports.results.values():
        if r['paired_with'] in reports.results:
            pairs.setdefault((r['paired_with'], r['id']), {})
    rows = []
    for (left, right), effect in sorted(pairs.items()):
        learned, control = reports.results.get(left), reports.results.get(right)
        delta = effect_metrics(effect, learned, control, policy, kind='adapter', direction=-1)
        values = []
        for r in (learned, control):
            values.append(policy_values(r, policy)['averages'].get('all', {}).get('mrr') if r else None)
        rows.append([escape(effect.get('dataset', learned['dataset'] if learned else None)), escape(left), escape(right),
                     escape(scope(learned) if learned else None), number(values[0]), number(values[1]),
                     number(delta.get('mrr'), signed=True), ci(delta.get('mrr_ci95'))])
    return rows






def dataset_name(name):
    if name is None:
        return MISSING
    if name.startswith('InductiveFB15k237Query:'):
        return 'FB15k-237 v' + name.split(':')[1]
    if name.startswith('WikiTopicsQuery:'):
        return 'WikiTopics ' + name.split(':')[1]
    return {'FB15k237LogicalQuery': 'FB15k-237', 'FB15kLogicalQuery': 'FB15k',
            'NELL995LogicalQuery': 'NELL995', 'FB15k237+H': 'FB15k-237+H'}.get(name, name)


def method_name(record):
    method = record.get('method')
    name = METHOD_NAMES.get(method, method or MISSING)
    entry = record.get('id', record.get('entry', ''))
    if method and method.endswith('-adapter'):
        if 'intersections' in entry:
            name += ' (2i/3i)'
        elif '14type' in entry:
            name += ' (14-type)'
        if record.get('paired_with') or record.get('calibration') == 'without-adapter' or '-without-adapter' in entry:
            name = name.replace(' + adapter', ' identity')
    options = (record.get('raw', {}).get('inference') or {}).get('options') or {}
    if options.get('observed_facts') in ('none', 'atomic', 'both'):
        name += ' (facts ' + options['observed_facts'] + ')'
    return name


def entry_record(reports, entry, dataset=None):
    if entry in reports.results:
        return reports.results[entry]
    method = next((m for m in sorted(METHOD_NAMES, key=len, reverse=True) if (entry or '').startswith(m + '-')), None)
    if method is None and (entry or '').startswith(('ultra-product-', 'trix-product-')):
        method = entry.split('-')[0] + '-adapter'
    return dict(id=entry or '', dataset=dataset, method=method, paired_with=None, calibration=None, raw={})


def labeled_records(reports):
    records = dict(reports.results)
    for row in reports.difficulty:
        entry = row.get('entry')
        if entry and entry not in records:
            records[entry] = hardness_record(reports, row)
    return records


def run_labels(reports):
    """Compact default names, unique labels whenever a dataset has multiple runs; seed replicates say so."""
    from . import summary
    records = labeled_records(reports)
    groups = defaultdict(list)
    for entry, record in records.items():
        groups[record['dataset'], short_method(method_name(record))].append(entry)
    labels = {}
    for (_, base), entries in sorted(groups.items(), key=lambda item: str(item[0])):
        full_names = [method_name(records[entry]) for entry in entries]
        systems = {entry: summary.system_of(records[entry]) if records[entry].get('dataset') else ((entry,), 0)
                   for entry in entries}
        # Several runs: name the recipe (without the backbone), for replicates the seed tag, and the
        # evaluation condition (answer filters, inference graph) where the runs differ in it.
        recipes = defaultdict(set)
        for entry in entries:
            recipes[systems[entry][0]].add(systems[entry][1])
        varying = [field for field in ('filter', 'graph') if len({records[entry].get(field) for entry in entries}) > 1]
        tags = {}
        for entry in entries:
            system, seed = systems[entry]
            parts = [] if len(recipes) == 1 else [summary.recipe_token(records[entry])] if records[entry].get('dataset') else []
            if len(recipes[system]) > 1:
                parts.append(f'seed tag {seed}')
            parts.extend(f'{records[entry][field]} {"filters" if field == "filter" else "graph"}'
                         for field in varying if records[entry].get(field))
            tags[entry] = ', '.join(p for p in parts if p)
        # Colliding names get recipe/seed labels when those name every run uniquely, else numbers.
        named = all(tags.values()) and len(set(tags.values())) == len(tags)
        for index, entry in enumerate(sorted(entries), 1):
            name = method_name(records[entry])
            labels[entry] = (base if len(entries) == 1 else name if full_names.count(name) == 1
                             else name + f' ({tags[entry]})' if named else name + f' [R{index}]')
    return labels


def display_method(reports, record):
    return run_labels(reports).get(record['id'], method_name(record))


def shown_labels(reports, records):
    """Names unique among the shown runs only: a table of one run per method needs no recipe, seed or condition tags."""
    shown = copy(reports)
    shown.results = {r['id']: r for r in records}
    shown.difficulty = [row for row in reports.difficulty if row.get('entry') in shown.results]
    return run_labels(shown)


@functools.lru_cache(maxsize=None)
def shipped_recipes():
    """{(suite, adapter method): recipe} of the checked-in kgfm_adapters.json manifests, without seed tags."""
    from . import summary
    recipes = {}
    for suite in SUITES:
        for entry in read_manifest(suite_directory(suite) / 'kgfm_adapters.json')['entries']:
            recipe = summary.SEED.sub('', entry['id'].replace(entry['dataset'], ''))
            recipes[suite, entry['method']] = re.sub('-+', '-', recipe).strip('-')
    return recipes


def paper_records(reports, *, identities=False):
    """One run per dataset, method and identity flag, as in the main tables (see summary.primary_runs)."""
    from . import summary
    return sorted(summary.primary_runs(reports, identities=identities), key=lambda r: (r['dataset'], method_name(r), r['id']))


def subset_reports(reports, records):
    subset = copy(reports)
    subset.results = {r['id']: r for r in records}
    return subset


def main_transfer_rows(reports, policy):
    selected = paper_records(reports)
    names = {escape(r['id'].replace(r['dataset'], '*')): escape(method_name(r)) for r in selected}
    rows = transfer_rows(subset_reports(reports, selected), policy)
    return [[names.get(row[0], row[0]), row[1], row[2], row[3], *row[4:]] for row in rows]


def main_plus_h_records(reports):
    """The +H tables compare one fact graph and answer-filter policy across methods."""
    records = [r for r in paper_records(reports) if r['dataset'] in PLUS_H_DATASETS]
    for field, label in (('graph', 'inference graph'), ('filter', 'answer filters')):
        applicable = [r for r in records if field != 'graph' or r['method'] not in GRAPH_INDEPENDENT_METHODS]
        conditions = {r[field] for r in applicable}
        if len(conditions) > 1 or any(is_missing(value) for value in conditions):
            details = ', '.join(f'{method_name(r)} ({dataset_name(r["dataset"])}): {r[field] or "unavailable"}'
                                for r in applicable)
            raise ValueError(f'The +H tables require the same {label}, explicitly recorded across methods and datasets; {details}')
    return records


def main_plus_h_protocol_notes(records):
    graphs = {r['graph'] for r in records if r['method'] not in GRAPH_INDEPENDENT_METHODS}
    filters = {r['filter'] for r in records}
    notes = []
    if graphs:
        graph = next(iter(graphs))
        graph_name = {'train': 'training', 'train+valid': 'training + validation'}.get(graph, graph)
        notes.append('Graph-based methods use ' + graph_name + ' facts.')
    if any(r['method'] in GRAPH_INDEPENDENT_METHODS for r in records):
        notes.append('ConE, CLMPT and plain CQD do not consume graph facts at inference.')
    if filters:
        notes.append('All methods use ' + next(iter(filters)) + ' answer filters.')
    return ' '.join(notes)


def main_plus_h_rows(reports, policy):
    records = main_plus_h_records(reports)
    labels = shown_labels(reports, records)
    return [[escape(dataset_name(r['dataset'])), escape(labels.get(r['id'], method_name(r))), escape(scope(r)),
             *[number(policy_values(r, policy)['per_shape'].get(s, {}).get('mrr')) for s in PLUS_H_TYPES]]
            for r in records]


def hardness_record(reports, row):
    r = entry_record(reports, row.get('entry'), row.get('dataset'))
    if not r.get('method') and row.get('method'):
        r = {**r, 'method': row['method']}
    return r


def hardness_scope(reports, row):
    """Coverage concerns parent queries, never the bin's participating queries."""
    if reports.template:
        return True, 'planned full test'
    parent, available = row.get('parent_queries'), row.get('available_queries')
    complete = row.get('complete_shape')
    if parent is not None and available is not None:
        if parent > available or parent < 0 or available < 0:
            raise ValueError('Invalid hardness parent-query coverage')
        if complete is True and parent != available:
            raise ValueError('Hardness claims complete coverage with missing parent queries')
        complete = parent == available if complete is None else complete
    if complete is None and row.get('entry') in reports.results:
        record = reports.results[row['entry']]
        online = (record['raw'].get('per_shape') or {}).get(row['shape']) or {}
        if online.get('available_queries') is not None:
            complete = online.get('queries') == online['available_queries']
        else:
            complete = (True if record['complete'] else (record['raw'].get('protocol') or {}).get('full_split'))
    text = ('full parent-type test' if complete is True else 'partial parent-type test' if complete is False
            else 'parent-query coverage unavailable')
    if parent is not None and available is not None:
        text += f' ({parent}/{available} parent queries)'
    return complete, text


def hardness_score(reports, row, policy, metric):
    hardness_scope(reports, row)
    return number((row.get(policy) or {}).get(metric))


def missing_link_count(row):
    """Exact answer-level counts; coarse/released bins cannot supply these scores."""
    label = row['label']
    if isinstance(label, bool) or not str(label).isdigit():
        raise ValueError('Missing-positive-link bins require integer labels')
    count = int(label)
    if not 0 <= count <= POSITIVE_EDGES[row['shape']]:
        raise ValueError('Missing-positive-link count outside parent-type bounds')
    return count


def selected_hardness(reports, main_types):
    """Pick one whole run/filter condition per dataset and method before pivoting."""
    primary = {r['id'] for r in paper_records(reports)}
    candidates = [r for r in paper_hardness_rows(reports) if r['grouping'] == 'inferred_positive_edges'
                  and r['shape'] in main_types and missing_link_count(r) > 0]
    candidates = [r for r in candidates if not (hardness_record(reports, r).get('paired_with')
                  or hardness_record(reports, r).get('calibration') == 'without-adapter'
                  or '-without-adapter' in (r.get('entry') or ''))]
    focused = [r for r in candidates if hardness_record(reports, r)['method'] in ('qto', 'ultra-adapter', 'trix-adapter')]
    groups = defaultdict(lambda: defaultdict(list))
    for row in focused or candidates:
        record = hardness_record(reports, row)
        method = record['method'] or method_name(record)
        key = (row.get('entry'), row.get('comparison_graph'), row.get('answer_filter'))
        groups[row.get('dataset'), method][key].append(row)
    selected = []
    for conditions in groups.values():
        def priority(key):
            entry, graph, filters = key
            rows = conditions[key]
            return (entry not in primary, graph != 'train+valid', filters != 'corrected',
                    any(hardness_scope(reports, r)[0] is not True for r in rows), entry or '', graph or '', filters or '')
        selected.extend(conditions[min(conditions, key=priority)])
    return selected


def main_hardness_notes(reports, main_types):
    conditions, coverage = defaultdict(set), defaultdict(set)
    for row in selected_hardness(reports, main_types):
        name = dataset_name(row.get('dataset')) + ' / ' + display_method(reports, hardness_record(reports, row))
        conditions[row.get('answer_filter') or MISSING].add(name)
        complete, description = hardness_scope(reports, row)
        if complete is not True:
            coverage[description].add(name + ' / ' + row['shape'])
    notes = ['Filters: ' + filters + ' (' +
             ('all shown rows' if len(conditions) == 1 else ', '.join(sorted(names))) + ')'
             for filters, names in sorted(conditions.items())]
    notes.extend(description + ': ' + ', '.join(sorted(names)) for description, names in sorted(coverage.items()))
    return '; '.join(notes)


def main_hardness_matrix(reports, policy, main_types):
    """One row per dataset/parent type; hardness levels are columns, never averaged."""
    candidates = selected_hardness(reports, main_types)
    methods = sorted({short_method(method_name(hardness_record(reports, r))) for r in candidates},
                     key=lambda name: (0 if name == 'QTO' else 1 if name.startswith('ULTRA') else 2 if name.startswith('TRIX') else 3, name))
    # Three method blocks fit a landscape page; other methods are fully retained in A6.
    methods = methods[:3]
    levels = range(1, max((POSITIVE_EDGES[s] for s in main_types), default=4) + 1)
    cells = {}
    for r in candidates:
        name = short_method(method_name(hardness_record(reports, r)))
        if name not in methods:
            continue
        key = (r.get('dataset'), r['shape'], name, missing_link_count(r))
        value = hardness_score(reports, r, policy, 'mrr')
        if key in cells and cells[key] != value:
            raise ValueError('Conflicting main hardness scores for the same run, condition and bin')
        cells[key] = value
    keys = sorted({(r.get('dataset'), r['shape']) for r in candidates},
                  key=lambda k: (k[0] or '', PLUS_H_TYPES.index(k[1])))
    headers = ['Dataset', 'Type', *[escape(m.replace(' + adapter', '') + ' / ' + h)
                                   for m in methods for h in map(str, levels)]]
    rows = [[escape(dataset_name(dataset)), escape(shape),
             *[cells.get((dataset, shape, m, h), MISSING) for m in methods for h in levels]]
            for dataset, shape in keys]
    if not methods:
        headers += [escape(m + ' / ' + str(h)) for m in ('QTO', 'ULTRA', 'TRIX') for h in levels]
    width = min(22, (248 - 44) // (len(headers) - 2))
    return headers, [32, 12, *([width] * (len(headers) - 2))], rows


def main_adapter_matrix(reports, policy):
    from . import summary
    selected = {r['id'] for r in paper_records(reports)}
    pairs = adapter_rows(reports, policy)  # validates paired coverage and preserves supplied CIs
    cells, scopes, names = {}, defaultdict(set), set()
    for row in pairs:
        learned_id = next((key for key in reports.results if escape(key) == row[1]), None)
        if learned_id and learned_id not in selected:
            continue
        effect = next((e for e in reports.effects['adapter'] if escape(e['learned']) == row[1]), {})
        r = entry_record(reports, learned_id or effect.get('learned'), effect.get('dataset'))
        dataset = r['dataset'] or effect.get('dataset')
        # The training recipe is recorded in A11; ULTRA and TRIX remain common column blocks.
        name = METHOD_NAMES.get(r['method'], r['method'] or MISSING).replace(' + adapter', '')
        names.add(name)
        recipe = ('2i/3i' if 'intersections' in r['id'] else '14-type' if '14type' in r['id']
                  else summary.recipe_token(r) if r.get('dataset') else MISSING)
        key = (dataset, recipe, name)
        if key in cells and cells[key] != row[4:]:
            raise ValueError('Ambiguous adapter recipes; supply one primary recipe per backbone')
        cells[key] = row[4:]
        scopes[dataset, recipe].add(row[3])
    names = sorted(names, key=lambda name: (name != 'ULTRA', name != 'TRIX', name)) or ['ULTRA', 'TRIX']
    headers = ['Dataset', 'Recipe', 'Scope', *[escape(name + ' / ' + field) if field != 'Delta' else escape(name) + r' / $\Delta$'
                                   for name in names for field in ('Learned', 'Identity', 'Delta', '95% CI')]]
    widths = [32, 18, 20, *[w for _ in names for w in (19, 19, 19, 29)]]
    rows = [[escape(dataset_name(dataset)), escape(recipe), '; '.join(sorted(scopes[dataset, recipe])),
             *[value for name in names for value in cells.get((dataset, recipe, name), [MISSING] * 4)]]
            for dataset, recipe in sorted(scopes)]
    return headers, widths, rows


def ultra_matrix_rows(reports, policy):
    rows = []
    for r in sorted(reports.results.values(), key=lambda r: (r['dataset'], method_name(r), r['id'])):
        if family(r['dataset']) is None:
            continue
        values = policy_values(r, policy)
        context = scope(r)
        if r['graph'] not in (None, 'dataset-defined') or r['filter'] not in (None, 'released'):
            context += '; ' + condition(r)
        for metric, label in (('mrr', 'MRR'), ('hits10', 'H@10')):
            rows.append([escape(dataset_name(r['dataset'])), escape(display_method(reports, r)), escape(context), label,
                         *[number(values['averages'].get(g, {}).get(metric)) for g in ('all', 'epfo', 'negation')],
                         *[number(values['per_shape'].get(s, {}).get(metric)) for s in ULTRA_TYPES]])
    return rows


def paper_hardness_rows(reports):
    """All paper hardness panels use the released test reference, train+valid."""
    return [row for row in reports.difficulty if row['grouping'] == 'overall' or
            (row.get('comparison_graph') == 'train+valid'
             and row.get('label_reference_graph', 'train+valid') == 'train+valid')]


def hardness_matrix_rows(reports, policy):
    """Pivot query types; merge identical count vectors, never scores or conditions."""
    groups = defaultdict(dict)
    inputs = paper_hardness_rows(reports)
    for record in reports.results.values():
        if record['dataset'] not in PLUS_H_DATASETS:
            continue
        for shape in set(record['counts']) | set(policy_values(record, policy)['per_shape']):
            inputs.append(dict(dataset=record['dataset'], entry=record['id'], shape=shape,
                               grouping='overall', label='all', comparison_graph=record['graph'],
                               answer_filter=record['filter'], **record['counts'].get(shape, {}),
                               **{p: policy_values(record, p)['per_shape'].get(shape, {}) for p in ('sort', 'expected')}))
    # Exact numeric bins replace duplicate coarse categories from the same run.
    # Keep coarse-only reports visibly identified, without inventing counts.
    exact = {(r.get('dataset'), r.get('entry'), r['shape'], r.get('comparison_graph'), r.get('answer_filter'))
             for r in inputs if r['grouping'] == 'inferred_positive_edges'}
    for r in inputs:
        if r['grouping'] == 'inferred_positive_edges':
            missing_link_count(r)
        if (r['grouping'] == 'difficulty' and r['label'] != 'observed'
                and (r.get('dataset'), r.get('entry'), r['shape'], r.get('comparison_graph'), r.get('answer_filter')) in exact):
            continue
        if r['grouping'] == 'difficulty' and r['label'] == 'observed':
            if (r.get('dataset'), r.get('entry'), r['shape'], r.get('comparison_graph'), r.get('answer_filter')) in exact:
                continue
            r = {**r, 'grouping': 'inferred_positive_edges', 'label': '0'}
        graph = r.get('comparison_graph')
        if r['grouping'] == 'overall':
            record = hardness_record(reports, r)
            graph = ('not used' if record.get('method') in GRAPH_INDEPENDENT_METHODS else
                     r.get('inference_graph') or record.get('graph') or graph)
        key = (r.get('dataset') or '', r.get('entry') or '', graph or '', r.get('answer_filter') or '', r['grouping'], str(r['label']))
        previous = groups[key].get(r['shape'])
        if previous is not None:
            merged = {**previous, **r}
            for p in ('sort', 'expected'):
                metrics = {}
                for metric in METRICS:
                    a, b = (previous.get(p) or {}).get(metric), (r.get(p) or {}).get(metric)
                    if not is_missing(a) and not is_missing(b) and not math.isclose(a, b, abs_tol=1e-11):
                        raise ValueError('Conflicting hardness rows for the same condition and bin')
                    metrics[metric] = a if is_missing(b) else b
                merged[p] = metrics
            for field in ('queries', 'hard_answers', 'parent_queries', 'available_queries', 'complete_shape'):
                a, b = previous.get(field), r.get(field)
                if not is_missing(a) and not is_missing(b) and a != b:
                    raise ValueError('Conflicting hardness counts for the same condition and bin')
                merged[field] = a if is_missing(b) else b
            r = merged
        groups[key][r['shape']] = r
    rows, counts = [], defaultdict(set)
    labels = {'difficulty': 'Coarse', 'inferred_positive_edges': 'Missing links',
              'released_reduction': 'Released reduction', 'overall': 'Overall'}
    for (dataset, entry, graph, filters, grouping, label), shapes in sorted(groups.items()):
        name = display_method(reports, entry_record(reports, entry, dataset))
        base = [escape(dataset_name(dataset)), escape(name), escape(graph), escape(filters),
                escape(labels.get(grouping, grouping) + ': ' + label)]
        for metric, title in (('mrr', 'MRR'), ('hits10', 'H@10')):
            rows.append([*base, title, *[hardness_score(reports, shapes[s], policy, metric) if s in shapes else MISSING
                                        for s in PLUS_H_TYPES]])
        for metric, title in (('queries', 'Queries'), ('hard_answers', 'Answers')):
            values = tuple(escape(shapes.get(s, {}).get(metric)) for s in PLUS_H_TYPES)
            counts[dataset, graph, filters, grouping, label, title, values].add(name)
    for (dataset, graph, filters, grouping, label, title, values), names in sorted(counts.items()):
        name = 'Counts unavailable' if all(v == MISSING for v in values) else 'Shared counts' if len(names) > 1 else next(iter(names))
        rows.append([escape(dataset_name(dataset)), escape(name),
                     escape(graph), escape(filters), escape(labels.get(grouping, grouping) + ': ' + label), title, *values])
    return rows


def author_reduction_panels(reports, policy):
    """Author categories as columns, with one whole run/condition per method.

    Structural labels come from released partitions. Negation uses supplied
    positive-tree partial/full scores; numeric-bin means are never combined.
    """
    records = {r['id']: r for r in paper_records(reports) if r['dataset'] in PLUS_H_DATASETS}
    labels = shown_labels(reports, records.values())
    groups = defaultdict(lambda: defaultdict(list))
    for row in paper_hardness_rows(reports):
        if row.get('dataset') not in PLUS_H_DATASETS:
            continue
        if row['grouping'] != 'released_reduction' and not (
                row['grouping'] == 'difficulty' and row['shape'] in ('2in', '3in', 'inp', 'pin', 'pni')):
            continue
        record = hardness_record(reports, row)
        if record.get('paired_with') or record.get('calibration') == 'without-adapter' or '-without-adapter' in record['id']:
            continue
        known = [r for r in records.values() if (r['dataset'], r['method']) == (row['dataset'], record.get('method'))]
        if known and row.get('entry') not in records:
            continue
        reference = row.get('label_reference_graph') if row['grouping'] == 'released_reduction' else row.get('comparison_graph')
        key = (row.get('entry'), row.get('comparison_graph'), row.get('answer_filter'), reference)
        groups[row['dataset'], record.get('method') or record['id']][key].append(row)
    selected = {}
    for conditions in groups.values():
        def priority(key):
            entry, graph, filters, reference = key
            record = records.get(entry)
            target_graph = 'train+valid'
            return (entry not in records, reference != 'train+valid', filters != (record['filter'] if record else 'corrected'),
                    graph != target_graph, entry.endswith('-released-filters') if entry else False,
                    any(hardness_scope(reports, row)[0] is not True for row in conditions[key]), str(key))
        key = min(conditions, key=priority)
        entry, graph, filters, reference = key
        selected[entry] = (graph, filters, reference, conditions[key])
        if entry not in records:
            records[entry] = hardness_record(reports, conditions[key][0])
    chosen, fact_graphs = {}, defaultdict(set)
    for entry, record in records.items():
        if entry in selected:
            graph, filters, reference, supplied = selected[entry]
        else:
            graph = 'train+valid'
            filters, reference, supplied = record['filter'], 'train+valid', []
        inference_graph = None if record.get('method') in GRAPH_INDEPENDENT_METHODS else record.get('graph')
        if inference_graph is None and record.get('method') not in GRAPH_INDEPENDENT_METHODS:
            inference_graph = next((row['inference_graph'] for row in supplied if row.get('inference_graph')), graph)
        chosen[entry] = (graph, filters, reference, supplied, inference_graph)
        if inference_graph:
            fact_graphs[record['dataset'], reference, filters].add(inference_graph)
    panels, coverage = defaultdict(list), defaultdict(lambda: defaultdict(set))
    for entry, record in records.items():
        graph, filters, reference, supplied, inference_graph = chosen[entry]
        if inference_graph is None:
            available_graphs = fact_graphs[record['dataset'], reference, filters]
            inference_graph = ('train+valid' if 'train+valid' in available_graphs else
                               min(available_graphs) if available_graphs else 'not used')
        by_shape = defaultdict(dict)
        # Supplied author partitions take precedence over broad negation summaries.
        for source in sorted(supplied, key=lambda row: row['grouping'] == 'released_reduction'):
            label = source['label']
            if source['grouping'] == 'difficulty':
                label = {'partial': 'Partial', 'full': 'Full'}.get(label)
            else:
                label = {'pos-exist': 'Partial', 'pos-only-miss': 'Full'}.get(label, label)
            if label not in (*AUTHOR_REDUCTION_TYPES, 'Partial', 'Full'):
                continue
            previous = by_shape[source['shape']].get(label)
            if previous is not None and previous['grouping'] == source['grouping']:
                a, b = (previous.get(policy) or {}).get('mrr'), (source.get(policy) or {}).get('mrr')
                if not is_missing(a) and not is_missing(b) and not math.isclose(a, b, abs_tol=1e-11):
                    raise ValueError('Conflicting author-reduction scores for the same run and category')
            by_shape[source['shape']][label] = source
        offline = {row['shape']: row for row in paper_hardness_rows(reports) if row.get('entry') == entry
                   and row['grouping'] == 'overall' and row.get('comparison_graph') == graph
                   and row.get('answer_filter') == filters}
        context = (record['dataset'], reference or MISSING, filters or MISSING, inference_graph or MISSING)
        name = labels.get(record['id'], method_name(record))
        shapes = set(by_shape) | set(offline)
        if entry in reports.results:
            shapes |= set(policy_values(record, policy)['per_shape'])
            if not record['complete']:
                coverage[context][scope(record)].add(name)
        for shape in PLUS_H_TYPES:
            if shape not in shapes:
                continue
            sources = list(by_shape[shape].values())
            overall = offline.get(shape)
            if overall is not None:
                overall_score = hardness_score(reports, overall, policy, 'mrr')
                sources.append(overall)
            elif (entry in reports.results and record['filter'] == filters
                  and (record['method'] in GRAPH_INDEPENDENT_METHODS or record['graph'] == inference_graph)
                  and all(hardness_scope(reports, row)[0] is True for row in sources)):
                overall_score = number(policy_values(record, policy)['per_shape'].get(shape, {}).get('mrr'))
            else:
                overall_score = MISSING
            values = {label: hardness_score(reports, source, policy, 'mrr') for label, source in by_shape[shape].items()}
            # These types are already irreducible in the authors' taxonomy.
            if shape in ('1p', '2u'):
                values.setdefault(shape, overall_score)
            elif shape in ('2in', 'pni'):
                values.setdefault('Full', overall_score)
            panels[context].append([shape, escape(name), overall_score,
                                    *[values.get(label, MISSING) for label in AUTHOR_REDUCTION_TYPES],
                                    values.get('Partial', MISSING), values.get('Full', MISSING)])
            for source in sources:
                complete, description = hardness_scope(reports, source)
                if complete is not True:
                    coverage[context][description].add(name + ' / ' + shape)
    output = []
    for (dataset, reference, filters, inference_graph), rows in sorted(panels.items()):
        context = (escape(dataset_name(dataset)) + ' | inference facts: ' + escape(inference_graph)
                   + '; filters: ' + escape(filters))
        if coverage[dataset, reference, filters, inference_graph]:
            context += '; ' + escape('; '.join(description + ': ' + ', '.join(sorted(names))
                                              for description, names in sorted(coverage[dataset, reference, filters, inference_graph].items())))
        rows.sort(key=lambda row: (PLUS_H_TYPES.index(row[0]), method_order(row[1])))
        rows = [[AUTHOR_QUERY_NAMES.get(row[0], row[0]), *row[1:]] for row in rows]
        output.append((context, rows))
    return output


def compact_protocol_rows(reports, policy):
    rows = []
    for r in sorted(reports.results.values(), key=lambda r: (r['dataset'], r['method'], r['id'])):
        a = policy_values(r, 'sort')['averages'].get('all', {}).get('mrr')
        b = policy_values(r, 'expected')['averages'].get('all', {}).get('mrr')
        rows.append(['Ties', escape(dataset_name(r['dataset'])), escape(display_method(reports, r)), 'Sort', 'Expected random',
                     number(a), number(b), number(b - a if not is_missing(a) and not is_missing(b) else None, signed=True), ci(r['tie_ci'])])
    for kind, first, second, title, a_name, b_name in (
            ('filter', 'released', 'corrected', 'Filters', 'Released', 'Corrected'),
            ('graph', 'train', 'train_valid', 'Graph', 'Train', 'Train+valid')):
        for effect in sorted(reports.effects[kind], key=lambda e: (e.get('dataset', ''), e[second])):
            values = [reports.results.get(effect[key]) for key in (first, second)]
            absolute = [policy_values(r, policy)['averages'].get('all', {}).get('mrr') if r else None for r in values]
            r = values[1] or values[0] or entry_record(reports, effect[second], effect.get('dataset'))
            delta = effect_metrics(effect, values[0], values[1], policy, kind=kind)
            rows.append([escape(title), escape(dataset_name(effect.get('dataset') or r['dataset'])),
                         escape(display_method(reports, r)), a_name, b_name, *[number(v) for v in absolute],
                         number(delta.get('mrr'), signed=True), ci(delta.get('mrr_ci95'))])
    return rows


def short_method(name):
    """Default training recipes belong in notes, not every result row."""
    return name.replace(' (14-type)', '').replace(' (2i/3i)', '')


def applicability(members, all_members):
    """Use a group name only when it describes exactly the applicable runs."""
    selections = (
        ('All configurations', all_members),
        ('+H suite', {m for m in all_members if m[0] in PLUS_H_DATASETS}),
        ('UltraQuery suite', {m for m in all_members if family(m[0])}),
        ('Learned adapters', {m for m in all_members if ' + adapter' in m[1]}),
        ('Identity controls', {m for m in all_members if ' identity' in m[1]}),
        ('All adapters', {m for m in all_members if ' + adapter' in m[1] or ' identity' in m[1]}),
        ('Baselines', {m for m in all_members if ' + adapter' not in m[1] and ' identity' not in m[1]}),
    )
    for label, selection in selections:
        if members == selection:
            return label
    datasets = {d for d, _ in members}
    if members == {m for m in all_members if m[0] in datasets}:
        return 'All methods (' + ', '.join(dataset_name(d) for d in sorted(datasets)) + ')'
    # A short exception list is often clearer than listing every included run.
    excluded = all_members - members
    excluded_methods = {m[1] for m in excluded}
    if len(excluded_methods) <= 2 and excluded == {m for m in all_members if m[1] in excluded_methods}:
        return 'All except ' + ', '.join(sorted(excluded_methods))
    by_method = defaultdict(set)
    for dataset, method in members:
        by_method[method].add(dataset)
    parts = []
    for method, datasets in sorted(by_method.items()):
        method_universe = {d for d, m in all_members if m == method}
        if datasets == method_universe:
            parts.append(method)
        elif datasets == {d for d in method_universe if d in PLUS_H_DATASETS}:
            parts.append(method + ' (+H)')
        elif datasets == {d for d in method_universe if family(d)}:
            parts.append(method + ' (UltraQuery)')
        else:
            parts.append(method + ' (' + ', '.join(dataset_name(d) for d in sorted(datasets)) + ')')
    return '; '.join(parts)


def setting_value(value):
    """Readable settings, without JSON punctuation or invented defaults."""
    if is_missing(value):
        return MISSING
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, str) and value.startswith(('{', '[')):
        try:
            return setting_value(json.loads(value))
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return '; '.join(str(k).replace('_', ' ') + ': ' + setting_value(v) for k, v in sorted(value.items())) or MISSING
    if isinstance(value, (list, tuple)):
        return ', '.join(setting_value(v) for v in value) or MISSING
    return str(value)


def shared_metadata_rows(reports):
    """Paper-facing settings: compact applicability and one named parameter per row."""
    groups = defaultdict(set)
    labels = run_labels(reports)
    all_members = {(r['dataset'], labels[r['id']]) for r in labeled_records(reports).values()}
    for r in reports.results.values():
        raw = r['raw']
        inference = raw.get('inference') or {}
        recipe = execution_settings(r)
        training = (inference.get('paper_protocol') or {}).get('training') or {}
        checkpoint = recipe.get('checkpoint')
        # Filenames identify checkpoints together with their dataset/method;
        # machine-local directory trees belong in the original JSON report.
        checkpoint_parts = str(checkpoint).replace('\\', '/').split('/') if checkpoint else []
        checkpoint_label = ('/'.join(checkpoint_parts[-3:]) if checkpoint_parts[-1] == 'checkpoint'
                            else checkpoint_parts[-1]) if checkpoint_parts else None
        values = {'scope': scope(r), 'inference graph': r['graph'], 'answer filters': r['filter'],
                  'execution profile': r['profile'], 'calibration': r['calibration'],
                  'observed facts': r['facts'], 'checkpoint': checkpoint_label,
                  'queries': raw.get('queries'), 'candidates': raw.get('num_candidates'),
                  'checkpoint SHA256': recipe.get('checkpoint_sha256'),
                  'selection protocol': recipe.get('selection_protocol'),
                  'seed': inference.get('seed'), 'device': inference.get('device'), 'PyTorch': inference.get('torch')}
        if 'query_batch_size' in recipe:
            values['query batch size'] = recipe['query_batch_size']
        for key, value in training.items():
            values['training: ' + key.replace('_', ' ')] = value
        if not training:
            values['training'] = None
        operators = recipe.get('operators') or {}
        by_operator = defaultdict(list)
        for shape, operator in operators.items():
            by_operator[setting_value(operator)].append(shape)
        values['operators'] = ('; '.join(operator + ' (' + ('all query types' if len(shapes) == len(operators)
                                    else ', '.join(shapes)) + ')' for operator, shapes in sorted(by_operator.items()))
                               if operators else None)
        options = recipe.get('options') or {}
        if not options:
            values['execution options'] = None
        overrides = options.get('per_shape') or {}
        option_keys = (set(options) - {'per_shape'}) | {k for v in overrides.values() for k in v}
        for key in sorted(option_keys):
            override_groups = defaultdict(list)
            for shape, settings in overrides.items():
                if key in settings:
                    override_groups[setting_value(settings[key])].append(shape)
            value = setting_value(options.get(key))
            if override_groups:
                value += ' (default); ' + '; '.join(', '.join(shapes) + ': ' + setting
                                                   for setting, shapes in sorted(override_groups.items()))
            if key.endswith('_bytes') and not override_groups and isinstance(options.get(key), (int, float)):
                value = f'{options[key] / 2**20:g} MiB'
            values[key.replace('_', ' ')] = value
        full_name = method_name(r)
        if ' (2i/3i)' in full_name or ' (14-type)' in full_name:
            values['adapter training types'] = '2i/3i' if ' (2i/3i)' in full_name else '14 query types'
        member = (r['dataset'], labels[r['id']])
        for setting, value in values.items():
            groups[setting, setting_value(value)].add(member)
    for r in labeled_records(reports).values():
        if labels[r['id']] != short_method(method_name(r)):
            groups['run ID', r['id']].add((r['dataset'], labels[r['id']]))
    return [[escape(applicability(members, all_members)), escape(setting), escape(value)]
            for (setting, value), members in sorted(groups.items())]


def empty_reports():
    """Plan the default test matrix using only the four checked-in manifests.

    No predictions, counts, trace labels or training outcomes are invented.
    Only primary evaluations and adapter pairs are planned. Protocol effects
    are separate appendix rows; no Cartesian graph/filter/reduction grid exists.
    """
    reports = Reports()
    reports.template = True
    for suite in ('ultraquery', 'plus_h'):
        for name in ('baselines.json', 'kgfm_adapters.json'):
            manifest = read_manifest(suite_directory(suite) / name)
            for entry in manifest['entries']:
                method = entry['method']
                adapter = method.endswith('-adapter')
                options = dict(entry.get('options', {}))
                profile = 'native' if adapter else 'reference'
                reference = dict(entry.get('reference', {}))
                if method in ('cqd', 'cqd-hybrid'):
                    profile = 'bounded'
                    options.update(reference_batching=False, row_batch_size=32, final_batch_size=32)
                    reference['status'] = 'bounded execution profile; independent validation required'
                answer_filter = SUITES[suite]['answer_filter']
                for identity in (False, True) if entry.get('adapter_ablation') else (False,):
                    suffix = '-without-adapter' if identity else ''
                    run_id = entry['id'] + suffix
                    learned_id = entry['id']
                    shapes = {shape: {} for shape in entry['query_types']}
                    raw = dict(dataset=entry['dataset'], split='test', benchmark_run={'entry': run_id},
                               per_shape=shapes, additional_tie_metrics={'expected': {'per_shape': shapes}},
                               protocol={'tie_policy': 'sort', 'full_split': True, 'answer_filter': answer_filter},
                               dataset_metadata={'inference_graph': entry.get('inference_graph', 'dataset-defined')},
                               execution_profile=profile, reference=reference,
                               inference=dict(method=method, checkpoint=entry.get('checkpoint'), options=options,
                                              query_batch_size=entry.get('query_batch_size'),
                                              operators=entry.get('operators'), seed=manifest.get('seed'),
                                              selection_protocol=entry.get('selection_protocol'),
                                              calibration=('without-adapter' if identity else 'learned') if adapter else None,
                                              observed_facts='checkpoint-default' if adapter else None),
                               paired_with=learned_id if identity else None)
                    reports.consume(raw)
                    if suite != 'plus_h':
                        continue
                    reports.effects['filter'].append(dict(
                        dataset=entry['dataset'], released=run_id + '-released-filters', corrected=run_id))
                    for graph in ('train+valid',):
                        for shape in entry['query_types']:
                            maximum = POSITIVE_EDGES[shape]
                            bins = [('inferred_positive_edges', str(k)) for k in range(1, maximum + 1)]
                            for grouping, label in bins:
                                reports.difficulty.append(dict(
                                    dataset=entry['dataset'], entry=run_id, shape=shape, grouping=grouping, label=label,
                                    comparison_graph=graph, answer_filter=answer_filter,
                                    label_reference_graph='train+valid' if grouping == 'released_reduction' else graph,
                                    complete_shape=True))
    return reports


def table(title, label, headers, widths, rows, note='', *, spanners=(), numeric_from=None,
          row_group=None, keep_group=False, banner=''):
    """Long tables paginate real reports; all unavailable cells use a dash."""
    columns = ''.join((r'>{\raggedleft\arraybackslash}' if numeric_from is not None and i >= numeric_from
                      else r'>{\raggedright\arraybackslash}') + f'p{{{width}mm}}'
                      for i, width in enumerate(widths))
    header = ' & '.join(headers) + r' \\'
    heading = []
    if banner:
        # Panel contexts consist of already escaped report cells.
        banner = banner.replace('/', r'/\allowbreak{}').replace(r'\_', r'\_\allowbreak{}')
        heading.extend([rf'\multicolumn{{{len(headers)}}}{{p{{{sum(widths)}mm}}}}{{\raggedright {banner}}}\\',
                        r'\addlinespace[2pt]'])
    if spanners:
        cells, rules, start = [], [], 1
        for name, count in spanners:
            cells.append(rf'\multicolumn{{{count}}}{{c}}{{{name}}}')
            if name:
                rules.append(rf'\cmidrule(lr){{{start}-{start + count - 1}}}')
            start += count
        if start - 1 != len(headers):
            raise ValueError('Grouped headers must cover every table column')
        heading.extend([' & '.join(cells) + r' \\', ''.join(rules)])
    heading.append(header)
    lines = [r'\begingroup', r'\scriptsize', r'\setlength{\tabcolsep}{1.5pt}',
             r'\setlength{\LTpre}{6pt}', r'\setlength{\LTpost}{6pt}',
             rf'\begin{{longtable}}{{{columns}}}', rf'\caption{{{escape(title)}}}\label{{{label}}}\\',
             r'\toprule', *heading, r'\midrule', r'\endfirsthead',
             rf'\multicolumn{{{len(headers)}}}{{l}}{{\tablename\ \thetable\ (continued)}}\\',
             r'\toprule', *heading, r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
    rows = rows or [[MISSING] * len(headers)]
    for i, row in enumerate(rows):
        if len(row) != len(headers):
            raise ValueError(f'Wrong number of cells in {title}')
        cells = [MISSING if is_missing(cell) else cell for cell in row]
        # Long paths/JSON identifiers can wrap without adding a package or changing their text.
        cells = [cell.replace('/', r'/\allowbreak{}').replace(r'\_', r'\_\allowbreak{}')
                 if len(cell) > 80 else cell for cell in cells]
        same_group = row_group is not None and i + 1 < len(rows) and row_group(row) == row_group(rows[i + 1])
        lines.append(' & '.join(cells) + (r' \\*' if same_group and keep_group else r' \\'))
        if row_group is not None and not same_group and i + 1 < len(rows):
            lines.append(r'\addlinespace[3pt]')
    lines.extend([r'\end{longtable}', r'\endgroup'])
    if note:
        lines.append(r'{\scriptsize ' + escape(note) + r'\par}')
    return '\n'.join(lines)


def protocol_panels(title, label, headers, widths, panels, note='', *, numeric_from=None, group_by=None, spanners=()):
    """One numbered table; each protocol panel repeats its context on continuation pages."""
    if not panels:
        return table(title, label, headers, widths, [], note, numeric_from=numeric_from, spanners=spanners)
    parts = []
    for index, panel_data in enumerate(panels):
        context, rows, *panel_headers = panel_data
        current_headers = panel_headers[0] if panel_headers else headers
        group_by = group_by or (lambda row: (row[0], row[1]))
        group_sizes = defaultdict(int)
        for row in rows:
            group_sizes[group_by(row)] += 1
        panel = table(title, label, current_headers, widths, rows, numeric_from=numeric_from,
                      banner=context, row_group=group_by, keep_group=max(group_sizes.values(), default=0) <= 20,
                      spanners=spanners)
        if index:
            # Longtable increments the table counter even without a caption.
            panel = r'\addtocounter{table}{-1}' + '\n' + panel
            panel = panel.replace(rf'\caption{{{escape(title)}}}\label{{{label}}}\\',
                                  '')
        parts.append(panel)
    if note:
        parts.append(r'{\scriptsize ' + escape(note) + r'\par}')
    return '\n'.join(parts)


def method_order(name):
    names = ('ConE', 'GNN-QE', 'UltraQuery', 'CLMPT', 'CQD', 'CQD-Hybrid', 'QTO',
             'ULTRA + adapter', 'ULTRA identity', 'TRIX + adapter', 'TRIX identity')
    base = short_method(name)
    return (names.index(base) if base in names else len(names), name)


def dataset_order(name):
    plus_h = [dataset_name(d) for d in PLUS_H_DATASETS]
    if name in plus_h:
        return 3, plus_h.index(name)
    if name in ('FB15k', 'FB15k-237', 'NELL995'):
        return 0, name
    if name.startswith('FB15k-237 v'):
        return 1, int(re.match(r'\d+', name.split(' v')[1]).group())
    if name.startswith('WikiTopics '):
        return 2, name
    return 4, name


def scope_notes(records):
    if not records:
        return 'Evaluation scope unavailable.'
    partial = defaultdict(set)
    for r in records:
        if not r['complete']:
            partial[scope(r)].add(short_method(method_name(r)) + ' (' + dataset_name(r['dataset']) + ')')
    return ('Evaluation scope: ' + '; '.join(
        value + ': ' + ', '.join(sorted(names)) for value, names in sorted(partial.items()))
        if partial else 'Full test evaluation.')


def hardness_panels(reports, policy):
    panels = defaultdict(list)
    bins = {'Overall: all': 'Overall', 'Missing links: 0': 'Diagnostic: 0',
            'Coarse: partial': 'Coarse: partial', 'Coarse: full': 'Coarse: full'}
    for row in hardness_matrix_rows(reports, policy):
        dataset, method, graph, filters, group, metric, *scores = row
        kind = 'inference facts' if group.startswith('Overall:') else 'hardness reference facts'
        context = (dataset, kind, graph, filters)
        method = 'Counts' if method == 'Counts unavailable' else method
        bin_label = (group.removeprefix('Missing links: ') if group.startswith('Missing links: ') and group != 'Missing links: 0'
                     else bins.get(group, group))
        panels[context].append([dataset, method, bin_label, metric, *scores])
    def order(row):
        bin_order = 0 if row[2] == 'Overall' else 1 if row[2].isdigit() else 2
        return (row[1] in ('Counts', 'Shared counts'), method_order(row[1]), bin_order,
                row[2], ('MRR', 'H@10', 'Queries', 'Answers').index(row[3]))
    output = []
    for (dataset, kind, graph, filters), rows in sorted(panels.items()):
        coverage = defaultdict(set)
        for source in paper_hardness_rows(reports):
            source_graph = source.get('comparison_graph')
            if source['grouping'] == 'overall':
                record = hardness_record(reports, source)
                source_graph = ('not used' if record.get('method') in GRAPH_INDEPENDENT_METHODS else
                                source.get('inference_graph') or record.get('graph') or source_graph)
            source_kind = 'inference facts' if source['grouping'] == 'overall' else 'hardness reference facts'
            if (escape(dataset_name(source.get('dataset'))), source_kind, escape(source_graph), escape(source.get('answer_filter'))) != (dataset, kind, graph, filters):
                continue
            complete, description = hardness_scope(reports, source)
            if complete is not True:
                coverage[description].add(display_method(reports, hardness_record(reports, source)) + ' / ' + source['shape'])
        context = f'{dataset} | ' + (f'inference facts: {graph}; ' if kind == 'inference facts' else '') + f'filters: {filters}'
        if coverage:
            context += '; ' + escape('; '.join(description + ': ' + ', '.join(sorted(names))
                                               for description, names in sorted(coverage.items())))
        output.append((context, sorted(rows, key=order)))
    return output


def render_tables(reports, *, policy='sort', fragment=False, main_types=MAIN_HARDNESS_TYPES):
    from . import summary
    if policy not in ('sort', 'expected'):
        raise ValueError('Tie policy must be sort or expected')
    if set(main_types) - set(PLUS_H_TYPES):
        raise ValueError('Unknown main hardness query type')
    if not (reports.results or reports.difficulty or any(reports.effects.values())):
        reports = empty_reports()
    lines = ['% Generated by python -m benchmarks.cqa.paper.',
             '% Required packages: booktabs, longtable, array; the appendix needs a landscape page (geometry).',
             '% Main tables are floats with one decimal; appendix tables are long tables with two.',
             '% Scores are x100. Missing data is shown as "-"; no scores are imputed.',
             f'% Selected tie policy: {policy}.',
             '% Empty templates: python -m benchmarks.cqa.paper',
             '% Populate: python -m benchmarks.cqa.paper REPORT.json ... -o tables.tex',
             '% Use --fragment to omit the preamble; retain both section headers.',
             '% Appendix table numbering is reset to A1--A11.']
    if not fragment:
        lines.extend([r'\documentclass[10pt]{article}', r'\usepackage[a4paper,landscape,margin=15mm]{geometry}',
                      r'\usepackage{booktabs,longtable,array}', r'\setlength{\LTcapwidth}{\textwidth}',
                      r'\setlength{\emergencystretch}{2em}', r'\begin{document}'])
    lines.append(r'\section*{Main paper}')
    tie_name = 'expected random ties' if policy == 'expected' else 'original sort ordering'
    lines.append(r'{\normalsize Scores and differences are shown as $\times 100$; counts are unscaled. Tie policy: '
                 + tie_name + r'. A dash (-) indicates unavailable data.\par}')
    if reports.template:
        lines.append(r'{\scriptsize No-data template: planned full test evaluation over all 23 UltraQuery and three +H datasets, '
                     r'using the default baselines and adapters. Identity and filter comparisons are separate tables. '
                     r'Dataset coverage is planned; no evaluation has been run. Scores, intervals and unavailable counts are "-". '
                     r'Hardness bins contain 1 to 4 missing positive links, as permitted by each parent type; further breakdowns appear only when supplied.\par}')
    for index, render in enumerate((summary.ultraquery_table, summary.plus_h_table, summary.ablation_table), 1):
        lines.extend([f'% BEGIN MAIN TABLE {index}', render(reports, policy), f'% END MAIN TABLE {index}'])
    hardness_headers, hardness_widths, hardness_rows = main_hardness_matrix(reports, policy, main_types)
    adapter_headers, adapter_widths, adapter_values = main_adapter_matrix(reports, policy)
    primary = paper_records(reports)
    plus_h_records = main_plus_h_records(reports)
    transfer = main_transfer_rows(reports, policy)
    variations = defaultdict(set)
    for row in transfer:
        variations[row[0]].add(row[1])
    markers, variation_notes = {}, []
    for method, conditions in sorted(variations.items()):
        if len(conditions) > 1:
            for details in sorted(conditions):
                marker = str(len(markers) + 1)
                markers[method, details] = marker
                variation_notes.append(marker + ': ' + method + ', ' + details)
    transfer = [[short_method(row[0]) + (rf'\textsuperscript{{{markers[row[0], row[1]]}}}'
                                         if (row[0], row[1]) in markers else ''), *row[2:]] for row in transfer]
    family_order = {label: i for i, (label, _) in enumerate(FAMILIES.values())}
    transfer.sort(key=lambda row: (method_order(row[0]), family_order[row[1]]))
    plus_h = [[row[0], row[1], *row[3:]]
              for row in main_plus_h_rows(reports, policy)]
    plus_h.sort(key=lambda row: (dataset_order(row[0]), method_order(row[1])))
    hardness_levels = max((POSITIVE_EDGES[s] for s in main_types), default=4)
    hardness_spanners = [('', 2), *[(hardness_headers[i].rsplit(' / ', 1)[0], hardness_levels)
                                   for i in range(2, len(hardness_headers), hardness_levels)]]
    hardness_headers = ['Dataset', 'Type', *([str(k) for k in range(1, hardness_levels + 1)] * (len(hardness_spanners) - 1))]
    adapter_spanners = [('', 1), *[(adapter_headers[i].rsplit(' / ', 1)[0], 4)
                                  for i in range(3, len(adapter_headers), 4)]]
    adapter_headers = ['Dataset', *(['Learned', 'Identity', r'$\Delta$', r'95\% CI'] * (len(adapter_spanners) - 1))]
    adapter_recipe_counts = defaultdict(set)
    for row in adapter_values:
        adapter_recipe_counts[row[0]].add(row[1])
    # Sort by dataset before a recipe label is appended to its name.
    adapter_values.sort(key=lambda row: (dataset_order(row[0]), row[1]))
    adapter_values = [[row[0] + (' (' + row[1] + ')' if len(adapter_recipe_counts[row[0]]) > 1 else ''),
                       *row[3:]] for row in adapter_values]
    # Tables that list every run show seed replicates once, at the lowest tag (A9 lists every tag).
    listed = summary.lowest_seed_reports(reports)
    ultra_panels = defaultdict(list)
    for row in ultra_matrix_rows(listed, policy):
        ultra_panels[row[2]].append([row[0], row[1], *row[3:]])
    for rows in ultra_panels.values():
        rows.sort(key=lambda row: (dataset_order(row[0]), method_order(row[1]), row[2] != 'MRR'))
    protocol = defaultdict(list)
    for row in compact_protocol_rows(listed, policy):
        factor, dataset, method, a_name, b_name, *scores = row
        protocol[factor, a_name, b_name].append([dataset, method, *scores])
    protocol_panels_data = []
    for (factor, a_name, b_name), rows in sorted(protocol.items(), key=lambda item: ('Ties', 'Filters', 'Graph').index(item[0][0])):
        rows.sort(key=lambda row: (dataset_order(row[0]), method_order(row[1])))
        protocol_panels_data.append((f'{factor}: {a_name} to {b_name}', rows,
                                     ['Dataset', 'Method', a_name + ' MRR', b_name + ' MRR', r'$\Delta$', r'95\% CI']))
    hardness_note = ('MRR by the minimum number of missing positive links in a valid witness for an answer (numbered columns). '
                     'For paths, these are hops requiring inference; negated atoms are not counted. Counts differ from released structural reductions, particularly for unions. '
                     'Within each bin, scores average answers per participating query, then participating queries; queries can occur in multiple bins. '
                     'Impossible or unavailable bins are "-". Coarse partial/full scores cannot supply numeric bins. '
                     'One run and filter condition is used per method/dataset. No cross-type hardness mean. '
                     + main_hardness_notes(reports, main_types))
    seed_rows, seed_recipes = summary.seed_rows(reports, policy)
    specifications = [
        dict(headers=['Method', 'Dataset family', 'Coverage', *(['MRR', 'H@10'] * 2)], widths=[42, 49, 18, 30, 30, 30, 30],
             rows=transfer, options=dict(spanners=(('', 3), ('EPFO', 2), ('Negation', 2)), numeric_from=3,
                                         row_group=lambda row: row[0], keep_group=True),
             note='Equal dataset averages over full tests with all 14 query types. Coverage is evaluated/total datasets '
                  '(planned in the empty template). Evaluation settings are in A11. Appendix tables use each method\'s default '
                  'recipe (adapters: the primary recipe) at its lowest seed tag, except A2 and A9, which report every seed '
                  'tag, and A3, A6, A10 and A11, which list every supplied run at its lowest seed tag.'
                  + (' Protocol variants: ' + '; '.join(variation_notes) if variation_notes else '')),
        dict(headers=['Method', 'EPFO MRR', 'Negation MRR', 'EPFO MRR', 'Negation MRR'], widths=[60, 34, 34, 34, 34],
             rows=summary.freebase_rows(reports, policy),
             options=dict(spanners=(('', 1), ('Freebase-derived (11)', 2), ('Other (12)', 2)), numeric_from=1),
             note='FB15k, FB15k-237 and the nine inductive FB15k-237 splits derive from Freebase, which UltraQuery '
                  'training queries, the pretraining of the ULTRA and TRIX backbones and the source queries of the adapters '
                  'also use. NELL995 and the eleven WikiTopics datasets do not derive from Freebase, but UltraQuery is '
                  'initialised from ULTRA 4g, whose pretraining includes NELL995. Equal query-type means per dataset, then '
                  'dataset means; adapter rows give the mean and sample standard deviation over seed tags, as in the main tables.'),
        dict(headers=['Dataset', 'Method', 'Metric', 'All', 'EPFO', 'Neg.', *ULTRA_TYPES], widths=[29, 40, 13, *([9] * 17)],
             panels=[(('Scope: ' + context) if context != 'full test' else '', values)
                     for context, values in sorted(ultra_panels.items())],
             options=dict(numeric_from=3, group_by=lambda row: row[0]),
             note='MRR and H@10, including identity controls. Means weight query types equally; protocol and partial '
                  'scopes are identified in panel headings. Adapter recipes are listed in A11.'),
        dict(headers=['Dataset', 'Method', *[escape(s) for s in PLUS_H_TYPES]], widths=[30, 40, *([11] * 16)], rows=plus_h,
             options=dict(numeric_from=2, row_group=lambda row: row[0], keep_group=True),
             note='MRR for all 16 query types, including five negation types. Adapter recipes are listed in A11. '
                  + main_plus_h_protocol_notes(plus_h_records) + ' ' + scope_notes(plus_h_records)),
        dict(headers=hardness_headers, widths=hardness_widths, rows=hardness_rows,
             options=dict(spanners=hardness_spanners, numeric_from=2, row_group=lambda row: row[0], keep_group=True),
             note='For +H test results, all hardness categories use training + validation facts as the fixed reference; '
                  'valid witnesses are checked offline against training + validation + test facts. ' + hardness_note),
        dict(headers=['Dataset', 'Method / counts', 'Bin', 'Metric', *PLUS_H_TYPES], widths=[25, 38, 19, 13, *([9] * 16)],
             panels=hardness_panels(listed, policy), options=dict(numeric_from=4),
             note='Overall panels identify the inference facts. Numbered bins count missing positive links, as in A5. '
                  'Hardness scores average bin answers within participating queries, then participating queries; query counts '
                  'can overlap across bins and cannot recombine these means. Parent-query coverage is described in panel '
                  'headings. Identical count vectors are shown once. Additional bins appear only when supplied.'),
        dict(headers=['Type', 'Method', 'Overall', *[AUTHOR_QUERY_NAMES.get(shape, shape) for shape in AUTHOR_REDUCTION_TYPES], 'Partial', 'Full'],
             widths=[16, 38, *([11] * 12), 16, 16], panels=author_reduction_panels(reports, policy),
             options=dict(numeric_from=3, group_by=lambda row: row[0],
                          spanners=(('', 3), ('Structural reductions', 11), ('Negated queries', 2))),
             note='MRR grouped by the +H authors\' structural query reductions. Columns describe the remaining query shape; '
                  '2u denotes two union branches, for which one predicted link can suffice. For negated queries, Partial and '
                  'Full refer to the positive reasoning tree. Query-type names follow the authors\' notation (for example, up '
                  'is 2u1p). One run and filter condition is used per method/dataset. Scores average category answers within '
                  'participating queries, then queries. Counts are in A6; unavailable or impossible categories are "-".'),
        dict(headers=adapter_headers, widths=[39, *adapter_widths[3:]], rows=adapter_values,
             options=dict(spanners=adapter_spanners, numeric_from=1),
             note='Paired MRR; delta is learned minus identity. Supplied 95% confidence intervals are conditional on frozen '
                  'weights. Adapter recipes are listed in A11. '
                  + scope_notes([r for r in primary if r['method'].endswith('-adapter')])),
        dict(headers=['Method', 'Seed tag', 'UltraQuery MRR', '+H MRR'], widths=[50, 18, 34, 34],
             rows=seed_rows, options=dict(numeric_from=1, row_group=lambda row: row[0]),
             note='Every seed tag of the primary adapter recipes (' + seed_recipes + '): MRR over all query types, '
                  'averaged over query types and then all datasets of the suite. Seed tags come from entry IDs (-seedN; '
                  'untagged runs are 0), not from training seeds. A tag without every dataset of a suite is "-". The main '
                  'tables report the mean and sample standard deviation over these tags; the other appendix tables use the '
                  'lowest tag.'),
        dict(headers=['Dataset', 'Method', 'MRR A', 'MRR B', r'$\Delta$', r'95\% CI'], widths=[35, 60, 35, 35, 28, 46],
             panels=protocol_panels_data, options=dict(numeric_from=2, group_by=lambda row: row[0]),
             note='Delta is the second condition minus the first, in MRR points. Supplied intervals are paired 95% '
                  'confidence intervals; unavailable values are "-".'),
        dict(headers=['Applies to', 'Setting', 'Value'], widths=[62, 49, 133], rows=shared_metadata_rows(listed),
             options=dict(row_group=lambda row: row[1]),
             note='Shared settings are listed once. Checkpoints are identified by filename and method/dataset; full paths '
                  'and run records remain in the input reports. Values in empty templates are manifest defaults; '
                  'unavailable values are "-".'),
    ]
    lines.extend([r'\clearpage', r'\section*{Appendix}', r'\setcounter{table}{0}', r'\renewcommand{\thetable}{A\arabic{table}}'])
    for index, spec in enumerate(specifications, 1):
        if index > 1 and not fragment:
            # In the standalone preview, each logical table starts together.
            lines.append(r'\clearpage')
        label = f'tab:benchmark-paper-{index}'
        lines.append(f'% BEGIN PAPER TABLE {index}')
        if 'panels' in spec:
            lines.append(protocol_panels(TABLE_TITLES[index - 1], label, spec['headers'], spec['widths'], spec['panels'],
                                         spec['note'], **spec['options']))
        else:
            lines.append(table(TABLE_TITLES[index - 1], label, spec['headers'], spec['widths'], spec['rows'],
                               spec['note'], **spec['options']))
        lines.append(f'% END PAPER TABLE {index}')
    if not fragment:
        lines.append(r'\end{document}')
    return '\n\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m benchmarks.cqa.paper', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples (render saved data only; never run benchmarks):
  python -m benchmarks.cqa.paper
      Full default evaluation layout; scores "-" -> console and results/paper_tables.tex.
  python -m benchmarks.cqa.paper result.json -o tables.tex
      Fill the applicable tables; missing values are shown as "-".
  python -m benchmarks.cqa.paper comparison.json inference-difficulty.json adapter-effects.json -o tables.tex
      Combine scores, hardness groups and paired adapter effects.
  python -m benchmarks.cqa.paper REPORT.json --fragment --tie-policy sort -o tables.tex
      Tables for an existing paper; use booktabs, longtable, array and a
      landscape page with at least 267 mm of text width.
  python -m benchmarks.cqa.paper REPORT.json ... --figures
      Also save PDF/SVG/PNG figures and captions/numerical data in
      results/paper_figures. With no reports, figure layouts show "-".
  python -m benchmarks.cqa.paper REPORT.json ... --figures --hardness-composition
      Also export the optional +H composition diagnostic for the appendix.

Main paper (compact floats, one decimal, best bold and second underlined):
  1. UltraQuery benchmark: methods as rows; EPFO/negation MRR per dataset
     family (transductive, inductive (e), inductive (e,r), all).
  2. +H: methods trained on each target graph vs transferred models; EPFO and
     negation MRR per dataset and their average.
  3. Ablations: each other adapter recipe against the primary recipe.
  Adapter rows aggregate seed replicates (entry IDs differing only by -seedN):
  mean and sample s.d. over seeds; --primary-recipe picks the main recipe.
Appendix (A1--A11, long tables, two decimals):
  A1. UltraQuery results by family: MRR and H@10, coverage per family.
  A2. UltraQuery results by Freebase derivation: the 11 Freebase-derived
      datasets apart from the 12 others, seed means and s.d.
  A3. Full UltraQuery results: query types as columns; MRR and H@10 rows.
  A4. +H per-query-type performance: all 16 types; partial scopes labeled.
  A5. +H performance by hardness: QTO/ULTRA/TRIX MRR by missing-positive-link
      count (1-4); all methods remain in A6.
  A6. Full +H hardness breakdowns: query types as columns; counts shared when
      identical; additional bins appear only when supplied.
  A7. +H performance by author query reduction: reduction columns and
      negation partial/full categories.
  A8. Learned vs identity adapter: dataset rows, backbone columns, paired CIs.
  A9. Adapter training seeds: suite MRR of every seed tag of the primary recipes.
  A10. Protocol sensitivity: ties, answer filters and inference graphs.
  A11. Data, training, and selection details: shared settings printed once.
  Appendix tables use the lowest seed tag of each primary recipe; A2 and A9
  report every seed.

Optional figures (--figures [DIRECTORY]):
  Main paper: adapter gain by dataset family (datasets as dots, family means);
      +H hardness profiles, MRR by missing positive links for 3p/4p/3i/4i.
  Appendix: per-dataset transfer gains over UltraQuery, EPFO/negation separate;
      per-dataset adapter gains with supplied paired CIs.
  figures.tex holds a float with an escaped caption for every figure.
  --hardness-composition adds an optional appendix diagnostic of evaluated
      QA pairs by minimum missing positive links; not a model-performance result.
  figures.json records placement, captions, numerical data, selected runs and
  missing-data reasons. No observed bin, inferred CI or invented score.

Inputs: raw result.json; runner {"results": [...]} outputs; comparison.json;
inference-difficulty.json; adapter-/filter-/graph-effects.json. A combined JSON
may contain results, comparison, difficulty, adapter_effects, filter_effects,
and graph_effects. Include raw results for detailed training metadata.
Missing data is shown as "-", never zero-filled. Metrics/deltas are x100; counts are
unscaled. Confidence intervals are copied, never estimated. Graphs, filters
and execution recipes remain separate. Hardness cannot be recovered from an
overall score. Large tables span pages and repeat headers.
With no data, the four default baselines.json/kgfm_adapters.json manifests
provide all 23 UltraQuery and three +H datasets, primary methods and adapter
pairs. Hardness plans positive missing-link counts within each parent type; no Cartesian grid of protocol
or reduction variants is created. Known labels/settings are filled; scores,
CIs and unavailable counts stay "-". Dataset coverage assumes full evaluation.
''')
    parser.add_argument('results', nargs='*', type=Path, metavar='REPORT.json', help='Saved JSON reports; omit for the full default evaluation layout with scores "-"')
    parser.add_argument('-o', '--output', type=Path, default=Path('results/paper_tables.tex'), help='LaTeX file (default: results/paper_tables.tex)')
    parser.add_argument('--fragment', action='store_true', help='Emit tables and section headers without a document preamble')
    parser.add_argument('--tie-policy', choices=('sort', 'expected'), default='sort',
                        help='Scores to print: original sort ordering (default, the protocol\'s primary metric) or expected random ties; missing policy is shown as "-"')
    parser.add_argument('--main-hardness-types', nargs='+', choices=PLUS_H_TYPES, metavar='TYPE', default=MAIN_HARDNESS_TYPES,
                        help='Parent types of the hardness table A5 (default: %(default)s); A6 retains all types')
    parser.add_argument('--primary-recipe', action='append', default=[], metavar='RECIPE',
                        help='Main adapter recipe, with or without its backbone prefix (for example types2); '
                             'default: the shipped recipe, else the recipe with the most seeds')
    parser.add_argument('--figures', nargs='?', const=Path('results/paper_figures'), type=Path, metavar='DIRECTORY',
                        help='Also export the paper figures as PDF/SVG/PNG plus figures.json and figures.tex (default directory: results/paper_figures)')
    parser.add_argument('--hardness-composition', action='store_true',
                        help='With --figures, also export the optional +H QA-pair composition diagnostic for the appendix')
    args = parser.parse_args(argv)
    if args.hardness_composition and args.figures is None:
        parser.error('--hardness-composition requires --figures')
    try:
        if args.output.resolve() in {path.resolve() for path in args.results}:
            raise ValueError('Output must not overwrite an input report')
        reports = load_reports(args.results)
        reports.primary_recipes = tuple(args.primary_recipe)
        latex = render_tables(reports, policy=args.tie_policy, fragment=args.fragment, main_types=args.main_hardness_types)
        if args.figures is not None:
            from .figures import figure_names, generate_figures
            targets = {(args.figures / (name + '.' + extension)).resolve()
                       for name in figure_names(hardness_composition=args.hardness_composition)
                       for extension in ('pdf', 'svg', 'png')}
            targets |= {(args.figures / name).resolve() for name in ('figures.json', 'figures.tex')}
            if targets & {path.resolve() for path in args.results} or args.output.resolve() in targets:
                raise ValueError('Figure outputs must not overwrite input reports or the LaTeX output')
            figures = generate_figures(reports, args.figures, policy=args.tie_policy,
                                       hardness_composition=args.hardness_composition)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex, encoding='utf-8')
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    sys.stdout.write(latex)
    print(f'Saved {args.output}', file=sys.stderr)
    if args.figures is not None:
        for figure in figures['figures']:
            print(f'{figure["placement"]}: {args.figures / figure["name"]} (.pdf, .svg, .png)', file=sys.stderr)
        print(f'Captions and numerical data: {args.figures / "figures.json"}', file=sys.stderr)


if __name__ == '__main__':
    main()
