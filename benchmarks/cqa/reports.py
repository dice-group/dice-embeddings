"""Rebuild paired tie-policy tables from compact answer ranks."""

import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from dicee.query_answering._checkpoint import write_json
from dicee.query_answering._query import PLUS_H_SHAPES
from dicee.query_answering.benchmark import METRICS, _averages
from dicee.query_answering.datasets import query_types_for_dataset
from dicee.query_answering.method_evaluation import GRAPH_INDEPENDENT_METHODS


def trace_queries(path):
    with sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro', uri=True) as connection:
        harmonic = np.zeros(1, dtype=np.float64)
        for query_id, shape, payload in connection.execute('SELECT query_id, shape, answers FROM queries ORDER BY position'):
            records = np.asarray(json.loads(payload), dtype=np.int64)
            if records.ndim != 2 or records.shape[1] != 4 or not len(records):
                raise ValueError('Malformed answer-rank records')
            ranks, greater, sizes = records[:, 1:].T
            if (greater < 0).any() or (sizes < 1).any() or (ranks <= greater).any() or (ranks > greater + sizes).any():
                raise ValueError('Inconsistent filtered rank / tie block')
            last = greater + sizes
            if last.max() >= len(harmonic):
                harmonic = np.concatenate(([0.], np.cumsum(1. / np.arange(1, last.max() + 1))))
            sort_metrics = dict(zip(METRICS, [np.mean(1. / ranks), *(np.mean(ranks <= k) for k in (1, 3, 10))]))
            expected = dict(zip(METRICS, [np.mean((harmonic[last] - harmonic[greater]) / sizes),
                                          *(np.mean(np.minimum(np.maximum(k - greater, 0), sizes) / sizes) for k in (1, 3, 10))]))
            yield dict(query_id=query_id, shape=shape, sort=sort_metrics, expected=expected,
                       hard_answers=len(records), tied_answers=int((sizes > 1).sum()), max_tie_size=int(sizes.max()))


def summarize_trace(path, *, bootstrap_samples=2000, seed=0, expected_types=None):
    groups = defaultdict(list)
    for query in trace_queries(path):
        groups[query['shape']].append(query)
    if not groups:
        raise ValueError('Empty rank trace')
    rng = np.random.default_rng(seed)
    per_shape, delta_draws = {}, np.zeros(bootstrap_samples)
    for shape, queries in sorted(groups.items()):
        policies = {policy: {metric: float(np.mean([query[policy][metric] for query in queries])) for metric in METRICS}
                    for policy in ('sort', 'expected')}
        deltas = np.array([query['expected']['mrr'] - query['sort']['mrr'] for query in queries])
        draws = (np.full(bootstrap_samples, deltas[0]) if np.ptp(deltas) == 0 else
                 np.array([rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(bootstrap_samples)]))
        delta_draws += draws / len(groups)
        hard = sum(query['hard_answers'] for query in queries)
        per_shape[shape] = dict(**policies, queries=len(queries), hard_answers=hard,
                                tied_answer_fraction=sum(query['tied_answers'] for query in queries) / hard,
                                max_tie_size=max(query['max_tie_size'] for query in queries),
                                delta_mrr=float(deltas.mean()),
                                delta_mrr_ci95=np.quantile(draws, [.025, .975]).tolist() if bootstrap_samples else None)
    averages = {policy: _averages({shape: values[policy] for shape, values in per_shape.items()}) for policy in ('sort', 'expected')}
    return dict(per_shape=per_shape, averages=averages, all_16_types=set(groups) == set(PLUS_H_SHAPES),
                all_benchmark_types=set(groups) == set(expected_types) if expected_types is not None else None,
                delta_macro_mrr_ci95=np.quantile(delta_draws, [.025, .975]).tolist() if bootstrap_samples else None,
                uncertainty=dict(unit='paired query', stratification='query type', resamples=bootstrap_samples, seed=seed))


def adapter_effects(learned_trace, control_trace, *, bootstrap_samples=2000, seed=0):
    """Paired learned-minus-identity differences, conditional on frozen weights."""
    learned = {query['query_id']: query for query in trace_queries(learned_trace)}
    groups = defaultdict(list)
    for control in trace_queries(control_trace):
        current = learned.pop(control['query_id'], None)
        if current is None or current['shape'] != control['shape'] or current['hard_answers'] != control['hard_answers']:
            raise ValueError('Adapter comparison queries or answer counts differ')
        groups[control['shape']].append({policy: {metric: current[policy][metric] - control[policy][metric]
                                                for metric in METRICS} for policy in ('sort', 'expected')})
    if learned or not groups:
        raise ValueError('Adapter comparison requires identical nonempty query coverage')
    rng = np.random.default_rng(seed)
    per_shape, draws = {}, {policy: np.zeros(bootstrap_samples) for policy in ('sort', 'expected')}
    for shape, queries in sorted(groups.items()):
        row = dict(queries=len(queries))
        for policy in draws:
            values = np.array([query[policy]['mrr'] for query in queries])
            samples = (np.full(bootstrap_samples, values[0]) if np.ptp(values) == 0 else
                       np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(bootstrap_samples)]))
            draws[policy] += samples / len(groups)
            row[policy] = {metric: float(np.mean([query[policy][metric] for query in queries])) for metric in METRICS}
            row[policy]['mrr_ci95'] = np.quantile(samples, [.025, .975]).tolist() if bootstrap_samples else None
        per_shape[shape] = row
    macro = {policy: {metric: float(np.mean([row[policy][metric] for row in per_shape.values()])) for metric in METRICS}
             | {'mrr_ci95': np.quantile(samples, [.025, .975]).tolist() if bootstrap_samples else None}
             for policy, samples in draws.items()}
    return dict(per_shape=per_shape, macro=macro, uncertainty=dict(unit='paired query', stratification='query type',
                resamples=bootstrap_samples, seed=seed, scope='Conditional on frozen models and adapters; not training-seed uncertainty.'))


def export_reports(results, published, output, *, bootstrap_samples=2000, seed=0, title='CQA benchmark'):
    """Compare matching coverage only; partial means never become full-suite means."""
    from dicee.query_answering._checkpoint import checksum, read
    published = read(published) if published is not None else {'source': None, 'datasets': {}}
    rows, details = [], []
    for path in sorted(Path(results).rglob('result.json')):
        report = read(path)
        run = report.get('benchmark_run', report.get('paper_run'))
        if run is None:
            continue
        trace = path.parent / 'ranks.sqlite3'
        if checksum(trace) != report['rank_trace_sha256']:
            raise ValueError(f'Rank trace checksum mismatch: {trace}')
        expected_types = report.get('dataset_metadata', {}).get('expected_query_types')
        if expected_types is None:
            expected_types = query_types_for_dataset(report['dataset'])
        summary = summarize_trace(trace, bootstrap_samples=bootstrap_samples, seed=seed, expected_types=expected_types)
        for shape, values in summary['per_shape'].items():
            for policy in ('sort', 'expected'):
                online = report['per_shape'][shape] if policy == 'sort' else report['additional_tie_metrics'][policy]['per_shape'][shape]
                if any(not math.isclose(online[metric], values[policy][metric], abs_tol=1e-11) for metric in METRICS):
                    raise ValueError(f'Offline rank metrics differ from evaluation: {path}')
        method = report['inference']['method']
        scores = published['datasets'].get(report['dataset'], {}).get(method, {})
        coverage = summary['all_benchmark_types'] and report['protocol']['full_split'] and report['split'] == 'test'
        sort_mrr = summary['averages']['sort']['all']['mrr'] * 100
        expected = summary['averages']['expected']['all']['mrr'] * 100
        paper = sum(scores.values()) / len(expected_types) if set(scores) == set(expected_types) and coverage else None
        graph = report.get('dataset_metadata', {}).get('inference_graph', 'dataset-defined')
        answer_filter = report['protocol'].get('answer_filter', 'released')
        independent = method in GRAPH_INDEPENDENT_METHODS
        native_graph = report['reference'].get('inference_graph')
        rows.append(dict(id=run['entry'], dataset=report['dataset'], method=method,
                         inference_graph=graph, answer_filter=answer_filter,
                         applicable_graphs=['train', 'train+valid'] if independent else [graph],
                         reference_graph=native_graph,
                         reference_graph_matches=True if independent else graph == native_graph if native_graph else None,
                         calibration=report['inference'].get('calibration'),
                         observed_facts=report['inference'].get('observed_facts'),
                         execution_profile=report.get('execution_profile', 'unspecified'),
                         phase=run['phase'], complete_test=coverage, expected_types=list(expected_types),
                         complete_16_type_test=coverage and summary['all_16_types'], types=len(summary['per_shape']),
                         published_mrr=paper, sort_mrr=sort_mrr, expected_random_mrr=expected,
                         tie_delta=expected-sort_mrr, published_delta=sort_mrr-paper if paper is not None else None,
                         reference=report['reference']))
        details.append(dict(id=run['entry'], dataset=report['dataset'], **summary,
                            inference_graph=graph, graph_ablation=report.get('graph_ablation') if run['phase'] == 'test' else None,
                            graph_recipe_sha256=report.get('graph_recipe_sha256'),
                            calibration=report['inference'].get('calibration'), answer_filter=answer_filter,
                            filter_paired_with=report.get('filter_paired_with'),
                            paired_with=report.get('paired_with'), published_per_shape=scores if coverage else {}, result=str(path)))
    output = Path(output)
    write_json(output / 'comparison.json', dict(rows=rows, details=details, published_source=published['source']))
    with (output / 'per-query-type.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['entry', 'dataset', 'inference_graph', 'answer_filter', 'type', 'queries', 'published_mrr', 'sort_mrr', 'expected_random_mrr', 'tie_delta', 'tied_answer_fraction'])
        for detail in details:
            for shape, values in detail['per_shape'].items():
                writer.writerow([detail['id'], detail['dataset'], detail['inference_graph'], detail['answer_filter'], shape, values['queries'], detail['published_per_shape'].get(shape, ''),
                                 values['sort']['mrr'] * 100, values['expected']['mrr'] * 100, values['delta_mrr'] * 100, values['tied_answer_fraction']])
    lines = [f'# {title}', '', 'MRR ×100. Published comparisons require complete test coverage of the dataset’s query types.', '',
             'Graph-independent methods apply to both graph conditions. Published values may use different graph or answer protocols.', '',
             '| Entry | Dataset | Inference facts | Filters | Execution | Scope | Types | Published | Sort | Expected random | Tie Δ |',
             '|---|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        paper = f'{row["published_mrr"]:.2f}' if row['published_mrr'] is not None else '—'
        scope = 'full test' if row['complete_test'] else row['phase'] + ' / partial'
        graph = 'independent' if len(row['applicable_graphs']) > 1 else row['inference_graph']
        lines.append(f'| {row["id"]} | {row["dataset"]} | {graph} | {row["answer_filter"]} | {row["execution_profile"]} | {scope} | {row["types"]} | {paper} | '
                     f'{row["sort_mrr"]:.2f} | {row["expected_random_mrr"]:.2f} | {row["tie_delta"]:+.2f} |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    lookup = {row['id']: row for row in details}
    pairs = []
    for control in details:
        if control['paired_with'] not in lookup:
            continue
        learned = lookup[control['paired_with']]
        effect = adapter_effects(Path(learned['result']).parent / 'ranks.sqlite3',
                                 Path(control['result']).parent / 'ranks.sqlite3',
                                 bootstrap_samples=bootstrap_samples, seed=seed)
        if learned['answer_filter'] != control['answer_filter']:
            raise ValueError('Adapter comparison changed answer filters')
        pairs.append(dict(learned=learned['id'], control=control['id'], dataset=control['dataset'], answer_filter=control['answer_filter'], **effect))
    write_json(output / 'adapter-effects.json', pairs)
    lines = ['# Learned adapter effect', '',
             'Learned minus identity calibration, MRR ×100. Observed facts, query plans, operators and beams are held fixed.', '',
             '| Entry | Dataset | Sort Δ | Expected random Δ | Expected random 95% CI |',
             '|---|---|---:|---:|---|']
    with (output / 'adapter-effects.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['entry', 'dataset', 'type', 'queries', 'sort_delta_mrr', 'expected_delta_mrr'])
        for pair in pairs:
            ci = pair['macro']['expected']['mrr_ci95']
            interval = f'[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]' if ci is not None else '—'
            lines.append(f'| {pair["learned"]} | {pair["dataset"]} | {pair["macro"]["sort"]["mrr"] * 100:+.2f} | '
                         f'{pair["macro"]["expected"]["mrr"] * 100:+.2f} | {interval} |')
            for shape, row in pair['per_shape'].items():
                writer.writerow([pair['learned'], pair['dataset'], shape, row['queries'],
                                 row['sort']['mrr'] * 100, row['expected']['mrr'] * 100])
    (output / 'adapter-effects.md').write_text('\n'.join(lines) + '\n')
    groups = defaultdict(dict)
    for detail in details:
        if detail['graph_ablation']:
            group = groups[detail['graph_ablation'], detail['calibration'], detail['answer_filter']]
            if detail['inference_graph'] in group:
                raise ValueError('Duplicate graph-ablation condition')
            group[detail['inference_graph']] = detail
    graph_pairs = []
    for group in groups.values():
        if set(group) != {'train', 'train+valid'}:
            continue
        train, extended = group['train'], group['train+valid']
        if not train['graph_recipe_sha256'] or train['graph_recipe_sha256'] != extended['graph_recipe_sha256']:
            raise ValueError('Graph ablation changed model, adapter, operators, beam, or query settings')
        effect = adapter_effects(Path(extended['result']).parent / 'ranks.sqlite3',
                                 Path(train['result']).parent / 'ranks.sqlite3',
                                 bootstrap_samples=bootstrap_samples, seed=seed)
        graph_pairs.append(dict(train=train['id'], train_valid=extended['id'], dataset=train['dataset'],
                                calibration=train['calibration'], answer_filter=train['answer_filter'], **effect))
    write_json(output / 'graph-effects.json', graph_pairs)
    lines = ['# Inference graph effect', '',
             'Train+valid minus train, MRR ×100. Checkpoints, operators, queries and answer-filter policy are fixed.', '',
             '| Train entry | Dataset | Sort Δ | Expected random Δ | Expected random 95% CI |',
             '|---|---|---:|---:|---|']
    with (output / 'graph-effects.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['train_entry', 'train_valid_entry', 'dataset', 'type', 'queries', 'sort_delta_mrr', 'expected_delta_mrr'])
        for pair in graph_pairs:
            ci = pair['macro']['expected']['mrr_ci95']
            interval = f'[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]' if ci is not None else '—'
            lines.append(f'| {pair["train"]} | {pair["dataset"]} | {pair["macro"]["sort"]["mrr"] * 100:+.2f} | '
                         f'{pair["macro"]["expected"]["mrr"] * 100:+.2f} | {interval} |')
            for shape, row in pair['per_shape'].items():
                writer.writerow([pair['train'], pair['train_valid'], pair['dataset'], shape, row['queries'],
                                 row['sort']['mrr'] * 100, row['expected']['mrr'] * 100])
    (output / 'graph-effects.md').write_text('\n'.join(lines) + '\n')
    filter_pairs = []
    for control in details:
        if control['filter_paired_with'] not in lookup:
            continue
        corrected = lookup[control['filter_paired_with']]
        if (corrected['answer_filter'], control['answer_filter']) != ('corrected', 'released'):
            raise ValueError('Invalid corrected/released filter comparison')
        effect = adapter_effects(Path(corrected['result']).parent / 'ranks.sqlite3',
                                 Path(control['result']).parent / 'ranks.sqlite3',
                                 bootstrap_samples=bootstrap_samples, seed=seed)
        filter_pairs.append(dict(corrected=corrected['id'], released=control['id'], dataset=control['dataset'], **effect))
    write_json(output / 'filter-effects.json', filter_pairs)
    lines = ['# Answer-filter effect', '',
             'Corrected minus released filters, MRR ×100. Predictions and released hard targets are identical.', '',
             '| Entry | Dataset | Sort Δ | Expected random Δ | Expected random 95% CI |',
             '|---|---|---:|---:|---|']
    with (output / 'filter-effects.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['entry', 'dataset', 'type', 'queries', 'sort_delta_mrr', 'expected_delta_mrr'])
        for pair in filter_pairs:
            ci = pair['macro']['expected']['mrr_ci95']
            interval = f'[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]' if ci is not None else '—'
            lines.append(f'| {pair["corrected"]} | {pair["dataset"]} | {pair["macro"]["sort"]["mrr"] * 100:+.2f} | '
                         f'{pair["macro"]["expected"]["mrr"] * 100:+.2f} | {interval} |')
            for shape, row in pair['per_shape'].items():
                writer.writerow([pair['corrected'], pair['dataset'], shape, row['queries'],
                                 row['sort']['mrr'] * 100, row['expected']['mrr'] * 100])
    (output / 'filter-effects.md').write_text('\n'.join(lines) + '\n')
    records = [dict(id=row['id'], dataset=row['dataset'], method=row['method'], calibration=detail['calibration'],
                    paired_with=detail['paired_with'], complete=row['complete_test'], answer_filter=detail['answer_filter'],
                    trace=Path(detail['result']).parent / 'ranks.sqlite3')
               for row, detail in zip(rows, details)]
    effects = suite_effects(records, bootstrap_samples=bootstrap_samples, seed=seed)
    write_json(output / 'suite-effects.json', effects)
    lines = ['# Suite-level paired effects', '',
             'Treatment minus baseline, MRR ×100, equal weight per query type within a dataset and per dataset within '
             'a group. Adapter seeds are averaged per query before differencing; intervals resample queries within '
             'dataset × type and are conditional on the frozen models, adapters and seeds.', '',
             '| Comparison | Recipe | Suite | Group | Seeds | Sort Δ | Sort 95% CI | Expected Δ | Seed range (sort) |',
             '|---|---|---|---|---:|---:|---|---:|---|']
    for effect in effects:
        for group, values in effect['macro']['sort'].items():
            ci = values['ci95']
            interval = f'[{ci[0]:+.2f}, {ci[1]:+.2f}]' if ci is not None else '—'
            seeds = [s['sort'][group] for s in effect['per_seed'] if group in s['sort']]
            spread = f'{min(seeds):+.2f} to {max(seeds):+.2f}' if len(seeds) > 1 else '—'
            lines.append(f'| {effect["comparison"]} | {effect["recipe"]} | {effect["suite"]} | {group} | {len(effect["per_seed"])} | '
                         f'{values["delta"]:+.2f} | {interval} | {effect["macro"]["expected"][group]["delta"]:+.2f} | {spread} |')
    (output / 'suite-effects.md').write_text('\n'.join(lines) + '\n')
    return rows


def query_scores(path):
    """{query_id: (shape, hard answers, sort MRR, expected MRR)} of one rank trace."""
    return {q['query_id']: (q['shape'], q['hard_answers'], q['sort']['mrr'], q['expected']['mrr']) for q in trace_queries(path)}


def paired_strata(treatments, baseline):
    """{(policy, shape): per-query MRR differences}, treatments averaged per query (seed means).

    Every trace must cover the same queries with the same types and hard-answer counts.
    """
    reference = query_scores(baseline)
    scores = [query_scores(path) for path in treatments]
    if any(set(s) != set(reference) for s in scores) or not reference:
        raise ValueError(f'Suite comparison needs identical nonempty query coverage: {baseline}')
    strata = defaultdict(list)
    for query_id, (shape, answers, sort_mrr, expected_mrr) in reference.items():
        rows = [s[query_id] for s in scores]
        if any(row[0] != shape or row[1] != answers for row in rows):
            raise ValueError(f'Suite comparison changes query types or answers: {baseline}')
        strata['sort', shape].append(100 * (sum(row[2] for row in rows) / len(rows) - sort_mrr))
        strata['expected', shape].append(100 * (sum(row[3] for row in rows) / len(rows) - expected_mrr))
    return {key: np.asarray(values) for key, values in strata.items()}


def macro_effects(by_dataset, groups, *, bootstrap_samples=2000, seed=0):
    """{policy: {group: delta, ci95, datasets}} with equal type and dataset weights.

    by_dataset maps a dataset to its paired strata; groups map a name to (datasets, type filter).
    The bootstrap resamples queries within each dataset × type stratum, in bounded memory.
    """
    rng = np.random.default_rng(seed)
    output = {}
    for policy in ('sort', 'expected'):
        output[policy] = {}
        for group, (datasets, keep) in groups.items():
            members = [d for d in datasets if d in by_dataset]
            shapes = {d: [s for p, s in by_dataset[d] if p == policy and keep(s)] for d in members}
            members = [d for d in members if shapes[d]]
            if not members:
                continue
            point = float(np.mean([np.mean([by_dataset[d][policy, s].mean() for s in shapes[d]]) for d in members]))
            interval = None
            if bootstrap_samples:
                draws = np.zeros(bootstrap_samples)
                for d in members:
                    for s in shapes[d]:
                        values = by_dataset[d][policy, s]
                        chunk = max(1, 4_000_000 // len(values))
                        means = np.concatenate([values[rng.integers(0, len(values), size=(min(chunk, bootstrap_samples - start), len(values)))].mean(1)
                                                for start in range(0, bootstrap_samples, chunk)])
                        draws += means / len(shapes[d]) / len(members)
                interval = [float(v) for v in np.quantile(draws, [.025, .975])]
            output[policy][group] = dict(delta=point, ci95=interval, datasets=len(members))
    return output


def suite_effects(records, *, bootstrap_samples=2000, seed=0):
    """Suite-level paired effects of every adapter recipe: against its no-adapter control and against UltraQuery.

    Seed replicates (entry IDs differing only by -seedN) are averaged per query; the control,
    which identity calibration makes independent of the adapter seed, is its lowest seed tag.
    Only complete test runs with corrected (+H) or released (UltraQuery) filters are compared.
    """
    from dicee.query_answering.catalog import BENCHMARK_DATASETS, PLUS_H_DATASETS, dataset_spec

    from .paper.summary import system_of
    suites = {'ultraquery': [d for d in BENCHMARK_DATASETS if dataset_spec(d)[0] in ('transductive', 'inductive-e', 'inductive-er')],
              'plus_h': list(PLUS_H_DATASETS)}
    negation = {s for s in PLUS_H_SHAPES if 'n' in s}
    systems = defaultdict(lambda: defaultdict(dict))
    for record in records:
        target_filter = 'corrected' if record['dataset'] in PLUS_H_DATASETS else 'released'
        if not record['complete'] or record['answer_filter'] != target_filter or record['id'].endswith('-released-filters'):
            continue
        system, tag = system_of(record)
        systems[system][record['dataset']][tag] = record['trace']
    output = []
    for (method, identity, recipe), runs in sorted(systems.items()):
        if identity or not method.endswith('-adapter'):
            continue
        control = systems.get((method, True, recipe), {})
        baseline = systems.get(('ultraquery', False, 'ultraquery'), {})
        for suite, datasets in suites.items():
            present = [d for d in datasets if d in runs]
            if not present:
                continue
            families = {'all': (datasets, lambda s: True), 'epfo': (datasets, lambda s: s not in negation),
                        'negation': (datasets, lambda s: s in negation)}
            if suite == 'ultraquery':
                families |= {family: ([d for d in datasets if dataset_spec(d)[0] == family], lambda s: True)
                             for family in ('transductive', 'inductive-e', 'inductive-er')}
            seeds = sorted(set.intersection(*(set(runs[d]) for d in present)))
            for comparison, reference in (('adapter-minus-identity', control), ('adapter-minus-ultraquery', baseline),
                                          ('identity-minus-ultraquery', baseline if control else {})):
                if comparison == 'identity-minus-ultraquery':
                    treatments = {d: [control[d][min(control[d])]] for d in present if d in control}
                else:
                    treatments = {d: [runs[d][tag] for tag in seeds] for d in present}
                shared = [d for d in present if d in reference and d in treatments and treatments[d]]
                if not shared:
                    continue
                base = {d: reference[d][min(reference[d])] for d in shared}
                strata = {d: paired_strata(treatments[d], base[d]) for d in shared}
                per_seed = []
                if comparison != 'identity-minus-ultraquery':
                    for tag in seeds:
                        single = {d: paired_strata([runs[d][tag]], base[d]) for d in shared}
                        values = macro_effects(single, families, bootstrap_samples=0)
                        per_seed.append(dict(seed_tag=tag, **{p: {g: v['delta'] for g, v in values[p].items()} for p in values}))
                output.append(dict(comparison=comparison, method=method, recipe=recipe, suite=suite,
                                   datasets=len(shared), complete_suite=len(shared) == len(datasets),
                                   seed_tags=seeds if comparison != 'identity-minus-ultraquery' else [],
                                   macro=macro_effects(strata, families, bootstrap_samples=bootstrap_samples, seed=seed),
                                   per_seed=per_seed,
                                   uncertainty=dict(unit='paired query', stratification='dataset x query type',
                                                    resamples=bootstrap_samples, seed=seed,
                                                    scope='Conditional on frozen models, adapters and the evaluated seeds.')))
    return output
