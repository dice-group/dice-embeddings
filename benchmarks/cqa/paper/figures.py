"""Scientific figures from the same saved reports as the paper tables.

Imported only by --figures. No models, datasets, rank traces or referenced files
are loaded. Fractions remain fractions in figures.json; displayed MRR is x100.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from . import summary, tables

# Okabe-Ito colours; the first two mark the two adapter backbones in every figure.
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
METHOD_STYLES = {'ULTRA + adapter': ('#0072B2', 'o', 1.6), 'TRIX + adapter': ('#D55E00', 'D', 1.6),
                 'QTO': ('#009E73', 's', 1.0), 'CQD-Hybrid': ('#CC79A7', 'v', 1.0), 'GNN-QE': ('#E69F00', '^', 1.0),
                 'UltraQuery': ('#56B4E9', 'P', 1.0), 'CQD': ('#999999', 'X', .9), 'CLMPT': ('#666666', '<', .9),
                 'ConE': ('#BBBBBB', '>', .9)}
BACKBONES = ('ULTRA', 'TRIX')
ADAPTERS = ('ultra-adapter', 'trix-adapter')
PROFILE_TYPES = ('3p', '4p', '3i', '4i')
FAMILY_ROWS = (('transductive', 'Transductive'), ('inductive-e', 'Inductive (e)'), ('inductive-er', 'Inductive (e,r)'),
               ('plus_h', '+H'))
FIGURE_NAMES = ('main-01-adapter-gains', 'main-02-hardness-profiles', 'appendix-01-transfer-gains',
                'appendix-02-adapter-gains-by-dataset')
COMPOSITION_NAME = 'appendix-03-hardness-composition'
TEXT_WIDTH = 5.5  # inches: one text column of NeurIPS/ICLR; two-column venues use figure*


def figure_names(*, hardness_composition=False):
    return FIGURE_NAMES + ((COMPOSITION_NAME,) if hardness_composition else ())


def default_datasets():
    defaults = tables.empty_reports()
    return sorted({r['dataset'] for r in defaults.results.values() if tables.family(r['dataset'])},
                  key=lambda d: tables.dataset_order(tables.dataset_name(d)))


def integer(value, name):
    if tables.is_missing(value):
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def composition_data(reports):
    """Answer fractions from one run/label condition; never pool model counts.

    Absent bins become zero only when the reported parent answer total proves
    that the supplied bins already partition all answers. Missing counts never
    become zero, and query counts cannot be used as answer denominators.
    """
    cohorts = defaultdict(lambda: defaultdict(dict))
    for row in reports.difficulty:
        if row['grouping'] != 'inferred_positive_edges':
            continue
        count = tables.missing_link_count(row)
        key = (row.get('dataset'), row.get('entry'), row.get('comparison_graph'), row.get('answer_filter'))
        previous = cohorts[key][row['shape']].get(count)
        if previous is not None:
            for field in ('hard_answers', 'parent_hard_answers', 'parent_queries', 'available_queries', 'complete_shape'):
                a, b = previous.get(field), row.get(field)
                if a is not None and b is not None and a != b:
                    raise ValueError('Conflicting hardness composition counts')
            row = {**previous, **{k: v for k, v in row.items() if v is not None}}
        cohorts[key][row['shape']][count] = row
    candidates = defaultdict(list)
    for (dataset, entry, graph, filters), shapes in cohorts.items():
        bins = []
        for shape in tables.PLUS_H_TYPES:
            rows = shapes.get(shape, {})
            bound = tables.POSITIVE_EDGES[shape]
            counts = [integer(rows[k].get('hard_answers'), 'Answer count') if k in rows else None
                      for k in range(1, bound + 1)]
            totals = {integer(r['parent_hard_answers'], 'Parent answer count') for r in rows.values()
                      if not tables.is_missing(r.get('parent_hard_answers'))}
            if len(totals) > 1:
                raise ValueError('Hardness bins disagree about parent answer counts')
            total = next(iter(totals), None)
            coverage = {tables.hardness_scope(reports, r)[0] for r in rows.values()}
            if len(coverage) > 1:
                raise ValueError('Hardness bins disagree about parent-query coverage')
            complete = next(iter(coverage), None)
            zero = integer(rows[0].get('hard_answers'), 'Zero-cost answer count') if 0 in rows else 0
            known = sum(v for v in counts if v is not None)
            if total is not None and known + (zero or 0) > total:
                raise ValueError('Hardness answer counts exceed the parent total')
            valid = total is not None and total > 0 and known == total and zero == 0
            fractions = [(v or 0) / total for v in counts] if valid else None
            if valid:
                counts = [v if v is not None else 0 for v in counts]
            bins.append(dict(shape=shape, counts=counts, total=total, fractions=fractions,
                             complete=complete, status='available' if valid else 'missing or incomplete answer counts'))
        candidates[dataset].append(dict(dataset=dataset, entry=entry, graph=graph, filters=filters, bins=bins))
    output = []
    for dataset in tables.PLUS_H_DATASETS:
        options = candidates[dataset]
        # Full evaluations on the same graph/filters must describe the same
        # answer distribution. They are replicated labels, not extra samples.
        evidence = {}
        for option in options:
            for item in option['bins']:
                if item['complete'] is not True or item['fractions'] is None:
                    continue
                key = (option['graph'], option['filters'], item['shape'])
                value = (item['total'], item['counts'])
                if key in evidence and evidence[key] != value:
                    raise ValueError('Full runs disagree about hardness answer distribution')
                evidence[key] = value
        def priority(option):
            valid = [b for b in option['bins'] if b['fractions'] is not None]
            return (not valid, option['filters'] != 'corrected', option['graph'] != 'train+valid',
                    any(b['complete'] is not True for b in valid), -len(valid), option['entry'] or '')
        if options:
            output.append(min(options, key=priority))
        else:
            output.append(dict(dataset=dataset, entry=None, graph=None, filters=None,
                               bins=[dict(shape=s, counts=None, total=None, fractions=None, complete=None,
                                          status='missing answer counts') for s in tables.PLUS_H_TYPES]))
    return output


def adapter_data(reports, policy):
    pairs = {(e['learned'], e['control']): e for e in reports.effects['adapter']}
    for record in reports.results.values():
        if record['paired_with'] in reports.results:
            pairs.setdefault((record['paired_with'], record['id']), {})
    primary = {r['id'] for r in tables.paper_records(reports)}
    points = []
    seen = set()
    for (learned_id, control_id), effect in sorted(pairs.items()):
        learned, control = reports.results.get(learned_id), reports.results.get(control_id)
        if learned and learned_id not in primary:
            continue
        record = tables.entry_record(reports, learned_id, effect.get('dataset'))
        if record['method'] not in ADAPTERS:
            continue
        key = (record['dataset'], record['method'])
        if key in seen:
            raise ValueError('Ambiguous adapter pairs for the same dataset/backbone; include raw results to select the primary recipe')
        seen.add(key)
        metric = tables.effect_metrics(effect, learned, control, policy, kind='adapter', direction=-1)
        value = metric.get('mrr')
        interval = metric.get('mrr_ci95')
        tables.number(value, signed=True)
        tables.ci(interval)
        if tables.is_missing(value):
            value = None
        if interval is not None and any(tables.is_missing(v) for v in interval):
            interval = None
        complete = learned['complete'] if learned else None
        identity_verified = bool(learned and control and all(
            tables.execution_settings(learned).get(field) is not None
            and tables.execution_settings(learned).get(field) == tables.execution_settings(control).get(field)
            for field in ('candidate_sha256', 'context_sha256')))
        points.append(dict(dataset=record['dataset'], method=BACKBONES[ADAPTERS.index(record['method'])],
                           learned=learned_id, identity=control_id, value=value, ci=interval,
                           complete=complete, identity_verified=identity_verified,
                           graph=record.get('graph'), filters=record.get('filter'),
                           scope=tables.scope(learned) if learned else 'coverage unavailable',
                           status='missing paired MRR' if value is None else 'available'))
    if not points:
        for dataset in (*tables.PLUS_H_DATASETS, *default_datasets()):
            for method in BACKBONES:
                points.append(dict(dataset=dataset, method=method, value=None, ci=None, complete=None,
                                   status='missing paired MRR', scope='coverage unavailable'))
    return points


def transfer_data(reports, policy):
    selected = {(r['dataset'], r['method']): r for r in tables.paper_records(reports)
                if tables.family(r['dataset'])}
    datasets = default_datasets() if reports.template else sorted({d for d, _ in selected},
                key=lambda d: tables.dataset_order(tables.dataset_name(d))) or default_datasets()
    points = []
    for dataset in datasets:
        baseline = selected.get((dataset, 'ultraquery'))
        for method, name in zip(ADAPTERS, BACKBONES):
            model = selected.get((dataset, method))
            status = 'available'
            if not baseline or not model:
                status = 'missing matched UltraQuery/adapter results'
            elif not baseline['complete'] or not model['complete']:
                status = 'requires full 14-type test results'
            else:
                for field in ('split', 'graph', 'filter'):
                    if baseline.get(field) is None or model.get(field) is None:
                        status = 'comparison conditions unavailable'
                    elif baseline[field] != model[field]:
                        raise ValueError(f'Transfer comparison changes {field}: {dataset}')
                if baseline['counts'] != model['counts']:
                    raise ValueError(f'Transfer comparison changes query/answer counts: {dataset}')
                a = tables.execution_settings(baseline).get('candidates')
                b = tables.execution_settings(model).get('candidates')
                if a is None or b is None:
                    status = 'candidate domain unavailable'
                elif a != b:
                    raise ValueError(f'Transfer comparison changes candidate domain: {dataset}')
                for field in ('candidate_sha256', 'context_sha256'):
                    a = tables.execution_settings(baseline).get(field)
                    b = tables.execution_settings(model).get(field)
                    if a is None or b is None:
                        status = 'candidate/context identity unavailable'
                    elif a != b:
                        raise ValueError(f'Transfer comparison changes {field}: {dataset}')
            for category in ('epfo', 'negation'):
                category_status = status
                value = None
                if status == 'available':
                    shapes = [s for s in tables.ULTRA_TYPES if ('n' in s) == (category == 'negation')]
                    scores = [tables.policy_values(r, policy)['per_shape'].get(s, {}).get('mrr')
                              for r in (model, baseline) for s in shapes]
                    for score in scores:
                        tables.number(score)
                    if all(not tables.is_missing(v) for v in scores):
                        n = len(shapes)
                        value = (sum(scores[:n]) - sum(scores[n:])) / n
                    else:
                        category_status = 'missing per-type MRR for selected tie policy'
                points.append(dict(dataset=dataset, family=tables.family(dataset), method=name, category=category,
                                   value=value, baseline=baseline['id'] if baseline else None,
                                   entry=model['id'] if model else None, status=category_status,
                                   graph=model.get('graph') if model else None,
                                   filters=model.get('filter') if model else None))
    return points


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#BBBBBB')
    ax.tick_params(length=2, color='#BBBBBB')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E6E6E6', linewidth=.5)


def composition_plot(plt, data):
    # Identical, fully measured distributions need only one panel. Missing or
    # partial coverage cannot establish equality between benchmark datasets.
    comparable = all(b['fractions'] is not None and b['complete'] is True
                     for cohort in data for b in cohort['bins'])
    signatures = [(c['graph'], c['filters'], [b['fractions'] for b in c['bins']]) for c in data]
    shared = comparable and all(s == signatures[0] for s in signatures)
    panels = data[:1] if shared else data
    fig, axes = plt.subplots(len(panels), 1, figsize=(TEXT_WIDTH, 2.4 if shared else 5.0),
                             squeeze=False, layout='constrained')
    axes = axes[:, 0]
    for ax, cohort in zip(axes, panels):
        for i, item in enumerate(cohort['bins']):
            fractions = item['fractions']
            if fractions is None:
                ax.text(i, 50, '-', ha='center', va='center', color='#666666')
                continue
            bottom = 0
            for level, value in enumerate(fractions):
                ax.bar(i, 100 * value, bottom=bottom, width=.73, color=COLORS[level],
                       edgecolor='white', linewidth=.3, hatch='//' if item['complete'] is not True else None)
                bottom += 100 * value
        ax.set_xticks(range(len(tables.PLUS_H_TYPES)), tables.PLUS_H_TYPES)
        ax.set_xlim(-.5, len(tables.PLUS_H_TYPES) - .5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel('Hard answers (%)')
        graph, filters = cohort['graph'] or '-', cohort['filters'] or '-'
        datasets = ', '.join(tables.dataset_name(c['dataset']) for c in data) if shared else tables.dataset_name(cohort['dataset'])
        separator = '\n' if shared else '  |  '
        ax.set_title(f"{datasets}{separator}label graph: {graph}; filters: {filters}", loc='left', fontsize=8)
        style_axes(ax)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', color='#E6E6E6', linewidth=.5)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=c, label=str(i)) for i, c in enumerate(COLORS, 1)],
                   title='Missing positive links', ncol=4, fontsize=7, title_fontsize=7,
                   loc='lower right', bbox_to_anchor=(1, 1.02), frameon=False)
    return fig


def forest_axes(ax, points, datasets, limits, *, legend=False):
    from matplotlib.lines import Line2D
    lookup = {(p['dataset'], p['method']): p for p in points}
    for i, dataset in enumerate(datasets):
        if i % 2 == 0:
            ax.axhspan(i - .48, i + .48, color='#F7F7F7', zorder=0)
        for j, method in enumerate(BACKBONES):
            point = lookup.get((dataset, method), {})
            value = point.get('value')
            y = i + (j - .5) * .26
            if value is None:
                ax.text(.97, y, '-', transform=ax.get_yaxis_transform(), ha='center', va='center',
                        color=COLORS[j], fontsize=8)
                continue
            interval = point.get('ci')
            if interval is not None:
                ax.hlines(y, 100 * interval[0], 100 * interval[1], color=COLORS[j], linewidth=1.2)
                ax.vlines([100 * v for v in interval], y - .06, y + .06, color=COLORS[j], linewidth=.7)
            ax.plot(100 * value, y, marker='o' if j == 0 else 'D', markersize=4,
                    markerfacecolor=COLORS[j] if point.get('complete', True) is True and point.get('identity_verified', True) else 'white',
                    markeredgecolor=COLORS[j], linestyle='none', zorder=3)
    ax.set_yticks(range(len(datasets)), [tables.dataset_name(d) for d in datasets])
    ax.set_ylim(len(datasets) - .5, -.5)
    ax.set_xlim(*limits)
    ax.axvline(0, color='#777777', linewidth=.7)
    style_axes(ax)
    if legend:
        ax.figure.legend(handles=[Line2D([], [], color=COLORS[j], marker='o' if j == 0 else 'D', linestyle='none',
                                         markersize=4, label=name) for j, name in enumerate(BACKBONES)],
                         loc='outside upper right', ncol=2, frameon=False, fontsize=7)


def effect_limits(points):
    """Zero and every value or interval endpoint stay visible, without an empty mirrored half."""
    endpoints = [100 * v for p in points for v in [p.get('value'), *(p.get('ci') or [])] if v is not None]
    low, high = min([0, *endpoints]), max([0, *endpoints])
    pad = max(.08 * (high - low), .5)
    return low - pad, high + pad


def adapter_plot(plt, points):
    datasets = sorted({p['dataset'] for p in points}, key=lambda d: tables.dataset_order(tables.dataset_name(d)))
    groups = [(name, ds) for name, ds in (
        ('+H', [d for d in datasets if d in tables.PLUS_H_DATASETS]),
        ('UltraQuery benchmark', [d for d in datasets if d not in tables.PLUS_H_DATASETS])) if ds]
    height = .9 * len(groups) + .17 * len(datasets) + .4
    fig, axes = plt.subplots(len(groups), 1, figsize=(TEXT_WIDTH, height), squeeze=False,
                             gridspec_kw={'height_ratios': [max(3, len(ds)) for _, ds in groups]}, layout='constrained')
    limits = effect_limits(points)
    for index, ((name, ds), ax) in enumerate(zip(groups, axes[:, 0])):
        forest_axes(ax, points, ds, limits, legend=index == 0)
        ax.set_title(name, loc='left', fontsize=8)
        ax.set_xlabel('MRR gain from the adapter (points)')
    return fig


def transfer_plot(plt, points):
    datasets = sorted({p['dataset'] for p in points}, key=lambda d: tables.dataset_order(tables.dataset_name(d)))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.2 + .17 * len(datasets)), sharey=True, layout='constrained')
    limits = effect_limits(points)
    for category, ax in zip(('epfo', 'negation'), axes):
        forest_axes(ax, [p for p in points if p['category'] == category], datasets, limits, legend=category == 'epfo')
        ax.set_title('Positive queries (EPFO)' if category == 'epfo' else 'Negated queries', fontsize=8)
        ax.set_xlabel('Adapter minus UltraQuery MRR (points)')
        for i in range(1, len(datasets)):
            if tables.family(datasets[i]) != tables.family(datasets[i - 1]):
                ax.axhline(i - .5, color='#AAAAAA', linewidth=.7)
    return fig


def validate_seed_pair(learned, control):
    """A learned seed and the shared no-adapter control must score the same queries the same way.

    Unlike same-run pairs, a seed's recipe hash covers its own adapter file, so only
    the adapter identity may differ: the backbone, execution options, query and
    answer counts, candidates and context must all match.
    """
    reference = dict(control, raw=dict(control.get('raw') or {}, graph_recipe_sha256=None),
                     detail=dict(control.get('detail') or {}, graph_recipe_sha256=None))
    candidate = dict(learned, raw=dict(learned.get('raw') or {}, graph_recipe_sha256=None),
                     detail=dict(learned.get('detail') or {}, graph_recipe_sha256=None))
    tables.validate_pair(candidate, reference, kind='adapter')


def family_gain_data(reports: 'tables.Reports', policy: str) -> list[dict]:
    """Learned-minus-identity MRR per dataset and backbone, averaged over adapter seeds.

    Every seed of the primary recipe is paired with the no-adapter control of the
    same dataset, which identity calibration makes independent of the seed. Pairs
    must pass the appendix pair checks except for the adapter identity itself. No
    intervals here: per-dataset paired intervals are in the appendix figure and table.
    """
    points = []
    for suite in ('ultraquery', 'plus_h'):
        selected = summary.selected_systems(reports, suite)
        for name, method in zip(BACKBONES, ADAPTERS):
            learned = next((runs for (m, identity, _), runs in selected.items() if m == method and not identity), {})
            control = next((runs for (m, identity, _), runs in selected.items() if m == method and identity), {})
            for dataset in summary.suite_datasets(suite):
                family = 'plus_h' if suite == 'plus_h' else tables.family(dataset)
                point = dict(dataset=dataset, family=family, method=name, value=None, seeds=0,
                             status='missing learned/identity pair')
                if dataset in learned and dataset in control:
                    reference = next(iter(control[dataset].values()))
                    base = summary.category_mean(reference, policy, 'all')
                    deltas = []
                    for record in learned[dataset].values():
                        validate_seed_pair(record, reference)
                        value = summary.category_mean(record, policy, 'all')
                        if value is not None and base is not None:
                            deltas.append(value - base)
                    if deltas:
                        point.update(value=sum(deltas) / len(deltas), seeds=len(deltas), status='available')
                    else:
                        point['status'] = 'missing per-type MRR for selected tie policy'
                points.append(point)
    return points


def family_gain_plot(plt, points: list[dict]):
    """Datasets as small dots and family means as large markers, one lane per backbone."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.3), layout='constrained')
    rows = [(key, label) for key, label in FAMILY_ROWS]
    for i, (key, label) in enumerate(rows):
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color='#F7F7F7', zorder=0)
        for j, name in enumerate(BACKBONES):
            y = i + (j - .5) * .34
            selected = [100 * p['value'] for p in points if p['family'] == key and p['method'] == name and p['value'] is not None]
            if not selected:
                ax.text(.97, y, '-', transform=ax.get_yaxis_transform(), ha='center', va='center', color=COLORS[j], fontsize=8)
                continue
            jitter = [(k % 5 - 2) * .025 for k in range(len(selected))]
            ax.scatter(selected, [y + d for d in jitter], s=9, color=COLORS[j], alpha=.45, linewidths=0, zorder=2)
            ax.plot(sum(selected) / len(selected), y, marker='o' if j == 0 else 'D', markersize=6.5, color=COLORS[j],
                    markeredgecolor='black', markeredgewidth=.6, linestyle='none', zorder=3)
    counts = {key: len({p['dataset'] for p in points if p['family'] == key}) for key, _ in rows}
    ax.set_yticks(range(len(rows)), [f'{label} ({counts[key]})' for key, label in rows])
    ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xlim(*effect_limits(points))
    ax.axvline(0, color='#777777', linewidth=.7)
    ax.set_xlabel('MRR gain from the adapter (points)')
    style_axes(ax)
    handles = [Line2D([], [], color=COLORS[j], marker='o' if j == 0 else 'D', linestyle='none', markersize=5,
                      markeredgecolor='black', markeredgewidth=.5, label=name) for j, name in enumerate(BACKBONES)]
    handles.append(Line2D([], [], color='#777777', marker='o', linestyle='none', markersize=3, alpha=.6, label='single dataset'))
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1.01), ncol=3, frameon=False, fontsize=7)
    return fig


def hardness_profile_data(reports: 'tables.Reports', policy: str, types: tuple[str, ...] = PROFILE_TYPES) -> list[dict]:
    """MRR by missing positive links for every method, one run and condition per dataset/method.

    Bins come only from supplied answer-level difficulty rows; zero-cost diagnostics
    and coarse or released groupings are excluded, and no bin is interpolated.
    """
    primary = {r['id'] for r in tables.paper_records(reports)}
    rows = [r for r in tables.paper_hardness_rows(reports) if r['grouping'] == 'inferred_positive_edges'
            and r['shape'] in types and tables.missing_link_count(r) > 0]
    rows = [r for r in rows if not summary.identity_run(tables.hardness_record(reports, r) | {'id': r.get('entry') or ''})]
    conditions = defaultdict(lambda: defaultdict(list))
    for row in rows:
        record = tables.hardness_record(reports, row)
        name = tables.short_method(tables.method_name(record))
        conditions[row.get('dataset'), name][row.get('entry'), row.get('comparison_graph'), row.get('answer_filter')].append(row)
    series = []
    for (dataset, name), options in sorted(conditions.items(), key=lambda item: str(item[0])):
        def priority(key):
            entry, graph, filters = key
            return (entry not in primary, graph != 'train+valid', filters != 'corrected',
                    any(tables.hardness_scope(reports, r)[0] is not True for r in options[key]), str(key))
        chosen = options[min(options, key=priority)]
        for shape in types:
            bins = {}
            for row in chosen:
                if row['shape'] == shape:
                    value = (row.get(policy) or {}).get('mrr')
                    tables.number(value)
                    bins[tables.missing_link_count(row)] = None if tables.is_missing(value) else value
            if bins:
                series.append(dict(dataset=dataset, shape=shape, method=name, entry=chosen[0].get('entry'),
                                   points=[dict(links=k, value=bins.get(k)) for k in range(1, tables.POSITIVE_EDGES[shape] + 1)]))
    if not series:
        for dataset in tables.PLUS_H_DATASETS:
            for shape in types:
                series.append(dict(dataset=dataset, shape=shape, method=None, entry=None,
                                   points=[dict(links=k, value=None) for k in range(1, tables.POSITIVE_EDGES[shape] + 1)]))
    return series


def hardness_profile_plot(plt, series: list[dict]):
    """Small multiples, datasets by query type; lines break at missing bins instead of bridging them."""
    from matplotlib.lines import Line2D
    datasets = [d for d in tables.PLUS_H_DATASETS if any(s['dataset'] == d for s in series)] or list(tables.PLUS_H_DATASETS)
    types = [t for t in PROFILE_TYPES if any(s['shape'] == t for s in series)] or list(PROFILE_TYPES)
    fig, axes = plt.subplots(len(datasets), len(types), figsize=(TEXT_WIDTH, 1.25 * len(datasets) + .55),
                             sharex='col', sharey='row', squeeze=False, layout='constrained')
    methods = sorted({s['method'] for s in series if s['method']},
                     key=lambda m: (list(METHOD_STYLES).index(m) if m in METHOD_STYLES else len(METHOD_STYLES), m))
    for i, dataset in enumerate(datasets):
        for j, shape in enumerate(types):
            ax = axes[i][j]
            bound = tables.POSITIVE_EDGES[shape]
            drawn = False
            for item in series:
                if item['dataset'] != dataset or item['shape'] != shape or not item['method']:
                    continue
                color, marker, width = METHOD_STYLES.get(item['method'], ('#444444', '.', .9))
                xs = [p['links'] for p in item['points']]
                ys = [math.nan if p['value'] is None else 100 * p['value'] for p in item['points']]
                if any(not math.isnan(y) for y in ys):
                    ax.plot(xs, ys, color=color, marker=marker, markersize=3.2, linewidth=width,
                            zorder=3 if 'adapter' in item['method'] else 2)
                    drawn = True
            if not drawn:
                ax.text(.5, .5, '-', transform=ax.transAxes, ha='center', va='center', color='#666666')
            ax.set_xticks(range(1, bound + 1))
            ax.set_xlim(.7, bound + .3)
            style_axes(ax)
            ax.grid(axis='y', color='#E6E6E6', linewidth=.5)
            if i == 0:
                ax.set_title(shape, fontsize=8)
            if j == 0:
                ax.set_ylabel(tables.dataset_name(dataset) + '\nMRR', fontsize=7)
            if i == len(datasets) - 1:
                ax.set_xlabel('missing links', fontsize=7)
    # Only once every panel is drawn: the shared row limits must cover all of its panels.
    for row in axes:
        row[0].set_ylim(bottom=0)
    if methods:
        handles = [Line2D([], [], color=METHOD_STYLES.get(m, ('#444444', '.', .9))[0],
                          marker=METHOD_STYLES.get(m, ('#444444', '.', .9))[1], markersize=3.5,
                          linewidth=METHOD_STYLES.get(m, ('#444444', '.', .9))[2], label=m) for m in methods]
        fig.legend(handles=handles, loc='outside upper center', ncol=min(len(handles), 5), frameon=False, fontsize=7)
    return fig


def figure_environment(name: str, caption: str, *, star: bool = False) -> str:
    """A LaTeX figure float for one exported figure; captions are escaped text."""
    environment = 'figure*' if star else 'figure'
    return '\n'.join([rf'\begin{{{environment}}}[t]', r'\centering',
                      rf'\includegraphics[width=\linewidth]{{{name}.pdf}}',
                      rf'\caption{{{tables.escape(caption)}}}\label{{fig:{name}}}', rf'\end{{{environment}}}'])


def generate_figures(reports, output, *, policy='sort', hardness_composition=False):
    if not (reports.results or reports.difficulty or any(reports.effects.values())):
        reports = tables.empty_reports()
    if policy not in ('sort', 'expected'):
        raise ValueError('Tie policy must be sort or expected')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ValueError('--figures requires matplotlib (already listed in project dependencies)') from error
    payload = dict(tie_policy=policy, score_units='fraction; displayed MRR differences x100',
                   template=reports.template, figures=[])
    specifications = (
        ('main-01-adapter-gains', 'Main paper', lambda r: family_gain_data(r, policy), family_gain_plot,
         'MRR gain (points) from the learned adapter over the same frozen backbone without it, over all query types. '
         'Small dots: single datasets (mean over adapter seeds); large markers: family means. Per-dataset paired 95% '
         'confidence intervals are in the appendix.'),
        ('main-02-hardness-profiles', 'Main paper', lambda r: hardness_profile_data(r, policy), hardness_profile_plot,
         'MRR on +H by the minimum number of links that must be predicted to reach an answer (x-axis), for the longest path '
         '(3p, 4p) and intersection (3i, 4i) query types. Missing bins are not drawn; all query types and counts are in the '
         'appendix.'),
        ('appendix-01-transfer-gains', 'Appendix', lambda r: transfer_data(r, policy), transfer_plot,
         'MRR of the adapter minus UltraQuery per dataset (points), for EPFO and negation query types. Circles: ULTRA; '
         'diamonds: TRIX.'),
        ('appendix-02-adapter-gains-by-dataset', 'Appendix', lambda r: adapter_data(r, policy), adapter_plot,
         'MRR of the adapter minus the same backbone without it per dataset (points), with paired 95% confidence '
         'intervals over queries. Open markers: partial coverage.'),
    )
    if hardness_composition:
        specifications += ((COMPOSITION_NAME, 'Appendix', composition_data, composition_plot,
                            'Optional benchmark-composition diagnostic: proportions of evaluated hard QA pairs by minimum missing positive links, relative to the selected label graph and answer filters. These numeric bins differ from the benchmark authors\' structural reduction groups, especially for unions. Fixed sampling alone is not a model result. QA-pair proportions do not give aggregate MRR weights because MRR averages answers within queries, then queries. Identical fully measured distributions share one panel. Missing counts are shown as dashes; hatched bars indicate partial or unknown parent-query coverage. Counts from different models are never pooled.'),)
    # Validate every chart's numerical data before writing any output.
    prepared = [(name, placement, build(reports), plot, caption)
                for name, placement, build, plot, caption in specifications]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
             'ytick.labelsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False}
    with plt.rc_context(style):
        for name, placement, data, plot, caption in prepared:
            fig = plot(plt, data)
            files = []
            for extension in ('pdf', 'svg', 'png'):
                path = output / f'{name}.{extension}'
                fig.savefig(path, dpi=220, facecolor='white')
                files.append(path.name)
            plt.close(fig)
            payload['figures'].append(dict(name=name, placement=placement, files=files, caption=caption, data=data))
    path = output / 'figures.json'
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    floats = [f'% {f["placement"]}\n' + figure_environment(f['name'], f['caption']) for f in payload['figures']]
    (output / 'figures.tex').write_text('% Generated by python -m benchmarks.cqa.paper --figures; needs graphicx.\n\n'
                                       + '\n\n'.join(floats) + '\n', encoding='utf-8')
    return payload
