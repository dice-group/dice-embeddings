"""Scientific figures from the same saved reports as the paper tables.

Imported only by --figures. No models, datasets, rank traces or referenced files
are loaded. Fractions remain fractions in figures.json; displayed MRR is x100.
"""

import json
from collections import defaultdict
from pathlib import Path

from . import tables

COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
BACKBONES = ('ULTRA', 'TRIX')
ADAPTERS = ('ultra-adapter', 'trix-adapter')
FIGURE_NAMES = ('main-01-adapter-gains', 'appendix-01-transfer-gains')
COMPOSITION_NAME = 'appendix-02-hardness-composition'


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
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 3.2 if shared else 6.0),
                             squeeze=False, layout='constrained')
    axes = axes[:, 0]
    fig.suptitle('+H hardness composition', fontsize=11)
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
        ax.set_title(f"{datasets}{separator}label graph: {graph}; filters: {filters}", loc='left', fontsize=9)
        style_axes(ax)
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', color='#E6E6E6', linewidth=.5)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=c, label=str(i)) for i, c in enumerate(COLORS, 1)],
                   title='Missing positive links', ncol=4, fontsize=7, title_fontsize=7,
                   loc='upper center', bbox_to_anchor=(.5, 1.38), frameon=False)
    fig.supxlabel("QA-pair proportions, not aggregate MRR weights\n"
                  "'-': missing counts; hatched bars: partial or unknown parent-query coverage", fontsize=7)
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
        ax.legend(handles=[Line2D([], [], color=COLORS[j], marker='o' if j == 0 else 'D', linestyle='none',
                                  markersize=4, label=name) for j, name in enumerate(BACKBONES)],
                  loc='lower right', bbox_to_anchor=(1, 1.02), ncol=2, frameon=False, fontsize=8)


def effect_limits(points):
    endpoints = [100 * v for p in points for v in [p.get('value'), *(p.get('ci') or [])] if v is not None]
    extent = max((abs(v) for v in endpoints), default=1)
    return -max(1, extent * 1.15), max(1, extent * 1.15)


def adapter_plot(plt, points):
    datasets = sorted({p['dataset'] for p in points}, key=lambda d: tables.dataset_order(tables.dataset_name(d)))
    groups = [(name, ds) for name, ds in (
        ('+H', [d for d in datasets if d in tables.PLUS_H_DATASETS]),
        ('UltraQuery benchmark', [d for d in datasets if d not in tables.PLUS_H_DATASETS])) if ds]
    height = 1.1 * len(groups) + .18 * len(datasets) + .8
    fig, axes = plt.subplots(len(groups), 1, figsize=(7.2, height), squeeze=False,
                             gridspec_kw={'height_ratios': [max(3, len(ds)) for _, ds in groups]}, layout='constrained')
    fig.suptitle('Learned vs identity adapter', fontsize=11)
    limits = effect_limits(points)
    for index, ((name, ds), ax) in enumerate(zip(groups, axes[:, 0])):
        forest_axes(ax, points, ds, limits, legend=index == 0)
        ax.set_title(name, loc='left', fontsize=9)
        ax.set_xlabel('Learned - identity MRR (points)')
    fig.supxlabel("Whiskers: supplied paired 95% CI; no whisker: CI unavailable\n"
                  "Open marker: partial/unknown coverage or pair identity; '-': missing paired MRR", fontsize=7)
    return fig


def transfer_plot(plt, points):
    datasets = sorted({p['dataset'] for p in points}, key=lambda d: tables.dataset_order(tables.dataset_name(d)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.8 + .2 * len(datasets)), sharey=True, layout='constrained')
    fig.suptitle('Transfer gains over UltraQuery', fontsize=11)
    limits = effect_limits(points)
    for category, ax in zip(('epfo', 'negation'), axes):
        forest_axes(ax, [p for p in points if p['category'] == category], datasets, limits, legend=category == 'epfo')
        ax.set_title('Positive queries (EPFO)' if category == 'epfo' else 'Negated queries', fontsize=9, pad=26)
        ax.set_xlabel('Adapter - UltraQuery MRR (points)')
        for i in range(1, len(datasets)):
            if tables.family(datasets[i]) != tables.family(datasets[i - 1]):
                ax.axhline(i - .5, color='#AAAAAA', linewidth=.7)
    fig.supxlabel("Equal query-type means within each dataset/category; full matched tests only\n"
                  "'-': missing or incomparable results; no confidence intervals estimated", fontsize=7)
    return fig


def generate_figures(reports, output, *, policy='expected', hardness_composition=False):
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
        ('main-01-adapter-gains', 'Main paper', lambda r: adapter_data(r, policy), adapter_plot,
         'Learned-minus-identity adapter macro MRR in points, under the selected tie policy. Whiskers are supplied paired 95% confidence intervals conditional on frozen weights, never estimated or combined. Supplied candidate/context identities must match. Open markers indicate partial/unknown coverage or unavailable pair identity; a point without a whisker has no supplied interval.'),
        ('appendix-01-transfer-gains', 'Appendix', lambda r: transfer_data(r, policy), transfer_plot,
         'Per-dataset adapter-minus-UltraQuery MRR in points, shown separately for EPFO and negation. Each category weights its query types equally. Only full 14-type tests with matching graph, filters, query/answer counts, candidate counts, and candidate/context identities are compared. These descriptive cross-method differences have no inferred confidence intervals.'),
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
    return payload
