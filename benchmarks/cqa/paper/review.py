"""Tables and figures of the rule-centred presentation (figure and table plan agreed 2026-10-08).

The training-free rule (QTO's softmax times observed degree in our beam executor) is a main system beside the adapter.
This module renders what the main tables in summary.py do not: where the gain over UltraQuery comes from (weights by
executor, and known facts by calibration), the calibration figure (gains over raw scores, and the rule's equivalence
to the adapter), the known-fact mechanism on +H, and the appendix tables of the review: paired subset contrasts,
mechanism diagnostics, +H strata and the protocol deviations.

Scores come from the saved reports; paired contrasts and diagnostics from the review-analyses bundle
(Reports.review), which are copied, never recomputed. Missing data is "-".
"""

from collections import defaultdict

from . import summary, tables

BACKBONE_NAMES = {'ultra': 'ULTRA', 'trix': 'TRIX', 'kgicl': 'KG-ICL', 'flock': 'Flock'}
SUITES = (('ultraquery', 'UQ-23'), ('plus_h', '+H'))


def contrast(reports: 'tables.Reports', name: str, suite: str, *, backbone: str | None = None, policy: str = 'sort') -> dict | None:
    """The named paired contrast of the review bundle for the suite (and backbone), or None."""
    for item in reports.review.get('contrasts', []):
        if (item['name'] == name and item['suite'] == suite and item.get('policy', 'sort') == policy
                and (backbone is None or item.get('backbone') == backbone)):
            return item
    return None


def family(item: dict | None, subset: str = 'all', category: str = 'all') -> dict | None:
    """One family row (all, epfo, negation) of a contrast's subset (all datasets or a prefix such as WikiTopicsQuery)."""
    if item is None:
        return None
    return (item.get('subsets') or {}).get(subset, {}).get(category)


def panel_float(caption: str, label: str, panels: list[tuple[str, str, list, list, str]], *, layout: str = 'paper',
                placement: str = 'H') -> str:
    """One table float with several tabulars: panels are (title, columns, header rows, body rows, notes)."""
    parts = []
    for index, (title, columns, header_rows, rows, notes) in enumerate(panels):
        body = summary.compact_table(caption, label, columns, header_rows, rows, star=False, notes=notes, layout=layout)
        lines = body.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith(r'\begin{threeparttable}'))
        end = max(i for i, line in enumerate(lines) if line.startswith(r'\end{threeparttable}'))
        # Each panel keeps its own threeparttable (one tabular and its notes), under its title; in the paper layout the
        # caption stays in the first panel, so it takes that table's width.
        inner = [line for line in lines[start:end + 1] if not line.startswith(r'\caption') or (index == 0 and layout != 'thesis')]
        if index:
            parts.append(r'\par\medskip')
        if layout == 'thesis':
            # The thesis's fitblock sets its body in one box, so each panel gets its own, under its title.
            parts.extend([rf'{{\small {title}}}\par\smallskip', r'\begin{fitblock}', *inner, r'\end{fitblock}'])
        else:
            parts.extend([rf'{{\footnotesize {title}}}\par\smallskip', *inner])
    if layout == 'thesis':
        lines = [rf'\begin{{table}}[{placement}]', rf'\caption{{{caption}}}\label{{{label}}}', *parts, r'\end{table}']
    else:
        lines = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', *parts, r'\end{table}']
    return '\n'.join(lines)


# Weights by executor, panel (a) of the gains table. Each cell lists candidate systems (method, identity, recipe); the
# first with runs on every dataset of the suite is shown. 'primary' is the main tables' adapter. The UltraQuery-LP
# arms of +H ('uqlp', 'uqlp-calibrated') are full-test runs of the review subsets driver.
EXECUTOR_COLUMNS = (('uq-plain', 'Plain'), ('uq-rule', '+facts+rule'), ('beam-rule', '+facts+rule'), ('beam-adapter', '+facts+adapter'))
WEIGHTS = (
    ('UltraQuery (query-trained from ULTRA 4g)', 'ultra', {
        'uq-plain': [('ultraquery', False, 'ultraquery')],
        'uq-rule': [('ultraquery', False, 'ultraquery-calibrated')],
        'beam-rule': [],  # paired subsets only (Table subset-contrasts); never shown in this full-test table
        'beam-adapter': [('ultra-adapter', False, 'ultra-uqweights')]}),
    ('ULTRA 3g, frozen', 'ultra', {
        'uq-plain': [('ultraquery-lp', False, 'ultraquery-lp'), ('ultraquery-lp', False, 'uqlp')],
        'uq-rule': [('ultraquery-lp', False, 'ultraquery-lp-calibrated'), ('ultraquery-lp', False, 'uqlp-calibrated')],
        'beam-rule': [('ultra-adapter', False, 'ultra-softmax-degree')],
        'beam-adapter': ['primary']}),
    ('ULTRA 4g, frozen', 'ultra', {
        'uq-plain': [('ultraquery-lp', False, 'ultraquery-lp-4g')],
        'uq-rule': [],
        'beam-rule': [],  # paired subsets only
        'beam-adapter': [('ultra-adapter', False, 'ultra-ultra_4g')]}),
)
ONE_HOP_ORDER = ('beam-adapter', 'beam-rule', 'uq-rule', 'uq-plain')  # 1p has one atom: monotone calibrations agree


def cell_runs(systems: dict, primary: dict, candidates: list, suite: str) -> dict:
    """The runs of the first candidate covering every dataset of the suite, else {}."""
    datasets = summary.suite_datasets(suite)
    for candidate in candidates:
        system = (('ultra-adapter', False, primary.get(('ultra-adapter', suite))) if candidate == 'primary' else candidate)
        runs = systems.get(system, {})
        # Full test splits only: a query sample never enters these tables.
        if runs and all(d in runs and all(r['complete'] for r in runs[d].values()) for d in datasets):
            return runs
    return {}


def one_hop(runs: dict, suite: str, policy: str) -> list[float]:
    """1p MRR per seed, mean over the suite's datasets."""
    datasets = summary.suite_datasets(suite)
    seeds = set.intersection(*(set(runs[d]) for d in datasets)) if runs and all(d in runs for d in datasets) else set()
    values = []
    for seed in sorted(seeds):
        scores = [tables.policy_values(runs[d][seed], policy)['per_shape'].get('1p', {}).get('mrr') for d in datasets]
        if all(not tables.is_missing(v) for v in scores):
            values.append(sum(scores) / len(scores))
    return values


def executor_rows(reports: 'tables.Reports', policy: str) -> list[dict]:
    """Panel (a): all-type MRR per weight set, suite and executor setting, and the weights' 1p MRR."""
    systems = summary.collect(reports)
    primary = summary.primary_recipes(reports, systems)
    rows = []
    for name, _, cells in WEIGHTS:
        for suite, suite_name in SUITES:
            values, one = {}, []
            for key, _ in EXECUTOR_COLUMNS:
                runs = cell_runs(systems, primary, cells[key], suite)
                values[key] = summary.seed_means(runs, summary.suite_datasets(suite), policy, 'all')
            for key in ONE_HOP_ORDER:
                one = one_hop(cell_runs(systems, primary, cells[key], suite), suite, policy)
                if one:
                    break
            rows.append(dict(weights=name, suite=suite_name, values=values, one_hop=one))
    return rows


# Known facts by calibration, panel (b): (row label, recipe suffix after the backbone, identity) with facts on, and the
# facts-off recipes tried in order (the +H full-test arms of the review subsets driver last).
FACT_ROWS = (('Raw scores', 'primary', True), ('Softmax × degree (rule)', 'softmax-degree', False), ('Adapter (ours)', 'primary', False))


def facts_rows(reports: 'tables.Reports', policy: str) -> list[dict]:
    """Panel (b): all-type MRR with known facts on and off, per backbone and calibration, both suites."""
    systems = summary.collect(reports)
    primary = summary.primary_recipes(reports, systems)
    rows = []
    for method in sorted({m for m, _ in primary}, key=lambda m: summary.system_order(m, False)):
        backbone = method.split('-')[0]
        for label, token, identity in FACT_ROWS:
            values = []
            for suite, _ in SUITES:
                recipe = primary.get((method, suite)) if token == 'primary' else f'{backbone}-{token}'
                on = systems.get((method, identity, recipe), {})
                if identity:
                    on = {d: {min(seeds): seeds[min(seeds)]} for d, seeds in on.items()}
                off_candidates = [(method, identity, f'{recipe}-facts-none')]
                if token != 'primary':
                    off_candidates.append((method, identity, f'{recipe}-facts-none-full'))
                off = cell_runs(systems, primary, off_candidates, suite)
                values.extend([summary.seed_means(on, summary.suite_datasets(suite), policy, 'all'),
                               summary.seed_means(off, summary.suite_datasets(suite), policy, 'all')])
            if any(values):
                rows.append(dict(backbone=BACKBONE_NAMES.get(backbone, backbone), name=label, values=values))
    return rows


def cost_rows(reports: 'tables.Reports') -> list[tuple[str, dict]]:
    """Cost of the executor columns for frozen ULTRA 3g on UQ-23 (review bundle 'cost'), as display strings."""
    cost = reports.review.get('cost') or {}
    rows = []
    for metric, label, digits in (('seconds_per_1000', 'Prediction time (s per 1000 queries)', 0),
                                  ('backbone_rows_per_query', 'Backbone rows per query', 1)):
        values = {key: (cost.get(key) or {}).get(metric) for key, _ in EXECUTOR_COLUMNS}
        if any(v is not None for v in values.values()):
            rows.append((label, {k: tables.MISSING if v is None else f'{v:.{digits}f}' for k, v in values.items()}))
    return rows


def gains_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    """Main table: where the gain over UltraQuery comes from, (a) weights by executor and (b) known facts by calibration."""
    rows = executor_rows(reports, policy)
    body_a = []
    for j, row in enumerate(rows):
        label = tables.escape(row['weights']) if j == 0 or rows[j - 1]['weights'] != row['weights'] else ''
        body_a.append((row['weights'], [label, row['suite'], *[summary.cell(row['values'][key]) for key, _ in EXECUTOR_COLUMNS],
                                        summary.cell(row['one_hop'])]))
    for label, values in cost_rows(reports):
        body_a.append(('cost', [tables.escape(label), 'UQ-23', *[values[key] for key, _ in EXECUTOR_COLUMNS], '']))
    header_a = [summary.grouped_header([("UltraQuery's executor", 2), ('Beam (ours)', 2)], 2),
                (['Weights', 'Suite', *[name for _, name in EXECUTOR_COLUMNS], '1p'], [])]
    where = r'Table~\ref{tab:subset-contrasts}' if layout == 'thesis' else 'the appendix'
    notes_a = ('MRR over all query types. Missing cells were not run on the full test split; subset results for the '
               f'UltraQuery and 4g weights with the rule in the beam are in {where}. UltraQuery is '
               'fine-tuned on FB15k-237 queries from ULTRA 4g, so its contrast with ULTRA 3g mixes the initialisation with query '
               'training; the same-initialisation contrast is UltraQuery against ULTRA 4g. In the beam, the adapter is refitted '
               'for the UltraQuery and 4g weights with the same recipe. 1p: single-link MRR of the weights. Plain UltraQuery LP '
               'keeps its published UQ-23 thresholds; +H has none, so the matching UQ-23 values are carried over (0.97 on '
               'NELL995+H, 0.8 elsewhere). Known facts: observed traversal in UltraQuery\'s executor, the known-fact override '
               'in the beam.')
    if any(group == 'cost' for group, _ in body_a):
        notes_a += (' Cost: frozen ULTRA 3g on UQ-23, prediction time without metric computation on one H100 NVL shared '
                    'with one other job; UltraQuery\'s executor ran its reference implementation and the beam our optimised '
                    'one, so times compare implementations as well as executors. Backbone rows: atom rows the beam requests, '
                    'cached rows included. One-off cost: UltraQuery fine-tunes for 10,000 steps of batch 32, 8 GPU-hours on '
                    'RTX 3090s (Galkin et al., 2024, Section 5.1); the rule fits nothing; the adapter fits 16 parameters on '
                    'source queries.')
    facts = facts_rows(reports, policy)
    body_b = []
    for j, row in enumerate(facts):
        label = tables.escape(row['backbone']) if j == 0 or facts[j - 1]['backbone'] != row['backbone'] else ''
        body_b.append((row['backbone'], [label, tables.escape(row['name']), *[summary.cell(v) for v in row['values']]]))
    header_b = [summary.grouped_header([('UQ-23', 2), ('+H', 2)], 2),
                (['Backbone', 'Calibration', 'Facts on', 'Facts off', 'Facts on', 'Facts off'], [])]
    notes_b = ('MRR over all query types. Facts off: the known-fact override disabled at test time, the same calibration '
               f'otherwise. A dash: not run on the full test split (subset contrasts in {where}).')
    caption = ('Where the gain over UltraQuery comes from: (a) weights by executor and (b) known facts by calibration.'
               + summary.seed_legend(len(v) for row in rows for v in row['values'].values()))
    panels = [('(a) Weights and executors', 'll' + 'c' * 5, header_a,
               body_a or [('', [tables.MISSING] * 7)], notes_a),
              ('(b) Known facts and calibration', 'llcccc', header_b, body_b or [('', [tables.MISSING] * 6)], notes_b)]
    return panel_float(caption, 'tab:main-gains', panels, layout=layout)


# Figure: calibration gains over raw scores (a) and the rule's equivalence to the adapter (b).
FIGURE_CALIBRATION = 'main-01-calibration'
FIGURE_MECHANISM = 'main-03-facts-mechanism'
FAMILY_ROWS = (('transductive', 'Transductive'), ('inductive-e', 'Inductive (e)'), ('inductive-er', 'Inductive (e,r)'),
               ('plus_h', '+H'))


def calibration_gain_points(reports: 'tables.Reports', policy: str) -> list[dict]:
    """Per backbone and dataset: the adapter's (seed mean) and the rule's all-type MRR minus the raw scores'."""
    systems = summary.collect(reports)
    primary = summary.primary_recipes(reports, systems)
    points = []
    for suite, _ in SUITES:
        for method in sorted({m for m, s in primary if s == suite}, key=lambda m: summary.system_order(m, False)):
            backbone = method.split('-')[0]
            control = systems.get((method, True, primary[method, suite]), {})
            runs = {'adapter': systems.get((method, False, primary[method, suite]), {}),
                    'rule': systems.get((method, False, f'{backbone}-{summary.RULE_TOKEN}'), {})}
            for dataset in summary.suite_datasets(suite):
                base = summary.category_mean(control[dataset][min(control[dataset])], policy, 'all') if dataset in control else None
                for kind, by_dataset in runs.items():
                    values = [summary.category_mean(r, policy, 'all') for r in by_dataset.get(dataset, {}).values()]
                    values = [v for v in values if v is not None]
                    points.append(dict(dataset=dataset, family='plus_h' if suite == 'plus_h' else tables.family(dataset),
                                       method=BACKBONE_NAMES.get(backbone, backbone), calibration=kind,
                                       value=sum(values) / len(values) - base if values and base is not None else None))
    return points


def equivalence_rows(reports: 'tables.Reports') -> list[dict]:
    """Rule minus adapter per backbone: UQ-23 and WikiTopics with dataset-clustered intervals, +H graphs query-level."""
    rows = []
    for backbone in ('ultra', 'trix', 'kgicl'):
        name = BACKBONE_NAMES[backbone]
        item = contrast(reports, 'rule-vs-adapter', 'ultraquery', backbone=backbone)
        for subset, label in (('all', 'UQ-23'), ('WikiTopicsQuery', 'WikiTopics, subset of UQ-23')):
            row = family(item, subset)
            if row:
                rows.append(dict(backbone=name, label=f'{label} ({row["datasets"]})', value=row['a_minus_b'],
                                 ci95=row['ci95_clustered'], ci90=row['ci90_clustered'], dots=list(row['per_dataset'].values()),
                                 clustered=True, margin=item.get('margin')))
        item = contrast(reports, 'rule-vs-adapter', 'plus_h', backbone=backbone)
        row = family(item)
        for dataset in tables.PLUS_H_DATASETS:
            if row and dataset in row.get('per_dataset', {}):
                interval = (row.get('per_dataset_intervals') or {}).get(dataset) or {}
                rows.append(dict(backbone=name, label=tables.dataset_name(dataset), value=row['per_dataset'][dataset],
                                 ci95=interval.get('ci95'), ci90=interval.get('ci90'), dots=[], clustered=False,
                                 margin=item.get('margin')))
    return rows


def verdict(row: dict) -> str:
    """TOST at the registered margin on the 90% interval, and the smallest margin the interval passes."""
    if not row.get('ci90') or row.get('margin') is None:
        return '-'
    low, high = row['ci90']
    smallest = max(abs(low), abs(high))
    if not row['clustered']:
        return f'descriptive (smallest margin {smallest:.2f})'
    passed = -row['margin'] < low and high < row['margin']
    interval = row.get('ci95') or (low, high)
    # An interval that excludes zero says more than "not shown equivalent": the rule is lower (or higher).
    if not passed and (interval[1] < 0 or interval[0] > 0):
        # An interval that excludes zero says more than "not shown equivalent", and no equivalence bound applies.
        return 'rule lower' if interval[1] < 0 else 'rule higher'
    return f'{"equivalent" if passed else "not shown equivalent"} (smallest margin {smallest:.2f})'


def calibration_data(reports: 'tables.Reports', policy: str) -> dict:
    return dict(gains=calibration_gain_points(reports, policy), equivalence=[dict(r, verdict=verdict(r)) for r in equivalence_rows(reports)])


def calibration_caption(data: dict) -> str:
    margin = next((r['margin'] for r in data['equivalence'] if r.get('margin') is not None), None)
    caption = ('(a) MRR gain over the raw scores of the same frozen backbone, over all query types: filled markers the '
               'adapter, open markers the training-free rule (softmax × degree); small dots single datasets, large markers '
               'family means. (b) Rule minus adapter, all-type MRR (points): thick bars 90% and whiskers 95% intervals, which '
               'resample datasets and queries (UQ-23 and its WikiTopics subset); light markers without bars are the three +H '
               'graphs, which resample queries only and are descriptive; faint dots are single datasets, and dots beyond the '
               'axis are counted at its edge.')
    if margin is not None:
        caption += (f' Shaded: the equivalence margin of {margin:g} MRR on either side of zero, fixed in the protocol on 8 '
                    'October 2026 after the dataset-clustered 95% intervals were known, 2.5 to 5 times the adapter-seed s.d. '
                    'of 0.1 to 0.2; each row states the two one-sided tests at that margin and the smallest margin its 90% '
                    'interval passes.')
    return caption


def calibration_plot(plt, data: dict):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from . import figures
    gains, rows = data['gains'], data['equivalence']
    names = [n for n in figures.BACKBONES if any(p['method'] == n and p['value'] is not None for p in gains)] or list(figures.BACKBONES[:2])
    height_b = .55 + .19 * max(len(rows), 3)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(figures.TEXT_WIDTH, 2.4 + height_b), layout='constrained',
                                 gridspec_kw={'height_ratios': [2.4, height_b]})
    for i, (key, label) in enumerate(FAMILY_ROWS):
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color='#F7F7F7', zorder=0)
        for position, name in enumerate(names):
            j = figures.BACKBONES.index(name)
            base = i + figures.lane(position, len(names), .3 if len(names) < 3 else .26)
            for k, kind in enumerate(('adapter', 'rule')):
                y = base + (k - .5) * .1
                selected = [100 * p['value'] for p in gains if p['family'] == key and p['method'] == name
                            and p['calibration'] == kind and p['value'] is not None]
                if not selected:
                    continue
                ax.scatter(selected, [y] * len(selected), s=7, color=figures.COLORS[j], alpha=.35, linewidths=0, zorder=2)
                ax.plot(sum(selected) / len(selected), y, marker=figures.MARKERS[j], markersize=6, linestyle='none', zorder=3,
                        color=figures.COLORS[j], markerfacecolor=figures.COLORS[j] if kind == 'adapter' else 'white',
                        markeredgecolor='black' if kind == 'adapter' else figures.COLORS[j], markeredgewidth=.7)
    counts = {key: len({p['dataset'] for p in gains if p['family'] == key}) for key, _ in FAMILY_ROWS}
    ax.set_yticks(range(len(FAMILY_ROWS)), [f'{label} ({counts[key]})' for key, label in FAMILY_ROWS])
    ax.set_ylim(len(FAMILY_ROWS) - .5, -.5)
    ax.set_xlim(*figures.effect_limits([p for p in gains if p['value'] is not None]))
    ax.axvline(0, color='#777777', linewidth=.7)
    ax.set_xlabel('(a) MRR gain over raw scores (points)')
    figures.style_axes(ax)
    # Two encodings, two labelled legend rows: colour and shape name the backbone (both panels), fill the calibration
    # (panel a only), drawn in neutral grey so it does not read as a further model.
    backbones = [Line2D([], [], color=figures.COLORS[figures.BACKBONES.index(n)], marker=figures.MARKERS[figures.BACKBONES.index(n)],
                        linestyle='none', markersize=5, markeredgecolor='black', markeredgewidth=.5) for n in names]
    calibrations = [Line2D([], [], color='#8C8C8C', marker='o', linestyle='none', markersize=5, markeredgecolor='black',
                           markeredgewidth=.5),
                    Line2D([], [], color='#8C8C8C', marker='o', linestyle='none', markersize=5, markerfacecolor='white',
                           markeredgewidth=.8)]
    blank = Patch(facecolor='none', edgecolor='none')
    columns = max(len(backbones), len(calibrations))
    rows_ = [([blank] + backbones + [blank] * (columns - len(backbones)), ['Backbone', *names] + [''] * (columns - len(names))),
             ([blank] + calibrations + [blank] * (columns - len(calibrations)),
              ['Calibration (a)', 'adapter (ours)', 'rule (softmax × degree)'] + [''] * (columns - len(calibrations)))]
    # Legends fill column by column, so interleave the two rows to get a grid with the headers in its first column.
    handles = [h for pair in zip(rows_[0][0], rows_[1][0]) for h in pair]
    texts = [s for pair in zip(rows_[0][1], rows_[1][1]) for s in pair]
    legend = ax.legend(handles, texts, loc='lower right', bbox_to_anchor=(1, 1.01), ncol=columns + 1, frameon=False, fontsize=7,
                       handletextpad=.4, columnspacing=1.2)
    for text in legend.get_texts():
        if text.get_text() in ('Backbone', 'Calibration (a)'):
            text.set_fontweight('bold')
            text.set_color('#444444')
    # (b) the forest of rule minus adapter.
    margin = next((r['margin'] for r in rows if r.get('margin') is not None), None)
    if margin is not None:
        bx.axvspan(-margin, margin, color='#EDEDED', zorder=0)
    labels = []
    # The axis covers every point estimate and interval; single-dataset dots beyond it are counted at its edge.
    endpoints = [v for r in rows for v in [r['value'], *(r.get('ci95') or [])] if v is not None]
    low, high = min([-1, *endpoints]) - .3, max([1, *endpoints]) + .3
    for i, row in enumerate(rows):
        j = figures.BACKBONES.index(row['backbone']) if row['backbone'] in figures.BACKBONES else 0
        color = figures.COLORS[j]
        inside = [d for d in row['dots'] if low <= d <= high]
        if inside:
            bx.scatter(inside, [i] * len(inside), s=5, color=color, alpha=.3, linewidths=0, zorder=2)
        for edge, beyond, marker, align in ((low, [d for d in row['dots'] if d < low], '<', 'left'),
                                            (high, [d for d in row['dots'] if d > high], '>', 'right')):
            if beyond:
                bx.plot(edge, i, marker=marker, markersize=3, color=color, alpha=.6, linestyle='none', zorder=2, clip_on=False)
                bx.text(edge + (.12 if marker == '<' else -.12), i - .32, f'{len(beyond)} beyond', fontsize=5, color=color,
                        ha=align, va='center')
        # Bars only where datasets are resampled; a +H graph's query-level interval is in its verdict, not drawn.
        if row.get('ci95') and row['clustered']:
            bx.hlines(i, *row['ci95'], color=color, linewidth=.8, zorder=3)
        if row.get('ci90') and row['clustered']:
            bx.hlines(i, *row['ci90'], color=color, linewidth=3, zorder=3)
        # The backbone's marker, always filled (fill means the calibration in panel a); light for the descriptive +H graphs.
        bx.plot(row['value'], i, marker=figures.MARKERS[j], markersize=4.5, linestyle='none', zorder=4, color=color,
                markerfacecolor=color, markeredgecolor='black', markeredgewidth=.7, alpha=1 if row['clustered'] else .45)
        labels.append(f'{row["backbone"]}: {row["label"]}')
        bx.text(1.01, i, row['verdict'], transform=bx.get_yaxis_transform(), ha='left', va='center', fontsize=6, color='#333333')
    if rows:
        bx.set_yticks(range(len(rows)), labels)
    else:
        bx.set_yticks([0], ['-'])
        bx.text(0, 0, '-', ha='center', va='center', color='#666666')
    bx.set_ylim(max(len(rows), 1) - .5, -.5)
    bx.set_xlim(low, high)
    bx.axvline(0, color='#777777', linewidth=.7)
    bx.set_xlabel('(b) Rule minus adapter, MRR (points)')
    figures.style_axes(bx)
    for i in range(1, len(rows)):
        if rows[i]['backbone'] != rows[i - 1]['backbone']:
            bx.axhline(i - .5, color='#BBBBBB', linewidth=.5)
    return fig


# Figure: what known facts do on +H, by the number of missing edges of an answer's cheapest proof.
MECHANISM_TYPES = ('2i', '3i', '4i', '2p', '3p', '4p')


def mechanism_data(reports: 'tables.Reports', policy: str) -> list[dict]:
    """Raw scores with known facts on and off on +H: per backbone, query type and missing edges c, the mean over the
    graphs that have both runs of (on - off), with the facts-off and facts-on levels (corrected filters, train+valid)."""
    values = defaultdict(dict)
    for row in tables.paper_hardness_rows(reports):
        method = row.get('method') or ''
        if (row['grouping'] != 'inferred_positive_edges' or row.get('answer_filter') != 'corrected'
                or row['shape'] not in MECHANISM_TYPES or not method.endswith('-adapter') or row['dataset'] not in tables.PLUS_H_DATASETS):
            continue
        backbone = method.split('-')[0]
        base = f'{backbone}-product-intersections-{row["dataset"]}'
        setting = {base + '-without-adapter': 'on', base + '-facts-none-without-adapter': 'off'}.get(row.get('entry'))
        value = (row.get(policy) or {}).get('mrr')
        if setting and not tables.is_missing(value):
            values[backbone, row['shape'], int(row['label'])][row['dataset'], setting] = value
    series = []
    for (backbone, shape, cost), found in sorted(values.items()):
        datasets = [d for d in tables.PLUS_H_DATASETS if (d, 'on') in found and (d, 'off') in found]
        if not datasets:
            continue
        on = sum(found[d, 'on'] for d in datasets) / len(datasets)
        off = sum(found[d, 'off'] for d in datasets) / len(datasets)
        series.append(dict(backbone=BACKBONE_NAMES.get(backbone, backbone), shape=shape, missing=cost, on=on, off=off,
                           delta=on - off, graphs=len(datasets), edges=tables.POSITIVE_EDGES[shape]))
    return series


def mechanism_caption(data: list[dict], reports: 'tables.Reports' = None) -> str:
    caption = ('Known facts on +H: MRR with the known-fact override on minus off (points), raw scores of the frozen backbones, '
               'by the number of missing edges in the cheapest proof of an answer (x-axis), mean of the three graphs. Top right, '
               'one line per backbone in its colour: MRR with facts off and on for answers whose every edge is missing. In '
               'intersections such an answer keeps its score, so every rank it loses goes to a non-answer lifted through a '
               'known edge; that no such answer improved follows from the scoring rule and is a consistency check of the '
               'implementation. The finding is how many non-answers overtake it')
    overtakers = (reports.review.get('accounting') or {}).get('rows') if reports is not None else None
    if overtakers:
        means = [r['overtakers_mean'] for r in overtakers]
        caption += f': {min(means):.0f} to {max(means):.0f} per answer on average, by graph, type and backbone (appendix).'
    else:
        caption += ' (appendix).'
    return caption


def mechanism_plot(plt, data: list[dict]):
    from matplotlib.lines import Line2D

    from . import figures
    fig, axes = plt.subplots(2, 3, figsize=(figures.TEXT_WIDTH, 3.6), sharey=True, layout='constrained')
    names = [n for n in figures.BACKBONES if any(s['backbone'] == n for s in data)] or list(figures.BACKBONES)
    limit = max([abs(100 * s['delta']) for s in data] or [1]) * 1.12
    for ax, shape in zip(axes.flat, MECHANISM_TYPES):
        edges = tables.POSITIVE_EDGES[shape]
        ax.axhline(0, color='#777777', linewidth=.7)
        drawn, labels = False, []
        for name in names:
            j = figures.BACKBONES.index(name)
            points = sorted((s for s in data if s['backbone'] == name and s['shape'] == shape), key=lambda s: s['missing'])
            if not points:
                continue
            drawn = True
            ax.plot([s['missing'] for s in points], [100 * s['delta'] for s in points], color=figures.COLORS[j],
                    marker=figures.MARKERS[j], markersize=3.5, linewidth=1.1)
            full = next((s for s in points if s['missing'] == edges), None)
            if full:
                labels.append((f'{100 * full["off"]:.1f} → {100 * full["on"]:.1f}', figures.COLORS[j]))
        # Facts off -> on for answers with every edge missing, one line per backbone in its colour, in the free corner.
        for k, (text, color) in enumerate(labels):
            ax.text(.98, .97 - .095 * k, text, transform=ax.transAxes, ha='right', va='top', fontsize=5.8, color=color)
        if not drawn:
            ax.text(.5, .5, '-', transform=ax.transAxes, ha='center', va='center', color='#666666')
        ax.set_title(shape, fontsize=8)
        ax.set_xticks(range(1, edges + 1))
        ax.set_xlim(.7, edges + .9)
        ax.set_ylim(-limit, limit)
        figures.style_axes(ax)
        ax.grid(axis='y', color='#E6E6E6', linewidth=.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('facts on − off (MRR)', fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel('missing edges', fontsize=7)
    handles = [Line2D([], [], color=figures.COLORS[figures.BACKBONES.index(n)], marker=figures.MARKERS[figures.BACKBONES.index(n)],
                      markersize=3.5, linewidth=1.1, label=n) for n in names]
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), frameon=False, fontsize=7)
    return fig


# Appendix figure: per-dataset agreement of rule and adapter (dumbbells of the gains over raw scores, Bland-Altman).
FIGURE_AGREEMENT = 'appendix-04-calibration-agreement'


def agreement_data(reports: 'tables.Reports', policy: str) -> list[dict]:
    """Per backbone and dataset: gains over raw scores of adapter and rule, and the rule-minus-adapter difference."""
    pairs = defaultdict(dict)
    for p in calibration_gain_points(reports, policy):
        pairs[p['method'], p['dataset']][p['calibration']] = p['value']
        pairs[p['method'], p['dataset']]['family'] = p['family']
    rows = []
    for (method, dataset), found in pairs.items():
        if found.get('adapter') is not None and found.get('rule') is not None:
            rows.append(dict(method=method, dataset=dataset, family=found['family'], adapter=found['adapter'], rule=found['rule']))
    return sorted(rows, key=lambda r: (tables.dataset_order(tables.dataset_name(r['dataset'])), r['method']))


def agreement_plot(plt, rows: list[dict]):
    from . import figures
    datasets = sorted({r['dataset'] for r in rows}, key=lambda d: tables.dataset_order(tables.dataset_name(d))) or ['-']
    names = [n for n in figures.BACKBONES if any(r['method'] == n for r in rows)] or list(figures.BACKBONES)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figures.TEXT_WIDTH, 1.0 + .16 * len(datasets)), layout='constrained',
                                 gridspec_kw={'width_ratios': [1.4, 1]})
    for i, dataset in enumerate(datasets):
        if i % 2 == 0:
            ax.axhspan(i - .48, i + .48, color='#F7F7F7', zorder=0)
        for position, name in enumerate(names):
            j = figures.BACKBONES.index(name)
            row = next((r for r in rows if r['dataset'] == dataset and r['method'] == name), None)
            if row is None:
                continue
            y = i + figures.lane(position, len(names), .24)
            ax.plot([100 * row['adapter'], 100 * row['rule']], [y, y], color=figures.COLORS[j], linewidth=.8, zorder=2)
            ax.plot(100 * row['adapter'], y, marker=figures.MARKERS[j], markersize=3.5, color=figures.COLORS[j], linestyle='none', zorder=3)
            ax.plot(100 * row['rule'], y, marker=figures.MARKERS[j], markersize=3.5, color=figures.COLORS[j], markerfacecolor='white',
                    linestyle='none', zorder=3)
    ax.set_yticks(range(len(datasets)), [tables.dataset_name(d) for d in datasets])
    ax.set_ylim(len(datasets) - .5, -.5)
    ax.axvline(0, color='#777777', linewidth=.7)
    ax.set_xlabel('Gain over raw scores (points): filled adapter, open rule')
    figures.style_axes(ax)
    for name in names:
        j = figures.BACKBONES.index(name)
        selected = [r for r in rows if r['method'] == name]
        if not selected:
            continue
        means = [50 * (r['adapter'] + r['rule']) for r in selected]
        differences = [100 * (r['rule'] - r['adapter']) for r in selected]
        bx.scatter(means, differences, s=10, color=figures.COLORS[j], marker=figures.MARKERS[j], linewidths=0, alpha=.8)
        mean = sum(differences) / len(differences)
        if len(differences) > 1:
            sd = (sum((d - mean) ** 2 for d in differences) / (len(differences) - 1)) ** .5
            for level, style in ((mean, '-'), (mean - 1.96 * sd, ':'), (mean + 1.96 * sd, ':')):
                bx.axhline(level, color=figures.COLORS[j], linewidth=.7, linestyle=style)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=figures.COLORS[figures.BACKBONES.index(n)], marker=figures.MARKERS[figures.BACKBONES.index(n)],
                      linestyle='none', markersize=4, label=n) for n in names]
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), frameon=False, fontsize=7)
    bx.axhline(0, color='#777777', linewidth=.5)
    bx.set_xlabel('Mean gain of rule and adapter (points)')
    bx.set_ylabel('Rule minus adapter (points)')
    figures.style_axes(bx)
    bx.grid(axis='y', color='#E6E6E6', linewidth=.5)
    return fig


def agreement_caption(rows: list[dict]) -> str:
    from . import figures
    names = [n for n in figures.BACKBONES if any(r['method'] == n for r in rows)] or list(figures.BACKBONES)
    return (figures.marker_legend(names) + ' '
            'Agreement of the training-free rule and the adapter per dataset (all-type MRR, adapter seed means). Left: gains '
            'over the raw scores of the same backbone, filled markers the adapter and open markers the rule, joined per '
            'dataset. Right: Bland-Altman plot of rule minus adapter against their mean gain, with each backbone\'s mean '
            'difference (solid) and limits of agreement 1.96 s.d. over datasets on either side (dotted).')


# Appendix: paired subset contrasts. Rows: (question, contrast name, A - B in words, prediction or None). A prediction is
# (suite, category, direction, threshold, text, registered): suite 'ultraquery', 'plus_h' or 'both' (judged per suite);
# direction '>' or '<' against the threshold (MRR points).
# Outcome: supported if the 95% interval lies beyond the threshold in the predicted direction, refuted if beyond it in
# the other, else inconclusive; '=' predicts equivalence within the threshold, judged on the 90% interval (two one-sided
# tests). UQ-23 uses dataset-clustered intervals, +H query-level ones (descriptive).
REGISTERED = '8 Oct 2026, 20:36'
SUBSET_ROWS = (
    ('KG-ICL rule', 'kgicl-softmax-vs-qto', 'softmax (d = 1) − softmax × degree',
     ('part', 'all', '<', 0, 'gives back part of the +H advantage over the adapter', REGISTERED)),
    ('KG-ICL rule', 'kgicl-softmax-vs-ties', 'level: softmax (d = 1) − tie-free cap',
     ('ultraquery', 'all', '>', 0, 'positive: d = 1 recovers most of the deficit', REGISTERED)),
    ('KG-ICL rule', 'kgicl-ties-vs-qto', 'tie repair: tie-free cap − softmax × degree', None),
    ('KG-ICL rule', 'kgicl-ties-vs-adapter', 'residual: tie-free cap − adapter', None),
    ('KG-ICL rule', 'kgicl-softmax-vs-adapter', 'softmax (d = 1) − adapter', None),
    ('ULTRA rule', 'ultra-softmax-vs-qto', 'softmax (d = 1) − softmax × degree',
     ('both', 'all', '<', 0, 'negative on both suites (crossover with KG-ICL)', REGISTERED)),
    ('TRIX mechanism', 'trix-ties-vs-qto', 'tie repair: tie-free cap − softmax × degree', None),
    ('TRIX mechanism', 'trix-masked-vs-ties', 'known-tail logits masked − unmasked (tie-free cap)',
     ('both', 'all', '<', 0, 'masking opens a deficit (both suites run)', REGISTERED)),
    ('Weights', 'ultra4g-vs-uqweights', 'ULTRA 4g − UltraQuery weights (beam, facts, rule)',
     ('both', 'all', '>', 0, '4g above UltraQuery\'s weights on both suites except FB15k', REGISTERED)),
    ('Weights', 'ultra4g-vs-3g', 'ULTRA 4g − ULTRA 3g (beam, facts, rule)', None),
    ('Selection', 'ultra-types14-vs-adapter', 'ULTRA 14-type adapter (one fit) − 2i/3i adapter (5 seeds)', None),
    ('Selection', 'trix-types14-vs-adapter', 'TRIX 14-type adapter (one fit) − 2i/3i adapter (5 seeds)', None),
    ('Generator bias', 'kgicl-hb-vs-adapter', 'KG-ICL hardness-balanced − standard adapter (5 seeds)', None),
    ('Generator bias', 'kgicl-hb-interaction', 'KG-ICL interaction: (hardness-balanced − standard) on +H minus on UQ-23, matched pairs',
     ('interaction', 'all', '>', 0, 'positive; matched pairs FB15k-237 and NELL995 against their +H versions', REGISTERED)),
    ('Generator bias', 'kgicl-hb-global-vs-global', 'KG-ICL hardness-balanced − standard global calibration', None),
    ('Generator bias', 'ultra-hb-vs-adapter-seed0', 'ULTRA hardness-balanced − standard adapter (seed 0)', None),
    ('Negation', 'ultra-negation-observed-vs-adapter', 'ULTRA adapter: negation from known facts − from memberships',
     ('ultraquery', 'negation', '<', -.5, 'model negation ahead by at least 0.5 on UQ-23; on +H the gap is smaller',
      '8 Oct 2026, 21:27')),
    ('Negation', 'ultra-sd-negation-observed-vs-sd', 'ULTRA rule: negation from known facts − from memberships',
     ('ultraquery', 'negation', '<', -.5, 'model negation ahead by at least 0.5 on UQ-23; on +H the gap is smaller',
      '8 Oct 2026, 21:27')),
    ('Facts off', 'trix-sd-facts-none-vs-sd', 'TRIX rule: facts off − on', None),
    ('Facts off', 'kgicl-sd-facts-none-vs-sd', 'KG-ICL rule: facts off − on', None),
    ('Flock', 'flock-sd-vs-softmax', 'Flock: softmax × degree − softmax (d = 1)',
     ('both', 'all', '>', .5, 'softmax × degree ahead by at least 0.5 on both suites', '8 Oct 2026, 21:28')),
    ('Flock', 'flock-sd-vs-adapter', 'Flock: softmax × degree − adapter',
     ('both', 'all', '=', .5, 'within 0.5 of the adapter on both suites', '8 Oct 2026, 21:28')),
    ('Flock', 'flock-softmax-vs-adapter', 'Flock: softmax (d = 1) − adapter', None),
    ('Flock', 'flock-adapter-vs-identity', 'Flock: adapter − raw scores', None),
)


def outcome(interval: list | None, direction: str, threshold: float) -> str:
    """'>' or '<': the 95% interval against the threshold; '=': equivalence within +-threshold on the 90% interval (TOST)."""
    if not interval:
        return 'pending'
    low, high = interval
    if direction == '=':
        return ('supported' if -threshold < low and high < threshold
                else 'refuted' if high < -threshold or low > threshold else 'inconclusive')
    if direction == '>':
        return 'supported' if low > threshold else 'refuted' if high < threshold else 'inconclusive'
    return 'supported' if high < threshold else 'refuted' if low > threshold else 'inconclusive'


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values (step-down), keyed like the input."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.
    for rank, key in enumerate(order):
        running = max(running, min(1., (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


MATCHED_PAIRS = (('FB15k237LogicalQuery', 'FB15k237+H'), ('NELL995LogicalQuery', 'NELL995+H'))


def graphs_in_direction(side: dict | None, direction: str, threshold: float) -> str:
    """+H, descriptive: how many of the graphs lie in the predicted direction (or within the margin for '=')."""
    if not side or not side.get('per_dataset'):
        return 'pending'
    values = list(side['per_dataset'].values())
    hits = sum((v > threshold) if direction == '>' else (v < threshold) if direction == '<' else (abs(v) < threshold)
               for v in values)
    return f'{hits}/{len(values)} graphs {"within the margin" if direction == "=" else "in the predicted direction"}'


def interaction_cells(reports: 'tables.Reports', name: str) -> tuple[str, str, str]:
    """(UQ-23 cell, +H cell, outcome) of the registered generator-bias interaction, judged on the matched pairs.

    Registered (PROTOCOL 20:36, operationalised before any hardness-balanced result): per pair, (HB - standard on the +H
    graph) minus (HB - standard on its UQ-23 original), FB15k-237 and NELL995, averaged over the two pairs. Each part has a
    query-level 95% interval conditional on its graph; the parts' variances add in a normal approximation, which ignores
    the five adapter seeds both share. The whole-suite difference is shown for reference only.
    """
    uq = family(contrast(reports, name, 'ultraquery'))
    side = family(contrast(reports, name, 'plus_h'))
    if not uq or not side:
        return tables.MISSING, tables.MISSING, 'pending'
    pairs, variances = [], []
    for original, plus_h in MATCHED_PAIRS:
        a, b = (side.get('per_dataset_intervals') or {}).get(plus_h), (uq.get('per_dataset_intervals') or {}).get(original)
        if plus_h in side['per_dataset'] and original in uq['per_dataset'] and a and b:
            pairs.append((tables.dataset_name(plus_h).replace('+H', ''), side['per_dataset'][plus_h] - uq['per_dataset'][original]))
            variances.append(((a['ci95'][1] - a['ci95'][0]) / 3.92) ** 2 + ((b['ci95'][1] - b['ci95'][0]) / 3.92) ** 2)
    whole = f'whole suites {side["a_minus_b"] - uq["a_minus_b"]:+.2f} (reference)'
    if len(pairs) < len(MATCHED_PAIRS):
        return tables.MISSING, whole, 'pending'
    value = sum(v for _, v in pairs) / len(pairs)
    se = sum(variances) ** .5 / len(pairs)
    interval = [value - 1.96 * se, value + 1.96 * se]
    cell = f'{value:+.2f} [{interval[0]:+.2f}, {interval[1]:+.2f}] (' + ', '.join(f'{d} {v:+.2f}' for d, v in pairs) + ')'
    return cell, whole, outcome(interval, '>', 0) + ' (matched pairs, approximate interval)'


def give_back_outcome(reports: 'tables.Reports', side: dict | None) -> str:
    """KG-ICL +H, registered as 'gives back part of the advantage over the adapter': below softmax x degree, still above
    the adapter. Descriptive, per graph."""
    versus_adapter = family(contrast(reports, 'kgicl-softmax-vs-adapter', 'plus_h'))
    if not side or not versus_adapter:
        return 'pending'
    below_rule = sum(v < 0 for v in side['per_dataset'].values())
    below_adapter = sum(v < 0 for v in versus_adapter['per_dataset'].values())
    n = len(side['per_dataset'])
    verdict = 'gives back all of it and more' if below_adapter == n else 'gives back part' if below_rule and not below_adapter else 'mixed'
    return f'+H: below softmax × degree on {below_rule}/{n} graphs, below the adapter on {below_adapter}/{n}: {verdict}'


def subset_rows(reports: 'tables.Reports') -> tuple[list[list[str]], float | None]:
    """Rows of the subset-contrast table and the largest change of a difference under expected ties.

    Holm adjusts the exact sign tests of the registered directional UQ-23 predictions against zero, a family that counts
    pending tests as p = 1; predictions with a nonzero threshold or of equivalence are judged by their interval only.
    """
    registered = {}
    for question, name, words, prediction in SUBSET_ROWS:
        if prediction and prediction[0] in ('ultraquery', 'both') and prediction[2] in ('<', '>') and prediction[3] == 0:
            row = family(contrast(reports, name, 'ultraquery'), 'all', prediction[1])
            registered[name] = row['sign_p'] if row and row.get('sign_p') is not None else 1.
    adjusted = holm(registered)
    rows, shifts = [], []
    for question, name, words, prediction in SUBSET_ROWS:
        predicted = f'{prediction[4]} ({prediction[5]})' if prediction else 'none (descriptive)'
        if prediction and prediction[0] == 'interaction':
            uq_cell, plus_h_cell, result = interaction_cells(reports, 'kgicl-hb-vs-adapter')
            rows.append([tables.escape(question), tables.escape(words), tables.escape(predicted), uq_cell, tables.escape(plus_h_cell),
                         tables.escape(result)])
            continue
        uq, plus_h = contrast(reports, name, 'ultraquery'), contrast(reports, name, 'plus_h')
        category = prediction[1] if prediction else 'all'
        main = family(uq, 'all', category)
        if main:
            uq_cell = (f'{main["a_minus_b"]:+.2f} [{main["ci95_clustered"][0]:+.2f}, {main["ci95_clustered"][1]:+.2f}], '
                       f'{main["wins"]}/{main["wins"] + main["losses"]}')
        else:
            uq_cell = tables.MISSING
        side = family(plus_h, 'all', category)
        plus_h_cell = ', '.join(f'{tables.dataset_name(d).replace("+H", "")} {v:+.2f}' for d, v in side['per_dataset'].items()) if side else tables.MISSING
        for item, suite in ((uq, 'ultraquery'), (plus_h, 'plus_h')):
            expected = contrast(reports, name, suite, policy='expected')
            a, b = family(item, 'all', category), family(expected, 'all', category)
            if a and b:
                shifts.append(abs(a['a_minus_b'] - b['a_minus_b']))
        if prediction:
            suite, _, direction, threshold, text, when = prediction
            results = []
            if suite == 'part':
                results.append(give_back_outcome(reports, side))
            if suite in ('ultraquery', 'both'):
                result = outcome((main or {}).get(('ci90' if direction == '=' else 'ci95') + '_clustered'), direction, threshold)
                if name in adjusted and result != 'pending':
                    result += f' (Holm p = {adjusted[name]:.2g})'
                results.append(('UQ-23: ' if suite == 'both' else '') + result)
            if suite == 'both':
                results.append('+H: ' + graphs_in_direction(side, direction, threshold) + ' (descriptive)')
            if category == 'negation' and main and side:
                # Registered alongside: on +H the gap is smaller, the gap being model negation's lead B - A (signed).
                lead_plus_h, lead_uq = -side['a_minus_b'], -main['a_minus_b']
                results.append(f'+H: model negation ahead by {lead_plus_h:+.2f} (UQ-23 {lead_uq:+.2f}), gap '
                               f'{"smaller" if lead_plus_h < lead_uq else "not smaller"} (descriptive)')
            result = '; '.join(results)
        else:
            result = '-'
        rows.append([tables.escape(question), tables.escape(words), tables.escape(predicted), uq_cell, plus_h_cell, tables.escape(result)])
    return rows, max(shifts) if shifts else None


def subset_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    rows, shift = subset_rows(reports)
    note = ('Paired comparisons on uniform samples of 1000 test queries per type (seed 0), each against its own baseline on the '
            'same queries; MRR differences A − B in points, sort ties. UQ-23: mean over datasets with a 95% interval resampling '
            'datasets and queries, and datasets won; +H: per graph (FB15k-237, NELL995, ICEWS18). UQ-23 outcomes compare the '
            'dataset-clustered 95% interval with a directional prediction and the 90% interval with an equivalence prediction; '
            'Holm adjusts the exact sign tests of the registered directional UQ-23 predictions against zero, counting pending '
            'ones. +H outcomes count graphs and are descriptive. The interaction is judged on the matched pairs (FB15k-237 and '
            'NELL995 against their +H versions), with their query-level 95% intervals combined in a normal approximation; '
            'its UQ-23 column holds the pair mean, the +H column the whole-suite difference for reference. A dash: not yet run.')
    if shift is not None:
        note += f' Under expected ties every difference changes by at most {shift:.2f}.'
    check = (reports.review.get('subset_representativeness') or {}).get('suites') or {}
    if all(check.get(suite, {}).get('error_diff') for suite in ('ultraquery', 'plus_h')):
        uq, plus_h = check['ultraquery']['error_diff'], check['plus_h']['error_diff']
        mean = f'{uq["mean"]:+.2f}'.replace('-0.00', '0.00').replace('+0.00', '0.00')
        note += (' Representativeness: on the full-test traces of ULTRA with softmax × degree and with its shipped (seed-0) '
                 f'adapter, restricting to the subset queries changes their difference by at most {uq["max_abs"]:.2f} on a UQ-23 dataset '
                 f'(suite mean {mean}) and {plus_h["max_abs"]:.2f} on a +H graph.')
    return tables.table('Paired Subset Contrasts of the Review', 'tab:subset-contrasts',
                        ['Question', 'Contrast (A − B)', 'Registered prediction', r'UQ-23 $\Delta$ [95\%], wins', r'+H $\Delta$ per graph',
                         'Outcome'],
                        [26, 54, 54, 44, 46, 30], rows, note, layout=layout)


# Appendix: what decides whether the rule works, per backbone (known-tail mass, calibration, cap ties, training loss).
TRAINING_LOSS = {'ULTRA': 'binary cross-entropy on sigmoid scores', 'TRIX': 'binary cross-entropy on sigmoid scores',
                 'KG-ICL': 'softmax cross-entropy over all entities', 'Flock': '-'}


def mechanism_table_rows(reports: 'tables.Reports') -> list[list[str]]:
    review = reports.review
    rows = []
    onep = {(r['suite'], r['backbone'], r['calibration']): r for r in ((review.get('onep_ties') or {}).get('rows') or [])}
    for key in ('ultra', 'trix', 'kgicl', 'flock'):
        name = BACKBONE_NAMES[key]
        mass = review.get('known_tail_mass', {}).get(key) or {}
        medians = sorted(v['known_softmax_mass']['median'] for v in mass.values())
        shares = sorted(v['known_in_top_degree'] for v in mass.values() if v.get('known_in_top_degree') is not None)
        mass_cell = (f'{medians[len(medians) // 2]:.3f} ({medians[0]:.3f}–{medians[-1]:.3f})' if medians else tables.MISSING)
        share_cell = f'{shares[len(shares) // 2]:.2f}' if shares else tables.MISSING
        top = review.get('top_k', {}).get(key) or {}
        uq = [v for d, v in top.items() if d not in tables.PLUS_H_DATASETS]
        ece = []
        for calibration in ('identity', 'adapter', 'softmax-degree'):
            values = [v['calibration_top_k']['10'][calibration]['ece'] for v in uq if calibration in v['calibration_top_k']['10']]
            ece.append(f'{sum(values) / len(values):.3f}' if values else tables.MISSING)
        cells = [f'{prefix}{ties["graphs_below_identity"]}/{ties["graphs"]}'
                 + (f', {ties["largest_drop"]:+.2f}' if ties['graphs_below_identity'] else '')
                 for prefix, ties in (('', onep.get(('ultraquery', key, 'softmax-degree'))),
                                      ('+H ', onep.get(('plus_h', key, 'softmax-degree')))) if ties]
        ties_cell = '; '.join(cells) if cells else tables.MISSING
        gap = family(contrast(reports, 'rule-vs-adapter', 'ultraquery', backbone=key))
        gap_cell = f'{gap["a_minus_b"]:+.2f}' if gap else tables.MISSING
        rows.append([name, mass_cell, share_cell, *ece, ties_cell, gap_cell, tables.escape(TRAINING_LOSS[name])])
    return rows


def mechanism_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    note = ('Known-tail mass: median over graphs (range) of each graph\'s median softmax mass on the observed tails of 300 '
            'validation 1p atoms; share: observed tails among the backbone\'s top d. ECE: expected calibration error on '
            'each atom\'s top 10 non-observed, non-easy candidates, UQ-23 mean. 1p ties: graphs whose 1p MRR under '
            'softmax × degree falls below the raw scores\', and the largest drop (points), on UQ-23 and on +H. Rule − adapter: all-type MRR, '
            'UQ-23. Flock: 128 walks and one prediction per atom, no adapter fitted when measured.')
    return tables.table('What Decides Whether the Rule Works', 'tab:mechanism',
                        ['Backbone', 'Known-tail mass', 'Share in top d', 'ECE raw', 'ECE adapter', 'ECE rule',
                         '1p ties', 'Rule − adapter', 'Training loss'],
                        [20, 34, 22, 18, 20, 18, 40, 24, 50], mechanism_table_rows(reports), note, numeric_from=1, layout=layout)


# Appendix: +H strata (full | partial by setting) and the exact accounting of the facts' effect on intersections.
STRATA_LABELS = {'identity': 'Raw scores', 'adapter (5 seeds)': 'Adapter (5 seeds)', 'softmax-degree': 'Softmax × degree',
                 'min-max': 'Min-max', 'identity, facts off': 'Raw scores, facts off', 'adapter, facts off': 'Adapter, facts off',
                 'softmax-degree, facts off': 'Softmax × degree, facts off'}
def strata_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    strata = reports.review.get('strata') or {}
    rows = []
    for key in ('ultra', 'trix', 'kgicl'):
        for method, cells in (strata.get(key) or {}).items():
            values = []
            for dataset in tables.PLUS_H_DATASETS:
                cell = cells.get(dataset)
                values.extend([tables.MISSING, tables.MISSING] if not cell else
                              [f'{cell["full"]:.2f}' if cell.get('full') is not None else tables.MISSING,
                               f'{cell["partial"]:.2f}' if cell.get('partial') is not None else tables.MISSING])
            rows.append([BACKBONE_NAMES[key], tables.escape(STRATA_LABELS.get(method, method)), *values])
    note = ('MRR on +H answers whose cheapest proof misses every positive edge (full) or some (partial), macro over the '
            'multi-edge query types that have both strata; corrected filters, train+valid reference.')
    spanners = [('', 2)] + [(tables.escape(tables.dataset_name(d)), 2) for d in tables.PLUS_H_DATASETS]
    return tables.table('+H Answer Strata by Setting', 'tab:facts-strata',
                        ['Backbone', 'Setting', *['Full', 'Partial'] * len(tables.PLUS_H_DATASETS)],
                        [18, 50, 20, 20, 20, 20, 20, 20], rows, note, spanners=spanners, numeric_from=2, layout=layout)


def accounting_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    accounting = (reports.review.get('accounting') or {}).get('rows') or []
    rows = []
    by = defaultdict(dict)
    for row in accounting:
        by[row['backbone'], row['dataset']][row['shape']] = row
    for (key, dataset), shapes in sorted(by.items(), key=lambda kv: (('ultra', 'trix', 'kgicl').index(kv[0][0]),
                                                                      list(tables.PLUS_H_DATASETS).index(kv[0][1]))):
        cells = []
        for shape in ('2i', '3i', '4i'):
            row = shapes.get(shape)
            cells.append(tables.MISSING if not row else f'{row["overtakers_mean"]:.1f} ({row["overtakers_median"]:.0f})')
        fell = min((s['full_not_improved'] for s in shapes.values()), default=None)
        improved = [s['with_known_edge_improved'] for s in shapes.values()]
        rows.append([BACKBONE_NAMES[key], tables.escape(tables.dataset_name(dataset)), *cells,
                     tables.MISSING if fell is None else rf'{100 * fell:.2f}\%',
                     rf'{100 * min(improved):.0f}–{100 * max(improved):.0f}\%' if improved else tables.MISSING])
    note = ('Raw scores, known facts on against off, corrected filters. New overtakers: non-answers newly scored strictly '
            'above an answer whose every edge is missing, mean (median) per answer. Such an answer keeps its score in '
            'an intersection, so its count can only rise; the column checks this on every answer and is a consistency '
            'check of the implementation, not evidence. Answers with a known edge: share whose count fell. Some '
            'overtakers may be true answers missing from the reference graph.')
    return tables.table('Known Facts on +H Intersections: Exact Accounting', 'tab:facts-accounting',
                        ['Backbone', 'Graph', '2i overtakers', '3i overtakers', '4i overtakers', 'Full: count did not fall',
                         'Known edge: count fell'], [18, 30, 28, 28, 28, 30, 30], rows, note, numeric_from=2, layout=layout)


def facts_by_dataset_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    """Per dataset: all-type MRR with known facts on and off, for raw scores, the rule and the adapter."""
    systems = summary.collect(reports)
    primary = summary.primary_recipes(reports, systems)
    rows = []
    for method in sorted({m for m, _ in primary}, key=lambda m: summary.system_order(m, False)):
        backbone = method.split('-')[0]
        for suite, _ in SUITES:
            recipe = primary.get((method, suite))
            settings = []
            for identity, token in ((True, recipe), (False, f'{backbone}-{summary.RULE_TOKEN}'), (False, recipe)):
                on = systems.get((method, identity, token), {})
                off = {}
                for candidate in (f'{token}-facts-none', f'{token}-facts-none-full'):
                    off = off or systems.get((method, identity, candidate), {})
                settings.append((on, off))
            for dataset in summary.suite_datasets(suite):
                cells = []
                for on, off in settings:
                    for runs in (on, off):
                        values = [summary.category_mean(r, policy, 'all') for r in runs.get(dataset, {}).values()
                                  if r['complete']]
                        values = [v for v in values if v is not None]
                        cells.append(tables.number(sum(values) / len(values)) if values else tables.MISSING)
                if any(c != tables.MISSING for c in cells):
                    rows.append([BACKBONE_NAMES.get(backbone, backbone), tables.escape(tables.dataset_name(dataset)), *cells])
    note = 'All-type MRR, full test splits; adapter: mean over its seeds. Facts off: the override disabled at test time.'
    spanners = [('', 2), ('Raw scores', 2), ('Softmax × degree', 2), ('Adapter', 2)]
    return tables.table('Known Facts On and Off per Dataset', 'tab:facts-by-dataset',
                        ['Backbone', 'Dataset', *['On', 'Off'] * 3], [18, 40, 18, 18, 18, 18, 18, 18], rows, note,
                        spanners=spanners, numeric_from=2, row_group=lambda row: row[0], layout=layout)


# Appendix: dated deviations from the registered protocol (Experiments/screens/PROTOCOL.md), with their direction.
DEVIATIONS = (
    ('5 Oct 2026', 'Beam 256 was not adopted after its validation run (+0.39 ULTRA, +0.49 TRIX on UQ-23 validation); the '
     'gain is reported only.', 'Main results use beam 64.'),
    ('8 Oct 2026, 18:21', 'Breaking beam ties among known facts by the backbone moved validation MRR by under 0.2 everywhere, '
     'so no test study followed (rule fixed at 17:20).', 'Null result, reported.'),
    ('8 Oct 2026, 20:36', 'The full target-validation bracket (52 fits and tests) was dropped except the three +H targets, '
     'and a temperature fitted on sources was not run.', 'Fewer upper references; budget.'),
    ('8 Oct 2026, 20:36', 'The 14-type adapters are evaluated on paired subsets, not the full test split.',
     'Robustness result only.'),
    ('8 Oct 2026, 20:36', 'The equivalence margin of 0.5 MRR either side of zero was set after the dataset-clustered 95% '
     'intervals were known.', 'Each row of the equivalence figure also states the smallest margin it passes.'),
    ('8 Oct 2026, 20:59', 'Three protocol entries had been labelled with times ahead of the clock (19:00, 21:00, 21:20 '
     'for 18:21, 20:36, 20:38); corrected with a note, contents unchanged.', 'No effect on any result.'),
    ('8 Oct 2026, 22:39', 'A run with known facts below membership one was dropped: for intersections it cannot fail; an '
     'exact accounting replaced it.', 'Mechanism shown by accounting, not by a graded run.'),
    ('8 Oct 2026, 23:14', 'The presentation was made rule-centred after the review results were known: the training-free '
     'rule became a main-table system beside the adapter, and the recipe ablations moved to the appendix.',
     'No result changed; the adapter stays in every table.'),
)


def deviations_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    rows = [[tables.escape(when), tables.escape(what), tables.escape(effect)] for when, what, effect in DEVIATIONS]
    return tables.table('Deviations from the Registered Protocol', 'tab:protocol-deviations', ['Date', 'Deviation', 'Direction'],
                        [30, 160, 70], rows, 'From the protocol log of the experiments; times are CEST.', layout=layout)


APPENDIX_TABLES = (('appendix-subset-contrasts', subset_table), ('appendix-mechanism', mechanism_table),
                   ('appendix-facts-strata', strata_table), ('appendix-facts-accounting', accounting_table),
                   ('appendix-facts-by-dataset', facts_by_dataset_table), ('appendix-deviations', deviations_table))
