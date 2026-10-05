"""Run selection and the compact main-paper tables, with seed aggregation.

A system is one method with one recipe. Entry IDs that differ only by their
dataset and an optional ``-seedN`` tag are seed replicates of one system;
``-graph-*`` and ``-released-filters`` suffixes name evaluation conditions,
not recipes. One selection
serves the main tables, the appendix and the figures: adapters use their
primary recipe per suite (requested, else shipped, else the one with most
seeds), other methods their default recipe, and every run its primary
evaluation condition.

Main tables average each seed over query types, then datasets, and report the
mean and sample standard deviation over the seeds covering a cell. Nothing is
imputed: a seed missing any dataset or query type of a cell is left out of that
cell, and a cell no seed covers is "-". Main tables show one decimal; the
appendix keeps two.
"""

import copy
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from . import tables

Record = dict[str, Any]
System = tuple[str, bool, str]
Runs = dict[str, dict[int, Record]]

SEED = re.compile(r'-seed(\d+)(?=-|$)')
CONDITION_SUFFIXES = ('-graph-train-valid', '-graph-train', '-released-filters')
BACKBONES = ('ultra', 'trix')
FAMILY_COLUMNS = (('transductive', 'Transductive'), ('inductive-e', r'Inductive ($e$)'),
                  ('inductive-er', r'Inductive ($e,r$)'), ('all', 'All'))
TRAINED_ON_TARGET = ('cone', 'clmpt', 'cqd', 'cqd-hybrid', 'gnnqe', 'qto', 'inductive-gnnqe', 'incoming-relation')
# Targets built from Freebase, whose graph UltraQuery's training queries, the
# backbones' pretraining and the adapters' source queries also come from.
FREEBASE_DERIVED = ('FB15kLogicalQuery', 'FB15k237LogicalQuery', 'FB15k237+H')
# Registered ablations, planned in empty templates; supplied runs replace them.
PLANNED_ABLATIONS = (('ultra', 'no-fb'), ('ultra', 'ultra_4g'), ('ultra', 'ultra_50g'), ('ultra', 'beam256'),
                     ('trix', 'no-fb'), ('trix', 'beam256'))
# Known recipe tokens in reading order: training data, adapter form, search, backbone.
ABLATION_NAMES = {'no-fb': 'Adapter fit without FB15k-237', 'types14': 'Adapter trained on 14 query types',
                  'balanced': 'Hardness-balanced training', 'target-fit': 'Adapter fit on the target graph',
                  'routed': 'Adapters routed by query type', 'beam128': 'Beam 128', 'beam256': 'Beam 256',
                  'ultra_4g': r'ULTRA 4g backbone$^\dagger$', 'ultra_50g': r'ULTRA 50g backbone$^\ddagger$'}
RECIPE_DESCRIPTIONS = {'product-intersections': 'adapter trained on 2i/3i queries',
                       'product-14types': 'adapter trained on 14 query types', 'product-14type': 'adapter trained on 14 query types'}
ABLATION_NOTES = {'ultra_4g': r'$^\dagger$Pretraining adds NELL995, a target dataset.',
                  'ultra_50g': r'$^\ddagger$Pretraining on 50 graphs, which may include target graphs.'}
# The baseline specific to each target graph per UltraQuery family. When all three are present, the main table
# shows them as one row, as Galkin et al. (2024) do; the appendix keeps each method's own rows.
PER_GRAPH_BASELINES = (('transductive', 'qto'), ('inductive-e', 'inductive-gnnqe'), ('inductive-er', 'incoming-relation'))
PER_GRAPH_NOTE = (r'$^\dagger$QTO on the transductive datasets, GNN-QE on the inductive ($e$) splits and the untrained '
                  r'incoming-relation heuristic on the inductive ($e,r$) graphs. All averages them over the 23 datasets, '
                  r'as in {cite}. Per-method results are in the appendix.')


def freebase_derived(dataset: str) -> bool:
    """FB15k, FB15k-237, the inductive FB15k-237 splits and FB15k-237+H."""
    return dataset in FREEBASE_DERIVED or dataset.startswith('InductiveFB15k237Query')


def fmt(value: float, *, signed: bool = False) -> str:
    """One decimal of a fractional metric x100, validated like the appendix tables."""
    tables.number(value, signed=signed)
    if signed and round(100 * value, 1) == 0:
        return '0.0'
    return f'{100 * value:+.1f}' if signed else f'{100 * value:.1f}'


def identity_run(record: Record) -> bool:
    """A no-adapter control: paired with a learned run or identity-calibrated."""
    return bool(record.get('paired_with') or record.get('calibration') == 'without-adapter'
                or '-without-adapter' in (record.get('id') or ''))


def system_of(record: Record) -> tuple[System, int]:
    """((method, identity, recipe), seed tag) of one result; recipes exclude dataset, seed and condition."""
    text = record['id'].replace(record['dataset'], '').replace('-without-adapter', '')
    for suffix in CONDITION_SUFFIXES:
        text = text.replace(suffix, '')
    match = SEED.search(text)
    recipe = re.sub('-+', '-', SEED.sub('', text)).strip('-') or record['method']
    return (record['method'], identity_run(record), recipe), int(match.group(1)) if match else 0


def recipe_token(record: Record) -> str:
    """The recipe without its backbone (adapters) or method prefix; 'default' for a method's own recipe."""
    (method, _, recipe), _ = system_of(record)
    prefix = method.split('-')[0] if method.endswith('-adapter') else method
    token = recipe[len(prefix) + 1:] if recipe.startswith(prefix + '-') else '' if recipe == prefix else recipe
    return token or 'default'


def condition_priority(record: Record) -> tuple:
    """The primary evaluation condition: target filters and graph, bounded CQD, full tests first."""
    target_filter = 'corrected' if record['dataset'] in tables.PLUS_H_DATASETS else 'released'
    target_graph = 'train' if record['method'] in tables.GRAPH_INDEPENDENT_METHODS else 'train+valid'
    return (record['filter'] not in (None, target_filter),
            record['dataset'] in tables.PLUS_H_DATASETS and record['graph'] not in (None, target_graph),
            record['profile'] == 'reference' and record['method'] in ('cqd', 'cqd-hybrid'),
            not record['complete'], record['id'])


def collect(reports: 'tables.Reports') -> dict[System, Runs]:
    """{system: {dataset: {seed: record}}}, one primary condition per seed and dataset."""
    systems: dict[System, Runs] = defaultdict(lambda: defaultdict(dict))
    for record in reports.results.values():
        if record['id'].endswith('-released-filters'):
            continue
        system, seed = system_of(record)
        previous = systems[system][record['dataset']].get(seed)
        if previous is None or condition_priority(record) < condition_priority(previous):
            systems[system][record['dataset']][seed] = record
    return systems


def suite_datasets(suite: str) -> list[str]:
    """Datasets of 'ultraquery' (23) or 'plus_h' (3), in catalog order."""
    if suite == 'plus_h':
        return list(tables.PLUS_H_DATASETS)
    return [d for d in tables.catalog().BENCHMARK_DATASETS if tables.family(d)]


def family_datasets(family: str) -> list[str]:
    """UltraQuery datasets of one family, or all of them for 'all'."""
    return [d for d in suite_datasets('ultraquery') if family == 'all' or tables.family(d) == family]


def primary_recipes(reports: 'tables.Reports', systems: dict[System, Runs]) -> dict[tuple[str, str], str]:
    """{(adapter method, suite): main recipe}: requested by name, else shipped, else most seeds."""
    requested = tuple(getattr(reports, 'primary_recipes', ()) or ())
    chosen = {}
    for method in sorted({m for (m, identity, _) in systems if m.endswith('-adapter') and not identity}):
        backbone = method.split('-')[0]
        for suite in ('ultraquery', 'plus_h'):
            datasets = set(suite_datasets(suite))
            recipes = [r for (m, identity, r), runs in systems.items()
                       if m == method and not identity and datasets & set(runs)]
            if not recipes:
                continue
            named = [r for r in recipes if any(r in (name, f'{backbone}-{name}') for name in requested)]
            if requested and not named:
                raise ValueError(f'No {method} runs on {suite} for the primary recipe(s) {", ".join(requested)}')
            shipped = [r for r in recipes if r == tables.shipped_recipes().get((suite, method))]
            seeds = {r: max(len(runs) for runs in systems[method, False, r].values()) for r in recipes}
            chosen[method, suite] = min(named or shipped or recipes, key=lambda r: (-seeds[r], len(r), r))
    return chosen


def default_recipe(method: str, recipes: Iterable[str]) -> str:
    """A baseline's default recipe: the one named after the method, else the shortest."""
    recipes = sorted(recipes)
    return method if method in recipes else min(recipes, key=lambda r: (len(r), r))


def selected_systems(reports: 'tables.Reports', suite: str) -> dict[System, Runs]:
    """Systems with runs in the suite: each method's default or primary recipe.

    No-adapter controls keep only their lowest seed: identity calibration takes
    nothing from the adapter's training seed, so further seeds repeat its scores.
    """
    systems = collect(reports)
    primary = primary_recipes(reports, systems)
    datasets = set(suite_datasets(suite))
    present = {key: runs for key, runs in systems.items() if datasets & set(runs)}
    defaults = {}
    for method, identity, recipe in present:
        if not method.endswith('-adapter'):
            defaults.setdefault((method, identity), []).append(recipe)
    selected = {}
    for key, runs in present.items():
        method, identity, recipe = key
        wanted = (primary.get((method, suite)) if method.endswith('-adapter')
                  else default_recipe(method, defaults[method, identity]))
        if recipe != wanted:
            continue
        selected[key] = {d: {min(seeds): seeds[min(seeds)]} for d, seeds in runs.items()} if identity else runs
    return selected


def primary_runs(reports: 'tables.Reports', *, identities: bool = False) -> list[Record]:
    """One run per (dataset, method, identity): the selected recipe's lowest seed in its primary condition."""
    output = []
    for suite in ('ultraquery', 'plus_h'):
        for (method, identity, _), runs in selected_systems(reports, suite).items():
            if identity and not identities:
                continue
            output.extend(seeds[min(seeds)] for dataset, seeds in runs.items() if dataset in suite_datasets(suite))
    # Datasets outside both suites keep their own lowest-seed, primary-condition run.
    known = set(suite_datasets('ultraquery')) | set(suite_datasets('plus_h'))
    others: dict[tuple, Record] = {}
    for (method, identity, _), runs in collect(reports).items():
        if identity and not identities:
            continue
        for dataset, seeds in runs.items():
            if dataset in known:
                continue
            record = seeds[min(seeds)]
            key = (dataset, method, identity)
            if key not in others or condition_priority(record) < condition_priority(others[key]):
                others[key] = record
    return output + list(others.values())


def lowest_seed_reports(reports: 'tables.Reports') -> 'tables.Reports':
    """The reports without seed replicates, for appendix tables that list every run.

    Of entry IDs that differ only by ``-seedN``, the lowest tag stays (untagged
    is 0), with its difficulty rows; effects that name a dropped run go too.
    """
    entries = set(reports.results) | {row['entry'] for row in reports.difficulty if row.get('entry')}
    kept: dict[str, tuple[int, str]] = {}
    for entry in entries:
        match = SEED.search(entry)
        candidate = (int(match.group(1)) if match else 0, entry)
        key = SEED.sub('', entry)
        kept[key] = min(kept.get(key, candidate), candidate)
    dropped = entries - {entry for _, entry in kept.values()}
    if not dropped:
        return reports
    subset = copy.copy(reports)
    subset.results = {entry: record for entry, record in reports.results.items() if entry not in dropped}
    subset.difficulty = [row for row in reports.difficulty if row.get('entry') not in dropped]
    subset.effects = {kind: [effect for effect in effects if not dropped & {v for v in effect.values() if isinstance(v, str)}]
                      for kind, effects in reports.effects.items()}
    return subset


def corrected_filter_reports(reports: 'tables.Reports') -> 'tables.Reports':
    """+H runs under corrected answer filters wherever a run has them; their released-filter twins are set aside.

    So are the paired filter effects, which the shared settings still use to state the size of the difference.
    """
    def key(record: Record) -> tuple:
        system, seed = system_of(record)
        return system, seed, record['dataset'], record.get('graph')

    corrected = {key(r) for r in reports.results.values() if r['dataset'] in tables.PLUS_H_DATASETS and r.get('filter') == 'corrected'}
    dropped = {entry for entry, r in reports.results.items() if r['dataset'] in tables.PLUS_H_DATASETS
               and r.get('filter') == 'released' and key(r) in corrected}
    if not dropped:
        return reports
    subset = copy.copy(reports)
    subset.results = {entry: r for entry, r in reports.results.items() if entry not in dropped}
    subset.difficulty = [row for row in reports.difficulty if row.get('entry') not in dropped]
    subset.effects = {kind: [effect for effect in effects if not dropped & {v for v in effect.values() if isinstance(v, str)}]
                      for kind, effects in reports.effects.items()}
    subset.set_aside_filter_effects = [e for e in reports.effects.get('filter', []) if e not in subset.effects['filter']]
    return subset


def system_order(method: str, identity: bool) -> tuple:
    """Baselines trained on the target first, then transferred baselines, then ULTRA and TRIX."""
    if method.endswith('-adapter'):
        backbone = method.split('-')[0]
        return (2, BACKBONES.index(backbone) if backbone in BACKBONES else len(BACKBONES), backbone, not identity)
    return (0 if method in TRAINED_ON_TARGET else 1, tables.method_order(tables.METHOD_NAMES.get(method, method)), method, False)


def display_name(method: str, identity: bool) -> str:
    """Row label of a system in the main tables."""
    name = tables.METHOD_NAMES.get(method, method)
    if method.endswith('-adapter'):
        return name.replace(' + adapter', '') + (' (no adapter)' if identity else ' + adapter (ours)')
    return name


def category_types(dataset: str, category: str) -> list[str]:
    """Query types of 'all', 'epfo' or 'negation' for the dataset's suite."""
    types = tables.PLUS_H_TYPES if dataset in tables.PLUS_H_DATASETS else tables.ULTRA_TYPES
    return [t for t in types if category == 'all' or ('n' in t) == (category == 'negation')]


def category_mean(record: Record, policy: str, category: str) -> float | None:
    """Equal query-type mean over every type of the category, or None if any is missing."""
    shapes = tables.policy_values(record, policy)['per_shape']
    values = [shapes.get(t, {}).get('mrr') for t in category_types(record['dataset'], category)]
    return None if any(tables.is_missing(v) for v in values) else sum(values) / len(values)


def seed_scores(by_dataset: Runs, datasets: Sequence[str], policy: str, category: str) -> dict[int, float]:
    """{seed: mean over datasets} for every seed covering all datasets and query types."""
    if not datasets:
        return {}
    seeds = set.intersection(*(set(by_dataset.get(d, {})) for d in datasets))
    output = {}
    for seed in sorted(seeds):
        values = [category_mean(by_dataset[d][seed], policy, category) for d in datasets]
        if all(v is not None for v in values):
            output[seed] = sum(values) / len(values)
    return output


def seed_means(by_dataset: Runs, datasets: Sequence[str], policy: str, category: str) -> list[float]:
    """Per seed covering every dataset (and query type): the mean over datasets."""
    return list(seed_scores(by_dataset, datasets, policy, category).values())


# Previews (--preview) show cells of planned runs without results with these placeholders, sized like real numbers.
PLACEHOLDER, SEED_PLACEHOLDER, DELTA_PLACEHOLDER = 'xx.x', r'xx.x$_{\pm x.x}$', '+x.x'


def planned_cell(by_dataset: Runs, datasets: Sequence[str]) -> bool:
    """Whether a system has a run planned for every dataset of a cell and at least one still awaits results."""
    return bool(datasets) and all(d in by_dataset for d in datasets) and any(
        record.get('raw', {}).get('planned') for d in datasets for record in by_dataset[d].values())


def placeholder(row: dict) -> str:
    """The placeholder of a pending main-table cell: adapter rows also reserve their seed spread."""
    return SEED_PLACEHOLDER if row['method'].endswith('-adapter') and not row['identity'] else PLACEHOLDER


def mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation (None for fewer than two values)."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    return mean, (math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else None)


def ranks(column: Sequence[Sequence[float]]) -> list[int | None]:
    """1 for the best and 2 for the second-best displayed mean; displayed ties share a rank."""
    shown = [round(100 * mean_sd(v)[0], 1) if v else None for v in column]
    levels = sorted({v for v in shown if v is not None}, reverse=True)[:2]
    return [None if v is None else levels.index(v) + 1 if v in levels else None for v in shown]


def cell(values: Sequence[float], *, rank: int | None = None, signed: bool = False) -> str:
    """Mean with a seed standard deviation subscript; best bold, second underlined."""
    mean, sd = mean_sd(values)
    if mean is None:
        return tables.MISSING
    text = fmt(mean, signed=signed)
    text = rf'\textbf{{{text}}}' if rank == 1 else rf'\underline{{{text}}}' if rank == 2 else text
    return text + (rf'$_{{\pm {100 * sd:.1f}}}$' if sd is not None else '')


def appendix_cell(values: Sequence[float]) -> str:
    """Two decimals and the seed standard deviation, for appendix long tables."""
    mean, sd = mean_sd(values)
    if mean is None:
        return tables.MISSING
    return tables.number(mean) + (rf' $\pm$ {100 * sd:.2f}' if sd is not None else '')


PREVIEW_NOTE = 'Preview: xx.x marks a planned result that is not available yet.'


def compact_table(caption: str, label: str, columns: str, header_rows: Sequence[tuple[list[str], list[tuple[int, int]]]],
                  rows: Sequence[tuple[str, list[str]]], *, star: bool = True, notes: str = '',
                  layout: str = 'paper') -> str:
    """A booktabs float; header rows are (cells, rules); row groups get space.

    Paper layout: the caption and notes take the table's own width (threeparttable), not the page's.
    Thesis layout (researchreport.sty): a single-column float whose body sits in a fitblock, which the
    thesis design sets to fit the reading column (a smaller size and tighter columns when needed); the
    caption keeps the page's caption column. Notes explain preview placeholders when cells have them.
    """
    if layout == 'thesis':
        lines = [r'\begin{table}[tbp]', rf'\caption{{{caption}}}\label{{{label}}}', r'\begin{fitblock}',
                 r'\begin{threeparttable}', rf'\begin{{tabular}}{{{columns}}}', r'\toprule']
        if any(PLACEHOLDER in cell or DELTA_PLACEHOLDER in cell for _, cells in rows for cell in cells):
            notes = (notes + ' ' + PREVIEW_NOTE).strip()
    else:
        environment = 'table*' if star else 'table'
        lines = [rf'\begin{{{environment}}}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
                 r'\begin{threeparttable}', rf'\caption{{{caption}}}\label{{{label}}}', rf'\begin{{tabular}}{{{columns}}}',
                 r'\toprule']
    for cells, rules in header_rows:
        lines.append(' & '.join(cells) + r' \\')
        if rules:
            lines.append(''.join(rf'\cmidrule(lr){{{a}-{b}}}' for a, b in rules))
    lines.append(r'\midrule')
    previous = None
    for group, cells in rows:
        if len(cells) != len(columns):
            raise ValueError(f'Wrong number of cells in {label}')
        if previous is not None and group != previous:
            lines.append(r'\addlinespace[3pt]')
        previous = group
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    if layout == 'thesis':
        if notes:
            lines.append(r'\begin{tablenotes}\item[] ' + notes + r'\end{tablenotes}')
        lines.extend([r'\end{threeparttable}', r'\end{fitblock}', r'\end{table}'])
        return '\n'.join(lines)
    if notes:
        lines.append(r'\begin{tablenotes}\scriptsize\item[] ' + notes + r'\end{tablenotes}')
    lines.extend([r'\end{threeparttable}', rf'\end{{{environment}}}'])
    return '\n'.join(lines)


def grouped_header(groups: Sequence[tuple[str, int]], first: int) -> tuple[list[str], list[tuple[int, int]]]:
    """A spanner row: (label, width) groups after `first` label columns, with their rules."""
    cells, rules, start = [''] * first, [], first + 1
    for name, width in groups:
        cells.append(rf'\multicolumn{{{width}}}{{c}}{{{name}}}')
        rules.append((start, start + width - 1))
        start += width
    return cells, rules


def scope_sentence(records: Sequence[Record]) -> str:
    """One short sentence; the appendix lists the scope of every run."""
    if not records:
        return 'No evaluated runs.'
    if all(r['complete'] for r in records):
        return 'Full test splits.'
    splits = sorted({r['split'] or 'unknown' for r in records if not r['complete']})
    return ('Partial evaluation (' + ', '.join(tables.escape(s) for s in splits)
            + ' split, query samples); not the full test splits.')


def seed_sentence(counts: Iterable[int]) -> str:
    """How many adapter seeds the adapter rows average; empty for single runs."""
    counts = sorted({n for n in counts if n})
    if not counts or counts == [1]:
        return ''
    return (r' Adapter rows: mean$_{\pm\mathrm{s.d.}}$ over ' + '/'.join(map(str, counts)) + ' adapter training seeds. '
            r'Baselines and backbones are single evaluations of fixed checkpoints; only the adapter is trained here, '
            r'so only adapter rows vary by seed.')


SELECTION_SENTENCE = (' The adapter recipe was chosen on validation queries of both suites; no choice used test queries.')


def seed_legend(counts: Iterable[int]) -> str:
    """The caption legend for seed spreads; empty when no cell averages several seeds."""
    counts = sorted({n for n in counts if n})
    return '' if not counts or counts == [1] else r' Subscripts: sample s.d. over adapter training seeds.'


def shared_settings(reports: 'tables.Reports', policy: str) -> str:
    """Settings every main table shares, stated once (for the paper's setup section) instead of in each caption."""
    ultraquery, plus_h = ultraquery_rows(reports, policy), plus_h_rows(reports, policy)
    rows = [*ultraquery, *plus_h]
    records = [r for row in rows for r in row['records']]
    sentences = [r'Scores and differences are MRR $\times 100$; counts are unscaled; a dash (-) marks unavailable data.',
                 scope_sentence(records) if records else '', seed_sentence(adapter_seed_counts(rows)).strip(),
                 SELECTION_SENTENCE.strip() if any(row['method'].endswith('-adapter') for row in rows) else '',
                 tie_sentence(records, policy)]
    plus_h_records = [r for row in plus_h for r in row['records']]
    if plus_h_records:
        notes = tables.escape(tables.main_plus_h_protocol_notes(plus_h_records))
        sentences.append('On +H, ' + notes[:1].lower() + notes[1:] if notes else '')
        sentences.append(filter_sentence(reports))
    uses = [*(['UltraQuery training'] if any(row['method'] == 'ultraquery' for row in rows) else []),
            'backbone pretraining', 'the source queries of the adapters']
    shared = ', '.join(uses[:-1]) + ' and ' + uses[-1]
    if any(row['method'] == 'ultraquery' for row in rows):
        sentences.append('UltraQuery is trained on FB15k-237 queries from an ULTRA 4g initialisation, whose pretraining '
                         'includes NELL995; the ULTRA and TRIX backbones (3g pretraining) stay frozen, and the no-adapter rows '
                         'rank their scores without calibration.')
    if ultraquery:
        sentences.append(f'FB15k, FB15k-237 and the nine inductive splits derive from Freebase, which {shared} also use; '
                         'the appendix reports them separately.')
    if plus_h:
        sentences.append(f'FB15k-237+H shares its graph with {shared}.')
    return ' '.join(s for s in sentences if s)


def tie_sentence(records: Sequence[Record], policy: str) -> str:
    """The tie policy and how much the other one changes any shown run's MRR (all query types)."""
    name = ('the expectation over uniformly random tie orders' if policy == 'expected'
            else "the sort order of the reference evaluators")
    differences = []
    for record in records:
        values = [tables.policy_values(record, p)['averages'].get('all', {}).get('mrr') for p in ('sort', 'expected')]
        if not any(tables.is_missing(v) for v in values):
            differences.append(abs(values[1] - values[0]) * 100)
    if not differences:
        return f'Tied scores are ranked by {name}.'
    return (f'Tied scores are ranked by {name}; the other tie policy changes any run\'s MRR by at most '
            f'{max(differences):.2f} points ({sum(differences) / len(differences):.2f} on average).')


def filter_sentence(reports: 'tables.Reports') -> str:
    """How much the released +H answer filters change MRR against the corrected ones, from the paired filter effects."""
    effects = [*reports.effects.get('filter', []), *getattr(reports, 'set_aside_filter_effects', [])]
    deltas = [-100 * e['macro']['sort']['mrr'] for e in effects if not tables.is_missing(e.get('macro', {}).get('sort', {}).get('mrr'))]
    if not deltas:
        return ''
    low, high = round(min(deltas), 2) + 0.0, round(max(deltas), 2) + 0.0
    if low >= 0:
        return f'With the authors\' released filters instead, a run\'s MRR is at most {high:.2f} points higher.'
    return f'The authors\' released filters change a run\'s MRR by {low:+.2f} to {high:+.2f} points.'


def ultraquery_rows(reports: 'tables.Reports', policy: str, columns=FAMILY_COLUMNS, datasets_of=family_datasets) -> list[dict]:
    """One row per selected system with seed values per (column, category)."""
    rows = []
    for (method, identity, _), runs in selected_systems(reports, 'ultraquery').items():
        values = {(f, c): seed_means(runs, datasets_of(f), policy, c) for f, _ in columns for c in ('epfo', 'negation')}
        pending = {(f, c): not values[f, c] and planned_cell(runs, datasets_of(f)) for f, c in values}
        rows.append(dict(method=method, identity=identity, name=display_name(method, identity), values=values, pending=pending,
                         records=[r for d, seeds in runs.items() if tables.family(d) for r in seeds.values()]))
    return sorted(rows, key=lambda row: system_order(row['method'], row['identity']))


def adapter_seed_counts(rows: Sequence[dict]) -> list[int]:
    return [len(v) for row in rows if row['method'].endswith('-adapter') and not row['identity'] for v in row['values'].values()]


def per_graph_row(reports: 'tables.Reports', policy: str) -> dict | None:
    """PER_GRAPH_BASELINES as one main-table row, each family scored by its own baseline; None unless all are present."""
    systems = {method: runs for (method, identity, _), runs in selected_systems(reports, 'ultraquery').items() if not identity}
    if any(method not in systems for _, method in PER_GRAPH_BASELINES):
        return None
    runs = {d: systems[method][d] for family, method in PER_GRAPH_BASELINES for d in family_datasets(family) if d in systems[method]}
    values = {(f, c): seed_means(runs, family_datasets(f), policy, c) for f, _ in FAMILY_COLUMNS for c in ('epfo', 'negation')}
    pending = {(f, c): not values[f, c] and planned_cell(runs, family_datasets(f)) for f, c in values}
    return dict(method='per-graph', identity=False, name='Best per-graph baseline', values=values, pending=pending,
                label=tables.escape('Best per-graph baseline') + r'$^\dagger$', group='trained')


def ultraquery_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    """Main table 1: MRR per method and dataset family, EPFO and negation."""
    rows = ultraquery_rows(reports, policy)
    merged = per_graph_row(reports, policy)
    if merged is not None:
        replaced = {method for _, method in PER_GRAPH_BASELINES}
        rows = [merged, *(row for row in rows if row['method'] not in replaced)]
    keys = [(f, c) for f, _ in FAMILY_COLUMNS for c in ('epfo', 'negation')]
    marks = [ranks([row['values'][key] for row in rows]) for key in keys]
    groups = [(f'{name} ({len(family_datasets(f))})', 2) for f, name in FAMILY_COLUMNS]
    body = [(row.get('group') or ('adapter' if row['method'].endswith('-adapter') else 'trained' if row['method'] in TRAINED_ON_TARGET
                                  else 'transferred'),
             [row.get('label') or tables.escape(row['name']),
              *[placeholder(row) if row['pending'][key] else cell(row['values'][key], rank=marks[i][j]) for i, key in enumerate(keys)]])
            for j, row in enumerate(rows)]
    first = (' The first row combines baselines specific to each target graph.' if merged is not None
             else ' The first group is trained on each target graph.' if any(row['method'] in TRAINED_ON_TARGET for row in rows) else '')
    caption = (r'Complex query answering on the 23 UltraQuery datasets: MRR on the 9 positive (EPFO) and 5 negated '
               r'query types, averaged over query types and then over the datasets of each family.'
               + first + ' Best in bold, second best underlined.' + seed_legend(adapter_seed_counts(rows)))
    notes = PER_GRAPH_NOTE.format(cite=r'\citet{galkin2024foundation}' if layout == 'thesis' else r'Galkin et al.\ (2024)')
    header = [grouped_header(groups, 1), (['Method', *(['EPFO', 'Neg.'] * len(groups))], [])]
    return compact_table(caption, 'tab:main-ultraquery', 'l' + 'c' * 2 * len(groups), header,
                         body or [('', [tables.MISSING] * (1 + 2 * len(groups)))], notes=notes if merged is not None else '',
                         layout=layout)


def plus_h_rows(reports: 'tables.Reports', policy: str) -> list[dict]:
    """One row per selected system with seed values per (dataset or average, category)."""
    rows = []
    for (method, identity, _), runs in selected_systems(reports, 'plus_h').items():
        values, pending = {}, {}
        for dataset in (*tables.PLUS_H_DATASETS, 'average'):
            datasets = list(tables.PLUS_H_DATASETS) if dataset == 'average' else [dataset]
            for category in ('epfo', 'negation'):
                values[dataset, category] = seed_means(runs, datasets, policy, category)
                pending[dataset, category] = not values[dataset, category] and planned_cell(runs, datasets)
        rows.append(dict(method=method, identity=identity, name=display_name(method, identity), values=values, pending=pending,
                         group='trained' if method in TRAINED_ON_TARGET else 'transferred',
                         records=[r for d, seeds in runs.items() if d in tables.PLUS_H_DATASETS for r in seeds.values()]))
    return sorted(rows, key=lambda row: system_order(row['method'], row['identity']))


def plus_h_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    """Main table 2: MRR per method and +H dataset, EPFO and negation, with their average."""
    rows = plus_h_rows(reports, policy)
    keys = [(d, c) for d in (*tables.PLUS_H_DATASETS, 'average') for c in ('epfo', 'negation')]
    marks = [ranks([row['values'][key] for row in rows]) for key in keys]
    groups = [(tables.escape(tables.dataset_name(d)), 2) for d in tables.PLUS_H_DATASETS] + [('Average', 2)]
    body = [(row['group'], [tables.escape(row['name']), *[placeholder(row) if row['pending'][key] else cell(row['values'][key], rank=marks[i][j])
                                                           for i, key in enumerate(keys)]])
            for j, row in enumerate(rows)]
    caption = (r'Complex query answering on +H: MRR on the 11 positive (EPFO) and 5 negated query types, averaged over '
               r'query types.'
               + (' The first group is trained on each target graph.' if any(row['group'] == 'trained' for row in rows) else '')
               + ' Best in bold, second best underlined.' + seed_legend(adapter_seed_counts(rows)))
    header = [grouped_header(groups, 1), (['Method', *(['EPFO', 'Neg.'] * len(groups))], [])]
    return compact_table(caption, 'tab:main-plus-h', 'l' + 'c' * 2 * len(groups), header,
                         body or [('', [tables.MISSING] * (1 + 2 * len(groups)))], layout=layout)


def seed_deltas(reference: Runs, variant: Runs, datasets: Sequence[str], policy: str) -> list[list[float]]:
    """[variant seed values, deltas]: each variant seed minus the same primary seed when every
    variant seed has one, else minus the primary recipe's mean over its seeds."""
    base = seed_scores(reference, datasets, policy, 'all')
    other = seed_scores(variant, datasets, policy, 'all')
    if not base or not other:
        return [list(other.values()), []]
    if set(other) <= set(base):
        return [list(other.values()), [other[s] - base[s] for s in other]]
    mean = sum(base.values()) / len(base)
    return [list(other.values()), [v - mean for v in other.values()]]


def ablation_rows(reports: 'tables.Reports', policy: str) -> tuple[list[dict], dict[tuple[str, str], str]]:
    """The primary recipe of each backbone, then every other recipe against it, per suite."""
    systems = collect(reports)
    primary = primary_recipes(reports, systems)
    rows = []
    for method in sorted({m for m, _ in primary}, key=lambda m: system_order(m, False)):
        backbone = method.split('-')[0]
        name = tables.METHOD_NAMES.get(method, method).replace(' + adapter', '')
        reference_values, reference_pending = [], []
        for suite in ('ultraquery', 'plus_h'):
            reference = systems.get((method, False, primary.get((method, suite))), {})
            reference_values.extend([seed_means(reference, suite_datasets(suite), policy, 'all'), None])
            reference_pending.extend([not reference_values[-2] and planned_cell(reference, suite_datasets(suite)), False])
        rows.append(dict(backbone=name, name='Primary recipe', values=reference_values, token=None, pending=reference_pending))
        mains = {r for (m, _), r in primary.items() if m == method}
        variants = []
        for recipe in sorted(r for (m, identity, r) in systems if m == method and not identity and r not in mains):
            variant = systems[method, False, recipe]
            values, pending = [], []
            for suite in ('ultraquery', 'plus_h'):
                datasets = suite_datasets(suite)
                reference = systems.get((method, False, primary.get((method, suite))), {})
                if not all(d in variant and d in reference for d in datasets):
                    values.extend([[], []])
                else:
                    values.extend(seed_deltas(reference, variant, datasets, policy))
                pending.extend([not values[-2] and planned_cell(variant, datasets)] * 2)
            token = recipe[len(backbone) + 1:] if recipe.startswith(backbone + '-') else recipe
            variants.append(dict(backbone=name, name=ABLATION_NAMES.get(token, tables.escape(token)), values=values, token=token,
                                 pending=pending))
        order = list(ABLATION_NAMES)
        variants.sort(key=lambda row: (order.index(row['token']) if row['token'] in order else len(order), row['token']))
        if not variants and reports.template:
            variants = [dict(backbone=name, name=ABLATION_NAMES[token], values=[[]] * 4, token=token)
                        for b, token in PLANNED_ABLATIONS if b == backbone]
        rows.extend(variants)
    if not primary and reports.template:
        for backbone in BACKBONES:
            name = tables.METHOD_NAMES[backbone + '-adapter'].replace(' + adapter', '')
            rows.append(dict(backbone=name, name='Primary recipe', values=[[], None, [], None], token=None))
            rows.extend(dict(backbone=name, name=ABLATION_NAMES[token], values=[[]] * 4, token=token)
                        for b, token in PLANNED_ABLATIONS if b == backbone)
    return rows, primary


def ablation_table(reports: 'tables.Reports', policy: str, *, layout: str = 'paper') -> str:
    """Main table 3: each recipe variant's MRR and its change from the primary recipe."""
    rows, primary = ablation_rows(reports, policy)
    groups = [('UltraQuery (23)', 2), ('+H (3)', 2)]
    body = []
    for j, row in enumerate(rows):
        pending = row.get('pending') or [False] * len(row['values'])
        values = [r'\textemdash{}' if v is None else (DELTA_PLACEHOLDER if i % 2 else SEED_PLACEHOLDER if row['token'] is None
                                                       else PLACEHOLDER) if pending[i]
                  else cell(v, signed=i % 2 == 1) for i, v in enumerate(row['values'])]
        label = tables.escape(row['backbone']) if j == 0 or rows[j - 1]['backbone'] != row['backbone'] else ''
        body.append((row['backbone'], [label, row['name'], *values]))
    references = primary_description(primary)
    notes = ' '.join(ABLATION_NOTES[row['token']] for row in rows if row.get('token') in ABLATION_NOTES)
    caption = (r'Ablations of the adapter recipe: MRR over all query types and its change ($\Delta$) from the primary '
               r'recipe of the same backbone'
               + (f' ({references})' if references else '') + r'. Each variant seed is compared with the same seed of '
               r'the primary recipe, or with its seed mean if that seed is missing.')
    header = [grouped_header(groups, 2), (['Backbone', 'Variant', *(['MRR', r'$\Delta$'] * len(groups))], [])]
    return compact_table(caption, 'tab:main-ablations', 'll' + 'c' * 2 * len(groups), header,
                         body or [('', [tables.MISSING] * (2 + 2 * len(groups)))], star=False, notes=notes,
                         layout=layout)


def recipe_description(method: str, recipe: str) -> str:
    """A recipe in words when known ('adapter trained on 2i/3i queries'), else its escaped name without the backbone."""
    backbone = method.split('-')[0]
    token = recipe[len(backbone) + 1:] if recipe.startswith(backbone + '-') else recipe
    return RECIPE_DESCRIPTIONS.get(token, tables.escape(token))


def primary_description(primary: dict[tuple[str, str], str]) -> str:
    """'ULTRA and TRIX: <recipe>' or 'ULTRA: <recipe>; TRIX: UltraQuery <recipe>, +H <recipe>' for captions and notes."""
    names: dict[str, dict[str, str]] = defaultdict(dict)
    for (method, suite), recipe in sorted(primary.items(), key=lambda item: (system_order(item[0][0], False), item[0][1] != 'ultraquery')):
        names[tables.METHOD_NAMES.get(method, method).replace(' + adapter', '')][suite] = recipe_description(method, recipe)
    described: dict[str, list[str]] = defaultdict(list)
    for name, recipes in names.items():
        text = (next(iter(recipes.values())) if len(set(recipes.values())) == 1 else
                ', '.join(('UltraQuery' if s == 'ultraquery' else '+H') + ' ' + r for s, r in recipes.items()))
        described[text].append(name)
    return '; '.join(' and '.join(backbones) + ': ' + text for text, backbones in described.items())


def seed_rows(reports: 'tables.Reports', policy: str) -> tuple[list[list[str]], str]:
    """Appendix: every seed tag of each adapter method; each suite uses its primary recipe."""
    systems = collect(reports)
    primary = primary_recipes(reports, systems)
    rows = []
    for method in sorted({m for m, _ in primary}, key=lambda m: system_order(m, False)):
        runs = {suite: systems[method, False, primary[method, suite]] for m, suite in primary if m == method}
        for seed in sorted({s for by_dataset in runs.values() for seeds in by_dataset.values() for s in seeds}):
            cells = []
            for suite in ('ultraquery', 'plus_h'):
                by_dataset = runs.get(suite, {})
                values = [category_mean(by_dataset[d][seed], policy, 'all') if seed in by_dataset.get(d, {}) else None
                          for d in suite_datasets(suite)]
                cells.append(tables.number(sum(values) / len(values)) if all(v is not None for v in values)
                             else tables.MISSING)
            rows.append([tables.escape(display_name(method, False)), str(seed), *cells])
    return rows, primary_description(primary)


def freebase_rows(reports: 'tables.Reports', policy: str) -> list[list[str]]:
    """Appendix: UltraQuery MRR on Freebase-derived datasets and on the others, per system."""
    groups = {'freebase': [d for d in suite_datasets('ultraquery') if freebase_derived(d)],
              'other': [d for d in suite_datasets('ultraquery') if not freebase_derived(d)]}
    rows = ultraquery_rows(reports, policy, columns=(('freebase', ''), ('other', '')), datasets_of=groups.__getitem__)
    return [[tables.escape(row['name']), *[appendix_cell(row['values'][g, c]) for g in groups for c in ('epfo', 'negation')]]
            for row in rows]
