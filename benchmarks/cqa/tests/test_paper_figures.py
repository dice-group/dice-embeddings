"""Saved synthetic reports verify chart mathematics; never benchmark inference."""

import json
import math
import subprocess
import sys

import pytest

from benchmarks.cqa.manifests import REPO
from benchmarks.cqa.paper import figures, tables

CLI = [sys.executable, '-m', 'benchmarks.cqa.paper']


def result(dataset='FB15k237+H', method='ultra-adapter', score=.3, *, complete=True):
    shapes = tables.PLUS_H_TYPES if dataset in tables.PLUS_H_DATASETS else tables.ULTRA_TYPES
    values = {s: dict(mrr=score, hits10=score + .1, queries=10, hard_answers=20) for s in shapes}
    return dict(dataset=dataset, split='test', benchmark_run={'entry': f'{method}-{dataset}'},
                per_shape=values, num_candidates=100,
                candidate_sha256='candidates-' + dataset, context_sha256='context-' + dataset,
                protocol={'tie_policy': 'expected', 'full_split': complete, 'answer_filter': 'corrected' if dataset in tables.PLUS_H_DATASETS else 'released'},
                dataset_metadata={'inference_graph': 'train+valid'},
                inference={'method': method, 'calibration': 'learned'})


def hardness(entry='qto', counts=(5, 5, 10), *, total=20, complete=True):
    return [dict(dataset='FB15k237+H', entry=entry, method='qto', shape='3p',
                 grouping='inferred_positive_edges', label=str(k), hard_answers=count,
                 parent_hard_answers=total, queries=3, parent_queries=4 if complete else 2,
                 available_queries=4, complete_shape=complete,
                 comparison_graph='train+valid', answer_filter='corrected')
            for k, count in enumerate(counts, 1) if count is not None]


def shape_data(reports):
    return next(b for b in figures.composition_data(reports)[0]['bins'] if b['shape'] == '3p')


def pair(reports, dataset='FB15k237+H', *, complete=True):
    learned = result(dataset, score=.3, complete=complete)
    identity = result(dataset, score=.2, complete=complete)
    identity['benchmark_run']['entry'] += '-without-adapter'
    identity['paired_with'] = learned['benchmark_run']['entry']
    identity['inference']['calibration'] = 'without-adapter'
    reports.consume([learned, identity])
    return learned, identity


def test_composition_uses_partitioned_answer_counts_and_does_not_pool_models():
    reports = tables.Reports()
    reports.consume(hardness('qto') + hardness('ultra'))
    data = shape_data(reports)
    assert data['counts'] == [5, 5, 10]
    assert data['total'] == 20 and data['fractions'] == [.25, .25, .5]
    # Bin query counts overlap; their sum must not be used as the denominator.
    assert data['complete'] is True


def test_missing_bins_are_not_normalized_into_a_complete_distribution():
    reports = tables.Reports()
    reports.consume(hardness(counts=(5, None, 10)))
    assert shape_data(reports)['fractions'] is None
    # An absent zero bin is provable only when the known parent total is met.
    reports = tables.Reports()
    reports.consume(hardness(counts=(5, None, 10), total=15))
    assert shape_data(reports)['counts'] == [5, 0, 10]
    assert shape_data(reports)['fractions'] == pytest.approx([1/3, 0, 2/3])


def test_zero_cost_answers_do_not_disappear_into_positive_bin_percentages():
    reports = tables.Reports()
    rows = hardness(counts=(5, 5, 9))
    rows.append({**rows[0], 'label': '0', 'hard_answers': 1})
    reports.consume(rows)
    assert shape_data(reports)['fractions'] is None


def test_full_runs_with_conflicting_label_distributions_are_rejected():
    reports = tables.Reports()
    reports.consume(hardness('qto') + hardness('ultra', counts=(5, 6, 9)))
    with pytest.raises(ValueError, match='disagree about hardness'):
        figures.composition_data(reports)


@pytest.mark.parametrize('count', [-1, 1.5, True])
def test_invalid_answer_counts_are_rejected(count):
    reports = tables.Reports()
    reports.consume(hardness(counts=(count, 5, 10)))
    with pytest.raises(ValueError, match='nonnegative integer'):
        figures.composition_data(reports)


def test_partial_composition_keeps_its_coverage_label():
    reports = tables.Reports()
    reports.consume(hardness(complete=False))
    assert shape_data(reports)['complete'] is False
    assert shape_data(reports)['fractions'] == [.25, .25, .5]


def test_missing_full_cohort_cannot_suppress_a_valid_partial_cohort():
    reports = tables.Reports()
    reports.consume(hardness('qto-full-missing', counts=(None, None, 10)))
    reports.consume(hardness('qto-partial-valid', complete=False))
    assert shape_data(reports)['complete'] is False
    assert shape_data(reports)['fractions'] == [.25, .25, .5]


def test_adapter_effect_direction_scale_and_supplied_ci_are_preserved():
    reports = tables.Reports()
    learned, identity = pair(reports)
    reports.consume(dict(dataset=learned['dataset'], learned=learned['benchmark_run']['entry'],
                         control=identity['benchmark_run']['entry'],
                         macro={'expected': {'mrr': .1, 'mrr_ci95': [.08, .12]}}))
    point = figures.adapter_data(reports, 'expected')[0]
    assert point['value'] == pytest.approx(.1) and point['ci'] == [.08, .12]
    assert point['complete'] is True
    reports.effects['adapter'][0]['macro']['expected']['mrr'] = .4
    with pytest.raises(ValueError, match='Inconsistent adapter'):
        figures.adapter_data(reports, 'expected')


def test_missing_adapter_ci_stays_missing_and_partial_pair_is_marked():
    reports = tables.Reports()
    pair(reports, complete=False)
    point = figures.adapter_data(reports, 'expected')[0]
    assert point['ci'] is None and point['complete'] is False
    assert figures.adapter_data(reports, 'sort')[0]['value'] is None


@pytest.mark.parametrize('field', ['candidate_sha256', 'context_sha256'])
def test_adapter_pair_rejects_conflicting_supplied_domain_identities(field):
    reports = tables.Reports()
    learned, identity = pair(reports)
    reports.results[identity['benchmark_run']['entry']]['raw'][field] = 'different-identity'
    with pytest.raises(ValueError, match=field):
        figures.adapter_data(reports, 'expected')


def test_adapter_pair_marks_unavailable_identity_evidence():
    reports = tables.Reports()
    learned, identity = pair(reports)
    reports.results[identity['benchmark_run']['entry']]['raw'].pop('context_sha256')
    point = figures.adapter_data(reports, 'expected')[0]
    assert point['value'] == pytest.approx(.1) and point['identity_verified'] is False


@pytest.mark.parametrize('field', ['candidate_sha256', 'context_sha256'])
@pytest.mark.parametrize('blank', ['', '  '])
def test_blank_identities_are_unavailable_for_adapter_and_transfer(field, blank):
    reports = tables.Reports()
    learned, identity = pair(reports)
    for raw in (learned, identity):
        reports.results[raw['benchmark_run']['entry']]['raw'][field] = blank
    point = figures.adapter_data(reports, 'expected')[0]
    assert point['value'] == pytest.approx(.1) and point['identity_verified'] is False
    dataset = 'FB15k237LogicalQuery'
    baseline, model = result(dataset, 'ultraquery', .2), result(dataset)
    baseline[field] = model[field] = blank
    reports = tables.Reports()
    reports.consume([baseline, model])
    assert all(p['value'] is None for p in figures.transfer_data(reports, 'expected'))


def test_multiple_effect_only_adapter_recipes_cannot_silently_overwrite_points():
    reports = tables.Reports()
    for recipe in ('intersections', '14type'):
        reports.consume(dict(dataset='FB15k237LogicalQuery',
                             learned=f'ultra-product-{recipe}-FB15k237LogicalQuery',
                             control=f'ultra-product-{recipe}-FB15k237LogicalQuery-without-adapter',
                             macro={'expected': {'mrr': .1}}))
    with pytest.raises(ValueError, match='Ambiguous adapter pairs'):
        figures.adapter_data(reports, 'expected')


def test_transfer_splits_categories_and_weights_types_equally():
    reports = tables.Reports()
    dataset = 'FB15k237LogicalQuery'
    baseline = result(dataset, 'ultraquery', .2)
    model = result(dataset, score=.3)
    model['per_shape']['1p']['mrr'] = .39
    model['per_shape']['1p']['queries'] = 100
    baseline['per_shape']['1p']['queries'] = 100
    reports.consume([baseline, model])
    values = {p['category']: p['value'] for p in figures.transfer_data(reports, 'expected') if p['method'] == 'ULTRA'}
    assert values == pytest.approx({'epfo': .11, 'negation': .1})


def test_missing_one_category_does_not_erase_the_other():
    reports = tables.Reports()
    dataset = 'FB15k237LogicalQuery'
    model = result(dataset)
    model['per_shape']['1p']['mrr'] = None
    reports.consume([result(dataset, 'ultraquery', .2), model])
    values = {p['category']: p['value'] for p in figures.transfer_data(reports, 'expected') if p['method'] == 'ULTRA'}
    assert values['epfo'] is None and values['negation'] == pytest.approx(.1)


@pytest.mark.parametrize('change', ['graph', 'counts', 'candidates', 'candidate_sha256', 'context_sha256'])
def test_transfer_rejects_incomparable_results(change):
    reports = tables.Reports()
    dataset = 'FB15k237LogicalQuery'
    model = result(dataset)
    if change == 'graph':
        model['dataset_metadata']['inference_graph'] = 'train'
    elif change == 'counts':
        model['per_shape']['1p']['hard_answers'] = 19
    elif change == 'candidates':
        model['num_candidates'] = 99
    else:
        model[change] = 'different-identity'
    reports.consume([result(dataset, 'ultraquery', .2), model])
    with pytest.raises(ValueError, match='Transfer comparison changes'):
        figures.transfer_data(reports, 'expected')


def test_transfer_requires_full_evaluation_and_known_candidate_domain():
    dataset = 'FB15k237LogicalQuery'
    for partial in (True, False):
        reports = tables.Reports()
        model = result(dataset, complete=not partial)
        if not partial:
            model.pop('num_candidates')
        reports.consume([result(dataset, 'ultraquery', .2), model])
        assert all(p['value'] is None for p in figures.transfer_data(reports, 'expected'))


def test_missing_candidate_context_identity_is_not_a_verified_transfer_point():
    reports = tables.Reports()
    dataset = 'FB15k237LogicalQuery'
    model = result(dataset)
    model.pop('context_sha256')
    reports.consume([result(dataset, 'ultraquery', .2), model])
    assert all(p['value'] is None for p in figures.transfer_data(reports, 'expected'))


def test_empty_figures_have_full_layouts_and_no_fabricated_values(tmp_path):
    payload = figures.generate_figures(tables.Reports(), tmp_path)
    assert [f['placement'] for f in payload['figures']] == ['Main paper', 'Main paper', 'Appendix', 'Appendix']
    assert not any('composition' in f['name'] for f in payload['figures'])
    family, hardness, transfer, adapter = [f['data'] for f in payload['figures']]
    assert len(family) == 26 * 2 and all(p['value'] is None for p in family)
    planned = {r['method'] for r in tables.empty_reports().results.values() if r['dataset'] in tables.PLUS_H_DATASETS
               and not r['paired_with']}
    assert len(hardness) == 3 * len(planned) * len(figures.PROFILE_TYPES)
    assert all(p['value'] is None for s in hardness for p in s['points'])
    assert len(adapter) == 26 * 2 and all(p['value'] is None and p['ci'] is None for p in adapter)
    assert len(transfer) == 23 * 2 * 2 and all(p['value'] is None for p in transfer)
    assert len(list(tmp_path.glob('*.pdf'))) == 4
    assert len(list(tmp_path.glob('*.svg'))) == len(list(tmp_path.glob('*.png'))) == 4
    assert json.loads((tmp_path / 'figures.json').read_text()) == payload
    latex = (tmp_path / 'figures.tex').read_text()
    assert latex.count(r'\begin{figure}') == 4 and r'\includegraphics[width=\linewidth]{main-01-adapter-gains.pdf}' in latex
    assert '95%' not in latex and r'95\%' in latex


def test_figures_cli_keeps_latex_console_and_file_identical(tmp_path):
    latex = tmp_path / 'tables.tex'
    directory = tmp_path / 'figures'
    run = subprocess.run([*CLI,
                          '--figures', str(directory), '-o', str(latex)], capture_output=True, text=True, cwd=REPO, check=True)
    assert run.stdout == latex.read_text()
    assert 'Main paper:' in run.stderr and 'Appendix:' in run.stderr
    assert (directory / 'figures.json').exists()


def test_optional_composition_is_appendix_and_keeps_missing_counts(tmp_path):
    payload = figures.generate_figures(tables.Reports(), tmp_path, hardness_composition=True)
    composition = payload['figures'][-1]
    assert composition['name'] == figures.COMPOSITION_NAME
    assert composition['placement'] == 'Appendix'
    assert len(composition['data']) == 3
    assert all(b['fractions'] is None for c in composition['data'] for b in c['bins'])
    assert len(list(tmp_path.glob('*.pdf'))) == 5


def test_only_identical_complete_compositions_share_a_panel():
    import matplotlib.pyplot as plt
    reports = tables.Reports()
    rows = [dict(row, dataset=dataset, shape=shape, label='1',
                 hard_answers=20, parent_hard_answers=20)
            for dataset in tables.PLUS_H_DATASETS for shape in tables.PLUS_H_TYPES
            for row in hardness(counts=(20,), total=20)]
    reports.consume(rows)
    data = figures.composition_data(reports)
    fig = figures.composition_plot(plt, data)
    assert len(fig.axes) == 1
    assert all(tables.dataset_name(d) in fig.axes[0].get_title(loc='left') for d in tables.PLUS_H_DATASETS)
    plt.close(fig)
    data[1]['bins'][0]['complete'] = False
    fig = figures.composition_plot(plt, data)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_composition_flag_requires_figures_before_writing(tmp_path):
    output = tmp_path / 'tables.tex'
    run = subprocess.run([*CLI,
                          '--hardness-composition', '-o', str(output)], capture_output=True, text=True, cwd=REPO)
    assert run.returncode == 2 and not output.exists()
    assert '--hardness-composition requires --figures' in run.stderr


def test_figures_cannot_overwrite_their_input_report(tmp_path):
    source = tmp_path / 'figures.json'
    raw = result()
    source.write_text(json.dumps(raw))
    run = subprocess.run([*CLI, str(source),
                          '--figures', str(tmp_path), '-o', str(tmp_path / 'tables.tex')], capture_output=True, text=True, cwd=REPO)
    assert run.returncode == 2 and json.loads(source.read_text()) == raw
    assert 'must not overwrite' in run.stderr


@pytest.mark.parametrize('filename', ['figures.json', figures.COMPOSITION_NAME + '.pdf'])
def test_figure_output_symlink_cannot_overwrite_an_input_report(tmp_path, filename):
    source = tmp_path / 'result.json'
    raw = result()
    source.write_text(json.dumps(raw))
    directory = tmp_path / 'figures'
    directory.mkdir()
    (directory / filename).symlink_to(source)
    run = subprocess.run([*CLI, str(source),
                          '--figures', str(directory), '--hardness-composition',
                          '-o', str(tmp_path / 'tables.tex')], capture_output=True, text=True, cwd=REPO)
    assert run.returncode == 2 and json.loads(source.read_text()) == raw
    assert 'must not overwrite' in run.stderr


def test_family_gains_pair_every_seed_with_the_seed_independent_control():
    reports = tables.Reports()
    for entry, score in (('ultra-r', .3), ('ultra-r-seed1', .4)):
        raw = result('FB15k237+H', 'ultra-adapter', score)
        raw['benchmark_run']['entry'] = entry + '-FB15k237+H'
        reports.consume(raw)
    control = result('FB15k237+H', 'ultra-adapter', .2)
    control['benchmark_run']['entry'] = 'ultra-r-FB15k237+H-without-adapter'
    control['paired_with'] = 'ultra-r-FB15k237+H'
    control['inference']['calibration'] = 'without-adapter'
    reports.consume(control)
    point = next(p for p in figures.family_gain_data(reports, 'expected') if p['dataset'] == 'FB15k237+H' and p['method'] == 'ULTRA')
    assert point['family'] == 'plus_h' and point['seeds'] == 2
    assert abs(point['value'] - .15) < 1e-12
    # The fixture has no sort-order scores: the gain stays missing instead of borrowing another policy.
    missing = next(p for p in figures.family_gain_data(reports, 'sort') if p['dataset'] == 'FB15k237+H' and p['method'] == 'ULTRA')
    assert missing['value'] is None and missing['status'] == 'missing per-type MRR for selected tie policy'


def test_hardness_profiles_use_numeric_bins_of_one_condition_only():
    reports = tables.Reports()
    rows = []
    for graph, filters, offset in (('train+valid', 'corrected', 0), ('train', 'released', .3)):
        for label, score in (('0', .9), ('1', .4), ('2', .3), ('3', .2)):
            rows.append(dict(dataset='FB15k237+H', method='qto', entry='qto-FB15k237+H', shape='3p',
                             grouping='inferred_positive_edges', label=label, comparison_graph=graph,
                             answer_filter=filters, sort=dict(mrr=score + offset)))
    rows.append(dict(dataset='FB15k237+H', method='qto', entry='qto-FB15k237+H', shape='3p', grouping='difficulty',
                     label='partial', comparison_graph='train+valid', answer_filter='corrected', sort=dict(mrr=.7)))
    reports.consume(rows)
    series = figures.hardness_profile_data(reports, 'sort')
    assert len(series) == 1 and series[0]['method'] == 'QTO'
    assert [p['value'] for p in series[0]['points']] == [.4, .3, .2]


def test_family_gains_accept_seed_recipe_hashes_but_reject_other_changes():
    reports = tables.Reports()
    for entry, score, digest in (('ultra-r', .3, 'a'), ('ultra-r-seed1', .4, 'b')):
        raw = result('FB15k237+H', 'ultra-adapter', score)
        raw['benchmark_run']['entry'] = entry + '-FB15k237+H'
        raw['graph_recipe_sha256'] = digest * 64
        reports.consume(raw)
    control = result('FB15k237+H', 'ultra-adapter', .2)
    control['benchmark_run']['entry'] = 'ultra-r-FB15k237+H-without-adapter'
    control['paired_with'] = 'ultra-r-FB15k237+H'
    control['inference']['calibration'] = 'without-adapter'
    control['graph_recipe_sha256'] = 'c' * 64
    reports.consume(control)
    # Frozen runs hash each seed's own adapter into its recipe; that alone must not break the pairing.
    point = next(p for p in figures.family_gain_data(reports, 'expected') if p['dataset'] == 'FB15k237+H' and p['method'] == 'ULTRA')
    assert point['seeds'] == 2 and abs(point['value'] - .15) < 1e-12
    reports.results['ultra-r-seed1-FB15k237+H']['raw']['candidate_sha256'] = 'other'
    with pytest.raises(ValueError, match='candidate_sha256'):
        figures.family_gain_data(reports, 'expected')


def test_hardness_profiles_break_lines_at_missing_bins():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = [dict(dataset='FB15k237+H', shape='4p', method='QTO', entry='qto',
                   points=[dict(links=1, value=.4), dict(links=2, value=None), dict(links=3, value=.2), dict(links=4, value=.1)])]
    fig = figures.hardness_profile_plot(plt, series)
    lines = [line for ax in fig.axes for line in ax.get_lines() if len(line.get_xdata()) == 4]
    assert len(lines) == 1 and math.isnan(lines[0].get_ydata()[1])
    plt.close(fig)
