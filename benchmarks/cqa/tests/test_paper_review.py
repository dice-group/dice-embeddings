"""The rule-centred tables and figures (review.py and the rule rows of the main tables), from synthetic reports."""

import pytest

from benchmarks.cqa.paper import figures, review, summary, tables
from benchmarks.cqa.tests.test_paper_tables import ULTRAQUERY_DATASETS, cells, main_table, main_table_section, result, suite_runs

BOTH = ULTRAQUERY_DATASETS + list(tables.PLUS_H_DATASETS)


def adapter_reports(*extra):
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-product-intersections', 'ultra-adapter', .3, datasets=BOTH)
                    + suite_runs('ultra-product-intersections-seed1', 'ultra-adapter', .4, datasets=BOTH)
                    + suite_runs('ultra-product-intersections', 'ultra-adapter', .2, datasets=BOTH,
                                 identity_of='ultra-product-intersections')
                    + [run for runs in extra for run in runs])
    return reports


def contrast_item(name, suite, backbone=None, *, value=.1, ci95=(-.1, .3), ci90=(-.05, .25), per_dataset=None, intervals=None,
                  sign_p=.5, wins=12, losses=11, policy='sort', margin=.5, category='all'):
    row = dict(a=30., b=30., a_minus_b=value, ci95=list(ci95), ci95_clustered=list(ci95), ci90=list(ci90), ci90_clustered=list(ci90),
               datasets=23 if suite == 'ultraquery' else 3, wins=wins, losses=losses, sign_p=sign_p,
               per_dataset=per_dataset or {}, per_dataset_intervals=intervals or {})
    return dict(name=name, suite=suite, policy=policy, backbone=backbone, a='A', b='B', margin=margin,
                subsets={'all': {category: row}})


def test_rule_rows_sit_between_raw_scores_and_the_adapter_with_their_note():
    reports = adapter_reports(suite_runs('ultra-softmax-degree', 'ultra-adapter', .33, datasets=BOTH))
    latex = tables.render_tables(reports)
    rows = [cells(row) for row in main_table(latex, 'tab:main-ultraquery').splitlines()]
    assert [row[0] for row in rows] == ['ULTRA (raw scores)', 'ULTRA + softmax × degree', 'ULTRA + adapter (ours)']
    # The rule is deterministic, so it has no seed spread; second to the adapter's 35.0, it is underlined.
    assert rows[1][1:3] == [r'\underline{33.0}', r'\underline{33.0}']
    plus_h = [cells(row)[0] for row in main_table(latex, 'tab:main-plus-h').splitlines()]
    assert plus_h == ['ULTRA (raw scores)', 'ULTRA + softmax × degree', 'ULTRA + adapter (ours)']
    section = main_table_section(latex, 'tab:main-ultraquery')
    assert "QTO's calibration" in section and 'neither implies a tested difference' in section
    assert 'Best in bold' not in section
    # The rule's runs are main-table systems, so the appendix lists them under their own name.
    assert 'ULTRA + softmax × degree' in latex[latex.index(r'\section*{Appendix}'):]


def test_kgicl_rule_row_is_marked_when_cap_ties_lower_its_one_hop_mrr():
    reports = tables.Reports()
    rule = suite_runs('kgicl-softmax-degree', 'kgicl-adapter', .3)
    for run in rule[:5]:  # 1p below the raw scores on five datasets
        run['per_shape']['1p'] = dict(run['per_shape']['1p'], mrr=.1)
        run['averages'] = tables.averages(run['per_shape'])
    reports.consume(suite_runs('kgicl-product-intersections', 'kgicl-adapter', .35)
                    + suite_runs('kgicl-product-intersections', 'kgicl-adapter', .2, identity_of='kgicl-product-intersections')
                    + rule)
    assert summary.tie_cap_drops(reports, 'kgicl-adapter') == (5, 23)
    latex = summary.ultraquery_table(reports, 'sort', layout='thesis')
    assert r'KG-ICL + softmax × degree$^{*}$' in latex and 'on 5 of 23 UQ-23 datasets' in latex
    assert r'\citealp{bai2022answering}' in latex
    # The tie counts live in the mechanism table; the subset table holds the tie-repair contrast.
    assert r'UQ-23 datasets (Table~\ref{tab:mechanism}); breaking the ties is the tie-repair contrast in Table~\ref{tab:subset-contrasts}' in latex


def test_gains_table_crosses_weights_with_executors_and_facts_with_calibration():
    reports = adapter_reports(suite_runs('ultra-softmax-degree', 'ultra-adapter', .33, datasets=BOTH),
                              suite_runs('ultraquery', 'ultraquery', .25, datasets=BOTH),
                              suite_runs('ultraquery-calibrated', 'ultraquery', .26),
                              suite_runs('ultraquery-lp', 'ultraquery-lp', .2),
                              suite_runs('ultraquery-lp-calibrated', 'ultraquery-lp', .27),
                              suite_runs('ultra-softmax-degree-facts-none', 'ultra-adapter', .28, datasets=BOTH),
                              suite_runs('ultra-product-intersections-facts-none', 'ultra-adapter', .29, datasets=BOTH),
                              suite_runs('ultra-product-intersections-facts-none', 'ultra-adapter', .21, datasets=BOTH,
                                         identity_of='ultra-product-intersections-facts-none'))
    # A query sample of the rule in the beam with UltraQuery's weights never enters the full-test table.
    sample = suite_runs('uqweights-softmax-degree', 'ultra-adapter', .9)
    for run in sample:
        run['protocol']['full_split'] = False
    reports.consume(sample)
    rows = {(r['weights'], r['suite']): r for r in review.executor_rows(reports, 'sort')}
    uq = rows['UltraQuery (query-trained from ULTRA 4g)', 'UQ-23']
    assert uq['values']['uq-plain'] == pytest.approx([.25]) and uq['values']['uq-rule'] == pytest.approx([.26])
    assert uq['values']['beam-rule'] == [] and uq['values']['beam-adapter'] == []
    frozen = rows['ULTRA 3g, frozen', 'UQ-23']
    assert frozen['values']['uq-plain'] == pytest.approx([.2]) and frozen['values']['uq-rule'] == pytest.approx([.27])
    assert frozen['values']['beam-rule'] == pytest.approx([.33]) and frozen['values']['beam-adapter'] == pytest.approx([.3, .4])
    # 1p of the weights from the beam with the adapter (one atom: every monotone calibration ranks alike).
    assert frozen['one_hop'] == pytest.approx([.3, .4])
    # The +H cells of UltraQuery LP exist only as full-test runs of their own, absent here.
    assert rows['ULTRA 3g, frozen', '+H']['values']['uq-plain'] == []
    facts = {(r['backbone'], r['name']): r['values'] for r in review.facts_rows(reports, 'sort')}
    assert [v for row in facts['ULTRA', 'Raw scores'] for v in row] == pytest.approx([.2, .21, .2, .21])
    assert [v for row in facts['ULTRA', 'Softmax × degree (rule)'] for v in row] == pytest.approx([.33, .28, .33, .28])
    assert facts['ULTRA', 'Adapter (ours)'][0] == pytest.approx([.3, .4]) and facts['ULTRA', 'Adapter (ours)'][1] == pytest.approx([.29])
    latex = review.gains_table(reports, 'sort', layout='thesis')
    assert latex.count(r'\begin{threeparttable}') == 2 and r'\label{tab:main-gains}' in latex
    assert '(a) Weights and executors' in latex and '(b) Known facts and calibration' in latex


def test_review_bundle_is_copied_and_conflicts_fail():
    reports = tables.Reports()
    bundle = dict(contrasts=[contrast_item('rule-vs-adapter', 'ultraquery', 'ultra')], onep_ties={'rows': []})
    reports.consume({'review_analyses': bundle})
    reports.consume({'review_analyses': bundle})  # the same contrast twice is kept once
    assert len(reports.review['contrasts']) == 1
    assert review.contrast(reports, 'rule-vs-adapter', 'ultraquery', backbone='ultra')['backbone'] == 'ultra'
    with pytest.raises(ValueError, match='Conflicting review analyses'):
        reports.consume({'review_analyses': dict(onep_ties={'rows': [1]})})


def test_equivalence_rows_state_tost_and_the_smallest_passing_margin():
    reports = tables.Reports()
    reports.consume({'review_analyses': dict(contrasts=[
        contrast_item('rule-vs-adapter', 'ultraquery', 'ultra', value=-.04, ci95=(-.2, .11), ci90=(-.17, .09),
                      per_dataset={'FB15k237LogicalQuery': -.1}),
        contrast_item('rule-vs-adapter', 'ultraquery', 'kgicl', value=-2.8, ci95=(-3.9, -1.8), ci90=(-3.7, -2.0)),
        contrast_item('rule-vs-adapter', 'plus_h', 'ultra', per_dataset={'FB15k237+H': .3},
                      intervals={'FB15k237+H': dict(ci95=[.1, .5], ci90=[.15, .45])})])})
    rows = review.calibration_data(reports, 'sort')['equivalence']
    by = {(r['backbone'], r['label']): r for r in rows}
    assert by['ULTRA', 'UQ-23 (23)']['verdict'] == 'equivalent (smallest margin 0.17)'
    # The 95% interval excludes zero: the rule is lower, which says more than "not shown equivalent".
    assert by['KG-ICL', 'UQ-23 (23)']['verdict'] == 'rule lower'
    # +H graphs have query-level intervals only: descriptive.
    assert by['ULTRA', 'FB15k-237+H']['verdict'].startswith('descriptive') and not by['ULTRA', 'FB15k-237+H']['clustered']
    caption = review.calibration_caption(dict(gains=[], equivalence=rows))
    assert 'after the dataset-clustered 95% intervals were known' in caption


def test_mechanism_series_subtract_facts_off_from_on_per_missing_edge_count():
    def row(entry, label, shape, mrr, dataset):
        return dict(entry=entry, method='ultra-adapter', dataset=dataset, grouping='inferred_positive_edges', label=str(label),
                    shape=shape, answer_filter='corrected', comparison_graph='train+valid', label_reference_graph='train+valid',
                    sort=dict(mrr=mrr))
    reports = tables.Reports()
    rows = []
    for dataset, shift in (('FB15k237+H', 0.), ('NELL995+H', .02)):
        base = f'ultra-product-intersections-{dataset}'
        rows += [row(base + '-without-adapter', 1, '2i', .3 + shift, dataset), row(base + '-without-adapter', 2, '2i', .1 + shift, dataset),
                 row(base + '-facts-none-without-adapter', 1, '2i', .2 + shift, dataset),
                 row(base + '-facts-none-without-adapter', 2, '2i', .2 + shift, dataset)]
    rows.append(row('ultra-product-intersections-ICEWS18+H-without-adapter', 1, '2i', .9, 'ICEWS18+H'))  # no facts-off twin
    reports.consume(rows)
    series = {(s['backbone'], s['shape'], s['missing']): s for s in review.mechanism_data(reports, 'sort')}
    assert series['ULTRA', '2i', 1]['delta'] == pytest.approx(.1) and series['ULTRA', '2i', 2]['delta'] == pytest.approx(-.1)
    assert series['ULTRA', '2i', 2]['off'] == pytest.approx(.21) and series['ULTRA', '2i', 2]['graphs'] == 2


def test_subset_outcomes_follow_the_registered_direction_with_holm():
    assert review.outcome([.2, .9], '>', 0) == 'supported' and review.outcome([-.9, -.2], '>', 0) == 'refuted'
    assert review.outcome([-.2, .3], '<', 0) == 'inconclusive' and review.outcome(None, '<', 0) == 'pending'
    assert review.outcome([-1.2, -.6], '<', -.5) == 'supported'
    # Equivalence within the threshold on the 90% interval.
    assert review.outcome([-.3, .4], '=', .5) == 'supported' and review.outcome([.6, 1.], '=', .5) == 'refuted'
    assert review.outcome([-.2, .7], '=', .5) == 'inconclusive'
    assert review.holm({'a': .01, 'b': .04, 'c': .03}) == pytest.approx({'a': .03, 'b': .06, 'c': .06})
    reports = tables.Reports()
    reports.consume({'review_analyses': dict(contrasts=[
        contrast_item('kgicl-softmax-vs-ties', 'ultraquery', value=-1., ci95=(-2., -.3), sign_p=.01),
        contrast_item('kgicl-softmax-vs-ties', 'ultraquery', value=-1.1, policy='expected'),
        contrast_item('ultra-softmax-vs-qto', 'ultraquery', value=-.5, ci95=(-.9, -.1), sign_p=.02),
        contrast_item('kgicl-softmax-vs-qto', 'plus_h', value=-1.5, per_dataset={'FB15k237+H': -1.4, 'NELL995+H': -3.0},
                      intervals={}),
        contrast_item('kgicl-softmax-vs-adapter', 'plus_h', value=-.3, per_dataset={'FB15k237+H': -.14, 'NELL995+H': -.43}),
        contrast_item('kgicl-hb-vs-adapter', 'ultraquery', value=-.2, ci95=(-.6, .2),
                      per_dataset={'FB15k237LogicalQuery': -.3, 'NELL995LogicalQuery': .1},
                      intervals={'FB15k237LogicalQuery': dict(ci95=[-.5, -.1]), 'NELL995LogicalQuery': dict(ci95=[-.1, .3])}),
        contrast_item('kgicl-hb-vs-adapter', 'plus_h', value=.8, ci95=(.6, 1.), per_dataset={'FB15k237+H': .9, 'NELL995+H': .5},
                      intervals={'FB15k237+H': dict(ci95=[.7, 1.1]), 'NELL995+H': dict(ci95=[.3, .7])})])})
    rows = {(row[0], row[1]): row for row in review.subset_rows(reports)[0]}
    level = rows['KG-ICL rule', 'level: softmax (d = 1) − tie-free cap']
    # Holm over the four registered directional UQ-23 predictions, pending ones counted at p = 1.
    assert level[-1] == 'refuted (Holm p = 0.04)' and level[3] == '-1.00 [-2.00, -0.30], 12/23'
    # Registered for both suites: UQ-23 by its interval, +H descriptively by graphs (none here yet).
    assert rows['ULTRA rule', 'softmax (d = 1) − softmax × degree'][-1] == 'UQ-23: supported (Holm p = 0.06); +H: pending (descriptive)'
    # "Gives back part of the advantage": below the rule on every graph and below the adapter on every graph is all and more.
    total = rows['KG-ICL rule', 'softmax (d = 1) − softmax × degree']
    assert total[-1] == '+H: below softmax × degree on 2/2 graphs, below the adapter on 2/2: gives back all of it and more'
    assert total[4] == 'FB15k-237 -1.40, NELL995 -3.00'
    # The registered interaction, judged on the matched pairs: (+0.9 - (-0.3) + 0.5 - 0.1) / 2 = +0.8, half-width 0.2.
    interaction = rows['Generator bias', 'KG-ICL interaction: (hardness-balanced − standard) on +H minus on UQ-23, matched pairs']
    assert interaction[3] == '+0.80 [+0.60, +1.00] (FB15k-237 +1.20, NELL995 +0.40)'
    assert interaction[4] == 'whole suites +1.00 (reference)'
    assert interaction[-1] == 'supported (matched pairs, approximate interval)'
    assert rows['Weights', 'ULTRA 4g − ULTRA 3g (beam, facts, rule)'][3:] == ['-', '-', '-']
    assert review.subset_rows(reports)[1] == pytest.approx(.1)
    latex = review.subset_table(reports, 'sort')
    assert 'Under expected ties every difference changes by at most 0.10' in latex
    assert 'Representativeness' not in latex  # no check in the bundle, no claim
    error = dict(mean=-.004, mean_abs=.05, max_abs=.124, datasets=23)
    reports.consume({'review_analyses': dict(subset_representativeness=dict(suites=dict(
        ultraquery=dict(error_diff=error), plus_h=dict(error_diff=dict(error, mean=-.04, max_abs=.109, datasets=3)))))})
    latex = review.subset_table(reports, 'sort')
    assert 'changes their difference by at most 0.12 on a UQ-23 dataset (suite mean 0.00) and 0.11 on a +H graph' in latex


def test_mechanism_table_shows_cap_ties_on_both_suites():
    reports = tables.Reports()
    tie = dict(backbone='kgicl', calibration='softmax-degree')
    ultra = dict(tie, backbone='ultra')
    reports.consume({'review_analyses': dict(onep_ties=dict(rows=[
        dict(tie, suite='ultraquery', graphs_below_identity=22, graphs=23, largest_drop=-5.585),
        dict(tie, suite='plus_h', graphs_below_identity=2, graphs=3, largest_drop=-1.069),
        dict(ultra, suite='ultraquery', graphs_below_identity=13, graphs=23, largest_drop=-.412),
        dict(ultra, suite='plus_h', graphs_below_identity=0, graphs=3, largest_drop=-.002)]))})
    rows = {row[0]: row for row in review.mechanism_table_rows(reports)}
    assert rows['KG-ICL'][6] == '22/23, -5.58; +H 2/3, -1.07'
    # No graph drops: no drop value (a sub-tolerance -0.002 would print as -0.00).
    assert rows['ULTRA'][6] == '13/23, -0.41; +H 0/3'
    assert rows['TRIX'][6] == tables.MISSING
    # UltraQuery LP on +H is our setting with carried-over UQ-23 thresholds: marked and explained in the main +H table.
    both = suite_runs('uqlp', 'ultraquery-lp', .2, datasets=list(tables.PLUS_H_DATASETS))
    lp = tables.Reports()
    lp.consume(both)
    latex = summary.plus_h_table(lp, 'sort', layout='thesis')
    assert r'UltraQuery-LP$^{\dagger}$' in latex and 'the matching UQ-23 values are carried over (0.97 on NELL995+H' in latex


def test_negation_rows_judge_the_threshold_and_the_signed_plus_h_gap():
    reports = tables.Reports()
    reports.consume({'review_analyses': dict(contrasts=[
        contrast_item('ultra-negation-observed-vs-adapter', 'ultraquery', value=.35, ci95=(-.9, 1.4), category='negation'),
        contrast_item('ultra-negation-observed-vs-adapter', 'plus_h', value=-.83, category='negation',
                      per_dataset={'FB15k237+H': -1.73, 'NELL995+H': -.81, 'ICEWS18+H': .04}),
        contrast_item('ultra-sd-negation-observed-vs-sd', 'ultraquery', value=-.3, ci95=(-.6, -.1), category='negation'),
        contrast_item('ultra-sd-negation-observed-vs-sd', 'plus_h', value=.8, category='negation', per_dataset={'FB15k237+H': .8})])})
    rows = {(row[0], row[1]): row for row in review.subset_rows(reports)[0]}
    adapter = rows['Negation', 'ULTRA adapter: negation from known facts − from memberships']
    # Registered: model negation ahead by at least 0.5 on UQ-23; the interval straddles -0.5.
    assert adapter[-1] == 'inconclusive; +H: model negation ahead by +0.83 (UQ-23 -0.35), gap not smaller (descriptive)'
    assert adapter[4] == 'FB15k-237 -1.73, NELL995 -0.81, ICEWS18 +0.04'
    # The gap is model negation's signed lead: behind on +H is a smaller gap, though |+0.8| > |-0.3|.
    rule = rows['Negation', 'ULTRA rule: negation from known facts − from memberships']
    assert rule[-1] == 'inconclusive; +H: model negation ahead by -0.80 (UQ-23 +0.30), gap smaller (descriptive)'


def test_rule_mode_hardness_profiles_draw_the_rule_not_the_adapter():
    def row(entry, label, mrr, method='ultra-adapter'):
        return dict(entry=entry, method=method, dataset='NELL995+H', grouping='inferred_positive_edges', label=str(label),
                    shape='3i', answer_filter='corrected', comparison_graph='train+valid', label_reference_graph='train+valid',
                    sort=dict(mrr=mrr))
    reports = tables.Reports()
    raw = result('NELL995+H', 'ultra-adapter', .3)
    raw['benchmark_run']['entry'] = 'ultra-product-intersections-NELL995+H'
    reports.consume([raw, row('ultra-product-intersections-NELL995+H', 1, .5), row('ultra-softmax-degree-NELL995+H', 1, .6),
                     row('ultra-product-intersections-NELL995+H-without-adapter', 1, .4)])
    rule = {s['method'] for s in figures.hardness_profile_data(reports, 'sort')}
    assert 'ULTRA + softmax × degree' in rule and 'ULTRA + adapter (2i/3i)' not in rule
    adapter = {s['method'] for s in figures.hardness_profile_data(reports, 'sort', main='adapter')}
    assert 'ULTRA + softmax × degree' not in adapter


def test_thesis_render_refuses_incomplete_runs(tmp_path):
    reports = adapter_reports()
    broken = result('FB15k237LogicalQuery', 'ultra-adapter', .3, complete=False)
    broken['benchmark_run']['entry'] = 'ultra-interrupted-FB15k237LogicalQuery'
    reports.consume([broken])
    with pytest.raises(ValueError, match='complete test runs'):
        tables.write_thesis(reports, tmp_path)

