"""Paper-table rendering uses synthetic saved reports, never model inference."""

import copy
import json
import re
import subprocess
import sys

import pytest

from benchmarks.cqa.manifests import REPO, read_manifest, suite_directory
from benchmarks.cqa.paper import summary, tables

CLI = [sys.executable, '-m', 'benchmarks.cqa.paper']


def metrics(mrr):
    return dict(mrr=mrr, hits1=mrr / 2, hits3=mrr, hits10=mrr + .1)


def result(dataset='FB15k237+H', method='qto', score=.2, *, shapes=None, complete=True):
    shapes = shapes or (tables.PLUS_H_TYPES if dataset in tables.PLUS_H_DATASETS else tables.ULTRA_TYPES)
    values = {s: dict(metrics(score), queries=10, hard_answers=12) for s in shapes}
    expected = {s: dict(metrics(score - .01), queries=10, hard_answers=12) for s in shapes}
    return dict(dataset=dataset, split='test', benchmark_run={'entry': f'{method}-{dataset}'},
                per_shape=values, averages=tables.averages(values),
                protocol={'tie_policy': 'sort', 'full_split': complete, 'answer_filter': 'released'},
                dataset_metadata={'inference_graph': 'train'},
                inference={'method': method, 'calibration': 'learned', 'selection_protocol': 'source-validation'},
                additional_tie_metrics={'expected': {'per_shape': expected, 'averages': tables.averages(expected)}})


def paper_bodies(latex):
    """Read data rows of the nine numbered tables, including continuation panels."""
    sections = re.findall(r'% BEGIN PAPER TABLE \d+\n(.*?)\n% END PAPER TABLE \d+', latex, re.DOTALL)
    return ['\n'.join(line for body in re.findall(r'\\endfoot\n(.*?)\n\\end\{longtable\}', section, re.DOTALL)
                       for line in body.splitlines() if ' & ' in line) for section in sections]


def main_bodies(latex):
    """Data rows of the main paper's compact floats."""
    sections = re.findall(r'% BEGIN MAIN TABLE \d+\n(.*?)\n% END MAIN TABLE \d+', latex, re.DOTALL)
    return ['\n'.join(line for line in section.split('\\midrule', 1)[1].splitlines() if ' & ' in line)
            for section in sections]


def cells(row):
    return [cell.strip() for cell in re.sub(r'\\\\\*?$', '', row).split(' & ')]


def test_no_data_has_full_evaluation_rows_and_missing_scores_in_paper_order():
    latex = tables.render_tables(tables.Reports())
    assert latex.startswith('% Generated') and r'\begin{document}' in latex
    assert latex.index(r'\section*{Main paper}') < latex.index(r'\section*{Appendix}')
    appendix = latex[latex.index(r'\section*{Appendix}'):]
    captions = re.findall(r'\\caption\{([^}]*)\}', appendix)
    # The benchmark appendix A1--A7, then the review's appendix tables.
    assert captions == list(tables.TABLE_TITLES) + [
        'Paired Subset Contrasts of the Review', 'What Decides Whether the Rule Works', '+H Answer Strata by Setting',
        'Known Facts on +H Intersections: Exact Accounting', 'Known Facts On and Off per Dataset',
        'Deviations from the Registered Protocol']
    mains = main_bodies(latex)
    # UltraQuery, +H, the calibrations (raw scores and adapter of each planned backbone), the ablations, then the gains.
    assert [len(body.splitlines()) for body in mains[:4]] == [5, 11, 4, 8]
    for body, count in zip(mains, (8, 8, 4, 4)):
        for row in body.splitlines():
            # Reference rows of the ablation table have no change from themselves.
            assert all(cell in ('-', r'\textemdash{}') for cell in cells(row)[-count:])
    # Gains, panel (a): three weight sets on two suites, every score missing.
    section = main_table_section(latex, 'tab:main-gains')
    panel = section.split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
    executor = [cells(row) for row in panel.splitlines() if ' & ' in row]
    assert len(executor) == 6 and all(cell == '-' for row in executor for cell in row[2:])
    bodies = paper_bodies(latex)
    assert len(bodies) == 7
    score_columns = (4, 5, 16, 4, 4, 2)
    for body, count in zip(bodies, score_columns):
        for row in body.splitlines():
            assert cells(row)[-count:] == ['-'] * count
    assert '0.00' not in ''.join(bodies[:6]) + ''.join(mains)
    assert all(len(body.splitlines()) > 1 for body in bodies)
    assert 'No-data template: planned full test' in latex
    assert 'a dash (-) marks unavailable data' in latex
    assert 'unless marked' not in latex and '* = ' not in latex
    assert all('*' not in cell for body in bodies for row in body.splitlines() for cell in cells(row))
    assert r'\renewcommand{\thetable}{A\arabic{table}}' in latex
    assert latex.count('Hardness uses training + validation facts as the fixed reference') == 1
    assert 'hardness reference facts:' not in latex and 'author reference facts:' not in latex


def test_no_data_matrix_matches_default_manifests_and_generated_controls():
    reports = tables.empty_reports()
    expected_ids = set()
    ultra_datasets = set()
    for suite in ('ultraquery', 'plus_h'):
        for name in ('baselines.json', 'kgfm_adapters.json'):
            manifest = read_manifest(suite_directory(suite) / name)
            for entry in manifest['entries']:
                if suite == 'ultraquery':
                    ultra_datasets.add(entry['dataset'])
                suffixes = ['', '-without-adapter'] if entry.get('adapter_ablation') else ['']
                for suffix in suffixes:
                    expected_ids.add(entry['id'] + suffix)
    assert len(ultra_datasets) == 23
    assert set(reports.results) == expected_ids
    assert len(expected_ids) == 148
    assert len(tables.ultra_matrix_rows(reports, 'expected')) == 115 * 2
    assert len(tables.main_plus_h_rows(reports, 'expected')) == 27
    for dataset in tables.PLUS_H_DATASETS:
        assert sum(row[0] == tables.dataset_name(dataset) for row in tables.main_plus_h_rows(reports, 'expected')) == 9
    assert len(tables.main_adapter_matrix(reports, 'expected')[2]) == 26
    assert len(reports.effects['filter']) == 33
    assert reports.effects['graph'] == []
    for method in ('cqd', 'cqd-hybrid'):
        for dataset in tables.PLUS_H_DATASETS:
            record = reports.results[f'{method}-{dataset}']
            assert record['complete']
            assert set(tables.policy_values(record, 'expected')['per_shape']) == set(tables.PLUS_H_TYPES)
            assert record['raw']['inference']['options']['atomic_negation']


def test_raw_result_scales_scores_and_marks_missing_shapes_with_dash():
    reports = tables.Reports()
    reports.consume(result(shapes=['1p', '2p']))
    rows = tables.main_plus_h_rows(reports, 'expected')
    assert rows[0][:5] == ['FB15k-237+H', 'QTO', 'test / partial', '19.00', '19.00']
    assert rows[0][5:] == ['-'] * 14


def test_table_two_requires_shared_graph_and_filters_but_accepts_graph_independent_methods():
    reports = tables.Reports()
    graph_based = result(method='qto')
    graph_based['dataset_metadata']['inference_graph'] = 'train+valid'
    reports.consume(graph_based)
    for method in ('cone', 'clmpt', 'cqd'):
        reports.consume(result(method=method))
    rows = tables.main_plus_h_rows(reports, 'expected')
    assert len(rows) == 4
    latex = tables.render_tables(reports)
    assert 'Graph-based methods use training + validation facts.' in latex
    assert 'ConE, CLMPT and plain CQD do not consume graph facts at inference.' in latex
    assert 'All methods use released answer filters.' in latex
    reports.consume(result(method='cqd-hybrid'))
    with pytest.raises(ValueError, match=r'The \+H tables require the same inference graph'):
        tables.render_tables(reports)


def test_table_two_rejects_mixed_answer_filters_for_graph_independent_methods():
    reports = tables.Reports()
    corrected = result(method='cqd')
    corrected['protocol']['answer_filter'] = 'corrected'
    reports.consume([corrected, result(method='clmpt')])
    with pytest.raises(ValueError, match=r'The \+H tables require the same answer filters'):
        tables.main_plus_h_rows(reports, 'expected')


@pytest.mark.parametrize('field', ['graph', 'filter'])
def test_table_two_requires_recorded_protocol_metadata(field):
    raw = result()
    if field == 'graph':
        raw['dataset_metadata'].pop('inference_graph')
    else:
        raw['protocol'].pop('answer_filter')
    reports = tables.Reports()
    reports.consume(raw)
    with pytest.raises(ValueError, match='explicitly recorded'):
        tables.main_plus_h_rows(reports, 'expected')


def test_additional_tie_macro_can_be_computed_from_recorded_per_type_scores():
    learned = result(method='ultra-adapter', score=.3)
    identity = result(method='ultra-adapter', score=.2)
    identity['benchmark_run']['entry'] += '-without-adapter'
    identity['paired_with'] = learned['benchmark_run']['entry']
    identity['inference']['calibration'] = 'without-adapter'
    for raw in (learned, identity):
        raw['additional_tie_metrics']['expected'].pop('averages')
    reports = tables.Reports()
    reports.consume([learned, identity])
    assert tables.adapter_rows(reports, 'expected')[0][4:7] == ['29.00', '19.00', '+10.00']


def test_comparison_details_match_raw_results_without_double_scaling():
    raw = result()
    detail = dict(id='qto-FB15k237+H', dataset=raw['dataset'],
                  per_shape={s: {'sort': metrics(.2), 'expected': metrics(.19), 'queries': 10, 'hard_answers': 12}
                             for s in tables.PLUS_H_TYPES},
                  averages={p: tables.averages({s: metrics(score) for s in tables.PLUS_H_TYPES})
                            for p, score in [('sort', .2), ('expected', .19)]})
    row = dict(id=detail['id'], dataset=raw['dataset'], method='qto', inference_graph='train',
               answer_filter='released', calibration='learned', phase='test', complete_test=True,
               sort_mrr=20, expected_random_mrr=19)
    reports = tables.Reports()
    reports.consume(raw)
    reports.consume(dict(rows=[row], details=[detail]))
    assert len(reports.results) == 1
    assert tables.main_plus_h_rows(reports, 'expected')[0][3:] == ['19.00'] * 16
    assert reports.results[detail['id']]['raw'] == raw
    summaries = tables.Reports()
    summaries.consume(dict(rows=[row], details=[]))
    assert tables.main_plus_h_rows(summaries, 'expected')[0][3:] == ['-'] * 16
    reports.consume(dict(rows=[row], details=[]))
    assert tables.main_plus_h_rows(reports, 'expected')[0][3:] == ['19.00'] * 16


def test_primary_tie_policy_and_unavailable_policy_are_respected():
    raw = result()
    raw['protocol']['tie_policy'] = 'expected'
    raw.pop('additional_tie_metrics')
    reports = tables.Reports()
    reports.consume(raw)
    assert tables.main_plus_h_rows(reports, 'expected')[0][3] == '20.00'
    assert tables.main_plus_h_rows(reports, 'sort')[0][3:] == ['-'] * 16


def test_paper_hardness_uses_only_the_fixed_test_reference():
    rows = [dict(dataset='FB15k237+H', entry='qto-FB15k237+H', shape='3p',
                 grouping=g, label=label, comparison_graph='train', label_reference_graph=graph,
                 answer_filter='corrected', queries=7, hard_answers=11, positive_edges=3,
                 sort=metrics(.25), expected=metrics(.24))
            for g, label, graph in [('difficulty', 'partial', 'train'),
                                    ('inferred_positive_edges', '1', 'train'),
                                    ('released_reduction', '1p', 'train+valid')]]
    reports = tables.Reports()
    reports.consume(dict(rows=rows, details=[], semantics='test'))
    appendix = tables.hardness_matrix_rows(reports, 'expected')
    assert appendix == []
    reports.consume([{**row, 'comparison_graph': 'train+valid', 'label_reference_graph': 'train+valid'} for row in rows])
    appendix = tables.hardness_matrix_rows(reports, 'expected')
    assert len(appendix) == 8
    released = next(r for r in appendix if r[4].startswith('Released reduction') and r[5] == 'MRR')
    assert released[2] == 'train+valid'
    assert released[8] == '24.00'
    assert any(r[4] == 'Missing links: 1' for r in appendix)
    assert not any(r[4].startswith('Coarse:') for r in appendix)


def author_category(shape, grouping, label, score, *, graph='train+valid', complete=True):
    return dict(dataset='FB15k237+H', entry='qto-FB15k237+H', shape=shape,
                grouping=grouping, label=label, comparison_graph=graph,
                label_reference_graph='train+valid' if grouping == 'released_reduction' else graph,
                answer_filter='corrected', complete_shape=complete,
                parent_queries=100 if complete else 10, available_queries=100,
                queries=5, hard_answers=7, sort=metrics(score), expected=metrics(score))


def author_result(shapes):
    raw = result(shapes=shapes, score=.4)
    raw['dataset_metadata']['inference_graph'] = 'train+valid'
    raw['protocol']['answer_filter'] = 'corrected'
    return raw


def test_adapter_deltas_and_supplied_intervals_are_scaled_once():
    learned = result(method='ultra-adapter', score=.3)
    control = result(method='ultra-adapter', score=.2)
    control['benchmark_run']['entry'] += '-without-adapter'
    control['paired_with'] = learned['benchmark_run']['entry']
    control['inference']['calibration'] = 'without-adapter'
    reports = tables.Reports()
    reports.consume([learned, control])
    assert tables.adapter_rows(reports, 'expected')[0][4:] == ['29.00', '19.00', '+10.00', '-']
    reports.consume([dict(learned=learned['benchmark_run']['entry'], control=control['benchmark_run']['entry'],
                          dataset='FB15k237+H', macro={'expected': {'mrr': .1, 'mrr_ci95': [.08, .12]}})])
    assert tables.adapter_rows(reports, 'expected')[0][-1] == '[+8.00, +12.00]'
    reports.results[control['benchmark_run']['entry']]['counts']['1p']['queries'] = 9
    with pytest.raises(ValueError, match='query/answer counts'):
        tables.adapter_rows(reports, 'expected')


def test_conflicting_results_and_unrecognized_data_fail():
    reports = tables.Reports()
    reports.consume(result())
    with pytest.raises(ValueError, match='Conflicting entry'):
        reports.consume(result(score=.5))
    with pytest.raises(ValueError, match='query-type coverage'):
        reports.consume(result(shapes=['1p']))
    with pytest.raises(ValueError, match='Unrecognized report'):
        reports.consume({'scores': [1, 2, 3]})
    with pytest.raises(ValueError, match='finite numeric'):
        tables.number('bad score')


def test_missing_values_are_dashes_and_real_zeros_are_preserved():
    for value in (None, '', '  ', float('nan'), float('inf'), float('-inf')):
        assert tables.number(value) == tables.escape(value) == '-'
    assert tables.number(0) == '0.00'
    assert tables.escape(0) == '0'
    assert tables.escape(False) == 'False'
    assert tables.ci([None, .2]) == tables.ci([.1, float('nan')]) == '-'
    assert tables.ci([0, 0]) == '[+0.00, +0.00]'


def test_partial_reports_keep_missing_metrics_types_and_metadata_visible():
    raw = result('FB15k237LogicalQuery', shapes=['1p'])
    raw['per_shape']['1p']['mrr'] = None
    raw['per_shape']['1p']['hits10'] = float('nan')
    raw['per_shape']['1p']['queries'] = None
    raw.pop('averages')
    raw.pop('additional_tie_metrics')
    raw.pop('inference')
    reports = tables.Reports()
    reports.consume(raw)
    rows = tables.ultra_matrix_rows(reports, 'sort')
    assert len(rows) == 2
    assert all(row[4:] == ['-'] * 17 for row in rows)
    assert next(row for row in tables.shared_metadata_rows(reports) if row[1] == 'checkpoint')[2] == '-'
    bodies = paper_bodies(tables.render_tables(reports, policy='sort'))
    for body in bodies:
        for row in body.splitlines():
            assert all(cells(row))


def test_query_score_columns_have_space_in_empty_and_populated_tables():
    reports = tables.Reports()
    reports.consume(result())
    for latex in (tables.render_tables(tables.Reports()), tables.render_tables(reports)):
        columns = dict((caption, spec) for spec, caption in re.findall(
            r'\\begin\{longtable\}\{([^\n]+)\}\n\\caption\{([^}]*)\}', latex))['+H Per-Query-Type Performance']
        assert [int(w) for w in re.findall(r'p\{(\d+)mm\}', columns)] == [30, 40, *([11] * 16)]


def test_null_shapes_policies_and_hardness_metrics_are_missing():
    raw = result(shapes=['1p', '2p'])
    raw['per_shape']['2p'] = None
    raw['averages'] = None
    raw['additional_tie_metrics']['expected'] = None
    raw['inference'] = None
    reports = tables.Reports()
    reports.consume(raw)
    assert tables.main_plus_h_rows(reports, 'sort')[0][3:5] == ['20.00', '-']
    assert tables.main_plus_h_rows(reports, 'expected')[0][3:] == ['-'] * 16
    reports.consume(dict(dataset=raw['dataset'], shape='2p', grouping='difficulty', label='full', expected=None))
    assert all(row[6:] == ['-'] * 16 for row in tables.hardness_matrix_rows(reports, 'expected') if row[4] == 'Coarse: full')
    tables.render_tables(reports)


def test_cli_prints_exact_saved_latex_and_escapes_input(tmp_path):
    path = tmp_path / 'result.json'
    raw = result()
    raw['inference']['method'] = r'model_%&\input{bad}'
    path.write_text(json.dumps(raw))
    destination = tmp_path / 'nested' / 'tables.tex'
    run = subprocess.run([*CLI, str(path), '--output', str(destination), '--fragment'],
                         capture_output=True, text=True, check=True, cwd=REPO)
    assert run.stdout == destination.read_text()
    assert r'model\_\%\&\textbackslash{}input\{bad\}' in run.stdout
    assert r'\documentclass' not in run.stdout and 'Saved ' in run.stderr
    rejected = subprocess.run([*CLI, str(path), '--output', str(path)], capture_output=True, text=True, cwd=REPO)
    assert rejected.returncode == 2 and json.loads(path.read_text()) == raw


def test_cli_empty_input_needs_no_dicee_imports_or_model_dependencies(tmp_path):
    destination = tmp_path / 'empty.tex'
    # -S disables site packages, so this also verifies the standard-library-only path.
    run = subprocess.run([sys.executable, '-S', '-m', 'benchmarks.cqa.paper', '-o', str(destination)],
                         capture_output=True, text=True, check=True, cwd=REPO)
    assert run.stdout == destination.read_text()
    from benchmarks.cqa.paper import review
    # The thesis's main tables, the ablations (a main float in the paper, an appendix float in the thesis), the benchmark
    # appendix A1--A7 and the review's appendix tables.
    assert run.stdout.count(r'\caption{') == len(tables.THESIS_MAIN_TABLES) + 1 + len(tables.TABLE_TITLES) + len(review.APPENDIX_TABLES)


def test_combined_report_populates_every_table(tmp_path):
    learned = result(method='ultra-adapter', score=.3)
    identity = result(method='ultra-adapter', score=.2)
    identity['benchmark_run']['entry'] += '-without-adapter'
    identity['paired_with'] = learned['benchmark_run']['entry']
    identity['inference']['calibration'] = 'without-adapter'
    difficulty = dict(dataset=learned['dataset'], entry=learned['benchmark_run']['entry'], shape='3p',
                      grouping='inferred_positive_edges', label='3', comparison_graph='train+valid', answer_filter='released',
                      queries=7, hard_answers=11, sort=metrics(.1), expected=metrics(.09))
    payload = dict(results=[learned, identity, result('WikiTopicsQuery:art', 'ultraquery')],
                   difficulty={'rows': [difficulty], 'semantics': 'test'},
                   adapter_effects=[dict(dataset=learned['dataset'], learned=identity['paired_with'],
                                         control=identity['benchmark_run']['entry'],
                                         macro={'expected': {'mrr': .1, 'mrr_ci95': [.08, .12]}})])
    source = tmp_path / 'combined.json'
    source.write_text(json.dumps(payload))
    reports = tables.load_reports([source])
    latex = tables.render_tables(reports, policy='expected')
    bodies = paper_bodies(latex)
    assert len(bodies) == 7
    assert all(body.replace('&', '').replace('\\', '').strip() for body in bodies + main_bodies(latex))
    assert 'uniformly random tie orders' in latex and '+10.00' in latex


def test_empty_presentation_is_bounded_and_every_table_fits_the_page():
    latex = tables.render_tables(tables.Reports())
    bodies = paper_bodies(latex)
    counts = [len(body.splitlines()) for body in bodies]
    assert counts[:6] == [5, 230, 27, 366, 26, 2]
    assert counts[6] < 120
    assert sum(counts) < 1500 and len(latex.encode()) < 300_000
    assert 'released-filters' not in bodies[2] and 'identity' not in bodies[2]
    assert r'\multicolumn{4}{c}{ULTRA}' in latex and r'\multicolumn{4}{c}{TRIX}' in latex
    for columns in re.findall(r'\\begin\{longtable\}\{([^\n]+)\}', latex):
        widths = [int(width) for width in re.findall(r'p\{(\d+)mm\}', columns)]
        assert sum(widths) + len(widths) * 3 * 25.4 / 72.27 <= 267
    settings = tables.shared_metadata_rows(tables.empty_reports())
    assert [row for row in settings if row[1] == 'seed'] == [['All configurations', 'seed', '0']]


def test_controls_stay_in_appendix_and_paired_scores_survive_pivoting():
    learned = result(method='ultra-adapter', score=.3)
    learned['protocol']['answer_filter'] = 'corrected'
    identity = copy.deepcopy(learned)
    identity['benchmark_run']['entry'] += '-without-adapter'
    identity['paired_with'] = learned['benchmark_run']['entry']
    identity['inference']['calibration'] = 'without-adapter'
    identity['per_shape'] = {s: metrics(.2) | {'queries': 10, 'hard_answers': 12} for s in tables.PLUS_H_TYPES}
    identity['averages'] = tables.averages(identity['per_shape'])
    identity['additional_tie_metrics']['expected']['per_shape'] = {
        s: metrics(.19) | {'queries': 10, 'hard_answers': 12} for s in tables.PLUS_H_TYPES}
    identity['additional_tie_metrics']['expected']['averages'] = tables.averages(identity['additional_tie_metrics']['expected']['per_shape'])
    released = copy.deepcopy(learned)
    released['benchmark_run']['entry'] += '-released-filters'
    released['protocol']['answer_filter'] = 'released'
    reports = tables.Reports()
    reports.consume([learned, identity, released])
    reports.consume({'adapter_effects': [dict(dataset=learned['dataset'], learned=identity['paired_with'],
        control=identity['benchmark_run']['entry'], macro={'expected': {'mrr': .1, 'mrr_ci95': [.08, .12]}})]})
    assert len(tables.main_plus_h_rows(reports, 'expected')) == 1
    assert tables.main_plus_h_rows(reports, 'expected')[0][3:] == ['29.00'] * 16
    assert tables.main_adapter_matrix(reports, 'expected')[2][0][-4:] == ['29.00', '19.00', '+10.00', '[+8.00, +12.00]']
    overall = [row for row in tables.hardness_matrix_rows(reports, 'expected') if row[4] == 'Overall: all' and row[5] == 'MRR']
    assert len(overall) == 3 and {row[3] for row in overall} == {'released', 'corrected'}


def test_overall_raw_and_difficulty_reports_merge_without_double_counting():
    raw = result(shapes=['3p'])
    reports = tables.Reports()
    reports.consume(raw)
    reports.consume(dict(dataset=raw['dataset'], entry=raw['benchmark_run']['entry'], shape='3p',
                         grouping='overall', label='all', comparison_graph='train', answer_filter='released',
                         queries=10, hard_answers=12, sort=metrics(.2), expected=metrics(.19)))
    rows = tables.hardness_matrix_rows(reports, 'expected')
    assert len(rows) == 4
    assert next(row for row in rows if row[5] == 'MRR')[8] == '19.00'
    assert next(row for row in rows if row[5] == 'Queries')[8] == '10'
    reports.difficulty[0]['expected']['mrr'] = .8
    with pytest.raises(ValueError, match='Conflicting hardness'):
        tables.hardness_matrix_rows(reports, 'expected')


def test_presentation_removes_redundant_columns_without_losing_paired_scores():
    learned = result(method='ultra-adapter', score=.3)
    identity = result(method='ultra-adapter', score=.2)
    identity['benchmark_run']['entry'] += '-without-adapter'
    identity['paired_with'] = learned['benchmark_run']['entry']
    identity['inference']['calibration'] = 'without-adapter'
    reports = tables.Reports()
    reports.consume([learned, identity])
    reports.consume(dict(dataset=learned['dataset'], learned=identity['paired_with'],
                         control=identity['benchmark_run']['entry'],
                         macro={'expected': {'mrr': .1, 'mrr_ci95': [.08, .12]}}))
    latex = tables.render_tables(reports, policy='expected')
    headers = '\n'.join(re.findall(r'\\toprule\n(.*?)\\midrule', latex, re.DOTALL))
    assert not any(column in headers for column in ('Condition A', 'Condition B', ' & Scope & ', ' & Recipe & '))
    assert r'\multicolumn{4}{c}{ULTRA}' in latex
    assert cells(paper_bodies(latex)[4].splitlines()[0])[-4:] == ['29.00', '19.00', '+10.00', '[+8.00, +12.00]']


def test_panel_context_preserves_graph_filter_scope_and_latex_escaping():
    reports = tables.Reports()
    raw = result('WikiTopicsQuery:art', shapes=['1p'])
    raw['dataset_metadata']['inference_graph'] = 'train_custom&other'
    raw['protocol']['answer_filter'] = 'corrected'
    reports.consume(raw)
    for graph, filters, score in [('train+valid', 'released', .25), ('train+valid', 'corrected', .5)]:
        reports.consume(dict(dataset='FB15k237+H', entry='qto-FB15k237+H', shape='3p',
                             grouping='difficulty', label='partial', comparison_graph=graph,
                             answer_filter=filters, expected=metrics(score)))
    latex = tables.render_tables(reports, policy='expected')
    assert 'FB15k-237+H | filters: released' in latex
    assert 'FB15k-237+H | filters: corrected' in latex
    assert 'hardness reference facts:' not in latex
    assert 'Scope: test /' in latex and r'train\_\allowbreak{}custom\&other' in latex
    rows = [cells(row) for row in paper_bodies(latex)[3].splitlines()]
    assert {row[6] for row in rows if row[3] == 'MRR'} == {'25.00', '50.00'}
    assert latex.count(r'\caption{Full +H Hardness Breakdowns}') == 1
    assert r'\addtocounter{table}{-1}' in latex


def test_paper_settings_are_readable_and_keep_overrides_and_checkpoint_identity():
    a, b = result(method='cone'), result(method='qto')
    for raw, model in ((a, 'ConE'), (b, 'QTO')):
        raw['inference']['checkpoint'] = '/machine/private/checkpoints/' + model + '/FB15k237/checkpoint'
        raw['inference']['options'] = {'beam_size': 2, 'cache_bytes': 2**20,
                                      'per_shape': {'3p': {'beam_size': 8}}}
    a['inference']['paper_protocol'] = {'training': {'learning_rate': 0, 'epochs': 12}}
    reports = tables.Reports()
    reports.consume([a, b])
    rows = tables.shared_metadata_rows(reports)
    assert {row[2] for row in rows if row[1] == 'checkpoint'} == {'ConE/FB15k237/checkpoint', 'QTO/FB15k237/checkpoint'}
    assert next(row[2] for row in rows if row[1] == 'beam size') == '2 (default); 3p: 8'
    assert next(row[2] for row in rows if row[1] == 'cache bytes') == '1 MiB'
    assert next(row[2] for row in rows if row[1] == 'training: learning rate') == '0'
    settings = '\n'.join(' '.join(row) for row in rows)
    assert '/machine/private' not in settings and 'per\\_shape' not in settings and '\\{' not in settings
    assert tables.applicability({('a', 'ConE')}, {('a', 'ConE'), ('b', 'ConE')}) == 'All methods (a)'


def difficulty(entry, label, score, *, graph='train+valid', filters='corrected', complete=True):
    return dict(dataset='FB15k237+H', method='qto', entry=entry, shape='3p',
                grouping='inferred_positive_edges', label=str(label), comparison_graph=graph, answer_filter=filters,
                parent_queries=1000 if complete else 10, available_queries=1000,
                complete_shape=complete, queries=7, hard_answers=11, expected=metrics(score))


def test_main_hardness_never_splices_bins_across_runs_or_conditions():
    reports = tables.Reports()
    reports.consume([difficulty('qto-preferred', '1', .8), difficulty('qto-preferred', '3', .4),
                     difficulty('qto-other', '2', .2, graph='train', filters='released')])
    bodies = paper_bodies(tables.render_tables(reports, policy='expected'))
    assert '80.00' in bodies[3] and '20.00' not in bodies[3]


def test_hardness_scope_reports_parent_coverage_without_score_markers():
    reports = tables.Reports()
    reports.consume(difficulty('qto-sampled', '3', .24, complete=False))
    latex = tables.render_tables(reports, policy='expected')
    bodies = paper_bodies(latex)
    assert '24.00' in bodies[3]
    assert '24.00*' not in latex and '24.00?' not in latex
    assert '10/\\allowbreak{}1000 parent queries' in latex
    assert 'partial parent-type test' in latex and 'queries can occur in several bins' in latex
    reports.difficulty[0]['complete_shape'] = True
    with pytest.raises(ValueError, match='complete coverage'):
        tables.render_tables(reports)


def test_three_hop_bins_keep_distinct_scores_and_zero_is_only_a_diagnostic():
    reports = tables.Reports()
    reports.consume([difficulty('qto', 0, .9), difficulty('qto', 1, .4),
                     difficulty('qto', 2, .3), difficulty('qto', 3, .2)])
    latex = tables.render_tables(reports, policy='expected')
    assert all(score in paper_bodies(latex)[3] for score in ('40.00', '30.00', '20.00'))
    assert 'Diagnostic: 0' in paper_bodies(latex)[3]
    assert 'Observed' not in latex and 'Obs.' not in latex
    assert 'Zero-cost answers are excluded' not in latex
    assert 'label graph' not in latex.lower()


def test_coarse_and_released_reductions_cannot_be_invented_as_numeric_scores():
    reports = tables.Reports()
    for grouping, label in [('difficulty', 'partial'), ('difficulty', 'full'), ('released_reduction', '2p')]:
        row = difficulty('qto', 1, .4)
        row.update(grouping=grouping, label=label)
        reports.consume(row)
    latex = tables.render_tables(reports, policy='expected')
    assert 'Coarse: partial' in paper_bodies(latex)[3]
    assert 'Released reduction: 2p' in paper_bodies(latex)[3]


@pytest.mark.parametrize('label', ['-1', '4', 'partial', True])
def test_invalid_missing_link_bins_are_rejected(label):
    reports = tables.Reports()
    reports.consume(difficulty('qto', label, .4))
    with pytest.raises(ValueError, match='Missing-positive-link'):
        tables.render_tables(reports)


def test_appendix_preserves_alternate_recipes_and_settings_applicability():
    a = result('WikiTopicsQuery:art', 'ultra-adapter', .2)
    b = result('WikiTopicsQuery:art', 'ultra-adapter', .5)
    a['benchmark_run']['entry'] = 'ultra-product-intersections-' + a['dataset']
    b['benchmark_run']['entry'] = 'ultra-product-14type-' + b['dataset']
    a['inference']['options'] = {'beam_size': 64}
    b['inference']['options'] = {'beam_size': 128}
    reports = tables.Reports()
    reports.consume([a, b])
    latex = tables.render_tables(reports, policy='expected')
    rows = [cells(row) for row in paper_bodies(latex)[1].splitlines() if cells(row)[2] == 'MRR']
    assert {row[1].replace(r'\allowbreak{}', '') for row in rows} == {
        'ULTRA + adapter (2i/3i)', 'ULTRA + adapter (14-type)'}
    assert {row[3] for row in rows} == {'19.00', '49.00'}
    settings = tables.shared_metadata_rows(reports)
    beams = [row for row in settings if row[1] == 'beam size']
    assert len(beams) == 2 and all(row[0] != 'All configurations' for row in beams)
    assert {row[2] for row in beams} == {'64', '128'}
    assert len([row for row in settings if row[1] == 'run ID']) == 2


def test_same_recipe_runs_get_consistent_distinct_labels_in_all_appendices():
    a = result('WikiTopicsQuery:art', 'ultraquery', .2)
    b = copy.deepcopy(a)
    b['benchmark_run']['entry'] += '-alternative'
    a['inference']['options'], b['inference']['options'] = {'beam_size': 64}, {'beam_size': 128}
    reports = tables.Reports()
    reports.consume([a, b])
    labels = set(tables.run_labels(reports).values())
    assert labels == {'UltraQuery (default)', 'UltraQuery (alternative)'}
    bodies = paper_bodies(tables.render_tables(reports))
    assert all(label in bodies[1] and label in bodies[6] for label in labels)


@pytest.mark.parametrize('field,left,right', [('options', {'beam_size': 64}, {'beam_size': 128}),
    ('operators', {'3p': 'product'}, {'3p': 'min'}), ('checkpoint_sha256', 'a' * 64, 'b' * 64)])
def test_adapter_ablation_rejects_changed_execution_recipe(field, left, right):
    a, b = result(method='ultra-adapter', score=.3), result(method='ultra-adapter', score=.2)
    b['benchmark_run']['entry'] += '-without-adapter'
    b['paired_with'] = a['benchmark_run']['entry']
    b['inference']['calibration'] = 'without-adapter'
    a['inference'][field], b['inference'][field] = left, right
    reports = tables.Reports()
    reports.consume([a, b])
    with pytest.raises(ValueError, match=field):
        tables.render_tables(reports)


def test_effect_delta_is_checked_against_scores_and_ci_only_effect_retains_difference():
    a, b = result(method='ultra-adapter', score=.3), result(method='ultra-adapter', score=.2)
    b['benchmark_run']['entry'] += '-without-adapter'
    b['paired_with'] = a['benchmark_run']['entry']
    b['inference']['calibration'] = 'without-adapter'
    reports = tables.Reports()
    reports.consume([a, b])
    effect = dict(dataset=a['dataset'], learned=a['benchmark_run']['entry'], control=b['benchmark_run']['entry'],
                  macro={'expected': {'mrr_ci95': [.08, .12]}})
    reports.consume(effect)
    assert tables.adapter_rows(reports, 'expected')[0][-2:] == ['+10.00', '[+8.00, +12.00]']
    reports.effects['adapter'][0]['macro']['expected']['mrr'] = .5
    with pytest.raises(ValueError, match='Inconsistent adapter'):
        tables.render_tables(reports, policy='expected')


def test_conflicting_run_recipes_and_inconsistent_saved_averages_are_rejected():
    a = result()
    a['inference']['options'] = {'beam_size': 64}
    b = copy.deepcopy(a)
    b['inference']['options']['beam_size'] = 128
    reports = tables.Reports()
    reports.consume(a)
    with pytest.raises(ValueError, match='Conflicting entry.*options'):
        reports.consume(b)
    a['averages']['all']['mrr'] = .8
    with pytest.raises(ValueError, match='Inconsistent sort all mrr average'):
        tables.Reports().consume(a)


def test_manifest_executor_settings_and_root_overrides_are_reported_and_validated():
    a, b = result(method='ultra-adapter', score=.3), result(method='ultra-adapter', score=.2)
    b['benchmark_run']['entry'] += '-without-adapter'
    b['paired_with'] = a['benchmark_run']['entry']
    b['inference']['calibration'] = 'without-adapter'
    for raw in (a, b):
        raw['inference']['options'] = None
        raw['inference']['manifest'] = {'query_batch_size': 64,
            'options': {'threshold': .001, 'negation_scale': 6, 'row_batch_size': 32,
                        'reference_batching': True, 'cache_bytes': 512 * 2**20},
            'per_shape': {'3p': {'threshold': .01}}}
    reports = tables.Reports()
    reports.consume([a, b])
    settings = tables.shared_metadata_rows(reports)
    assert next(row[2] for row in settings if row[1] == 'threshold') == '0.001 (default); 3p: 0.01'
    assert next(row[2] for row in settings if row[1] == 'query batch size') == '64'
    assert next(row[2] for row in settings if row[1] == 'cache bytes') == '512 MiB'
    assert next(row[2] for row in settings if row[1] == 'reference batching') == 'yes'
    assert not any(row[1] == 'execution options' for row in settings)
    assert tables.adapter_rows(reports, 'expected')[0][-2] == '+10.00'
    reports.results[b['benchmark_run']['entry']]['raw']['inference']['manifest']['per_shape']['3p']['threshold'] = .02
    with pytest.raises(ValueError, match='changes options'):
        tables.adapter_rows(reports, 'expected')


def test_adapter_effect_rejects_a_control_marked_as_another_learned_adapter():
    a, b = result(method='ultra-adapter', score=.3), result(method='ultra-adapter', score=.2)
    b['benchmark_run']['entry'] += '-another-learned-run'
    b['paired_with'] = a['benchmark_run']['entry']
    reports = tables.Reports()
    reports.consume([a, b])
    with pytest.raises(ValueError, match='learned and identity'):
        tables.adapter_rows(reports, 'expected')


def test_invalid_absolute_score_range_is_not_formatted_as_a_result():
    assert tables.number(0) == '0.00' and tables.number(1) == '100.00'
    assert tables.number(-.1, signed=True) == '-10.00'
    for value in (-.01, 1.01):
        with pytest.raises(ValueError, match='fractional range'):
            tables.number(value)


ULTRAQUERY_DATASETS = [d for d in tables.catalog().BENCHMARK_DATASETS if tables.family(d)]


def suite_runs(entry, method, score, *, datasets=ULTRAQUERY_DATASETS, identity_of=None):
    """One result per dataset; entry is the recipe ID without the dataset, e.g. 'ultra-r-seed1'."""
    runs = []
    for dataset in datasets:
        raw = result(dataset, method, score)
        raw['benchmark_run']['entry'] = f'{entry}-{dataset}'
        if identity_of:
            raw['benchmark_run']['entry'] += '-without-adapter'
            raw['paired_with'] = f'{identity_of}-{dataset}'
            raw['inference']['calibration'] = 'without-adapter'
        runs.append(raw)
    return runs


def main_table(latex, label):
    """Data rows of the main float with this label."""
    sections = re.findall(r'% BEGIN MAIN TABLE \d+\n(.*?)\n% END MAIN TABLE \d+', latex, re.DOTALL)
    section = next(s for s in sections if rf'\label{{{label}}}' in s)
    return '\n'.join(line for line in section.split('\\midrule', 1)[1].splitlines() if ' & ' in line)


def main_table_section(latex, label):
    """The whole main float with this label."""
    sections = re.findall(r'% BEGIN MAIN TABLE \d+\n(.*?)\n% END MAIN TABLE \d+', latex, re.DOTALL)
    return next(s for s in sections if rf'\label{{{label}}}' in s)


def ablation_rows(latex):
    """Ablation cells keyed by variant name (the backbone is named in its first row only)."""
    return {cells(row)[1]: cells(row)[2:] for row in main_table(latex, 'tab:main-ablations').splitlines()}


def main_rows(latex, index):
    return {cells(row)[0]: cells(row)[1:] for row in main_bodies(latex)[index].splitlines()}


def test_main_ultraquery_table_reports_seed_mean_sd_and_marks_best_and_second():
    reports = tables.Reports()
    reports.consume(suite_runs('ultraquery', 'ultraquery', .2) + suite_runs('ultra-r', 'ultra-adapter', .3)
                    + suite_runs('ultra-r-seed1', 'ultra-adapter', .4)
                    + suite_runs('ultra-r', 'ultra-adapter', .25, identity_of='ultra-r')
                    + suite_runs('ultra-r-seed1', 'ultra-adapter', .25, identity_of='ultra-r-seed1'))
    latex = tables.render_tables(reports)
    rows = main_rows(latex, 0)
    assert rows['UltraQuery'] == ['20.0'] * 8
    # Mean .35 with sample s.d. .0707 over two seeds; the control's seed replicates are one run.
    assert rows['ULTRA + adapter (ours)'] == [r'\textbf{35.0}$_{\pm 7.1}$'] * 8
    assert rows['ULTRA (raw scores)'] == [r'\underline{25.0}'] * 8
    assert 'over 2 adapter training seeds' in latex and 'Full test splits.' in latex
    assert 'only adapter rows vary by seed' in latex


def test_main_table_cells_use_only_seeds_with_every_dataset():
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-r', 'ultra-adapter', .3)
                    + suite_runs('ultra-r-seed1', 'ultra-adapter', .4, datasets=ULTRAQUERY_DATASETS[1:]))
    rows = main_rows(tables.render_tables(reports), 0)['ULTRA + adapter (ours)']
    families = [f for f, _ in tables.FAMILIES.items()]
    missing = tables.family(ULTRAQUERY_DATASETS[0])
    for index, family in enumerate(families):
        for cell_value in rows[2 * index:2 * index + 2]:
            if family in (missing, 'all'):
                assert cell_value == r'\textbf{30.0}'  # seed 1 lacks a dataset of this cell
            else:
                assert cell_value == r'\textbf{35.0}$_{\pm 7.1}$'


def test_main_ultraquery_table_merges_the_per_graph_baselines_into_one_row():
    scores = {'transductive': ('qto', .4), 'inductive-e': ('inductive-gnnqe', .3), 'inductive-er': ('incoming-relation', .01)}
    runs = [result(d, *scores[tables.family(d)]) for d in ULTRAQUERY_DATASETS]
    reports = tables.Reports()
    reports.consume(runs + suite_runs('ultraquery', 'ultraquery', .2))
    latex = tables.render_tables(reports)
    rows = main_rows(latex, 0)
    assert not {'QTO', 'Inductive GNN-QE', 'Incoming relation'} & set(rows)
    # Each family takes its own baseline; All averages the mix over the 23 datasets: (3 * 40 + 9 * 30 + 11 * 1) / 23.
    assert rows[r'Best per-graph baseline$^\dagger$'] == [r'\textbf{40.0}'] * 2 + [r'\textbf{30.0}'] * 2 + [r'\underline{1.0}'] * 2 \
        + [r'\underline{17.4}'] * 2
    assert 'The first row combines baselines specific to each target graph.' in latex
    assert r'$^\dagger$QTO on the transductive datasets' in latex and r'as in Galkin et al.\ (2024)' in latex
    assert 'The All columns average them over the 23 datasets' in latex
    appendix = latex[latex.index(r'\section*{Appendix}'):]
    assert ' & QTO & ' in appendix and ' & Incoming relation & ' in appendix
    assert r'as in \citet{galkin2024foundation}' in tables.render_thesis(reports)['main-ultraquery.tex']
    # Without a baseline for every family, each method keeps its own row and no note is added.
    partial = tables.Reports()
    partial.consume([run for run in runs if run['inference']['method'] != 'incoming-relation'])
    latex = tables.render_tables(partial)
    assert {'QTO', 'Inductive GNN-QE'} <= set(main_rows(latex, 0)) and 'per-graph' not in latex


def test_hardness_appendix_shows_bins_of_the_adapters_and_the_best_baseline_only():
    scores = (('clmpt', .3), ('qto', .2), ('ultra-adapter', .25))
    reports = tables.Reports()
    reports.consume([result('FB15k237+H', method, score) for method, score in scores])
    reports.consume([dict(dataset='FB15k237+H', method=method, entry=f'{method}-FB15k237+H', shape='3p',
                          grouping='inferred_positive_edges', label=str(k), comparison_graph='train+valid',
                          answer_filter='released', sort=dict(mrr=score / k))
                     for method, score in scores for k in (1, 2, 3)])
    rows = tables.hardness_matrix_rows(reports, 'sort')
    binned = {row[1] for row in rows if row[4].startswith('Missing links') and row[5] == 'MRR'}
    overall = {row[1] for row in rows if row[4].startswith('Overall') and row[5] == 'MRR'}
    # QTO trails CLMPT on the main table's EPFO MRR: it keeps its overall rows but not its bins.
    assert binned == {'CLMPT', 'ULTRA + adapter'} and {'CLMPT', 'QTO'} <= overall


def test_requested_primary_recipe_drives_main_and_seed_matched_ablations():
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-a', 'ultra-adapter', .3) + suite_runs('ultra-a-seed1', 'ultra-adapter', .4)
                    + suite_runs('ultra-b', 'ultra-adapter', .5))
    reports.primary_recipes = ('b',)
    latex = tables.render_tables(reports)
    assert main_rows(latex, 0)['ULTRA + adapter (ours)'][-1] == r'\textbf{50.0}'
    ablations = ablation_rows(latex)
    assert ablations['Primary recipe'] == ['50.0', r'\textemdash{}', '-', r'\textemdash{}']
    assert ablations['a'] == ['35.0$_{\\pm 7.1}$', '-15.0$_{\\pm 7.1}$', '-', '-']
    reports.primary_recipes = ('a',)
    ablations = ablation_rows(tables.render_tables(reports))
    assert ablations['Primary recipe'] == ['35.0$_{\\pm 7.1}$', r'\textemdash{}', '-', r'\textemdash{}']
    # Seed 0 of the variant is compared with seed 0 of the primary recipe: .5 - .3.
    assert ablations['b'] == ['50.0', '+20.0', '-', '-']
    reports.primary_recipes = ('missing',)
    with pytest.raises(ValueError, match='primary recipe'):
        tables.render_tables(reports)


def test_main_tables_share_ranks_for_displayed_ties_and_name_partial_scope():
    reports = tables.Reports()
    for method in ('qto', 'gnnqe', 'cqd'):
        reports.consume([result(d, method, {'qto': .3, 'gnnqe': .3004, 'cqd': .2}[method], complete=False)
                         for d in tables.PLUS_H_DATASETS])
    latex = tables.render_tables(reports)
    rows = main_rows(latex, 1)
    assert rows['QTO'][-1] == rows['GNN-QE'][-1] == r'\textbf{30.0}'
    assert rows['CQD'][-1] == r'\underline{20.0}'
    assert 'Partial evaluation (test split, query samples); not the full test splits.' in latex
    assert 'mean$_{\\pm' not in latex


def test_appendix_uses_the_main_tables_primary_recipe_without_mutating_reports():
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-a', 'ultra-adapter', .3) + suite_runs('ultra-b', 'ultra-adapter', .5))
    reports.primary_recipes = ('b',)
    latex = tables.render_tables(reports)
    assert main_rows(latex, 0)['ULTRA + adapter (ours)'][-1] == r'\textbf{50.0}'
    by_family = [cells(row) for row in paper_bodies(latex)[0].splitlines()]
    assert {row[3] for row in by_family if row[0].startswith('ULTRA')} == {'50.00'}
    assert {r['id'].split('-')[1] for r in tables.paper_records(reports)} == {'b'}
    assert not hasattr(reports, 'primary_recipe_names')


def test_baseline_condition_variants_share_one_main_row():
    reports = tables.Reports()
    for dataset in tables.PLUS_H_DATASETS:
        for suffix, graph, score in (('', 'train+valid', .3), ('-graph-train', 'train', .6)):
            raw = result(dataset, 'qto', score)
            raw['benchmark_run']['entry'] = f'qto-{dataset}{suffix}'
            raw['dataset_metadata']['inference_graph'] = graph
            raw['protocol']['answer_filter'] = 'corrected'
            reports.consume(raw)
    rows = main_rows(tables.render_tables(reports), 1)
    # One QTO row, from the target graph (training + validation facts).
    assert list(rows) == ['QTO'] and rows['QTO'][-2] == r'\textbf{30.0}'


def test_seed_replicates_are_labeled_by_seed_tag_in_the_appendix():
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-r', 'ultra-adapter', .3, datasets=['WikiTopicsQuery:art'])
                    + suite_runs('ultra-r-seed1', 'ultra-adapter', .4, datasets=['WikiTopicsQuery:art']))
    assert set(tables.run_labels(reports).values()) == {'ULTRA + adapter (seed tag 0)', 'ULTRA + adapter (seed tag 1)'}


@pytest.mark.parametrize('suite', ['ultraquery', 'plus_h'])
def test_public_seed_and_ablation_recipes_are_read_as_planned(suite):
    shipped = tables.shipped_recipes()
    entries = [e for name in ('kgfm_seeds.json', 'kgfm_ablations.json')
               for e in read_manifest(suite_directory(suite) / name)['entries']]
    systems = {summary.system_of(e) for e in entries}
    seeds = {seed for (method, _, recipe), seed in systems if recipe == shipped[suite, method]}
    assert seeds == {1, 2, 3, 4}
    tokens = {(method.split('-')[0], recipe.split('-', 1)[1]) for (method, _, recipe), _ in systems if recipe != shipped[suite, method]}
    assert tokens == set(summary.PLANNED_ABLATIONS) - {('ultra', 'beam256'), ('trix', 'beam256')}
    assert all(not e['adapter_ablation'] for e in entries)


def test_filter_companions_are_labeled_by_condition_and_captions_describe_the_recipe():
    corrected = suite_runs('ultra-product-intersections', 'ultra-adapter', .3, datasets=['FB15k237+H'])
    released = suite_runs('ultra-product-intersections', 'ultra-adapter', .2, datasets=['FB15k237+H'])
    corrected[0]['protocol']['answer_filter'] = 'corrected'
    released[0]['benchmark_run']['entry'] += '-released-filters'
    reports = tables.Reports()
    reports.consume(corrected + released)
    labels = set(tables.run_labels(reports).values())
    assert labels == {'ULTRA + adapter (2i/3i) (corrected filters)', 'ULTRA + adapter (2i/3i) (released filters)'}
    # Tables of one run per method name it plainly.
    assert [row[1] for row in tables.main_plus_h_rows(reports, 'sort')] == ['ULTRA + adapter']
    latex = tables.render_tables(reports)
    assert 'from the primary recipe of the same backbone (ULTRA: adapter trained on 2i/3i queries)' in latex
    assert 'FB15k-237+H shares its graph with backbone pretraining and the source queries of the adapters.' in latex
    # The command line keeps the corrected-filter run and sets its released twin aside.
    assert set(summary.corrected_filter_reports(reports).results) == {'ultra-product-intersections-FB15k237+H'}


def test_tables_listing_every_run_show_each_seed_replicate_once():
    reports = tables.Reports()
    art = ['WikiTopicsQuery:art']
    reports.consume(suite_runs('ultra-r', 'ultra-adapter', .3, datasets=art)
                    + suite_runs('ultra-r', 'ultra-adapter', .2, datasets=art, identity_of='ultra-r')
                    + suite_runs('ultra-r-seed1', 'ultra-adapter', .4, datasets=art))
    listed = summary.lowest_seed_reports(reports)
    assert set(listed.results) == {'ultra-r-WikiTopicsQuery:art', 'ultra-r-WikiTopicsQuery:art-without-adapter'}
    assert len(reports.results) == 3 and summary.lowest_seed_reports(listed) is listed
    bodies = paper_bodies(tables.render_tables(reports))
    full = [cells(row) for row in bodies[1].splitlines()]
    assert sorted(row[3] for row in full if row[2] == 'MRR') == ['20.00', '30.00']
    assert [cells(row)[1] for row in bodies[5].splitlines()] == ['0', '1']


def test_freebase_split_averages_each_group_separately():
    reports = tables.Reports()
    for dataset in ULTRAQUERY_DATASETS:
        reports.consume(suite_runs('ultraquery', 'ultraquery', .4 if summary.freebase_derived(dataset) else .2, datasets=[dataset]))
    assert sum(map(summary.freebase_derived, ULTRAQUERY_DATASETS)) == 11
    assert summary.freebase_rows(reports, 'sort') == [['UltraQuery', '40.00', '40.00', '20.00', '20.00']]


def test_two_backbones_with_custom_recipe_names_share_adapter_rows():
    reports = tables.Reports()
    for backbone in ('ultra', 'trix'):
        reports.consume(suite_runs(f'{backbone}-r', f'{backbone}-adapter', .3)
                        + suite_runs(f'{backbone}-r', f'{backbone}-adapter', .2, identity_of=f'{backbone}-r'))
    rows = [cells(row) for row in paper_bodies(tables.render_tables(reports))[4].splitlines()]
    assert len(rows) == 23 and all('(' not in row[0] for row in rows)
    assert all(row[1:4] == ['30.00', '20.00', '+10.00'] and row[5:8] == ['30.00', '20.00', '+10.00'] for row in rows)


def test_main_table_captions_and_notes_take_the_table_width():
    latex = tables.render_tables(tables.Reports())
    sections = re.findall(r'% BEGIN MAIN TABLE \d+\n(.*?)\n% END MAIN TABLE \d+', latex, re.DOTALL)
    # The thesis's main tables and the ablations, which the thesis moves to its appendix.
    assert len(sections) == len(tables.THESIS_MAIN_TABLES) + 1
    for section in sections:
        assert section.index(r'\begin{threeparttable}') < section.index(r'\caption{') < section.index(r'\end{threeparttable}')
    assert all(r'\begin{tablenotes}' in sections[i] for i in (2, 3))
    assert r'\usepackage{booktabs,longtable,array,threeparttable}' in latex


def test_preview_fills_planned_cells_with_placeholders_in_the_main_tables_only():
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-product-intersections', 'ultra-adapter', .3))
    latex = tables.render_tables(reports, preview=True)
    rows = main_rows(latex, 0)
    # The per-graph baselines are planned for their own families, so their merged row covers every column.
    assert rows[r'Best per-graph baseline$^\dagger$'] == ['xx.x'] * 8 and 'QTO' not in rows
    assert rows['UltraQuery'] == ['xx.x'] * 8
    assert rows['ULTRA + adapter (ours)'] == [r'\textbf{30.0}'] * 8
    assert rows['TRIX + adapter (ours)'] == [r'xx.x$_{\pm x.x}$'] * 8
    variants = ablation_rows(latex)
    assert variants['ULTRA 4g backbone$^\\dagger$'] == ['xx.x', '+x.x', 'xx.x', '+x.x']
    assert all('xx.x' not in body and 'QTO' not in body for body in paper_bodies(latex))
    assert 'xx.x' not in tables.render_tables(reports)


def test_thesis_tables_are_single_column_floats_that_leave_typesetting_to_the_thesis():
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-product-intersections', 'ultra-adapter', .3))
    files = tables.render_thesis(reports, preview=True)
    assert set(files) == {name + '.tex' for name in tables.THESIS_MAIN_TABLES} | {'settings.tex', 'appendix.tex', 'appendix-ablations.tex'}
    # The ablations float in the appendix.
    assert r'\begin{table}[tbp]' in files['appendix-ablations.tex'] and r'\label{tab:main-ablations}' in files['appendix-ablations.tex']
    for name in tables.THESIS_MAIN_TABLES:
        latex = files[name + '.tex']
        assert latex.count(r'\begin{table}[H]') == 1 and 'table*' not in latex
        # The caption keeps the page's caption column; the thesis fits the body to its reading column.
        assert latex.index(r'\caption{') < latex.index(r'\begin{fitblock}') < latex.index(r'\begin{tabular}')
        assert r'\footnotesize' not in latex and r'\tabcolsep' not in latex and r'\scriptsize' not in latex
    assert summary.PREVIEW_NOTE in files['main-ultraquery.tex'] and 'xx.x' in files['main-ultraquery.tex']
    assert summary.PREVIEW_NOTE not in tables.render_thesis(reports)['main-ultraquery.tex']
    appendix = files['appendix.tex']
    assert r'\section*' not in appendix and r'\thetable' not in appendix.split(r'\begin{longtable}')[0]
    assert r'\scriptsize' not in appendix and re.search(r'p\{\d+mm\}', appendix) is None
    assert r'\linewidth-2\tabcolsep' in appendix and r'\tablenote{' in appendix
    assert r'\ref{tab:benchmark-paper-7}' in appendix and ' A7' not in appendix


def test_thesis_option_writes_tables_and_figures_into_the_thesis_project(tmp_path):
    (tmp_path / 'thesis').mkdir()
    subprocess.run([*CLI, '--thesis', str(tmp_path / 'thesis'), '-o', str(tmp_path / 'tables.tex')],
                   cwd=REPO, check=True, capture_output=True)
    written = sorted(str(p.relative_to(tmp_path / 'thesis')) for p in (tmp_path / 'thesis').rglob('*') if p.is_file())
    assert [p for p in written if p.startswith(tables.THESIS_TABLES)] == sorted(
        f'{tables.THESIS_TABLES}/{name}.tex' for name in (*tables.THESIS_MAIN_TABLES, 'settings', 'appendix', 'appendix-ablations'))
    # Three main figures (calibration, hardness profiles, facts mechanism) and three appendix figures, each a PDF and a float.
    assert len([p for p in written if p.startswith('figures/cqa/')]) == 2 * 6
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([*CLI, '--thesis', str(tmp_path / 'missing'), '-o', str(tmp_path / 'tables.tex')],
                       cwd=REPO, check=True, capture_output=True)


def test_bracket_table_places_the_adapter_between_fixed_calibrations_and_target_fits():
    both = ULTRAQUERY_DATASETS + list(tables.PLUS_H_DATASETS)
    reports = tables.Reports()
    reports.consume(suite_runs('ultra-product-intersections', 'ultra-adapter', .3, datasets=both)
                    + suite_runs('ultra-product-intersections-seed1', 'ultra-adapter', .4, datasets=both)
                    + suite_runs('ultra-product-intersections', 'ultra-adapter', .2, datasets=both,
                                 identity_of='ultra-product-intersections')
                    + suite_runs('ultra-uqlp', 'ultra-adapter', .1)
                    + suite_runs('ultra-global', 'ultra-adapter', .25, datasets=both)
                    + suite_runs('ultra-target-fit', 'ultra-adapter', .36, datasets=both))
    latex = tables.render_tables(reports)
    rows = {cells(row)[1]: cells(row)[2:] for row in main_table(latex, 'tab:main-brackets').splitlines()}
    # By fitted parameters: raw scores, the published threshold, one global scale and shift, the adapter; below the rule,
    # target-fitted adapters and the test-selected oracle.
    assert list(rows) == ['Raw scores (sigmoid)', 'UltraQuery-LP threshold t, beam executor', 'Global scale and shift', 'Adapter (ours)',
                          'Adapter per target graph', 'Oracle: best zero-shot row per target and type']
    assert rows['Raw scores (sigmoid)'] == ['0', 'none'] + ['20.0'] * 4
    assert rows['Global scale and shift'] == ['2', 'source queries'] + ['25.0'] * 4
    # The thresholds exist for UQ-23 only.
    assert rows['UltraQuery-LP threshold t, beam executor'] == ['0', 'published t', '10.0', '10.0', '-', '-']
    assert rows['Adapter (ours)'] == ['16', 'source queries'] + [r'35.0$_{\pm 7.1}$'] * 4
    assert rows['Adapter per target graph'] == ['16 per target', r'target inference graph, 30\% facts masked'] + ['36.0'] * 4
    # The oracle takes, per target and type, the best zero-shot row (here the adapter's seed mean); target fits are no candidates.
    assert rows['Oracle: best zero-shot row per target and type'] == ['-', 'selected on test'] + ['35.0'] * 4
    section = main_table_section(latex, 'tab:main-brackets')
    # The target-fitted rows and the oracle sit below a dashed rule, defined in the float that uses it.
    assert r'\cqaDashedRule{7}' in section and section.index(r'\providecommand{\cqaDashedRule}') < section.index(r'\cqaDashedRule{7}')
    assert 'Rows below a dashed line use target data' in section
    assert 'applied in our beam executor' in latex and 'no target query or answer is used' in latex
    # Notes are LaTeX: a bare % would comment out the end of the table.
    section = next(s for s in re.findall(r'% BEGIN MAIN TABLE \d+\n(.*?)\n% END MAIN TABLE \d+', latex, re.DOTALL)
                   if r'\label{tab:main-brackets}' in s)
    assert not re.search(r'(?<!\\)%', section)
    assert r'30\% of the facts' in latex
    # The bracket runs are not ablations of the recipe.
    assert set(ablation_rows(latex)) == {'Primary recipe'}
    assert 'main-brackets.tex' in tables.render_thesis(reports)
    # The appendix tables that list every run leave the bracket runs to the bracket table.
    appendix = latex[latex.index(r'\section*{Appendix}'):]
    assert not any(token in appendix for token in ('ultra-global', 'ultra-uqlp', 'ultra-target-fit'))
    listed = [cells(row)[1].replace(r'\allowbreak{}', '') for row in paper_bodies(latex)[1].splitlines() if cells(row)[2] == 'MRR']
    assert listed.count('ULTRA + adapter') == len(ULTRAQUERY_DATASETS)  # the primary recipe only


def test_kgicl_follows_trix_in_the_main_tables_and_observed_facts_off_is_an_ablation():
    facts_off = suite_runs('kgicl-product-intersections', 'kgicl-adapter', .29)
    for raw in facts_off:
        raw['benchmark_run']['entry'] += '-facts-none'
    reports = tables.Reports()
    reports.consume(suite_runs('ultraquery', 'ultraquery', .2) + suite_runs('ultra-product-intersections', 'ultra-adapter', .3)
                    + suite_runs('trix-product-intersections', 'trix-adapter', .32)
                    + suite_runs('kgicl-product-intersections', 'kgicl-adapter', .31)
                    + suite_runs('kgicl-product-intersections', 'kgicl-adapter', .05,
                                 identity_of='kgicl-product-intersections') + facts_off)
    latex = tables.render_tables(reports)
    names = list(main_rows(latex, 0))
    assert names.index('TRIX + adapter (ours)') < names.index('KG-ICL (raw scores)') < names.index('KG-ICL + adapter (ours)')
    assert main_rows(latex, 0)['KG-ICL + adapter (ours)'][-1] == r'\underline{31.0}'
    # Observed facts off is an ablation of KG-ICL's recipe: seed 0 against seed 0 of the primary recipe.
    assert ablation_rows(latex)['Observed facts off'][:2] == ['29.0', '-2.0']
    assert "KG-ICL (the authors' 6-layer checkpoint" in latex
    assert 'facts-none' not in latex[latex.index(r'\section*{Appendix}'):]
    # The main UQ-23 table names the targets that are not zero-shot, for KG-ICL also NELL995.
    assert summary.ZERO_SHOT_NOTE in latex and summary.KGICL_OVERLAP_NOTE in latex
    assert 'Complex query answering on the 23 UQ-23 datasets' in latex and 'UltraQuery (23)' not in latex
    assert 'NELL995 shares data with the pretraining of KG-ICL.' in latex  # UQ-23 runs only
