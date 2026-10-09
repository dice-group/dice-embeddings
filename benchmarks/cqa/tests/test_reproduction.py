"""The final-study registry, the reproduction driver, adapter fits and the thesis analyses, on CPU without datasets."""

import json
import pickle
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest
import torch

from benchmarks.cqa import cli
from benchmarks.cqa.manifests import REPO
from benchmarks.cqa.reproduction import analyses, registry
from benchmarks.cqa.reproduction import cli as reproduction
from benchmarks.cqa.reproduction.registry import FITS, STUDIES, Study
from benchmarks.cqa.tests.test_protocol import paper_fixture, plus_h_fixture, use_ultraquery_fixture
from dicee.query_answering.context import fingerprint, state_fingerprint

PINNED = json.loads(Path(__file__).with_name('final_studies.json').read_text())['studies']
FROZEN = REPO / 'Experiments' / 'final-manifests'


@pytest.fixture(autouse=True)
def torch_settings():
    """Fits and analyses set threads and float32 precision for their process; tests restore them."""
    from dicee.models._inference import float32_precision_backends
    threads, precision = torch.get_num_threads(), [(backend, backend.fp32_precision) for backend in float32_precision_backends()]
    yield
    torch.set_num_threads(threads)
    for backend, value in precision:
        backend.fp32_precision = value


def shipped_sha256(path):
    return registry.sha256(REPO / path)


@pytest.mark.parametrize('study', STUDIES, ids=lambda study: study.id)
def test_registry_resolves_every_frozen_final_study(study):
    """Every scientific field of the frozen manifest, adapters by content; documentation and adapter paths may differ."""
    manifest = registry.resolved_manifest(study)
    pinned = PINNED[study.id]
    assert fingerprint({key: value for key, value in manifest.items() if key != 'entries'}) == pinned['manifest']
    assert [(e['id'], fingerprint(registry.scientific_entry(e, shipped_sha256))) for e in manifest['entries']] == list(pinned['entries'].items())
    assert len(manifest['entries']) == study.entries
    assert all(not entry_id.startswith(study.superseded) for entry_id in pinned['entries'])
    assert bool(pinned['superseded']) == bool(study.superseded)


@pytest.mark.skipif(not FROZEN.is_dir(), reason='needs the frozen manifests of the final runs (Experiments/final-manifests)')
def test_frozen_manifests_differ_only_where_documented():
    frozen_adapters = REPO / 'Experiments/final-adapters/adapters'

    def frozen_sha256(path):
        prefix = reproduction.SCREEN_ADAPTERS
        return registry.sha256(frozen_adapters / path.removeprefix(prefix) if path.startswith(prefix) else REPO / path)

    for study in STUDIES:
        result = registry.compare_manifest(study, json.loads((FROZEN / study.id / 'manifest.json').read_text()), frozen_sha256)
        assert result['scientific'], study.id
        # Bracket and KG-ICL adapters moved from the GPU host's Experiments/screens/adapters into benchmarks/adapters.
        assert bool(result['adapter_paths']) == (study.bracket is not None or 'kgicl' in study.name), study.id
        # The frozen KG-ICL entries carry the notes of the then provisional recipe; the tracked recipe pins the same adapters.
        assert bool(result['reference']) == ('kgicl' in study.name and study.bracket is None), study.id
        assert bool(result['superseded']) == bool(study.superseded), study.id


def test_every_adapter_of_the_final_studies_is_shipped_with_its_fit_recipe():
    used = {path for study in STUDIES for path in registry.study_adapters(study)}
    assert used == set(FITS) and all((REPO / registry.ADAPTER_ROOT / path).is_file() for path in used)
    # Seed replicates, ablations without a source and the global fit share the reference fit's score banks.
    assert {FITS[f'seeds/ultra_product_intersections_seed{seed}.json'].group for seed in (1, 2, 3, 4)} == {FITS['ultra_product_intersections.json'].group}
    assert FITS['ablations/ultra_4g_product_intersections.json'].group != FITS['ultra_product_intersections.json'].group
    assert len({spec.group for spec in FITS.values() if spec.kind == 'target'}) == 52
    assert registry.thresholds()['NELL995LogicalQuery'] == 0.97 and set(registry.thresholds().values()) == {0.8, 0.97}
    assert sum(s.hours for s in STUDIES if s.suite == 'ultraquery') == pytest.approx(414.3)
    assert sum(s.hours for s in STUDIES if s.suite == 'plus_h') == pytest.approx(54.2)


def test_study_selection_by_id_name_and_pattern():
    assert [s.id for s in registry.select_studies(['kgfm-b64'])] == ['ultraquery-kgfm-b64', 'plus_h-kgfm-b64']
    assert len(registry.select_studies(['plus_h-*'])) == 11 and len(registry.select_studies(None)) == len(STUDIES) == 24
    assert [s.id for s in registry.select_studies(['plus_h-qto', 'pergraph'])] == ['ultraquery-pergraph', 'plus_h-qto']
    with pytest.raises(ValueError, match='No study matches'):
        registry.select_studies(['uqlp-plus'])


def test_bracket_recipes_swap_adapters_and_drop_the_shared_control():
    study = next(s for s in STUDIES if s.id == 'ultraquery-kgfm-b64-uqlp')
    entries = registry.recipe_manifest(study, adapter_root='results/final/adapters')['entries']
    nell = next(e for e in entries if e['id'] == 'trix-uqlp-NELL995LogicalQuery')
    assert nell['adapters'] == {'product': 'results/final/adapters/brackets/trix/uqlp-threshold-0.97.json'}
    assert not nell['adapter_ablation'] and nell['finalized'] and nell['blockers'] == []
    assert registry.prepare_options(study) == ['--hardware-profile', 'h100']
    facts = next(s for s in STUDIES if s.id == 'plus_h-kgfm-b64-kgicl-facts-none')
    assert registry.prepare_options(facts) == ['--hardware-profile', 'h100', '--observed-facts', 'none']
    assert all(e['id'].endswith('-facts-none') and e['options']['observed_facts'] == 'none' for e in registry.resolved_manifest(facts)['entries'])


def lines(capsys):
    return [line for line in capsys.readouterr().out.splitlines() if line]


def test_dry_run_prints_every_stage_without_writing(tmp_path, capsys):
    root = tmp_path / 'final'
    cli.main(['reproduce', 'plus_h-kgfm-b64-global', '--input-root', str(tmp_path), '--output', str(root), '--dry-run',
              '--gpus', '0', '1', '--workers-per-gpu', '2', '--image', 'dicee/cqa:local'])
    printed = lines(capsys)
    study = root / 'plus_h-kgfm-b64-global'
    assert printed[0] == '# plus_h-kgfm-b64-global: 6 entries, 3.3 test hours measured'
    assert printed[1] == f'# write {study / "recipes.json"} (6 entries)'
    commands = [line.split()[3:5] for line in printed[2:]]
    assert commands == [['plus_h', 'prepare'], ['plus_h', 'verify'], ['plus_h', 'run'], ['plus_h', 'report']]
    assert printed[2].endswith(f'--manifests {study / "recipes.json"} --hardware-profile h100 --image dicee/cqa:local')
    assert '--gpus 0 1 --workers-per-gpu 2' in printed[3] and '--gpus 0 1 --workers-per-gpu 2' in printed[4]
    assert not root.exists()
    # --all adds the combined reports, the analyses and the tables, which a dry run assumes earlier stages produce.
    cli.main(['reproduce', '--all', '--input-root', str(tmp_path), '--output', str(root), '--dry-run'])
    printed = lines(capsys)
    assert f'# link 102 entry directories of 11 studies into {root / "plus_h-all"}' in printed
    assert sum(' analysis ' in line for line in printed) == 8 and not root.exists()
    paper = next(line for line in printed if ' benchmarks.cqa.paper ' in line).split()
    assert paper[3:16] == [str(path) for path in reproduction.saved_reports(root, {
        *(root / f'{suite}-all-reports' / f'{name}.json' for suite in ('ultraquery', 'plus_h') for name in reproduction.REPORTS),
        *(root / study / 'difficulty/inference-difficulty.json' for study in reproduction.DIFFICULTY)})]
    assert set(reproduction.DIFFICULTY) == {s.id for s in STUDIES if s.difficulty}
    # Baselines need upstream oracles; the dry run names the pinned checkouts.
    assert any(line.startswith('#   git clone https://github.com/bys0318/QTO UPSTREAM/QTO') for line in printed)
    # --fit refits the study's adapters first and freezes the recipes with them.
    cli.main(['reproduce', 'plus_h-kgfm-b64-global', '--fit', '--input-root', str(tmp_path), '--output', str(root), '--dry-run'])
    printed = lines(capsys)
    assert printed[0].split()[3:6] == ['fit', 'brackets/ultra/global.json', 'brackets/trix/global.json']
    assert f'--output {root / "adapters"}' in printed[0] and not root.exists()


def test_completed_stages_are_skipped_and_prepared_studies_must_match(tmp_path, capsys):
    root = tmp_path / 'final'
    study = next(s for s in STUDIES if s.id == 'plus_h-kgfm-b64-kgicl')
    directory = root / study.id
    (directory / 'bundle').mkdir(parents=True)
    (directory / 'bundle' / 'bundle.sha256.json').write_text('{}')
    (directory / 'manifest.json').write_text(json.dumps(registry.resolved_manifest(study)))
    (directory / 'verified-bundle').mkdir()
    (directory / 'verified-bundle' / 'bundle.json').write_text('{}')
    entry = registry.resolved_manifest(study)['entries'][0]['id']
    (directory / 'test' / entry).mkdir(parents=True)
    (directory / 'test' / entry / 'result.json').write_text(json.dumps({'queries': 3}))
    args = ['reproduce', study.id, '--input-root', str(tmp_path), '--output', str(root), '--dry-run']
    cli.main(args)
    commands = [line.split()[3:5] for line in lines(capsys)[1:]]
    assert commands == [['plus_h', 'run'], ['plus_h', 'report'], ['plus_h', 'difficulty-report']]
    (directory / 'manifest.json').write_text(json.dumps(registry.resolved_manifest(next(s for s in STUDIES if s.id == 'plus_h-kgfm-b64-kgicl-global'))))
    with pytest.raises(SystemExit):
        cli.main(args)
    assert 'prepared from other recipes' in capsys.readouterr().err


def test_oracles_are_exported_with_pinned_checkouts_under_a_temporary_name(tmp_path, monkeypatch):
    import subprocess
    study = next(s for s in STUDIES if s.id == 'plus_h-qto')
    context = reproduction.Context(tmp_path, tmp_path / 'final', upstream=tmp_path / 'upstream')
    commands = []

    def export(command, env):
        commands.append((command, env['PATH']))
        Path(command[command.index('--output') + 1]).write_bytes(b'oracle')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(reproduction.subprocess, 'run', export)
    reproduction.stage_oracles(study, context)
    references = tmp_path / 'final/plus_h-qto/references'
    assert sorted(p.name for p in references.iterdir()) == ['qto-FB15k237+H.pt', 'qto-ICEWS18+H.pt', 'qto-NELL995+H.pt']
    assert all(command[command.index('--upstream') + 1] == str(tmp_path / 'upstream/QTO') for command, _ in commands)
    assert all(path.startswith(str(Path(sys.executable).parent)) for _, path in commands)
    reproduction.stage_oracles(study, context)
    assert len(commands) == 3  # Existing oracles are kept.
    with pytest.raises(ValueError, match='git clone https://github.com/bys0318/QTO'):
        reproduction.stage_oracles(study, reproduction.Context(tmp_path, tmp_path / 'other'))


def test_missing_upstream_checkouts_stop_a_reproduction_before_any_work(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(['reproduce', 'plus_h-kgfm-b64-global', 'qto', '--input-root', str(tmp_path), '--output', str(tmp_path / 'final')])
    assert 'git clone https://github.com/bys0318/QTO UPSTREAM/QTO' in capsys.readouterr().err
    assert not (tmp_path / 'final').exists()


@pytest.mark.parametrize('args', [['reproduce'], ['reproduce', 'qto', '--all'], ['reproduce', 'qto', '--stages', 'fit'],
                                  ['reproduce', 'qto', '--gpus', '0', '0'], ['reproduce', 'qto', '--output', '/elsewhere'],
                                  ['reproduce', 'no-such-study'], ['fit', '--all'], ['fit', 'seeds/unknown.json', '--output', 'x'],
                                  ['analysis', 'calibration-profile', '--output', 'x.json'], ['render', '--reports', '/nonexistent']])
def test_invalid_reproduction_commands_fail_before_any_work(args, tmp_path):
    with pytest.raises(SystemExit):
        cli.main([*args, *(['--input-root', str(tmp_path)] if args[0] == 'reproduce' else [])])


def test_list_reports_entries_prerequisites_and_measured_hours(capsys):
    cli.main(['reproduce', '--list'])
    printed = lines(capsys)
    assert printed[-1].split()[-2:] == ['102', '54.2'] and any(line.startswith('ultraquery total') and line.split()[-2:] == ['736', '414.3'] for line in printed)
    pergraph = next(line for line in printed if line.startswith('ultraquery-pergraph'))
    assert '43 oracles, 9 trained checkpoints' in pergraph
    cli.main(['fit', '--list'])
    assert len(lines(capsys)) == len(FITS) == 78


def test_fit_runs_one_worker_per_score_bank_group(tmp_path, capsys):
    cli.main(['fit', '--all', '--output', str(tmp_path / 'adapters'), '--dry-run'])
    printed = lines(capsys)
    assert len(printed) == len({spec.group for spec in FITS.values()}) == 59
    ultra = next(line for line in printed if line.endswith(f'{tmp_path / "adapters" / "logs" / "ultra.log"} 2>&1'))
    assert ultra.split().index('fit') < ultra.split().index('ultra_product_intersections.json') < ultra.split().index('--worker')
    assert all(f'seeds/ultra_product_intersections_seed{seed}.json' in ultra for seed in (1, 2, 3, 4))


def test_fits_use_the_recorded_settings(tmp_path, monkeypatch):
    """Source and target fits call fit_query_adapter exactly as refit.py and target_fit.py did."""
    from benchmarks.cqa.reproduction import fit
    calls = []

    class Result:
        history = [{'epoch': 1}]

        def __init__(self, model, data, **settings):
            from dicee.query_answering import QueryScoreAdapter
            calls.append((data, settings))
            self.adapter = QueryScoreAdapter('context_scores', 1., bias_bound=8., metadata=dict(
                backbone_state_sha256=state_fingerprint(model), training=dict(selected_epoch=1, selected_validation_mrr=.5)))

    from dicee.models import ULTRA
    model = ULTRA(dict(num_entities=1, num_relations=1))
    monkeypatch.setattr(fit, 'load_backbone', lambda backbone, root, **kwargs: model)
    monkeypatch.setattr(fit, 'source_data', lambda name, root, cache: name)
    target = 'brackets/ultra/target-fit/WikiTopicsQuery-sci.json'
    recorded = json.loads((REPO / registry.ADAPTER_ROOT / target).read_text())['sources']['WikiTopicsQuery:sci']
    prepared = type('Prepared', (), dict(save=lambda self, path: None, train=(1,), metadata={'masked_fact_pairs': 2}, to_dict=lambda self: recorded))
    monkeypatch.setattr(fit, 'target_data', lambda dataset, root: prepared())
    monkeypatch.setattr(fit, 'fingerprint', lambda value: value)
    monkeypatch.setattr(fit, 'fit_query_adapter', Result)
    report = fit.fit('ablations/ultra_product_intersections_no_fb15k237.json', tmp_path, tmp_path / 'out', device='cpu')
    data, settings = calls[0]
    assert data == ['WN18RR', 'CoDExMedium'] and settings['training_sources'] == settings['validation_sources'] == ['WN18RR', 'CoDExMedium']
    assert settings | {'checkpoint_path': None, 'cache_dir': None} == dict(
        feature_mode='context_scores_v1', bias_bound=8.0, scale_bound=2.0, observed_mix=1., epochs=500, train_shapes=['2i', '3i'],
        validation_every=5, validation_shapes=['1p', '2p', '3p', '2i', '3i', '2in', '3in', 'inp', 'pin', 'pni'], early_stopping_patience=100,
        seed=2026090851, row_batch_size=8, training_device='cpu', device_cache_bytes=512 * 2**20, checkpoint_path=None, cache_dir=None,
        train_per_shape=140, training_sources=['WN18RR', 'CoDExMedium'], validation_sources=['WN18RR', 'CoDExMedium'],
        validation_device='cpu', training_cache_bytes=6144 * 2**20)
    assert report['adapter_sha256'] == registry.sha256(tmp_path / 'out/ablations/ultra_product_intersections_no_fb15k237.json')
    fit.fit(target, tmp_path, tmp_path / 'out', device='cpu')
    _, settings = calls[1]
    assert (settings['train_per_shape'], settings['training_sources'], settings['validation_device'], settings['training_cache_bytes'],
            settings['seed']) == (280, ['WikiTopicsQuery:sci'], 'cpu', 2048 * 2**20, 2026090851)
    assert fit.fit(target, tmp_path, tmp_path / 'out') and len(calls) == 2  # kept
    # Target data must be the data the shipped adapter was fitted on.
    prepared.to_dict = lambda self: 'other data'
    with pytest.raises(ValueError, match='differs from the data of the shipped adapter'):
        fit.fit('brackets/trix/target-fit/WikiTopicsQuery-sci.json', tmp_path, tmp_path / 'out', device='cpu')


@pytest.mark.skipif(not (REPO / 'KGs/UltraQuery/WikiTopics_QE/sci').is_dir(), reason='needs the UltraQuery WikiTopics data')
def test_target_fit_data_regenerates_the_shipped_adapters_data():
    from benchmarks.cqa.reproduction import fit
    data = fit.target_data('WikiTopicsQuery:sci', REPO)
    for backbone in ('ultra', 'trix'):
        shipped = json.loads((REPO / registry.ADAPTER_ROOT / f'brackets/{backbone}/target-fit/WikiTopicsQuery-sci.json').read_text())
        assert fingerprint(data.to_dict()) == shipped['sources']['WikiTopicsQuery:sci']


@pytest.mark.skipif(not (REPO / 'checkpoints/ultra_3g.pth').is_file() or not (REPO / 'checkpoints/trix/entity_prediction.pth').is_file(),
                    reason='needs the released ULTRA and TRIX checkpoints')
def test_threshold_adapters_rebuild_byte_identically(tmp_path):
    from benchmarks.cqa.reproduction import fit
    for path in (p for p, spec in FITS.items() if spec.kind == 'threshold'):
        fit.fit(path, REPO, tmp_path, device='cpu')
        assert (tmp_path / path).read_bytes() == (REPO / registry.ADAPTER_ROOT / path).read_bytes()


@pytest.mark.skipif(not (REPO / registry.SOURCES['WN18RR'][0]).is_file(), reason='needs the WN18RR training split')
def test_adapter_source_data_regenerates_the_pinned_queries(tmp_path):
    from benchmarks.cqa.reproduction import fit
    data = fit.source_data('WN18RR', REPO, tmp_path)
    assert (len(data.train), len(data.validation)) == (668, 224) and data.metadata['masked_fact_pairs'] == 26050
    with pytest.raises(ValueError, match='differs from the pinned'):
        bad = json.loads((tmp_path / 'WN18RR.json').read_text())
        bad['train'] = bad['train'][1:]
        (tmp_path / 'WN18RR.json').write_text(json.dumps(bad))
        fit.source_data('WN18RR', REPO, tmp_path)


def test_adapter_weights_match_the_five_seed_analysis():
    report = analyses.adapter_weights()
    shift = {b: {k: round(v['mean'], 2) for k, v in report[b]['shift'].items()} for b in ('ultra', 'trix')}
    assert [shift[b]['observed_tails'] for b in ('ultra', 'trix')] == [1.81, 1.90]
    assert [shift[b]['score_entropy'] for b in ('ultra', 'trix')] == [-1.15, -0.83]
    assert [round(report[b]['scale']['head_degree']['mean'], 2) for b in ('ultra', 'trix')] == [2.58, 1.67]


def test_observed_links_count_intersection_answers_with_observed_atoms(tmp_path):
    plus_h_fixture(tmp_path / 'KGs/query-benchmarks-plus-h', 'FB15k237+H')
    report = analyses.observed_links(tmp_path, names=['FB15k237+H'])['FB15k237+H']
    assert set(report) == set(analyses.INTERSECTIONS)
    assert all(row['queries'] == 1 and row['answers'] == 2 and 0 <= row['share'] <= 1 for row in report.values())


def test_answer_classes_follow_the_cheapest_grounding():
    full = {0: {0: {1, 2}}, 1: {1: {3}}, 2: {1: {3, 4}}, 3: {}, 4: {}}
    train = {0: {0: {1}}, 1: {1: set()}, 2: {1: {3}}}
    train = {h: {r: set(t) for r, t in rows.items()} for h, rows in train.items()}
    for h in full:
        train.setdefault(h, {})
    pairs = {(0, 1), (1, 0), (2, 3), (3, 2), (1, 3)}
    # 3 via 0-1 (observed) and 1-3 (missing, pair connected) beats 0-2 (missing) and 2-3 (observed); 4 needs both hops.
    assert analyses.answer_class('2p', (0, (0, 1)), frozenset({3, 4}), full, train, pairs) == {
        3: ('1 missing', 'hop 2', True), 4: ('2 missing', 'both', False)}
    assert analyses.answer_class('1p', (0, (0,)), frozenset({2, 1}), full, train, pairs) == {
        2: ('1 missing', 'hop 1', False), 1: ('1 missing', 'hop 1', True)}


def test_answer_classes_read_each_method_from_its_rank_trace(tmp_path):
    from benchmarks.cqa.difficulty import root_queries
    manifest = paper_fixture(tmp_path)
    use_ultraquery_fixture(manifest, tmp_path)
    (tmp_path / 'KGs').mkdir()
    (tmp_path / 'data').rename(tmp_path / 'KGs/UltraQuery')
    folder = tmp_path / 'KGs/UltraQuery/FB15k-237-betae'
    records = root_queries(folder, ['1p', '2p'])
    trace = tmp_path / 'final/ultraquery-kgfm-b64/test/ultra-product-intersections-FB15k237LogicalQuery/ranks.sqlite3'
    trace.parent.mkdir(parents=True)
    with closing(sqlite3.connect(trace)) as db:
        db.execute('CREATE TABLE queries(position INTEGER, query_id TEXT, shape TEXT, answers TEXT)')
        for i, (key, record) in enumerate(records.items()):
            db.execute('INSERT INTO queries VALUES (?, ?, ?, ?)', (i, key, record['shape'], json.dumps([[a, 2, 1, 1] for a in sorted(record['hard'])])))
        db.commit()
    report = analyses.answer_classes(tmp_path, tmp_path / 'final', names=['FB15k237LogicalQuery'])['datasets']['FB15k237LogicalQuery']
    assert report['traces']['ULTRA+ad'] == str(trace) and report['traces']['UltraQuery'] is None
    # The fixture's 2p hard answers have no grounding in its complete graph; their trace ranks stay visible as unclassified.
    assert [(row['shape'], row['missing'], row['answers']) for row in report['classes']] == [('1p', '1 missing', 2), ('2p', 'unclassified', 0)]
    assert all(row['mrr'] == {'ULTRA+ad': .5} for row in report['classes'])
    with pytest.raises(ValueError, match='transductive'):
        analyses.answer_classes(tmp_path, tmp_path / 'final', names=['WikiTopicsQuery:art'])


def test_pretraining_overlap_matches_original_identifiers(tmp_path, monkeypatch):
    plus_h_fixture(tmp_path / 'KGs/query-benchmarks-plus-h', 'FB15k237+H')
    folder = tmp_path / 'KGs/query-benchmarks-plus-h/iscqa-compl-benchmarks/new_benchmarks/FB15k-237+H'
    with (folder / 'id2rel.pkl').open('wb') as stream:
        pickle.dump({0: '+likes', 1: '-likes', 2: '+knows', 3: '-knows'}, stream)
    graph = tmp_path / 'train.txt'
    graph.write_text('0\tlikes\t5\n5\tknows\t6\n9\tknows\t9\n')  # Test links (0, 0, 5) and (5, 2, 6) by name.
    monkeypatch.setattr(analyses, 'SOURCES', {'G': ('train.txt', registry.sha256(graph), '')})
    report = analyses.pretraining_overlap(tmp_path, names=['FB15k237+H'])['datasets']['FB15k237+H']
    assert report['sizes'] == {'train': 2, 'valid': 2, 'test': 2}
    assert report['overlap']['G']['train'] == dict(train=0, valid=0, test=2, pretraining_in_target=2, shared_entities=3)
    graph.write_text('changed')
    with pytest.raises(ValueError, match='differs from the pinned'):
        analyses.pretraining_overlap(tmp_path, names=['FB15k237+H'])


def tiny_backbone(root):
    """A random ULTRA saved as the 3g checkpoint, and an adapter bound to it."""
    from dicee.models import ULTRA
    from dicee.query_answering.score_adapter import QueryScoreAdapter
    model = ULTRA(dict(num_entities=1, num_relations=1))
    (root / 'checkpoints').mkdir(parents=True)
    torch.save({'model': model.state_dict()}, root / registry.CHECKPOINTS['ultra'])
    (root / 'adapters').mkdir()
    QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=torch.randn(2, 8, generator=torch.Generator().manual_seed(4)).double() / 5,
                      metadata={'backbone_state_sha256': state_fingerprint(model)}).save(root / 'adapters/ultra_product_intersections.json')
    return model


def test_calibration_profile_and_negation_probe_on_a_tiny_backbone(tmp_path):
    tiny_backbone(tmp_path)
    plus_h_fixture(tmp_path / 'KGs/query-benchmarks-plus-h', 'FB15k237+H')
    profile = analyses.calibration_profile('ultra', tmp_path, names=['FB15k237+H'], atoms=5, device='cpu', adapter_root=tmp_path / 'adapters')
    row = profile['datasets']['FB15k237+H']
    assert row['atoms'] == 1 and row['entities'] == 8 and 0 <= row['with_adapter']['ece'] <= 1
    assert row['scale']['p10'] <= row['scale']['median'] <= row['scale']['p90']
    probe = analyses.negation_probe(tmp_path, backbone='ultra', names=['FB15k237+H'], device='cpu', adapter_root=tmp_path / 'adapters')
    row = probe['datasets']['FB15k237+H']
    assert {'2in_raw', '2in_adapted'} <= row.keys() and 0 <= row['2in_adapted']['mrr'] <= 1


def test_analysis_command_writes_its_report(tmp_path):
    cli.main(['analysis', 'adapter-weights', '--output', str(tmp_path / 'weights.json')])
    assert set(json.loads((tmp_path / 'weights.json').read_text())) == set(registry.BACKBONES)


def test_render_passes_the_saved_reports_in_render_order(tmp_path, monkeypatch, capsys):
    from benchmarks.cqa.paper import tables
    calls = []
    monkeypatch.setattr(tables, 'main', calls.append)
    for study in reproduction.DIFFICULTY[::-1]:
        (tmp_path / study / 'difficulty').mkdir(parents=True)
        (tmp_path / study / 'difficulty/inference-difficulty.json').write_text('{}')
    cli.main(['render', '--reports', str(tmp_path), '-o', str(tmp_path / 'tables.tex'), '--figures', str(tmp_path / 'figures')])
    assert calls == [[*(str(tmp_path / s / 'difficulty/inference-difficulty.json') for s in reproduction.DIFFICULTY),
                      '-o', str(tmp_path / 'tables.tex'), '--figures', str(tmp_path / 'figures')]]


def test_verify_runs_checks_in_parallel_workers(tmp_path, monkeypatch):
    from benchmarks.cqa.study import freeze
    study = tmp_path / 'study'
    manifest = paper_fixture(tmp_path)
    manifest['entries'].append(dict(manifest['entries'][0], id='second'))
    freeze(manifest, study / 'bundle', tmp_path)
    for name in ('cone-fixture', 'second'):
        (tmp_path / f'{name}.pt').touch()
    jobs = []

    def workers(batch, *, status, gpus, workers_per_gpu):
        jobs.extend(batch)
        for entry, _, command, _ in batch:
            cli.write_json(command[command.index('--output') + 1], dict(passed=True))
        return []

    monkeypatch.setattr(cli, 'run_workers', workers)
    cli.main(['plus_h', 'verify', '--input-root', str(tmp_path), '--output', str(study), '--references', str(tmp_path),
              '--gpus', '0', '1', '--workers-per-gpu', '2'])
    assert [(entry, method, log) for entry, method, _, log in jobs] == [
        ('cone-fixture', 'cone', study / 'verification/cone-fixture.log'), ('second', 'cone', study / 'verification/second.log')]
    assert (study / 'verified-bundle/bundle.json').is_file()


def test_reproduce_one_study_end_to_end_and_resume(tmp_path, monkeypatch, capsys):
    """manifest, prepare, verify, run, report and combine of a one-entry KGFM study on CPU; a second run does nothing."""
    from dicee.models import ULTRA
    from dicee.query_answering.score_adapter import QueryScoreAdapter
    manifest = paper_fixture(tmp_path)
    use_ultraquery_fixture(manifest, tmp_path)
    model = ULTRA(dict(num_entities=1, num_relations=1))
    torch.save({'model': model.state_dict()}, tmp_path / 'kgfm.pt')
    QueryScoreAdapter('context_scores', 1., bias_bound=8., weights=torch.randn(2, 8, generator=torch.Generator().manual_seed(4)).double() / 5,
                      metadata={'backbone_state_sha256': state_fingerprint(model)}).save(tmp_path / 'adapter.json')
    entry = manifest['entries'][0]
    entry.update(id='ultra-tiny', method='ultra-adapter', checkpoint='kgfm.pt', adapters={'product': 'adapter.json'},
                 options={'beam_size': 2, 'row_batch_size': 1, 'backend_batch_size': 1, 'cache_bytes': 4096, 'raw_cache_bytes': 4096,
                          'raw_cache_device': 'model', 'relation_cache_mb': 1, 'projection_cache_mb': 1},
                 operators={shape: 'product' for shape in entry['query_types']}, selection_protocol='source-validation', finalized=True,
                 adapter_ablation=True, query_batch_size=4, query_order='relation')
    (tmp_path / 'suite').mkdir()
    (tmp_path / 'suite/tiny.json').write_text(json.dumps(manifest))
    tiny = Study('ultraquery', 'tiny', ('tiny.json',), entries=1, hours=0., hardware_profile='default')
    monkeypatch.setattr(registry, 'suite_directory', lambda suite: tmp_path / 'suite')
    monkeypatch.setattr(registry, 'STUDIES', (tiny,))
    monkeypatch.setattr(reproduction, 'STUDIES', (tiny,))
    args = ['reproduce', '--all', '--input-root', str(tmp_path), '--output', str(tmp_path / 'final'), '--device', 'cpu', '--report-workers', '1',
            '--stages', 'manifest', 'prepare', 'verify', 'run', 'report', 'combine']
    cli.main(args)
    root = tmp_path / 'final'
    result = json.loads((root / 'ultraquery-tiny/test/ultra-tiny/result.json').read_text())
    assert result['queries'] == 14 and (root / 'ultraquery-tiny/test/ultra-tiny/without-adapter/result.json').is_file()
    assert json.loads((root / 'ultraquery-tiny/manifest.json').read_text()) == registry.resolved_manifest(tiny)
    assert json.loads((root / 'ultraquery-all/sources.json').read_text()) == {'ultraquery-tiny': ['ultra-tiny']}
    rows = json.loads((root / 'ultraquery-all-reports/comparison.json').read_text())['rows']
    assert sorted(row['id'] for row in rows) == ['ultra-tiny', 'ultra-tiny-without-adapter']
    capsys.readouterr()
    cli.main(args)
    assert lines(capsys) == ['# ultraquery-tiny: 1 entries, 0.0 test hours measured']
