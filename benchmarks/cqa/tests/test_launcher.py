"""Portable preparation profiles and launcher checks without Docker or inference."""
import json
import subprocess

import pytest

from benchmarks.cqa import cli
from benchmarks.cqa.manifests import prepare_manifest, read_manifest, suite_directory, validate_entry
from benchmarks.cqa.study import freeze, read, run_job
from benchmarks.cqa.tests.test_protocol import paper_fixture

PUBLIC = suite_directory('plus_h')
IMAGE = ['--image', 'example/plus-h:local', '--dry-run']


@pytest.mark.parametrize('suite', ['plus_h', 'ultraquery'])
def test_h100_profile_preserves_protocol_and_records_execution_options(suite):
    paths = [suite_directory(suite) / name for name in ('baselines.json', 'kgfm_adapters.json')]
    settings = dict(suite=suite, answer_filter='corrected' if suite == 'plus_h' else 'released')
    baseline = prepare_manifest(paths, **settings)
    h100 = prepare_manifest(paths, hardware_profile='h100', **settings)
    for original, changed in zip(baseline['entries'], h100['entries']):
        validate_entry(changed)
        for key in original.keys() - {'options', 'reference'}:
            assert changed[key] == original[key]
        options = changed['options']
        if original['method'].endswith('-adapter'):
            assert options['raw_cache_device'] == 'model'
            assert options['row_batch_size'] == options['backend_batch_size'] == 16
            assert options['cache_bytes'] == 4 * 2**30 and options['raw_cache_bytes'] == 2 * 2**30
            assert options['beam_size'] == original['options']['beam_size']
            assert options['relation_cache_mb'] == 256 and options['projection_cache_mb'] == 512
        else:
            assert changed == original
    override = prepare_manifest(paths, hardware_profile='h100', kgfm_batch_size=8, **settings)
    assert all(e['options']['backend_batch_size'] == e['options']['row_batch_size'] == 8
               for e in override['entries'] if e['method'].endswith('-adapter'))
    assert prepare_manifest(paths, **settings) == baseline


@pytest.mark.parametrize('settings', [dict(hardware_profile='bad'), dict(kgfm_batch_size=0),
                                     dict(kgfm_batch_size=True), dict(kgfm_batch_size=1, methods=['qto'])])
def test_invalid_hardware_settings_fail_before_inputs(settings):
    with pytest.raises(ValueError):
        prepare_manifest([PUBLIC / 'baselines.json'], **settings)


@pytest.mark.parametrize('suite', ['plus_h', 'ultraquery'])
def test_hardware_profile_cli_and_container_forwarding(tmp_path, capsys, suite):
    output = tmp_path / 'run'
    cli.main([suite, 'evaluate', '--split', 'test', '--hardware-profile', 'h100', '--kgfm-batch-size', '16', '--methods', 'ultra-adapter',
              '--output', str(output), '--dry-run'])
    entries = json.loads(capsys.readouterr().out)['entries']
    assert all(e['options']['backend_batch_size'] == 16 and e['options']['raw_cache_device'] == 'model' for e in entries)
    common = ['--input-root', str(tmp_path), '--output', str(output), *IMAGE]
    flags = ['--hardware-profile', 'h100', '--kgfm-batch-size', '16']
    cli.main([suite, 'prepare', *common, *flags])
    assert '--hardware-profile h100 --kgfm-batch-size 16' in capsys.readouterr().out
    assert not output.exists()
    # Frozen bundles fix the execution options; a run cannot change them.
    with pytest.raises(SystemExit):
        cli.main([suite, 'run', *common, *flags])
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_profiles_preserve_beams_and_per_shape_settings():
    paths = [PUBLIC / name for name in ('baselines.json', 'kgfm_adapters.json')]
    reference = prepare_manifest(paths, profile='reference')
    bounded = prepare_manifest(paths, profile='bounded')
    original = sum((read_manifest(path)['entries'] for path in paths), [])
    assert reference['entries'] == original
    assert reference['answer_filter'] == bounded['answer_filter'] == 'corrected'
    assert prepare_manifest(paths, answer_filter='released')['answer_filter'] == 'released'
    assert len(bounded['entries']) == len(reference['entries'])
    for expected, actual in zip(reference['entries'], bounded['entries']):
        if expected['method'] not in ('cqd', 'cqd-hybrid'):
            assert actual == expected
            continue
        assert actual['options']['reference_batching'] is False
        assert actual['options']['row_batch_size'] == actual['options']['final_batch_size'] == 32
        for key, value in expected['options'].items():
            if key not in ('reference_batching', 'row_batch_size', 'final_batch_size'):
                assert actual['options'][key] == value
        assert actual['query_types'] == expected['query_types']
        assert actual['query_batch_size'] == expected['query_batch_size']
        assert 'bounded' in actual['reference']['status']
    selected = prepare_manifest(paths, entries=[original[0]['id']])
    assert [entry['id'] for entry in selected['entries']] == [original[0]['id']]
    with pytest.raises(ValueError, match='distinct'):
        prepare_manifest(paths, entries=['missing'])


def test_method_dataset_query_combinations_preserve_recipes_and_trim_overrides():
    paths = [PUBLIC / name for name in ('baselines.json', 'kgfm_adapters.json')]
    original = prepare_manifest(paths, profile='reference')
    selected = prepare_manifest(paths, profile='reference', methods=['cqd', 'qto', 'ultra-adapter'],
                                datasets=['FB15k237+H', 'NELL995+H'], query_types=['epfo', '2in'], atomic_negation=True)
    assert len(selected['entries']) == 6
    assert {(e['method'], e['dataset']) for e in selected['entries']} == {
        (method, dataset) for method in ('cqd', 'qto', 'ultra-adapter') for dataset in ('FB15k237+H', 'NELL995+H')}
    for entry in selected['entries']:
        validate_entry(entry)
        assert len(entry['query_types']) == 12
        assert '2in' in entry['query_types'] and 'pni' not in entry['query_types']
        assert set(entry['options'].get('per_shape', {})) <= set(entry['query_types'])
        if 'operators' in entry:
            assert set(entry['operators']) == set(entry['query_types'])
        baseline = next(e for e in original['entries'] if e['id'] == entry['id'])
        assert entry['checkpoint'] == baseline['checkpoint']
        assert {k: v for k, v in entry['options'].items() if k not in ('atomic_negation', 'per_shape')} == {
            k: v for k, v in baseline['options'].items() if k not in ('atomic_negation', 'per_shape')}
    assert prepare_manifest(paths, profile='reference') == original
    assert prepare_manifest(paths, methods=['all'], datasets=['all']) == prepare_manifest(paths)


@pytest.mark.parametrize('selection', [dict(methods=['missing']), dict(datasets=['missing']),
                                      dict(methods=['cqd', 'cqd']), dict(query_types=['bad']),
                                      dict(query_types=['all', '2p']), dict(query_types=['2p', '2p']),
                                      dict(methods=['qto'], entries=['cqd-FB15k237+H'])])
def test_invalid_combinations_fail_before_opening_inputs(selection):
    with pytest.raises(ValueError):
        prepare_manifest([PUBLIC / 'baselines.json'], **selection)


@pytest.mark.parametrize('query_group', [None, 'all', 'epfo', 'negation'])
def test_evaluate_dry_run_resolves_default_cqd_coverage_without_inputs(tmp_path, capsys, query_group):
    from dicee.query_answering._query import PLUS_H_SHAPES
    output = tmp_path / 'no-results'
    args = ['plus_h', 'evaluate', '--split', 'test', '--methods', 'cqd', 'cqd-hybrid', '--input-root', str(tmp_path / 'missing-inputs'),
            '--output', str(output), '--dry-run']
    if query_group is not None:
        args += ['--query-types', query_group]
    cli.main(args)
    manifest = json.loads(capsys.readouterr().out)
    assert {(entry['method'], entry['dataset']) for entry in manifest['entries']} == {
        (method, dataset) for method in ('cqd', 'cqd-hybrid')
        for dataset in ('FB15k237+H', 'ICEWS18+H', 'NELL995+H')}
    expected = set(PLUS_H_SHAPES)
    if query_group in ('epfo', 'negation'):
        expected = {shape for shape in expected if ('n' in shape) == (query_group == 'negation')}
    for entry in manifest['entries']:
        validate_entry(entry)
        assert set(entry['query_types']) == expected
        assert entry['options']['atomic_negation']
        assert not entry['options']['reference_batching']
        assert set(entry['options']['per_shape']) <= expected
        assert any('signed-atom negation with +H scoring' in note for note in entry['reference']['notes'])
    assert not output.exists()


def test_direct_subset_evaluation_scores_only_selected_methods_datasets_and_queries(tmp_path, monkeypatch):
    import torch

    from dicee.models.complex import ComplEx
    from dicee.query_answering.methods.cqd import CQD

    manifest = paper_fixture(tmp_path)
    model = ComplEx(dict(model='ComplEx', num_entities=8, num_relations=4, embedding_dim=8, normalization=None))
    torch.save(model.state_dict(), tmp_path / 'complex.pt')
    for method in ('cqd', 'cqd-hybrid'):
        manifest['entries'].append(dict(manifest['entries'][0], id=f'{method}-FB15k237+H', method=method,
                                       checkpoint='complex.pt', options={'beam_size': 2, 'atomic_negation': True,
                                                                         'per_shape': {'2p': {'beam_size': 3}}}))
    manifest['entries'].append(dict(manifest['entries'][-1], id='excluded-dataset', dataset='NELL995+H', checkpoint='missing.pt'))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    output = tmp_path / 'results'
    args = ['plus_h', 'evaluate', '--split', 'test', '--manifests', str(path), '--answer-filter', 'released', '--methods', 'cqd', 'cqd-hybrid',
            '--datasets', 'FB15k237+H', '--query-types', 'negation', '--input-root', str(tmp_path), '--output', str(output),
            '--device', 'cpu', '--threads', '2']
    cli.main(args)
    assert sorted(p.name for p in output.iterdir()) == ['cqd-FB15k237+H', 'cqd-hybrid-FB15k237+H']
    for method in ('cqd', 'cqd-hybrid'):
        report_path = output / f'{method}-FB15k237+H' / 'result.json'
        report = json.loads(report_path.read_text())
        assert report['queries'] == 5
        assert set(report['per_shape']) == {'2in', '3in', 'inp', 'pin', 'pni'}
        assert report['inference']['configuration']['atomic_negation']
        assert report['inference']['negation_scope'] == 'CQD-A signed-atom negation with +H scoring'
    monkeypatch.setattr(CQD, 'predict', lambda *args: pytest.fail('Completed subset was recomputed'))
    cli.main(args)


def test_observed_fact_recipes_reuse_checkpoints_and_select_only_kgfms():
    paths = [PUBLIC / name for name in ('baselines.json', 'kgfm_adapters.json')]
    originals = {entry['id']: entry for entry in prepare_manifest(paths)['entries']}
    expanded = prepare_manifest(paths, observed_facts=['all'])['entries']
    assert len(expanded) == 18
    for entry in expanded:
        validate_entry(entry)
        mode = entry['options']['observed_facts']
        original = originals[entry['id'].removesuffix(f'-facts-{mode}')]
        assert entry['method'].endswith('-adapter')
        assert {k: v for k, v in entry.items() if k not in ('id', 'options', 'reference')} == {
            k: v for k, v in original.items() if k not in ('id', 'options', 'reference')}
        assert {k: v for k, v in entry['options'].items() if k != 'observed_facts'} == original['options']
    selected = prepare_manifest(paths, observed_facts=['none', 'atomic'])['entries']
    assert len(selected) == 12
    assert {entry['options']['observed_facts'] for entry in selected} == {'none', 'atomic'}
    one = prepare_manifest(paths, observed_facts=['all'], entries=[expanded[0]['id']])['entries']
    assert one == expanded[:1]
    for invalid in ([], ['all', 'none'], ['atomic', 'atomic'], ['invalid']):
        with pytest.raises(ValueError, match='observed-fact'):
            prepare_manifest(paths, observed_facts=invalid)
    one[0]['options']['observed_facts'] = 'invalid'
    with pytest.raises(ValueError, match='Observed facts'):
        validate_entry(one[0])


def test_graph_ablation_freezes_weights_and_reuses_graph_independent_methods():
    paths = [PUBLIC / name for name in ('baselines.json', 'kgfm_adapters.json')]
    original = {e['id']: e for e in prepare_manifest(paths)['entries']}
    expanded = prepare_manifest(paths, inference_graphs=['train', 'train+valid'])['entries']
    assert len(expanded) == 45
    assert sum(1 + int(e.get('adapter_ablation', False)) for e in expanded) == 57
    for entry in expanded:
        validate_entry(entry)
        if entry['method'] in ('cone', 'clmpt', 'cqd'):
            assert entry['id'] in original and entry['reference']['graph_independent']
        else:
            base = original[entry['graph_ablation']]
            assert entry['options'] == base['options']
            assert entry['checkpoint'] == base['checkpoint']
            assert entry.get('adapters') == base.get('adapters')
            assert entry.get('operators') == base.get('operators')
    for invalid in ([], ['train', 'train'], ['test']):
        with pytest.raises(ValueError, match='inference graphs'):
            prepare_manifest(paths, inference_graphs=invalid)


def test_public_test_graph_defaults_and_explicit_train_override():
    from dicee.query_answering.method_evaluation import GRAPH_INDEPENDENT_METHODS
    paths = [PUBLIC / name for name in ('baselines.json', 'kgfm_adapters.json')]
    defaults = prepare_manifest(paths)['entries']
    for entry in defaults:
        expected = 'train' if entry['method'] in GRAPH_INDEPENDENT_METHODS else 'train+valid'
        assert entry['inference_graph'] == expected
        if entry['method'] in ('gnnqe', 'ultraquery', 'qto'):
            assert entry['reference']['inference_graph'] == 'train'
    overridden = prepare_manifest(paths, inference_graphs=['train'])['entries']
    assert len(overridden) == len(defaults)
    assert all(entry['inference_graph'] == 'train' for entry in overridden)


def test_container_dry_run_forwards_selection_and_rejects_bad_arguments(tmp_path, capsys):
    output = tmp_path / 'run'
    common = ['--input-root', str(tmp_path), '--output', str(output), *IMAGE]

    def launched(command, *extra):
        cli.main(['plus_h', command, *common, *extra])
        return capsys.readouterr().out

    result = launched('prepare', '--entries', 'cqd-FB15k237+H')
    assert result.startswith('docker run --rm --network none --read-only')
    assert f'source={tmp_path},target=/inputs,readonly' in result and f'source={output},target=/results' in result
    assert 'example/plus-h:local plus_h prepare --input-root /inputs --output /results --entries cqd-FB15k237+H --no-setup' in result
    assert not output.exists()
    assert '--observed-facts none atomic' in launched('prepare', '--observed-facts', 'none', '--observed-facts', 'atomic')
    assert '--inference-graphs train train+valid' in launched('prepare', '--inference-graphs', 'train', '--inference-graphs', 'train+valid')
    assert '--answer-filter released' in launched('prepare', '--answer-filter', 'released')
    for extra in (['--profile', 'invalid'], ['--output', str(tmp_path.parent / 'outside')], ['--unknown', 'x']):
        with pytest.raises(SystemExit):
            launched('prepare', *extra)
    assert 'plus_h run --input-root /inputs --output /results --no-setup' in launched('run')
    selected = launched('run', '--entries', 'ultra-product-14type-FB15k237+H-facts-none', '--entries', 'trix-product-14type-NELL995+H-facts-atomic')
    assert '--entries ultra-product-14type-FB15k237+H-facts-none trix-product-14type-NELL995+H-facts-atomic' in selected
    for extra in (['--answer-filter', 'corrected'], ['--observed-facts', 'none']):
        with pytest.raises(SystemExit):
            launched('run', *extra)


def test_container_forwards_combination_selectors_and_keeps_queries_frozen(tmp_path, capsys):
    common = ['--input-root', str(tmp_path), '--output', str(tmp_path / 'run'), *IMAGE]

    def launched(command, *extra):
        cli.main(['plus_h', command, *common, *extra])
        return capsys.readouterr().out

    result = launched('evaluate', '--methods', 'cqd', 'cqd-hybrid', '--datasets', 'FB15k237+H', 'NELL995+H', '--query-types', 'negation',
                      '--atomic-negation', '--split', 'valid', '--max-queries-per-shape', '3')
    assert '--methods cqd cqd-hybrid --datasets FB15k237+H NELL995+H --query-types negation --atomic-negation' in result
    assert '--split valid --max-queries-per-shape 3' in result
    assert not (tmp_path / 'run').exists()
    assert '--methods qto --datasets ICEWS18+H --query-types 2p 3p' in launched('prepare', '--methods', 'qto', '--datasets', 'ICEWS18+H',
                                                                                 '--query-types', '2p', '3p')
    assert '--methods qto --datasets ICEWS18+H' in launched('run', '--methods', 'qto', '--datasets', 'ICEWS18+H')
    for command, extra in [('run', ['--query-types', '2p']), ('run', ['--atomic-negation']), ('run', ['--split', 'valid']),
                           ('evaluate', ['--split', 'valid', '--methods']), ('evaluate', ['--split', 'valid', '--max-queries-per-shape', '0']),
                           ('evaluate', ['--methods', 'qto'])]:
        with pytest.raises(SystemExit):
            launched(command, *extra)


class _FakeWorkers:
    """Popen stand-in: each worker stays running for one poll, then exits."""

    def __init__(self, failing=()):
        self.started, self.active, self.peak, self.failing = [], {}, {}, set(failing)
        outer = self

        class Worker:
            pid = 7

            def __init__(self, command, *, env, **kwargs):
                self.entry, self.gpu, self.returncode, self.polls = command[0], env['CUDA_VISIBLE_DEVICES'], None, 0
                outer.started.append((self.entry, self.gpu))
                outer.active[self.gpu] = outer.active.get(self.gpu, 0) + 1
                outer.peak[self.gpu] = max(outer.peak.get(self.gpu, 0), outer.active[self.gpu])

            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                if self.returncode is None:
                    outer.active[self.gpu] -= 1
                    self.returncode = int(self.entry in outer.failing)
                return self.returncode

        self.module = __import__('types').SimpleNamespace(Popen=Worker, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)


def test_parallel_workers_order_long_entries_first_and_bound_each_gpu(tmp_path, monkeypatch):
    fake = _FakeWorkers()
    monkeypatch.setattr(cli, 'subprocess', fake.module)
    monkeypatch.setattr(cli.time, 'sleep', lambda seconds: None)
    methods = ['cone', 'ultraquery', 'trix-adapter', 'cqd', 'ultra-adapter', 'gnnqe', 'clmpt']
    jobs = [(f'e{i}-{method}', method, [f'e{i}-{method}'], tmp_path / f'{i}.log') for i, method in enumerate(methods)]
    assert cli.run_workers(jobs, gpus=['0', 'GPU-a'], workers_per_gpu=2, status=tmp_path / 'status.json') == []
    order = [entry.split('-', 1)[1] for entry, _ in fake.started]
    assert order[:4] == ['trix-adapter', 'ultra-adapter', 'gnnqe', 'cqd'] and sorted(order) == sorted(methods)
    assert {gpu for _, gpu in fake.started} == {'0', 'GPU-a'} and max(fake.peak.values()) == 2
    assert json.loads((tmp_path / 'status.json').read_text())['running'] == {}


def test_parallel_failure_stops_new_entries_but_lets_running_workers_finish(tmp_path, monkeypatch):
    fake = _FakeWorkers(failing={'a'})
    monkeypatch.setattr(cli, 'subprocess', fake.module)
    monkeypatch.setattr(cli.time, 'sleep', lambda seconds: None)
    jobs = [(name, 'cone', [name], tmp_path / f'{name}.log') for name in ('a', 'b', 'c')]
    assert cli.run_workers(jobs, gpus=['0', '1'], workers_per_gpu=1, status=tmp_path / 'status.json') == [('a', 1)]
    assert [entry for entry, _ in fake.started] == ['a', 'b']
    assert json.loads((tmp_path / 'status.json').read_text())['failed'] == ['a']


def pilot_study(tmp_path):
    """A frozen two-entry study and the fake worker processes that run its pilot in this process."""
    from types import SimpleNamespace
    manifest = paper_fixture(tmp_path)
    manifest['entries'].append(dict(manifest['entries'][0], id='second'))
    study = tmp_path / 'study'
    freeze(manifest, study / 'bundle', tmp_path)
    seen = []

    class Worker:
        pid, returncode = 1, 0

        def __init__(self, command, *, env, **kwargs):
            entry = command[command.index('--entries') + 1]
            device = command[command.index('--device') + 1] if '--device' in command else 'cuda'
            seen.append((entry, env.get('CUDA_VISIBLE_DEVICES'), device))
            run_job(study / 'bundle', entry, tmp_path, study / 'pilot' / entry)

        def poll(self):
            return 0

    workers = SimpleNamespace(Popen=Worker, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)
    return study, ['plus_h', 'run', '--phase', 'pilot', '--input-root', str(tmp_path), '--output', str(study)], workers, seen


def test_parallel_run_uses_fresh_workers_and_resumes_like_sequential_runs(tmp_path, monkeypatch):
    study, args, workers, seen = pilot_study(tmp_path)
    monkeypatch.setattr(cli, 'subprocess', workers)
    args += ['--device', 'cuda']
    cli.main(args + ['--gpus', '3', '5'])
    assert sorted(seen) == [('cone-fixture', '3', 'cuda'), ('second', '5', 'cuda')]
    assert read(study / 'pilot/status.json') == dict(state='complete', entries=['cone-fixture', 'second'])
    stamp = (study / 'pilot/second/result.json').stat().st_mtime_ns
    cli.main(args + ['--gpus', '0', '--workers-per-gpu', '2'])
    assert (study / 'pilot/second/result.json').stat().st_mtime_ns == stamp
    for invalid in (['--gpus', '0', '0'], ['--gpus', '0', '--workers-per-gpu', '0'], ['--workers-per-gpu', '2']):
        with pytest.raises(SystemExit):
            cli.main(args + invalid)
    with pytest.raises(SystemExit):
        cli.main(args[:-2] + ['--device', 'cpu', '--gpus', '0'])


def test_container_forwards_parallel_gpus_only_for_evaluate_and_run(tmp_path, capsys):
    common = ['--input-root', str(tmp_path), '--output', str(tmp_path / 'out'), *IMAGE]
    cli.main(['plus_h', 'run', *common, '--gpus', '0', '1', 'GPU-2c', '--workers-per-gpu', '2'])
    assert '--gpus 0 1 GPU-2c --workers-per-gpu 2' in capsys.readouterr().out
    cli.main(['plus_h', 'evaluate', *common, '--split', 'test', '--gpus', '1'])
    assert '--gpus 1' in capsys.readouterr().out
    for command, extra in [('prepare', ['--gpus', '0']), ('run', ['--workers-per-gpu', '2']), ('run', ['--gpus'])]:
        with pytest.raises(SystemExit):
            cli.main(['plus_h', command, *common, *extra])


def test_run_subsets_share_lock_and_resume_without_recomputing(tmp_path, monkeypatch, capsys):
    from dicee.query_answering._checkpoint import BenchmarkCheckpoint, checksum
    study, args, workers, seen = pilot_study(tmp_path)
    monkeypatch.setattr(cli, 'subprocess', workers)
    args += ['--device', 'cpu']
    cli.main(args + ['--entries', 'cone-fixture'])
    first_result = study / 'pilot/cone-fixture/result.json'
    stamp = first_result.stat().st_mtime_ns
    assert not (study / 'pilot/second/result.json').exists()
    cli.main(args + ['--entries', 'second'])
    cli.main(args)
    assert first_result.stat().st_mtime_ns == stamp
    assert read(study / 'pilot/status.json')['entries'] == ['cone-fixture', 'second']
    identity = dict(bundle=checksum(study / 'bundle/bundle.json'), phase='pilot', device='cpu', pilot_queries=2)
    with BenchmarkCheckpoint(study / 'pilot/suite', identity):
        with pytest.raises(RuntimeError, match='already running'):
            cli.main(args + ['--entries', 'second'])
    assert [entry for entry, _, _ in seen] == ['cone-fixture', 'second', 'cone-fixture', 'second']
    with pytest.raises(SystemExit):
        cli.main(args + ['--pilot-queries', '3'])
    assert 'settings changed' in capsys.readouterr().err


def test_verification_runs_each_entry_in_a_fresh_process_and_requires_oracles(tmp_path, monkeypatch, capsys):
    study = tmp_path / 'study'
    manifest = paper_fixture(tmp_path)
    manifest['entries'].append(dict(manifest['entries'][0], id='cqd-example', method='cqd', options={'atomic_negation': True}))
    freeze(manifest, study / 'bundle', tmp_path)
    references, comparisons = tmp_path / 'references', tmp_path / 'dense'
    references.mkdir()
    comparisons.mkdir()
    args = ['plus_h', 'verify', '--input-root', str(tmp_path), '--output', str(study), '--references', str(references),
            '--comparison-references', str(comparisons)]

    def rejected(message):
        with pytest.raises(SystemExit):
            cli.main(args)
        assert message in capsys.readouterr().err

    rejected('Missing baseline oracle cone-fixture.pt')
    for name in ('cone-fixture', 'cqd-example'):
        (references / f'{name}.pt').touch()
    rejected('Missing comparison oracle')
    (comparisons / 'cqd-example.pt').touch()
    commands = []

    def parity(command):
        commands.append(command)
        evidence = command[command.index('--output') + 1]
        cli.write_json(evidence, dict(passed=True))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, 'run', parity)
    cli.main(args)
    assert [command[command.index('--entry') + 1] for command in commands] == ['cone-fixture', 'cqd-example']
    assert all(command[5] == 'parity' for command in commands)
    assert '--comparison-reference' in commands[1] and '--comparison-reference' not in commands[0]
    verified = read(study / 'verified-manifest.json')
    assert [entry['verification'] for entry in verified['entries']] == ['study/evidence/cone-fixture.json', 'study/evidence/cqd-example.json']
    assert (study / 'verified-bundle/bundle.json').is_file()


def test_container_rejects_retagged_image_before_launch(tmp_path, monkeypatch):
    from benchmarks.cqa import docker
    (tmp_path / 'container-image.json').write_text(json.dumps({'image_id': 'sha256:original'}))
    monkeypatch.setattr(docker.subprocess, 'check_output', lambda *a, **k: json.dumps([{'Id': 'sha256:changed'}]))
    launches = []
    monkeypatch.setattr(docker.subprocess, 'run', lambda *a, **k: launches.append(a))
    with pytest.raises(ValueError, match='Image differs'):
        docker.run('example/plus-h:local', ['plus_h', 'verify'], inputs=tmp_path, results=tmp_path)
    assert launches == []
