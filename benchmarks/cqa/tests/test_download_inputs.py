"""Input setup must be selective, bounded, and preserve immutable inputs."""

import hashlib
import io
import json
import stat
import subprocess
import sys
import zipfile

import pytest

from benchmarks.cqa import cli, docker, inputs
from benchmarks.cqa.manifests import REPO, SUITES, default_recipes, prepare_manifest

DATASET = {'plus_h': 'FB15k237+H', 'ultraquery': 'WikiTopicsQuery:art'}


def recipes(suite, **selection):
    return prepare_manifest(default_recipes(suite), suite=suite, answer_filter=SUITES[suite]['answer_filter'], **selection)


def test_setup_plan_downloads_only_selected_methods_and_datasets():
    jobs, weights = inputs.plan(recipes('plus_h', methods=['cqd', 'cqd-hybrid'], datasets=['NELL995+H']), 'plus_h')
    assert len(jobs) == 1
    assert list(jobs[0]['folders']) == ['iscqa-compl-benchmarks/new_benchmarks/NELL995+H']
    assert weights == ['checkpoints/query-baselines/iscqa-compl-models/models/CQD:CQD-HYBRID:QTO/QTO/nellcheckpoint']
    manifest = recipes('ultraquery', methods=['ultra-adapter'], datasets=['WikiTopicsQuery:art', 'InductiveFB15k237Query:106'])
    jobs, weights = inputs.plan(manifest, 'ultraquery')
    assert [job['id'] for job in jobs] == ['ultra-inductive-106', 'ultra-wikitopics']
    assert list(jobs[1]['folders']) == ['WikiTopics_QE/art']
    assert weights == ['checkpoints/ultra_3g.pth']
    with pytest.raises(ValueError):
        recipes('plus_h', methods=['misspelled'])


def test_plus_h_plan_includes_every_released_report_category_and_corrected_filter():
    from benchmarks.cqa.difficulty import REDUCTIONS
    required = inputs.plus_h_files()
    for shape, labels in REDUCTIONS.items():
        for label in labels:
            assert f'test-query-reduction/{shape}/{label}/test-hard-answers.pkl' in required
    assert 'test-query-reduction/2in/all/test-easy-answers.pkl' in required
    assert 'test-query-reduction/pni/all/test-hard-answers.pkl' in required
    icews = inputs.plus_h_files(True)
    assert 'KG_splits/test.txt' in icews and 'test.txt' not in icews
    assert not any(name.startswith('test-query-reduction') for name in icews)


def test_streamed_download_rejects_bad_hash_without_touching_existing_input(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'official bytes')
    target = tmp_path / 'checkpoint'
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    inputs.download(source.as_uri(), target, expected)
    assert target.read_bytes() == b'official bytes'
    source.write_bytes(b'corrupted download')
    with pytest.raises(ValueError, match='SHA-256'):
        inputs.download(source.as_uri(), target, expected)
    assert target.read_bytes() == b'official bytes'
    assert not list(tmp_path.glob('.input-*'))
    with pytest.raises(ValueError, match='refusing to replace'):
        inputs.install(io.BytesIO(b'different input'), target)
    assert target.read_bytes() == b'official bytes'


def test_interrupted_stream_never_installs_partial_input_and_reads_are_bounded(tmp_path):
    class Broken:
        calls = 0

        def read(self, size):
            assert size == inputs.CHUNK
            self.calls += 1
            if self.calls == 2:
                raise OSError('connection dropped')
            return b'partial'

    with pytest.raises(OSError, match='connection dropped'):
        inputs.install(Broken(), tmp_path / 'checkpoint')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('unsafe', ['../escape', '/absolute', 'symlink'])
def test_archive_is_validated_before_any_file_is_installed(tmp_path, unsafe):
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('valid.txt', b'valid')
        if unsafe == 'symlink':
            member = zipfile.ZipInfo('symlink')
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            zipped.writestr(member, '../escape')
        else:
            zipped.writestr(unsafe, b'unsafe')
    root = tmp_path / 'inputs'
    root.mkdir()
    with pytest.raises(ValueError, match='Unsafe|Symlink'):
        inputs.extract(archive, root, lambda name: name if name == 'valid.txt' else None)
    assert not list(root.iterdir())


def test_complete_public_inputs_do_not_contact_network_and_install_shipped_adapter(tmp_path, monkeypatch):
    manifest = recipes('ultraquery', methods=['ultra-adapter'], datasets=['WikiTopicsQuery:art'])
    jobs, weights = inputs.plan(manifest, 'ultraquery')
    for job in jobs:
        for name in inputs.missing_files(tmp_path, job):
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'existing labels')
    weight = tmp_path / weights[0]
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b'weight fixture')
    monkeypatch.setitem(inputs.WEIGHT_HASHES, weights[0], inputs.digest(weight))
    monkeypatch.setattr(inputs, 'download', lambda *a, **kw: pytest.fail('Unexpected network request'))
    inputs.ensure_inputs(manifest, tmp_path, suite='ultraquery')
    adapter = next(iter(manifest['entries'][0]['adapters'].values()))
    assert (tmp_path / adapter).read_bytes() == (REPO / adapter).read_bytes()
    inputs.ensure_inputs(manifest, tmp_path, suite='ultraquery', download_missing=False)


def test_custom_labels_are_not_mixed_with_public_data_and_missing_custom_weights_fail(tmp_path, monkeypatch):
    manifest = recipes('plus_h', methods=['cqd'], datasets=['FB15k237+H'])
    manifest['archives'] = {}
    manifest['entries'][0]['checkpoint'] = 'my-model.pt'
    monkeypatch.setattr(inputs, 'download', lambda *a, **kw: pytest.fail('Unexpected download for custom inputs'))
    with pytest.raises(ValueError, match='custom checkpoint'):
        inputs.ensure_inputs(manifest, tmp_path, suite='plus_h')
    (tmp_path / 'my-model.pt').write_bytes(b'custom weights')
    inputs.ensure_inputs(manifest, tmp_path, suite='plus_h')
    assert not (tmp_path / manifest['data_root']).exists()


@pytest.mark.parametrize('suite,method', [('plus_h', 'cqd'), ('ultraquery', 'ultraquery')])
def test_host_commands_need_only_the_standard_library(tmp_path, suite, method):
    """Setup and Docker launches run with ``python -S``: no PyTorch or other site packages."""
    root = tmp_path / 'inputs'
    common = [sys.executable, '-S', '-m', 'benchmarks.cqa', suite]
    selection = ['--input-root', str(root), '--methods', method, '--datasets', DATASET[suite]]
    output = subprocess.check_output([*common, 'setup', *selection, '--dry-run'], text=True, cwd=REPO)
    assert 'https://' in output and method in output.lower()
    checked = subprocess.run([*common, 'setup', *selection, '--check'], capture_output=True, text=True, cwd=REPO)
    assert checked.returncode == 1 and 'Missing:' in checked.stdout
    root.mkdir()
    launch = subprocess.check_output([*common, 'evaluate', *selection, '--output', str(root / 'results'),
                                      '--image', 'example:test', '--dry-run'], text=True, cwd=REPO)
    assert launch.startswith('docker run') and f'example:test {suite} evaluate --input-root /inputs' in launch
    assert list(root.iterdir()) == []


def test_run_sets_up_the_frozen_bundle_inputs_not_the_current_recipes(tmp_path, monkeypatch):
    manifest = recipes('plus_h', methods=['cqd'], datasets=['ICEWS18+H'])
    study = tmp_path / 'study'
    (study / 'bundle').mkdir(parents=True)
    (study / 'bundle/bundle.json').write_text(json.dumps({'manifest': manifest, 'files': {'pinned': 'digest'}}))
    before = (study / 'bundle/bundle.json').read_bytes()
    seen = []

    def setup(manifest, root, **kwargs):
        seen.append(manifest)
        raise RuntimeError('stop after setup')

    monkeypatch.setattr(inputs, 'ensure_inputs', setup)
    with pytest.raises(RuntimeError, match='stop after setup'):
        cli.main(['plus_h', 'run', '--phase', 'pilot', '--input-root', str(tmp_path), '--output', str(study)])
    assert seen == [manifest]
    assert (study / 'bundle/bundle.json').read_bytes() == before


@pytest.mark.parametrize('suite', SUITES)
def test_evaluate_sets_up_selected_inputs_before_evaluation(tmp_path, monkeypatch, suite):
    from benchmarks.cqa import evaluate
    calls = []
    monkeypatch.setattr(inputs, 'ensure_inputs', lambda manifest, root, **kwargs: calls.append(('setup', manifest, kwargs)))

    def evaluated(manifest, **kwargs):
        assert calls[-1][:2] == ('setup', manifest)
        calls.append(('evaluate', manifest, kwargs))

    monkeypatch.setattr(evaluate, 'evaluate_manifest', evaluated)
    args = [suite, 'evaluate', '--methods', 'ultraquery', '--datasets', DATASET[suite], '--input-root', str(tmp_path),
            '--output', str(tmp_path / 'results')]
    cli.main(args)
    assert [call[0] for call in calls] == ['setup', 'evaluate']
    assert [entry['dataset'] for entry in calls[0][1]['entries']] == [DATASET[suite]]
    assert calls[0][2]['suite'] == suite and calls[0][2].get('download_missing', True)
    calls.clear()
    cli.main([*args, '--no-setup'])
    assert calls[0][2]['download_missing'] is False


@pytest.mark.parametrize('suite', SUITES)
def test_setup_dry_run_lists_sources_without_writing(tmp_path, capsys, suite):
    cli.main([suite, 'setup', '--methods', 'ultraquery', '--datasets', DATASET[suite],
              '--input-root', str(tmp_path / 'absent'), '--dry-run'])
    output = capsys.readouterr().out
    assert 'https://' in output and 'ultraquery.pth' in output
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('suite', SUITES)
def test_docker_launch_fetches_on_host_before_isolated_container(tmp_path, monkeypatch, suite):
    calls = []
    monkeypatch.setattr(inputs, 'ensure_inputs', lambda manifest, root, **kwargs: calls.append(('setup', root, kwargs['suite'])))
    monkeypatch.setattr(docker, 'run', lambda image, argv, **kwargs: calls.append(('container', image, argv)))
    root = tmp_path / 'inputs'
    root.mkdir()
    args = [suite, 'evaluate', '--input-root', str(root), '--output', str(root / 'results'), '--image', 'test', '--device', 'cpu',
            '--methods', 'ultraquery', '--datasets', DATASET[suite]]
    cli.main(args)
    assert [call[0] for call in calls] == ['setup', 'container'] and calls[0][2] == suite
    assert '--no-setup' in calls[1][2] and '--image' not in calls[1][2]
    calls.clear()
    cli.main([*args, '--no-setup'])
    assert [call[0] for call in calls] == ['container']
