"""Shared launch interface, independent input reading, and final-run parity gates."""

import json
import os
import pickle
import subprocess
import sys

import pytest
import torch

from benchmarks.cqa import cli, docker
from benchmarks.cqa.manifests import REPO, SUITES, prepare_manifest, read_manifest, suite_directory, validate_entry
from benchmarks.cqa.oracles import verify_predictions
from benchmarks.cqa.study import dataset_key, entry_inputs_identity, freeze, run_job
from benchmarks.cqa.verification import export_reference as exporter
from dicee.query_answering import BENCHMARK_DATASETS, QueryContext, load_benchmark
from dicee.query_answering._query import QUERY_SHAPES, ULTRAQUERY_SHAPES
from dicee.query_answering.context import fingerprint
from dicee.query_answering.datasets import dataset_spec
from dicee.query_answering.methods import REFERENCES, UltraQuery

PUBLIC = suite_directory('ultraquery')


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def fixture(root, name):
    group, folder, _ = dataset_spec(name)
    path = root / 'data' / folder
    path.mkdir(parents=True)
    pairs = ((0, 1), (2, 3)) if group == 'transductive' else ((0, 2), (1, 3))

    def graph(stem, direct):
        triples = QueryContext(direct, 8, 4, pairs).triples
        (path / f'{stem}.txt').write_text(''.join(f'{h}\t{r}\t{t}\n' for h, r, t in triples))

    def dump(filename, value):
        (path / filename).write_bytes(pickle.dumps(value))

    train = [(0, 0, 1), (1, 2 if group == 'transductive' else 1, 2), (2, 0, 3)]
    if group == 'transductive':
        graph('train', train)
        dump('id2ent.pkl', {i: str(i) for i in range(8)})
        dump('id2rel.pkl', {i: str(i) for i in range(4)})
    else:
        graph('train_graph', train)
        if group == 'inductive-e':
            graph('val_inference', [(2, 1, 5)])
            graph('test_inference', [(2, 1, 6)])
        else:
            graph('test_inference', [(0, 1, 3), (3, 0, 1), (1, 1, 2)])

    def instantiate(shape, counts):
        if shape in ('n', 'u'):
            return -2 if shape == 'n' else -1
        if shape in ('e', 'r'):
            i = int(shape == 'r')
            value = counts[i]
            counts[i] += 1
            return value % (4 if i else 8)
        return tuple(instantiate(part, counts) for part in shape)

    queries = {QUERY_SHAPES[s]: {instantiate(QUERY_SHAPES[s], [0, 0])} for s in ULTRAQUERY_SHAPES}
    for split in ('valid', 'test'):
        if group == 'transductive':
            dump(f'{split}-queries.pkl', queries)
            dump(f'{split}-easy-answers.pkl', {q: {1} for qs in queries.values() for q in qs})
            dump(f'{split}-hard-answers.pkl', {q: {3} for qs in queries.values() for q in qs})
        else:
            dump(f'{split}_queries.pkl', queries)
            dump(f'{split}_answers_easy.pkl', {s: {q: {1} for q in qs} for s, qs in queries.items()})
            dump(f'{split}_answers_hard.pkl', {s: {q: {3} for q in qs} for s, qs in queries.items()})
    data = load_benchmark(root / 'data', name, split='valid')
    model = UltraQuery(data.context, dim=8, num_layers=2)
    torch.save({'model': {'model.model.' + k: v for k, v in model.state_dict().items()}}, root / 'model.pt')
    entry = dict(id='ultraquery-fixture', method='ultraquery', dataset=name, checkpoint='model.pt',
                 options={'threshold': 0., 'logic': 'product'}, query_types=list(ULTRAQUERY_SHAPES),
                 query_batch_size=32, query_order='upstream-pickle', reference={'status': 'test fixture'}, blockers=[])
    return dict(version=1, data_root='data', entries=[entry], seed=0, threads=2, archives={}, answer_filter='released')


def test_public_recipes_cover_all_datasets_and_selected_adapters():
    paths = [PUBLIC / name for name in ('baselines.json', 'kgfm_adapters.json')]
    manifest = prepare_manifest(paths, suite='ultraquery', answer_filter='released')
    assert len(manifest['entries']) == 69
    assert all('-product-intersections-' in e['id'] for e in manifest['entries'] if e['method'].endswith('-adapter'))
    extras = prepare_manifest([*paths, PUBLIC / 'kgfm_14types.json'], suite='ultraquery', answer_filter='released')
    assert len(extras['entries']) == 115
    for entry in manifest['entries']:
        validate_entry(entry)
        assert set(entry['query_types']) == set(ULTRAQUERY_SHAPES)
        assert 'inference_graph' not in entry
    for method in ('ultraquery', 'ultra-adapter', 'trix-adapter'):
        assert {e['dataset'] for e in manifest['entries'] if e['method'] == method} == set(BENCHMARK_DATASETS)
    for backbone in ('ultra', 'trix'):
        adapter = json.loads((REPO / 'benchmarks/adapters' / f'{backbone}_product_intersections.json').read_text())
        assert adapter['training']['shapes'] == ['2i', '3i']
    # Both suites share one copy of every released adapter.
    shared = {path for suite in SUITES for name in ('kgfm_adapters.json', 'kgfm_14types.json')
              if (suite_directory(suite) / name).is_file()
              for entry in read_manifest(suite_directory(suite) / name)['entries'] for path in entry['adapters'].values()}
    assert len(shared) == 4 and all(path.startswith('benchmarks/adapters/') and (REPO / path).is_file() for path in shared)


def test_cli_subset_and_suite_policies_fail_before_opening_inputs(tmp_path, capsys):
    cli.main(['ultraquery', 'evaluate', '--methods', 'ultraquery', 'ultra-adapter', '--datasets', 'WikiTopicsQuery:art',
          'InductiveFB15k237Query:106', '--query-types', 'negation', '--input-root', str(tmp_path),
          '--output', str(tmp_path / 'results'), '--dry-run'])
    manifest = json.loads(capsys.readouterr().out)
    assert len(manifest['entries']) == 4 and manifest['answer_filter'] == 'released'
    assert all(set(e['query_types']) == {'2in', '3in', 'inp', 'pin', 'pni'} for e in manifest['entries'])
    assert not (tmp_path / 'results').exists()
    for selection in (['--answer-filter', 'corrected'], ['--datasets', 'FB15k237+H'], ['--methods', 'bad']):
        with pytest.raises(SystemExit):
            cli.main(['ultraquery', 'evaluate', '--output', str(tmp_path / 'results'), '--dry-run', *selection])


@pytest.mark.parametrize('action', ['prepare', 'evaluate', 'run', 'report'])
def test_docker_actions_use_the_shared_runtime_and_correct_suite(tmp_path, capsys, action):
    args = ['ultraquery', action, '--image', 'dicee/cqa:test', '--input-root', str(tmp_path), '--output', str(tmp_path / 'results'),
            '--gpu', 'none', '--dry-run']
    if action in ('evaluate', 'run'):
        args += ['--device', 'cpu']
    if action == 'report':
        args += ['--results', str(tmp_path / 'results/test')]
    else:
        args += ['--methods', 'ultraquery', '--datasets', 'WikiTopicsQuery:art']
    if action in ('evaluate', 'prepare'):
        args += ['--query-types', 'negation']
    cli.main(args)
    output = capsys.readouterr().out
    assert f'dicee/cqa:test ultraquery {action} --input-root /inputs --output /results' in output
    assert '--gpus' not in output and '--published' not in output
    if action == 'run':
        with pytest.raises(SystemExit):
            cli.main([*args, '--query-types', '1p'])
    assert not (tmp_path / 'results').exists()


def test_docker_custom_manifests_are_mapped_and_confined_to_inputs(tmp_path, capsys):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    manifests = [inputs / name for name in ('native.json', 'adapters.json')]
    for path in manifests:
        path.write_text('{}')
    args = ['ultraquery', 'prepare', '--image', 'dicee/cqa:test', '--input-root', str(inputs), '--output', str(inputs / 'results'), '--dry-run']
    cli.main([*args, '--manifests', str(manifests[0]), '--manifests', str(manifests[1])])
    assert '--manifests /inputs/native.json /inputs/adapters.json' in capsys.readouterr().out
    outside = tmp_path / 'outside.json'
    outside.write_text('{}')
    with pytest.raises(SystemExit):
        cli.main([*args, '--manifests', str(outside)])
    assert 'below --input-root' in capsys.readouterr().err
    assert not (inputs / 'results').exists()


def test_container_runs_the_cli_entrypoint_with_a_pinned_image(tmp_path, monkeypatch):
    launches = []
    monkeypatch.setattr(docker.subprocess, 'check_output', lambda *args, **kwargs: '[{"Id":"sha256:test"}]')
    monkeypatch.setattr(docker.subprocess, 'run', lambda command, **kwargs: launches.append(command))
    results = tmp_path / 'results'
    results.mkdir()
    docker.run('dicee/cqa:test', ['ultraquery', 'evaluate', '--output', '/results'], inputs=tmp_path, results=results)
    command = launches[0]
    assert '--entrypoint' not in command and command[command.index('--network') + 1] == 'none'
    assert command[command.index('sha256:test') + 1:] == ['ultraquery', 'evaluate', '--output', '/results']
    assert json.loads((results / 'container-image.json').read_text())['image_id'] == 'sha256:test'


@pytest.mark.parametrize('name', ['FB15k237LogicalQuery', 'InductiveFB15k237Query:106', 'WikiTopicsQuery:art'])
def test_independent_exporter_reads_all_three_published_layouts(tmp_path, name):
    manifest = fixture(tmp_path, name)
    bundle = freeze(manifest, tmp_path / 'bundle', tmp_path)
    prefix, structured, easy, hard, graph, graphs = exporter.reference_dataset(bundle, manifest['entries'][0], tmp_path)
    assert set(graph['triples']) == set(load_benchmark(tmp_path / 'data', name, split='valid').context.triples)
    assert {q for qs in structured.values() for q in qs} == set(easy) == set(hard)
    assert graphs == {split: bundle['datasets'][dataset_key(manifest['entries'][0], split)]['context'] for split in ('valid', 'test')}
    assert prefix.endswith(dataset_spec(name)[1] + '/')


def test_final_runs_require_matching_parity_and_preserve_14_type_coverage(tmp_path):
    manifest = fixture(tmp_path, 'WikiTopicsQuery:art')
    bundle = freeze(manifest, tmp_path / 'bundle', tmp_path)
    with pytest.raises(ValueError, match='requires pinned reference'):
        run_job(tmp_path / 'bundle', 'ultraquery-fixture', tmp_path, tmp_path / 'test', phase='test')
    scores = {}
    pilot = run_job(tmp_path / 'bundle', 'ultraquery-fixture', tmp_path, tmp_path / 'pilot',
                    on_prediction=lambda q, values: scores.update({q.identity: values.clone()}))
    entry = manifest['entries'][0]
    oracle = dict(version=1, entry_sha256=fingerprint(entry), inputs_sha256=entry_inputs_identity(bundle, entry),
                  validation_plan_sha256=bundle['plans']['ultraquery-fixture/valid']['sha256'],
                  reference_commit=REFERENCES['ultraquery'][1], pilot_queries=2, scores=scores,
                  orders={k: v.argsort(descending=True) for k, v in scores.items()},
                  environment=pilot['benchmark_run']['environment'],
                  graphs={split: bundle['datasets'][dataset_key(entry, split)]['context'] for split in ('valid', 'test')})
    reference = tmp_path / 'reference.pt'
    torch.save(oracle, reference)
    evidence = verify_predictions(tmp_path / 'bundle', entry['id'], tmp_path, reference, tmp_path / 'evidence.json')
    assert evidence['passed'] and len(evidence['queries']) == 14
    missing_graphs = dict(oracle, graphs={})
    torch.save(missing_graphs, reference)
    with pytest.raises(ValueError, match='inference graphs'):
        verify_predictions(tmp_path / 'bundle', entry['id'], tmp_path, reference, tmp_path / 'bad.json')
    oracle['scores'][next(iter(scores))] = torch.zeros_like(next(iter(scores.values())))
    oracle['orders'] = {k: v.argsort(descending=True) for k, v in oracle['scores'].items()}
    torch.save(oracle, reference)
    assert not verify_predictions(tmp_path / 'bundle', entry['id'], tmp_path, reference, tmp_path / 'failed.json')['passed']
    entry['verification'] = 'evidence.json'
    freeze(manifest, tmp_path / 'verified', tmp_path)
    report = run_job(tmp_path / 'verified', entry['id'], tmp_path, tmp_path / 'test', phase='test')
    assert report['protocol']['full_split'] and report['coverage']['complete_benchmark_types']
    assert not report['coverage']['complete_16_types']
    assert set(report['per_shape']) == set(ULTRAQUERY_SHAPES)
    assert set(report['additional_tie_metrics']) == {'expected'}
    cli.main(['ultraquery', 'report', '--results', str(tmp_path / 'test'), '--output', str(tmp_path / 'reports'), '--bootstrap-samples', '0'])
    rows = json.loads((tmp_path / 'reports/comparison.json').read_text())['rows']
    assert rows[0]['complete_test'] and rows[0]['types'] == 14 and rows[0]['published_mrr'] is None


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('DICEE_ULTRAQUERY_REFERENCE_ROOT'), reason='Set the pinned upstream root and dependencies')
@pytest.mark.parametrize('name', ['FB15k237LogicalQuery', 'InductiveFB15k237Query:106', 'WikiTopicsQuery:art'])
def test_independent_ultraquery_export_parity_and_full_run(tmp_path, name):
    manifest = fixture(tmp_path, name)
    freeze(manifest, tmp_path / 'bundle', tmp_path)
    reference = tmp_path / 'reference.pt'
    subprocess.run([sys.executable, str(REPO / 'benchmarks/cqa/verification/export_reference.py'), '--bundle', str(tmp_path / 'bundle'),
                    '--input-root', str(tmp_path), '--entry', 'ultraquery-fixture', '--upstream',
                    os.environ['DICEE_ULTRAQUERY_REFERENCE_ROOT'], '--output', str(reference)], check=True)
    evidence = verify_predictions(tmp_path / 'bundle', 'ultraquery-fixture', tmp_path, reference, tmp_path / 'evidence.json')
    assert evidence['passed'] and len(evidence['queries']) == 14
    manifest['entries'][0]['verification'] = 'evidence.json'
    freeze(manifest, tmp_path / 'verified', tmp_path)
    report = run_job(tmp_path / 'verified', 'ultraquery-fixture', tmp_path, tmp_path / 'test', phase='test')
    assert report['protocol']['full_split'] and report['coverage']['complete_benchmark_types']
