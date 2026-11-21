"""UltraQuery suite: public recipes, launch interface and the three released dataset layouts."""

import json
import pickle

import pytest
import torch

from benchmarks.cqa import cli, docker
from benchmarks.cqa.evaluate import evaluate_manifest
from benchmarks.cqa.manifests import REPO, SUITES, prepare_manifest, read_manifest, suite_directory, validate_entry
from dicee.query_answering import BENCHMARK_DATASETS, QueryContext, load_benchmark
from dicee.query_answering._query import QUERY_SHAPES, ULTRAQUERY_SHAPES
from dicee.query_answering.datasets import dataset_spec
from dicee.query_answering.methods import UltraQuery

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
                 query_batch_size=32, query_order='upstream-pickle', reference={'status': 'test fixture'})
    return dict(version=1, data_root='data', entries=[entry], seed=0, threads=2, archives={}, answer_filter='released')


def test_public_recipes_cover_all_datasets_and_selected_adapters():
    paths = [PUBLIC / name for name in ('baselines.json', 'kgfm_adapters.json')]
    manifest = prepare_manifest(paths, suite='ultraquery', answer_filter='released')
    assert len(manifest['entries']) == 69
    assert all('-product-intersections-' in e['id'] for e in manifest['entries'] if e['method'].endswith('-adapter'))
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
    shared = {path for suite in SUITES for name in ('kgfm_adapters.json',)
              if (suite_directory(suite) / name).is_file()
              for entry in read_manifest(suite_directory(suite) / name)['entries'] for path in entry['adapters'].values()}
    assert shared and all(path.startswith('benchmarks/adapters/') and (REPO / path).is_file() for path in shared)


def test_cli_subset_and_suite_policies_fail_before_opening_inputs(tmp_path, capsys):
    cli.main(['ultraquery', 'evaluate', '--split', 'test', '--methods', 'ultraquery', 'ultra-adapter', '--datasets', 'WikiTopicsQuery:art',
          'InductiveFB15k237Query:106', '--query-types', 'negation', '--input-root', str(tmp_path),
          '--output', str(tmp_path / 'results'), '--dry-run'])
    manifest = json.loads(capsys.readouterr().out)
    assert len(manifest['entries']) == 4 and manifest['answer_filter'] == 'released'
    assert all(set(e['query_types']) == {'2in', '3in', 'inp', 'pin', 'pni'} for e in manifest['entries'])
    assert not (tmp_path / 'results').exists()
    for selection in (['--answer-filter', 'corrected'], ['--datasets', 'FB15k237+H'], ['--methods', 'bad']):
        with pytest.raises(SystemExit):
            cli.main(['ultraquery', 'evaluate', '--split', 'test', '--output', str(tmp_path / 'results'), '--dry-run', *selection])


def test_docker_evaluation_uses_the_shared_runtime_and_correct_suite(tmp_path, capsys):
    cli.main(['ultraquery', 'evaluate', '--image', 'dicee/cqa:test', '--input-root', str(tmp_path), '--output', str(tmp_path / 'results'),
              '--gpu', 'none', '--dry-run', '--device', 'cpu', '--split', 'test', '--methods', 'ultraquery',
              '--datasets', 'WikiTopicsQuery:art', '--query-types', 'negation'])
    output = capsys.readouterr().out
    assert 'dicee/cqa:test ultraquery evaluate --input-root /inputs --output /results' in output
    assert '--gpus' not in output
    assert not (tmp_path / 'results').exists()


def test_docker_custom_manifests_are_mapped_and_confined_to_inputs(tmp_path, capsys):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    manifests = [inputs / name for name in ('native.json', 'adapters.json')]
    for path in manifests:
        path.write_text('{}')
    args = ['ultraquery', 'evaluate', '--split', 'test', '--image', 'dicee/cqa:test', '--input-root', str(inputs),
            '--output', str(inputs / 'results'), '--dry-run']
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
@pytest.mark.parametrize('split', ['valid', 'test'])
def test_native_ultraquery_evaluates_all_three_published_layouts(tmp_path, name, split):
    manifest = fixture(tmp_path, name)
    evaluate_manifest(manifest, input_root=tmp_path, output=tmp_path / 'results', split=split)
    report = json.loads((tmp_path / 'results/ultraquery-fixture/result.json').read_text())
    assert report['split'] == split and report['queries'] == 14
    assert report['protocol']['full_split'] and report['protocol']['all_benchmark_shapes']
    assert set(report['per_shape']) == set(ULTRAQUERY_SHAPES)
    assert set(report['additional_tie_metrics']) == {'expected'}
    # Each split is answered over its released inference graph.
    assert report['context_sha256'] == load_benchmark(tmp_path / 'data', name, split=split).context.identity
