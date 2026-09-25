"""Resumable source-only adapter sweep on paired logical-query validation subsets."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from dicee.models import TRIX, ULTRA
from dicee.models._inference import float32_precision_backends
from dicee.query_answering import AdapterTrainingData, QueryContext, QueryScoreAdapter, benchmark_model, fit_query_adapter, load_benchmark, prepare_adapter_data, summarize_benchmarks
from dicee.query_answering._checkpoint import BenchmarkCheckpoint, write_json
from dicee.query_answering.benchmark import _implementation_fingerprint, _query_plan
from dicee.query_answering.context import fingerprint, state_fingerprint
from dicee.query_answering.datasets import BENCHMARK_DATASETS
from dicee.query_answering.training import STANDARD_TRAINING_SHAPES, TRAINING_SHAPES

SOURCE_NAMES = {'FB15k237': 'FB15k-237', 'WN18RR': 'WN18RR', 'CoDExMedium': 'CoDEx-Medium'}
SHAPES = ['2i', '3i', '2in', '3in']
VARIANTS = {
    'scores_positive': dict(feature_mode='context_scores_v1', shapes=['2i', '3i']),
    'scores_negation': dict(feature_mode='context_scores_v1', shapes=SHAPES),
    'scores_positive_wide': dict(feature_mode='context_scores_v1', shapes=['2i', '3i'], bias_bound=8.),
    'scores_negation_wide': dict(feature_mode='context_scores_v1', shapes=SHAPES, bias_bound=8.),
    'global_negation': dict(feature_mode='global', shapes=SHAPES),
    'context_negation': dict(feature_mode='context', shapes=SHAPES),
    'normalized_global_negation': dict(feature_mode='global', shapes=SHAPES, normalization='standard'),
    'mlp_negation': dict(feature_mode='context_scores_v1', shapes=SHAPES, hidden_dim=16),
}


def study_variants(study):
    settings = dict(feature_mode='context_scores_v1', bias_bound=8., scale_bound=2.)
    if study == 'query-types':
        groups = [SHAPES[:2], SHAPES, STANDARD_TRAINING_SHAPES, TRAINING_SHAPES]
        return {f'types_{len(shapes)}': dict(settings, shapes=list(shapes), train_per_shape=280 // len(shapes))
                for shapes in groups}
    if study == 'scale':
        return {name: dict(settings, shapes=SHAPES[:2], train_per_shape=140, scale_bound=bound)
                for name, bound in [('types_2', 2.), ('scale_4', 4.), ('scale_8', 8.), ('scale_unbounded', None)]}
    return VARIANTS


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def status(out, **fields):
    write_json(out/'status.json', dict(updated=time.time(), **fields))


def runtime():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    for backend in float32_precision_backends():
        backend.fp32_precision = 'ieee'


def load_model(config, backbone, batch_size):
    cls = {'ultra': ULTRA, 'trix': TRIX}[backbone]
    if sha(config['checkpoints'][backbone]['path']) != config['checkpoints'][backbone]['sha256']:
        raise ValueError('Backbone checkpoint changed')
    return cls(dict(num_entities=1, num_relations=1, graph_inference_backend='auto',
                    **{f'{backbone}_query_batch_size': batch_size})).load_pretrained(
                        config['checkpoints'][backbone]['path']).cuda().eval().requires_grad_(False)


def prepare_sources(out, config):
    directory = out/'sources'
    directory.mkdir(exist_ok=True)
    for name, source in config['sources'].items():
        source_bytes = Path(source['path']).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != source['sha256']:
            raise ValueError('Source file changed')
        path = directory/f'{name}.json'
        if config.get('source_data_from') and not path.exists():
            original = Path(config['source_data_from'])/'sources'/f'{name}.json'
            if sha(original) != config['prepared_sources'][name]:
                raise ValueError('Prepared source data changed')
            shutil.copy2(original, path)
        if path.exists():
            if name in config.get('prepared_sources', {}) and sha(path) != config['prepared_sources'][name]:
                raise ValueError('Prepared source data changed')
            data = AdapterTrainingData.load(path)
            if data.metadata['train_sha256'] != source['sha256']:
                raise ValueError('Source file changed')
            continue
        status(out, state='preparing', source=name)
        triples = [line.split() for line in source_bytes.decode().splitlines() if line.strip()]
        entities = {v: i for i, v in enumerate(sorted({v for h, _, t in triples for v in (h, t)}))}
        relations = {v: i for i, v in enumerate(sorted({r for _, r, _ in triples}))}
        count = len(relations)
        context = QueryContext(tuple((entities[h], relations[r], entities[t]) for h, r, t in triples),
                               len(entities), 2 * count, tuple((r, r + count) for r in range(count)))
        previous = config.get('data_scaling_from')
        existing = AdapterTrainingData.load(Path(previous)/'sources'/f'{name}.json') if previous else None
        count = max(v.get('train_per_shape', 96) for v in config['variants'].values())
        data = prepare_adapter_data(context, name=name, seed=config['training_seed'], shapes=config.get('source_shapes', SHAPES),
                                    train_per_shape=count, validation_per_shape=config.get('source_validation_per_shape', 32),
                                    train_counts=config.get('source_training_counts'), extend=existing,
                                    max_attempts=config.get('preparation_attempts', 100_000))
        data.metadata.update(train_sha256=source['sha256'], source_path=source['path'])
        data.save(path)
        log(f'Prepared {name}: {len(data.train)} training / {len(data.validation)} validation queries')


def fit_worker(out, config, backbone, batch_size):
    runtime()
    model = load_model(config, backbone, batch_size)
    sources = [AdapterTrainingData.load(out/'sources'/f'{name}.json') for name in SOURCE_NAMES]
    directory = out/'adapters'/backbone
    directory.mkdir(parents=True, exist_ok=True)
    for variant, options in config['variants'].items():
        if variant in config.get('reuse_variants', []):
            continue
        settings = dict(options)
        shapes = settings.pop('shapes')
        train_per_shape = settings.pop('train_per_shape', 192 // len(shapes))
        for heldout in [None, *SOURCE_NAMES]:
            label = heldout or 'all_sources'
            path = directory/variant/label
            path.mkdir(parents=True, exist_ok=True)
            if (path/'report.json').exists():
                QueryScoreAdapter.load(path/'adapter.json', model=model)
                continue
            training = [name for name in SOURCE_NAMES if name != heldout]
            validation = [heldout] if heldout else list(SOURCE_NAMES)
            start = time.monotonic()
            status(out, state='fitting', backbone=backbone, variant=variant, fold=label, epoch=0)

            def progress(record):
                if record['epoch'] % 5 == 0:
                    status(out, state='fitting', backbone=backbone, variant=variant, fold=label, **record)
                    log(f'{backbone} {variant} {label}: epoch {record["epoch"]}/{config["epochs"]}')

            result = fit_query_adapter(model, sources, **settings, observed_mix=1., epochs=config['epochs'],
                                       train_shapes=shapes, train_per_shape=train_per_shape,
                                       training_sources=training, validation_sources=validation,
                                       validation_every=config.get('validation_every', 5),
                                       validation_shapes=config.get('validation_shapes'),
                                       early_stopping_patience=config.get('early_stopping_patience'),
                                       seed=config['training_seed'], on_epoch=progress,
                                       row_batch_size=batch_size, cache_dir=Path(config.get('score_bank_dir', out/'score-banks'))/backbone)
            result.adapter.save(path/'adapter.json')
            loaded = QueryScoreAdapter.load(path/'adapter.json', model=model)
            assert loaded.to_dict() == result.adapter.to_dict()
            write_json(path/'report.json', dict(history=result.history, validation=result.validation,
                                                metadata=result.adapter.metadata, seconds=time.monotonic() - start,
                                                parameters=sum(p.numel() for p in loaded.parameters()),
                                                adapter_sha256=sha(path/'adapter.json')))
            training_report = result.adapter.metadata['training']
            log(f'{backbone} {variant} {label}: finished {training_report["epochs_completed"]} epochs, '
                f'selected {training_report["selected_epoch"]} ({training_report["stop_reason"]})')
        log(f'Fitted {backbone} {variant}, including all three source holdouts')
    for name, mix in ([] if config.get('data_scaling_from') or config.get('study') else [('sigmoid', 0.), ('observed', 1.)]):
        path = directory/name/'all_sources'
        path.mkdir(parents=True, exist_ok=True)
        QueryScoreAdapter('global', mix, metadata={'backbone_state_sha256': state_fingerprint(model)}).save(path/'adapter.json')
    if backbone == 'ultra' and config.get('reference_adapter'):
        path = directory/'reference_500ep'/'all_sources'
        path.mkdir(parents=True, exist_ok=True)
        QueryScoreAdapter.load(config['reference_adapter']['path'], model=model).save(path/'adapter.json')
    write_json(directory/'complete.json', dict(backbone=backbone, state_sha256=state_fingerprint(model)))


def variant_names(config, backbone):
    names = list(config['variants'])
    if not config.get('data_scaling_from') and not config.get('study'):
        names = ['observed', *names, 'sigmoid']
    if backbone == 'ultra' and config.get('reference_adapter'):
        names.insert(0, 'reference_500ep')
    return names


def evaluate_worker(out, config, backbone, variant, dataset, batch_size, cache_mb):
    runtime()
    directory = out/'benchmark'/backbone/variant/dataset.replace(':', '-')
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    status(out, state='loading', backbone=backbone, variant=variant, dataset=dataset)
    model = load_model(config, backbone, batch_size)
    adapter = QueryScoreAdapter.load(out/'adapters'/backbone/variant/'all_sources'/'adapter.json', model=model)
    data = load_benchmark(Path(config['root'])/'KGs'/'UltraQuery', dataset, split='valid')
    plan = _query_plan(data, config['size'], 'relation', sampling='uniform', seed=config['sampling_seed'])
    selection = [dict(shape=q.shape, query=q.query) for q in plan]
    if (directory/'selection.json').exists():
        assert fingerprint(json.loads((directory/'selection.json').read_text())) == fingerprint(selection)
    write_json(directory/'selection.json', selection)
    reference = None
    if config.get('data_scaling_from'):
        reference = json.loads((out/'benchmark'/backbone/config.get('comparison_baseline', 'scores_negation_wide')/
                                dataset.replace(':', '-')/'result.json').read_text())
        if sha(directory/'selection.json') != reference['selection_sha256']:
            raise ValueError('Data-scaling evaluation queries differ from the baseline')
    trace = directory/'metrics.jsonl'
    checkpoint = directory/'checkpoint'/'state.json'
    completed = json.loads(checkpoint.read_text())['state']['completed'] if checkpoint.exists() else 0
    records = [json.loads(line) for line in trace.read_text().splitlines()[:completed]] if trace.exists() else []
    assert len(records) == completed
    for row, query in zip(records, plan):
        assert row['id'] == fingerprint(query.query) and row['shape'] == query.shape
    trace.write_text(''.join(json.dumps(row) + '\n' for row in records))
    load_seconds = time.monotonic() - start

    with trace.open('a', buffering=1) as stream:
        def record(query, shape, metrics):
            expected = plan[len(records)]
            assert query == expected.query and shape == expected.shape
            row = dict(id=fingerprint(query), shape=shape, **metrics)
            records.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

        def progress(report):
            write_json(directory/'progress.json', report)
            status(out, state='evaluating', backbone=backbone, variant=variant, dataset=dataset,
                   completed=report['queries'], total=len(plan), seconds=report['seconds'])

        report = benchmark_model(model, data, adapter=adapter, beam_size=64, row_batch_size=batch_size,
                                 backend_batch_size=batch_size, cache_bytes=cache_mb * 2**20,
                                 query_batch_size=1, max_queries_per_shape=config['size'],
                                 query_sampling='uniform', sampling_seed=config['sampling_seed'],
                                 checkpoint_dir=directory/'checkpoint', checkpoint_every=10,
                                 on_checkpoint=progress, on_query=record)
    assert len(records) == len(plan)
    if reference is not None:
        for key in ('context_sha256', 'candidate_sha256'):
            if report[key] != reference['report'][key]:
                raise ValueError(f'Data-scaling evaluation changed {key}')
    for shape, values in report['per_shape'].items():
        rows = [r for r in records if r['shape'] == shape]
        assert abs(sum(r['mrr'] for r in rows) / len(rows) - values['mrr']) < 1e-12
    write_json(directory/'result.json', dict(report=report, load_seconds=load_seconds,
                                            wall_seconds=time.monotonic() - start,
                                            metrics_sha256=sha(trace), selection_sha256=sha(directory/'selection.json')))
    log(f'{backbone} {variant} {dataset}: {report["seconds"]:.1f}s')


def publish(out, config):
    results: dict[str, dict[str, Any]] = {}
    for backbone in config['backbones']:
        results[backbone] = {}
        for variant in variant_names(config, backbone):
            reports = []
            for dataset in BENCHMARK_DATASETS:
                path = out/'benchmark'/backbone/variant/dataset.replace(':', '-')/'result.json'
                if path.exists():
                    job = json.loads(path.read_text())
                    if sha(path.parent/'metrics.jsonl') != job['metrics_sha256']:
                        raise ValueError('Benchmark trace checksum mismatch')
                    reports.append(job['report'])
            folds = []
            for name in SOURCE_NAMES:
                path = out/'adapters'/backbone/variant/name/'report.json'
                if path.exists():
                    report = json.loads(path.read_text())
                    folds.append(report['metadata']['training']['selected_validation_mrr'])
            results[backbone][variant] = dict(datasets=len(reports),
                source_holdout_mrr=sum(folds) / len(folds) if folds else None,
                summary=summarize_benchmarks(reports) if reports else None,
                seconds=sum(r['seconds'] for r in reports))
    write_json(out/'comparison.json', results)
    lines = ['# Adapter screening', '',
             f'Validation: {config["size"]} fixed queries per type, all 23 datasets. MRR × 100.',
             'Partial rows may cover different datasets; compare target scores only after completion.', '',
             '| Backbone | Adapter | Source holdout | Datasets | EPFO | Negation | Minutes |',
             '|---|---|---:|---:|---:|---:|---:|']
    for backbone, variants in results.items():
        for variant, result in variants.items():
            avg = result['summary']['groups']['all'] if result['summary'] else None
            source = f'{100 * result["source_holdout_mrr"]:.2f}' if result['source_holdout_mrr'] is not None else '—'
            epfo = f'{100 * avg["epfo"]["mrr"]:.2f}' if avg else '—'
            negation = f'{100 * avg["negation"]["mrr"]:.2f}' if avg else '—'
            lines.append(f'| {backbone} | {variant} | {source} | {result["datasets"]}/23 | {epfo} | {negation} | {result["seconds"] / 60:.1f} |')
    (out/'comparison.md').write_text('\n'.join(lines) + '\n')


def reuse_baseline(out, config):
    previous = Path(config['data_scaling_from'])
    if sha(previous/'configuration.json') != config['predecessor_sha256']:
        raise ValueError('Predecessor configuration changed')
    parent_config = json.loads((previous/'configuration.json').read_text())
    with BenchmarkCheckpoint(previous/'run', parent_config):
        if json.loads((previous/'status.json').read_text())['state'] != 'complete':
            raise ValueError('The preceding adapter sweep must finish first')
        for backbone in config['backbones']:
            for variant in config['reuse_variants']:
                source = previous/'adapters'/backbone/variant
                for fold in ['all_sources', *SOURCE_NAMES]:
                    report = json.loads((source/fold/'report.json').read_text())
                    if sha(source/fold/'adapter.json') != report['adapter_sha256']:
                        raise ValueError('Predecessor adapter checksum mismatch')
                shutil.copytree(source, out/'adapters'/backbone/variant, dirs_exist_ok=True)
                for dataset in BENCHMARK_DATASETS:
                    source = previous/'benchmark'/backbone/variant/dataset.replace(':', '-')
                    result = json.loads((source/'result.json').read_text())
                    for filename, field in [('selection.json', 'selection_sha256'), ('metrics.jsonl', 'metrics_sha256')]:
                        if sha(source/filename) != result[field]:
                            raise ValueError('Predecessor evaluation checksum mismatch')
                    destination = out/'benchmark'/backbone/variant/dataset.replace(':', '-')
                    destination.mkdir(parents=True, exist_ok=True)
                    for filename in ('selection.json', 'metrics.jsonl', 'result.json'):
                        shutil.copy2(source/filename, destination/filename)


def supervise(out, config):
    worker = out/'source'/'dicee'/'scripts'/'benchmark_query_adapters.py'
    env = dict(os.environ, PYTHONPATH=str(out/'source'), PYTORCH_ALLOC_CONF='expandable_segments:True')
    active: subprocess.Popen | None = None

    def stop(signum, _frame):
        if active is not None and active.poll() is None:
            active.terminate()
        status(out, state='stopped', signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def run(job, batch_size):
        nonlocal active
        for attempt in range(3):
            command = [sys.executable, '-u', str(worker), '--output', str(out), '--worker', *job,
                       '--batch-size', str(batch_size), '--cache-mb', '512']
            with (out/'workers.log').open('a') as stream:
                active = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                (out/'worker-pid.txt').write_text(str(active.pid) + '\n')
                code = active.wait()
            if code == 0:
                return
            failure = json.loads((out/'status.json').read_text())
            if 'OutOfMemoryError' not in failure.get('error', ''):
                raise RuntimeError(failure)
            log(f'Fresh-process OOM retry {attempt + 1}: {job}')
        raise RuntimeError(f'Out-of-memory retries exhausted: {job}')

    with BenchmarkCheckpoint(out/'run', config):
        if config.get('after_run'):
            previous = Path(config['after_run'])
            if sha(previous/'configuration.json') != config['after_run_sha256']:
                raise ValueError('Predecessor configuration changed')
            with BenchmarkCheckpoint(previous/'run', json.loads((previous/'configuration.json').read_text())):
                if json.loads((previous/'status.json').read_text())['state'] != 'complete':
                    raise ValueError('The preceding adapter sweep must finish first')
        if config.get('data_scaling_from'):
            reuse_baseline(out, config)
        prepare_sources(out, config)
        for backbone in config['backbones']:
            batch_size = 8 if backbone == 'ultra' else 1
            if not (out/'adapters'/backbone/'complete.json').exists():
                run(['fit', backbone], batch_size)
            publish(out, config)
            for dataset in BENCHMARK_DATASETS:
                for variant in variant_names(config, backbone):
                    path = out/'benchmark'/backbone/variant/dataset.replace(':', '-')/'result.json'
                    if not path.exists():
                        run(['evaluate', backbone, variant, dataset], batch_size)
                    publish(out, config)
        status(out, state='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--backbones', nargs='+', choices=['ultra', 'trix'], default=['ultra', 'trix'])
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--validation-every', type=int, default=5)
    parser.add_argument('--early-stopping-patience', type=int, default=100, help='Epochs without validation improvement')
    parser.add_argument('--after-run', type=Path)
    parser.add_argument('--study', choices=['query-types', 'scale'])
    parser.add_argument('--reuse-from', type=Path)
    parser.add_argument('--prepared-from', type=Path)
    parser.add_argument('--reference-adapter', type=Path)
    parser.add_argument('--data-scaling-from', type=Path)
    parser.add_argument('--training-multipliers', type=int, nargs='+', default=[4, 16])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', nargs='+')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--cache-mb', type=int, default=512)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        config_path = out/'configuration.json'
        if args.worker:
            config = json.loads(config_path.read_text())
            if sha(__file__) != config['runner'] or _implementation_fingerprint() != config['implementation']:
                raise ValueError('Worker code differs from the recorded snapshot')
            if args.worker == ['prepare']:
                prepare_sources(out, config)
                status(out, state='queued', predecessor=config.get('data_scaling_from') or config.get('after_run'))
            elif args.worker[0] == 'fit':
                fit_worker(out, config, args.worker[1], args.batch_size)
            elif args.worker[0] == 'evaluate':
                _, backbone, variant, dataset = args.worker
                evaluate_worker(out, config, backbone, variant, dataset, args.batch_size, args.cache_mb)
            else:
                raise ValueError('Unknown worker action')
            return
        if config_path.exists():
            config = json.loads(config_path.read_text())
            snapshot = out/'source'/'dicee'/'scripts'/'benchmark_query_adapters.py'
            if Path(__file__).resolve() != snapshot:
                command = [sys.executable, '-u', str(snapshot), '--output', str(out)]
                if args.prepare_only:
                    command.append('--prepare-only')
                os.execve(sys.executable, command,
                          dict(os.environ, PYTHONPATH=str(out/'source')))
        else:
            if min(args.size, args.epochs, args.validation_every, args.early_stopping_patience) < 1:
                raise ValueError('Positive sample, epoch, validation, and patience counts required')
            root = args.root.resolve()
            checkpoints = {'ultra': root/'checkpoints'/'ultra_3g.pth', 'trix': root/'checkpoints'/'trix'/'entity_prediction.pth'}
            config = dict(root=str(root), size=args.size, epochs=args.epochs, backbones=args.backbones,
                          split='valid', sampling_seed=20260923, training_seed=2026090851,
                          validation_every=args.validation_every, early_stopping_patience=args.early_stopping_patience,
                          variants=VARIANTS, implementation=_implementation_fingerprint(), runner=sha(__file__),
                          checkpoints={k: dict(path=str(checkpoints[k]), sha256=sha(checkpoints[k])) for k in args.backbones},
                          sources={k: dict(path=str(root/'KGs'/v/'train.txt'), sha256=sha(root/'KGs'/v/'train.txt'))
                                   for k, v in SOURCE_NAMES.items()},
                          reference_adapter=dict(path=str(args.reference_adapter.resolve()), sha256=sha(args.reference_adapter))
                          if args.reference_adapter else None)
            if args.after_run:
                previous = args.after_run.resolve()
                config.update(after_run=str(previous), after_run_sha256=sha(previous/'configuration.json'))
            if args.data_scaling_from:
                previous = args.data_scaling_from.resolve()
                parent = json.loads((previous/'configuration.json').read_text())
                for key in ('root', 'size', 'epochs', 'split', 'sampling_seed', 'training_seed', 'checkpoints', 'sources'):
                    if config[key] != parent[key]:
                        raise ValueError(f'Data-scaling comparison must preserve {key}')
                for key, default in [('validation_every', 5), ('early_stopping_patience', None)]:
                    if config[key] != parent.get(key, default):
                        raise ValueError(f'Data-scaling comparison must preserve {key}')
                if parent.get('data_scaling_from') or config['backbones'] != parent['backbones']:
                    raise ValueError('Use the original paired adapter sweep as the predecessor')
                if any(n <= 1 for n in args.training_multipliers) or len(set(args.training_multipliers)) != len(args.training_multipliers):
                    raise ValueError('Training multipliers must be distinct integers greater than one')
                base = parent['variants']['scores_negation_wide']
                variants = {'scores_negation_wide': dict(base, train_per_shape=48)}
                for multiplier in sorted(args.training_multipliers):
                    variants[f'scores_negation_wide_{multiplier}x'] = dict(base, train_per_shape=48 * multiplier)
                config.update(data_scaling_from=str(previous), predecessor_sha256=sha(previous/'configuration.json'),
                              reuse_variants=['scores_negation_wide'], variants=variants, reference_adapter=None,
                              preparation_attempts=500_000)
            if args.study:
                if args.data_scaling_from or args.reference_adapter:
                    raise ValueError('Study presets require their own controlled baseline')
                config.update(study=args.study, variants=study_variants(args.study),
                              validation_shapes=list(STANDARD_TRAINING_SHAPES), source_shapes=list(TRAINING_SHAPES),
                              source_validation_per_shape=16, preparation_attempts=500_000)
                coverage = study_variants('query-types')
                config['source_training_counts'] = {
                    shape: max(v['train_per_shape'] for v in coverage.values() if shape in v['shapes'])
                    for shape in TRAINING_SHAPES}
                if args.study == 'scale':
                    if args.reuse_from is None:
                        raise ValueError('The scale study requires --reuse-from query-types-run')
                    previous = args.reuse_from.resolve()
                    parent = json.loads((previous/'configuration.json').read_text())
                    for key in ('root', 'size', 'epochs', 'backbones', 'split', 'sampling_seed', 'training_seed',
                                'checkpoints', 'sources', 'validation_every', 'early_stopping_patience',
                                'validation_shapes', 'source_shapes', 'source_validation_per_shape', 'source_training_counts'):
                        if config[key] != parent[key]:
                            raise ValueError(f'Scale comparison must preserve {key}')
                    if parent.get('study') != 'query-types' or config['variants']['types_2'] != parent['variants']['types_2']:
                        raise ValueError('Scale study requires a matching two-type baseline')
                    config.update(data_scaling_from=str(previous), predecessor_sha256=sha(previous/'configuration.json'),
                                  reuse_variants=['types_2'], comparison_baseline='types_2', source_data_from=str(previous),
                                  score_bank_dir=str(previous/'score-banks'),
                                  prepared_sources={name: sha(previous/'sources'/f'{name}.json') for name in SOURCE_NAMES})
                elif args.reuse_from:
                    raise ValueError('--reuse-from applies only to the scale study')
            if args.prepared_from:
                if args.study != 'query-types':
                    raise ValueError('--prepared-from applies only to the query-type study')
                previous = args.prepared_from.resolve()
                config.update(source_data_from=str(previous),
                              prepared_sources={name: sha(previous/'sources'/f'{name}.json') for name in SOURCE_NAMES})
            shutil.copytree(root/'dicee', out/'source'/'dicee', ignore=shutil.ignore_patterns('__pycache__'))
            write_json(config_path, config)
        if args.prepare_only:
            if not (out/'status.json').exists():
                status(out, state='queued', predecessor=config.get('data_scaling_from') or config.get('after_run'))
            return
        supervise(out, config)
    except BaseException as error:
        if not isinstance(error, SystemExit):
            status(out, state='failed', job=args.worker, error=repr(error))
        raise


if __name__ == '__main__':
    main()
