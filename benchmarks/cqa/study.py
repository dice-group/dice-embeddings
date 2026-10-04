"""Frozen CQA inputs, reference batches, verification gates, and job execution."""

import importlib.metadata
import os
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dicee.query_answering._checkpoint import BenchmarkCheckpoint, checksum, implementation_fingerprint, input_path, read, source_fingerprint, write_json
from dicee.query_answering._query import PLUS_H_SHAPES, compile_query, relation_signature
from dicee.query_answering.context import fingerprint
from dicee.query_answering.datasets import PLUS_H_DATASETS, dataset_spec, inference_graph_for_split, load_benchmark, source_query_groups
from dicee.query_answering.method_evaluation import evaluate_method, save_results
from dicee.query_answering.methods import REFERENCES

from .manifests import validate_entry
from .provenance import validate_training

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def dataset_key(entry, split):
    key = f'{entry["dataset"]}/{split}'
    graph = inference_graph_for_split(entry['dataset'], split, entry.get('inference_graph'))
    return key + '/train+valid' if graph == 'train+valid' else key


def code_identity():
    """Source that can change frozen-study results.

    The dicee package without its command-line scripts, and this harness without
    tests or paper tooling, so local scripts never break verification elsewhere.
    """
    package, harness = SOURCE_ROOT / 'dicee', Path(__file__).resolve().parent
    paths = [path for path in package.rglob('*.py') if path.relative_to(package).parts[0] != 'scripts']
    paths += [path for path in harness.rglob('*.py') if path.relative_to(harness).parts[0] not in ('tests', 'paper')]
    return source_fingerprint(paths, SOURCE_ROOT)


def environment(device):
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return dict(python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__),
                cuda=torch.version.cuda, device=str(device), driver=driver,
                gpu=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
                image_id=os.environ.get('DICEE_IMAGE_ID'), source_commit=os.environ.get('DICEE_SOURCE_COMMIT'),
                source_dirty=os.environ.get('DICEE_SOURCE_DIRTY'),
                packages={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'lightning', 'pandas')},
                threads=torch.get_num_threads(), precision='float32 IEEE; no autocast',
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def make_plan(data, entry, root, query_ids=None):
    """Keep each structure's original conversion order and batch membership."""
    wanted = set(entry['query_types'])
    lookup = {query.query: query for query in data.queries if query.shape in wanted}
    if entry['query_order'] == 'upstream-pickle':
        groups = source_query_groups(data, input_path(root, entry['_data_root']), entry['query_types'])
    else:
        groups = [[query for query in data.queries if query.shape == shape] for shape in entry['query_types']]
        if entry['query_order'] == 'relation':
            groups = [sorted(group, key=lambda query: (relation_signature(compile_query(query.query)), query.query)) for group in groups]
    queries, ends = [], []
    for group in groups:
        for start in range(0, len(group), entry['query_batch_size']):
            queries.extend(group[start:start + entry['query_batch_size']])
            ends.append(len(queries))
    if len(queries) != len(lookup) or len(set(queries)) != len(queries):
        raise ValueError('Reference order does not cover each selected query exactly once')
    return dict(queries=[query_ids[query.query] if query_ids is not None else query.identity for query in queries], batch_ends=ends,
                query_counts=dict(Counter(query.shape for query in queries)), order=entry['query_order'])


def freeze(manifest, output, input_root):
    """Pin inputs and plans without loading model weights or running test inference."""
    manifest = read(manifest) if isinstance(manifest, (str, Path)) else manifest
    if (set(manifest) - {'answer_filter'} != {'version', 'data_root', 'entries', 'seed', 'threads', 'archives'}
            or manifest['version'] != 1 or manifest.get('answer_filter', 'released') not in ('released', 'corrected')):
        raise ValueError('Expected version, data_root, entries, seed, threads, archives')
    if type(manifest['seed']) is not int or type(manifest['threads']) is not int or manifest['threads'] < 1:
        raise ValueError('Invalid seed or thread count')
    entries = manifest['entries']
    if not entries or len({entry['id'] for entry in entries}) != len(entries):
        raise ValueError('Entries need unique IDs')
    for entry in entries:
        validate_entry(entry)
    if manifest.get('answer_filter') == 'corrected' and any(e['dataset'] not in PLUS_H_DATASETS for e in entries):
        raise ValueError('Corrected answer filters are only defined for the +H suite')
    output = Path(output)
    if (output / 'bundle.json').exists():
        raise FileExistsError('Use a new bundle directory to change a frozen study')
    pins = {}

    def pin(relative):
        pins[relative] = checksum(input_path(input_root, relative))

    for entry in entries:
        pin(entry['checkpoint'])
        if entry.get('training'):
            pin(entry['training'])
        _, training_pins = validate_training(entry, input_root, manifest['data_root'], file_hashes=pins)
        pins.update(training_pins)
        for relative in entry.get('adapters', {}).values():
            pin(relative)
        if entry.get('verification'):
            pin(entry['verification'])
    datasets, plans, corrections = {}, {}, {}
    for name in dict.fromkeys(entry['dataset'] for entry in entries):
        for split in ('valid', 'test'):
            matching = [entry for entry in entries if entry['dataset'] == name]
            policies = list(dict.fromkeys(inference_graph_for_split(name, split, entry.get('inference_graph')) for entry in matching))
            data = load_benchmark(input_path(input_root, manifest['data_root']), name, split=split,
                                  inference_graph=policies[0])
            query_ids = {query.query: query.identity for query in data.queries}
            key = dataset_key(dict(dataset=name, inference_graph=policies[0]), split)
            datasets[key] = dict(context=data.context.identity, candidates=fingerprint(data.candidates),
                                 metadata=data.metadata, queries=len(data.queries),
                                 query_ids=fingerprint(sorted(query_ids.values())))
            for filename, digest in data.metadata['files'].items():
                relative = str(Path(manifest['data_root']) / dataset_spec(name)[1] / filename)
                pins[relative] = digest
            if split == 'test' and manifest.get('answer_filter') == 'corrected':
                from dicee.query_answering.datasets import audit_plus_h_filters
                audit = audit_plus_h_filters(input_path(input_root, manifest['data_root']), data)
                relative = f'filters/{name}.json'
                write_json(output / relative, audit)
                corrections[name] = dict(path=relative, sha256=checksum(output / relative), per_shape=audit['per_shape'])
                for filename, digest in audit['source_files'].items():
                    pins[str(Path(manifest['data_root']) / dataset_spec(name)[1] / filename)] = digest
            for policy in policies[1:]:
                variant = load_benchmark(input_path(input_root, manifest['data_root']), name, split=split,
                                         inference_graph=policy)
                variant_key = dataset_key(dict(dataset=name, inference_graph=policy), split)
                datasets[variant_key] = dict(datasets[key], context=variant.context.identity, metadata=variant.metadata)
                for filename, digest in variant.metadata['files'].items():
                    pins[str(Path(manifest['data_root']) / dataset_spec(name)[1] / filename)] = digest
            prepared_plans = {}
            for entry in entries:
                if entry['dataset'] != name:
                    continue
                layout = fingerprint({key: entry[key] for key in ('query_types', 'query_batch_size', 'query_order')})
                if layout not in prepared_plans:
                    plan = make_plan(data, dict(entry, _data_root=manifest['data_root']), input_root, query_ids)
                    relative = f'plans/{name}-{split}-{layout[:16]}.json'
                    write_json(output / relative, plan)
                    prepared_plans[layout] = dict(path=relative, sha256=checksum(output / relative))
                plans[f'{entry["id"]}/{split}'] = prepared_plans[layout]
    bundle = dict(version=1, manifest=manifest, files=pins, datasets=datasets, plans=plans, filter_corrections=corrections,
                  source_sha256=code_identity(), implementation_sha256=implementation_fingerprint(),
                  created=time.time(), python=platform.python_version(),
                  protocol=dict(tie_policies=['sort', 'expected'], tie_equality='exact',
                                filtering=manifest.get('answer_filter', 'released'),
                                averaging='answers per query; queries per type; equal types',
                                observed_graphs=('entry-specific test graph; validation uses train; pinned by context hashes'
                                                 if all(e['dataset'] in PLUS_H_DATASETS for e in entries) else
                                                 'published dataset-specific validation/test graphs; pinned by context hashes'),
                                candidates='dataset-specific; pinned by candidate hashes'))
    write_json(output / 'bundle.json', bundle)
    write_json(output / 'bundle.sha256.json', {'sha256': checksum(output / 'bundle.json')})
    return bundle


def verify_bundle(directory, input_root):
    directory = Path(directory)
    if checksum(directory / 'bundle.json') != read(directory / 'bundle.sha256.json')['sha256']:
        raise ValueError('Bundle checksum mismatch')
    bundle = read(directory / 'bundle.json')
    if bundle['source_sha256'] != code_identity():
        raise ValueError('Source differs from the frozen bundle; freeze a new study')
    for relative, digest in bundle['files'].items():
        if checksum(input_path(input_root, relative)) != digest:
            raise ValueError(f'Input checksum mismatch: {relative}')
    for plan in bundle['plans'].values():
        if checksum(directory / plan['path']) != plan['sha256']:
            raise ValueError('Query plan checksum mismatch')
    for correction in bundle.get('filter_corrections', {}).values():
        if checksum(directory / correction['path']) != correction['sha256']:
            raise ValueError('Answer-filter correction checksum mismatch')
    return bundle


def bundle_entry(bundle, entry_id):
    """The frozen entry ``entry_id`` of ``bundle``."""
    matches = [entry for entry in bundle['manifest']['entries'] if entry['id'] == entry_id]
    if len(matches) != 1:
        raise ValueError(f'No frozen entry {entry_id!r} in this bundle')
    return matches[0]


def run_job(bundle_dir, entry_id, input_root, output, *, device='cpu', phase='pilot', pilot_queries=2,
            on_prediction=None, probe_only=False, on_control_prediction=None):
    """Pilot validation or full test; never choose settings from test labels."""
    from dicee.models._inference import float32_precision_backends
    bundle = verify_bundle(bundle_dir, input_root)
    entry = bundle_entry(bundle, entry_id)
    if phase not in ('pilot', 'test'):
        raise ValueError('Choose the pilot or test phase')
    if probe_only and phase != 'pilot':
        raise ValueError('Probe selection is only available for validation pilots')
    if phase == 'test':
        validate_training(entry, input_root, bundle['manifest']['data_root'], final=True, file_hashes=bundle['files'])
        if entry.get('blockers') or (entry['method'].endswith('-adapter') and not entry.get('finalized')):
            raise ValueError(f'Entry is not ready for final testing: {entry.get("blockers", ["adapter recipe is provisional"])}')
        if not entry.get('verification'):
            raise ValueError('Final testing requires pinned reference parity evidence')
    seed, threads = bundle['manifest']['seed'], bundle['manifest']['threads']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    for backend in float32_precision_backends():
        backend.fp32_precision = 'ieee'
    env = environment(device)
    if phase == 'test':
        evidence = read(input_path(input_root, entry['verification']))
        expected = dict(entry_sha256=fingerprint({key: value for key, value in entry.items() if key != 'verification'}),
                        source_sha256=bundle['source_sha256'], environment=env,
                        inputs_sha256=entry_inputs_identity(bundle, entry))
        if not evidence.get('passed') or any(evidence.get(key) != value for key, value in expected.items()):
            raise ValueError('Parity evidence does not match this entry, source, and execution hardware')
    output = Path(output)
    identity = dict(bundle=checksum(Path(bundle_dir) / 'bundle.json'), entry=entry_id, phase=phase,
                    pilot_queries=pilot_queries if phase == 'pilot' else None, probe_only=probe_only, environment=env)
    with BenchmarkCheckpoint(output / 'job', identity):
        if (output / 'result.json').exists():
            result = read(output / 'result.json')
            if result.get('benchmark_run', result.get('paper_run')) != identity:
                raise ValueError('Result identity mismatch')
            if checksum(output / 'ranks.sqlite3') != result['rank_trace_sha256']:
                raise ValueError('Completed rank trace checksum mismatch')
            for name, expected_checksum in result.get('comparison_results', {}).items():
                comparison = read(output / name / 'result.json')
                if (checksum(output / name / 'result.json') != expected_checksum or
                        checksum(output / name / 'ranks.sqlite3') != comparison['rank_trace_sha256']):
                    raise ValueError('Completed comparison checksum mismatch')
            return result
        write_json(output / 'status.json', dict(state='loading', updated=time.time(), **identity))
        started = time.monotonic()
        split = 'valid' if phase == 'pilot' else 'test'
        data = load_benchmark(input_path(input_root, bundle['manifest']['data_root']), entry['dataset'], split=split,
                              inference_graph=entry.get('inference_graph'))
        if data.context.identity != bundle['datasets'][dataset_key(entry, split)]['context']:
            raise ValueError('Loaded inference graph differs from the frozen graph')
        lookup = {query.identity: query for query in data.queries}
        filter_corrections = None
        if phase == 'test' and bundle['manifest'].get('answer_filter') == 'corrected':
            from dicee.query_answering.datasets import corrected_answer_filters
            correction = bundle['filter_corrections'][entry['dataset']]
            filter_corrections = corrected_answer_filters(data, read(Path(bundle_dir) / correction['path']))
        query_lookup = {query.query: query for query in data.queries}
        prediction_hook = (lambda query, scores: on_prediction(query_lookup[query], scores)) if on_prediction else None
        control_hook = (lambda query, scores: on_control_prediction(query_lookup[query], scores)) if on_control_prediction else None
        saved_plan = read(Path(bundle_dir) / bundle['plans'][f'{entry_id}/{split}']['path'])
        plan = [lookup[key] for key in saved_plan['queries']]
        ends = saved_plan['batch_ends']
        reference_batches = {}
        if phase == 'pilot':
            if type(pilot_queries) is not int or pilot_queries < 1:
                raise ValueError('Positive pilot query budget required')
            selected, selected_ends, counts, start = [], [], Counter(), 0
            for end in ends:
                batch = plan[start:end]
                if counts[batch[0].shape] < pilot_queries:
                    probes = batch[:pilot_queries - counts[batch[0].shape]] if probe_only else batch
                    reference_batches[probes[0].query] = [query.query for query in batch]
                    selected.extend(probes)
                    selected_ends.append(len(selected))
                    counts[batch[0].shape] += len(probes)
                start = end
            plan, ends = selected, selected_ends
        loading = time.monotonic() - started
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        write_json(output / 'status.json', dict(state='evaluating', updated=time.time(), **identity))
        try:
            recipe = {key: entry[key] for key in ('method', 'dataset', 'query_types', 'query_batch_size')}
            recipe.update(checkpoint=str(input_path(input_root, entry['checkpoint'])),
                          root=str(input_path(input_root, bundle['manifest']['data_root'])),
                          options=dict(entry['options']))
            if 'inference_graph' in entry:
                recipe['inference_graph'] = entry['inference_graph']
            if entry['method'] in REFERENCES:
                recipe['vocabulary_dataset'] = entry['dataset']
                if 'per_shape' in recipe['options']:
                    recipe['per_shape'] = recipe['options'].pop('per_shape')
            else:
                recipe.update(adapters={name: str(input_path(input_root, path)) for name, path in entry['adapters'].items()},
                              operators=entry['operators'], selection_protocol=entry['selection_protocol'],
                              adapter_ablation=entry.get('adapter_ablation', False))
            report = evaluate_method(recipe, output=output, data=data, split=split, device=device, seed=seed,
                                     query_plan=plan, batch_ends=ends, rank_trace_path=output / 'ranks.sqlite3', provenance=identity,
                                     query_order='published' if entry['method'] in REFERENCES else entry['query_order'],
                                     write_result=False, on_prediction=prediction_hook, reference_batches=reference_batches,
                                     on_control_prediction=control_hook if entry['method'] not in REFERENCES else None,
                                     filter_corrections=filter_corrections)
            profile = ('bounded' if not entry['options'].get('reference_batching', False) else 'reference') if entry['method'] in ('cqd', 'cqd-hybrid') else ('reference' if entry['method'] in REFERENCES else 'native')
            common = dict(execution_profile=profile, reference=entry['reference'],
                          graph_ablation=entry.get('graph_ablation'),
                          graph_recipe_sha256=fingerprint(dict(
                              recipe={key: value for key, value in entry.items()
                                  if key not in ('id', 'inference_graph', 'graph_ablation', 'verification', 'reference')},
                              inputs=entry_inputs_identity(bundle, entry), source=bundle['source_sha256'],
                              filters=bundle.get('filter_corrections', {}).get(entry['dataset']),
                              plan=bundle['plans'][f'{entry_id}/{split}']['sha256'], seed=seed, environment=env)), coverage=dict(
                selected_types=entry['query_types'], expected_types=data.metadata['expected_query_types'],
                complete_benchmark_types=set(entry['query_types']) == set(data.metadata['expected_query_types']),
                complete_16_types=set(entry['query_types']) == set(PLUS_H_SHAPES)),
                dataset_loading_seconds=loading, total_wall_seconds=time.monotonic() - started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None)
            save_results(report, output, run=identity, metadata=common)
            write_json(output / 'status.json', dict(state='complete', updated=time.time(), **identity))
            return report
        except BaseException as error:
            write_json(output / 'status.json', dict(state='failed', error=str(error), updated=time.time(), **identity))
            raise


def entry_inputs_identity(bundle, entry):
    selected = {entry['checkpoint'], *entry.get('adapters', {}).values()}
    if entry.get('training'):
        selected.add(entry['training'])
    prefix = str(Path(bundle['manifest']['data_root']) / dataset_spec(entry['dataset'])[1]) + '/'
    return fingerprint({key: value for key, value in bundle['files'].items() if key in selected or key.startswith(prefix)})


def integration_passed(path, bundle, entry):
    """Whether ``path`` holds a passing KGFM integration check of this frozen entry, inputs and source."""
    if not Path(path).is_file():
        return False
    evidence = read(path)
    return bool(evidence.get('passed') and evidence.get('source_sha256') == bundle['source_sha256']
                and evidence.get('entry_sha256') == fingerprint({k: v for k, v in entry.items() if k != 'verification'})
                and evidence.get('inputs_sha256') == entry_inputs_identity(bundle, entry))


def parity_passed(path, bundle, entry, reference, comparison_reference=None):
    """Whether ``path`` holds passing parity evidence of this frozen entry, inputs and source against these oracles."""
    if not integration_passed(path, bundle, entry):
        return False
    evidence = read(path)
    comparison = evidence.get('upstream_comparison', {}).get('reference_sha256')
    return (evidence.get('reference_sha256') == checksum(reference)
            and comparison == (checksum(comparison_reference) if comparison_reference is not None else None))
