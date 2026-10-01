"""Benchmark suites and their public recipes, resolved without PyTorch.

A recipe (manifest entry) fixes a method, checkpoint, dataset, query types and
execution options. Resolution only selects and annotates entries; it never
opens datasets or models, so the host launcher can run it before Docker.

Public manifests group entries into ``recipes``: each entry is the recipe's
``defaults`` updated by the entry's own fields, recursively for objects, and
``{dataset}`` in an ID is replaced by the entry's dataset. Manifests may also
list resolved ``entries`` directly.
"""

import copy
import importlib.util
import json
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUITES = {
    'plus_h': dict(title='+H benchmark', answer_filter='corrected'),
    'ultraquery': dict(title='UltraQuery benchmark (released PyG)', answer_filter='released'),
}
KGFM_EXECUTION = {
    # H100 / H100 NVL starting point for ULTRA/TRIX adapters: larger atomic
    # batches and bounded GPU caches. Native methods need no profile.
    'h100': dict(row_batch_size=16, backend_batch_size=16, cache_bytes=4 * 2**30, raw_cache_bytes=2 * 2**30,
                 raw_cache_device='model', relation_cache_mb=256, projection_cache_mb=512),
}


@cache
def catalog():
    """The library's benchmark catalog, loaded without importing dicee or PyTorch."""
    spec = importlib.util.spec_from_file_location('_dicee_cqa_catalog', REPO / 'dicee/query_answering/catalog.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suite_directory(suite):
    if suite not in SUITES:
        raise ValueError(f'Choose a suite: {", ".join(SUITES)}')
    return REPO / 'benchmarks' / suite


def default_recipes(suite):
    """Public recipe files of a suite: native baselines, then KGFM adapters."""
    directory = suite_directory(suite)
    return [path for path in (directory / 'baselines.json', directory / 'kgfm_adapters.json') if path.is_file()]


def merge(defaults, override):
    """``defaults`` updated by ``override``; objects merge recursively, other values are replaced."""
    result = copy.deepcopy(defaults)
    for key, value in override.items():
        nested = isinstance(value, dict) and isinstance(result.get(key), dict)
        result[key] = merge(result[key], value) if nested else copy.deepcopy(value)
    return result


def expand(recipes):
    """Resolved entries of recipe groups, in order."""
    entries = []
    for recipe in recipes:
        for override in recipe['entries']:
            entry = merge(recipe['defaults'], override)
            entry['id'] = entry['id'].replace('{dataset}', entry['dataset'])
            entries.append(entry)
    return entries


def read_manifest(path):
    """A recipe manifest with resolved entries, or the manifest pinned in a frozen bundle."""
    value = json.loads(Path(path).read_text())
    value = value.get('manifest', value)
    if 'recipes' in value:
        value = {key: item for key, item in value.items() if key != 'recipes'} | {'entries': expand(value['recipes'])}
    return value


def select_entries(entries, *, entry_ids=None, methods=None, datasets=None, query_types=None, atomic_negation=False):
    """Resolve recipe subsets before opening datasets or model checkpoints."""
    shapes_by_dataset = catalog().query_types_for_dataset
    all_shapes = catalog().PLUS_H_SHAPES
    identifiers = [entry.get('id', f'{entry["method"]}-{entry["dataset"]}') for entry in entries]
    if len(set(identifiers)) != len(identifiers) or any(Path(name).name != name or name in ('', '.', '..') for name in identifiers):
        raise ValueError('Manifest entry IDs must be unique, plain directory names')

    def selection(values, available, label, allow_all=True):
        if values is None or (allow_all and values == ['all']):
            return None
        if not values or len(set(values)) != len(values):
            raise ValueError(f'Select distinct {label}')
        missing = set(values) - set(available)
        if missing:
            raise ValueError(f'Select distinct {label}; unknown: {", ".join(sorted(missing))}. Available: {", ".join(sorted(available))}')
        return set(values)

    ids = selection(entry_ids, identifiers, 'entry IDs', allow_all=False)
    wanted_methods = selection(methods, {e['method'] for e in entries}, 'methods')
    wanted_datasets = selection(datasets, {e['dataset'] for e in entries}, 'datasets')
    selected = [copy.deepcopy(entry) for name, entry in zip(identifiers, entries)
                if (ids is None or name in ids) and (wanted_methods is None or entry['method'] in wanted_methods)
                and (wanted_datasets is None or entry['dataset'] in wanted_datasets)]
    if not selected:
        raise ValueError('No manifest entries match the selected methods, datasets, and entry IDs')
    groups = {'all', 'epfo', 'negation'}
    if query_types is not None:
        if not query_types or len(set(query_types)) != len(query_types):
            raise ValueError('Select distinct query types or groups: all, epfo, negation')
        unknown = set(query_types) - set(all_shapes) - groups
        if unknown or ('all' in query_types and len(query_types) != 1):
            raise ValueError('Select supported query types or groups: all, epfo, negation; use all alone. '
                             f'Available types: {", ".join(all_shapes)}')
    for entry in selected:
        changed = False
        if query_types is not None:
            available = shapes_by_dataset(entry['dataset'])
            expanded_groups = {'all': available, 'epfo': [s for s in available if 'n' not in s],
                               'negation': [s for s in available if 'n' in s]}
            shapes = []
            for shape in query_types:
                shapes.extend(expanded_groups.get(shape, [shape]))
            shapes = list(dict.fromkeys(shapes))
            if set(shapes) - set(available):
                raise ValueError(f'Query types {shapes} are not supported by {entry["dataset"]}')
            changed = shapes != entry.get('query_types')
            entry['query_types'] = shapes
            for owner in (entry, entry.get('options', {})):
                if 'per_shape' in owner:
                    owner['per_shape'] = {s: value for s, value in owner['per_shape'].items() if s in shapes}
            if 'operators' in entry:
                missing = set(shapes) - entry['operators'].keys()
                if missing:
                    raise ValueError(f'{entry.get("id", entry["method"])} has no operator/adapter recipe for {sorted(missing)}')
                entry['operators'] = {s: entry['operators'][s] for s in shapes}
        if entry['method'] in ('cqd', 'cqd-hybrid'):
            options = entry.setdefault('options', {})
            if atomic_negation and not options.get('atomic_negation'):
                options['atomic_negation'] = True
                changed = True
            if any('n' in s for s in entry.get('query_types', [])) and not options.get('atomic_negation'):
                raise ValueError('CQD negated query types require atomic_negation=true or --atomic-negation')
            if changed and options.get('atomic_negation') and 'reference' in entry:
                entry['reference'].setdefault('notes', []).append(
                    'CQD-A signed-atom negation with +H scoring.')
        if changed:
            entry.pop('verification', None)
            if 'reference' in entry:
                entry['reference']['status'] = 'custom query selection or executor; independent validation required'
    return selected


def validate_entry(entry):
    required = {'id', 'method', 'dataset', 'checkpoint', 'options', 'query_types', 'query_batch_size', 'query_order', 'reference'}
    optional = {'adapters', 'operators', 'selection_protocol', 'finalized', 'blockers', 'verification', 'adapter_ablation',
                'inference_graph', 'graph_ablation', 'training'}
    if required - entry.keys() or entry.keys() - required - optional:
        raise ValueError(f'Invalid benchmark entry fields: {entry.get("id")}')
    if (entry['method'] not in catalog().METHODS
            or Path(entry['id']).name != entry['id'] or entry['id'] in ('', '.', '..')):
        raise ValueError('Invalid benchmark method, dataset, or entry ID')
    shapes = entry['query_types']
    if not shapes or len(set(shapes)) != len(shapes) or set(shapes) - set(catalog().query_types_for_dataset(entry['dataset'])):
        raise ValueError('Invalid query types for the selected dataset')
    catalog().check_recipe(entry, shapes)
    if entry['query_order'] not in ('canonical', 'relation', 'upstream-pickle'):
        raise ValueError('Choose canonical, relation, or upstream-pickle query ordering')
    if entry.get('inference_graph') not in (None, 'train', 'train+valid'):
        raise ValueError('Inference graph must be train or train+valid')
    if entry.get('graph_ablation') is not None and not isinstance(entry['graph_ablation'], str):
        raise ValueError('Graph ablation must identify the shared frozen recipe')
    if entry['method'] == 'clmpt' and entry['query_order'] != 'upstream-pickle':
        raise ValueError('CLMPT requires frozen reference conversion order')


def prepare_manifest(paths, *, profile='bounded', entries=None, methods=None, datasets=None, query_types=None,
                     atomic_negation=False, observed_facts=None, inference_graphs=None, answer_filter='corrected', suite='plus_h',
                     hardware_profile='default', kgfm_batch_size=None):
    """Merge public recipes without changing beams or per-shape operator settings."""
    manifests = [read_manifest(path) for path in paths]
    if not manifests or profile not in ('bounded', 'reference'):
        raise ValueError('Choose a bounded or reference profile and at least one manifest')
    if hardware_profile not in ('default', *KGFM_EXECUTION):
        raise ValueError('Choose a default or h100 hardware profile')
    if kgfm_batch_size is not None and (type(kgfm_batch_size) is not int or kgfm_batch_size < 1):
        raise ValueError('KGFM batch size must be a positive integer')
    if answer_filter not in ('released', 'corrected'):
        raise ValueError('Answer filter must be released or corrected')
    if suite not in SUITES:
        raise ValueError(f'Choose a suite: {", ".join(SUITES)}')
    if suite == 'ultraquery' and (answer_filter != 'released' or inference_graphs is not None):
        raise ValueError('UltraQuery uses released answer filters and dataset-defined inference graphs')
    manifest = {key: value for key, value in manifests[0].items() if key != 'entries'}
    combined = []
    for source in manifests:
        if {key: value for key, value in source.items() if key != 'entries'} != manifest:
            raise ValueError('Source manifests disagree on dataset, seed, threads, or archive metadata')
        combined.extend(source['entries'])
    if observed_facts is not None:
        modes = ('none', 'atomic', 'both') if observed_facts == ['all'] else observed_facts
        if not modes or len(modes) != len(set(modes)) or set(modes) - {'none', 'atomic', 'both'}:
            raise ValueError('Select distinct observed-fact modes, or all')
        variants = []
        for entry in combined:
            if not entry['method'].endswith('-adapter'):
                continue
            for mode in modes:
                variant = copy.deepcopy(entry)
                variant['id'] += f'-facts-{mode}'
                variant['options']['observed_facts'] = mode
                variant['reference']['status'] = 'observed-fact ablation; verification pending'
                variant['reference'].setdefault('notes', []).append(
                    f'Inference observed-fact mode: {mode}. Keep the trained weights and graph inputs fixed; '
                    'apply this mode to both learned and identity calibration. No adapter refitting.')
                variants.append(variant)
        if not variants:
            raise ValueError('Observed-fact ablation requires KGFM entries')
        combined = variants
    if inference_graphs is not None:
        if (not inference_graphs or len(set(inference_graphs)) != len(inference_graphs)
                or set(inference_graphs) - {'train', 'train+valid'}):
            raise ValueError('Select distinct inference graphs: train, train+valid')
        variants = []
        for entry in combined:
            if entry['method'] in catalog().GRAPH_INDEPENDENT_METHODS:
                entry['inference_graph'] = 'train'
                entry['reference']['graph_independent'] = True
                variants.append(entry)
                continue
            for graph in inference_graphs:
                variant = copy.deepcopy(entry)
                variant['graph_ablation'] = entry['id']
                variant['inference_graph'] = graph
                variant['id'] += '-graph-' + graph.replace('+', '-')
                variants.append(variant)
        combined = variants
    ids = [entry['id'] for entry in combined]
    if len(ids) != len(set(ids)):
        raise ValueError('Source manifests contain duplicate entry IDs')
    manifest['entries'] = select_entries(combined, entry_ids=entries, methods=methods, datasets=datasets,
                                         query_types=query_types, atomic_negation=atomic_negation)
    allowed_datasets = catalog().PLUS_H_DATASETS if suite == 'plus_h' else catalog().BENCHMARK_DATASETS
    if any(entry['dataset'] not in allowed_datasets for entry in manifest['entries']):
        raise ValueError(f'Selected entries must belong to the {suite} dataset suite')
    manifest['answer_filter'] = answer_filter
    kgfm_methods = catalog().KGFM_ADAPTERS
    if kgfm_batch_size is not None and not any(e['method'] in kgfm_methods for e in manifest['entries']):
        raise ValueError('KGFM batch size requires at least one selected KGFM adapter')
    for entry in manifest['entries']:
        entry.pop('verification', None)
        if profile == 'bounded' and entry['method'] in ('cqd', 'cqd-hybrid'):
            entry['options'].update(reference_batching=False, row_batch_size=32, final_batch_size=32)
            entry['reference']['status'] = 'bounded execution profile; independent validation required'
            entry['reference'].setdefault('notes', []).append(
                'Bounded profile: reference_batching=false, row_batch_size=32, and final_batch_size=32; beams and per-shape '
                'recipes retained. Requires independent bounded-profile validation scores, orders, and ties.')
        kgfm = entry['method'] in kgfm_methods
        if hardware_profile != 'default' and kgfm:
            entry['options'].update(KGFM_EXECUTION[hardware_profile])
            entry['reference']['status'] = 'H100 execution profile; independent validation required'
            entry['reference'].setdefault('notes', []).append(
                'H100 / H100 NVL starting configuration, not a measured optimum. Float32 IEEE, query batches, beams, '
                'operators and observed facts retained. Validate memory, scores, orders and ties on the target GPU.')
        if kgfm and kgfm_batch_size is not None:
            entry['options'].update(row_batch_size=kgfm_batch_size, backend_batch_size=kgfm_batch_size)
            entry['reference']['status'] = 'custom KGFM atomic batch; independent validation required'
            entry['reference'].setdefault('notes', []).append(
                f'Atomic row and backbone batch size {kgfm_batch_size}; frozen before verification.')
    return manifest
