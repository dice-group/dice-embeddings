"""Evaluate selected entries of a portable study manifest directly, without a frozen bundle."""

import json
from pathlib import Path

from dicee.query_answering._checkpoint import checksum, input_path
from dicee.query_answering.method_evaluation import evaluate_method
from dicee.query_answering.methods import REFERENCES

from .provenance import validate_training


def evaluate_manifest(manifest, *, input_root, output, device='cpu', split='test', limit=None, seed=None, threads=None):
    """Evaluate each entry into ``output/ID``, resolving its inputs below ``input_root``."""
    seed = manifest.get('seed', 0) if seed is None else seed
    threads = manifest.get('threads') if threads is None else threads
    for entry in manifest['entries']:
        entry = dict(entry)
        name, query_order = entry['id'], entry.pop('query_order', 'relation')
        record, _ = validate_training(entry, input_root, manifest['data_root'])
        for key in ('reference', 'finalized', 'blockers', 'verification', 'graph_ablation'):
            entry.pop(key, None)
        entry['root'] = str(input_path(input_root, manifest['data_root']))
        entry['checkpoint'] = str(input_path(input_root, entry['checkpoint']))
        provenance = None
        if entry.get('training'):
            provenance = dict(training=record, training_sha256=checksum(input_path(input_root, entry.pop('training'))))
        entry['options'] = dict(entry.get('options', {}))
        if entry['method'] in REFERENCES:
            entry['vocabulary_dataset'] = entry['dataset']
            if 'per_shape' in entry['options']:
                entry['per_shape'] = entry['options'].pop('per_shape')
        else:
            entry['adapters'] = {key: str(input_path(input_root, path)) for key, path in entry['adapters'].items()}
        data = filter_corrections = None
        if manifest.get('answer_filter') == 'corrected' and split == 'test':
            from dicee.query_answering.datasets import audit_plus_h_filters, corrected_answer_filters, load_benchmark
            data = load_benchmark(entry['root'], entry['dataset'], split='test', query_types=entry.get('query_types'),
                                  inference_graph=entry.get('inference_graph'))
            filter_corrections = corrected_answer_filters(data, audit_plus_h_filters(entry['root'], data))
        report = evaluate_method(entry, output=Path(output) / name, device=device, split=split, limit=limit, seed=seed,
                                 threads=threads, query_order=query_order, data=data, filter_corrections=filter_corrections,
                                 provenance=provenance)
        print(json.dumps(dict(id=name, queries=report['queries'], seconds=report['seconds'], averages=report['averages'])), flush=True)
