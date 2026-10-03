"""Validate independently exported reference scores on the intended hardware."""

import tempfile
from collections import defaultdict

import torch

from dicee.query_answering._checkpoint import write_json
from dicee.query_answering.benchmark import METRICS, QueryMetrics
from dicee.query_answering.context import fingerprint
from dicee.query_answering.datasets import BENCHMARK_DATASETS
from dicee.query_answering.methods import REFERENCES

from .study import bundle_entry, checksum, dataset_key, entry_inputs_identity, run_job, verify_bundle


def verify_predictions(bundle_dir, entry_id, input_root, reference, output, *, device='cpu', atol=3e-5, rtol=3e-5,
                       comparison_reference=None):
    bundle = verify_bundle(bundle_dir, input_root)
    entry = bundle_entry(bundle, entry_id)
    oracle = torch.load(reference, map_location='cpu', weights_only=True)
    expected = dict(version=1, entry_sha256=fingerprint({key: value for key, value in entry.items() if key != 'verification'}),
                    inputs_sha256=entry_inputs_identity(bundle, entry),
                    validation_plan_sha256=bundle['plans'][f'{entry_id}/valid']['sha256'])
    if any(oracle.get(key) != value for key, value in expected.items()):
        raise ValueError('Reference does not match the frozen entry, inputs, and validation batches')
    if entry['method'] in REFERENCES and oracle.get('reference_commit') != REFERENCES[entry['method']][1]:
        raise ValueError('Reference commit mismatch')
    graphs = {split: bundle['datasets'][dataset_key(entry, split)]['context'] for split in ('valid', 'test')}
    if (entry['method'] in REFERENCES and ('inference_graph' in entry or entry['dataset'] in BENCHMARK_DATASETS)
            and oracle.get('graphs') != graphs):
        raise ValueError('Reference inference graphs differ from the frozen validation/test protocol')
    control = None
    comparisons = {}
    if comparison_reference is not None:
        control = torch.load(comparison_reference, map_location='cpu', weights_only=True)
        fields = ('inputs_sha256', 'validation_plan_sha256', 'reference_commit', 'environment')
        if any(control.get(key) != oracle.get(key) for key in fields):
            raise ValueError('Comparison reference uses different inputs, query batches, upstream code, or hardware')
        if set(control['scores']) != set(oracle['scores']) or set(control['orders']) != set(oracle['orders']):
            raise ValueError('Comparison reference query coverage differs')
    results = {}

    def compare(query, actual):
        key = query.identity
        gold = oracle['scores'][key].to(actual.device)
        order = oracle['orders'][key].to(actual.device)
        if gold.shape != actual.shape or order.shape != actual.shape:
            raise ValueError('Reference must score and rank every candidate')
        if not torch.equal(order.sort().values, torch.arange(len(order), device=order.device)):
            raise ValueError('Reference order is not a candidate permutation')
        if not bool((gold[order][:-1] >= gold[order][1:]).all()):
            raise ValueError('Reference order contradicts its scores')
        scores_match = torch.allclose(actual, gold, atol=atol, rtol=rtol)
        ranks_match = torch.equal(actual.argsort(descending=True), order)
        ties_match = torch.equal(actual[order][1:] == actual[order][:-1], gold[order][1:] == gold[order][:-1])
        finite = torch.isfinite(actual) & torch.isfinite(gold)
        errors = (actual[finite] - gold[finite]).abs()
        results[key] = dict(shape=query.shape, max_error=float(errors.max()) if errors.numel() else 0.,
                            scores_match=scores_match, ranks_match=ranks_match, ties_match=ties_match,
                            passed=scores_match and ranks_match and ties_match)
        if control is not None:
            previous = control['scores'][key].to(actual.device)
            previous_order = control['orders'][key].to(actual.device)
            if previous.shape != actual.shape or not torch.equal(previous.argsort(descending=True), previous_order):
                raise ValueError('Comparison reference has invalid candidate scores or order')
            finite = torch.isfinite(actual) & torch.isfinite(previous)
            errors = (actual[finite] - previous[finite]).abs()
            metric = QueryMetrics(len(actual))
            current_metrics = metric.evaluate(actual, query.easy, query.hard)[0]
            previous_metrics = metric.evaluate(previous, query.easy, query.hard)[0]
            comparisons[key] = dict(shape=query.shape, max_error=float(errors.max()) if errors.numel() else 0.,
                ranks_match=torch.equal(actual.argsort(descending=True), previous_order),
                ties_match=torch.equal(actual[previous_order][1:] == actual[previous_order][:-1],
                                       previous[previous_order][1:] == previous[previous_order][:-1]),
                current=current_metrics, comparison=previous_metrics)

    with tempfile.TemporaryDirectory(prefix='dicee-cqa-parity-') as temporary:
        report = run_job(bundle_dir, entry_id, input_root, temporary, device=device, phase='pilot',
                         pilot_queries=oracle['pilot_queries'], on_prediction=compare,
                         probe_only=oracle.get('probe_only', False))
    env = report['benchmark_run']['environment']
    hardware_keys = ('python', 'torch', 'cuda', 'gpu', 'driver', 'precision', 'threads')
    if any(oracle.get('environment', {}).get(key) != env[key] for key in hardware_keys):
        raise ValueError('Reference was produced with different hardware, precision, or runtime versions')
    if set(results) != set(oracle['scores']) or set(results) != set(oracle['orders']):
        raise ValueError('Reference query coverage differs from the frozen pilot batches')
    if {row['shape'] for row in results.values()} != set(entry['query_types']):
        raise ValueError('Reference omits a requested query type')
    evidence = dict(entry_sha256=expected['entry_sha256'], inputs_sha256=expected['inputs_sha256'],
                    source_sha256=bundle['source_sha256'], environment=env, reference_sha256=checksum(reference),
                    reference_commit=oracle.get('reference_commit'), queries=results,
                    graphs=graphs,
                    reference_adjustments=oracle.get('runtime_adjustments', []),
                    probe_only=oracle.get('probe_only', False),
                    scope='paired full-candidate validation predictions, exact frozen batch membership',
                    atol=atol, rtol=rtol, passed=all(value['passed'] for value in results.values()))
    if control is not None:
        groups = defaultdict(list)
        for row in comparisons.values():
            groups[row['shape']].append(row)
        differences = {shape: {policy: {metric: sum(row['current'][policy][metric] - row['comparison'][policy][metric]
                       for row in rows) / len(rows) for metric in METRICS} for policy in ('sort', 'expected')}
                       for shape, rows in groups.items()}
        evidence['upstream_comparison'] = dict(reference_sha256=checksum(comparison_reference),
            comparison_entry_sha256=control.get('entry_sha256'), queries=comparisons,
            delta_per_type=differences,
            delta_macro={policy: {metric: sum(row[policy][metric] for row in differences.values()) / len(differences)
                                 for metric in METRICS} for policy in ('sort', 'expected')},
            scope='Diagnostic comparison only; declared reference scores, ranks and ties remain the verification gate.')
    write_json(output, evidence)
    return evidence
