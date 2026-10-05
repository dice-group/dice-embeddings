"""Validate the frozen KGFM pipeline against independent calibration/composition."""

import json
from pathlib import Path
from unittest.mock import patch

import torch

from dicee.query_answering.context import fingerprint
from dicee.query_answering.engine import AtomicScorer, QueryAnswerer

from ..study import bundle_entry, checksum, entry_inputs_identity, read, run_job, verify_bundle
from .reference_kgfm import ReferenceKGFM


def verify(bundle_dir, entry_id, root, output, *, device='cuda', probes=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'pilot').exists():
        raise FileExistsError('Use a fresh directory so every prediction is checked')
    bundle = verify_bundle(bundle_dir, root)
    entry = bundle_entry(bundle, entry_id)
    if not entry['method'].endswith('-adapter') or len(entry['adapters']) != 1:
        raise ValueError('Select a frozen single-operator KGFM entry')
    adapter = read(Path(root) / next(iter(entry['adapters'].values())))
    observed_facts = entry['options'].get('observed_facts')
    if observed_facts is not None:
        adapter['observed_mix'] = float(observed_facts != 'none')
    references, scorers, comparisons, scores, orders = {}, {}, {}, {}, {}
    original = AtomicScorer.rows
    original_init = QueryAnswerer.__init__

    def initialize(engine, *args, **kwargs):
        original_init(engine, *args, **kwargs)
        scorer = engine.scorer
        variant = 'without-adapter' if engine.adapter.metadata.get('calibration') == 'identity' else 'learned'
        expected_adapter = adapter if variant == 'learned' else dict(adapter, feature_mode='global', weights=[[0.], [0.]],
                                                                     membership_threshold=0.)
        reference = ReferenceKGFM(scorer.context.triples, scorer.n, scorer.nr, expected_adapter,
                                  beam_size=entry['options']['beam_size'], device=scorer.device,
                                  restore_observed=observed_facts in (None, 'both'))
        scorers[id(scorer)] = reference
        references[variant] = reference

    def capture(scorer, conditions):
        raw = original(scorer, conditions)
        scorers[id(scorer)].capture(conditions, raw)
        return raw

    def compare(query, actual, variant='learned'):
        key = query.identity
        expected = references[variant].predict(query.query, entry['operators'][query.shape])
        order = expected.argsort(descending=True)
        finite = actual.isfinite() & expected.isfinite()
        error = (actual[finite] - expected[finite]).abs().max().item() if finite.any() else 0.
        result = dict(shape=query.shape, variant=variant, max_error=error,
                      scores_match=torch.allclose(actual, expected, atol=1e-12, rtol=1e-12),
                      ranks_match=torch.equal(actual.argsort(descending=True), order),
                      ties_match=torch.equal(actual[order][1:] == actual[order][:-1],
                                             expected[order][1:] == expected[order][:-1]))
        result['passed'] = all(result[name] for name in ('scores_match', 'ranks_match', 'ties_match'))
        comparisons[f'{variant}:{key}'] = result
        scores.setdefault(variant, {})[key], orders.setdefault(variant, {})[key] = expected.cpu(), order.cpu()
        print(json.dumps(dict(query=len(comparisons), **result)), flush=True)

    with patch.object(AtomicScorer, 'rows', capture), patch.object(QueryAnswerer, '__init__', initialize):
        report = run_job(bundle_dir, entry_id, root, output / 'pilot', device=device, phase='pilot',
                         pilot_queries=probes, probe_only=True, on_prediction=compare,
                         on_control_prediction=lambda query, actual: compare(query, actual, 'without-adapter'))
    variants = {'learned', 'without-adapter'} if entry.get('adapter_ablation') else {'learned'}
    if set(references) != variants or any({row['shape'] for row in comparisons.values() if row['variant'] == variant}
                                         != set(entry['query_types']) for variant in variants):
        raise ValueError('Integration check missed a query type or calibration variant')
    identity = dict(entry_sha256=fingerprint({k: v for k, v in entry.items() if k != 'verification'}),
                    inputs_sha256=entry_inputs_identity(bundle, entry))
    scope = ('Independent context features, linear adapter, tuple interpretation, stable fixed-k pruning, '
             'fuzzy composition and positive observed-answer restoration. Shared native backbone logits; '
             'this is integration evidence, not an independent backbone reproduction. '
             'Paired entries also verify the identity-calibration control on every query type.')
    oracle = dict(version=1, **identity, validation_plan_sha256=bundle['plans'][f'{entry_id}/valid']['sha256'],
                  pilot_queries=probes, probe_only=True, scores=scores['learned'], orders=orders['learned'],
                  comparison_scores={key: value for key, value in scores.items() if key != 'learned'},
                  comparison_orders={key: value for key, value in orders.items() if key != 'learned'},
                  environment=report['benchmark_run']['environment'], runtime_adjustments=[scope])
    torch.save(oracle, output / 'reference.pt')
    evidence = dict(**identity, source_sha256=bundle['source_sha256'], environment=report['benchmark_run']['environment'],
                    reference_sha256=checksum(output / 'reference.pt'), reference_commit=None,
                    reference_implementation={p.name: checksum(p) for p in (Path(__file__), Path(__file__).with_name('reference_kgfm.py'))},
                    queries=comparisons, scope=scope, probe_only=True,
                    reference_adjustments=[scope], atol=1e-12, rtol=1e-12,
                    passed=all(row['passed'] for row in comparisons.values()),
                    raw_rows=sum(reference.raw_rows for reference in references.values()),
                    unique_rows=sum(len(reference.rows) for reference in references.values()),
                    peak_cuda_allocated_bytes=report['peak_cuda_allocated_bytes'],
                    total_wall_seconds=report['total_wall_seconds'])
    (output / 'integration.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if not evidence['passed']:
        raise RuntimeError('Integration parity failed; inspect integration.json')
    return evidence
