"""Matched operator fitting and source-only policy selection."""

import json

import numpy as np
import pytest

from dicee.query_answering import QueryScoreAdapter, fit_query_adapter
from dicee.query_answering._checkpoint import write_json
from dicee.query_answering.benchmark import METRICS, _averages
from dicee.query_answering.context import fingerprint
from dicee.query_answering.training import STANDARD_TRAINING_SHAPES, _bank, _baseline_validation
from dicee.scripts import benchmark_query_adapters as sweep
from tests.test_query_adapter_training import data, source  # noqa: F401
from tests.test_query_engine import TableModel


def test_norms_share_raw_scores_but_not_validation_or_fit_checkpoints(data, tmp_path):  # noqa: F811
    model = TableModel(np.random.default_rng(20).normal(size=(8, 6, 8)))
    options = dict(epochs=4, validation_every=2, cache_dir=tmp_path)
    first = fit_query_adapter(model, [data], checkpoint_path=tmp_path/'fit.pt', **options)
    calls = len(model.calls)
    second = fit_query_adapter(model, [data], tnorm='min', **options)
    assert len(model.calls) == calls
    assert first.adapter.metadata['training']['tnorm'] == 'prod'
    assert second.adapter.metadata['training']['tnorm'] == 'min'
    assert first.history[0]['loss'] != second.history[0]['loss']
    assert len(list(tmp_path.glob('validation-*.json'))) == 4
    with pytest.raises(ValueError, match='checkpoint inputs/settings changed'):
        fit_query_adapter(model, [data], checkpoint_path=tmp_path/'fit.pt', tnorm='min', **options)
    with pytest.raises(ValueError, match='t-norm'):
        fit_query_adapter(model, [data], tnorm='invalid', **options)


def test_validation_cache_is_operator_specific(data, tmp_path, monkeypatch):  # noqa: F811
    from dicee.query_answering import training
    model = TableModel(np.zeros((8, 6, 8)))
    bank = _bank(model, data, cache_dir=tmp_path, row_batch_size=2, seed=0, samples=None)
    adapter = QueryScoreAdapter('global')
    calls = []
    def validation(_adapter, _sources, banks):
        calls.append(banks[0]['tnorm'])
        return {'tnorm': banks[0]['tnorm']}
    monkeypatch.setattr(training, '_validation', validation)
    for tnorm in ('prod', 'min', 'prod', 'min'):
        bank['tnorm'] = tnorm
        assert _baseline_validation(adapter, [data], [bank], tmp_path) == {'tnorm': tnorm}
    assert calls == ['prod', 'min']


def policy_fixture(out):
    config = dict(study='tnorm', variants=sweep.study_variants('tnorm'),
                  tnorm_policy=dict(select_shapes=list(STANDARD_TRAINING_SHAPES),
                                    fixed_product_shapes=['ip', 'pi', '2u', 'up'], minimum_gain=1e-12))
    write_json(out/'configuration.json', config)
    for name in sweep.SOURCE_NAMES:
        write_json(out/'sources'/f'{name}.json', {'source': name})
    for variant, norm in [('product', 'prod'), ('min', 'min')]:
        directory = out/'adapters'/'ultra'/variant/'all_sources'
        directory.mkdir(parents=True)
        adapter = QueryScoreAdapter('global', metadata={'training': {'tnorm': norm}})
        adapter.save(directory/'adapter.json')
        metrics = {f'{name}/{shape}': dict(mrr=.5, queries=16)
                   for name in sweep.SOURCE_NAMES for shape in STANDARD_TRAINING_SHAPES}
        if variant == 'min':
            for name in sweep.SOURCE_NAMES:
                metrics[f'{name}/2i']['mrr'] = .6
                metrics[f'{name}/3i']['mrr'] = .6
            metrics['CoDExMedium/3i']['mrr'] = .4
        write_json(directory/'report.json', dict(metadata=adapter.metadata, validation={'fitted': metrics},
                                                 adapter_sha256=sweep.sha(directory/'adapter.json')))
    return config


def test_policy_uses_consistent_source_gains_and_preserves_excluded_shapes(tmp_path):
    config = policy_fixture(tmp_path)
    sweep.freeze_tnorm_policy(tmp_path, config, 'ultra')
    policy = json.loads((tmp_path/'policies'/'ultra.json').read_text())
    assert policy['choices']['2i'] == 'min'
    assert all(v == 'product' for k, v in policy['choices'].items() if k != '2i')
    for shape in ('ip', 'pi', '2u', 'up'):
        assert shape not in policy['source_validation_mrr_gains']
    target = tmp_path/'benchmark'/'ultra'/'min'/'target'
    target.mkdir(parents=True)
    (target/'result.json').write_text('Target results must not be read for selection')
    sweep.freeze_tnorm_policy(tmp_path, config, 'ultra')
    report_path = tmp_path/'adapters'/'ultra'/'min'/'all_sources'/'report.json'
    report = json.loads(report_path.read_text())
    report['validation']['fitted']['WN18RR/2i']['mrr'] = .1
    write_json(report_path, report)
    with pytest.raises(ValueError, match='Frozen t-norm policy inputs changed'):
        sweep.freeze_tnorm_policy(tmp_path, config, 'ultra')


def test_policy_cannot_be_first_selected_after_target_evaluation(tmp_path):
    config = policy_fixture(tmp_path)
    trace = tmp_path/'benchmark'/'ultra'/'product'/'toy'/'metrics.jsonl'
    trace.parent.mkdir(parents=True)
    trace.write_text('')
    with pytest.raises(ValueError, match='before target evaluation'):
        sweep.freeze_tnorm_policy(tmp_path, config, 'ultra')


def benchmark_fixture(out, config):
    sweep.freeze_tnorm_policy(out, config, 'ultra')
    selection = [dict(shape=s, query=[i, [0]]) for i, s in enumerate(('2i', '3i', 'ip'))]
    for variant, norm in [('product', 'prod'), ('min', 'min')]:
        directory = out/'benchmark'/'ultra'/variant/'toy'
        write_json(directory/'selection.json', selection)
        value = .2 if variant == 'product' else .8
        rows = [dict(id=fingerprint(q['query']), shape=q['shape'], **dict.fromkeys(METRICS, value)) for q in selection]
        trace = directory/'metrics.jsonl'
        trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        per_shape = {q['shape']: dict(queries=1, **dict.fromkeys(METRICS, value)) for q in selection}
        adapter = json.loads((out/'adapters'/'ultra'/variant/'all_sources'/'adapter.json').read_text())
        report = dict(dataset='toy', split='valid', group='transductive', dataset_metadata={}, context_sha256='graph',
                      candidate_sha256='candidates', protocol={'tie_policy': 'sort'}, queries=3, per_shape=per_shape,
                      averages=_averages(per_shape), seconds=1., inference=dict(tnorm=norm, adapter=adapter))
        write_json(directory/'result.json', dict(report=report, metrics_sha256=sweep.sha(trace),
                   selection_sha256=sweep.sha(directory/'selection.json'), operator_policy_sha256=sweep.sha(out/'policies'/'ultra.json')))


def test_policy_result_reuses_exact_pair_metrics_without_target_selection(tmp_path):
    config = policy_fixture(tmp_path)
    benchmark_fixture(tmp_path, config)
    sweep.derive_tnorm_results(tmp_path, config, 'ultra', 'toy')
    output = tmp_path/'benchmark'/'ultra'/'source_selected'/'toy'
    result = json.loads((output/'result.json').read_text())
    scores = result['report']['per_shape']
    assert scores['2i']['mrr'] == .8
    assert scores['3i']['mrr'] == scores['ip']['mrr'] == .2
    assert result['report']['averages']['all']['mrr'] == pytest.approx(.4)
    assert result['derived']
    rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert [row['mrr'] for row in rows] == [.8, .2, .2]


@pytest.mark.parametrize('change', ['trace', 'selection', 'context', 'operator', 'adapter', 'policy'])
def test_policy_rejects_mismatched_global_runs(tmp_path, change):
    config = policy_fixture(tmp_path)
    benchmark_fixture(tmp_path, config)
    directory = tmp_path/'benchmark'/'ultra'/'min'/'toy'
    job = json.loads((directory/'result.json').read_text())
    if change == 'trace':
        rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        rows.reverse()
        (directory/'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        job['metrics_sha256'] = sweep.sha(directory/'metrics.jsonl')
    elif change == 'selection':
        write_json(directory/'selection.json', [])
        job['selection_sha256'] = sweep.sha(directory/'selection.json')
    elif change == 'context':
        job['report']['context_sha256'] = 'other graph'
    elif change == 'operator':
        job['report']['inference']['tnorm'] = 'prod'
    elif change == 'adapter':
        job['report']['inference']['adapter'] = {}
    else:
        job['operator_policy_sha256'] = 'changed'
    write_json(directory/'result.json', job)
    with pytest.raises(ValueError):
        sweep.derive_tnorm_results(tmp_path, config, 'ultra', 'toy')


def test_study_only_changes_operator_and_matches_training_budget():
    variants = sweep.study_variants('tnorm')
    first, second = (dict(variants[v]) for v in ('product', 'min'))
    assert first.pop('tnorm') == 'prod' and second.pop('tnorm') == 'min'
    assert first == second == sweep.study_variants('query-types')['types_2']
    assert 3 * len(first['shapes']) * first['train_per_shape'] == 840
