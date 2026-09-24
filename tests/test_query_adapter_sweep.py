"""The data-scaling sweep waits for and verifies its reused baseline."""

import json

import pytest

from dicee.query_answering._checkpoint import BenchmarkCheckpoint, write_json
from dicee.scripts import benchmark_query_adapters as sweep


@pytest.mark.parametrize('cached', [False, True])
def test_source_changes_rejected_before_preparation_or_cache_reuse(tmp_path, monkeypatch, cached):
    from dicee.query_answering import AdapterQuery, AdapterTrainingData, QueryContext

    source = tmp_path/'train.txt'
    source.write_text('a r b\n')
    recorded_sha = sweep.sha(source)
    config = dict(sources={'source': dict(path=str(source), sha256=recorded_sha)})
    if cached:
        data = AdapterTrainingData('source', QueryContext((), 3, 2),
                                   (AdapterQuery(((0, (0,)), (1, (1,))), {2}),),
                                   metadata={'train_sha256': recorded_sha})
        (tmp_path/'sources').mkdir()
        data.save(tmp_path/'sources'/'source.json')
    source.write_text('a r c\n')
    monkeypatch.setattr(sweep, 'prepare_adapter_data', lambda *_args, **_kwargs: pytest.fail('Prepared changed source'))
    with pytest.raises(ValueError, match='Source file changed'):
        sweep.prepare_sources(tmp_path, config)


def test_scaling_variants_do_not_add_unrequested_baselines():
    config = dict(data_scaling_from='parent', variants={'baseline': {}, 'larger': {}})
    assert sweep.variant_names(config, 'ultra') == ['baseline', 'larger']
    config = dict(variants=sweep.VARIANTS, reference_adapter={'path': 'adapter.json'})
    assert sweep.variant_names(config, 'ultra') == ['reference_500ep', 'observed', *sweep.VARIANTS, 'sigmoid']


def test_reuse_requires_completed_unlocked_verified_predecessor(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, 'SOURCE_NAMES', {'source': 'source'})
    monkeypatch.setattr(sweep, 'BENCHMARK_DATASETS', ['toy'])
    previous, out = tmp_path/'previous', tmp_path/'new'
    parent = {'backbones': ['ultra']}
    write_json(previous/'configuration.json', parent)
    config = dict(data_scaling_from=str(previous), predecessor_sha256=sweep.sha(previous/'configuration.json'),
                  backbones=['ultra'], reuse_variants=['baseline'])
    for fold in ('all_sources', 'source'):
        directory = previous/'adapters'/'ultra'/'baseline'/fold
        write_json(directory/'adapter.json', {'test': fold})
        write_json(directory/'report.json', {'adapter_sha256': sweep.sha(directory/'adapter.json')})
    source = previous/'benchmark'/'ultra'/'baseline'/'toy'
    write_json(source/'selection.json', [{'query': [0, [1]]}])
    (source/'metrics.jsonl').write_text('{"mrr": 0.5}\n')
    write_json(source/'result.json', dict(selection_sha256=sweep.sha(source/'selection.json'),
                                         metrics_sha256=sweep.sha(source/'metrics.jsonl')))
    write_json(previous/'status.json', {'state': 'evaluating'})
    with pytest.raises(ValueError, match='finish first'):
        sweep.reuse_baseline(out, config)
    write_json(previous/'status.json', {'state': 'complete'})
    with BenchmarkCheckpoint(previous/'run', parent):
        with pytest.raises(RuntimeError, match='already running'):
            sweep.reuse_baseline(out, config)
    sweep.reuse_baseline(out, config)
    target = out/'benchmark'/'ultra'/'baseline'/'toy'
    assert json.loads((target/'result.json').read_text()) == json.loads((source/'result.json').read_text())
    assert (target/'metrics.jsonl').read_bytes() == (source/'metrics.jsonl').read_bytes()
    (source/'metrics.jsonl').write_text('{"mrr": 1.0}\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        sweep.reuse_baseline(out, config)


def test_evaluation_keeps_trace_open_until_queries_finish(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from dicee.query_answering import QueryScoreAdapter
    from dicee.query_answering.context import state_fingerprint

    monkeypatch.setattr(sweep, 'runtime', lambda: None)
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(sweep, 'load_model', lambda *_: model)
    monkeypatch.setattr(sweep, 'load_benchmark', lambda *_args, **_kwargs: None)
    queries = [SimpleNamespace(shape='1p', query=(0, (1,)))]
    monkeypatch.setattr(sweep, '_query_plan', lambda *_args, **_kwargs: queries)
    adapter_path = tmp_path/'adapters'/'ultra'/'baseline'/'all_sources'
    adapter_path.mkdir(parents=True)
    QueryScoreAdapter('global', metadata={'backbone_state_sha256': state_fingerprint(model)}).save(adapter_path/'adapter.json')

    def benchmark(_model, _data, **kwargs):
        kwargs['on_query'](queries[0].query, '1p', {'mrr': .5})
        return dict(per_shape={'1p': {'mrr': .5}}, seconds=1.)

    monkeypatch.setattr(sweep, 'benchmark_model', benchmark)
    sweep.evaluate_worker(tmp_path, dict(root=str(tmp_path), size=1, sampling_seed=0),
                          'ultra', 'baseline', 'toy', 1, 1)
    result = json.loads((tmp_path/'benchmark'/'ultra'/'baseline'/'toy'/'result.json').read_text())
    assert result['report']['per_shape']['1p']['mrr'] == .5
