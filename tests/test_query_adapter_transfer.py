"""Evaluate frozen adapters on a paired new benchmark without fitting."""

import json

import numpy as np
import pytest

from dicee.query_answering import PLUS_H_DATASETS, QueryScoreAdapter
from dicee.query_answering._checkpoint import BenchmarkCheckpoint, write_json
from dicee.query_answering.context import state_fingerprint
from dicee.scripts import benchmark_query_adapters as sweep
from tests.test_query_benchmark_plus_h import plus_h_fixture
from tests.test_query_engine import TableModel


def source_run(path, model, *, variants=('baseline',), **settings):
    config = dict(study='fixture', backbones=['ultra'], checkpoints={'ultra': {'path': 'weights', 'sha256': 'same'}},
                  variants={v: {'tnorm': 'prod'} for v in variants}, **settings)
    write_json(path/'configuration.json', config)
    write_json(path/'status.json', {'state': 'complete'})
    write_json(path/'adapters'/'ultra'/'complete.json', {'state_sha256': state_fingerprint(model)})
    for variant in variants:
        directory = path/'adapters'/'ultra'/variant/'all_sources'
        directory.mkdir(parents=True)
        adapter = QueryScoreAdapter('global', metadata={'backbone_state_sha256': state_fingerprint(model)})
        adapter.save(directory/'adapter.json')
        write_json(directory/'report.json', {'adapter_sha256': sweep.sha(directory/'adapter.json')})
    return config


def test_catalog_pins_adapters_and_skips_only_included_duplicate_baselines(tmp_path):
    model = TableModel(np.zeros((8, 4, 8)))
    first, second = tmp_path/'first', tmp_path/'second'
    config = source_run(first, model)
    source_run(second, model, variants=('baseline', 'larger'), data_scaling_from=str(first), reuse_variants=['baseline'])
    catalog, variants = sweep.evaluation_catalog(config, [first, second])
    assert list(catalog['ultra']) == list(variants) == ['baseline', 'larger']
    assert catalog['ultra']['baseline']['adapter_sha256'] == sweep.sha(first/'adapters/ultra/baseline/all_sources/adapter.json')
    catalog, _ = sweep.evaluation_catalog(config, [second])
    assert list(catalog['ultra']) == ['baseline', 'larger']
    other = tmp_path/'other'
    source_run(other, model)
    catalog, _ = sweep.evaluation_catalog(config, [first, other])
    assert list(catalog['ultra']) == ['baseline', 'baseline__run2']


def test_import_requires_completed_unlocked_pinned_source(tmp_path):
    model = TableModel(np.zeros((8, 4, 8)))
    previous, out = tmp_path/'previous', tmp_path/'new'
    parent = source_run(previous, model)
    entries, variants = sweep.evaluation_catalog(parent, [previous])
    config = dict(evaluation_adapters=entries, variants=variants)
    write_json(previous/'status.json', {'state': 'fitting'})
    with pytest.raises(ValueError, match='finish first'):
        sweep.import_evaluation_adapters(out, config)
    write_json(previous/'status.json', {'state': 'complete'})
    with BenchmarkCheckpoint(previous/'run', parent):
        with pytest.raises(RuntimeError, match='already running'):
            sweep.import_evaluation_adapters(out, config)
    sweep.import_evaluation_adapters(out, config)
    sweep.import_evaluation_adapters(out, config)
    assert (out/'adapters/ultra/baseline/all_sources/adapter.json').read_bytes() == (previous/'adapters/ultra/baseline/all_sources/adapter.json').read_bytes()
    with (previous/'adapters/ultra/baseline/all_sources/adapter.json').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='Pinned evaluation adapter changed'):
        sweep.import_evaluation_adapters(out, config)


def test_unfinished_fits_are_resolved_and_verified_when_imported(tmp_path):
    model = TableModel(np.zeros((8, 4, 8)))
    previous, out = tmp_path/'previous', tmp_path/'new'
    parent = source_run(previous, model)
    adapter = previous/'adapters/ultra/baseline/all_sources/adapter.json'
    saved = adapter.read_bytes()
    adapter.unlink()
    entries, variants = sweep.evaluation_catalog(parent, [previous])
    assert entries['ultra']['baseline']['adapter_sha256'] is None
    adapter.write_bytes(saved)
    sweep.import_evaluation_adapters(out, dict(evaluation_adapters=entries, variants=variants))
    manifest = json.loads((out/'adapter-imports.json').read_text())
    assert manifest['ultra']['baseline']['adapter_sha256'] == sweep.sha(adapter)


def test_plus_h_screen_end_to_end_never_trains_and_covers_all_datasets(tmp_path, monkeypatch):
    data_root = tmp_path/'data'
    for name in PLUS_H_DATASETS:
        plus_h_fixture(data_root, name)
    model = TableModel(np.random.default_rng(8).normal(size=(8, 4, 8)))
    previous, out = tmp_path/'previous', tmp_path/'new'
    parent = source_run(previous, model)
    entries, variants = sweep.evaluation_catalog(parent, [previous])
    config = dict(root=str(tmp_path), study='evaluation', evaluation_adapters=entries, variants=variants,
                  backbones=['ultra'], datasets=list(PLUS_H_DATASETS), benchmark_root=str(data_root),
                  split='valid', size=25, sampling_seed=17)
    config['benchmark_inputs'] = sweep.prepare_evaluation_selection(out, config)
    monkeypatch.setattr(sweep, 'load_model', lambda *_args: model)
    monkeypatch.setattr(sweep, 'prepare_sources', lambda *_args: pytest.fail('Training sources requested'))
    monkeypatch.setattr(sweep, 'fit_worker', lambda *_args: pytest.fail('Adapters were retrained'))
    class Worker:
        pid = 1
        def __init__(self, command, **_kwargs):
            job = command[command.index('--worker') + 1:command.index('--batch-size')]
            assert job[0] == 'evaluate'
            sweep.evaluate_worker(out, config, *job[1:], 1, 1)
        def wait(self):
            return 0
    monkeypatch.setattr(sweep.subprocess, 'Popen', Worker)
    monkeypatch.setattr(sweep.signal, 'signal', lambda *_args: None)
    sweep.supervise(out, config)
    assert json.loads((out/'status.json').read_text())['state'] == 'complete'
    comparison = json.loads((out/'comparison.json').read_text())['ultra']['baseline']
    assert comparison['datasets'] == 3
    assert not comparison['summary']['complete_plus_h_test_suite']
    for name in PLUS_H_DATASETS:
        directory = out/'benchmark/ultra/baseline'/name
        report = json.loads((directory/'result.json').read_text())['report']
        assert report['dataset_metadata']['suite'] == 'plus-h'
        assert report['dataset_metadata']['inference_graph'] == 'train'
        assert report['split'] == 'valid'
        assert len(report['per_shape']) == 16
        assert set(report['per_shape']) >= {'4p', '4i'}
        assert sweep.sha(directory/'selection.json') == config['benchmark_inputs'][name]['selection_sha256']
    text = (out/'comparison.md').read_text()
    assert 'all 3 datasets' in text and '| 3/3 |' in text and '/23' not in text
    config['benchmark_inputs'][PLUS_H_DATASETS[0]]['selection_sha256'] = 'changed'
    with pytest.raises(ValueError, match='Frozen evaluation dataset or query selection changed'):
        sweep.evaluate_worker(out, config, 'ultra', 'baseline', PLUS_H_DATASETS[0], 1, 1)
