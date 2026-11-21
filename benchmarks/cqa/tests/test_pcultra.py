"""PCUltra as a CQA backbone: the catalog entry, loading from a ProofPFN checkpoint and the recipe option score_input."""

from pathlib import Path

import pytest
import torch

from benchmarks.cqa import cli
from benchmarks.cqa.tests.test_protocol import fixture_manifest, read
from dicee.query_answering import QueryScoreAdapter
from dicee.query_answering._checkpoint import write_json
from dicee.query_answering.catalog import KGFM_ADAPTERS, KGFM_BACKBONES, NATIVE_KGFM_BACKBONES, check_recipe
from dicee.query_answering.context import state_fingerprint
from dicee.query_answering.method_evaluation import evaluate_method, load_kgfm_backbone
from dicee.synthetic_kg.checkpoint import CheckpointError, model_from_checkpoint, read_checkpoint
from tests.proofpfn_fixtures import clean_checkpoints


@pytest.fixture(scope='module')
def checkpoints(tmp_path_factory):
    """Checkpoints of the initial weights of PCUltra-S for seeds 42, 43 and 44 (the format of the pretraining exports)."""
    folder = tmp_path_factory.mktemp('pcultra')
    return clean_checkpoints(folder, 'PCUltra-S')


def test_every_kgfm_adapter_has_one_backbone_source_and_recipes_check_the_score_input():
    assert set(KGFM_ADAPTERS) == set(KGFM_BACKBONES) | set(NATIVE_KGFM_BACKBONES)
    assert not set(KGFM_BACKBONES) & set(NATIVE_KGFM_BACKBONES) and 'pcultra-adapter' in NATIVE_KGFM_BACKBONES
    base = dict(method='pcultra-adapter', operators={'1p': 'product'}, adapters={'product': 'a.json'},
                selection_protocol='source-validation')
    for options in ({}, dict(score_input='raw'), dict(score_input='log_softmax')):
        check_recipe(dict(base, options=options), ['1p'])
        check_recipe(dict(base, method='ultra-adapter', options=options), ['1p'])
    for options in (dict(score_input='softmax'), dict(relation_conditioning='query'), dict(prompt_seed=1)):
        with pytest.raises(ValueError):
            check_recipe(dict(base, options=options), ['1p'])


def test_the_pcultra_backbone_loads_as_its_checkpoint_says(checkpoints, tmp_path):
    path = checkpoints.format(seed=42)
    model, record = load_kgfm_backbone('pcultra', path, options=dict(backend_batch_size=3, relation_cache_mb=1))
    assert (model.name, model.relation_weighting, model.pair_channels) == ('ULTRA', 'normalized', 'on')
    assert model.query_batch_size == 3 and model.relation_cache_mb == 1 and not model.training
    assert state_fingerprint(model) == state_fingerprint(model_from_checkpoint(read_checkpoint(path)))
    assert (record['variant'], record['seed'], record['weights'], record['source']) == ('PCUltra-S', 42, 'raw', 'test')
    assert record['export_sha256'] is None and record['model_config']['ultra_pair_channels'] == 'on'
    # The global random state is not consumed by building the model.
    state = torch.random.get_rng_state()
    load_kgfm_backbone('pcultra', path)
    assert torch.equal(state, torch.random.get_rng_state())
    from dicee.models import ULTRA
    torch.save({'model': ULTRA(dict(num_entities=1, num_relations=1)).state_dict()}, tmp_path / 'ultra.pt')
    with pytest.raises(CheckpointError):
        load_kgfm_backbone('pcultra', tmp_path / 'ultra.pt')
    with pytest.raises(ValueError, match='KGFM backbone'):
        load_kgfm_backbone('pcultra2', path)


def pcultra_entry(tmp_path, checkpoints, *, adapter, score_input, ablation=True):
    """The fixture manifest with one pcultra-adapter entry of the given adapter file and score input."""
    manifest = fixture_manifest(tmp_path)
    (tmp_path / 'pcultra.pt').write_bytes(Path(checkpoints.format(seed=42)).read_bytes())
    entry = manifest['entries'][0]
    options = {'beam_size': 2, 'row_batch_size': 1, 'backend_batch_size': 1, 'cache_bytes': 1 << 20, 'raw_cache_bytes': 1 << 20}
    if score_input is not None:
        options['score_input'] = score_input
    entry.update(id='pcultra-paired', method='pcultra-adapter', checkpoint='pcultra.pt', options=options,
                 adapters={'product': adapter}, operators={shape: 'product' for shape in entry['query_types']},
                 selection_protocol='source-validation', adapter_ablation=ablation, query_batch_size=4, query_order='relation')
    return manifest, entry


def recipe_of(entry, folder, adapter):
    """The entry as a recipe of ``evaluate_method`` with the files below ``folder``."""
    recipe = {k: entry[k] for k in ('method', 'dataset', 'query_types', 'query_batch_size', 'options', 'operators',
                                    'selection_protocol', 'adapter_ablation')}
    recipe.update(checkpoint=str(folder / 'pcultra.pt'), root=str(folder / 'data'), adapters={'product': str(adapter)})
    return recipe


def learned_adapter(model, path, score_input):
    QueryScoreAdapter('context_scores', 1., bias_bound=8., score_input=score_input,
                      weights=torch.randn(2, 8, generator=torch.Generator().manual_seed(4)).double() / 5,
                      metadata={'backbone_state_sha256': state_fingerprint(model)}).save(path)


def test_paired_pcultra_evaluation_runs_with_log_softmax_input(checkpoints, tmp_path):
    model, _ = load_kgfm_backbone('pcultra', checkpoints.format(seed=42))
    learned_adapter(model, tmp_path / 'adapter.json', 'log_softmax')
    manifest, entry = pcultra_entry(tmp_path, checkpoints, adapter='adapter.json', score_input='log_softmax')
    write_json(tmp_path / 'manifest.json', manifest)
    cli.main(['plus_h', 'evaluate', '--manifests', str(tmp_path / 'manifest.json'), '--answer-filter', 'released', '--input-root', str(tmp_path),
              '--output', str(tmp_path / 'results'), '--split', 'test', '--device', 'cpu'])
    result = read(tmp_path / 'results' / entry['id'] / 'result.json')
    control = read(tmp_path / 'results' / entry['id'] / 'without-adapter/result.json')
    assert result['queries'] == control['queries'] == len(entry['query_types'])
    assert control['paired_with'] == entry['id'] and control['inference']['calibration'] == 'without-adapter'
    for report in (result, control):
        assert report['inference']['options']['score_input'] == 'log_softmax'
        backbone = report['inference']['backbone']
        assert (backbone['variant'], backbone['seed'], backbone['format']) == ('PCUltra-S', 42, 'proofpfn-checkpoint-v1')


@pytest.mark.parametrize('adapter_input,recipe_input', [('raw', 'log_softmax'), ('log_softmax', None), ('log_softmax', 'raw')])
def test_the_recipe_score_input_must_match_every_learned_adapter(checkpoints, tmp_path, adapter_input, recipe_input):
    model, _ = load_kgfm_backbone('pcultra', checkpoints.format(seed=42))
    learned_adapter(model, tmp_path / 'adapter.json', adapter_input)
    _, entry = pcultra_entry(tmp_path, checkpoints, adapter='adapter.json', score_input=recipe_input)
    with pytest.raises(ValueError, match='score_input'):
        evaluate_method(recipe_of(entry, tmp_path, tmp_path / 'adapter.json'), output=tmp_path / 'out', split='valid')


def test_rule_calibration_reads_raw_rows_and_its_control_reads_the_recipe_input(checkpoints, tmp_path):
    model, _ = load_kgfm_backbone('pcultra', checkpoints.format(seed=42))
    rule = QueryScoreAdapter.rule(model)   # the degree-scaled softmax with ordered candidates at the cap
    assert rule.fixed_calibration == 'softmax-degree-ties' and rule.score_input == 'raw'
    rule.save(tmp_path / 'rule.json')
    with pytest.raises(ValueError, match='raw scores'):
        QueryScoreAdapter('global', 1., fixed_calibration='softmax-degree-ties', score_input='log_softmax')
    predictions = {}
    for score_input in ('raw', 'log_softmax'):
        _, entry = pcultra_entry(tmp_path / score_input, checkpoints, adapter='unused', score_input=score_input)
        learned, control = {}, {}
        evaluate_method(recipe_of(entry, tmp_path / score_input, tmp_path / 'rule.json'), output=tmp_path / score_input / 'out', split='valid',
                        on_prediction=lambda query, scores: learned.__setitem__(query, scores.clone()),
                        on_control_prediction=lambda query, scores: control.__setitem__(query, scores.clone()))
        predictions[score_input] = learned, control
    (raw_learned, raw_control), (log_learned, log_control) = predictions['raw'], predictions['log_softmax']
    assert raw_learned.keys() == log_learned.keys() == raw_control.keys() == log_control.keys()
    # The rule calibration is the same computation either way; only the sigmoid identity control changes its input.
    assert all(torch.equal(raw_learned[q], log_learned[q]) for q in raw_learned)
    assert any(not torch.equal(raw_control[q], log_control[q]) for q in raw_control)
