"""Compact public recipes resolve to exactly the pinned benchmark entries."""

import json
from pathlib import Path

import pytest

from benchmarks.cqa.manifests import REPO, expand, merge, read_manifest
from dicee.query_answering.context import fingerprint

PINNED = json.loads(Path(__file__).with_name('recipe_fingerprints.json').read_text())


@pytest.mark.parametrize('name', sorted(PINNED))
def test_public_recipes_expand_to_the_pinned_entries(name):
    """Any change to a public recipe must update recipe_fingerprints.json in the same review."""
    entries = read_manifest(REPO / 'benchmarks' / name)['entries']
    assert {entry['id']: fingerprint(entry) for entry in entries} == PINNED[name]
    assert len(entries) == len(PINNED[name])


def test_entries_merge_defaults_recursively_and_fill_dataset_ids():
    recipes = [{'defaults': {'id': 'm-{dataset}', 'method': 'm', 'options': {'a': 1, 'b': {'c': 2}}, 'query_types': ['1p', '2p']},
                'entries': [{'dataset': 'x'}, {'dataset': 'y', 'id': 'custom', 'options': {'b': {'d': 3}}, 'query_types': ['1p']}]}]
    assert expand(recipes) == [
        {'id': 'm-x', 'method': 'm', 'options': {'a': 1, 'b': {'c': 2}}, 'query_types': ['1p', '2p'], 'dataset': 'x'},
        {'id': 'custom', 'method': 'm', 'options': {'a': 1, 'b': {'c': 2, 'd': 3}}, 'query_types': ['1p'], 'dataset': 'y'}]
    defaults = {'options': {'a': 1}}
    assert merge(defaults, {'options': None}) == {'options': None} and defaults == {'options': {'a': 1}}


def test_resolved_and_bundled_manifests_read_unchanged(tmp_path):
    manifest = dict(version=1, data_root='data', entries=[dict(id='a', method='cone', dataset='FB15k237+H')])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'bundle.json').write_text(json.dumps(dict(manifest=manifest, files={})))
    assert read_manifest(tmp_path / 'manifest.json') == read_manifest(tmp_path / 'bundle.json') == manifest
