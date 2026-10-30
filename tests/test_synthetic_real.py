"""The real subgraph worlds (``priors.real``) of arms R and M against golden G2, and their source graphs.

The golden checks read the source graphs from ``PROOFPFN_REAL_SOURCES`` (a folder with ``<name>/data.pt``) and skip
when it is unset. The rebuild of the source graphs reads the split files below ``KGs/`` and skips without them.
"""
import os
from pathlib import Path

import pytest
import torch

from dicee.synthetic_kg.arms import ARMS
from dicee.synthetic_kg.digests import file_sha256
from dicee.synthetic_kg.priors import FamilySpec, Slot, real, world
from dicee.synthetic_kg.recipe import REAL_SOURCE_SHA256
from tests.synthetic_worlds import check_world, golden_names

KGS = Path(__file__).resolve().parents[1] / 'KGs'


def require_real_sources():
    if not os.environ.get(real.SOURCES_ENV):
        pytest.skip(f'real worlds need their source graphs (set {real.SOURCES_ENV})')


def test_real_world_equals_the_research_world():
    require_real_sources()
    check_world('44/R/real/1')


@pytest.mark.slow
def test_every_golden_real_world():
    require_real_sources()
    names = golden_names({'real'})
    assert len(names) == 10
    for name in names:
        check_world(name)


def test_real_sources_are_checked_against_their_pins(tmp_path, monkeypatch):
    monkeypatch.delenv(real.SOURCES_ENV, raising=False)
    with pytest.raises(RuntimeError, match=real.SOURCES_ENV):
        real.source_folder()
    with pytest.raises(RuntimeError, match=real.SOURCES_ENV) as caught:   # a generator error names the world
        world(42, Slot('real', 'hook', 400_000), ARMS['R'].prior)
    assert caught.value.__notes__ == ["in the world of seed 42, slot ('real', 'hook', 400000)"]
    (tmp_path / 'WN18RR').mkdir()
    torch.save({'splits': {'train': torch.zeros(1, 3, dtype=torch.long)}, 'entities': ['a']}, tmp_path / 'WN18RR' / 'data.pt')
    with pytest.raises(ValueError, match='the pin of WN18RR'):
        real.load_source('WN18RR', tmp_path)
    assert real.NAMES == ('FB15k-237', 'WN18RR', 'CoDEx-M')


@pytest.mark.slow
@pytest.mark.parametrize('name', real.NAMES)
def test_sources_rebuild_from_their_split_files(name, tmp_path):
    graph = KGS / name
    if not all((graph / f'{split}.txt').is_file() for split in ('train', 'valid', 'test')):
        pytest.skip(f'{graph} holds no split files (KGs.zip holds FB15k-237 and WN18RR; docs/proofpfn.md names the CoDEx-M source)')
    assert real.build_source(name, graph, tmp_path) == tmp_path / name / 'data.pt'
    assert file_sha256(tmp_path / name / 'data.pt') == REAL_SOURCE_SHA256[name]


def test_a_source_with_other_split_files_is_refused(tmp_path):
    graph = tmp_path / 'graph'
    graph.mkdir()
    for split, rows in (('train', 'a r b\nb r c\n'), ('valid', 'a r c\n'), ('test', 'c r a\n')):
        (graph / f'{split}.txt').write_text(rows)
    with pytest.raises(ValueError, match='not the pin'):
        real.build_source('WN18RR', graph, tmp_path / 'sources')
    assert list((tmp_path / 'sources' / 'WN18RR').iterdir()) == []
    with pytest.raises(ValueError, match='unknown real source graph'):
        real.build_source('YAGO3-10', graph, tmp_path / 'sources')


def test_real_family_parameters():
    assert FamilySpec('real', 'real', 400_000, (('max_entities', 5000),)).overrides == {'max_entities': 5000}
    with pytest.raises(ValueError, match='unknown parameters'):
        FamilySpec('real', 'real', 400_000, (('rows', (1, 2)),))
