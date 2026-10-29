"""The real subgraph worlds (``priors.real``) of arms R and M against golden G2, and the checks of their sources.

The golden checks read the source graphs from ``PROOFPFN_REAL_SOURCES`` (a folder with ``<name>/data.pt``) and skip
when it is unset.
"""
import os

import pytest
import torch

from dicee.synthetic_kg.priors import FamilySpec, real
from tests.synthetic_worlds import check_world, golden_names


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
    (tmp_path / 'WN18RR').mkdir()
    torch.save({'splits': {'train': torch.zeros(1, 3, dtype=torch.long)}, 'entities': ['a']}, tmp_path / 'WN18RR' / 'data.pt')
    with pytest.raises(ValueError, match='the pin of WN18RR'):
        real.load_source('WN18RR', tmp_path)
    assert real.NAMES == ('FB15k-237', 'WN18RR', 'CoDEx-M')


def test_real_family_parameters():
    assert FamilySpec('real', 'real', 400_000, (('max_entities', 5000),)).overrides == {'max_entities': 5000}
    with pytest.raises(ValueError, match='unknown parameters'):
        FamilySpec('real', 'real', 400_000, (('rows', (1, 2)),))
