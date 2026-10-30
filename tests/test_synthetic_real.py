"""The real subgraph worlds (``priors.real``) of arms R and M against golden G2, and their source graphs.

The golden checks read the source graphs from ``PROOFPFN_REAL_SOURCES`` (a folder with ``<name>/data.pt``) and skip
when it is unset. The rebuild of the source graphs reads the split files below ``KGs/`` and skips without them; the
rebuild of CoDEx-M writes its upstream split files from the pinned source first.
"""
import os
from pathlib import Path

import pytest
import torch

from dicee.synthetic_kg.arms import ARMS
from dicee.synthetic_kg.digests import file_sha256
from dicee.synthetic_kg.priors import FamilySpec, Slot, real, world
from dicee.synthetic_kg.recipe import REAL_SOURCE_CONTENT_SHA256
from tests.synthetic_worlds import check_world, golden_names

KGS = Path(__file__).resolve().parents[1] / 'KGs'
RESEARCH_FILE_SHA256 = {'FB15k-237': '653525529c02f9b7d88ec66702a27a128024693dbc0ffb0c2336576122ddb1ef',
                        'WN18RR': '19d9dc38e66247a7e6680801eff6c45564efa5b22a49975eff8d64ecac17a4e7',
                        'CoDEx-M': 'aebdfbc3130be64766dc5714ac048d84f985898957c6daae9b46bc29a0d0e218'}
"""The SHA-256 of the research files (``pcu-training-20261026/plan.json``), the bytes of torch 2.5.1's ``torch.save``."""
CODEX_M_SPLITS = {'train': 'd99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556',
                  'valid': '11c323096367354940846b9ed940c5dabddc1d20a70474222babe7e7fd28b28d',
                  'test': '0575ce05e4ce915e395bb4f708e8df9547989dd4c8c71407cbfd78c19daffcf7'}
"""The SHA-256 of the CoDEx-M split files of tsafavi/codex at ``3132e426`` (``data/triples/codex-m/``)."""


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
    with pytest.raises(ValueError, match='not a real source graph'):
        real.load_source('WN18RR', tmp_path)
    torch.save({'splits': {'train': torch.zeros(1, 3, dtype=torch.long)}, 'entities': ['a'], 'relations': ['r']}, tmp_path / 'WN18RR' / 'data.pt')
    with pytest.raises(ValueError, match='other content than the research file of WN18RR'):
        real.load_source('WN18RR', tmp_path)
    assert real.NAMES == ('FB15k-237', 'WN18RR', 'CoDEx-M')


def check_rebuild(name: str, graph: Path, folder: Path) -> None:
    """``build_source`` writes the pinned content; with torch 2.5.1 also the bytes of the research file."""
    path = real.build_source(name, graph, folder)
    assert path == folder / name / 'data.pt'
    assert real.content_digest(torch.load(path, map_location='cpu', weights_only=True)) == REAL_SOURCE_CONTENT_SHA256[name]
    if torch.__version__.split('+')[0] == '2.5.1':
        assert file_sha256(path) == RESEARCH_FILE_SHA256[name]


@pytest.mark.slow
@pytest.mark.parametrize('name', real.NAMES)
def test_sources_rebuild_from_their_split_files(name, tmp_path):
    graph = KGS / name
    if not all((graph / f'{split}.txt').is_file() for split in ('train', 'valid', 'test')):
        pytest.skip(f'{graph} holds no split files (KGs.zip holds FB15k-237 and WN18RR; docs/proofpfn.md names the CoDEx-M source)')
    check_rebuild(name, graph, tmp_path / 'sources')


@pytest.mark.slow
def test_codex_m_rebuilds_from_its_upstream_split_files(tmp_path):
    """The upstream split files of CoDEx-M, written back from the pinned source, rebuild the pinned source."""
    require_real_sources()
    data = real.read_source('CoDEx-M', real.source_folder() / 'CoDEx-M' / 'data.pt')
    graph = tmp_path / 'CoDEx-M'
    graph.mkdir()
    entities, relations = data['entities'], data['relations']
    for split, ids in data['splits'].items():
        (graph / f'{split}.txt').write_bytes(''.join(f'{entities[h]}\t{relations[r]}\t{entities[t]}\n' for h, r, t in ids.tolist()).encode())
        assert file_sha256(graph / f'{split}.txt') == CODEX_M_SPLITS[split], split
    check_rebuild('CoDEx-M', graph, tmp_path / 'sources')


def test_a_source_with_other_split_files_is_refused(tmp_path):
    graph = tmp_path / 'graph'
    graph.mkdir()
    for split, rows in (('train', 'a r b\nb r c\n'), ('valid', 'a r c\n'), ('test', 'c r a\n')):
        (graph / f'{split}.txt').write_text(rows)
    with pytest.raises(ValueError, match='other triples than the research files of WN18RR'):
        real.build_source('WN18RR', graph, tmp_path / 'sources')
    assert not (tmp_path / 'sources').exists()   # nothing is written
    with pytest.raises(ValueError, match='unknown real source graph'):
        real.build_source('YAGO3-10', graph, tmp_path / 'sources')


def test_real_family_parameters():
    assert FamilySpec('real', 'real', 400_000, (('max_entities', 5000),)).overrides == {'max_entities': 5000}
    with pytest.raises(ValueError, match='unknown parameters'):
        FamilySpec('real', 'real', 400_000, (('rows', (1, 2)),))
