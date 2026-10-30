"""The stage-2 worlds (``priors.stage2``: structured, statistical and stage-2 plain worlds) against golden G2."""
import pytest

from tests.synthetic_worlds import check_world, golden_names

FAST = ('42/S/structured/50', '42/S/statistical/2')


@pytest.mark.parametrize('name', FAST)
def test_stage2_world_equals_the_research_world(name):
    check_world(name)


@pytest.mark.slow
def test_every_golden_stage2_world():
    names = golden_names({'structured', 'statistical'})
    assert len(names) == 22
    for name in names:
        check_world(name)
