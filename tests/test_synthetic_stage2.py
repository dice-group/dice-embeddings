"""The stage-2 worlds (``priors.stage2``: structured, statistical and stage-2 plain worlds) against their pinned worlds."""
import pytest

from tests.synthetic_worlds import check_world, pinned_names

FAST = ('42/synthetic/structured/50', '42/synthetic/statistical/2')


@pytest.mark.parametrize('name', FAST)
def test_stage2_world_equals_its_pin(name):
    check_world(name)


@pytest.mark.slow
def test_every_pinned_stage2_world():
    names = pinned_names({'structured', 'statistical'})
    assert len(names) == 22
    for name in names:
        check_world(name)
