"""The record families (``priors.records``: categorical2, attr2, rec1, list1) of the synthetic prior against their pinned worlds."""
import pytest

from tests.synthetic_worlds import check_world, pinned_names

FAST = ('42/synthetic/categorical2/0',
        '42/synthetic/attr2/3',     # its template is a rule world with pair-affinity rewiring
        '42/synthetic/list1/4',
        '44/synthetic/rec1/0')      # its template is a control world of the transitivity family


@pytest.mark.parametrize('name', FAST)
def test_record_world_equals_its_pin(name):
    check_world(name)


@pytest.mark.slow
def test_every_pinned_record_world():
    names = pinned_names({'categorical2', 'attr2', 'rec1', 'list1'}, {'synthetic'})
    assert len(names) == 44
    for name in names:
        check_world(name)
