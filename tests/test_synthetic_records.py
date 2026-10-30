"""The record families (``priors.records``: categorical2, attr2, rec1, list1) of arm S against golden G2."""
import pytest

from tests.synthetic_worlds import check_world, golden_names

FAST = ('42/S/categorical2/0',
        '42/S/attr2/3',     # its template is a rule world with pair-affinity rewiring
        '42/S/list1/4',
        '44/S/rec1/0')      # its template is a control world of the transitivity family


@pytest.mark.parametrize('name', FAST)
def test_record_world_equals_the_research_world(name):
    check_world(name)


@pytest.mark.slow
def test_every_golden_record_world():
    names = golden_names({'categorical2', 'attr2', 'rec1', 'list1'}, {'S'})
    assert len(names) == 44
    for name in names:
        check_world(name)
