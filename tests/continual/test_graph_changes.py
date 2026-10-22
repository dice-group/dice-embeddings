import numpy as np
import pytest

from dicee.continual.graph_changes import GraphSnapshot, diff_snapshots


def test_diff_added_triples():
    prev = GraphSnapshot.from_array(np.array([[0, 0, 1], [1, 0, 2]]))
    cur = GraphSnapshot.from_array(np.array([[0, 0, 1], [1, 0, 2], [2, 1, 3]]))
    diff = diff_snapshots(prev, cur)
    assert diff.added_triples == frozenset({(2, 1, 3)})
    assert diff.active_entities == {0, 1, 2, 3}
    assert diff.new_entities == {3}
    assert diff.new_relations == {1}


def test_diff_no_changes():
    prev = GraphSnapshot.from_array(np.array([[0, 0, 1]]))
    diff = diff_snapshots(prev, prev)
    assert diff.added_triples == frozenset()
    assert diff.new_entities == frozenset()


def test_diff_requires_superset():
    prev = GraphSnapshot.from_array(np.array([[0, 0, 1], [5, 0, 6]]))
    cur = GraphSnapshot.from_array(np.array([[0, 0, 1]]))
    with pytest.raises(ValueError):
        diff_snapshots(prev, cur)
