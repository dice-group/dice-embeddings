"""The pinned worlds for the tests of the world generators (``tests/test_synthetic_<group>.py``).

The pins (``tests/proofpfn_fixtures.py``) hold 123 worlds by ``'<seed>/<prior>/<family>/<position>'``: at seed 42 the
stream positions 0 to 4, 50 and 500 of each of the seven families of the synthetic prior, positions 0 to 4 of the four
families with the structural null and 0 to 4 and 50 of the real family; at seeds 43 and 44 positions 0 and 1 of each.
A generated world equals its pin when its content digest and the digest of its record (task order, metadata and the
template provenance of each task) are equal.

Each group's test file checks a few small worlds and, in a test marked ``slow`` (deselect with ``-m "not slow"``), all
of its pinned worlds.
"""
from typing import Optional

from tests.proofpfn_fixtures import content_sha256, pinned_world, pins, record_sha256

WORLDS = pins()['worlds']
"""The pinned worlds by ``'<seed>/<prior>/<family>/<position>'`` (prior: ``synthetic``, ``null`` or ``real``)."""


def pinned_names(families: set[str], priors: Optional[set[str]] = None) -> list[str]:
    """The names of the pinned worlds of some families (and priors)."""
    return [name for name in WORLDS if name.split('/')[2] in families and (priors is None or name.split('/')[1] in priors)]


def check_world(name: str) -> None:
    """Generate a pinned world and compare its content and its record."""
    _, generated = pinned_world(name)
    assert content_sha256(generated.batch) == WORLDS[name]['content'], f'{name}: the content differs'
    assert record_sha256(generated) == WORLDS[name]['record'], f'{name}: the task order, the metadata or the template provenance differs'
