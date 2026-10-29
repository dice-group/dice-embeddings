"""The golden worlds of G2 for the tests of the world generators (``tests/test_synthetic_<group>.py``).

G2 (``tests/fixtures/proofpfn/g2_worlds.json.gz``) holds 123 research worlds: at seed 42 the stream positions 0 to 4,
50 and 500 of each of S's seven families, positions 0 to 4 of N's four nulled families and 0 to 4 and 50 of the real
family; at seeds 43 and 44 positions 0 and 1 of each. A clean world equals its golden world when its content digest
(``batch.content_sha256``), its research fingerprint (``priors.research_fingerprint``: the content plus the template's
RecipeBatch provenance, so also the latent legacy world), its metadata (the research keys, timings excluded), its
template provenance and its task order are equal.

Each group's test file checks a few small worlds in the default run and all of its golden worlds in a slow test, which
runs with ``-m slow`` or ``PROOFPFN_SLOW=1``.
"""
import os
from typing import Optional

import pytest
import torch

from dicee.synthetic_kg.arms import ARMS
from dicee.synthetic_kg.batch import content_sha256, content_tensors
from dicee.synthetic_kg.digests import per_key_digests
from dicee.synthetic_kg.priors import Slot, research_fingerprint, world
from tests.proofpfn_fixtures import fixture

G2 = fixture('g2_worlds.json.gz')
WORLDS = {f"{w['seed']}/{w['arm']}/{w['family']}/{w['k']}": w for w in G2['worlds']}
"""The golden worlds by ``'<seed>/<arm>/<family>/<k>'`` (k: the position in the family's stream)."""

TASK_FIELDS = ('world_sha256', 'sampling_seed', 'requested_positives')


def golden_names(families: set[str], arms: Optional[set[str]] = None) -> list[str]:
    """The names of the golden worlds of some families (and arms)."""
    return [name for name, w in WORLDS.items() if w['family'] in families and (arms is None or w['arm'] in arms)]


def require_slow(request: pytest.FixtureRequest) -> None:
    """Skip a slow check unless ``-m`` names the slow marker or ``PROOFPFN_SLOW`` is set."""
    if 'slow' not in (request.config.getoption('markexpr') or '') and not os.environ.get('PROOFPFN_SLOW'):
        pytest.skip('slow check: run with -m slow or PROOFPFN_SLOW=1')


def check_world(name: str) -> None:
    """Build a golden world with the clean generator (2 torch threads) and compare every recorded digest."""
    golden = WORLDS[name]
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = world(golden['seed'], Slot(golden['family'], golden['source'], golden['index']), ARMS[golden['arm']].prior)
    finally:
        torch.set_num_threads(threads)
    if content_sha256(result.batch) != golden['content_sha256']:
        keys = per_key_digests(content_tensors(result.batch))
        pytest.fail(f"{name}: the content differs in {[k for k in keys if keys[k] != golden['content_per_key'].get(k)]}")
    assert list(result.batch) == list(golden['research_fingerprint']), name
    assert research_fingerprint(result) == golden['research_fingerprint'], name
    assert {k: v for k, v in result.meta.items() if k != 'recipe_batch'} == golden['meta'], name
    assert result.meta['recipe_batch'] == {t: {k: golden['tasks'][t][k] for k in TASK_FIELDS} for t in result.batch}, name
