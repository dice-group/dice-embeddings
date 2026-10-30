"""The golden worlds of G2 for the tests of the world generators (``tests/test_synthetic_<group>.py``).

G2 (``tests/fixtures/proofpfn/g2_worlds.json.gz``) holds 123 research worlds: at seed 42 the stream positions 0 to 4,
50 and 500 of each of S's seven families, positions 0 to 4 of N's four nulled families and 0 to 4 and 50 of the real
family; at seeds 43 and 44 positions 0 and 1 of each. A clean world equals its golden world when its content digest
(``content_sha256``), its metadata (the research keys, timings excluded), its template provenance (the sampling
seed and requested positives of each task) and its task order are equal.

Each group's test file checks a few small worlds and, in a test marked ``slow`` (deselect with ``-m "not slow"``), all
of its golden worlds.
"""
from typing import Mapping, Optional

import pytest
import torch

from dicee.synthetic_kg.arms import ARMS
from dicee.synthetic_kg.batch import WorldBatch
from dicee.synthetic_kg.digests import tensor_digest
from dicee.synthetic_kg.priors import Slot, world
from tests.proofpfn_fixtures import fixture, per_key_digests

G2 = fixture('g2_worlds.json.gz')
WORLDS = {f"{w['seed']}/{w['arm']}/{w['family']}/{w['k']}": w for w in G2['worlds']}
"""The golden worlds by ``'<seed>/<arm>/<family>/<k>'`` (k: the position in the family's stream)."""

TASK_FIELDS = ('sampling_seed', 'requested_positives')


def content_tensors(batch: Mapping[str, WorldBatch]) -> dict[str, torch.Tensor]:
    """The training content of a world under the keys ``'<task>/<name>'``: per task ``groups`` (int64), ``head_prediction``
    (bool), ``context`` and ``truth`` as sorted int64 ``[n, 3]``, and ``num_entities`` and ``num_relations`` as int64
    scalars."""
    out = {}
    for task, b in batch.items():
        out.update({f'{task}/groups': b.groups.to(torch.long), f'{task}/head_prediction': b.head_prediction.to(torch.bool),
                    f'{task}/context': torch.tensor(sorted(b.context), dtype=torch.long).reshape(-1, 3),
                    f'{task}/truth': torch.tensor(sorted(b.truth), dtype=torch.long).reshape(-1, 3),
                    f'{task}/num_entities': torch.tensor(int(b.num_entities), dtype=torch.long),
                    f'{task}/num_relations': torch.tensor(int(b.num_relations), dtype=torch.long)})
    return out


def content_sha256(batch: Mapping[str, WorldBatch]) -> str:
    """The content digest of a world (golden G2's ``content_sha256``): the tensor digest of its ``content_tensors``, which
    does not depend on the task order."""
    return tensor_digest(content_tensors(batch))


def golden_names(families: set[str], arms: Optional[set[str]] = None) -> list[str]:
    """The names of the golden worlds of some families (and arms)."""
    return [name for name, w in WORLDS.items() if w['family'] in families and (arms is None or w['arm'] in arms)]


def check_world(name: str) -> None:
    """Build a golden world with the clean generator and compare every recorded digest."""
    golden = WORLDS[name]
    result = world(golden['seed'], Slot(golden['family'], golden['source'], golden['index']), ARMS[golden['arm']].prior)
    if content_sha256(result.batch) != golden['content_sha256']:
        keys = per_key_digests(content_tensors(result.batch))
        pytest.fail(f"{name}: the content differs in {[k for k in keys if keys[k] != golden['content_per_key'].get(k)]}")
    assert list(result.batch) == list(golden['research_fingerprint']), name   # the task order
    assert {k: v for k, v in result.meta.items() if k != 'recipe_batch'} == golden['meta'], name
    assert result.meta['recipe_batch'] == {t: {k: golden['tasks'][t][k] for k in TASK_FIELDS} for t in result.batch}, name
