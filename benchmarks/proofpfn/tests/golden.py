"""Golden fixtures and test inputs of the ProofPFN benchmark tests.

Small fixtures are copies in ``golden/`` (see its README); the G6 ranks and the UMLS scores, which the dicee tests use
too, are read from ``tests/fixtures/proofpfn``. Large golden files (research exports, full row files) stay outside the
repository: ``golden_large`` and ``golden_root`` (``tests/proofpfn_fixtures.py``) skip the test unless
``PROOFPFN_GOLDEN`` names the proofpfn-golden folder.
"""
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.proofpfn.manifests import FORMAT, content_sha256, file_sha256
from tests.proofpfn_fixtures import FIXTURES, golden_large, golden_root

GOLDEN = Path(__file__).resolve().parent / 'golden'
SHARED = ('g6_ranks.json.gz', 'umls_research_scores.json')
"""The fixtures of ``tests/fixtures/proofpfn`` that the benchmark tests read too."""
G6_KEYS = {'gs': ('S', 'REC10A-XLE-PCU-noA'), 'gcu': ('CU', 'REC10A-XLE-CU-noA'), 'gul': ('UL', 'REC10A-XLE-UL-noA'),
           'ght': ('HT', 'PCU-HT')}   # the test keys of G6: arm and research --arm of the G4 exports
G6_REGISTERED = {'gs': 'rec10axlepcunoa', 'gcu': 'rec10axlecunoa', 'gul': 'rec10axleulnoa', 'ght': 'pcuht'}


def fixture(name: str) -> Any:
    """A JSON fixture of ``golden/`` (of ``tests/fixtures/proofpfn`` for ``SHARED``), gzip-compressed when the name ends
    with ``.gz``."""
    data = ((FIXTURES if name in SHARED else GOLDEN) / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)


def fixture_bytes(name: str) -> bytes:
    """The bytes of a fixture of ``golden/``, decompressed when the name ends with ``.gz``."""
    data = (GOLDEN / name).read_bytes()
    return gzip.decompress(data) if name.endswith('.gz') else data


def g4_export(arm_key: str) -> str:
    """The template with ``{seed}`` of a G4 export at update 20 (skips without the large goldens)."""
    _, run = G6_KEYS[arm_key]
    golden_large(f'g4/golden-runs/{run}/training/42/export-20.pt')
    return str(golden_root() / 'golden' / 'large' / 'g4' / 'golden-runs' / run / 'training' / '{seed}' / 'export-20.pt')


def int_digest(values: Any) -> str:
    """The rank digest of the goldens: SHA-256 of int64 little-endian values."""
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def write_graph(folder: Path, train: list[str], valid: list[str], test: list[str]) -> Path:
    """A graph folder from lines of tab-separated triples."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, lines in (('train', train), ('valid', valid), ('test', test)):
        (folder / f'{name}.txt').write_text(''.join(f'{line}\n' for line in lines), encoding='utf-8')
    return folder


def write_manifest(path: Path, root: Path, folders: dict[str, Path], name: str = 'tiny', blind: bool = False,
                   sample_rule: str = 'first-1000') -> Path:
    """A manifest of graph folders below ``root`` with their current file and content hashes."""
    graphs = []
    for graph, folder in folders.items():
        train = [line.split() for line in (folder / 'train.txt').read_text().splitlines() if line.strip()]
        graphs.append(dict(key=f'tiny/{graph}', name=graph, path=str(folder.relative_to(root)), collection='tiny', view='tiny',
                           code=None, group=None, verdict=None, weight=0.0, origin_class=None, suite_id=None,
                           size=dict(entities=len({e for h, _, t in train for e in (h, t)}), inference_edges=len({tuple(r) for r in train})),
                           sha256={s: file_sha256(folder / f'{s}.txt') for s in ('train', 'valid', 'test')},
                           content_sha256={s: content_sha256(folder / f'{s}.txt') for s in ('train', 'test')}, counts=None, tail_only_view=None))
    path.write_text(json.dumps(dict(format=FORMAT, name=name, blind=blind, sample_rule=sample_rule, graphs=graphs), indent=1))
    return path


def tiny_graph(folder: Path, entities: int = 16, relations: int = 3, seed: int = 0) -> Path:
    """A small random graph: self-loops only in r0 (open under the rule), a duplicate train row, and a test fact of r1
    whose answer is its anchor (closed under the rule)."""
    import random
    rng = random.Random(seed)
    names = [f'e{i:02d}' for i in range(entities)]
    rels = [f'r{i}' for i in range(relations)]
    facts = sorted({(h, r, t) for h, r, t in ((rng.choice(names), rng.choice(rels), rng.choice(names)) for _ in range(6 * entities)) if h != t})
    rng.shuffle(facts)
    train = facts[: 4 * len(facts) // 5]
    rest = facts[len(train):]
    test, valid = rest[: len(rest) // 2], rest[len(rest) // 2:]
    train += [train[0], ('e00', 'r0', 'e00'), ('e01', 'r0', 'e01')]   # a duplicate row and self-loops (r0 is reflexive)
    test.append(('e02', 'r1', 'e02'))                                  # the anchor is the answer of both queries
    return write_graph(folder, ['\t'.join(f) for f in train], ['\t'.join(f) for f in valid], ['\t'.join(f) for f in test])
