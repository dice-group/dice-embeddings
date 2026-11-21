"""Fixtures and test inputs of the ProofPFN benchmark tests.

Small fixtures are in ``fixtures/`` (see its README). The regression pins of the gate graphs and of the evaluation on
worlds of the prior are in ``tests/fixtures/proofpfn/pins.json`` (``tests.proofpfn_fixtures.pins``).
"""
import json
from pathlib import Path
from typing import Any, Optional

from benchmarks.proofpfn.manifests import FORMAT, content_sha256, file_sha256
from tests.proofpfn_fixtures import int_digest, pins

__all__ = ['FIXTURES', 'GATE', 'fixture', 'int_digest', 'pins', 'tiny_graph', 'write_graph', 'write_manifest']

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
GATE = FIXTURES / 'gate.json'
"""The manifest of three small graphs with pinned sizes and row orders: UMLS, NELLInductive_v1 and grail-WN18RR_v1."""


def fixture(name: str) -> Any:
    """A JSON fixture of ``fixtures/``."""
    return json.loads((FIXTURES / name).read_text())


def write_graph(folder: Path, train: list[str], valid: list[str], test: list[str]) -> Path:
    """A graph folder from lines of tab-separated triples."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, lines in (('train', train), ('valid', valid), ('test', test)):
        (folder / f'{name}.txt').write_text(''.join(f'{line}\n' for line in lines), encoding='utf-8')
    return folder


def write_manifest(path: Path, root: Path, folders: dict[str, Path], name: str = 'tiny', sample_rule: str = 'first-1000',
                   views: Optional[dict[str, str]] = None) -> Path:
    """A manifest of graph folders below ``root`` with their current file and content hashes (``views``: graph to view)."""
    graphs = []
    for graph, folder in folders.items():
        train = [line.split() for line in (folder / 'train.txt').read_text().splitlines() if line.strip()]
        graphs.append(dict(name=graph, path=str(folder.relative_to(root)), view=(views or {}).get(graph, 'tiny'), suite_id=None,
                           size=dict(entities=len({e for h, _, t in train for e in (h, t)}), inference_edges=len({tuple(r) for r in train})),
                           sha256={s: file_sha256(folder / f'{s}.txt') for s in ('train', 'valid', 'test')},
                           content_sha256={s: content_sha256(folder / f'{s}.txt') for s in ('train', 'test')}, counts=None))
    path.write_text(json.dumps(dict(format=FORMAT, name=name, sample_rule=sample_rule, graphs=graphs), indent=1))
    return path


def tiny_graph(folder: Path, entities: int = 16, relations: int = 3, seed: int = 0) -> Path:
    """A small random graph: self-loops only in r0, a duplicate train row, and a test fact of r1 whose answer is its
    anchor."""
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
