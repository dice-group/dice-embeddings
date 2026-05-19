"""Vocabulary construction and persistence (spec §3).

The embedding matrix only ever contains schema tokens and a fixed Z pool.
Specific instance entities never get their own embeddings — they are mapped
to anonymous [Z_i] tokens at sample time (see dataset.py).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

PAD = "[PAD]"
X = "[X]"
INV_SUFFIX = "__inv"


def inverse_relation(r: str) -> str:
    """Idempotent: inverse(inverse(r)) == r. See dataset.augment_with_inverse."""
    if r.endswith(INV_SUFFIX):
        return r[: -len(INV_SUFFIX)]
    return r + INV_SUFFIX


def build_vocab(
    triples: Iterable[tuple[str, str, str]],
    z_pool_size: int = 100,
    cardinality_cutoff: int = 100,
    type_relation: str = "rdf:type",
) -> tuple[dict[str, int], set[str]]:
    """Build vocab from training triples.

    Returns (vocab, fixed_values). `fixed_values` are the entity strings that
    received their own [VAL_*] token (classes + controlled-vocabulary values).
    Anything outside `fixed_values` is treated as an instance and anonymized
    at sample time.
    """
    triples = list(triples)
    relations = sorted({r for _, r, _ in triples})
    classes = sorted({o for s, r, o in triples if r == type_relation})

    fixed_values: set[str] = set(classes)
    range_by_rel: dict[str, set[str]] = defaultdict(set)
    for _, r, o in triples:
        range_by_rel[r].add(o)
    for r, vals in range_by_rel.items():
        if len(vals) < cardinality_cutoff:
            fixed_values.update(vals)

    vocab: dict[str, int] = {PAD: 0, X: 1}
    for i in range(z_pool_size):
        vocab[f"[Z_{i}]"] = len(vocab)
    for r in relations:
        vocab[f"[REL_{r}]"] = len(vocab)
        vocab[f"[REL_{inverse_relation(r)}]"] = len(vocab)
    for v in sorted(fixed_values):
        vocab[f"[VAL_{v}]"] = len(vocab)
    return vocab, fixed_values


def is_schema(entity: str, fixed_values: set[str]) -> bool:
    return entity in fixed_values


def z_token_ids(vocab: dict[str, int], z_pool_size: int) -> list[int]:
    return [vocab[f"[Z_{i}]"] for i in range(z_pool_size)]


def save_vocab(
    vocab: dict[str, int],
    fixed_values: set[str],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vocab": vocab, "fixed_values": sorted(fixed_values)}))


def load_vocab(path: str | Path) -> tuple[dict[str, int], set[str]]:
    data = json.loads(Path(path).read_text())
    return data["vocab"], set(data["fixed_values"])
