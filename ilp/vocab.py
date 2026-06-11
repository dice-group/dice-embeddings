"""Vocabulary construction and persistence (spec §3).

The embedding matrix only ever contains schema tokens and a fixed Z pool.
Specific instance entities never get their own embeddings — they are mapped
to anonymous [Z_i] tokens at sample time (see dataset.py).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

PAD = "[PAD]"
X = "[X]"
INV_SUFFIX = "__inv"

# Per-entity hop-distance tokens (see dataset.build_sample). Encodes BFS
# distance from the anchor — analogous to positional encoding in NLP but
# over graph distance, not sequence index. Vocab size is coupled to
# `subgraph_hops`: a checkpoint trained at depth K allocates [HOP_NONE]
# + [HOP_DIST_0..K]. Untrained slots aren't useful at inference, so we
# don't reserve them.
HOP_NONE = "[HOP_NONE]"


def hop_distance_token(distance: int) -> str:
    """Map a non-negative integer hop distance to its token string.

    Distances < 0 return HOP_NONE (relation positions, disconnected).
    The caller is responsible for not querying distances beyond the depth
    the vocab was built for — a k-hop subgraph cannot produce d > k.
    """
    if distance < 0:
        return HOP_NONE
    return f"[HOP_DIST_{distance}]"


def inverse_relation(r: str) -> str:
    """Idempotent: inverse(inverse(r)) == r. See dataset.augment_with_inverse."""
    if r.endswith(INV_SUFFIX):
        return r[: -len(INV_SUFFIX)]
    return r + INV_SUFFIX


def build_vocab(
    triples: Iterable[tuple[str, str, str]],
    z_pool_size: int = 100,
    cardinality_cutoff: int = 0,
    tail_diversity_cutoff: float = 0.0,
    type_relation: str | Iterable[str] | None = "rdf:type",
    subgraph_hops: int = 2,
) -> tuple[dict[str, int], set[str]]:
    """Build vocab from training triples.

    Returns (vocab, fixed_values). `fixed_values` are the entity strings that
    received their own [VAL_*] token (classes + controlled-vocabulary values).
    Anything outside `fixed_values` is treated as an instance and anonymized
    at sample time.

    `type_relation` lists relations whose tails are *always* promoted to
    schema-level [VAL_*] tokens regardless of cardinality. Use this for
    type-bearing relations like `rdf:type` or WordNet's `_hypernym` /
    `_instance_hypernym` whose ranges form the taxonomic backbone. Accepts:

    - a string (one relation, back-compat): ``"rdf:type"``
    - a list/tuple of strings: ``["_hypernym", "_instance_hypernym"]``
    - an empty string or ``None``: no relations promoted by name.

    Two cardinality heuristics auto-promote a relation's *entire* range to
    [VAL_*]; a relation is promoted if **either** fires (both default off):

    - `tail_diversity_cutoff` (preferred): promote when the tail-diversity
      ratio ``distinct_tails / triples_with_this_relation < cutoff``. This is
      scale-invariant — it asks "how reusable is the average tail value?", so
      a 700-code controlled vocabulary seen 20× each is promoted while a 1:1
      instance relation is not, regardless of absolute size. ``0.0`` disables.
    - `cardinality_cutoff`: promote when a relation has fewer than `cutoff`
      *distinct* tails. Absolute-count fallback; brittle as the graph grows
      (misses large-but-low-diversity code lists). ``0`` disables.
    """
    if type_relation is None or type_relation == "":
        schema_rels: set[str] = set()
    elif isinstance(type_relation, str):
        schema_rels = {type_relation}
    else:
        schema_rels = {r for r in type_relation if r}

    triples = list(triples)
    relations = sorted({r for _, r, _ in triples})
    classes = sorted({o for s, r, o in triples if r in schema_rels})

    fixed_values: set[str] = set(classes)
    range_by_rel: dict[str, set[str]] = defaultdict(set)
    count_by_rel: dict[str, int] = defaultdict(int)
    for _, r, o in triples:
        range_by_rel[r].add(o)
        count_by_rel[r] += 1
    for r, vals in range_by_rel.items():
        n_distinct = len(vals)
        promote = n_distinct < cardinality_cutoff
        if tail_diversity_cutoff > 0.0:
            diversity = n_distinct / count_by_rel[r]
            promote = promote or diversity < tail_diversity_cutoff
        if promote:
            fixed_values.update(vals)

    vocab: dict[str, int] = {PAD: 0, X: 1}
    for i in range(z_pool_size):
        vocab[f"[Z_{i}]"] = len(vocab)
    for r in relations:
        vocab[f"[REL_{r}]"] = len(vocab)
        vocab[f"[REL_{inverse_relation(r)}]"] = len(vocab)
    for v in sorted(fixed_values):
        vocab[f"[VAL_{v}]"] = len(vocab)
    # Hop-distance tokens. Allocated to exactly cover the training subgraph
    # depth: `[HOP_NONE]` + `[HOP_DIST_0..subgraph_hops]`. Always present
    # (so the use_hop_distance_tokens flag can flip on/off without rebuilding
    # the vocab), but sized so unused slots can't accumulate.
    vocab[HOP_NONE] = len(vocab)
    for h in range(subgraph_hops + 1):
        vocab[f"[HOP_DIST_{h}]"] = len(vocab)
    return vocab, fixed_values


def is_schema(entity: str, fixed_values: set[str]) -> bool:
    return entity in fixed_values


def format_vocab_summary(
    triples: Iterable[tuple[str, str, str]],
    fixed_values: set[str],
    type_relation: str | Iterable[str] | None = None,
    cardinality_cutoff: int = 0,
    tail_diversity_cutoff: float = 0.0,
) -> str:
    """Return a banner-style summary of vocab schema/instance breakdown.

    Highlights what fraction of entities became schema [VAL_*] vs anonymized
    [Z_*], attributes them to each schema relation, and warns loudly when
    no schema was found.
    """
    triples = list(triples)
    ents = {s for s, _, _ in triples} | {o for _, _, o in triples}
    total = len(ents)
    n_schema = len(fixed_values)
    n_inst = total - n_schema
    pct = (100.0 * n_schema / total) if total else 0.0

    if type_relation is None or type_relation == "":
        rels: list[str] = []
    elif isinstance(type_relation, str):
        rels = [type_relation]
    else:
        rels = [r for r in type_relation if r]

    bar = "─" * 72
    lines = [bar, "  Vocab construction summary", bar,
             f"  Total entities:     {total:>6,}",
             f"  Schema  [VAL_*]:    {n_schema:>6,}  ({pct:5.1f}%)",
             f"  Instances [Z_*]:    {n_inst:>6,}  ({100 - pct:5.1f}%)"]

    if rels:
        from collections import defaultdict as _dd
        range_by_rel: dict[str, set[str]] = _dd(set)
        for _h, r, t in triples:
            range_by_rel[r].add(t)
        lines.append("")
        lines.append("  Schema relations (tails promoted to [VAL_*]):")
        for r in rels:
            n_tails = len(range_by_rel.get(r, set()))
            present = "" if r in range_by_rel else "  (NOT in training KG)"
            lines.append(f"    {r:35s} → {n_tails:>5,} tails{present}")
    else:
        lines.append("")
        lines.append("  type_relation: <none declared>")

    if tail_diversity_cutoff and tail_diversity_cutoff > 0.0:
        from collections import defaultdict as _dd
        rng_by_rel: dict[str, set[str]] = _dd(set)
        cnt_by_rel: dict[str, int] = _dd(int)
        for _h, r, t in triples:
            rng_by_rel[r].add(t)
            cnt_by_rel[r] += 1
        promoted = sorted(
            (len(v) / cnt_by_rel[r], r, len(v))
            for r, v in rng_by_rel.items()
            if len(v) / cnt_by_rel[r] < tail_diversity_cutoff
        )
        lines.append("")
        lines.append(f"  tail_diversity_cutoff: {tail_diversity_cutoff:.2f} "
                     "(relations with distinct_tails/triples below this are auto-promoted)")
        lines.append(f"  → {len(promoted)} relation(s) promoted by diversity:")
        for div, r, n_tails in promoted:
            lines.append(f"    {r:35s} → {n_tails:>5,} tails  (diversity {div:6.2%})")
    else:
        lines.append("  tail_diversity_cutoff: 0.0 (diversity auto-promotion disabled)")

    if cardinality_cutoff and cardinality_cutoff > 0:
        lines.append(f"  cardinality_cutoff: {cardinality_cutoff} "
                     "(relations with fewer distinct tails are auto-promoted)")
    else:
        lines.append("  cardinality_cutoff: 0 (count auto-promotion disabled)")

    lines.append(bar)
    if n_schema == 0:
        lines += [
            "  ⚠  WARNING: no schema entities found.",
            "     All entities will anonymize to [Z_*]. The model loses access to",
            "     any stable type identity. To fix this, either:",
            "       • set `type_relation` to a list of schema-bearing relations",
            "         (e.g. ['rdf:type', '_hypernym'])",
            "       • set `tail_diversity_cutoff` > 0 (e.g. 0.2) to auto-promote",
            "         low-diversity controlled-vocabulary relations.",
            bar,
        ]
    return "\n".join(lines)


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
