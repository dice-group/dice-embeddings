"""KG container, subgraph extraction, sample construction (spec §4).

The dataset returns dicts of tensors ready for the model. Anonymization
happens at sample draw time — that's what gives Z-randomization its
regularization effect (spec §6).
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .vocab import inverse_relation

Triple = tuple[str, str, str]


def augment_with_inverse(triples: Iterable[Triple]) -> list[Triple]:
    """Return original triples plus (t, r_inv, h) for each (h, r, t).

    Used to make the KG bidirectional so the model sees inverse relations
    in subgraph context, not only as target tokens.
    """
    out: list[Triple] = []
    for h, r, t in triples:
        out.append((h, r, t))
        out.append((t, inverse_relation(r), h))
    return out


class KnowledgeGraph:
    """Adjacency over a triple set. Entities are looked up by string."""

    def __init__(self, triples: Iterable[Triple]):
        self.triples: list[Triple] = list(triples)
        self.triple_set: set[Triple] = set(self.triples)
        self.entities: list[str] = sorted(
            {s for s, _, _ in self.triples} | {o for _, _, o in self.triples}
        )
        self.relations: list[str] = sorted({r for _, r, _ in self.triples})
        self.adj: dict[str, set[Triple]] = defaultdict(set)
        for s, r, o in self.triples:
            self.adj[s].add((s, r, o))
            self.adj[o].add((s, r, o))

    def __contains__(self, triple: Triple) -> bool:
        return triple in self.triple_set


# Triple file formats. Each entry maps a column-order name to the indices
# of (head, relation, tail) within a whitespace/tab-split line. Add new
# formats here as needed — read_triples and configs reference these names.
TRIPLE_FORMATS: dict[str, tuple[int, int, int]] = {
    "head_tail_relation": (0, 2, 1),  # DBpedia50: h <TAB> t <TAB> r
    "head_relation_tail": (0, 1, 2),  # standard KG-completion: h <TAB> r <TAB> t
}


def read_triples(
    path: str | Path,
    fmt: str = "head_tail_relation",
) -> list[Triple]:
    """Parse a triples file into canonical (head, relation, tail) order.

    `fmt` selects the column layout of the source file (see TRIPLE_FORMATS).
    """
    if fmt not in TRIPLE_FORMATS:
        raise ValueError(
            f"Unknown triple format {fmt!r}. Known: {sorted(TRIPLE_FORMATS)}"
        )
    h_idx, r_idx, t_idx = TRIPLE_FORMATS[fmt]
    out: list[Triple] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Bad triple line: {line!r}")
        out.append((parts[h_idx], parts[r_idx], parts[t_idx]))
    return out


def two_hop_neighborhood(node: str, kg: KnowledgeGraph) -> set[Triple]:
    """All triples within two hops of `node` (inclusive of 1-hop)."""
    one_hop = kg.adj.get(node, set())
    neighbors: set[str] = set()
    for s, _, o in one_hop:
        neighbors.add(s)
        neighbors.add(o)
    two_hop = set(one_hop)
    for n in neighbors:
        two_hop |= kg.adj.get(n, set())
    return two_hop


def build_sample(
    anchor: str,
    relation: str,
    candidate: str,
    label: float,
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    max_triples: int = 64,
    z_pool: int = 100,
    rng: random.Random | None = None,
    exclude_triple: Triple | None = None,
    collapse_z: bool = False,
) -> dict[str, torch.Tensor]:
    """Build one anonymized sample anchored on `anchor`.

    `anchor` gets the [X] token. `candidate` gets either its [VAL_*] (schema)
    or a [Z_i] consistent with the subgraph's anonymization.

    `exclude_triple`: drop this triple from the subgraph if present. For
    positive training samples pass (anchor, relation, candidate) so the
    target is never leaked into context. For negatives, leave it None.

    `collapse_z`: when True, if the Z pool is exhausted, distinct entities
    are mapped to a uniformly-random already-existing [Z_i] instead of
    raising. Used for the z_pool=1 ablation that asks "does variable
    binding across triples matter?".
    """
    rng = rng or random
    subgraph = two_hop_neighborhood(anchor, kg)
    if exclude_triple is not None:
        s, r, o = exclude_triple
        subgraph.discard((s, r, o))
        subgraph.discard((o, inverse_relation(r), s))

    if len(subgraph) > max_triples:
        subgraph_list = rng.sample(list(subgraph), max_triples)
    else:
        subgraph_list = list(subgraph)

    z_indices = list(range(z_pool))
    rng.shuffle(z_indices)
    entity_map: dict[str, int] = {anchor: vocab["[X]"]}

    def tok_entity(node: str) -> int:
        if node in entity_map:
            return entity_map[node]
        if node in fixed_values:
            return vocab[f"[VAL_{node}]"]
        if not z_indices:
            if collapse_z:
                entity_map[node] = vocab[f"[Z_{rng.randrange(z_pool)}]"]
                return entity_map[node]
            raise RuntimeError(
                f"Z pool exhausted (size={z_pool}); subsample more aggressively "
                f"(max_triples={max_triples}, |subgraph|={len(subgraph_list)})"
            )
        entity_map[node] = vocab[f"[Z_{z_indices.pop()}]"]
        return entity_map[node]

    def tok_rel(rel: str) -> int:
        return vocab[f"[REL_{rel}]"]

    triples_tok = [[tok_entity(s), tok_rel(rel), tok_entity(o)] for s, rel, o in subgraph_list]
    rng.shuffle(triples_tok)

    candidate_tok = tok_entity(candidate)
    relation_tok = tok_rel(relation)

    n = len(triples_tok)
    pad = [[0, 0, 0]] * (max_triples - n)
    return {
        "triples": torch.tensor(triples_tok + pad, dtype=torch.long),
        "mask": torch.tensor([True] * n + [False] * (max_triples - n), dtype=torch.bool),
        "target_relation": torch.tensor(relation_tok, dtype=torch.long),
        "target_tail": torch.tensor(candidate_tok, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.float),
    }


class InductiveKGDataset(Dataset):
    """Training dataset with on-the-fly negative sampling.

    Each positive triple yields (1 + neg_per_pos) samples per __getitem__
    call slot. We also randomly flip anchor direction so the model learns
    both `(h, r, ?)` and `(?, r, t)` queries (spec §7.1 evaluates both).
    """

    def __init__(
        self,
        positive_triples: Sequence[Triple],
        kg: KnowledgeGraph,
        vocab: dict[str, int],
        fixed_values: set[str],
        entity_pool: Sequence[str],
        max_triples: int = 64,
        z_pool: int = 100,
        neg_per_pos: int = 4,
        both_directions: bool = True,
        seed: int | None = None,
        collapse_z: bool = False,
    ):
        self.pos = list(positive_triples)
        self.kg = kg
        self.vocab = vocab
        self.fixed_values = fixed_values
        self.entity_pool = list(entity_pool)
        self.max_triples = max_triples
        self.z_pool = z_pool
        self.neg_per_pos = neg_per_pos
        self.both_directions = both_directions
        self._seed = seed
        self.collapse_z = collapse_z

    def __len__(self) -> int:
        return len(self.pos) * (1 + self.neg_per_pos)

    def _rng(self, idx: int) -> random.Random:
        # Per-call RNG keeps DataLoader workers reproducible if a seed is set.
        if self._seed is None:
            return random
        return random.Random(self._seed + idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = self._rng(idx)
        pos_idx = idx // (1 + self.neg_per_pos)
        slot = idx % (1 + self.neg_per_pos)
        h, r, t = self.pos[pos_idx]

        if self.both_directions and rng.random() < 0.5:
            anchor, candidate = t, h
            r_use = inverse_relation(r)
        else:
            anchor, candidate = h, t
            r_use = r
        exclude = (h, r, t)  # build_sample also drops the inverse form

        if slot == 0:
            return build_sample(
                anchor, r_use, candidate, 1.0, self.kg, self.vocab, self.fixed_values,
                self.max_triples, self.z_pool, rng, exclude_triple=exclude,
                collapse_z=self.collapse_z,
            )

        # Negative: corrupt the candidate side until we leave the known triple set.
        for _ in range(100):
            corrupt = rng.choice(self.entity_pool)
            corrupted_triple = (anchor, r_use, corrupt)
            if corrupted_triple not in self.kg.triple_set:
                candidate = corrupt
                break
        return build_sample(
            anchor, r_use, candidate, 0.0, self.kg, self.vocab, self.fixed_values,
            self.max_triples, self.z_pool, rng, exclude_triple=None,
            collapse_z=self.collapse_z,
        )


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def verify_inductive_split(train: Sequence[Triple], test: Sequence[Triple]) -> bool:
    """Spec §2: assert no test entity appears in train."""
    train_ents = {s for s, _, _ in train} | {o for _, _, o in train}
    test_ents = {s for s, _, _ in test} | {o for _, _, o in test}
    overlap = train_ents & test_ents
    if overlap:
        print(f"[WARN] Inductive split violated: {len(overlap)} entities appear in both train and test")
        return False
    return True
