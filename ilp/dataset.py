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

from .vocab import HOP_NONE, hop_distance_token, inverse_relation

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
        # Memo for k_hop_neighborhood: the graph is immutable after __init__, so
        # a node's k-hop subgraph is invariant across sample draws. Caching it
        # turns the per-draw BFS (the data-pipeline bottleneck) into a one-time
        # cost per (node, k). Anonymization stays per-draw downstream, so this
        # changes nothing observable about the samples.
        self._khop_cache: dict[tuple[str, int], tuple[set[Triple], dict[str, int]]] = {}

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


def k_hop_neighborhood(
    node: str, kg: KnowledgeGraph, k: int = 2,
) -> tuple[set[Triple], dict[str, int]]:
    """BFS-extract the k-hop subgraph and per-entity hop distances.

    Returns (triples, distance) where:
    - `triples` is the set of all triples involving any entity at distance ≤ k-1
      (i.e., extending the frontier k times from `node`). Matches the original
      `two_hop_neighborhood` semantics for k=2.
    - `distance[e]` is the shortest-path hop count from `node` to `e` along
      relation-agnostic edges. `node` itself has distance 0.

    Generalizes to arbitrary k so subgraph depth can be tuned without code
    changes elsewhere.

    The result is memoized on `kg` (see `KnowledgeGraph._khop_cache`) and is
    therefore shared and **read-only** — callers must not mutate the returned
    set/dict (copy first if they need to).
    """
    cached = kg._khop_cache.get((node, k))
    if cached is not None:
        return cached

    distance: dict[str, int] = {node: 0}
    frontier: set[str] = {node}
    triples: set[Triple] = set()
    for d in range(1, k + 1):
        next_frontier: set[str] = set()
        for ent in frontier:
            for tr in kg.adj.get(ent, set()):
                triples.add(tr)
                s, _, o = tr
                for x in (s, o):
                    if x not in distance:
                        distance[x] = d
                        next_frontier.add(x)
        frontier = next_frontier
    kg._khop_cache[(node, k)] = (triples, distance)
    return triples, distance


def two_hop_neighborhood(node: str, kg: KnowledgeGraph) -> set[Triple]:
    """All triples within two hops of `node`. Back-compat wrapper."""
    triples, _ = k_hop_neighborhood(node, kg, k=2)
    return triples


def _build_subgraph_rows(
    center: str,
    kg: KnowledgeGraph,
    vocab: dict[str, int],
    fixed_values: set[str],
    max_triples: int,
    z_pool: int,
    rng: random.Random,
    exclude_triple: Triple | None,
    collapse_z: bool,
    subgraph_hops: int,
    use_hop_distance_tokens: bool,
    center_mode: str = "xtoken",
    inherit_map: dict[str, int] | None = None,
    inherit_z: list[int] | None = None,
):
    """Extract, anonymize and tokenize the k-hop subgraph around `center`.

    Returns `(triples_tok, hop_tok, tok_entity, entity_map, z_remaining, none_id)`:
    - `tok_entity` is the (stateful) closure that mapped this subgraph's entities
      to token ids — the caller reuses it so the candidate token is consistent
      with the anonymization (single-tower behaviour).
    - `entity_map` is the same dict the closure mutates (entity → token id), and
      `z_remaining` the still-unused `[Z_i]` indices. Scoring callers read these
      to assign candidate tokens without re-running the closure (see
      `eval._tokenize_center`). This is the single tokenizer shared by the
      training path (`build_sample`) and the eval/score path.

    `center_mode` controls how the subgraph root is tokenized:
    - `xtoken`  (default): the center gets the special `[X]` token (which is also
      the SAB's prepended CLS), so it has a role marker but no shareable identity.
    - `cls_role`: the center gets an ordinary `[Z_i]` like any other entity, so it
      has a *shareable* identity that binds across the dual towers. `[X]` is then
      used only as the CLS pool token. Rooting is carried by the `HOP_DIST_0`
      hop-distance token, so this mode requires `use_hop_distance_tokens=True`.

    `inherit_map`/`inherit_z` enable shared anonymization across the two dual
    towers: pass the anchor subgraph's `entity_map`/`z_remaining` so an entity
    common to both subgraphs reuses the *same* `[Z_i]`, restoring cross-tower
    variable binding. The shared labels are still randomized per draw — nothing
    becomes a persistent per-entity embedding. Under `xtoken` the new `center` is
    re-keyed to `[X]` and the other tower's center drops back to a fresh `[Z_i]`;
    under `cls_role` the inherited map is reused as-is, so *both* centers bind too.
    """
    subgraph, entity_distance = k_hop_neighborhood(center, kg, k=subgraph_hops)
    # `subgraph` is the cached, read-only neighborhood — filter the excluded
    # target triple into a fresh list rather than mutating the cache.
    if exclude_triple is not None:
        s, r, o = exclude_triple
        excluded = {(s, r, o), (o, inverse_relation(r), s)}
        pool = [t for t in subgraph if t not in excluded]
    else:
        pool = list(subgraph)

    if len(pool) > max_triples:
        subgraph_list = rng.sample(pool, max_triples)
    else:
        subgraph_list = pool

    x_id = vocab["[X]"]
    if inherit_z is None:
        z_indices = list(range(z_pool))
        rng.shuffle(z_indices)
    else:
        z_indices = list(inherit_z)
    if center_mode == "cls_role":
        # Center is an ordinary (shareable) [Z]; [X] is reserved for the CLS pool.
        # Reuse the inherited map wholesale so both centers bind across towers;
        # the center's own [Z] is assigned lazily by tok_entity (inherited if shared).
        entity_map: dict[str, int] = dict(inherit_map) if inherit_map is not None else {}
    elif inherit_map is None:
        entity_map = {center: x_id}
    else:
        entity_map = {e: tid for e, tid in inherit_map.items()
                      if tid != x_id and e != center}
        entity_map[center] = x_id

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

    none_id = vocab[HOP_NONE]

    def hop_for(node: str) -> int:
        if not use_hop_distance_tokens:
            return none_id
        d = entity_distance.get(node, -1)
        return vocab[hop_distance_token(d)]

    rows = [
        ([tok_entity(s), vocab[f"[REL_{rel}]"], tok_entity(o)],
         [hop_for(s), none_id, hop_for(o)])
        for s, rel, o in subgraph_list
    ]
    # Shuffle triples *and* their parallel hop-distance rows together so the
    # alignment isn't broken — without this, hop tokens would refer to
    # different entities than the ones they sit beside.
    rng.shuffle(rows)
    triples_tok = [t for t, _ in rows]
    hop_tok = [r for _, r in rows]
    return triples_tok, hop_tok, tok_entity, entity_map, z_indices, none_id


def _pad_subgraph(triples_tok, hop_tok, max_triples, none_id):
    """Pad a tokenized subgraph to `max_triples` → (triples, hop_distances, mask)."""
    n = len(triples_tok)
    pad_t = [[0, 0, 0]] * (max_triples - n)
    pad_r = [[none_id, none_id, none_id]] * (max_triples - n)
    return (
        torch.tensor(triples_tok + pad_t, dtype=torch.long),
        torch.tensor(hop_tok + pad_r, dtype=torch.long),
        torch.tensor([True] * n + [False] * (max_triples - n), dtype=torch.bool),
    )


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
    subgraph_hops: int = 2,
    use_hop_distance_tokens: bool = False,
    dual_subgraph: bool = False,
    shared_anonymization: bool = False,
    center_mode: str = "xtoken",
) -> dict[str, torch.Tensor]:
    """Build one anonymized sample anchored on `anchor`.

    `anchor` is the subgraph center (`[X]` under `center_mode='xtoken'`, else a
    shareable `[Z_i]`). `candidate` gets either its [VAL_*] (schema) or a [Z_i]
    consistent with the subgraph's anonymization.

    `exclude_triple`: drop this triple from the subgraph if present. For
    positive training samples pass (anchor, relation, candidate) so the
    target is never leaked into context. For negatives, leave it None.

    `collapse_z`: when True, if the Z pool is exhausted, distinct entities
    are mapped to a uniformly-random already-existing [Z_i] instead of
    raising. Used for the z_pool=1 ablation that asks "does variable
    binding across triples matter?".

    `dual_subgraph`: also emit `cand_triples`/`cand_hop_distances`/`cand_mask`
    for the candidate's *own* k-hop subgraph (centered on `candidate` with its
    own independent anonymization, same `exclude_triple` so the target edge is
    never leaked). The model's candidate tower consumes these to ground
    candidates that fall outside the anchor's neighborhood.

    `shared_anonymization` (dual only): label the candidate subgraph from the
    anchor subgraph's `[Z_i]` assignment so entities shared by both towers map
    to the same slot — restoring cross-tower variable binding. Still randomized
    per draw; no persistent per-entity embedding.
    """
    rng = rng or random
    triples_tok, hop_tok, tok_entity, _entity_map, _z_remaining, none_id = _build_subgraph_rows(
        anchor, kg, vocab, fixed_values, max_triples, z_pool, rng,
        exclude_triple, collapse_z, subgraph_hops, use_hop_distance_tokens,
        center_mode=center_mode,
    )
    # Candidate token consistent with the anchor subgraph's anonymization
    # (single-tower representation; unused but harmless under dual_subgraph).
    candidate_tok = tok_entity(candidate)

    triples_t, hop_t, mask_t = _pad_subgraph(triples_tok, hop_tok, max_triples, none_id)
    out = {
        "triples": triples_t,
        "hop_distances": hop_t,
        "mask": mask_t,
        "target_relation": torch.tensor(vocab[f"[REL_{relation}]"], dtype=torch.long),
        "target_tail": torch.tensor(candidate_tok, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.float),
    }

    if dual_subgraph:
        c_triples_tok, c_hop_tok, *_ = _build_subgraph_rows(
            candidate, kg, vocab, fixed_values, max_triples, z_pool, rng,
            exclude_triple, collapse_z, subgraph_hops, use_hop_distance_tokens,
            center_mode=center_mode,
            inherit_map=_entity_map if shared_anonymization else None,
            inherit_z=_z_remaining if shared_anonymization else None,
        )
        c_triples_t, c_hop_t, c_mask_t = _pad_subgraph(
            c_triples_tok, c_hop_tok, max_triples, none_id
        )
        out["cand_triples"] = c_triples_t
        out["cand_hop_distances"] = c_hop_t
        out["cand_mask"] = c_mask_t

    return out


class NegativeSampler:
    """Picks a candidate `c` such that `(anchor, relation, c)` is a negative.

    Strategy lives here so `InductiveKGDataset` doesn't have to know about
    type-priors, neighborhoods, or mixtures. Implementations precompute
    their indexes at __init__ and must be pickleable for DataLoader workers.

    The protocol returns an entity even on failure (after MAX_TRIES); the
    caller treats the sample as a negative regardless. In pathological KGs
    where every candidate forms a known triple, this can leak a positive as
    label 0 — same behavior as the original inline loop.
    """

    MAX_TRIES = 100

    def __init__(self, kg: "KnowledgeGraph", entity_pool: Sequence[str]):
        self.kg = kg
        self.entity_pool = list(entity_pool)

    def __call__(
        self, anchor: str, relation: str, true_tail: str, rng: random.Random
    ) -> str:
        raise NotImplementedError


class UniformNegativeSampler(NegativeSampler):
    """Default: sample uniformly from `entity_pool`, reject known triples."""

    def __call__(self, anchor, relation, true_tail, rng):
        candidate = true_tail
        for _ in range(self.MAX_TRIES):
            corrupt = rng.choice(self.entity_pool)
            if (anchor, relation, corrupt) not in self.kg.triple_set:
                candidate = corrupt
                break
        return candidate


def _build_true_tails(kg: "KnowledgeGraph") -> dict[tuple[str, str], set[str]]:
    """Index (head, rel) → set of all true tails. Used to filter hard negatives."""
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for h, r, t in kg.triples:
        out[(h, r)].add(t)
    return out


class RelationTailPriorSampler(NegativeSampler):
    """Hard: sample from entities that ever appear as tail of `relation`.

    Removes the easy "wrong type" mass — corruptions are at least
    relation-feasible. Falls back to uniform if `relation` is unseen
    (e.g., inverse relation not yet indexed) or the pool is empty after
    filtering.
    """

    def __init__(self, kg, entity_pool):
        super().__init__(kg, entity_pool)
        self.tails_by_relation: dict[str, list[str]] = defaultdict(list)
        seen: dict[str, set[str]] = defaultdict(set)
        for _, r, t in kg.triples:
            if t not in seen[r]:
                seen[r].add(t)
                self.tails_by_relation[r].append(t)
        self.true_tails = _build_true_tails(kg)
        self._fallback = UniformNegativeSampler(kg, entity_pool)

    def __call__(self, anchor, relation, true_tail, rng):
        pool = self.tails_by_relation.get(relation)
        if not pool:
            return self._fallback(anchor, relation, true_tail, rng)
        filt = self.true_tails.get((anchor, relation), frozenset())
        for _ in range(self.MAX_TRIES):
            corrupt = rng.choice(pool)
            if corrupt not in filt:
                return corrupt
        return self._fallback(anchor, relation, true_tail, rng)


class TwoHopNeighborhoodSampler(NegativeSampler):
    """Hard: sample from entities reachable within 2 hops of `anchor`.

    These candidates already appear in the subgraph context, so the model
    can't reject them by absence — it has to use relational structure.
    Falls back to uniform when the neighborhood (minus filter) is empty.
    """

    def __init__(self, kg, entity_pool):
        super().__init__(kg, entity_pool)
        self.true_tails = _build_true_tails(kg)
        self._fallback = UniformNegativeSampler(kg, entity_pool)

    def __call__(self, anchor, relation, true_tail, rng):
        nb = two_hop_neighborhood(anchor, self.kg)
        ents: set[str] = set()
        for s, _, o in nb:
            ents.add(s)
            ents.add(o)
        ents.discard(anchor)
        ents -= self.true_tails.get((anchor, relation), frozenset())
        if not ents:
            return self._fallback(anchor, relation, true_tail, rng)
        return rng.choice(list(ents))


class MixtureSampler(NegativeSampler):
    """Weighted mixture over child samplers. Weights are normalized."""

    def __init__(self, components: Sequence[tuple[float, NegativeSampler]]):
        if not components:
            raise ValueError("MixtureSampler requires at least one component")
        total = sum(w for w, _ in components)
        if total <= 0:
            raise ValueError("MixtureSampler weights must sum to > 0")
        self.weights = [w / total for w, _ in components]
        self.samplers = [s for _, s in components]
        # Don't call super().__init__ — child samplers own their KG refs.

    def __call__(self, anchor, relation, true_tail, rng):
        s = rng.choices(self.samplers, weights=self.weights, k=1)[0]
        return s(anchor, relation, true_tail, rng)


SAMPLER_REGISTRY: dict[str, type[NegativeSampler]] = {
    "uniform": UniformNegativeSampler,
    "relation_tail_prior": RelationTailPriorSampler,
    "two_hop": TwoHopNeighborhoodSampler,
}


def build_negative_sampler(
    spec: str | dict | None,
    kg: "KnowledgeGraph",
    entity_pool: Sequence[str],
) -> NegativeSampler:
    """Construct a sampler from a config spec.

    - None / "uniform"     → UniformNegativeSampler
    - "relation_tail_prior" / "two_hop" → that single sampler
    - {"mixture": {"uniform": 0.5, "two_hop": 0.5, ...}} → MixtureSampler
    """
    if spec is None or spec == "uniform":
        return UniformNegativeSampler(kg, entity_pool)
    if isinstance(spec, str):
        if spec not in SAMPLER_REGISTRY:
            raise ValueError(
                f"Unknown sampler {spec!r}. Known: {sorted(SAMPLER_REGISTRY)} or 'mixture'."
            )
        return SAMPLER_REGISTRY[spec](kg, entity_pool)
    if isinstance(spec, dict) and "mixture" in spec:
        components: list[tuple[float, NegativeSampler]] = []
        for name, weight in spec["mixture"].items():
            if name not in SAMPLER_REGISTRY:
                raise ValueError(
                    f"Unknown sampler {name!r} in mixture. Known: {sorted(SAMPLER_REGISTRY)}."
                )
            components.append((float(weight), SAMPLER_REGISTRY[name](kg, entity_pool)))
        return MixtureSampler(components)
    raise ValueError(f"Unrecognized neg_sampler spec: {spec!r}")


class InductiveKGDataset(Dataset):
    """Training dataset with on-the-fly negative sampling.

    Each positive triple yields (1 + neg_per_pos) samples per __getitem__
    call slot. We also randomly flip anchor direction so the model learns
    both `(h, r, ?)` and `(?, r, t)` queries (spec §7.1 evaluates both).

    `neg_sampler` controls negative generation. Defaults to
    `UniformNegativeSampler` over `entity_pool` (current behavior).
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
        neg_sampler: NegativeSampler | None = None,
        subgraph_hops: int = 2,
        use_hop_distance_tokens: bool = False,
        dual_subgraph: bool = False,
        shared_anonymization: bool = False,
        center_mode: str = "xtoken",
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
        self.neg_sampler = neg_sampler or UniformNegativeSampler(kg, self.entity_pool)
        self.subgraph_hops = subgraph_hops
        self.use_hop_distance_tokens = use_hop_distance_tokens
        self.dual_subgraph = dual_subgraph
        self.shared_anonymization = shared_anonymization
        self.center_mode = center_mode

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
                subgraph_hops=self.subgraph_hops,
                use_hop_distance_tokens=self.use_hop_distance_tokens,
                dual_subgraph=self.dual_subgraph,
                shared_anonymization=self.shared_anonymization,
                center_mode=self.center_mode,
            )

        candidate = self.neg_sampler(anchor, r_use, candidate, rng)
        return build_sample(
            anchor, r_use, candidate, 0.0, self.kg, self.vocab, self.fixed_values,
            self.max_triples, self.z_pool, rng, exclude_triple=None,
            collapse_z=self.collapse_z,
            subgraph_hops=self.subgraph_hops,
            use_hop_distance_tokens=self.use_hop_distance_tokens,
            dual_subgraph=self.dual_subgraph,
            shared_anonymization=self.shared_anonymization,
            center_mode=self.center_mode,
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
