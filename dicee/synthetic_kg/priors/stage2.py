"""The stage-2 worlds: statistical (``st_prior``) and structured (``sw_prior``) worlds and the stage-2 plain worlds.

A stage-2 index ``i`` (from ``STAGE2_OFFSET``) is statistical with probability ``STATISTICAL_SHARE``, else structured
with probability ``STRUCTURED_SHARE``, else plain (``streams.stage2_family``). Every stage-2 world starts from the
legacy template of index ``i``:

- plain: the template itself;
- structured: the template's world and queries, plus an attribute block (subjects in latent clusters with Zipf-popular
  attribute values and homophilous links), self relations and a many-relation background; four base tail queries ask
  for observed attribute facts and two base head queries for self facts; every negative is drawn anew; ids are
  permuted;
- statistical: a world whose facts come only from latent similarity (clustered entity vectors and popularity), in the
  template's task layout.

``query_world`` holds the observation, query and negative sampling that the statistical, record and real worlds share.
"""
import random
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from ..batch import WorldBatch
from ..recipe import NEGATIVES
from ..streams import is_statistical, is_structured, statistical_stream, structured_stream
from .factorial import Triple
from .legacy import Template, template

ATTRIBUTE_QUERIES, SELF_QUERIES = 4, 2
"""Base queries of a structured world that ask for attribute facts and self facts (``sw_prior``)."""


def query_world(layout: Template, facts: Sequence[Triple], domains: Sequence[Iterable[int]], ranges: Sequence[Iterable[int]], n: int,
                nr: int, rng: np.random.Generator, stream: Callable[[str], int], observed: tuple[float, float], *, cap: bool = False,
                fill: bool = False, permute: bool = True) -> tuple[dict[str, WorldBatch], int]:
    """A world's batch in the task layout of ``layout``: a random share of the facts observed, the queries uniform over
    the hidden facts, 256 negatives from the relation's domain (head queries) or range (tail queries) and 256 from all
    entities, all false (``st_prior.statistical``; ``pf_prior2f.build`` with ``cap``; ``rw_prior.real_world`` with
    ``fill`` and without ``permute``).

    Args:
        layout: The legacy template: task names, query counts and head/tail pattern.
        facts: The unique facts of the world, in the generator's order.
        domains: Per relation the permitted heads, in the generator's iteration order.
        ranges: Per relation the permitted tails.
        n: Entity count.
        nr: Relation count.
        rng: The world generator; it draws the fact order and the observed share.
        stream: Seeds of the named query and negative streams of the world.
        observed: The range of the observed share.
        cap: Keep at least as many hidden facts as queries (the record worlds).
        fill: Draw further queries with replacement when the hidden facts are too few (the real worlds).
        permute: Permute entity and relation ids with the query generator.

    Returns:
        The batch and the number of observed facts.
    """
    needed = sum(len(b.groups) for b in layout.batch.values())
    order = rng.permutation(len(facts))
    cut = int(rng.uniform(*observed) * len(facts))
    if cap:
        cut = min(cut, len(facts) - needed)
    observed_facts = [facts[i] for i in order[:cut]]
    hidden = [facts[i] for i in order[cut:]]
    tails_of: dict[tuple[int, int], set[int]] = {}
    heads_of: dict[tuple[int, int], set[int]] = {}
    for h, r, t in facts:
        tails_of.setdefault((h, r), set()).add(t)
        heads_of.setdefault((r, t), set()).add(h)
    prng = random.Random(stream('queries'))
    queries = prng.sample(hidden, min(needed, len(hidden)) if fill else needed)
    while len(queries) < needed:
        queries.append(prng.choice(hidden))
    ent, rel = list(range(n)), list(range(nr))
    if permute:
        prng.shuffle(ent)
        prng.shuffle(rel)

    def mapped(f: Triple) -> Triple:
        return ent[f[0]], rel[f[1]], ent[f[2]]

    all_entities = list(range(n))
    context, truth = frozenset(map(mapped, observed_facts)), frozenset(map(mapped, facts))
    out, position = {}, 0
    for task, b in layout.batch.items():
        rows = []
        nrng = random.Random(stream(f'negatives:{task}'))
        for head in b.head_prediction.tolist():
            h, r, t = queries[position]
            position += 1
            forbidden = heads_of[r, t] if head else tails_of[h, r]
            pool = [e for e in (domains[r] if head else ranges[r]) if e not in forbidden] or [e for e in all_entities if e not in forbidden]
            rest = [e for e in all_entities if e not in forbidden]
            negs = nrng.choices(pool, k=NEGATIVES // 2) + nrng.choices(rest, k=NEGATIVES - NEGATIVES // 2)
            rows.append([mapped(x) for x in [(h, r, t)] + [((e, r, t) if head else (h, r, e)) for e in negs]])
        out[task] = WorldBatch(torch.tensor(rows, dtype=torch.long), b.head_prediction, context, truth, n, nr)
    return out, cut


# ----------------------------------------------------------------------------------------------- statistical worlds

def similarity_world(rng: np.random.Generator) -> tuple[int, int, list[Triple], list[np.ndarray], list[np.ndarray]]:
    """Facts from latent similarity (``st_prior.latent_world``): 400 to 5,000 entities in 4 to 30 clusters with d = 16
    vectors and a popularity bias; per relation, heads of random domain clusters pick 1 + Poisson(lam) tails of random
    range clusters by softmax((<e_h, w_r, e_t> / 4 + beta * b_t) / tau).

    Returns:
        Entities, relations, the sorted facts, and per relation the domain and range entities.
    """
    n = int(np.exp(rng.uniform(np.log(400), np.log(5000))))
    relations = int(np.exp(rng.uniform(np.log(4), np.log(80))))
    facts_target = int(np.exp(rng.uniform(np.log(3000), np.log(50000))))
    clusters = int(rng.integers(4, 31))
    tau = float(np.exp(rng.uniform(np.log(0.15), np.log(1.0))))
    noise, beta, lam = rng.uniform(0.3, 1.0), rng.uniform(0.0, 2.5), rng.uniform(0.3, 2.0)
    d = 16
    centers = rng.standard_normal((clusters, d)) * 2.0
    member = rng.integers(0, clusters, n)
    emb = centers[member] + noise * rng.standard_normal((n, d))
    bias = rng.standard_normal(n)
    triples, domains, ranges = set(), [], []
    per_relation = max(20, facts_target // relations)
    for r in range(relations):
        w = rng.standard_normal(d)
        dom, ran = rng.random(clusters) < 0.5, rng.random(clusters) < 0.5
        dom[rng.integers(clusters)] = True
        ran[rng.integers(clusters)] = True
        heads_all, tails = np.flatnonzero(dom[member]), np.flatnonzero(ran[member])
        heads = rng.choice(heads_all, size=min(len(heads_all), max(1, int(per_relation / (1 + lam)))), replace=False)
        logits = ((emb[heads] * w) @ emb[tails].T / 4.0 + beta * bias[tails][None, :]) / tau
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        for i, h in enumerate(heads):
            k = min(1 + rng.poisson(lam), len(tails))
            for t in rng.choice(tails, size=k, replace=False, p=probs[i]):
                if t != h:
                    triples.add((int(h), r, int(t)))
        domains.append(heads_all)
        ranges.append(tails)
    return n, relations, sorted(triples), domains, ranges


def statistical(seed: int, index: int) -> tuple[dict[str, WorldBatch], dict[str, Any]]:
    """A statistical world (``st_prior.statistical``): a similarity world, 70 to 90 % observed, in the layout of the
    index's legacy template.

    Args:
        seed: The run seed.
        index: The stage-2 index.

    Returns:
        The batch and its metadata.
    """
    layout = template(seed, index)
    rng = np.random.default_rng(statistical_stream(seed, index, 'world'))
    n, nr, facts, domains, ranges = similarity_world(rng)
    batch, observed = query_world(layout, facts, [d.tolist() for d in domains], [r.tolist() for r in ranges], n, nr, rng,
                                  lambda name: statistical_stream(seed, index, name), (0.7, 0.9))
    return batch, dict(family='statistical', program_slot=layout.program_slot, entities=n, relations_total=nr, context=observed,
                       facts=len(facts), recipe_batch=layout.recipe_batch)


# ----------------------------------------------------------------------------------------------- structured worlds

def attribute_block(n0: int, original: np.ndarray, rng: np.random.Generator) -> tuple[int, list, list, dict, np.ndarray]:
    """Subjects in latent clusters with 2 to 6 attribute relations and 1 to 3 homophilous link relations; up to half of
    the original entities hold attributes too (``sw_prior.attribute_block``).

    Returns:
        The number of new entities, the relation kinds, the facts ``(head, kind, tail)``, the value entities of each
        attribute kind and the attribute holders.
    """
    ent = n0
    n_subj = int(np.exp(rng.uniform(np.log(150), np.log(2500))))
    subjects = np.arange(ent, ent + n_subj)
    ent += n_subj
    attach = rng.random(len(original)) < rng.uniform(0.0, 0.5)
    holders = np.concatenate([subjects, original[attach]])
    n_clusters = int(rng.integers(3, 16))
    cluster = rng.choice(n_clusters, len(holders), p=rng.dirichlet(np.full(n_clusters, 1.5)))
    facts: list[tuple[int, tuple[str, int], int]] = []
    kinds: list[tuple[str, int]] = []
    value_sets: dict[tuple[str, int], np.ndarray] = {}
    for a in range(int(rng.integers(2, 7))):
        m = int(np.exp(rng.uniform(np.log(20), np.log(800))))
        values = np.arange(ent, ent + m)
        ent += m
        base = (rng.permutation(m) + 1.0) ** -rng.uniform(0.6, 1.4)
        logits = np.log(base)[None, :] + rng.uniform(0.5, 2.5) * rng.standard_normal((n_clusters, m))
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        functional = rng.random() < 0.7
        coverage = rng.uniform(0.6, 1.0)
        rel = ('attribute', a)
        for holder, c in zip(holders, cluster):
            if rng.random() > coverage:
                continue
            k = 1 if functional else int(rng.integers(1, 4))
            for v in rng.choice(values, size=min(k, m), replace=False, p=probs[c]):
                facts.append((int(holder), rel, int(v)))
        kinds.append(rel)
        value_sets[rel] = values
    by_cluster = defaultdict(list)
    for holder, c in zip(holders, cluster):
        by_cluster[c].append(int(holder))
    for link in range(int(rng.integers(1, 4))):
        rel = ('link', link)
        homophily = rng.uniform(0.6, 0.95)
        for _ in range(int(len(holders) * rng.uniform(0.5, 3.0))):
            i = int(rng.integers(len(holders)))
            x, c = int(holders[i]), cluster[i]
            pool = by_cluster[c] if rng.random() < homophily else holders
            y = int(pool[int(rng.integers(len(pool)))])
            if x != y:
                facts.append((x, rel, y))
        kinds.append(rel)
    return ent - n0, kinds, facts, value_sets, holders


def self_relations(pools: list[np.ndarray], rng: np.random.Generator) -> tuple[list, list]:
    """1 or 2 relations where many entities of one pool relate to themselves, plus a few other edges (``sw_prior.self_relations``)."""
    facts, kinds = [], []
    for s in range(int(rng.integers(1, 3))):
        pool = pools[int(rng.integers(len(pools)))]
        rel = ('self', s)
        p_self, p_other = rng.uniform(0.3, 0.95), rng.uniform(0.0, 0.3)
        for x in pool:
            if rng.random() < p_self:
                facts.append((int(x), rel, int(x)))
            if rng.random() < p_other:
                y = int(pool[int(rng.integers(len(pool)))])
                if y != x:
                    facts.append((int(x), rel, y))
        kinds.append(rel)
    return kinds, facts


def background(n0: int, original: np.ndarray, context_size: int, rng: np.random.Generator) -> tuple[int, list, list]:
    """A many-relation background of LogUniform(1, 6) times the context, on the original and new entities (``sw_prior.background``)."""
    factor = float(np.exp(rng.uniform(0.0, np.log(6.0))))
    budget = int((factor - 1.0) * context_size / 0.7)
    if budget < 100:
        return 0, [], []
    new = int(max(32, len(original) * np.exp(rng.uniform(np.log(0.5), np.log(4.0)))))
    pool = np.concatenate([original, np.arange(n0, n0 + new)])
    kinds: list[tuple[str, int]] = []
    facts: list[tuple[int, tuple[str, int], int]] = []
    nrel = int(np.exp(rng.uniform(np.log(4), np.log(64))))
    shares = rng.dirichlet(np.full(nrel, 0.7))
    for k in range(nrel):
        rel = ('background', k)
        dom = pool if rng.random() < 0.5 else pool[len(original):]
        ran = pool if rng.random() < 0.5 else pool[len(original):]
        hp = (rng.permutation(len(dom)) + 1.0) ** -rng.uniform(0.0, 1.0)
        tp = (rng.permutation(len(ran)) + 1.0) ** -rng.uniform(0.2, 1.5)
        count = max(20, int(budget * shares[k]))
        hs = rng.choice(dom, count, p=hp / hp.sum())
        ts = rng.choice(ran, count, p=tp / tp.sum())
        facts.extend((int(h), rel, int(t)) for h, t in zip(hs, ts) if h != t)
        kinds.append(rel)
    return new, kinds, facts


def structured(seed: int, index: int) -> tuple[dict[str, WorldBatch], dict[str, Any]]:
    """A structured world on the index's legacy template, or the template itself where the index's hash says plain
    (``sw_prior.structured``).

    Args:
        seed: The run seed.
        index: The stage-2 index.

    Returns:
        The batch and its metadata.
    """
    layout = template(seed, index)
    batch = layout.batch
    if not is_structured(seed, index):
        return batch, dict(family='plain', program_slot=layout.program_slot, recipe_batch=layout.recipe_batch)
    first = next(iter(batch.values()))
    n, nr, truth = first.num_entities, first.num_relations, first.truth
    rng = np.random.default_rng(structured_stream(seed, index, 'world'))
    original = np.arange(n)
    context_size = max(len(b.context) for b in batch.values())
    n_attr, attr_kinds, attr_facts, value_sets, holders = attribute_block(n, original, rng)
    n_bg, bg_kinds, bg_facts = background(n + n_attr, original, context_size, rng)
    self_kinds, self_facts = self_relations([original, holders, np.arange(n + n_attr, n + n_attr + n_bg) if n_bg else holders], rng)
    kinds = attr_kinds + self_kinds + bg_kinds
    rel_id = {k: nr + i for i, k in enumerate(kinds)}
    n_all, nr_all = n + n_attr + n_bg, nr + len(kinds)
    new_facts = sorted({(h, rel_id[k], t) for h, k, t in attr_facts + self_facts + bg_facts})
    observed = [f for f, keep in zip(new_facts, rng.random(len(new_facts)) < rng.uniform(0.5, 0.85)) if keep]
    full_truth = frozenset(truth) | frozenset(new_facts)
    tails, heads = defaultdict(set), defaultdict(set)
    for h, r, t in full_truth:
        tails[h, r].add(t)
        heads[r, t].add(h)
    prng = random.Random(structured_stream(seed, index, 'queries'))
    attr_ids = {rel_id[k]: value_sets[k] for k in attr_kinds if k in value_sets}
    self_ids = {rel_id[k] for k in self_kinds}
    attr_pool = [f for f in observed if f[1] in attr_ids]
    self_pool = [f for f in observed if f[1] in self_ids and f[0] == f[2]]
    heads_flag = batch['base'].head_prediction.tolist()
    tail_rows = [j for j, hd in enumerate(heads_flag) if not hd]
    head_rows = [j for j, hd in enumerate(heads_flag) if hd]
    plan = {}
    for j, q in zip(prng.sample(tail_rows, ATTRIBUTE_QUERIES), prng.sample(attr_pool, ATTRIBUTE_QUERIES)):
        plan[j] = ('attribute', q)
    if len(self_pool) >= SELF_QUERIES:
        for j, q in zip(prng.sample(head_rows, SELF_QUERIES), prng.sample(self_pool, SELF_QUERIES)):
            plan[j] = ('self', q)
    targets = {q for _, q in plan.values()}
    all_entities = list(range(n_all))
    ent = list(range(n_all))
    prng.shuffle(ent)
    rel = list(range(nr_all))
    prng.shuffle(rel)
    holder_set = set(holders.tolist())

    def mapped(f: Triple) -> Triple:
        return ent[f[0]], rel[f[1]], ent[f[2]]

    def sample(pool: Iterable[int], forbidden: set[int], k: int, r: random.Random) -> list[int]:
        cands = [e for e in pool if e not in forbidden] or [e for e in all_entities if e not in forbidden]
        return r.choices(cands, k=k)

    out = {}
    for task, b in batch.items():
        rows = []
        nrng = random.Random(structured_stream(seed, index, f'negatives:{task}'))
        for j, (q, hd) in enumerate(zip(map(tuple, b.groups[:, 0].tolist()), b.head_prediction.tolist(), strict=True)):
            kind = 'original'
            if task == 'base' and j in plan:
                kind, q = plan[j]
            h, r, t = q
            forbidden = heads[r, t] if hd else tails[h, r]
            if kind == 'original':
                local: Iterable[int] = range(n)
            elif kind == 'attribute':
                local = attr_ids[r].tolist()
            else:
                local = [int(x) for x in (holders if h in holder_set else original)]
            negs = sample(local, forbidden, NEGATIVES // 2, nrng) + sample(all_entities, forbidden, NEGATIVES - NEGATIVES // 2, nrng)
            rows.append([mapped(x) for x in [q] + [((e, r, t) if hd else (h, r, e)) for e in negs]])
        context = (set(b.context) | set(observed)) - targets
        out[task] = WorldBatch(torch.tensor(rows, dtype=torch.long), b.head_prediction, frozenset(map(mapped, context)),
                               frozenset(map(mapped, full_truth)), n_all, nr_all)
    meta = dict(family='structured', program_slot=layout.program_slot, entities=n_all, relations_total=nr_all,
                context=max(len(b.context) for b in out.values()), original_context=context_size, attribute_relations=len(attr_ids),
                self_relations=len(self_kinds), background_relations=len(bg_kinds),
                attribute_queries=sum(k == 'attribute' for k, _ in plan.values()), self_queries=sum(k == 'self' for k, _ in plan.values()),
                recipe_batch=layout.recipe_batch)
    return out, meta


def stage2_world(seed: int, index: int) -> tuple[dict[str, WorldBatch], dict[str, Any]]:
    """The world of stage-2 index ``index`` (``st_prior.structured``): statistical, structured or plain by its hashes."""
    if is_statistical(seed, index):
        return statistical(seed, index)
    return structured(seed, index)
