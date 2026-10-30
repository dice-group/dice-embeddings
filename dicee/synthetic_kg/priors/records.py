"""The record families: worlds of subjects and their attribute values, read as facts (``prior-fix-v1-runtime``).

- categorical2 (``pf_prior2f``): subjects of 2 to 8 types in latent groups; attribute relations subject -> value carried
  by some groups of a type, with reused Zipf-popular values; a subject takes its group's value with a reliability, or
  values of its group's set for multi-valued attributes; identity attributes, twin relations, homophilous links.
- attr2 (``pf_attr2``): categorical2's world with one world-level strength for every fidelity (null worlds have none)
  and an optional value chain, drawn by a second generator so that every other draw is categorical2's.
- rec1 (``pf_rec1``): relational tables of rows in latent taxonomies; key and property columns read the taxonomy at
  one level; value chains, references between tables and inherited columns; null, weak and strong worlds.
- list1 (``pf_list1``): anchors with lists of items from latent communities, hub items, item and anchor attributes and
  item links; null, weak and strong worlds.

Each world takes the task layout of the legacy template of its stream index and samples its queries and negatives
with ``stage2.query_world`` (observed share from the family's parameters, at least 30 facts hidden). With
``FamilySpec.null_swap_budget`` the structural null (``null.swap_truth``) replaces the truth before the observation.
"""
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from ..batch import WorldBatch
from ..recipe import record_params
from ..streams import ATTR2_VERSION, CATEGORICAL2_VERSION, LIST1_VERSION, NULL_VERSION, REC1_VERSION, world_stream
from .legacy import template
from .null import swap_truth
from .stage2 import query_world

if TYPE_CHECKING:
    from . import FamilySpec

VERSIONS = {'categorical2': CATEGORICAL2_VERSION, 'attr2': ATTR2_VERSION, 'rec1': REC1_VERSION, 'list1': LIST1_VERSION}
"""The stream version of each record family."""

LatentWorld = tuple[int, int, np.ndarray, list[np.ndarray], list[np.ndarray], dict[str, Any]]
"""Entities, relations, the sorted unique facts ``[F, 3]``, per relation the domain and range entities, and the
generator's metadata."""


def _lu(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A log-uniform draw (``hi`` raised to just above ``lo`` when it is not above it)."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(max(hi, lo * 1.0001)))))


def _zipf(rng: np.random.Generator, m: int, lo: float, hi: float) -> np.ndarray:
    """A Zipf popularity over ``m`` values in random order, exponent U(lo, hi)."""
    pop = (rng.permutation(m) + 1.0) ** -rng.uniform(lo, hi)
    normalized: np.ndarray = pop / pop.sum()
    return normalized


def _compact(facts: np.ndarray, domains: list[np.ndarray], ranges: list[np.ndarray], next_id: int) -> tuple[int, int, np.ndarray, list, list]:
    """Renumber the entities that occur in a fact densely, keeping their order; domains and ranges follow."""
    used = np.unique(facts[:, [0, 2]])
    dense = np.full(next_id, -1, dtype=np.int64)
    dense[used] = np.arange(len(used))
    facts[:, 0], facts[:, 2] = dense[facts[:, 0]], dense[facts[:, 2]]
    return len(used), len(domains), facts, [dense[d][dense[d] >= 0] for d in domains], [dense[r][dense[r] >= 0] for r in ranges]


def categorical_world(rng: np.random.Generator, P: dict[str, Any], frng: Optional[np.random.Generator] = None) -> LatentWorld:
    """The latent world of categorical2 (``pf_prior2f.latent_world``), or with ``frng`` of attr2 (``pf_attr2.latent_world``).

    Args:
        rng: The world generator.
        P: The family's parameters.
        frng: attr2's fidelity generator: the world strength, each attribute's fidelity, the null values and the value
            chain. ``rng`` then makes exactly categorical2's draws.

    Returns:
        The latent world; attr2's metadata holds ``strength``, ``null`` and ``chain``.
    """
    null, s = False, 0.0
    if frng is not None:
        null = frng.random() < P['null']
        s = 0.0 if null else float(frng.uniform(*P['strength']))
    ns = int(_lu(rng, *P['subjects']))
    lowest = P['groups'][0] * ns if P['groups'][0] < 1 else P['groups'][0]
    groups = max(2, int(_lu(rng, max(lowest, 2.0), P['groups'][1] * ns)))
    gw = (np.arange(groups) + 1.0) ** -rng.uniform(*P['group_zipf'])
    group = rng.choice(groups, ns, p=gw / gw.sum())
    types = int(rng.integers(P['types'][0], P['types'][1] + 1))
    group_type = rng.integers(0, types, groups)   # a group lies inside one type
    subject_type = group_type[group]
    members = [np.flatnonzero(group == g) for g in range(groups)]
    attributes = max(1, int(round(_lu(rng, *P['attributes']))))
    subjects = np.arange(ns)
    budget, next_id = P['max_entities'] - ns, ns
    parts: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    chainable = []   # attr2: (values, group value, carrying groups) of the single-valued attributes
    if types > 1 and rng.random() < P['type_relation']:
        nodes = np.arange(next_id, next_id + types)
        next_id, budget = next_id + types, budget - types
        subj = subjects[rng.random(ns) < rng.uniform(*P['type_coverage'])]
        parts.append(np.stack([subj, np.zeros(len(subj), dtype=np.int64), nodes[subject_type[subj]]], 1))
        domains.append(subjects)
        ranges.append(nodes)
    for a in range(attributes):
        # every draw below is made for every attribute, so that a parameter changes only what it names
        u_typed, pick_type, u_group, rate = rng.random(), int(rng.integers(types)), rng.random(), rng.uniform(*P['group_rate'])
        group_coin, u_full, full, part = rng.random(groups), rng.random(), rng.uniform(0.95, 1.0), rng.uniform(*P['coverage'])
        domain = subjects
        if types > 1 and u_typed < P['typed']:
            chosen = subjects[subject_type == pick_type]
            if len(chosen) >= 10:
                domain = chosen
        if u_group < P['group_domain']:
            carries = group_coin < rate
            carries[int(np.argmin(group_coin))] = True
            chosen = domain[carries[group[domain]]]
            if len(chosen) >= 10:
                domain = chosen
        cov = full if u_full < P['full_coverage'] else part
        identity = rng.random() < P['identity']
        m = groups if identity else int(_lu(rng, P['vocabulary'][0], min(P['vocabulary'][1], max(len(domain), P['vocabulary'][0] + 1))))
        m = int(max(2, min(m, budget - 2 * (attributes - a - 1))))
        values = np.arange(next_id, next_id + m)
        next_id, budget = next_id + m, budget - m
        pop = (rng.permutation(m) + 1.0) ** -rng.uniform(*P['value_zipf'])
        pop /= pop.sum()
        group_value = (np.arange(groups) % m) if identity else rng.choice(m, groups, p=pop)
        rho = rng.uniform(*P['identity_reliability']) if identity else rng.uniform(*P['reliability'])
        functional = rng.random() < P['functional']
        lam, set_rho = rng.uniform(*P['extra']), rng.uniform(*P['set_reliability'])
        subj = domain[rng.random(len(domain)) < cov]
        if frng is not None:   # attr2: the fidelity comes from the world's strength
            rho = 0.0 if null else float(np.clip(frng.uniform(s - P['spread'], s + P['spread']), 0.0, 1.0))
        if functional:
            follow = rng.random(len(subj)) < rho
            val = np.where(follow, group_value[group[subj]], rng.choice(m, len(subj), p=pop))
            heads = subj
            if frng is not None:
                carrying = np.unique(group[subj])
                if len(carrying):
                    chainable.append((values, group_value, carrying))
        else:
            size = 1 + rng.poisson(lam, groups)
            start = np.concatenate([[0], np.cumsum(size)])
            flat = rng.choice(m, int(size.sum()), p=pop)
            flat[start[:-1]] = group_value
            reps = size[group[subj]]
            heads = np.repeat(subj, reps)
            offset = np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps)
            val = flat[np.repeat(start[group[subj]], reps) + offset]
            keep = rng.random(len(heads)) < set_rho
            none = np.setdiff1d(subj, heads[keep])
            heads = np.concatenate([heads[keep], none])
            val = np.concatenate([val[keep], rng.choice(m, len(none), p=pop)])
            if frng is not None and null:   # attr2: a null world keeps the number of values and draws them by popularity
                val = frng.choice(m, len(val), p=pop)
        r = len(domains)
        pairs = np.unique(np.stack([heads, values[val]], 1), axis=0)
        parts.append(np.stack([pairs[:, 0], np.full(len(pairs), r), pairs[:, 1]], 1))
        domains.append(domain)
        ranges.append(values)
        # a specific relation (fewer than half of its facts have a value with twin_uses or more uses) has a twin with
        # probability twin_rare, a categorical relation with probability twin
        uses = np.unique(pairs[:, 1], return_counts=True)[1]
        specific = uses[uses >= P['twin_uses']].sum() < 0.5 * len(pairs)
        u_twin, share, inverse = rng.random(), rng.uniform(*P['twin_share']), rng.random() < 0.5
        if u_twin < (P['twin_rare'] if specific else P['twin']):
            keep = pairs[rng.random(len(pairs)) < share]
            r = len(domains)
            cols = (keep[:, 1], keep[:, 0]) if inverse else (keep[:, 0], keep[:, 1])
            parts.append(np.stack([cols[0], np.full(len(keep), r), cols[1]], 1))
            domains.append(values if inverse else domain)
            ranges.append(domain if inverse else values)
    for _ in range(int(rng.integers(P['links'][0], P['links'][1] + 1))):
        count = int(ns * rng.uniform(*P['link_degree']))
        homophily = rng.uniform(*P['homophily'])
        x = rng.integers(0, ns, count)
        inside = rng.random(count) < homophily
        pick = rng.random(count)
        same = np.array([members[g][int(p * len(members[g]))] for g, p in zip(group[x], pick)]) if count else np.empty(0, dtype=np.int64)
        y = np.where(inside, same, rng.integers(0, ns, count))
        keep = x != y
        r = len(domains)
        x, y = x[keep], y[keep]
        if rng.random() < P['symmetric']:
            x, y = np.concatenate([x, y]), np.concatenate([y, x])
        pairs = np.unique(np.stack([x, y], 1), axis=0)
        parts.append(np.stack([pairs[:, 0], np.full(len(pairs), r), pairs[:, 1]], 1))
        domains.append(subjects)
        ranges.append(subjects)
    info: dict[str, Any] = {}
    if frng is not None:   # attr2: at most one value chain from a finer to a coarser vocabulary
        made_chain = False
        if frng.random() < P['chain'] and len(chainable) >= 2:
            order = frng.permutation(len(chainable))
            for i in order:
                for j in order:
                    if i == j or made_chain:
                        continue
                    va, ga, ca = chainable[i]
                    vb, gb, cb = chainable[j]
                    both = np.intersect1d(ca, cb)
                    if len(both) and len(va) > len(vb):
                        pairs = np.unique(np.stack([va[ga[both]], vb[gb[both]]], 1), axis=0)
                        parts.append(np.stack([pairs[:, 0], np.full(len(pairs), len(domains)), pairs[:, 1]], 1))
                        domains.append(va)
                        ranges.append(vb)
                        made_chain = True
        info = dict(strength=s, null=bool(null), chain=bool(made_chain))
    return (*_compact(np.unique(np.concatenate(parts).astype(np.int64), axis=0), domains, ranges, next_id), info)


def _taxonomy(rng: np.random.Generator, n: int, P: dict[str, Any]) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
    """A taxonomy of depth 1 to 3 over ``n`` rows: the cluster of every row per level (level 0 the coarsest), the
    cluster counts, and the parent of every cluster of level ``l`` at ``parents[l - 1]`` (``pf_rec1.taxonomy``)."""
    depth = int(rng.integers(P['depth'][0], P['depth'][1] + 1))
    cap = max(2, n // 4)
    counts = [int(min(cap, max(2, round(_lu(rng, *P['top'])))))]
    parents = []
    for _ in range(1, depth):
        children = np.maximum(1, np.round([_lu(rng, *P['branch']) for _ in range(counts[-1])]).astype(np.int64))
        if children.sum() > cap:
            children = np.maximum(1, np.floor(children * cap / children.sum())).astype(np.int64)
        parents.append(np.repeat(np.arange(counts[-1]), children))
        counts.append(int(children.sum()))
    leaf = rng.choice(counts[-1], n, p=_zipf(rng, counts[-1], *P['leaf_zipf']))
    anc = [leaf]
    for par in reversed(parents):
        anc.insert(0, par[anc[0]])
    return anc, counts, parents


def record_world_latent(rng: np.random.Generator, P: dict[str, Any]) -> LatentWorld:
    """The latent world of rec1 (``pf_rec1.latent_world``): tables, columns, chains, references and inherited columns.

    Returns:
        The latent world; the metadata holds the regime, strength, table and row counts, the mean fidelity and the
        counts of column kinds, chains, references and inherited columns.
    """
    ns = int(_lu(rng, *P['rows']))
    T = int(rng.integers(P['tables'][0], P['tables'][1] + 1))
    share = rng.dirichlet(np.ones(T))
    sizes = np.maximum(50, np.round(share * ns)).astype(np.int64)
    ns = int(sizes.sum())
    starts = np.concatenate([[0], np.cumsum(sizes)])
    u = rng.random()
    regime = 'null' if u < P['null'] else ('weak' if u < P['null'] + P['weak'] else 'strong')
    s = 0.0 if regime == 'null' else float(rng.uniform(*(P['weak_strength'] if regime == 'weak' else P['strength'])))
    null = regime == 'null'
    budget, next_id = P['max_entities'] - ns, ns
    parts: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    props: list[tuple[np.ndarray, np.ndarray]] = []   # (values, pop) of earlier property columns (shared vocabularies)
    n_cols = [max(1, int(round(_lu(rng, *P['columns'])))) for _ in range(T)]
    remaining = sum(n_cols)
    rhos: list[float] = []
    tabs: list[dict[str, Any]] = []
    counts_info = dict(functional=0, multi=0, key=0, shared=0, chains=0, references=0, inherited=0)

    def add(pairs: np.ndarray, dom: np.ndarray, ran: np.ndarray) -> np.ndarray:
        pairs = np.unique(pairs, axis=0)
        parts.append(np.stack([pairs[:, 0], np.full(len(pairs), len(domains)), pairs[:, 1]], 1))
        domains.append(dom)
        ranges.append(ran)
        return pairs

    def fidelity(u_free: float) -> float:
        d = rng.uniform(s - P['spread'], s + P['spread'])   # drawn for every column (draw order)
        return 0.0 if (null or u_free < P['free']) else float(np.clip(d, 0.0, 1.0))

    for t in range(T):
        subj_t = np.arange(starts[t], starts[t + 1])
        nt = len(subj_t)
        anc, counts, parents = _taxonomy(rng, nt, P)
        depth = len(counts)
        func, keys = [], []   # single-valued columns (level, values, mapping, pop, row value); keys
        for _ in range(n_cols[t]):
            remaining -= 1
            # every draw below is made for every column, so that a parameter changes only what it names
            level = int(rng.integers(0, depth))
            u_key, u_shared, u_free, u_func = rng.random(), rng.random(), rng.random(), rng.random()
            rho = fidelity(u_free)
            lam, cov = rng.uniform(*P['extra']), rng.uniform(*P['coverage'])
            m_draw = int(round(_lu(rng, P['vocabulary'][0], min(P['vocabulary'][1], max(P['vocab_rows'] * nt, P['vocabulary'][0] + 1)))))
            clusters = counts[level]
            is_key = u_key < P['key'] and clusters <= budget - 2 * remaining
            if is_key:
                values = np.arange(next_id, next_id + clusters)
                next_id, budget = next_id + clusters, budget - clusters
                pop = _zipf(rng, clusters, *P['value_zipf'])
                mapping = np.arange(clusters)
            elif props and u_shared < P['shared']:
                values, pop = props[int(rng.integers(len(props)))]
                mapping = rng.choice(len(values), clusters, p=pop)
                counts_info['shared'] += 1
            else:
                m = int(max(2, min(m_draw, budget - 2 * remaining)))
                values = np.arange(next_id, next_id + m)
                next_id, budget = next_id + m, budget - m
                pop = _zipf(rng, m, *P['value_zipf'])
                mapping = rng.choice(m, clusters, p=pop)
                props.append((values, pop))
            m = len(values)
            rhos.append(rho)
            functional = u_func < P['functional']
            subj = subj_t[rng.random(nt) < cov]
            if not len(subj):
                continue
            cl = anc[level][subj - starts[t]]
            if functional:
                follow = rng.random(len(subj)) < rho
                val = np.where(follow, mapping[cl], rng.choice(m, len(subj), p=pop))
                heads = subj
            else:
                k = 1 + rng.poisson(lam, len(subj))
                kmax = int(k.max())
                sets = np.concatenate([mapping[:, None], rng.choice(m, (clusters, max(0, kmax - 1)), p=pop)], 1)
                heads = np.repeat(subj, k)
                slot = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
                follow = rng.random(len(heads)) < rho
                val = np.where(follow, sets[np.repeat(cl, k), slot], rng.choice(m, len(heads), p=pop))
            pairs = add(np.stack([heads, values[val]], 1), subj_t, values)
            counts_info['functional' if functional else 'multi'] += 1
            if functional:
                row_value = np.full(nt, -1, dtype=np.int64)
                row_value[subj - starts[t]] = values[val]
                func.append((level, values, mapping, pop, row_value))
            if is_key:
                counts_info['key'] += 1
                keys.append((level, values, np.unique(np.searchsorted(values, pairs[:, 1]))))
        # value chains: a finer key's values link to the values of a coarser single-valued column
        made = 0
        for kl, kvalues, kused in keys:
            for fl, fvalues, fmap, _, _ in func:
                uc = rng.random()
                if fl < kl and made < P['chains_max'] and uc < P['chain'] and len(kused):
                    up = kused.copy()
                    for lv in range(kl, fl, -1):   # the level-fl ancestor of every used level-kl cluster
                        up = parents[lv - 1][up]
                    add(np.stack([kvalues[kused], fvalues[fmap[up]]], 1), kvalues, fvalues)
                    made += 1
        counts_info['chains'] += made
        # a reference to an earlier table, and the columns inherited through it
        if t > 0:
            u_ref, target, level, u_free = rng.random(), int(rng.integers(t)), int(rng.integers(0, depth)), rng.random()
            rho, cov = fidelity(u_free), rng.uniform(*P['coverage'])
            if u_ref < P['reference']:
                tb = tabs[target]
                rows_u = np.arange(tb['start'], tb['start'] + tb['n'])
                anchor_row = rng.choice(rows_u, counts[level])   # one referenced row per cluster
                subj = subj_t[rng.random(nt) < cov]
                follow = rng.random(len(subj)) < rho
                ref = np.where(follow, anchor_row[anc[level][subj - starts[t]]], rng.choice(rows_u, len(subj)))
                if len(subj):
                    add(np.stack([subj, ref], 1), subj_t, rows_u)
                    counts_info['references'] += 1
                    made_i = 0
                    for _, vals_u, _, pop_u, row_value_u in tb['func']:
                        ui, rho_i, cov_i = rng.random(), fidelity(rng.random()), rng.uniform(*P['coverage'])
                        if ui < P['inherit'] and made_i < 2:
                            pick = rng.random(len(subj)) < cov_i
                            sel, ref_sel = subj[pick], ref[pick]
                            copied = row_value_u[ref_sel - tb['start']]
                            follow = (rng.random(len(sel)) < rho_i) & (copied >= 0)
                            noise = vals_u[rng.choice(len(vals_u), len(sel), p=pop_u)]
                            val = np.where(follow, copied, noise)
                            if len(sel):
                                add(np.stack([sel, val], 1), subj_t, vals_u)
                                made_i += 1
                    counts_info['inherited'] += made_i
        tabs.append(dict(start=int(starts[t]), n=nt, func=func))
    info = dict(regime=regime, strength=s, null=bool(null), tables=T, rows=ns, rho_mean=float(np.mean(rhos)) if rhos else 0.0, **counts_info)
    return (*_compact(np.unique(np.concatenate(parts).astype(np.int64), axis=0), domains, ranges, next_id), info)


def _pick(cdf_rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Row-wise categorical draw: ``cdf_rows`` (n, k) cumulative weights per row (last column 1), ``u`` (n,) uniforms."""
    picked: np.ndarray = np.minimum((u[:, None] > cdf_rows).sum(1), cdf_rows.shape[1] - 1)
    return picked


def list_world(rng: np.random.Generator, P: dict[str, Any]) -> LatentWorld:
    """The latent world of list1 (``pf_list1.latent_world``): anchors, item lists from communities, hubs, attributes and links.

    Returns:
        The latent world; the metadata holds the regime, strength, the sizes and the counts of relation kinds.
    """
    na = int(_lu(rng, *P['anchors']))
    kinds = int(rng.integers(P['kinds'][0], P['kinds'][1] + 1))
    rich = rng.random() < P['rich']
    attr_range = P['rich_attributes'] if rich else P['attributes']
    anchor_range = P['rich_anchor_attributes'] if rich else P['anchor_attributes']
    u_reg = rng.random()
    regime = 'null' if u_reg < P['null'] else ('weak' if u_reg < P['null'] + P['weak'] else 'strong')
    s = 0.0 if regime == 'null' else float(rng.uniform(*(P['weak_strength'] if regime == 'weak' else P['strength'])))
    null = regime == 'null'

    def fidelity() -> float:
        d, u_free = rng.uniform(s - P['spread'], s + P['spread']), rng.random()   # both drawn for every relation
        return 0.0 if (null or u_free < P['free']) else float(np.clip(d, 0.0, 1.0))

    # list relations first (kind, fidelity, mean and spread of the list size, storage direction)
    specs = []
    for _ in range(int(rng.integers(P['lists'][0], P['lists'][1] + 1))):
        k = int(rng.integers(kinds))
        rho = fidelity()
        mu, sigma, rev = _lu(rng, *P['list_mean']), rng.uniform(*P['list_spread']), rng.random() < P['reverse']
        specs.append([k, rho, mu, sigma, rev])
    # items per kind = communities x items per community, items per community = cover x the largest mean list size on
    # the kind; the entity budget scales the items down
    C = int(max(2, min(round(_lu(rng, *P['communities'])), na // 8)))
    mu_kind = [max([sp[2] for sp in specs if sp[0] == k] or [3.0]) for k in range(kinds)]
    cover = [rng.uniform(*P['cover']) for _ in range(kinds)]
    m = np.array([max(30, int(round(C * max(2.0, cover[k] * mu_kind[k])))) for k in range(kinds)], dtype=np.int64)
    room = P['max_entities'] - na - (attr_range[1] * kinds + anchor_range[1]) * P['vocabulary'][1]
    if m.sum() > room:
        m = np.maximum(30, np.floor(m * room / m.sum())).astype(np.int64)
    for sp in specs:
        sp[2] = min(sp[2], 0.5 * m[sp[0]])
    cw = _zipf(rng, C, *P['community_zipf'])
    # anchors: 1-2 communities by size (Gumbel top-k), Dirichlet(1) weights
    top = min(P['memberships'][1], C)
    k_mem = rng.integers(P['memberships'][0], P['memberships'][1] + 1, na)
    k_mem = np.minimum(k_mem, top)
    keys = np.log(cw)[None, :] - np.log(-np.log(rng.random((na, C))))
    order = np.argsort(-keys, axis=1)[:, :top]
    g = rng.gamma(1.0, 1.0, (na, top))
    g[np.arange(top)[None, :] >= k_mem[:, None]] = 0.0
    weights = g / g.sum(1, keepdims=True)
    mem_cdf = np.cumsum(weights, 1)
    mem_cdf[:, -1] = 1.0
    main_comm = order[np.arange(na), np.argmax(weights, 1)]
    next_id = na
    item_ids, item_comm, comm_items, comm_cdf, glob_p = [], [], [], [], []
    for k in range(kinds):
        ids = np.arange(next_id, next_id + m[k])
        next_id += int(m[k])
        comm = rng.choice(C, int(m[k]), p=cw)
        z = rng.uniform(*P['item_zipf'])
        w = np.empty(int(m[k]))
        per_items, per_cdf = [], []
        for c in range(C):
            sel = np.flatnonzero(comm == c)
            wc = (rng.permutation(len(sel)) + 1.0) ** -z
            w[sel] = wc
            per_items.append(sel)
            per_cdf.append(np.cumsum(wc) / wc.sum() if len(sel) else np.ones(0))
        item_ids.append(ids)
        item_comm.append(comm)
        comm_items.append(per_items)
        comm_cdf.append(per_cdf)
        glob_p.append(w / w.sum())
    hub_items = []
    for k in range(kinds):
        u_hub, n_hub = rng.random(), int(round(_lu(rng, *P['hub_count'])))
        picks = rng.choice(int(m[k]), min(n_hub, int(m[k])), replace=False)
        hub_items.append(picks if u_hub < P['hubs'] else picks[:0])
    parts: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    counts = dict(lists=0, reversed=0, item_attributes=0, anchor_attributes=0, shared=0, item_links=0, list_facts=0)
    anchors = np.arange(na)

    def add(pairs: np.ndarray, dom: np.ndarray, ran: np.ndarray) -> np.ndarray:
        pairs = np.unique(pairs, axis=0)
        parts.append(np.stack([pairs[:, 0], np.full(len(pairs), len(domains)), pairs[:, 1]], 1))
        domains.append(dom)
        ranges.append(ran)
        return pairs

    expected = 0.0
    for sp in specs:   # a plain float loop: Python 3.12's sum() adds with compensation, the research Python did not
        expected += na * sp[2]
    scale = min(1.0, 0.9 * P['max_facts'] / max(expected, 1.0))
    rhos = []
    for k, rho, mu, sigma, rev in specs:
        mu = max(1.0, mu * scale)
        rhos.append(rho)
        size = np.exp(rng.normal(np.log(mu) - sigma ** 2 / 2, sigma, na))
        size = np.clip(np.round(size), 1, max(1, int(0.5 * m[k]))).astype(np.int64)
        draws = np.ceil(1.6 * size).astype(np.int64) + 3
        who = np.repeat(anchors, draws)
        nd = len(who)
        follow = rng.random(nd) < rho
        cpos = _pick(mem_cdf[who], rng.random(nd))
        comm_of = order[who, cpos]
        u_item = rng.random(nd)
        noise = np.where(rng.random(nd) < P['noise_pop'], rng.choice(int(m[k]), nd, p=glob_p[k]), rng.integers(0, int(m[k]), nd))
        item = noise.copy()
        for c in range(C):
            sel = np.flatnonzero(follow & (comm_of == c))
            if len(sel) and len(comm_items[k][c]):
                item[sel] = comm_items[k][c][np.minimum(np.searchsorted(comm_cdf[k][c], u_item[sel]), len(comm_items[k][c]) - 1)]
        # no repeat inside a list: first occurrence per (anchor, item), then the first `size` per anchor in draw order
        key = who * int(m[k]) + item
        _, first = np.unique(key, return_index=True)
        first = np.sort(first)
        w_first = who[first]
        rank = np.arange(len(first)) - np.searchsorted(w_first, w_first)
        keep = first[rank < size[w_first]]
        pairs = np.stack([who[keep], item_ids[k][item[keep]]], 1)
        lo, hi = P['hub_rate']
        rates = rng.uniform(lo, hi, len(hub_items[k]))
        for hub, rate in zip(hub_items[k].tolist(), rates.tolist()):
            holders = anchors[rng.random(na) < rate]
            pairs = np.concatenate([pairs, np.stack([holders, np.full(len(holders), item_ids[k][hub])], 1)])
        if rev:
            add(pairs[:, ::-1], item_ids[k], anchors)
            counts['reversed'] += 1
        else:
            add(pairs, anchors, item_ids[k])
        counts['lists'] += 1
        counts['list_facts'] += len(pairs)
    # item attributes (value entities per attribute; a community -> value map)
    item_attr = []
    for k in range(kinds):
        for _ in range(int(rng.integers(attr_range[0], attr_range[1] + 1))):
            rho, cov = fidelity(), rng.uniform(*P['coverage'])
            V = int(max(2, min(round(_lu(rng, *P['vocabulary'])), m[k] // 2)))
            values = np.arange(next_id, next_id + V)
            next_id += V
            pop = _zipf(rng, V, *P['value_zipf'])
            cmap = rng.choice(V, C, p=pop)
            subj = np.flatnonzero(rng.random(int(m[k])) < cov)
            val = np.where(rng.random(len(subj)) < rho, cmap[item_comm[k][subj]], rng.choice(V, len(subj), p=pop))
            add(np.stack([item_ids[k][subj], values[val]], 1), item_ids[k], values)
            item_attr.append((values, pop))
            rhos.append(rho)
            counts['item_attributes'] += 1
    # anchor attributes (own value entities, or those of an item attribute)
    for _ in range(int(rng.integers(anchor_range[0], anchor_range[1] + 1))):
        rho, cov = fidelity(), rng.uniform(*P['coverage'])
        u_shared, pick = rng.random(), int(rng.integers(max(1, len(item_attr))))
        if item_attr and u_shared < P['shared_values']:
            values, pop = item_attr[pick]
            V = len(values)
            counts['shared'] += 1
        else:
            V = int(max(2, min(round(_lu(rng, *P['vocabulary'])), na // 2)))
            values = np.arange(next_id, next_id + V)
            next_id += V
            pop = _zipf(rng, V, *P['value_zipf'])
        cmap = rng.choice(V, C, p=pop)
        subj = np.flatnonzero(rng.random(na) < cov)
        val = np.where(rng.random(len(subj)) < rho, cmap[main_comm[subj]], rng.choice(V, len(subj), p=pop))
        add(np.stack([subj, values[val]], 1), anchors, values)
        rhos.append(rho)
        counts['anchor_attributes'] += 1
    # item-item links inside the communities
    for k in range(kinds):
        u_link, deg, hom = rng.random(), rng.uniform(*P['link_degree']), rng.uniform(*P['homophily'])
        if u_link >= P['item_links']:
            continue
        cnt = int(m[k] * deg)
        x = rng.integers(0, int(m[k]), cnt)
        inside = rng.random(cnt) < hom
        pos = rng.random(cnt)
        y = rng.integers(0, int(m[k]), cnt)
        for c in range(C):
            sel = np.flatnonzero(inside & (item_comm[k][x] == c))
            if len(sel) and len(comm_items[k][c]):
                y[sel] = comm_items[k][c][np.minimum((pos[sel] * len(comm_items[k][c])).astype(np.int64), len(comm_items[k][c]) - 1)]
        ok = x != y
        if ok.any():
            add(np.stack([item_ids[k][x[ok]], item_ids[k][y[ok]]], 1), item_ids[k], item_ids[k])
            counts['item_links'] += 1
    facts = np.unique(np.concatenate(parts).astype(np.int64), axis=0)
    if len(facts) > P['max_facts']:   # the last cut (rare): a uniform subset
        facts = facts[np.sort(rng.choice(len(facts), P['max_facts'], replace=False))]
    info = dict(regime=regime, strength=s, null=bool(null), rich=bool(rich), anchors=na, kinds=kinds, items=int(m.sum()), communities=C,
                rho_mean=float(np.mean(rhos)) if rhos else 0.0, list_scale=float(scale), **counts)
    return (*_compact(facts, domains, ranges, next_id), info)


def record_world(seed: int, index: int, spec: 'FamilySpec') -> tuple[dict[str, WorldBatch], dict[str, Any]]:
    """The world of a record family at stream index ``index`` (``pf_*.sample``; ``s5_null`` with the null).

    Args:
        seed: The run seed.
        index: The stream index.
        spec: The family; its generator names the record family, ``params`` override its parameters and
            ``null_swap_budget`` asks for the structural null.

    Returns:
        The batch and its metadata (the research keys and order, without the null's timing).
    """
    P = record_params(spec.generator, spec.overrides)
    layout = template(seed, index)
    version = VERSIONS[spec.generator]

    def stream(name: str) -> int:
        return world_stream(version, seed, index, name)

    rng = np.random.default_rng(stream('world'))
    if spec.generator in ('categorical2', 'attr2'):
        frng = np.random.default_rng(stream('fidelity')) if spec.generator == 'attr2' else None
        n, nr, facts, domains, ranges, info = categorical_world(rng, P, frng)
    else:
        latent: Callable[[np.random.Generator, dict[str, Any]], LatentWorld] = record_world_latent if spec.generator == 'rec1' else list_world
        n, nr, facts, domains, ranges, info = latent(rng, P)
    summary: dict[str, Any] = {}
    if spec.null_swap_budget is not None:
        facts, summary = swap_truth(facts, domains, ranges, n, seed, index, spec.generator, float(spec.null_swap_budget))
    fact_list = [tuple(f) for f in facts.tolist()]
    batch, observed = query_world(layout, fact_list, [d.tolist() for d in domains], [r.tolist() for r in ranges], n, nr, rng, stream,
                                  P['observed'], cap=True)
    meta: dict[str, Any] = dict(family=spec.generator, entities=n, relations_total=nr, context=observed, facts=len(fact_list), **info)
    if spec.generator != 'categorical2':   # the null's summary travelled in the generator's own metadata
        meta.update(summary)
    meta['program_slot'] = layout.program_slot
    if summary:
        meta.update({k: v for k, v in summary.items() if k not in meta})
        meta['structural_null'] = NULL_VERSION
    meta['recipe_batch'] = layout.recipe_batch
    return batch, meta
