"""The legacy plain generator: the worlds of the ``plain`` family and the template of every other family.

Stream index ``x`` of the prior has the legacy world ``confirm_common.samples(seed, 300,000 + x)['new']`` of the research
code. Through its wrapper chain that is legacy position ``p = 350,000 + x`` (``program``) and sampler index
``i = 362,000 + x`` (``aa_common.batches`` and below). By hashes of these indices the world is

- a control world (one index in five, ``joint-scaling common.source_arm``): ``factorial.control_world``;
- otherwise a sparse rule world (joint-worlds-v2 ``worlds.generate``), rewired by pair affinities in two of the eight
  rule worlds of a block of ten (``pair_batches``), and given new rule programs in four of them (``program``).

The two tasks of the batch sample 15 rule targets and 15 base targets with 512 negatives each (``training.make_batches``,
``pair_batches.from_world``, ``program.make_batch_values``), and a targeted observation mask removes the observed answers
of up to three frequent anchors per task (``oc_prior``, arm ``targeted``). Every family keeps from this template its
task names and order, the head and tail pattern of each task and the program-slot flag; the record and real families
also keep its RecipeBatch provenance (``Template.recipe_batch``).

The configuration values come from ``recipe.recipe_config(seed)``, the top of the research chain (``aa_common.cfg``);
the generator reads its seed, batch size, negatives and world sizes. The research rebuilt the same worlds in several
layers; they are built once here, and the batches of worlds that a later layer replaces are not sampled, only their
target pools are checked (the one way they could fail).
"""
import random
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Optional

import numpy as np

from ..batch import WorldBatch
from ..recipe import LEGACY_OFFSET, RecipeConfig, recipe_config
from ..streams import joint_world_stream, observation_mask_stream, pair_affinity_stream, pair_slot_stream, rule_program_stream, source_schedule_stream
from .factorial import LegacyBatch, Proofs, Rule, Triple, control_batches, control_world, derive, json_sha256, sample_groups, task_quotas

POSITION_OFFSET = 50_000
"""``confirm_common.OFFSET``: legacy position = ``LEGACY_OFFSET + stream index + POSITION_OFFSET``."""

SAMPLER_OFFSET = 12_000
"""``aa_common.OFFSET``: sampler index = legacy position + ``SAMPLER_OFFSET``."""

SCHEDULE_FORK = 250
"""``joint-scaling common.FORK``: the first sampler index of the source schedule."""

MAX_FACTS = 200_000
"""``worlds.MAX_FACTS``: the fact budget of a rule world's closure."""

OBSERVATION_CAP = 20_000
"""Most observed facts of a rule world (``worlds.observe``)."""

WORLDS_VERSION = 'joint-worlds-v2'
"""``worlds.VERSION``: the version field of a rule world's record."""

MAX_ANCHORS, MAX_REMOVAL_FRACTION, MIN_ROLE_COUNT, ROLE_QUANTILE = 3, .20, 2, .75
"""The targeted mask (``oc_prior``): anchors per task, largest removed share of the context, frequent-role rule."""


# ----------------------------------------------------------------------------------------------- rule worlds

@dataclass(frozen=True)
class RuleWorld:
    """A typed rule world with exact truth and an independent observation (joint-worlds-v2 ``World``, profile ``sparse``)."""

    seed: int
    n: int
    nr: int
    base: frozenset[Triple]
    truth: frozenset[Triple]
    proofs: Proofs
    rules: tuple[Rule, ...]
    context: frozenset[Triple]
    observation_seed: int
    observed_fraction: float
    types: tuple[int, ...]
    signatures: tuple[tuple[int, int], ...]

    def fingerprint(self) -> str:
        """SHA-256 of the research world record (``World.fingerprint``); the ``world_sha256`` of its batches."""
        return json_sha256(dict(version=WORLDS_VERSION, seed=self.seed, profile='sparse', n=self.n, R=self.nr, base=sorted(self.base),
                                truth=sorted(self.truth), context=sorted(self.context),
                                rules=[dict(body=r.body, head=r.head, reverse=r.reverse) for r in self.rules],
                                proofs=[[q, rid, premises] for q, (rid, premises) in sorted(self.proofs.items())],
                                observation_seed=self.observation_seed, observed_fraction=self.observed_fraction,
                                types=self.types, signatures=self.signatures), compact=True)


def observe(truth: frozenset[Triple], seed: int) -> tuple[frozenset[Triple], float]:
    """Observe each fact with one probability U(0.5, 0.9), at most ``OBSERVATION_CAP`` facts (``worlds.observe``)."""
    rng = np.random.default_rng(seed)
    fraction = float(rng.uniform(.5, .9))
    ordered = sorted(truth)
    kept = [q for q, flag in zip(ordered, rng.random(len(ordered)) < fraction) if flag]
    if len(kept) > OBSERVATION_CAP:
        kept = [kept[i] for i in rng.choice(len(kept), OBSERVATION_CAP, replace=False)]
    return frozenset(kept), fraction


def rule_world(seed: int) -> RuleWorld:
    """A sparse rule world (``worlds.generate('sparse', seed)``): 256 to 768 typed entities, 10 to 14 relations, five
    rules (two unary reversals, chains of two and three, and a chain over a derived relation); ids permuted before the
    closure and the observation."""
    rng = np.random.default_rng(joint_world_stream(seed, 'world'))
    n, nr = int(rng.integers(256, 769)), int(rng.integers(10, 15))
    nbase = nr - 4
    nt = int(rng.integers(3, 6))
    types = rng.integers(nt, size=n)
    types[:nt] = np.arange(nt)
    a, b, c, d = map(int, rng.integers(nt, size=4))
    domains = [(a, a), (a, b), (b, c), (c, d), (a, c), (a, d)]
    while len(domains) < nbase:
        choices = [(a, b), (a, c), (a, d), (b, c), (c, d)]
        domains.append(choices[int(rng.integers(len(choices)))])
    rules: tuple[Rule, ...] = (Rule((0,), 0, True), Rule((1,), nbase, True), Rule((1, 2), nbase + 1), Rule((1, 2, 3), nbase + 2), Rule((nbase + 1, 3), nbase + 3))
    signatures = [*domains, (b, a), (a, c), (a, d), (a, d)]
    pools = [np.where(types == i)[0] for i in range(nt)]
    alpha = float(rng.uniform(0., 1.5))
    popularity = np.ones(n)
    for pool in pools:
        popularity[pool] = (rng.permutation(len(pool)) + 1.) ** -alpha
    frequencies = rng.dirichlet(np.full(nbase, 1.2))
    counts = rng.multinomial(int(n * rng.uniform(3, 7)), frequencies)
    base: set[Triple] = set()
    for relation, (dom, ran) in enumerate(domains):
        heads, tails = pools[dom], pools[ran]
        head_weights = popularity[heads] ** float(rng.uniform(0., .8))
        tail_weights = popularity[tails]
        head_weights /= head_weights.sum()
        tail_weights = tail_weights / tail_weights.sum()
        count = max(64, int(counts[relation]))
        pairs = zip(rng.choice(heads, count, p=head_weights), rng.choice(tails, count, p=tail_weights))
        base.update((int(h), relation, int(t)) for h, t in pairs if h != t)
    entities, relations = rng.permutation(n).tolist(), rng.permutation(nr).tolist()
    base_ids = frozenset((entities[h], relations[r], entities[t]) for h, r, t in base)
    rules = tuple(Rule(tuple(relations[r] for r in rule.body), relations[rule.head], rule.reverse) for rule in rules)
    remapped_types, remapped_signatures = [0] * n, [(0, 0)] * nr
    for old, new in enumerate(entities):
        remapped_types[new] = int(types[old])
    for old, new in enumerate(relations):
        remapped_signatures[new] = signatures[old]
    truth, proofs = derive(base_ids, rules, MAX_FACTS)
    observation_seed = joint_world_stream(seed, 'observation')
    context, fraction = observe(truth, observation_seed)
    return RuleWorld(seed, n, nr, base_ids, truth, proofs, rules, context, observation_seed, fraction, tuple(remapped_types),
                     tuple(remapped_signatures))


def affinity_rewiring(source: RuleWorld) -> RuleWorld:
    """Degree- and type-preserving swaps of each relation's base pairs toward latent pair affinities, then the new
    closure and observation (``affinity.transform(world, 'affinity')``: 20 Metropolis proposals per fact, beta 2)."""
    z = np.random.default_rng(pair_affinity_stream(source.seed, 'latent')).normal(size=(source.n, 4))
    z = 2 * z / np.linalg.norm(z, axis=1, keepdims=True)
    energy = z @ z.T / 2
    facts = set(source.base)
    rng = np.random.default_rng(pair_affinity_stream(source.seed, 'rewire'))
    beta = 2.
    tiny = float(np.finfo(float).tiny)
    for r in range(source.nr):
        edges = sorted((h, t) for h, relation, t in facts if relation == r)
        pairs = set(edges)
        if len(edges) < 2:
            continue
        for _ in range(20 * len(edges)):
            i, j = map(int, rng.integers(len(edges), size=2))
            u = rng.random()   # drawn for every proposal, valid or not
            h, t = edges[i]
            x, y = edges[j]
            if i == j or h == x or t == y or h == y or x == t or (h, y) in pairs or (x, t) in pairs:
                continue
            delta = energy[h, y] + energy[x, t] - energy[h, t] - energy[x, y]
            if np.log(max(u, tiny)) > beta * delta:
                continue
            pairs.remove((h, t))
            pairs.remove((x, y))
            pairs.update(((h, y), (x, t)))
            edges[i], edges[j] = (h, y), (x, t)
        facts.difference_update(q for q in source.base if q[1] == r)
        facts.update((h, r, t) for h, t in pairs)
    truth, proofs = derive(facts, source.rules, MAX_FACTS)
    context, fraction = observe(truth, source.observation_seed)
    return replace(source, base=frozenset(facts), truth=truth, proofs=proofs, context=context, observed_fraction=fraction)


def alternatives(signatures: tuple[tuple[int, int], ...], available: set[int], target: tuple[int, int]) -> list[tuple[tuple[int, ...], bool]]:
    """Every typed rule body over ``available`` relations that ends in the ``target`` signature: chains of length 1 to 3,
    then reversed unary bodies (``program.alternatives``)."""
    options = []
    for length in (1, 2, 3):
        for body in product(sorted(available), repeat=length):
            if any(signatures[a][1] != signatures[b][0] for a, b in zip(body, body[1:])):
                continue
            if (signatures[body[0]][0], signatures[body[-1]][1]) == target:
                options.append((body, False))
    for relation in sorted(available):
        if tuple(reversed(signatures[relation])) == target:
            options.append(((relation,), True))
    return options


def rule_programs(source: RuleWorld) -> RuleWorld:
    """New typed rule programs (1 to 3 bodies) for every derived relation of a rule world, on its unchanged base
    (``program.transform``); a head is dropped when its typed fact bound would pass ``MAX_FACTS``."""
    rng = random.Random(rule_program_stream(source.seed, 'program'))
    base_relations = {r for _, r, _ in source.base}
    heads = sorted({r.head for r in source.rules} - base_relations)
    rng.shuffle(heads)
    rules = [r for r in source.rules if r.head in base_relations]   # the unary symmetry of a base relation stays
    available = set(base_relations)
    sizes = Counter(source.types)
    bound = 2 * len(source.base)   # the base and at most its symmetric copies
    for head in heads:
        signature = source.signatures[head]
        additional = sizes[signature[0]] * sizes[signature[1]]
        if bound + additional > MAX_FACTS:
            continue
        candidates = alternatives(source.signatures, available, signature)
        if not candidates:
            continue
        chosen: list[tuple[tuple[int, ...], bool]] = []
        for _ in range(min(rng.randint(1, 3), len(candidates))):
            categories = {length: [v for v in candidates if len(v[0]) == length and v not in chosen] for length in (1, 2, 3)}
            lengths = [length for length, pool in categories.items() if pool]
            length = rng.choices(lengths, weights=[{1: .2, 2: .5, 3: .3}[x] for x in lengths])[0]
            chosen.append(rng.choice(categories[length]))
        rules.extend(Rule(body, head, reverse) for body, reverse in chosen)
        available.add(head)
        bound += additional
    truth, proofs = derive(source.base, tuple(rules), MAX_FACTS)
    context, fraction = observe(truth, source.observation_seed)
    return replace(source, rules=tuple(rules), truth=truth, proofs=proofs, context=context, observed_fraction=fraction)


# ----------------------------------------------------------------------------------------------- source selection

def source_arm(seed: int, index: int) -> str:
    """``'control'`` for one sampler index in each block of five, else ``'joint'`` (``source_arm(..., 'replay20')``)."""
    if index < SCHEDULE_FORK:
        raise ValueError(f'the source schedule starts at sampler index {SCHEDULE_FORK}')
    choices = ['joint'] * 4 + ['control']
    block, offset = divmod(index - SCHEDULE_FORK, 5)
    random.Random(source_schedule_stream(seed, block)).shuffle(choices)
    return choices[offset]


def pair_selected(seed: int, index: int) -> bool:
    """Whether a rule world gets the pair-affinity rewiring: two of the eight per ten sampler indices (``pair_batches.selected``)."""
    if source_arm(seed, index) != 'joint':
        return False
    block = index // 10 * 10
    choices = [i for i in range(block, block + 10) if source_arm(seed, i) == 'joint']
    return index in sorted(choices, key=lambda i: pair_slot_stream(seed, i))[:2]


def program_selected(seed: int, position: int) -> bool:
    """Whether a rule world gets new rule programs: four of the eight per ten legacy positions (``program.selected``)."""
    block = position // 10 * 10
    offsets = [j for j in range(block, block + 10) if source_arm(seed, SAMPLER_OFFSET + j) == 'joint']
    return position in sorted(offsets, key=lambda j: rule_program_stream(seed, f'slot:{j}'))[:4]


# ----------------------------------------------------------------------------------------------- batches

def _pools(world: RuleWorld) -> dict[str, list[Triple]]:
    """The observed targets of the two tasks: derived facts for ``rule``, base facts for ``base``."""
    return {'rule': sorted(world.context - world.base), 'base': sorted(world.context & world.base)}


def check_pools(world: RuleWorld, total: int) -> None:
    """Refuse a rule world with fewer observed targets than the batch size in a task (the research's only failure of
    the batches of a world that a later layer replaces).

    Raises:
        ValueError: For a short pool.
    """
    for task, pool in _pools(world).items():
        if len(pool) < total:
            raise ValueError(f'{task}: {len(pool)} observed targets; require {total}; no resampling')


def rule_batches(world: RuleWorld, cfg: RecipeConfig, index: int, joint_exclusion: bool) -> dict[str, LegacyBatch]:
    """The two tasks of a rule world: 30 targets per task drawn, 15 kept by the split of ``task_quotas``; the context of
    a task lacks its own targets, or every target of both tasks with ``joint_exclusion`` (the rewired worlds).

    Raises:
        ValueError: For a short pool, or a context without facts.
    """
    total = cfg.batch_size
    fingerprint = world.fingerprint()
    pools = _pools(world)
    result = {}
    for task, (count, tails) in task_quotas(index, total).items():
        pool = pools[task]
        if len(pool) < total:
            raise ValueError(f'{task}: {len(pool)} observed targets; require {total}; no resampling')
        sampling_seed = joint_world_stream(world.seed, f'sample:{task}')
        rng = random.Random(sampling_seed)
        groups, directions = sample_groups(rng.sample(pool, total), world.truth, world.n, cfg.num_negative, rng)
        indices = [*range(tails), *range(total // 2, total // 2 + count - tails)]
        groups, directions = groups[indices], directions[indices]
        context = world.context - set(map(tuple, groups[:, 0].tolist()))
        result[task] = LegacyBatch(groups, directions, context, world.truth, world.n, world.nr, fingerprint, sampling_seed, count)
    if joint_exclusion:
        targets = {tuple(q) for batch in result.values() for q in batch.groups[:, 0].tolist()}
        result = {task: batch._replace(context=world.context - targets) for task, batch in result.items()}
    if not all(batch.context for batch in result.values()):
        raise ValueError('a task context without facts')
    return result


def _roles(batch: LegacyBatch, context: frozenset[Triple]) -> list[tuple[int, int, float]]:
    """(role degree, observed answers, role quantile) of each query group under ``context`` (``oc_prior.covariates``)."""
    heads: Counter = Counter()
    tails: Counter = Counter()
    pools: dict[tuple[bool, int], list[int]] = defaultdict(list)
    for h, r, t in context:
        heads[h, r] += 1
        tails[r, t] += 1
    for (_, r), count in heads.items():
        pools[True, r].append(count)
    for (r, _), count in tails.items():
        pools[False, r].append(count)
    for pool in pools.values():
        pool.sort()
    rows = []
    for (h, r, t), head in zip(batch.groups[:, 0].tolist(), batch.head_prediction.tolist(), strict=True):
        degree = heads[h, r] if head else tails[r, t]
        observed = tails[r, t] if head else heads[h, r]
        pool = pools[head, r]
        rows.append((degree, observed, bisect_left(pool, degree) / len(pool) if pool else 0.))
    return rows


def _frequent(row: tuple[int, int, float]) -> bool:
    return row[0] >= MIN_ROLE_COUNT and row[2] >= ROLE_QUANTILE


def targeted_context(batch: LegacyBatch) -> frozenset[Triple]:
    """The context without the observed answers of up to three frequent anchors (``oc_prior.masks``, arm ``targeted``).

    Anchors are tried in a shuffled order; one is taken when its removal keeps at most a fifth of the context removed,
    leaves the context non-empty, and leaves every taken anchor frequent and without an observed answer.
    """
    rng = random.Random(observation_mask_stream(batch.sampling_seed))
    candidates = [i for i, row in enumerate(_roles(batch, batch.context)) if _frequent(row) and row[1] > 0]
    rng.shuffle(candidates)
    selected: list[int] = []
    removed: set[Triple] = set()
    for index in candidates:
        h, r, t = batch.groups[index, 0].tolist()
        head = bool(batch.head_prediction[index])
        trial = removed | {q for q in batch.context if q[1] == r and (q[2] == t if head else q[0] == h)}
        if trial == removed or len(trial) > int(MAX_REMOVAL_FRACTION * len(batch.context)):
            continue
        observed = batch.context - trial
        if not observed:
            continue
        rows = _roles(batch, observed)
        if not all(_frequent(rows[i]) and rows[i][1] == 0 for i in [*selected, index]):
            continue
        selected.append(index)
        removed = trial
        if len(selected) == MAX_ANCHORS:
            break
    return batch.context - removed


# ----------------------------------------------------------------------------------------------- the template

@dataclass(frozen=True)
class Template:
    """The legacy plain world of a stream index.

    Attributes:
        batch: ``'rule'`` and ``'base'`` with 15 query groups each, as an update reads them.
        recipe_batch: Per task the research RecipeBatch fields that no update reads: ``world_sha256`` (the fingerprint of
            the world the batch was sampled from), ``sampling_seed`` and ``requested_positives``.
        program_slot: Whether the world has new rule programs.
    """

    batch: dict[str, WorldBatch]
    recipe_batch: dict[str, dict[str, Any]]
    program_slot: bool


def template(seed: int, index: int, cfg: Optional[RecipeConfig] = None) -> Template:
    """The legacy plain world of stream index ``index`` (``confirm_common.samples(seed, 300,000 + index)['new']``).

    Args:
        seed: The run seed.
        index: The stream index (``s1`` and ``s2`` indices, and the indices of the hook families).
        cfg: The recipe, ``recipe_config(seed)`` by default.

    Returns:
        The template.

    Raises:
        ValueError: When a world cannot give a batch (no resampling).
    """
    cfg = cfg or recipe_config(seed)
    position = POSITION_OFFSET + LEGACY_OFFSET + index
    sampler = SAMPLER_OFFSET + position
    program = program_selected(seed, position)
    if source_arm(seed, sampler) == 'control':
        batches = control_batches(control_world(cfg, sampler), cfg, sampler)
    else:
        world = rule_world(joint_world_stream(seed, f'train-world:{sampler}'))
        pair = pair_selected(seed, sampler)
        if pair:
            check_pools(world, cfg.batch_size)
            world = affinity_rewiring(world)
        if program:
            check_pools(world, cfg.batch_size)
            world = rule_programs(world)
        batches = rule_batches(world, cfg, sampler, pair)
    batch = {task: WorldBatch(b.groups, b.head_prediction, targeted_context(b), b.truth, b.num_entities, b.num_relations)
             for task, b in batches.items()}
    provenance = {task: dict(world_sha256=b.world_sha256, sampling_seed=b.sampling_seed, requested_positives=b.requested_positives)
                  for task, b in batches.items()}
    return Template(batch, provenance, program)


def plain(seed: int, index: int) -> tuple[dict[str, WorldBatch], dict[str, Any]]:
    """A ``plain`` world of the stage-1 stream (``op_worlds.world(seed, 's1', index)``): the template itself.

    Returns:
        The batch and ``{'family': 'plain', 'program_slot': ..., 'recipe_batch': ...}``.
    """
    t = template(seed, index)
    return t.batch, dict(family='plain', program_slot=t.program_slot, recipe_batch=t.recipe_batch)
