"""The control worlds of the legacy generator, and the Horn-world primitives that its rule worlds share.

A fifth of the legacy stream indices are control worlds (``legacy.source_arm``). The research built them through six
generator layers, each a treatment of the one below; the port keeps their order and draws:

1. source world: a world-first world (``world_first.generate_world``, family ``current``) or a world of one logical
   family (``prior_mechanisms.family_world``: split or repeated composition, transitivity, inverse, symmetry), by the
   family schedule of ``prior_factorial``, with its observation;
2. sparse demonstrations: isolated example paths that reveal each rule (``prior_factorial.add_sparse_demonstrations``);
3. the factorial treatment: a fresh background relation, attached to the paths or spread over the core
   (``prior_factorial.make_world``, by the topology schedule of ``topology_mix``), and the final id permutation;
4. the observation dose: later sparse examples hidden, matched background edges added (``demo_context``, arm ``dose``);
5. the cardinality treatment: a second, disjoint matching of each grid relabelled to the rule
   (``cardinality_prior``, arm ``multiple``);
6. endpoint reuse: sixteen copies with merged head-only and tail-only nodes (``endpoint_prior``, arm
   ``concentrated``).

``control_batches`` samples the two tasks of the update from the final world. The research also hashed every layer into
a provenance record of the batches (``world_sha256``); no update reads it and no draw depends on it, so the port omits
it. Only the training split is reachable; the research checks that the construction guarantees (for example the local
rule applications of a closure) are not repeated.
"""
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any, Callable, Iterable, NamedTuple, Optional, Sequence

import torch

from ..batch import Triple
from ..recipe import NEGATIVES, QUERIES, WORLD
from ..streams import world_first_stream

Proofs = dict[Triple, tuple[int, tuple[Triple, ...]]]
"""A derived fact's witness: the index of its rule and the premises of one application (from an earlier round)."""

FACTORIAL_FAMILIES = ('composition_split', 'composition_repeated', 'transitivity', 'inverse', 'symmetry')
"""The logical families of the factorial schedule next to ``current`` (``prior_factorial.NEW_FAMILIES``)."""

ENDPOINT_COPIES = 16
"""Copies of the endpoint-reuse world (``endpoint_prior.COPIES``)."""


# ----------------------------------------------------------------------------------------------- Horn worlds

@dataclass(frozen=True)
class Rule:
    """A relation path of length 1 to 3 implies ``head`` between its endpoints; a unary rule may reverse them."""

    body: tuple[int, ...]
    head: int
    reverse: bool = False


def derive(base: Iterable[Triple], rules: tuple[Rule, ...], limit: Optional[int] = None) -> tuple[frozenset[Triple], Proofs]:
    """The closure of ``base`` under ``rules`` with one witness per derived fact (``generator.derive``, ``worlds.closure``).

    Each round indexes the facts in sorted order and joins rule bodies left to right; the first path found for an
    endpoint pair is its witness. Both research implementations make exactly these choices.

    Args:
        base: The base facts.
        rules: The rules.
        limit: The fact budget of the joint worlds (``worlds.MAX_FACTS``), or ``None``.

    Returns:
        The truth and the witnesses of the derived facts.

    Raises:
        ValueError: For a rule outside the path language, or a closure above ``limit`` facts (never truncated).
    """
    for rule in rules:
        if not 1 <= len(rule.body) <= 3 or (rule.reverse and len(rule.body) != 1):
            raise ValueError('rules must be paths of length 1 to 3; only unary rules can reverse')
    facts = set(base)
    if limit is not None and len(facts) > limit:
        raise ValueError('the world exceeds its fact budget')
    proofs: Proofs = {}
    while True:
        index: dict[int, dict[int, list[Triple]]] = {}
        for fact in sorted(facts):
            index.setdefault(fact[1], {}).setdefault(fact[0], []).append(fact)
        pending: Proofs = {}
        for rule_id, rule in enumerate(rules):
            paths: list[tuple[int, int, tuple[Triple, ...]]] = [(f[0], f[2], (f,)) for values in index.get(rule.body[0], {}).values() for f in values]
            for relation in rule.body[1:]:
                joined: dict[tuple[int, int], tuple[int, int, tuple[Triple, ...]]] = {}
                for start, end, premises in paths:
                    for f in index.get(relation, {}).get(end, ()):
                        joined.setdefault((start, f[2]), (start, f[2], premises + (f,)))
                paths = list(joined.values())
            for start, end, premises in paths:
                fact = (end, rule.head, start) if rule.reverse else (start, rule.head, end)
                if fact not in facts:
                    pending.setdefault(fact, (rule_id, premises))
        if not pending:
            return frozenset(facts), proofs
        if limit is not None and len(facts) + len(pending) > limit:
            raise ValueError('the closure exceeds the fact budget')
        facts.update(pending)
        proofs.update(pending)


def proof_leaves(fact: Triple, base: frozenset[Triple], proofs: Proofs) -> frozenset[Triple]:
    """The base facts that the recorded witness of a fact expands to (``generator.proof_leaves``).

    Args:
        fact: A fact of the closure.
        base: The base facts.
        proofs: The witnesses of ``derive``.

    Returns:
        The base facts of the witness tree.
    """
    leaves, pending, visited = set(), [fact], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in base:
            leaves.add(current)
        else:
            pending.extend(proofs[current][1])
    return frozenset(leaves)


@dataclass(frozen=True)
class Episode:
    """A complete world and its observation (the fields of ``generator.Episode`` that a world reads).

    Attributes:
        seed: The world's own seed; it seeds the streams of the later layers.
        num_entities: Entity count.
        num_relations: Relation count.
        rules: The rules of the world.
        base: The base facts.
        truth: The closure of the base under the rules.
        context: The observed facts.
        support: The facts that reveal the rules (the demonstration paths and their correct endpoint pairs).
        protected: The support facts that the observation must keep.
        background_relation: The relation outside every rule.
    """

    seed: int
    num_entities: int
    num_relations: int
    rules: tuple[Rule, ...]
    base: frozenset[Triple]
    truth: frozenset[Triple]
    context: frozenset[Triple]
    support: frozenset[Triple]
    protected: frozenset[Triple]
    background_relation: int


def _validate_world(ep: Episode) -> None:
    """Refuse an episode whose truth is not its closure, or whose observation is empty or not true (``validate_world``).

    Raises:
        ValueError: With the violated rule.
    """
    truth, _ = derive(ep.base, ep.rules)
    if truth != ep.truth or not ep.context <= truth or not ep.protected <= ep.context:
        raise ValueError('world closure, observations or protected facts disagree')
    n, nr = ep.num_entities, ep.num_relations
    if not ep.context or any(not (0 <= h < n and 0 <= t < n and 0 <= r < nr) for h, r, t in truth):
        raise ValueError('empty context or out-of-vocabulary fact')


def _fact_map(entities: Sequence[int], relations: Sequence[int]) -> Callable[[Triple], Triple]:
    """The map of a fact to its new ids: entity ``e`` becomes ``entities[e]``, relation ``r`` becomes ``relations[r]``."""
    def fact(q: Triple) -> Triple:
        return entities[q[0]], relations[q[1]], entities[q[2]]

    return fact


def _remap_episode(ep: Episode, entities: list[int], relations: list[int]) -> Episode:
    """The episode under new entity and relation ids (``redundancy._remap``)."""
    fact = _fact_map(entities, relations)

    def facts(values: Iterable[Triple]) -> frozenset[Triple]:
        return frozenset(map(fact, values))

    return replace(ep, num_entities=len(entities),
                   rules=tuple(Rule(tuple(relations[r] for r in rule.body), relations[rule.head], rule.reverse) for rule in ep.rules),
                   base=facts(ep.base), truth=facts(ep.truth), context=facts(ep.context),
                   support=facts(ep.support), protected=facts(ep.protected), background_relation=relations[ep.background_relation])


# ----------------------------------------------------------------------------------------------- batches

class LegacyBatch(NamedTuple):
    """One task of a legacy world: the fields of the research ``RecipeBatch``."""

    groups: torch.Tensor
    head_prediction: torch.Tensor
    context: frozenset[Triple]
    truth: frozenset[Triple]
    num_entities: int
    num_relations: int
    sampling_seed: int
    requested_positives: int


def sample_groups(positives: list[Triple], truth: frozenset[Triple], num_entities: int, num_negative: int,
                  rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Query groups with oracle-false negatives drawn with replacement; the first half tail, the second half head
    queries (``standard_recipe.sample_groups``).

    Args:
        positives: The query facts.
        truth: Every true fact of the world.
        num_entities: Entity count.
        num_negative: Negatives per group.
        rng: The sampling generator; it drew the positives.

    Returns:
        ``groups`` int64 ``[len(positives), 1 + num_negative, 3]`` and ``head_prediction`` bool.

    Raises:
        ValueError: For fewer than two positives, a positive that is false or outside the world, or a query without a
            false candidate.
    """
    if len(positives) < 2 or num_negative < 1 or num_entities < 1:
        raise ValueError('require at least two positive groups and one negative')
    if any(q not in truth or min(q) < 0 or max(q[0], q[2]) >= num_entities for q in positives):
        raise ValueError('positive targets must be valid oracle facts')
    tail_truth: dict[tuple[int, int], set[int]] = defaultdict(set)
    head_truth: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in truth:
        tail_truth[h, r].add(t)
        head_truth[r, t].add(h)
    groups = torch.tensor(positives, dtype=torch.long)[:, None].repeat(1, 1 + num_negative, 1)
    heads = torch.arange(len(positives)) >= len(positives) // 2
    for index, (h, r, t) in enumerate(positives):
        head = bool(heads[index])
        forbidden = head_truth[r, t] if head else tail_truth[h, r]
        candidates = [e for e in range(num_entities) if e not in forbidden]
        if not candidates:
            raise ValueError(f'no false candidates for the query {(h, r, t)}')
        groups[index, 1:, 0 if head else 2] = torch.tensor(rng.choices(candidates, k=num_negative))
    return groups, heads


def task_quotas(index: int, total: int) -> dict[str, tuple[int, int]]:
    """The query groups and tail groups of the two tasks of a legacy world (``training.make_batches``; the same split in
    ``pair_batches``, ``program`` and ``prior_factorial``).

    Each task gets half the groups. When a half is odd, the rule task of an odd index takes its extra tail query.

    Args:
        index: The sampler index.
        total: The query groups of the world.

    Returns:
        ``{'rule': (groups, tail groups), 'base': (groups, tail groups)}``.
    """
    rules = total // 2
    rule_tails = rules // 2 + (index % 2 if rules % 2 else 0)
    return {'rule': (rules, rule_tails), 'base': (total - rules, total // 2 - rule_tails)}


# ----------------------------------------------------------------------------------------------- source worlds

def _remap_grids(groups: list[dict], fact: Callable[[Triple], Triple], relations: Optional[list[int]] = None) -> list[dict]:
    """Grid blocks with remapped facts; with ``relations``, the relation order of every bundle is remapped too."""
    out = []
    for g in groups:
        bundles = []
        for b in g['bundles']:
            bundle = {**b, 'triple': fact(b['triple']), 'paths': [tuple(fact(q) for q in path) for path in b['paths']]}
            if relations is not None:
                bundle['order'] = tuple(relations[r] for r in b['order'])
            bundles.append(bundle)
        out.append({**g, 'bundles': bundles})
    return out


def _first_world(seed: int, index: int) -> tuple[Episode, list[dict]]:
    """A world-first world of the training split with its candidate grids (``world_first.generate_world``), with the
    world sizes ``recipe.WORLD``.

    Args:
        seed: The stream seed of the source worlds.
        index: The world index.

    Returns:
        The episode and its grid blocks (remapped, with ``correct`` per bundle).
    """
    world_seed = world_first_stream(seed, 'world', ('train', index))
    obs_seed = world_first_stream(world_seed, 'observation', 0)
    rng, obs = random.Random(world_seed), random.Random(obs_seed)
    ne = rng.randint(WORLD.min_core_entities, WORLD.max_core_entities)
    nr = rng.randint(WORLD.min_relations, WORLD.max_relations)
    output, noise = nr - 2, nr - 1
    rules = tuple(Rule(tuple(rng.sample(range(output), length)), output) for length in (2, 3))
    active = sorted({r for rule in rules for r in rule.body} | {noise})
    capacity = ne * (ne - 1) * len(active)
    density = rng.uniform(WORLD.min_density, WORLD.max_density)
    base: set[Triple] = set()
    for code in rng.sample(range(capacity), max(1, int(capacity * density))):
        pair, rel = divmod(code, len(active))
        h, t = divmod(pair, ne - 1)
        base.add((h, active[rel], t + (t >= h)))
    background = frozenset(base)
    grid = WORLD.grid_size
    endpoints = rng.sample(range(ne), 4 * WORLD.blocks_per_rule * grid)
    next_node, offset = ne, 0
    blocks: list[dict[str, Any]] = []
    for rid, rule in enumerate(rules):
        wrong = [order for order in permutations(rule.body) if order != rule.body]
        for ordinal in range(WORLD.blocks_per_rule):
            heads, tails = endpoints[offset:offset + grid], endpoints[offset + grid:offset + 2 * grid]
            offset += 2 * grid
            mapping = rng.sample(range(grid), grid)
            copies = rng.randint(1, WORLD.max_copies)
            bundles = []
            for i, h in enumerate(heads):
                for j, t in enumerate(tails):
                    order = rule.body if mapping[i] == j else rng.choice(wrong)
                    paths = []
                    for _ in range(copies):
                        nodes = [h, *range(next_node, next_node + len(order) - 1), t]
                        next_node += len(order) - 1
                        path = tuple((a, r, b) for a, r, b in zip(nodes, order, nodes[1:]))
                        base.update(path)
                        paths.append(path)
                    bundles.append({'triple': (h, output, t), 'order': order, 'paths': paths})
            blocks.append({'rule': rid, 'demonstration': ordinal == 0, 'copies': copies, 'bundles': bundles})
    truth, _ = derive(base, rules)
    # One complete path copy per pair; further copies observed by a mask coupled by path position within a block.
    observed = obs.uniform(WORLD.min_observed, WORLD.max_observed)
    context = {q for q in sorted(background) if obs.random() < observed}
    support: set[Triple] = set()
    for block in blocks:
        rule = rules[block['rule']]
        keep = [[True] * len(rule.body)] + [[obs.random() < observed for _ in rule.body] for _ in range(block['copies'] - 1)]
        for bundle in block['bundles']:
            for copy, path in enumerate(bundle['paths']):
                context.update(q for q, seen in zip(path, keep[copy]) if seen or block['demonstration'])
            if block['demonstration']:
                support.update(q for path in bundle['paths'] for q in path)
                if tuple(bundle['order']) == rule.body:
                    support.add(bundle['triple'])
    context.update(support)
    # The research drew a masked view of the context from ``obs`` here, for its world record only.
    entities = rng.sample(range(next_node), next_node)
    relations = rng.sample(range(nr), nr)
    ep = _remap_episode(Episode(world_seed, next_node, nr, rules, frozenset(base), truth, frozenset(context), frozenset(support),
                               frozenset(support), noise), entities, relations)
    _validate_world(ep)
    groups = [{**block, 'bundles': [{**b, 'correct': tuple(b['order']) == ep.rules[block['rule']].body} for b in block['bundles']]}
              for block in _remap_grids(blocks, _fact_map(entities, relations), relations)]
    return ep, groups


def _family_rules(family: str) -> tuple[Rule, ...]:
    """The rules of a logical family of the factorial schedule (``prior_mechanisms.family_rules``)."""
    rules = {'composition_split': (Rule((0, 1), 4), Rule((0, 1, 2), 5)), 'composition_repeated': (Rule((0, 0), 4), Rule((0, 0, 0), 4)),
             'transitivity': (Rule((0, 0), 0),), 'inverse': (Rule((0,), 4, True),), 'symmetry': (Rule((0,), 0, True),)}
    if family not in rules:
        raise ValueError(f'unknown logical family {family!r}')
    return rules[family]


def _family_world(family: str, seed: int, index: int, blocks: int) -> tuple[Episode, list[dict]]:
    """A world of balanced endpoint grids of one logical family, split ``test`` (``prior_mechanisms.family_world``).

    Returns:
        The episode and its grid blocks (remapped, with ``correct`` per bundle).

    Raises:
        ValueError: When a grid pair is true by accident (no resampling).
    """
    world_seed = world_first_stream(seed, 'mechanism-family-world', ('test', index))
    rng = random.Random(world_seed)
    obs = random.Random(world_first_stream(world_seed, 'mechanism-family-observation', 0))
    rules = _family_rules(family)
    lengths = (1, 1) if family in ('inverse', 'symmetry') else (2, 3)
    endpoints = rng.sample(range(96), 12 * blocks)
    base: set[Triple] = set()
    density = rng.uniform(.001, .003)
    for code in rng.sample(range(96 * 95), max(1, int(96 * 95 * density))):
        h, t = divmod(code, 95)
        base.add((h, 9, t + (t >= h)))
    observed = obs.uniform(.4, .9)
    context = {q for q in sorted(base) if obs.random() < observed}
    support: set[Triple] = set()
    next_node, offset = 96, 0
    groups: list[dict[str, Any]] = []
    for group, length in enumerate(lengths):
        rule = rules[min(group, len(rules) - 1)]
        body = rule.body if family != 'transitivity' else (0,) * length
        wrong = [p for p in permutations(body) if p != body]
        for ordinal in range(blocks):
            heads, tails = endpoints[offset:offset + 3], endpoints[offset + 3:offset + 6]
            offset += 6
            mapping = rng.sample(range(3), 3)
            copies = rng.randint(1, 2) if length > 1 else 1
            bundles = []
            for i, h in enumerate(heads):
                for j, t in enumerate(tails):
                    correct = mapping[i] == j
                    order = body
                    if not correct:
                        if wrong:
                            order = rng.choice(wrong)
                        else:
                            position = rng.randrange(length)
                            order = tuple(3 if p == position else r for p, r in enumerate(body))
                    paths = []
                    for _ in range(copies):
                        nodes = [t, h] if rule.reverse else [h, *range(next_node, next_node + length - 1), t]
                        next_node += length - 1
                        path = tuple((a, r, b) for a, r, b in zip(nodes, order, nodes[1:]))
                        base.update(path)
                        context.update(path)
                        paths.append(path)
                    query = (h, rule.head, t)
                    if ordinal == 0:
                        support.update(q for path in paths for q in path)
                        if correct:
                            support.add(query)
                    bundles.append({'triple': query, 'correct': correct, 'paths': paths})
            groups.append({'demonstration': ordinal == 0, 'bundles': bundles})
    truth, _ = derive(base, rules)
    context.update(support)
    ep = Episode(world_seed, next_node, 10, rules, frozenset(base), truth, frozenset(context), frozenset(support), frozenset(support), 9)
    entities, relations = rng.sample(range(next_node), next_node), rng.sample(range(10), 10)
    ep = _remap_episode(ep, entities, relations)
    remapped = _remap_grids(groups, _fact_map(entities, relations))
    for group_value in remapped:
        for bundle in group_value['bundles']:
            if (bundle['triple'] in ep.truth) != bundle['correct']:
                raise ValueError('a controlled family has an accidentally true endpoint pair; no resampling')
    _validate_world(ep)
    return ep, remapped


def _source_world(seed: int, index: int, family: str) -> tuple[Episode, list[dict]]:
    """The observed source world of a family (``prior_factorial.source_world``, split ``train``).

    Returns:
        The episode and its grid blocks.
    """
    if family == 'current':
        return _first_world(world_first_stream(seed, 'factorial-original-v1', 'train'), index)
    if family not in FACTORIAL_FAMILIES:
        raise ValueError(f'unknown family {family!r}')
    ep, groups = _family_world(family, world_first_stream(seed, 'factorial-fresh-family-v1', 'train'), index, WORLD.blocks_per_rule)
    planted = {q for block in groups for b in block['bundles'] for path in b['paths'] for q in path}
    context = set(ep.context - planted)
    obs = random.Random(world_first_stream(ep.seed, 'factorial-family-observations-v1', 'train'))
    fraction = obs.uniform(WORLD.min_observed, WORLD.max_observed)
    # The supported observation: the first path complete, further copies masked by position, coupled within a block.
    for block in groups:
        example = block['bundles'][0]['paths']
        masks = [[True] * len(example[0])] + [[obs.random() < fraction for _ in p] for p in example[1:]]
        for bundle in block['bundles']:
            for path, mask in zip(bundle['paths'], masks):
                context.update(q for q, keep in zip(path, mask) if keep or block['demonstration'])
    context.update(ep.support)
    observed = replace(ep, context=frozenset(context))
    _validate_world(observed)
    return observed, groups


def _add_sparse_demonstrations(ep: Episode, groups: list[dict]) -> tuple[Episode, list[dict]]:
    """Isolated example paths, three correct and three wrong per demonstration block (``add_sparse_demonstrations``).

    Raises:
        ValueError: When the closure contradicts an example's label.
    """
    rng = random.Random(world_first_stream(ep.seed, 'factorial-sparse-demonstrations-v1', 0))
    base, context, support = set(ep.base), set(ep.context), set(ep.support)
    next_node = ep.num_entities
    added: list[dict[str, Any]] = []
    for block in groups:
        if not block['demonstration']:
            continue
        examples = []
        for correct in (True, False):
            choices = [b for b in block['bundles'] if b['correct'] == correct]
            for example in rng.sample(choices, 3):
                old_path = example['paths'][0]
                h, t = next_node, next_node + len(old_path)
                nodes = list(range(h, t + 1))
                next_node = t + 1
                path = tuple((a, old[1], b) for a, old, b in zip(nodes, old_path, nodes[1:]))
                reverse = old_path[0][0] == example['triple'][2]
                query = (t, example['triple'][1], h) if reverse else (h, example['triple'][1], t)
                base.update(path)
                context.update(path)
                support.update(path)
                if correct:
                    context.add(query)
                    support.add(query)
                examples.append({'triple': query, 'correct': correct, 'paths': [path]})
        added.append({'demonstration': True, 'bundles': examples})
    truth, _ = derive(base, ep.rules)
    for block in added:
        if any((b['triple'] in truth) != b['correct'] for b in block['bundles']):
            raise ValueError('sparse demonstration labels disagree with the complete closure')
    result = replace(ep, num_entities=next_node, base=frozenset(base), truth=truth, context=frozenset(context),
                     support=frozenset(support), protected=frozenset(support))
    return result, groups + added


# ----------------------------------------------------------------------------------------------- treatments

@dataclass(frozen=True)
class FactorialWorld:
    """A treated world with its target pools (the fields of ``prior_factorial.FactorialWorld`` that a world reads).

    Attributes:
        episode: The treated world.
        rule_pool: The held-out rule targets.
        base_pool: The observed base targets.
        family: The family of the source world; it seeds the query sample.
        witnesses: Per rule target the paths that prove it.
    """

    episode: Episode
    rule_pool: tuple[Triple, ...]
    base_pool: tuple[Triple, ...]
    family: str
    witnesses: dict[Triple, tuple[tuple[Triple, ...], ...]]


def _scheduled_family(seed: int, index: int) -> str:
    """The family of training world ``index``: five ``current`` and the five logical families per ten worlds."""
    values = ['current'] * 5 + list(FACTORIAL_FAMILIES)
    random.Random(world_first_stream(seed, 'factorial-family-schedule-v1', ('train', index // 10))).shuffle(values)
    return values[index % 10]


def _scheduled_topology(seed: int, index: int) -> str:
    """``'core'`` or ``'attached'``: each family half and half within twenty training worlds (``topology_mix``).

    Raises:
        ValueError: When a family occurs an odd number of times in the block of twenty.
    """
    block = index // 20
    positions: dict[str, list[int]] = defaultdict(list)
    for offset in range(20):
        positions[_scheduled_family(seed, 20 * block + offset)].append(offset)
    values = [''] * 20
    for family, indices in sorted(positions.items()):
        if len(indices) % 2:
            raise ValueError('the family schedule is not balanced over twenty worlds')
        random.Random(world_first_stream(seed, 'topology-mix-assignment-v1', ('train', block, family))).shuffle(indices)
        for i, offset in enumerate(indices):
            values[offset] = 'core' if i < len(indices) // 2 else 'attached'
    return values[index % 20]


def _factorial_world(attached: bool, family: str, sparse: tuple[Episode, list[dict]], ids: tuple[list[int], list[int]]) -> FactorialWorld:
    """The factorial treatment of a source world with sparse demonstrations (``prior_factorial.make_world``, arm
    ``both`` with ``attached``, else ``rules``).

    Raises:
        ValueError: When the world has no held-out planted target, or the fresh relation changed a label.
    """
    ep, groups = sparse
    witnesses = {b['triple']: tuple(tuple(p) for p in b['paths']) for block in groups if not block['demonstration']
                 for b in block['bundles'] if b['correct'] and b['triple'] in ep.truth}
    if not witnesses or any(q in ep.context for q in witnesses):
        raise ValueError('need held-out positive rule targets')
    base_pool = sorted(ep.base & ep.context)
    private = {q[2] for block in groups for b in block['bundles'] for path in b['paths'] for q in path[:-1]}
    core = sorted(set(range(ep.num_entities)) - private)
    # Unary rules have no inner path nodes: they attach at the path endpoints.
    anchors = sorted(private or {e for block in groups for b in block['bundles'] for e in (b['triple'][0], b['triple'][2])})
    rng = random.Random(world_first_stream(ep.seed, 'factorial-augmentation-v1', 0))
    new_relation = ep.num_relations
    attached_edges = frozenset((node, new_relation, rng.choice([c for c in core if c != node])) for node in anchors)
    background = set()
    for code in rng.sample(range(len(core) * (len(core) - 1)), len(anchors)):
        h, t = divmod(code, len(core) - 1)
        background.add((core[h], new_relation, core[t + (t >= h)]))
    extra = attached_edges if attached else frozenset(background)
    if len(extra) != len(anchors):
        raise ValueError('topology edge budgets differ')
    truth, _ = derive(ep.base | extra, ep.rules)
    if {q for q in truth if q[1] != new_relation} != set(ep.truth):
        raise ValueError('the irrelevant augmentation changed an original label')
    result = replace(ep, num_relations=ep.num_relations + 1, base=ep.base | extra, truth=truth, context=ep.context | extra,
                     background_relation=new_relation)
    fact = _fact_map(*ids)
    result = _remap_episode(result, *ids)
    _validate_world(result)
    return FactorialWorld(result, tuple(sorted(fact(q) for q in witnesses)), tuple(sorted(fact(q) for q in base_pool)), family,
                          {fact(q): tuple(tuple(fact(e) for e in p) for p in paths) for q, paths in witnesses.items()})


def _observation_dose(world: FactorialWorld, source: tuple[Episode, list[dict]], sparse: tuple[Episode, list[dict]],
                      ids: tuple[list[int], list[int]]) -> FactorialWorld:
    """Hide every sparse example after the first of its kind and add as many background edges in the core
    (``demo_context.layout``, ``_background`` and ``transform`` with view ``dose``).

    Raises:
        ValueError: When the treatment would leak a target or remove its last witness or a base target.
    """
    before, groups = source
    after, expanded = sparse
    entities, fact = ids[0], _fact_map(*ids)
    private = {q[2] for g in groups for b in g['bundles'] for p in b['paths'] for q in p[:-1]}
    core = sorted(entities[i] for i in range(before.num_entities) if i not in private)
    sparse_nodes = range(before.num_entities, after.num_entities)
    removed: set[Triple] = set()
    for gi, g in enumerate(expanded[len(groups):]):
        for role in (True, False):
            values = [b for b in g['bundles'] if b['correct'] == role]
            random.Random(world_first_stream(after.seed, 'demo-dose-order-v1', (gi, role))).shuffle(values)
            for b in values[1:]:
                removed.update(fact(q) for q in b['paths'][0])
                if role:
                    removed.add(fact(b['triple']))
    added = frozenset(fact(q) for q in after.context - before.context)
    ep = world.episode
    if not added <= ep.context:
        raise ValueError('require the full observation of the examples before the dose')
    relation = ep.background_relation
    if any(relation in r.body or relation == r.head for r in ep.rules):
        raise ValueError('the background relation must be outside every rule')
    rng = random.Random(world_first_stream(ep.seed, 'demo-embedding-background-v1', 0))
    available = [(h, relation, t) for h in core for t in core if h != t and (h, relation, t) not in ep.base]
    extra = frozenset(rng.sample(available, len(sparse_nodes)))
    if len(extra) != len(sparse_nodes):
        raise ValueError('the matched background edge budget changed')
    context = (ep.context - removed) | extra
    result = replace(ep, base=ep.base | extra, truth=ep.truth | extra, context=context, protected=ep.protected & context)
    if any(q in context for q in world.rule_pool) or any(q[1] == ep.background_relation for q in world.rule_pool):
        raise ValueError('a rule target leaked or uses the augmentation relation')
    if any(not any(set(path) <= context for path in world.witnesses[q]) for q in world.rule_pool):
        raise ValueError('the dose removed the evidence of a query')
    pool = tuple(q for q in world.base_pool if q not in added)
    if not set(pool) <= context:
        raise ValueError('a common base target is no longer observed')
    return replace(world, episode=result, base_pool=pool)


def _second_matching(groups: list[dict], seed: int) -> tuple[dict[Triple, Triple], frozenset[Triple]]:
    """Relabel the path slots of one disjoint matching per grid to the correct order (``cardinality_prior.second_matching``).

    Raises:
        ValueError: When a grid has no one-to-one primary matching of three endpoints, or two relabellings conflict.
    """
    replacements: dict[Triple, Triple] = {}
    promoted: set[Triple] = set()
    for index, block in enumerate(groups):
        correct = {b['triple'][0]: b for b in block['bundles'] if b['correct']}
        heads = sorted(correct)
        if len(heads) < 3 or len({b['triple'][2] for b in correct.values()}) != len(heads):
            raise ValueError('require a primary one-to-one matching and at least three endpoints')
        rng = random.Random(world_first_stream(seed, 'cardinality-second-matching-v1', index))
        rng.shuffle(heads)
        shift = rng.randrange(1, len(heads))
        for offset, h in enumerate(heads):
            target = correct[heads[(offset + shift) % len(heads)]]['triple'][2]
            bundle = next(b for b in block['bundles'] if b['triple'][0] == h and b['triple'][2] == target)
            if bundle['correct']:
                raise ValueError('the secondary matching must be disjoint')
            order = tuple(q[1] for q in correct[h]['paths'][0])
            for path in bundle['paths']:
                if len(path) != len(order):
                    raise ValueError('path geometry changed within a block')
                for edge, relation in zip(path, order):
                    new = (edge[0], relation, edge[2])
                    if edge in replacements and replacements[edge] != new:
                        raise ValueError('a shared path slot has conflicting relabelings')
                    replacements[edge] = new
            promoted.add(bundle['triple'])
    return replacements, frozenset(promoted)


def _multiple_matchings(original: FactorialWorld, groups: list[dict]) -> FactorialWorld:
    """The world with two true matchings per grid (``cardinality_prior.variants``, arm ``multiple``).

    The research also built the arms ``observed`` and ``multiple_observed``, whose checks follow from those of
    ``control`` and ``multiple``; nothing of them reaches the world.

    Raises:
        ValueError: When the relabelled paths miss a positive, change the geometry or the observation, or a world fails
            its validation.
    """
    mapping, promoted = _second_matching(groups, original.episode.seed)

    def fact(q: Triple) -> Triple:
        return mapping.get(q, q)

    def facts(values: Iterable[Triple]) -> frozenset[Triple]:
        return frozenset(map(fact, values))

    ep = original.episode
    new_base = facts(ep.base)
    truth, _ = derive(new_base, ep.rules)
    if not promoted <= truth or not set(original.rule_pool) <= truth:
        raise ValueError('the relabelled paths do not imply every designated positive')
    if len(new_base) != len(ep.base) or {(h, t) for h, _, t in new_base} != {(h, t) for h, _, t in ep.base}:
        raise ValueError('the cardinality treatment changed the path geometry or the fact count')
    context = facts(ep.context)
    if len(context) != len(ep.context) or not context <= truth:
        raise ValueError('the relabelled observation changed its size or became false')
    changed = replace(ep, base=new_base, truth=truth, context=context, support=facts(ep.support), protected=facts(ep.protected))
    multiple = replace(original, episode=changed, base_pool=tuple(sorted(fact(q) for q in original.base_pool)),
                       witnesses={q: tuple(tuple(map(fact, path)) for path in paths) for q, paths in original.witnesses.items()})
    for world in (original, multiple):
        _validate_world(world.episode)
        if not set(world.base_pool) <= world.episode.context:
            raise ValueError('the common base target pool is not observed')
    return multiple


def _endpoint_reuse(world: FactorialWorld) -> FactorialWorld:
    """Sixteen copies whose head-only and tail-only nodes merge across copies (``endpoint_prior.construct``, arm
    ``concentrated``: five copies share their endpoints, three keep their own, separately for heads and tails).

    Raises:
        ValueError: When the copied world fails its validation or has fewer than 30 rule or base targets.
    """
    ep = world.episode
    heads = {h for h, _, _ in world.rule_pool}
    tails = {t for _, _, t in world.rule_pool}
    order = list(range(ENDPOINT_COPIES))
    random.Random(world_first_stream(ep.seed, 'endpoint-reuse-copy-order-v1', 0)).shuffle(order)
    mapping = {(copy, node): (copy, node) for copy in order for node in range(ep.num_entities)}
    for selected, nodes in ((order[:8], heads - tails), (order[8:], tails - heads)):
        for group in (selected[:5], *[[copy] for copy in selected[5:]]):
            for node in nodes:
                for copy in group:
                    mapping[copy, node] = group[0], node
    values = sorted(set(mapping.values()))
    permuted = list(range(len(values)))
    random.Random(world_first_stream(ep.seed, 'endpoint-reuse-final-ids-v1', len(values))).shuffle(permuted)
    ids = dict(zip(values, permuted))

    def facts(original: Iterable[Triple]) -> frozenset[Triple]:
        original = tuple(original)
        return frozenset((ids[mapping[copy, h]], r, ids[mapping[copy, t]]) for copy in range(ENDPOINT_COPIES) for h, r, t in original)

    base, context = facts(ep.base), facts(ep.context)
    truth, _ = derive(base, ep.rules)
    supported, observed_proofs = derive(context, ep.rules)
    rule_pool = supported - context - base
    witnesses: dict[Triple, tuple[tuple[Triple, ...], ...]] = {q: (tuple(sorted(proof_leaves(q, context, observed_proofs))),) for q in sorted(rule_pool)}
    result_ep = replace(ep, num_entities=len(values), base=base, truth=truth, context=context, support=facts(ep.support),
                        protected=facts(ep.protected))
    _validate_world(result_ep)
    base_pool = tuple(sorted(facts(world.base_pool)))
    if not (set(base_pool) <= context & base and len(rule_pool) >= 30 and len(base_pool) >= 30):
        raise ValueError('the endpoint-reuse world needs 30 observed rule and base targets')
    return replace(world, episode=result_ep, rule_pool=tuple(sorted(rule_pool)), base_pool=base_pool, witnesses=witnesses)


def control_world(seed: int, index: int) -> FactorialWorld:
    """The control world of a sampler index (``endpoint_prior.make_world(cfg, index, 'concentrated')``).

    The research rebuilt the source world and its sparse demonstrations in three layers; they are built once here.

    Args:
        seed: The run seed.
        index: The sampler index.

    Returns:
        The world after all six layers (module docstring).
    """
    attached = _scheduled_topology(seed, index) == 'attached'
    family = _scheduled_family(seed, index)
    source = _source_world(seed, index, family)
    sparse = _add_sparse_demonstrations(source[0], source[1])
    ep = sparse[0]
    ids = random.Random(world_first_stream(ep.seed, 'factorial-final-ids-v1', 0))
    permutation = (ids.sample(range(ep.num_entities), ep.num_entities), ids.sample(range(ep.num_relations + 1), ep.num_relations + 1))
    world = _factorial_world(attached, family, sparse, permutation)
    world = _observation_dose(world, source, sparse, permutation)
    return _endpoint_reuse(_multiple_matchings(world, _remap_grids(source[1], _fact_map(*permutation))))


def control_batches(world: FactorialWorld, seed: int, index: int) -> dict[str, LegacyBatch]:
    """The two tasks of a control world (``prior_factorial.batches``): rule targets from the rule pool, base targets
    from the base pool, each task's own targets removed from its context.

    Args:
        world: The control world.
        seed: The run seed.
        index: The sampler index.

    Returns:
        ``'rule'`` and ``'base'`` with the research RecipeBatch fields.

    Raises:
        ValueError: For a pool below the batch size, or a rule target that lost every planted witness.
    """
    total = QUERIES
    ep = world.episode
    result = {}
    for task, (count, tails) in task_quotas(index, total).items():
        pool = world.rule_pool if task == 'rule' else world.base_pool
        if len(pool) < total:
            raise ValueError('insufficient distinct positives; no resampling or duplication')
        sampling_seed = world_first_stream(seed, 'factorial-sampling-' + task, (world.family, index))
        rng = random.Random(sampling_seed)
        all_groups, heads = sample_groups(rng.sample(pool, total), ep.truth, ep.num_entities, NEGATIVES, rng)
        indices = [*range(tails), *range(total // 2, total // 2 + count - tails)]
        groups, directions = all_groups[indices], heads[indices]
        positives = set(map(tuple, groups[:, 0].tolist()))
        context = ep.context - positives
        if task == 'rule' and any(not any(set(path) <= context for path in world.witnesses[q]) for q in positives):
            raise ValueError('rule training lost every planted witness')
        if positives & context or not context:
            raise ValueError('invalid task context')
        result[task] = LegacyBatch(groups, directions, context, ep.truth, ep.num_entities, ep.num_relations, sampling_seed, count)
    return result
