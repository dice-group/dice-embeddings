"""Fixed values of the synthetic prior: the resolved research recipe, the stream layout and the families' parameters.

The research code built its recipe through a chain of 12 configuration wrappers (from ``demo_context_experiment.recipe``
up to ``architecture-affinity-v1-runtime/aa_common.cfg``). ``RecipeConfig`` holds the values at the top of that chain,
``aa_common.cfg(seed, 'cpu')``, which the legacy world generator reads. The levels differ only in fields that no
world reads (updates, validation seed and interval, saving, logging, threads, device, dimension); the world, the query
and negative counts and the observation policy are the same at every level. The test compares these values with a
dump of the research chain.

The parameters of the record families are the research ``PARAMS`` dictionaries (``pf_prior2f``, ``pf_attr2``,
``pf_rec1``, ``pf_list1``) with their key order. A family spec may override entries (``FamilySpec.params``).
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

QUERIES = 30
"""Query groups per world, over all tasks (the research batch check)."""

HEAD_QUERIES = 15
"""Head-prediction query groups per world, over all tasks."""

NEGATIVES = 512
"""Negatives per query group (``pf_prior.NEG``, ``st_prior.NEG``, ``sw_prior.NEG``)."""

STAGE1_WORLDS = 20_000
"""Worlds of the stage-1 stream ``s1`` (indices 0 to 19,999; ``op_common.N1``)."""

STAGE2_OFFSET = 20_000
"""First index of the stage-2 stream ``s2`` (``op_common.S2_OFFSET``)."""

LEGACY_OFFSET = 300_000
"""Index offset of the legacy generator for a stream index (``cp_prior.OFFSET``)."""

STATISTICAL_SHARE = 0.35
"""Share of stage-2 indices that are statistical worlds (``st_prior.P_STAT``)."""

STRUCTURED_SHARE = 0.5
"""Share of the other stage-2 indices that are structured worlds (``sw_prior.V2_SHARE``)."""

CONSTANT_UPDATES = 20_000
"""Updates at the constant learning rate (``op_common.N1``)."""

DECAY_UPDATES = 4_000
"""Updates of the linear decay to zero after them (``op_common.N2``)."""

TOTAL_UPDATES = CONSTANT_UPDATES + DECAY_UPDATES
"""Updates of a run, and the length of its world order."""


@dataclass(frozen=True)
class WorldConfig:
    """Rule-world sizes of the legacy generator (``world_first.WorldConfig``, default values)."""

    min_core_entities: int = 96
    max_core_entities: int = 192
    min_relations: int = 5
    max_relations: int = 10
    min_density: float = 0.001
    max_density: float = 0.003
    min_observed: float = 0.4
    max_observed: float = 0.9
    blocks_per_rule: int = 6
    grid_size: int = 3
    max_copies: int = 2
    mask_keep: float = 0.5


@dataclass(frozen=True)
class RecipeConfig:
    """``standard_recipe.RecipeConfig`` with the values of the top of the research configuration chain.

    The fields after ``num_negative`` describe the research training of the chain's last experiment. The pretraining
    of the arms has its own recipe (one AdamW step per world, microbatch 2, 24,000 updates); no ported component reads
    those fields.
    """

    model: str = 'CountULTRA'
    relation_weighting: str = 'normalized'
    world: WorldConfig = field(default_factory=WorldConfig)
    positive_pool: str = 'observed'
    observation: str = 'supported'
    batch_size: int = QUERIES
    num_negative: int = NEGATIVES
    adversarial_temperature: float = 1.0
    microbatch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: Optional[float] = None
    dim: int = 64
    layers: int = 6
    seed: int = 42
    updates: int = 12_000
    validation_seed: int = 20_263_000
    validation_worlds: int = 1
    validation_queries: int = 4
    validation_every: int = 1_000
    save_every: int = 50
    log_every: int = 25
    device: str = 'cpu'
    threads: int = 1


def recipe_config(seed: int) -> RecipeConfig:
    """The recipe that the legacy generator of a run with this seed reads (``aa_common.cfg(seed, 'cpu')``).

    Args:
        seed: The run seed.

    Returns:
        The resolved research recipe.
    """
    return RecipeConfig(seed=seed)


def _frozen(values: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(values)


CATEGORICAL2_PARAMS: Mapping[str, Any] = _frozen(dict(   # pf_prior2f.PARAMS (prior-fix-v2f, fitted 12 October 2026)
    max_entities=5000,            # hard limit of a world (subjects + values)
    subjects=(400, 4000),         # log-uniform
    groups=(8, 0.3),              # G ~ LogUniform(groups[0], groups[1] * subjects); a first entry below 1 is a share
    group_zipf=(0.0, 0.8),        # exponent of the group sizes
    types=(2, 8),                 # number of subject types, uniform integer
    typed=0.5,                    # P(attribute applies to one type only)
    group_domain=1.0,             # P(only some groups of the domain carry the attribute)
    group_rate=(0.02, 0.15),      # P(a group carries the attribute), one draw per attribute
    full_coverage=0.0,            # P(the attribute has full coverage inside its domain)
    coverage=(0.75, 0.95),        # coverage inside the domain otherwise
    type_relation=0.0,            # P(a world with 2 or more types has a class relation)
    type_coverage=(0.5, 1.0),     # share of the subjects that carry their class fact
    attributes=(12, 50),          # log-uniform, rounded
    vocabulary=(10, 3500),        # values per attribute, log-uniform (at most the subjects of the domain)
    value_zipf=(0.2, 1.0),        # exponent of the value popularity
    reliability=(0.1, 0.7),       # P(subject takes its group's value), functional attributes
    functional=0.4,               # P(attribute is functional)
    extra=(1.5, 4.0),             # Poisson mean of the further values in a group's set
    set_reliability=(0.1, 0.45),  # P(subject takes a value of its group's set)
    identity=0.15,                # P(attribute names the group: membership)
    identity_reliability=(0.8, 1.0),
    twin=0.14,                    # P(a categorical attribute has a redundant twin relation)
    twin_rare=0.9,                # P(twin) of a specific attribute
    twin_uses=4,
    twin_share=(0.3, 0.9),        # share of the attribute's facts that the twin repeats
    links=(0, 2),                 # number of subject-subject relations, uniform integer
    link_degree=(0.2, 1.5),       # edges per subject of a link relation
    homophily=(0.3, 0.8),         # P(link stays inside the group)
    symmetric=0.6,                # P(a link relation is symmetric)
    observed=(0.6, 0.95),         # share of observed facts
))

ATTR2_PARAMS: Mapping[str, Any] = _frozen(dict(          # pf_attr2.PARAMS (attr-v2): categorical2's plus four entries
    CATEGORICAL2_PARAMS,
    null=0.2,                     # P(null world: every fidelity is 0)
    strength=(0.6, 1.0),          # world strength s of the other worlds
    spread=0.15,                  # fidelity ~ U(s - spread, s + spread), clipped to [0, 1]
    chain=0.3,                    # P(the world has one value chain relation)
))

REC1_PARAMS: Mapping[str, Any] = _frozen(dict(           # pf_rec1.PARAMS (rec-v1)
    max_entities=5000,            # hard limit of a world (rows + values)
    rows=(300, 2500),             # rows of all tables together, log-uniform
    tables=(1, 3),                # tables, uniform integer
    depth=(1, 3),                 # taxonomy levels of a table, uniform integer
    top=(2, 30),                  # clusters at level 1, log-uniform
    branch=(2, 8),                # children per cluster and level, log-uniform
    leaf_zipf=(0.0, 1.0),         # exponent of the leaf sizes
    columns=(2, 10),              # value columns per table, log-uniform, rounded
    key=0.25,                     # P(column is a key of its level)
    vocabulary=(3, 5000),         # values of a property column, log-uniform, at most vocab_rows x the table's rows
    vocab_rows=3.0,               # upper end of the vocabulary as a multiple of the table's rows
    value_zipf=(0.0, 1.2),        # exponent of the value popularity
    functional=0.6,               # P(column is single-valued)
    extra=(0.3, 1.5),             # multi-valued columns: 1 + Poisson(extra) values per row
    coverage=(0.6, 1.0),          # share of the table's rows with a value in the column
    shared=0.15,                  # P(property column reuses the value entities of an earlier property column)
    chain=0.3,                    # P(value chain) per pair (finer key, coarser single-valued column)
    chains_max=3,                 # at most this many chains per table
    reference=0.7,                # P(a table after the first references an earlier table)
    inherit=0.5,                  # P(inherited column) per single-valued column of the referenced table, at most 2
    null=0.2,                     # P(null world: every fidelity is 0)
    weak=0.2,                     # P(weak world)
    weak_strength=(0.3, 0.6),     # world strength s of a weak world
    strength=(0.6, 1.0),          # world strength s of a strong world
    spread=0.15,                  # rho ~ U(s - spread, s + spread), clipped to [0, 1]
    free=0.2,                     # P(column of a non-null world ignores the clusters)
    observed=(0.6, 0.95),         # share of observed facts
))

LIST1_PARAMS: Mapping[str, Any] = _frozen(dict(          # pf_list1.PARAMS (list-v2)
    max_entities=20000,           # hard limit of a world (anchors + items + values)
    max_facts=120000,             # hard limit of the facts of a world
    anchors=(200, 3000),          # anchors, log-uniform
    kinds=(1, 3),                 # item kinds, uniform integer
    communities=(3, 80),          # latent communities, log-uniform (at most anchors / 8)
    cover=(1.2, 3.0),             # items per community of a kind / the largest mean list size on the kind, uniform
    community_zipf=(0.0, 1.0),    # exponent of the community sizes
    memberships=(1, 2),           # communities per anchor, uniform integer; Dirichlet(1) weights
    item_zipf=(0.5, 1.5),         # exponent of the item popularity inside a community
    lists=(1, 3),                 # list relations, uniform integer
    list_mean=(1.5, 12.0),        # mean list size of a list relation, log-uniform
    list_spread=(0.5, 1.2),       # sigma of the log-normal list size
    reverse=0.0,                  # P(a list relation is stored item -> anchor)
    noise_pop=1.0,                # P(a noise item of a list is drawn by the kind's popularity), else uniformly
    hubs=0.4,                     # P(an item kind has global hub items)
    hub_count=(1, 10),            # hub items of such a kind, log-uniform
    hub_rate=(0.2, 0.8),          # P(an anchor's list holds a hub item), one draw per hub and list relation
    attributes=(0, 3),            # single-valued attributes per item kind, uniform integer
    anchor_attributes=(0, 2),     # single-valued attributes of the anchors, uniform integer
    rich=0.1,                     # P(attribute-rich world: the two ranges below replace the two above)
    rich_attributes=(2, 8),       # attributes per item kind of a rich world
    rich_anchor_attributes=(1, 6),  # anchor attributes of a rich world
    vocabulary=(3, 200),          # values of an attribute, log-uniform (at most half the subjects)
    value_zipf=(0.0, 1.0),        # exponent of the value popularity
    shared_values=0.2,            # P(an anchor attribute reuses the value entities of an item attribute)
    item_links=0.2,               # P(an item kind has an item-item link relation)
    link_degree=(0.3, 1.5),       # links per item
    homophily=(0.5, 0.95),        # P(a link stays inside the community)
    coverage=(0.6, 1.0),          # share of the subjects with a value in an attribute
    null=0.2,                     # P(null world: every fidelity is 0)
    weak=0.2,                     # P(weak world)
    weak_strength=(0.3, 0.6),     # world strength s of a weak world
    strength=(0.8, 1.0),          # world strength s of a strong world
    spread=0.1,                   # rho ~ U(s - spread, s + spread), clipped to [0, 1]
    free=0.1,                     # P(relation of a non-null world ignores the communities)
    observed=(0.6, 0.95),         # share of observed facts
))

RECORD_PARAMS: Mapping[str, Mapping[str, Any]] = _frozen(dict(
    categorical2=CATEGORICAL2_PARAMS, attr2=ATTR2_PARAMS, rec1=REC1_PARAMS, list1=LIST1_PARAMS))
"""The record families' default parameters, by generator name."""

REAL_PARAMS: Mapping[str, Any] = _frozen(dict(  # pt_real (realworlds-v1 with realworlds-v3's size cap)
    max_entities=20000,           # RW_MAXN: the largest entity target of a subgraph, log-uniform from 400
))

REAL_SOURCE_SHA256: Mapping[str, str] = _frozen({   # pcu-training-20261026/plan.json; rw_prior.SOURCES order
    'FB15k-237': '653525529c02f9b7d88ec66702a27a128024693dbc0ffb0c2336576122ddb1ef',
    'WN18RR': '19d9dc38e66247a7e6680801eff6c45564efa5b22a49975eff8d64ecac17a4e7',
    'CoDEx-M': 'aebdfbc3130be64766dc5714ac048d84f985898957c6daae9b46bc29a0d0e218',
})
"""The pinned SHA-256 of each real source graph's ``data.pt`` (the train splits are read); a world draws its source
from these names in this order."""

GENERATOR_PARAMS: Mapping[str, Mapping[str, Any]] = _frozen(dict(RECORD_PARAMS, real=REAL_PARAMS))
"""The default parameters of every hook generator."""


def record_params(generator: str, overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """A hook generator's parameters with overrides; JSON lists become tuples, as the research hooks did.

    Args:
        generator: ``'categorical2'``, ``'attr2'``, ``'rec1'``, ``'list1'`` or ``'real'``.
        overrides: Entries that replace defaults; every key must exist.

    Returns:
        A new dictionary in the default key order.

    Raises:
        ValueError: For an unknown generator or override key.
    """
    if generator not in GENERATOR_PARAMS:
        raise ValueError(f'unknown hook generator {generator!r}')
    values = dict(GENERATOR_PARAMS[generator])
    for key, value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f'{generator} has no parameter {key!r}')
        values[key] = tuple(value) if isinstance(value, list) else value
    return values
