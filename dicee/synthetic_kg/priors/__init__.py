"""The synthetic prior: its configuration, the world order of a run, and the world of an order slot.

A prior is a mixture of world families. Update ``i`` of a run draws its family from the shares
(``streams.mixture_draw``, families sorted by name) and takes the next world of that family's stream:

- ``plain``: stage-1 index ``k`` (source ``'s1'``) for its first ``STAGE1_WORLDS`` worlds, then the plain indices of
  the stage-2 stream;
- ``structured`` and ``statistical``: the stage-2 indices from ``STAGE2_OFFSET`` whose family by the generators' own
  rule (``streams.stage2_family``) matches (source ``'s2'``);
- a hook family (``FamilySpec``): ``first``, ``first + 1``, ... (source ``'hook'``).

Both kinds of stream pass over the reserved index ranges, which held-out worlds and generator fitting used. The order
is prefix-stable: slot ``i`` does not depend on the run length. ``order_sha256`` pins it.
"""
import hashlib
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence

from ..batch import World, WorldBatch, check_batch, world_sizes
from ..recipe import GENERATOR_PARAMS, RECORD_PARAMS, STAGE1_WORLDS, STAGE2_OFFSET, TOTAL_UPDATES
from ..streams import mixture_draw, stage2_family

BUILTIN_FAMILIES = ('plain', 'structured', 'statistical')
"""Families of the built-in streams; every other family of a mixture needs a ``FamilySpec``."""

GENERATORS = (*RECORD_PARAMS, 'real')
"""Generators of hook families: the record families and the real subgraph worlds."""

RESERVED_STAGE2 = ((90_000, 90_048), (120_000, 160_000))
"""Stage-2 index ranges that no family stream takes (held-out and fitting worlds; ``pt_common.RESERVED_S2``)."""

RESERVED_HOOK = ((130_000, 160_000), (500_000, 501_000))
"""Hook index ranges that no family stream takes (fitting and gate worlds; ``pt_common.RESERVED_HOOK``)."""


class Slot(NamedTuple):
    """The world of one update: its family, its stream source (``'s1'``, ``'s2'`` or ``'hook'``) and stream index."""

    family: str
    source: str
    index: int  # type: ignore[assignment]  # the stream index; the field shadows tuple.index


@dataclass(frozen=True)
class FamilySpec:
    """A hook family of a mixture.

    Attributes:
        name: The family name in the mixture.
        generator: The world generator, one of ``GENERATORS``.
        first: The first index of the family's stream.
        params: Overrides of the generator's parameters as ``(key, value)`` pairs (``recipe.record_params``; the real
            worlds take ``max_entities``).
        null_swap_budget: The structural null of arm N: the swap budget of each world's truth, or ``None`` for intact
            worlds.
    """

    name: str
    generator: str
    first: int
    params: tuple[tuple[str, Any], ...] = ()
    null_swap_budget: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name or self.name in BUILTIN_FAMILIES:
            raise ValueError(f'a hook family needs a name other than {BUILTIN_FAMILIES}')
        if self.generator not in GENERATORS:
            raise ValueError(f'unknown generator {self.generator!r}; expected one of {GENERATORS}')
        if self.first < 0:
            raise ValueError('the first stream index must be nonnegative')
        keys = [key for key, _ in self.params]
        if len(set(keys)) != len(keys):
            raise ValueError(f'{self.name}: repeated parameter')
        unknown = sorted(set(keys) - set(GENERATOR_PARAMS[self.generator]))
        if unknown:
            raise ValueError(f'{self.name}: unknown parameters {unknown}')
        if self.null_swap_budget is not None and self.generator not in RECORD_PARAMS:
            raise ValueError('the structural null applies to the record families only')
        if self.null_swap_budget is not None and self.null_swap_budget < 1:
            raise ValueError('the null swap budget must be positive')

    @property
    def overrides(self) -> dict[str, Any]:
        """The parameter overrides as a dictionary."""
        return dict(self.params)


def parse_mix(text: str) -> dict[str, float]:
    """Normalized family shares of a mixture string ``'name=share,...'``, sorted by family name (``pt_common.parse_mix``).

    The total is a plain left-to-right float sum, as on the Python of the research runs.

    Args:
        text: The mixture, for example ``'plain=0.5,rec1=0.5'``.

    Returns:
        Family name to share, positive shares only.

    Raises:
        ValueError: For a malformed, repeated or negative entry, or no positive share.
    """
    shares: dict[str, float] = {}
    for part in text.split(','):
        name, _, number = part.partition('=')
        name = name.strip()
        if not name or not number or name in shares:
            raise ValueError(f'bad or repeated family in the mixture: {part!r}')
        shares[name] = float(number)
        if shares[name] < 0:
            raise ValueError(f'negative share: {part!r}')
    shares = {k: v for k, v in shares.items() if v > 0}
    total = 0.0
    for share in shares.values():
        total += share
    if not total > 0:
        raise ValueError('the mixture needs a positive share')
    return {k: shares[k] / total for k in sorted(shares)}


@dataclass(frozen=True)
class PriorConfig:
    """A mixture of world families.

    Attributes:
        mix: The mixture string, verbatim (``parse_mix``).
        order_name: The name that seeds the family draws; priors with the same mixture and order name give the same order.
        families: The hook families of the mixture.
    """

    mix: str
    order_name: str
    families: tuple[FamilySpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.order_name:
            raise ValueError('a prior needs an order name')
        shares = parse_mix(self.mix)
        names = [spec.name for spec in self.families]
        if len(set(names)) != len(names):
            raise ValueError('repeated family spec')
        missing = sorted(set(shares) - set(BUILTIN_FAMILIES) - set(names))
        unused = sorted(set(names) - set(shares))
        if missing or unused:
            raise ValueError(f'families without a spec: {missing}; specs outside the mixture: {unused}')

    @property
    def shares(self) -> dict[str, float]:
        """The normalized shares, sorted by family name."""
        return parse_mix(self.mix)

    def family(self, name: str) -> Optional[FamilySpec]:
        """The spec of a hook family, or ``None`` for a built-in family."""
        return next((spec for spec in self.families if spec.name == name), None)


def mixture_bounds(prior: PriorConfig) -> list[tuple[str, float]]:
    """The families sorted by name with the cumulative shares that bound their draws (plain float sums in order).

    Args:
        prior: The prior.

    Returns:
        ``(family, upper bound)`` pairs; a draw ``u`` picks the first family with ``u < bound``, else the last family.
    """
    shares = prior.shares
    bounds, cumulative = [], 0.0
    for name in sorted(shares):
        cumulative += shares[name]
        bounds.append((name, cumulative))
    return bounds


def family_sequence(seed: int, prior: PriorConfig, total: int = TOTAL_UPDATES) -> list[str]:
    """The family of each of the first ``total`` updates (``pt_common.family_sequence``).

    Args:
        seed: The run seed.
        prior: The prior.
        total: The number of updates.

    Returns:
        One family name per update.
    """
    bounds = mixture_bounds(prior)
    last = bounds[-1][0]
    out = []
    for i in range(total):
        u = mixture_draw(seed, prior.order_name, i)
        out.append(next((name for name, bound in bounds if u < bound), last))
    return out


def reserved(index: int, ranges: Sequence[tuple[int, int]]) -> bool:
    """Whether a stream index lies in one of the half-open ranges."""
    return any(low <= index < high for low, high in ranges)


def stage2_indices(seed: int, family: str, count: int) -> list[int]:
    """The first ``count`` stage-2 indices of a family, outside ``RESERVED_STAGE2`` (``pt_common.stage2_indices``).

    Args:
        seed: The run seed.
        family: ``'plain'``, ``'structured'`` or ``'statistical'``.
        count: The number of indices.

    Returns:
        Increasing stage-2 indices from ``STAGE2_OFFSET``.
    """
    out: list[int] = []
    index = STAGE2_OFFSET
    while len(out) < count:
        if not reserved(index, RESERVED_STAGE2) and stage2_family(seed, index) == family:
            out.append(index)
        index += 1
    return out


def hook_indices(first: int, count: int) -> list[int]:
    """The first ``count`` indices of a hook family's stream, outside ``RESERVED_HOOK`` (``pt_common.hook_indices``).

    Args:
        first: The family's first stream index.
        count: The number of indices.

    Returns:
        Increasing indices from ``first``.
    """
    out: list[int] = []
    index = first
    while len(out) < count:
        if not reserved(index, RESERVED_HOOK):
            out.append(index)
        index += 1
    return out


def build_order(seed: int, prior: PriorConfig, total: int = TOTAL_UPDATES) -> list[Slot]:
    """The world of each of the first ``total`` updates of a run (``pt_common.build_order``).

    Slot ``i`` depends on the seed, the prior and ``i`` only: the order of a longer run starts with the order of a
    shorter one.

    Args:
        seed: The run seed.
        prior: The prior.
        total: The number of updates.

    Returns:
        One ``Slot`` per update.
    """
    families = family_sequence(seed, prior, total)
    streams: dict[str, list[tuple[str, int]]] = {}
    for name in prior.shares:
        count = families.count(name)
        spec = prior.family(name)
        if spec is not None:
            streams[name] = [('hook', i) for i in hook_indices(spec.first, count)]
        elif name == 'plain':
            extra = stage2_indices(seed, 'plain', count - STAGE1_WORLDS) if count > STAGE1_WORLDS else []
            streams[name] = [('s1', k) for k in range(min(count, STAGE1_WORLDS))] + [('s2', i) for i in extra]
        else:
            streams[name] = [('s2', i) for i in stage2_indices(seed, name, count)]
    position = dict.fromkeys(streams, 0)
    out = []
    for name in families:
        source, index = streams[name][position[name]]
        position[name] += 1
        out.append(Slot(name, source, index))
    return out


def order_sha256(order: Sequence[Sequence[Any]]) -> str:
    """SHA-256 of the lines ``'<family> <source> <index>'`` joined by newlines (``pt_common.order_sha256``).

    Args:
        order: Slots, or ``(family, source, index)`` triples.

    Returns:
        The hex digest.
    """
    return hashlib.sha256('\n'.join(f'{f} {s} {i}' for f, s, i in order).encode()).hexdigest()


def world(seed: int, slot: Slot, prior: PriorConfig) -> World:
    """The world of an order slot, checked (the research worker's ``pt_worlds.world`` and ``check_batches``).

    The world depends on the seed, the slot and the family's spec only; generators draw from their own seeded streams
    and leave every global random state unchanged. A generator error propagates: a run stops on it.

    Args:
        seed: The run seed.
        slot: A slot of ``build_order(seed, prior)``.
        prior: The prior of the order.

    Returns:
        The world; ``meta['family']`` is the slot's family, and hook families keep the generator's own family name as
        ``meta['generator_family']``. ``meta['recipe_batch']`` holds per task the research RecipeBatch fields of the
        legacy template that no update reads (``sampling_seed``, ``requested_positives``).

    Raises:
        ValueError: For a slot that does not belong to the prior, or a world that fails ``check_batch`` (a generator's
            own errors propagate too; the research generators never resample a world).
    """
    spec = prior.family(slot.family)
    sources = ('hook',) if spec is not None else ('s1', 's2') if slot.family == 'plain' else ('s2',)
    stage1 = slot.source == 's1' and slot.index >= STAGE1_WORLDS
    if slot.family not in prior.shares or slot.source not in sources or slot.index < 0 or stage1 or (slot.source == 's2' and slot.index < STAGE2_OFFSET):
        raise ValueError(f'slot {tuple(slot)} does not belong to this prior')
    batch, meta = _generate(seed, slot, spec)
    check_batch(batch)
    if spec is not None:
        meta = dict(meta, generator_family=meta.get('family'), family=slot.family)
    elif meta.get('family') != slot.family:
        raise ValueError(f'slot {tuple(slot)} gave a {meta.get("family")} world')
    return World(batch, {**meta, **world_sizes(batch)})


def _generate(seed: int, slot: Slot, spec: Optional[FamilySpec]) -> tuple[dict[str, WorldBatch], dict[str, Any]]:
    """Dispatch to the family generator: ``(task name -> WorldBatch, generator metadata)``.

    The generators live in the modules of this package, imported here at call time: ``legacy`` (stage 1, ``'s1'``),
    ``stage2`` (``'s2'``: structured, statistical and stage-2 plain worlds), ``records`` (categorical2, rec1, attr2,
    list1, with the structural null of ``null``) and ``real`` (real subgraph worlds).
    """
    if spec is None:
        if slot.source == 's1':
            from .legacy import plain
            return plain(seed, slot.index)
        from .stage2 import stage2_world
        return stage2_world(seed, slot.index)
    if spec.generator == 'real':
        from .real import real_world
        return real_world(seed, slot.index, spec)
    from .records import record_world
    return record_world(seed, slot.index, spec)

