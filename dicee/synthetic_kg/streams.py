"""Seeded streams of the synthetic prior, with the exact version strings of the research code.

A stream maps a text key to an integer: the first 8 bytes of the key's SHA-256, big-endian, unless its function says
otherwise. The version strings are part of the keys. They name the research generators and must not change, because
they fix the world order, the worlds and the initial weights of every run. Each stream kind has one function here;
generators never hash keys themselves.
"""
import hashlib
from typing import Literal

from .recipe import STATISTICAL_SHARE, STRUCTURED_SHARE

MIXTURE_VERSION = 'prior-fix-v1'
"""``pt_common.VERSION``: the family draw of every update."""

CAPACITY_VERSION = 'capacity-prior-v1'
"""``cp_prior.VERSION``: the capacity-prior streams, among them the initial weights."""

STATISTICAL_VERSION = 'statworlds-v1'
"""``st_prior.VERSION``: statistical stage-2 worlds and the stage-2 routing."""

STRUCTURED_VERSION = 'structured-worlds-v1'
"""``sw_prior.VERSION``: structured stage-2 worlds and the stage-2 routing."""


def sha256_int(key: str, byteorder: Literal['big', 'little'] = 'big') -> int:
    """The first 8 bytes of SHA-256 of ``key`` (UTF-8) as an unsigned integer.

    Args:
        key: The stream key, version string included.
        byteorder: Byte order of the 8 bytes; every stream here is big-endian.

    Returns:
        An integer in ``[0, 2**64)``.
    """
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], byteorder)


def world_stream(version: str, seed: int, index: int, name: str) -> int:
    """The per-world stream ``'<version>:<seed>:<index>:<name>'`` of a world generator.

    Args:
        version: The generator's research version string.
        seed: The run seed.
        index: The world's stream index.
        name: The purpose of the stream inside the generator.

    Returns:
        The stream integer.
    """
    return sha256_int(f'{version}:{seed}:{index}:{name}')


# ----------------------------------------------------------------------------------------------- world order

def mixture_draw(seed: int, order_name: str, index: int) -> float:
    """The uniform number in ``[0, 1)`` that picks the family of update ``index`` (``pt_common.draw``).

    Args:
        seed: The run seed.
        order_name: The order name of the prior.
        index: The 0-based update.

    Returns:
        The stream integer divided by ``2**64``.
    """
    return sha256_int(f'{MIXTURE_VERSION}:mix:{seed}:{order_name}:{index}') / 2 ** 64


# ----------------------------------------------------------------------------------------------- initial weights

def capacity_stream(seed: int, name: str) -> int:
    """The capacity-prior stream ``'capacity-prior-v1:<seed>:<name>'`` (``cp_prior.stream``).

    Args:
        seed: The run seed.
        name: The purpose of the stream.

    Returns:
        The stream integer.
    """
    return sha256_int(f'{CAPACITY_VERSION}:{seed}:{name}')


def initial_seed(seed: int, width: str = 'd64') -> int:
    """The ``torch.manual_seed`` value of a run's initial weights (``cp_common.create``).

    Args:
        seed: The run seed.
        width: The model width tag of the research arms; every ported arm is ``'d64'``.

    Returns:
        ``capacity_stream(seed, 'initial:<width>') % (2**63 - 1)``.
    """
    return capacity_stream(seed, f'initial:{width}') % (2 ** 63 - 1)


# ----------------------------------------------------------------------------------------------- stage-2 routing

def statistical_stream(seed: int, index: int, name: str) -> int:
    """A stream of the statistical generator (``st_prior.stream``).

    Args:
        seed: The run seed.
        index: The stage-2 stream index.
        name: The purpose of the stream.

    Returns:
        The stream integer.
    """
    return world_stream(STATISTICAL_VERSION, seed, index, name)


def structured_stream(seed: int, index: int, name: str) -> int:
    """A stream of the structured generator (``sw_prior.stream``).

    Args:
        seed: The run seed.
        index: The stage-2 stream index.
        name: The purpose of the stream.

    Returns:
        The stream integer.
    """
    return world_stream(STRUCTURED_VERSION, seed, index, name)


def is_statistical(seed: int, index: int) -> bool:
    """Whether stage-2 index ``index`` is a statistical world (``st_prior.is_statistical``)."""
    return statistical_stream(seed, index, 'family') / 2 ** 64 < STATISTICAL_SHARE


def is_structured(seed: int, index: int) -> bool:
    """Whether a non-statistical stage-2 index is a structured world (``sw_prior.is_structured``)."""
    return structured_stream(seed, index, 'family') / 2 ** 64 < STRUCTURED_SHARE


def stage2_family(seed: int, index: int) -> str:
    """The family of a stage-2 index by the generators' own rule (``op_worlds.family`` for source ``'s2'``).

    Args:
        seed: The run seed.
        index: The stage-2 stream index.

    Returns:
        ``'statistical'``, ``'structured'`` or ``'plain'``.
    """
    if is_statistical(seed, index):
        return 'statistical'
    return 'structured' if is_structured(seed, index) else 'plain'


# ----------------------------------------------------------------------------------------------- legacy generator
# The legacy generator has streams of three layouts, two of them with an 8-byte BLAKE2b digest.

WORLD_FIRST_VERSION = 'world-first-v1'
"""The prefix of ``world_first.stream`` (world-first version 1): the control worlds of the legacy generator."""

JOINT_WORLDS_STREAM_VERSION = 'joint-worlds-v1'
"""``worlds.STREAM_VERSION`` of joint-worlds-v2: the rule worlds, their batches and the source schedule."""

SOURCE_SCHEDULE_VERSION = 'joint-retention-v1'
"""``joint-scaling common.SCHEDULE_VERSION``: which legacy indices are control worlds."""

PAIR_SLOT_VERSION = 'pair-affinity-training-v1'
"""``pair_batches.VERSION``: which rule worlds get the pair-affinity rewiring."""

OBSERVATION_MASK_VERSION = 'role-observation-v1'
"""``oc_common.VERSION``: the targeted observation masks of the legacy batches."""

RULE_PROGRAM_VERSION = 'rule-program-v1'
"""``program.VERSION``: which rule worlds get new rule programs, and the programs."""

PAIR_AFFINITY_VERSION = 'pair-affinity-v1'
"""``affinity.VERSION`` (pair-affinity-v2 runtime): the latent pair affinities and the rewiring."""


def blake2b_int(key: str) -> int:
    """The 8-byte BLAKE2b digest of ``key`` (UTF-8) as a big-endian unsigned integer.

    Args:
        key: The stream key, version string included.

    Returns:
        An integer in ``[0, 2**64)``.
    """
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'big')


def world_first_stream(seed: int, name: str, value: object) -> int:
    """A stream of the control worlds: ``'world-first-v1:<seed>:<name>:<value>'`` with BLAKE2b (``world_first.stream``).

    Args:
        seed: A run seed or a world's own seed.
        name: The purpose of the stream.
        value: Formatted with ``str``, so a tuple appears in Python's notation, for example ``('train', 3)``.

    Returns:
        The stream integer.
    """
    return blake2b_int(f'{WORLD_FIRST_VERSION}:{seed}:{name}:{value}')


def joint_world_stream(seed: int, name: str) -> int:
    """A stream of the rule worlds: ``'joint-worlds-v1:<seed>:<name>'`` with BLAKE2b (joint-worlds-v2 ``worlds.stream``).

    Args:
        seed: A run seed, a world seed or a batch's sampling seed.
        name: The purpose of the stream.

    Returns:
        The stream integer.
    """
    return blake2b_int(f'{JOINT_WORLDS_STREAM_VERSION}:{seed}:{name}')


def source_schedule_stream(seed: int, block: int) -> int:
    """The shuffle of the source schedule's block of five legacy indices (``joint-scaling common.source_arm``)."""
    return joint_world_stream(seed, f'{SOURCE_SCHEDULE_VERSION}:block:{block}')


def pair_slot_stream(seed: int, index: int) -> int:
    """The rank of a rule-world index among the candidates of the pair-affinity rewiring (``pair_batches.selected``)."""
    return joint_world_stream(seed, f'{PAIR_SLOT_VERSION}:slot:{index}')


def observation_mask_stream(sampling_seed: int) -> int:
    """The anchor order of the targeted observation mask of a legacy batch (``oc_prior.masks``)."""
    return joint_world_stream(sampling_seed, f'{OBSERVATION_MASK_VERSION}:targeted')


def rule_program_stream(seed: int, name: str) -> int:
    """A rule-program stream ``'rule-program-v1:<seed>:<name>'`` with SHA-256 (``program.seed_for``)."""
    return sha256_int(f'{RULE_PROGRAM_VERSION}:{seed}:{name}')


def pair_affinity_stream(seed: int, name: str) -> int:
    """A pair-affinity stream ``'pair-affinity-v1:<seed>:<name>'`` with SHA-256 (``affinity.seed_for``)."""
    return sha256_int(f'{PAIR_AFFINITY_VERSION}:{seed}:{name}')
