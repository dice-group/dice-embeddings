"""Exact incidence-count and entity-pair weights for ULTRA's relation graph (pure PyTorch, CPU).

ULTRA's relation graph links two relations when they share an entity in a role pair: head-head, tail-tail, head-tail
or tail-head (edge types 0 to 3, ``INCIDENCE_PAIRS``). Official ULTRA uses only the binary support of these links.
``relation_count_weights`` gives each link the normalized count ``sqrt(c^2 / (d_a d_b))``, where ``c`` counts the
entities that carry both relations in their roles and ``d_a``, ``d_b`` count the entities of each relation in its role.
The value is formed from the reduced exact fraction, so equal fractions give equal float64 weights.
``pair_channel_weights`` gives the two entity-pair channels (edge types 4 and 5): how many entity pairs carry both
relations in the same orientation, or in the reverse one, divided by ``sqrt(deg_a deg_b)`` of the relations' triple
counts.

Every function takes the direct observed triples ``[E, 3]`` (CPU, int64) with relation ids below ``num_relations``.
Technical inverses ``(t, r + R, h)`` are added inside, so relation ids of the results run over ``2R``. No truth,
label or rule enters. Training and inference both build their weights here.
"""
import torch

INCIDENCE_PAIRS = ((0, 0), (1, 1), (0, 1), (1, 0))
"""Role pairs (0 = head, 1 = tail) of ULTRA's relation-graph edge types 0 to 3: hh, tt, ht, th."""

PAIR_EDGE_TYPES = (4, 5)
"""Relation-graph edge types of the same-orientation and the reverse-orientation entity-pair channel."""


def _with_inverses(direct: torch.Tensor, num_entities: int, num_relations: int) -> torch.Tensor:
    """Unique direct triples followed by their technical inverses, after the input checks."""
    if min(num_entities, num_relations) < 1 or num_entities ** 2 > torch.iinfo(torch.int64).max:
        raise ValueError('Require positive graph sizes with safe exact int64 products')
    if direct.device.type != 'cpu' or direct.dtype != torch.long or direct.ndim != 2 or direct.shape[1] != 3:
        raise ValueError('Require a CPU int64 tensor of direct triples with shape [E, 3]')
    if direct.numel() and (direct.min() < 0 or direct[:, (0, 2)].max() >= num_entities or direct[:, 1].max() >= num_relations):
        raise ValueError('Direct triples must use entity and relation ids within bounds')
    unique = direct.unique(dim=0)
    inverse = unique[:, [2, 1, 0]].clone()
    inverse[:, 1] += num_relations
    return torch.cat((unique, inverse))


def incidence_counts(direct: torch.Tensor, num_entities: int, num_relations: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Incidence co-occurrence counts and role degrees of every relation, technical inverses included.

    An entity contributes once to a relation's role, however many neighbours it has under that relation.

    Args:
        direct: Direct observed triples, CPU int64 ``[E, 3]``.
        num_entities: Number of entities.
        num_relations: Number of direct relations ``R``.

    Returns:
        ``counts`` int64 ``[4, 2R, 2R]`` (``counts[k, a, b]``: entities in role ``INCIDENCE_PAIRS[k][0]`` of relation
        ``a`` and role ``INCIDENCE_PAIRS[k][1]`` of relation ``b``) and ``degrees`` int64 ``[2, 2R]`` (entities per role).
    """
    triples = _with_inverses(direct, num_entities, num_relations)
    incidence = torch.zeros((2, num_entities, 2 * num_relations), dtype=torch.float64)
    for side, column in enumerate((0, 2)):
        incidence[side, triples[:, column], triples[:, 1]] = 1
    # Float64 products of 0/1 matrices are exact: every partial sum is an integer of at most num_entities < 2^53.
    counts = torch.stack([incidence[a].T @ incidence[b] for a, b in INCIDENCE_PAIRS]).long()
    return counts, incidence.sum(1).long()


def relation_count_weights(direct: torch.Tensor, num_entities: int, num_relations: int) -> torch.Tensor:
    """Normalized incidence counts ``sqrt(c^2 / (d_a d_b))`` of all relation pairs, from the reduced exact fraction.

    Args:
        direct: Direct observed triples, CPU int64 ``[E, 3]``.
        num_entities: Number of entities.
        num_relations: Number of direct relations ``R``.

    Returns:
        Float64 ``[4, 2R, 2R]``, indexed by edge type, then the two relations of an edge.
    """
    counts, degrees = incidence_counts(direct, num_entities, num_relations)
    denominator = torch.stack([degrees[a, :, None] * degrees[b, None, :] for a, b in INCIDENCE_PAIRS]).clamp_min(1)
    numerator = counts.square()
    common = torch.gcd(numerator, denominator)
    reduced_numerator: torch.Tensor = numerator // common
    reduced_denominator: torch.Tensor = denominator // common
    return (reduced_numerator.double() / reduced_denominator.double()).sqrt()


def pair_counts(direct: torch.Tensor, num_entities: int, num_relations: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Entity-pair co-occurrence counts of all relation pairs, in O(E + R^2) space instead of O(N^2).

    The trivial matches of a relation with itself (same orientation) and with its own technical inverse (reverse
    orientation) are removed.

    Args:
        direct: Direct observed triples, CPU int64 ``[E, 3]``.
        num_entities: Number of entities.
        num_relations: Number of direct relations ``R``.

    Returns:
        ``counts`` int64 ``[2, 2R, 2R]`` (0: pairs ``(h, t)`` under both relations; 1: ``(h, t)`` under the first and
        ``(t, h)`` under the second) and ``degree`` int64 ``[2R]`` (triples per relation).
    """
    triples = _with_inverses(direct, num_entities, num_relations)
    relations = 2 * num_relations
    degree = torch.bincount(triples[:, 1], minlength=relations)
    if not len(triples):
        return torch.zeros((2, relations, relations), dtype=torch.long), degree
    # Float64 sparse sums of unit entries are exact below 2^53.
    if len(triples) >= 2 ** 53:
        raise ValueError('Pair counts exceed exact float64 integer arithmetic')
    keys, ids = torch.unique(triples[:, 0] * num_entities + triples[:, 2], sorted=True, return_inverse=True)
    reverse_ids = torch.searchsorted(keys, triples[:, 2] * num_entities + triples[:, 0])
    matrices = [torch.sparse_coo_tensor(torch.stack((row, triples[:, 1])), torch.ones(len(triples), dtype=torch.float64),
                                        (len(keys), relations)).coalesce() for row in (ids, reverse_ids)]
    counts = torch.stack([torch.sparse.mm(matrices[0].T, right).to_dense().long() for right in matrices])
    relation = torch.arange(relations)
    counts[0, relation, relation] = 0
    counts[1, relation, (relation + num_relations) % relations] = 0
    return counts, degree


def pair_channel_weights(direct: torch.Tensor, num_entities: int, num_relations: int) -> torch.Tensor:
    """Entity-pair channel weights ``count / sqrt(deg_a deg_b)`` of all relation pairs.

    Args:
        direct: Direct observed triples, CPU int64 ``[E, 3]``.
        num_entities: Number of entities.
        num_relations: Number of direct relations ``R``.

    Returns:
        Float64 ``[2, 2R, 2R]``: channel (same, reverse orientation), then the two relations of an edge.
    """
    counts, degree = pair_counts(direct, num_entities, num_relations)
    denominator = (degree[:, None] * degree[None, :]).clamp_min(1).double().sqrt()
    return counts.double() / denominator
