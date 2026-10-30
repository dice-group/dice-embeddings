"""Graph loading and query structure of the ProofPFN benchmark, by the rules of the research evaluation.

- Loading (``load_graph``): the graph's files are hash-checked against its manifest entry first. Entities and
  relations get the ids of their sorted names over train, valid and test; the triples of each split keep their file
  order, duplicates included (``inductive-benchmark-v1-runtime/ib_eval.load_dataset``). A line holds three
  whitespace-separated names.
- Test sample (``ranked_positions``): with the ``first-1000`` rule, a test split of at most 1,000 rows is ranked whole;
  a larger one is sampled once: the first occurrence of every distinct test fact in file order is the pool, then
  ``sorted(random.Random(0).sample(pool, 1000))`` (``allgraph-v1/ag_eval.sample_positions``). The ``all`` rule ranks
  every row. A row's ``index`` is its position in ``test.txt``; unsampled test facts stay in the filter.
- Filter (``known_sets``): every known fact of train, valid and test, per (relation, anchor) and direction.
- Query groups (``query_groups``): (head direction, relation, anchor) with its test members in index order, stably
  sorted by relation (``lockbox-v5-runtime/lb5_eval.query_groups``).
- Self-loop rule (``reflexive_relations``): ``share1`` opens a relation when at least 1% of its train facts are
  self-loops (``ib_rule_eval.open_relations``).
- GT2 inputs (``gt2_inputs``): per direction and relation, the candidates unseen in the query role of train.txt and
  the Good-Turing estimate ``(n1 + 0.5) / (N + 1)`` (``lockbox-v6-runtime/fair_eval.gt_inputs``).
- Mean-tie rank (``rank_of``): the rank of the cells and of the N gate, with the banned candidates excluded and ties at
  their mean rank (``realsrc-v1/xrel/xr_diag.rank_of``), on purpose not the pessimistic rank of the evaluation.

dicee's ``KG`` is not used: its pandas reader turns tokens such as ``NA`` and ``null`` into missing values, which
merges distinct entities (``tests/test_data.py`` shows it); the research loader keeps every token.
"""
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from dicee.synthetic_kg.batch import Triple

from .manifests import SAMPLE, SAMPLE_SEED, SPLITS, GraphEntry, Manifest, ManifestError, PathLike, graph_problems

GroupKey = tuple[bool, int, int]


@dataclass(frozen=True)
class GraphData:
    """A loaded graph.

    Attributes:
        entities: The entity names, in id order (sorted).
        relations: The relation names, in id order (sorted).
        train: ``[N, 3]`` ids of train.txt in file order (the inference graph).
        valid: ``[N, 3]`` ids of valid.txt.
        test: ``[N, 3]`` ids of test.txt.
        hashes: The hashes computed by the manifest check (``manifests.graph_problems``).
    """

    entities: tuple[str, ...]
    relations: tuple[str, ...]
    train: torch.Tensor
    valid: torch.Tensor
    test: torch.Tensor
    hashes: dict[str, str]

    @property
    def num_entities(self) -> int:
        """The number of entities over all splits."""
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        """The number of relations over all splits."""
        return len(self.relations)


def read_triples(path: PathLike) -> list[tuple[str, str, str]]:
    """The non-empty lines of a triple file as name triples, in file order.

    Raises:
        ManifestError: For a line that does not hold three names.
    """
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ManifestError(f'{path}, line {number}: {len(parts)} fields instead of 3')
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def read_graph(folder: PathLike, hashes: Optional[dict[str, str]] = None) -> GraphData:
    """Load a graph folder by the research rules (sorted ids over all splits, triples in file order).

    Args:
        folder: The folder with train.txt, valid.txt and test.txt.
        hashes: The hashes of its checked files, kept in the result.

    Returns:
        The graph.
    """
    splits = {s: read_triples(Path(folder) / f'{s}.txt') for s in SPLITS}
    entities = sorted({e for rows in splits.values() for h, _, t in rows for e in (h, t)})
    relations = sorted({r for rows in splits.values() for _, r, _ in rows})
    ei, ri = {e: i for i, e in enumerate(entities)}, {r: i for i, r in enumerate(relations)}
    ids = {s: torch.tensor([[ei[h], ri[r], ei[t]] for h, r, t in rows], dtype=torch.long).reshape(-1, 3) for s, rows in splits.items()}
    return GraphData(tuple(entities), tuple(relations), ids['train'], ids['valid'], ids['test'], dict(hashes or {}))


def load_graph(manifest: Manifest, entry: GraphEntry, root: PathLike) -> GraphData:
    """Hash-check a graph of a manifest, then load it.

    Args:
        manifest: The graph's manifest (its blind flag decides whether a blind folder may load).
        entry: The graph.
        root: The data root that the manifest paths are relative to.

    Returns:
        The graph, with the computed hashes.

    Raises:
        ManifestError: When a file is missing, a hash or a manifest count differs.
        manifests.BlindSetError: For a blind folder outside a released blind manifest.
    """
    folder = manifest.folder(entry, root)
    problems, hashes = graph_problems(entry, folder)
    if problems:
        raise ManifestError('; '.join(problems))
    graph = read_graph(folder, hashes)
    if entry.counts is not None:
        c = entry.counts
        found = (graph.num_entities, graph.num_relations, len(graph.train), len(graph.test))
        if found != (c['entities'], c['relations'], c['inference_rows'], c['test_rows']):
            raise ManifestError(f'{entry.key}: entities, relations, train rows and test rows {found} differ from the manifest counts')
    return graph


def as_triples(ids: torch.Tensor) -> list[Triple]:
    """An ``[N, 3]`` id tensor as a list of int triples."""
    return [(int(h), int(r), int(t)) for h, r, t in ids.tolist()]


def sample_positions(test: Sequence[Triple]) -> Optional[list[int]]:
    """The ``first-1000`` sample: ``None`` (rank every row) for at most 1,000 rows, else the 1,000 sampled positions."""
    if len(test) <= SAMPLE:
        return None
    first: dict[Triple, int] = {}
    for i, fact in enumerate(test):
        first.setdefault(fact, i)
    return sorted(random.Random(SAMPLE_SEED).sample(list(first.values()), SAMPLE))


def ranked_positions(test: Sequence[Triple], rule: str) -> Optional[list[int]]:
    """The positions of the ranked test rows under a sample rule, ``None`` for all rows."""
    if rule == 'all':
        return None
    if rule == 'first-1000':
        return sample_positions(test)
    raise ValueError(f'unknown sample rule {rule!r}')


def known_sets(triples: Sequence[Triple]) -> dict[bool, dict[tuple[int, int], set[int]]]:
    """The filter: ``{False: {(r, h): tails}, True: {(r, t): heads}}`` of the given facts."""
    known: dict[bool, dict[tuple[int, int], set[int]]] = {False: defaultdict(set), True: defaultdict(set)}
    for h, r, t in triples:
        known[False][r, h].add(t)
        known[True][r, t].add(h)
    return known


def query_groups(test: Sequence[Triple]) -> list[tuple[GroupKey, list[int]]]:
    """``[((head, relation, anchor), member indices)]``: both directions of every fact, stably sorted by relation."""
    groups: 'OrderedDict[GroupKey, list[int]]' = OrderedDict()
    for index, (h, r, t) in enumerate(test):
        for head in (False, True):
            groups.setdefault((head, r, t if head else h), []).append(index)
    return sorted(groups.items(), key=lambda item: item[0][1])


def reflexive_relations(train: Sequence[Triple], rule: str = 'share1') -> set[int]:
    """The relations the self-loop rule leaves open: ``share1`` (at least 1% self-loops) or ``any`` (at least one)."""
    facts = Counter(r for _, r, _ in train)
    loops = Counter(r for h, r, t in train if h == t)
    return {r for r in loops if rule == 'any' or loops[r] >= 0.01 * facts[r]}


def gt2_inputs(train: Sequence[Triple], n: int) -> dict[tuple[bool, int], tuple[np.ndarray, float]]:
    """Per (head direction, relation) of train: the mask of candidates unseen in the query role and ``(n1 + 0.5) / (N + 1)``."""
    counts: dict[bool, dict[int, Counter]] = {False: defaultdict(Counter), True: defaultdict(Counter)}
    for h, r, t in train:
        counts[False][r][t] += 1
        counts[True][r][h] += 1
    out = {}
    for head in (False, True):
        for r, c in counts[head].items():
            seen = np.zeros(n, bool)
            seen[list(c)] = True
            total, n1 = sum(c.values()), sum(1 for k in c.values() if k == 1)
            out[head, r] = (~seen, (n1 + 0.5) / (total + 1))
    return out


@dataclass(frozen=True)
class Queries:
    """The ranked test facts of a graph and everything their ranks need.

    Attributes:
        facts: The ranked test facts, in test order (all rows, or the sample in position order).
        positions: The position in test.txt of each ranked fact, or ``None`` when every row is ranked.
        test_rows: The number of rows of test.txt.
        groups: The query groups over ``facts`` (members are indices into ``facts``).
        known: The filter of all facts of train, valid and test.
        reflexive: The relations open under the self-loop rule.
        gt2: The GT2 inputs per (head direction, relation).
    """

    facts: list[Triple]
    positions: Optional[list[int]]
    test_rows: int
    groups: list[tuple[GroupKey, list[int]]]
    known: dict[bool, dict[tuple[int, int], set[int]]]
    reflexive: set[int]
    gt2: dict[tuple[bool, int], tuple[np.ndarray, float]]

    def index(self, member: int) -> int:
        """The position in test.txt of a ranked fact."""
        return member if self.positions is None else self.positions[member]


def prepare_queries(graph: GraphData, sample_rule: str) -> Queries:
    """The ranked facts, query groups, filter, open relations and GT2 inputs of a graph under a sample rule."""
    test, train = as_triples(graph.test), as_triples(graph.train)
    positions = ranked_positions(test, sample_rule)
    facts = test if positions is None else [test[i] for i in positions]
    return Queries(facts=facts, positions=positions, test_rows=len(test), groups=query_groups(facts),
                   known=known_sets(train + test + as_triples(graph.valid)), reflexive=reflexive_relations(train),
                   gt2=gt2_inputs(train, graph.num_entities))


def inference_stats(train: torch.Tensor) -> tuple[int, int]:
    """``(|V|, |E|)`` of an inference graph: its distinct entities (heads and tails) and its distinct triples."""
    return int(torch.unique(train[:, [0, 2]]).numel()), int(torch.unique(train, dim=0).shape[0])


def rank_of(score: np.ndarray, target: int, banned: Iterable[int]) -> float:
    """The filtered rank of the target with the banned candidates excluded and ties at their mean rank.

    Args:
        score: The scores over all entities.
        target: The target entity.
        banned: The candidates that leave the ranking (the other true answers, the anchor).

    Returns:
        ``1 + #{s > s[target]} + (#{s == s[target]} - 1) / 2`` over the candidates that are not banned.
    """
    s = score.astype(np.float64).copy()
    s[list(banned)] = -np.inf
    v = s[target]
    return 1.0 + float((s > v).sum()) + 0.5 * float((s == v).sum() - 1)
