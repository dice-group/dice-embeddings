"""The query cells of the registered analysis: rule-confidence predictors on the observed train graph (CPU, no model).

Per graph (``graph_cells``), as ``fresh-v1-runtime/fv1_cells.py`` runs ``icml-roadmap-20261025/step5/s5_measure.evaluate_rows``
with the predictors of ``realsrc-v1/xrel/xr_diag.py``:

- the observed graph is train.txt with 2R relations (r and its inverse r + R); the filter is train, valid and test;
- the CSR matrix of a relation lists the edges of each row in train.txt order, a repeated fact as its own entry
  (``_input_order_matrix``); R2 samples facts by their position in this order, so ``r2_at`` does not depend on the
  scipy version;
- the queries are every test fact in both directions, in the order of ``rng.sample(facts, len(facts))`` with
  ``rng = random.Random(0)``: the head queries ``(t, r + R, h)`` of that order, then the tail queries ``(h, r, t)``;
- ``R1`` scores a candidate e of a query ``(x, q, ?)`` by the sum over relations ``r' != q`` with an observed
  ``(x, r', e)`` of ``conf(r' => q)`` (shared entity pairs over the pairs of r', rules below 0.05 dropped);
  ``r1_at`` is the score of the answer;
- ``R2`` sums ``conf(b)`` over the bodies ``b = (b1, b2)`` of q that reach e from x: up to 100 sampled facts of q propose
  bodies, the 8 most frequent are kept (hubs capped at 2,000 neighbours), and the confidence is measured on 50 sampled
  heads of q; ``r2_at`` is the score of the answer; the samples draw from the same ``rng``, relation by relation;
- ``PEER`` votes for e through the peers x' that share a value v of another relation a with x, weighted by
  ``1 / deg_a(v)``; ``rank_peer`` is the filtered rank of the answer with the anchor excluded and ties at their mean
  rank (``data.rank_of``; on purpose a different rule from the pessimistic ranks of the evaluation).

The predictors POP, VOTE and CF of ``xr_diag`` are not ported: no output holds them, and they draw no random numbers.

Output (``write_queries``): ``queries.csv`` with the columns graph, query, r1_at, rank_peer, r2_at, floats as ``repr``,
``query`` = ``<index>:h`` (head query, the anchor is the tail) or ``<index>:t``, index = the position in test.txt,
sorted by index with the tail query first, CSV line ends ``\\r\\n``; and ``<out>.meta.json`` with the input hashes.
"""
import csv
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

from dicee.synthetic_kg.checkpoint import atomic_write_bytes

from .data import GraphData, load_graph, rank_of
from .manifests import REPO, SPLITS, GraphEntry, Manifest, ManifestError, PathLike, file_sha256

SEED = 0
MIN_CONFIDENCE = 0.05
NBODIES, BODY_FACTS, CONF_HEADS, HUB = 8, 100, 50, 2000
HEADER = ('graph', 'query', 'r1_at', 'rank_peer', 'r2_at')
Cell = tuple[str, float, float, float]


def _input_order_matrix(rows: np.ndarray, cols: np.ndarray, n: int) -> sparse.csr_matrix:
    """The ``n x n`` CSR matrix with one stored 1.0 per edge ``rows[i] -> cols[i]``, each row in input order.

    A stable sort by row gives the columns and the row counts give ``indptr``. A repeated edge stays its own entry. scipy
    1.13.0, the version of the frozen run, built ``csr_matrix((data, (rows, cols)))`` this way. Later versions sort each
    row, which changes the facts that ``_rules_r2`` samples.
    """
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return sparse.csr_matrix((np.ones(len(rows)), cols[np.argsort(rows, kind='stable')], indptr), shape=(n, n))


class ObservedGraph:
    """The observed graph with 2R relations: per relation a CSR matrix (source x destination, rows in input order) and its
    transpose, the sorted edge list, and per relation the number of sources of each destination (its popularity)."""

    def __init__(self, edges: np.ndarray, n: int, nr: int):
        """Build the matrices of an ``[N, 3]`` int64 edge array over ``n`` entities and ``nr`` relations."""
        self.n, self.nr = n, nr
        h, r, t = edges[:, 0], edges[:, 1], edges[:, 2]
        src, rel, dst = np.concatenate([h, t]), np.concatenate([r, r + nr]), np.concatenate([t, h])
        order = np.lexsort((dst, rel, src))
        self.src, self.rel, self.dst = src[order], rel[order], dst[order]
        self.ptr = np.searchsorted(self.src, np.arange(n + 1))
        self.mat: dict[int, sparse.csr_matrix] = {}
        for q in np.unique(rel).tolist():
            m = rel == q
            self.mat[q] = _input_order_matrix(src[m], dst[m], n)
        self.pop = {q: np.asarray(m.sum(0)).ravel() for q, m in self.mat.items()}
        self.matT = {q: m.T.tocsr() for q, m in self.mat.items()}
        self.invdeg = {q: np.where(self.pop[q] > 0, 1.0 / np.maximum(self.pop[q], 1), 0.0) for q in self.mat}

    def inv(self, q: int) -> int:
        """The inverse of a relation of the 2R relations.

        Args:
            q: A relation id in ``0..2R-1``.

        Returns:
            The relation ``q + R`` for ``q < R``, else ``q - R``.
        """
        return q + self.nr if q < self.nr else q - self.nr

    def out(self, x: int) -> tuple[np.ndarray, np.ndarray]:
        """The relations and destinations of the edges that leave x.

        Args:
            x: The source entity.

        Returns:
            The arrays ``(relations, destinations)``, slices of the sorted edge list (by relation, then destination), a
            repeated fact as its own entry.
        """
        a, b = self.ptr[x], self.ptr[x + 1]
        return self.rel[a:b], self.dst[a:b]


def _rules_r1(g: ObservedGraph, qs: Iterable[int]) -> dict[int, dict[int, float]]:
    """``conf(r' => q)`` for every q of ``qs``: the pairs of r' that are pairs of q over the pairs of r' (at least 0.05)."""
    pairs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for s, r, d in zip(g.src.tolist(), g.rel.tolist(), g.dst.tolist()):
        pairs[s, d].add(r)
    co: Counter = Counter()
    size = Counter(g.rel.tolist())
    for rs in pairs.values():
        if len(rs) > 1:
            for a in rs:
                for b in rs:
                    if a != b:
                        co[a, b] += 1
    return {q: {a: c / size[a] for (a, b), c in co.items() if b == q and c / size[a] >= MIN_CONFIDENCE} for q in qs}


def _rules_r2(g: ObservedGraph, q: int, rng: random.Random) -> list[tuple[float, int, int]]:
    """``[(conf, b1, b2)]`` of the most frequent length-2 bodies of q with a positive confidence."""
    m = g.mat.get(q)
    if m is None or m.nnz == 0:
        return []
    hs, ts = m.nonzero()
    pick = rng.sample(range(len(hs)), min(BODY_FACTS, len(hs)))
    count: Counter = Counter()
    for i in pick:
        h, t = int(hs[i]), int(ts[i])
        r1, x1 = g.out(h)
        r2, x2 = g.out(t)
        if len(x1) > HUB:
            s = np.random.default_rng(h).choice(len(x1), HUB, replace=False)
            r1, x1 = r1[s], x1[s]
        if len(x2) > HUB:
            s = np.random.default_rng(t).choice(len(x2), HUB, replace=False)
            r2, x2 = r2[s], x2[s]
        at_t = defaultdict(list)
        for c, x in zip(r2.tolist(), x2.tolist()):
            if x not in (h, t):
                at_t[x].append(c)
        seen = set()
        for a, x in zip(r1.tolist(), x1.tolist()):
            for c in at_t.get(x, ()):
                body = (a, g.inv(c))   # h -a-> x and t -c-> x, so x -inv(c)-> t
                if body not in seen:
                    seen.add(body)
                    count[body] += 1
    bodies = []
    heads = np.unique(hs)
    sample = rng.sample(list(heads.tolist()), min(CONF_HEADS, len(heads)))
    for (b1, b2), _ in count.most_common(NBODIES):
        hit = tot = 0
        for h in sample:
            reach = g.mat[b1][h] @ g.mat[b2]
            idx = reach.indices[reach.indices != h]
            if len(idx):
                tot += len(idx)
                hit += int(m[h, idx].sum())
        if tot:
            bodies.append((hit / tot, b1, b2))
    return [b for b in bodies if b[0] > 0]


def _peer_scores(g: ObservedGraph, anchor: int, q: int) -> np.ndarray:
    """The PEER votes of a query ``(anchor, q, ?)`` over all entities."""
    m = g.mat.get(q)
    if m is None:
        return np.zeros(g.n)
    w = np.zeros(g.n)
    rels, vals = g.out(anchor)
    for a in np.unique(rels).tolist():
        if a in (q, g.inv(q)):
            continue
        v = vals[rels == a]
        wv = np.zeros(g.n)
        wv[v] = g.invdeg[a][v]
        w += np.asarray(wv @ g.matT[a]).ravel()
    w[anchor] = 0
    return np.asarray(w @ m).ravel()


def _query_cells(observed: np.ndarray, filt: np.ndarray, facts: Sequence[tuple[int, int, int]], n: int, nr: int,
                rng: random.Random) -> list[tuple[tuple[int, int, int], bool, float, float, float]]:
    """``[(fact, head, r1_at, rank_peer, r2_at)]`` of every fact in both directions, in the draw order of the research.

    Args:
        observed: ``[N, 3]`` int64 train facts.
        filt: ``[N, 3]`` int64 facts of the filter (train, valid, test).
        facts: The test facts.
        n: The number of entities.
        nr: The number of relations.
        rng: The random stream (``random.Random(0)``); the facts' order and the R2 samples draw from it.
    """
    g = ObservedGraph(observed, n, nr)
    rows = rng.sample(list(facts), len(facts))
    qlist = [((t, r + nr, h), True, (h, r, t)) for h, r, t in rows] + [((h, r, t), False, (h, r, t)) for h, r, t in rows]
    known: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in filt.tolist():
        known[h, r].add(t)
        known[t, r + nr].add(h)
    qs = sorted({q for (_, q, _), _, _ in qlist})
    r1 = _rules_r1(g, qs)
    r2 = {q: _rules_r2(g, q, rng) for q in qs}
    out = []
    for (anchor, q, target), head, fact in qlist:
        banned = (known[anchor, q] - {target}) | {anchor}
        s1 = np.zeros(n)
        for a, c in r1.get(q, {}).items():
            row = g.mat[a][anchor]
            s1[row.indices] += c
        s2 = np.zeros(n)
        for c, b1, b2 in r2.get(q, []):
            reach = g.mat[b1][anchor] @ g.mat[b2]
            s2[reach.indices] += c
        out.append((fact, head, float(s1[target]), rank_of(_peer_scores(g, anchor, q), target, banned), float(s2[target])))
    return out


def graph_cells(graph: GraphData) -> list[Cell]:
    """``[(query, r1_at, rank_peer, r2_at)]`` of a graph, sorted by test index with the tail query first.

    Args:
        graph: The loaded graph. Every row of test.txt gives a query in both directions.

    Returns:
        Two cells per test fact. ``query`` is ``<index>:t`` (tail query) or ``<index>:h`` (head query), with ``index``
        the position in test.txt.

    Raises:
        ManifestError: When test.txt holds a fact twice (a value could not join its fact).
    """
    split = {s: getattr(graph, s).numpy().astype('int64').reshape(-1, 3) for s in SPLITS}
    facts = [(int(h), int(r), int(t)) for h, r, t in split['test'].tolist()]
    index_of = {f: i for i, f in enumerate(facts)}
    if len(index_of) != len(facts):
        raise ManifestError('test.txt holds a fact twice')
    filt = np.concatenate([split['train'], split['valid'], split['test']])
    cells = [(f'{index_of[fact]}:{"h" if head else "t"}', r1, peer, r2)
             for fact, head, r1, peer, r2 in _query_cells(split['train'], filt, facts, graph.num_entities, graph.num_relations, random.Random(SEED))]
    return sorted(cells, key=lambda c: (int(c[0].split(':')[0]), c[0].endswith(':h')))


def write_queries(manifest: Manifest, entries: Sequence[GraphEntry], out: PathLike, root: PathLike = REPO) -> dict[str, Any]:
    """Write ``queries.csv`` of the graphs (hash-checked before they load) and its ``.meta.json``.

    Args:
        manifest: The graph set.
        entries: The graphs, in output order.
        out: The CSV file.
        root: The data root of the manifest paths.

    Returns:
        The meta record.

    Raises:
        ManifestError: When a graph fails to load (``data.load_graph``) or its test.txt holds a fact twice.
        manifests.BlindSetError: For a blind folder outside a released blind manifest.
    """
    out = Path(out)
    counts, inputs = {}, {}
    text = io.StringIO(newline='')
    writer = csv.writer(text)
    writer.writerow(HEADER)
    for entry in entries:
        graph = load_graph(manifest, entry, root)
        rows = graph_cells(graph)
        for query, r1, peer, r2 in rows:
            writer.writerow([entry.name, query, repr(r1), repr(peer), repr(r2)])
        counts[entry.name] = len(rows)
        inputs[entry.name] = {f'{s}.txt': graph.hashes[f'{s}.txt'] for s in SPLITS}
    atomic_write_bytes(out, text.getvalue().encode())
    meta = dict(manifest=manifest.name, manifest_sha256=manifest.sha256, seed=SEED, graphs=counts, out_sha256=file_sha256(out),
                inputs=inputs, code_sha256={f'benchmarks/proofpfn/{name}': file_sha256(Path(__file__).with_name(name)) for name in ('cells.py', 'data.py')})
    atomic_write_bytes(out.with_name(out.name + '.meta.json'), (json.dumps(meta, indent=1, sort_keys=True) + '\n').encode())
    return meta

