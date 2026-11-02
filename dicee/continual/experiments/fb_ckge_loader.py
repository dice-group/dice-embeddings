"""Loader/adapter for the FastKGE ``FB_CKGE`` benchmark format.

``FastKGE/data/FB_CKGE`` ships its own, pre-built continual-learning split
of FB15k-237 as a sequence of numbered snapshot folders (``0/`` .. ``4/``),
one per "time slice". This is the format produced by FastKGE's own
``data_to_id.py`` (see that script's ``KnowledgeGraph.load_data``):

* ``<i>/entity2id.txt`` / ``<i>/relation2id.txt`` -- the entity/relation
  vocabulary **as known up to and including snapshot i** (ids are stable
  and strictly growing across snapshots: every id assigned in snapshot
  ``i`` keeps the same meaning in every later snapshot's file).
* ``<i>/train.txt`` / ``<i>/valid.txt`` / ``<i>/test.txt`` -- the facts
  **newly introduced at snapshot i** (a *delta*, not a cumulative set),
  each independently split ~60/20/20 into train/valid/test.

This is therefore a different on-disk convention than DICE's own
``dicee.continual.materialize`` (which always writes *cumulative*
``train.txt`` per snapshot folder); :func:`load_fb_ckge_sequence` bridges
the two by concatenating deltas internally and returning the exact same
``(GraphSnapshot, List[UpdateStep])`` objects the rest of
``dicee.continual`` already knows how to consume.

Usage
-----
    t0, steps, num_entities, num_relations, full_filter_triples, cumulative_test = \\
        load_fb_ckge_sequence("FastKGE/data/FB_CKGE")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..graph_changes import GraphSnapshot, diff_snapshots
from ..update_sequence import UpdateStep, _density

logger = logging.getLogger(__name__)


def _read_id_map(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            k, v = parts
            mapping[k] = int(v)
    return mapping


def _read_triples_as_ids(path: Path, entity_to_idx: Dict[str, int],
                          relation_to_idx: Dict[str, int]) -> np.ndarray:
    rows: List[Tuple[int, int, int]] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            h, r, t = parts
            rows.append((entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]))
    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def load_fb_ckge_sequence(
    root: str,
) -> Tuple[GraphSnapshot, List[UpdateStep], int, int, np.ndarray, List[np.ndarray]]:
    """Load the FB_CKGE snapshot folders into DICE's continual-learning types.

    Parameters
    ----------
    root : str
        Path to the ``FB_CKGE`` directory (containing numbered subfolders
        ``0``, ``1``, ``2``, ...).

    Returns
    -------
    t0 : GraphSnapshot
        Cumulative training graph at snapshot 0.
    steps : List[UpdateStep]
        One ``UpdateStep`` per subsequent snapshot (1 .. N-1), with
        ``added_triples`` = that snapshot's delta and ``cumulative_triples``
        = the running cumulative training set.
    num_entities, num_relations : int
        Final (snapshot N-1) vocabulary sizes -- entity/relation ids are
        stable and only grow across snapshots, so this is the correct,
        fixed width for the model's embedding tables from the start if you
        want to avoid online growth, or the ceiling ``new_num_entities``
        should reach after the last step if growing online.
    full_filter_triples : np.ndarray
        Concatenation of every train/valid/test triple across *all*
        snapshots -- the standard "everything known" filtering vocabulary
        for filtered link-prediction evaluation (``er_vocab``/``re_vocab``),
        independent of which step is currently being evaluated.
    cumulative_test : List[np.ndarray]
        ``cumulative_test[i]`` is the union of every snapshot's test split
        from snapshot 0 up to and including snapshot i (one entry per
        snapshot, same length/order as ``[t0] + steps``), mirroring
        FastKGE's own ``test_all`` semantics: evaluation always measures
        retention of old facts plus acquisition of new ones.
    """
    root_path = Path(root)
    folders = sorted(
        (p for p in root_path.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not folders:
        raise ValueError(f"No numbered snapshot folders found under {root_path}")

    # Entity/relation ids are stable and cumulative across snapshots, so the
    # LAST folder's vocabulary is a superset covering every id ever used.
    last_folder = folders[-1]
    entity_to_idx = _read_id_map(last_folder / "entity2id.txt")
    relation_to_idx = _read_id_map(last_folder / "relation2id.txt")
    num_entities = len(entity_to_idx)
    num_relations = len(relation_to_idx)
    logger.info("FB_CKGE: %s snapshots, %s entities, %s relations (from %s)",
                len(folders), num_entities, num_relations, last_folder.name)

    cumulative_train_parts: List[np.ndarray] = []
    cumulative_valid_parts: List[np.ndarray] = []
    cumulative_test_parts: List[np.ndarray] = []
    all_parts: List[np.ndarray] = []
    cumulative_test: List[np.ndarray] = []

    prev_snapshot = None
    t0: GraphSnapshot = None  # type: ignore[assignment]
    steps: List[UpdateStep] = []

    for i, folder in enumerate(folders):
        train = _read_triples_as_ids(folder / "train.txt", entity_to_idx, relation_to_idx)
        valid = _read_triples_as_ids(folder / "valid.txt", entity_to_idx, relation_to_idx)
        test = _read_triples_as_ids(folder / "test.txt", entity_to_idx, relation_to_idx)

        cumulative_train_parts.append(train)
        cumulative_valid_parts.append(valid)
        cumulative_test_parts.append(test)
        all_parts.extend([train, valid, test])

        cumulative_train = np.concatenate(cumulative_train_parts, axis=0)
        cumulative_test.append(np.concatenate(cumulative_test_parts, axis=0))

        current_snapshot = GraphSnapshot.from_array(cumulative_train)

        if i == 0:
            t0 = current_snapshot
            logger.info("  snapshot 0 (T0): %s cumulative training triples", len(cumulative_train))
        else:
            diff = diff_snapshots(prev_snapshot, current_snapshot)
            steps.append(UpdateStep(
                step=i,
                added_triples=train,  # this snapshot's delta IS the newly added training facts
                cumulative_triples=cumulative_train,
                num_added_triples=len(train),
                num_active_entities=len(diff.active_entities),
                num_new_entities=len(diff.new_entities),
                num_active_relations=len(diff.active_relations),
                num_new_relations=len(diff.new_relations),
                density=_density(len(cumulative_train), len(diff.active_entities)),
                new_entity_ids=tuple(sorted(diff.new_entities)),
                new_relation_ids=tuple(sorted(diff.new_relations)),
            ))
            logger.info("  snapshot %s: +%s train triples, +%s entities, +%s relations",
                        i, len(train), len(diff.new_entities), len(diff.new_relations))

        prev_snapshot = current_snapshot

    full_filter_triples = np.concatenate(all_parts, axis=0)
    return t0, steps, num_entities, num_relations, full_filter_triples, cumulative_test
