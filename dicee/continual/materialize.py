"""Materialize fixed, on-disk continual-learning dataset snapshots.

This is the actual deliverable requested: given a path to a dataset (e.g.
``KGs/UMLS``) and a scenario, build literal, human-readable per-snapshot
dataset folders ONCE and reuse them forever after -- no random
regeneration inside every training run.

For ``dataset_dir=KGs/UMLS``, ``scenario="new_entities"``, ``num_steps=3``,
this creates::

    Experiments/continual_datasets/UMLS_new_entities/
        UMLS_0_new_entities/train.txt   # T0  (cumulative triples at snapshot 0)
        UMLS_0_new_entities/valid.txt   # copied, unchanged, from the original split
        UMLS_0_new_entities/test.txt    # copied, unchanged, from the original split
        UMLS_1_new_entities/train.txt   # T1 = T0 + DeltaT_1 (cumulative, not just the delta)
        UMLS_1_new_entities/valid.txt
        UMLS_1_new_entities/test.txt
        UMLS_2_new_entities/train.txt   # T2
        UMLS_2_new_entities/valid.txt
        UMLS_2_new_entities/test.txt
        UMLS_3_new_entities/train.txt   # T3 == the full original train.txt
        UMLS_3_new_entities/valid.txt
        UMLS_3_new_entities/test.txt
        metadata.json                  # per-snapshot entity/relation/triple counts

Every snapshot's ``train.txt`` is **cumulative** (``T_i``), not just the
delta, so each ``UMLS_i_<scenario>`` folder is independently a complete,
plain ``dicee.knowledge_graph.KG``-loadable dataset directory -- you can
point ``KG(dataset_dir=...)`` (or any other DICE entry point) straight at
``UMLS_3_new_entities`` and get exactly the snapshot-3 training graph, with
the *same* original valid/test files every snapshot uses for evaluation.

This is the single source of truth for a given
(dataset, scenario, num_steps, initial_fraction, seed, ...) combination:
every ``run_experiment.py`` invocation with that same configuration reads
triples straight from these folders instead of generating a new random
partition, so results are reproducible and comparable run to run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dicee.knowledge_graph import KG

from .graph_changes import GraphSnapshot, diff_snapshots
from .update_sequence import (
    VALID_SCENARIOS,
    UpdateStep,
    _density,
    get_or_create_update_sequence,
)

logger = logging.getLogger(__name__)

DEFAULT_SEQUENCE_CACHE_ROOT = "Experiments/continual_sequence_cache"
DEFAULT_DATASET_ROOT = "Experiments/continual_datasets"


def _write_triples_txt(path: Path, triples: np.ndarray, idx_to_entity: Dict[int, str],
                        idx_to_relation: Dict[int, str]) -> None:
    with open(path, "w") as f:
        for h, r, t in triples:
            f.write(f"{idx_to_entity[int(h)]}\t{idx_to_relation[int(r)]}\t{idx_to_entity[int(t)]}\n")


def _write_array_triples_txt(path: Path, array: Optional[np.ndarray], idx_to_entity: Dict[int, str],
                              idx_to_relation: Dict[int, str]) -> None:
    if array is None:
        open(path, "w").close()
        return
    _write_triples_txt(path, array, idx_to_entity, idx_to_relation)


def _idx_to_str_maps(kg: KG) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Build idx->string lookup dicts for entities/relations, handling the
    different internal representations ``KG.entity_to_idx``/``relation_to_idx``
    can take (plain dict, pandas DataFrame, or polars DataFrame)."""
    def _to_dict(mapping, column: str) -> Dict[int, str]:
        if isinstance(mapping, dict):
            return {v: k for k, v in mapping.items()}
        # pandas DataFrame: single `column`, integer id is the row index.
        if hasattr(mapping, "to_dict") and not hasattr(mapping, "to_pandas"):
            return mapping[column].to_dict()
        # polars DataFrame: explicit "index" + `column` columns.
        if hasattr(mapping, "to_pandas"):
            pdf = mapping.to_pandas()
            return dict(zip(pdf["index"], pdf[column]))
        raise TypeError(f"Unsupported entity/relation mapping type: {type(mapping)}")

    return _to_dict(kg.entity_to_idx, "entity"), _to_dict(kg.relation_to_idx, "relation")


def scenario_folder_name(dataset_name: str, scenario: str) -> str:
    return f"{dataset_name}_{scenario}"


def snapshot_folder_name(dataset_name: str, step: int, scenario: str) -> str:
    return f"{dataset_name}_{step}_{scenario}"


def materialize_scenario_snapshots(
    dataset_dir: str,
    scenario: str,
    num_steps: int,
    initial_fraction: float = 0.5,
    seed: int = 0,
    output_root: str = DEFAULT_DATASET_ROOT,
    separator: str = r"\s+",
    update_batch_size: Optional[int] = None,
    min_incident_facts: int = 1,
    sequence_cache_root: str = DEFAULT_SEQUENCE_CACHE_ROOT,
    force_regenerate: bool = False,
) -> Path:
    """Build (or reuse) fixed, on-disk per-snapshot dataset folders.

    Returns the scenario folder, e.g.
    ``Experiments/continual_datasets/UMLS_new_entities``, which directly
    contains one ``<dataset>_<step>_<scenario>`` subfolder per snapshot
    (``step`` 0 .. ``num_steps``), each a complete KG-loadable dataset_dir.
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {VALID_SCENARIOS}")

    dataset_name = Path(dataset_dir).resolve().name
    scenario_dir = Path(output_root) / scenario_folder_name(dataset_name, scenario)
    success_marker = scenario_dir / "_SUCCESS"

    if success_marker.exists() and not force_regenerate:
        logger.info("Reusing already-materialized dataset snapshots at %s", scenario_dir)
        return scenario_dir

    logger.info("Materializing dataset snapshots for %s / %s (K=%s) at %s",
                dataset_name, scenario, num_steps, scenario_dir)

    kg = KG(dataset_dir=dataset_dir, separator=separator, eval_model="train_val_test")
    idx_to_entity, idx_to_relation = _idx_to_str_maps(kg)

    # The actual triple partition is computed/cached via the existing
    # (index-based) sequence generator -- that machinery already implements
    # the three scenarios correctly and deterministically; here we only add
    # a text-file materialization layer on top of it.
    t0, steps, _seq_cache_dir, _was_cached = get_or_create_update_sequence(
        dataset_dir=dataset_dir,
        train_triples=kg.train_set,
        scenario=scenario,
        num_steps=num_steps,
        initial_fraction=initial_fraction,
        seed=seed,
        cache_root=sequence_cache_root,
        update_batch_size=update_batch_size,
        min_incident_facts=min_incident_facts,
        force_regenerate=force_regenerate,
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    per_snapshot_meta: List[Dict] = []

    def _write_snapshot(step_idx: int, cumulative: np.ndarray, num_added: int,
                         num_new_entities: int, num_new_relations: int) -> None:
        snap_dir = scenario_dir / snapshot_folder_name(dataset_name, step_idx, scenario)
        snap_dir.mkdir(parents=True, exist_ok=True)
        _write_triples_txt(snap_dir / "train.txt", cumulative, idx_to_entity, idx_to_relation)
        _write_array_triples_txt(snap_dir / "valid.txt", kg.valid_set, idx_to_entity, idx_to_relation)
        _write_array_triples_txt(snap_dir / "test.txt", kg.test_set, idx_to_entity, idx_to_relation)
        snap = GraphSnapshot.from_array(cumulative)
        per_snapshot_meta.append({
            "step": step_idx,
            "folder": snap_dir.name,
            "num_cumulative_triples": int(len(cumulative)),
            "num_added_triples": int(num_added),
            "num_new_entities": int(num_new_entities),
            "num_new_relations": int(num_new_relations),
            "num_active_entities": len(snap.entities),
            "num_active_relations": len(snap.relations),
        })
        logger.info("  wrote %s (%s cumulative triples, +%s new entities, +%s new relations)",
                    snap_dir.name, len(cumulative), num_new_entities, num_new_relations)

    t0_array = t0.to_array()
    _write_snapshot(0, t0_array, num_added=len(t0_array),
                     num_new_entities=len(t0.entities), num_new_relations=len(t0.relations))
    for s in steps:
        _write_snapshot(s.step, s.cumulative_triples, num_added=s.num_added_triples,
                         num_new_entities=s.num_new_entities, num_new_relations=s.num_new_relations)

    with open(scenario_dir / "metadata.json", "w") as f:
        json.dump({
            "dataset_dir": dataset_dir,
            "dataset_name": dataset_name,
            "scenario": scenario,
            "num_steps": num_steps,
            "initial_fraction": initial_fraction,
            "seed": seed,
            "update_batch_size": update_batch_size,
            "min_incident_facts": min_incident_facts,
            "snapshots": per_snapshot_meta,
        }, f, indent=2)

    success_marker.touch()
    logger.info("Done. %s snapshot folders written under %s", len(per_snapshot_meta), scenario_dir)
    return scenario_dir


def load_snapshot_metadata(scenario_dir: str) -> Dict:
    """Load the ``metadata.json`` written by :func:`materialize_scenario_snapshots`."""
    with open(Path(scenario_dir) / "metadata.json") as f:
        return json.load(f)


def get_snapshot_dataset_dir(scenario_dir: str, step: int) -> Path:
    """Path to the materialized ``<dataset>_<step>_<scenario>`` dataset_dir for one snapshot."""
    meta = load_snapshot_metadata(scenario_dir)
    for s in meta["snapshots"]:
        if s["step"] == step:
            return Path(scenario_dir) / s["folder"]
    raise ValueError(f"No snapshot with step={step} found in {scenario_dir}")


def _str_to_idx_maps(kg: KG) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Inverse of ``_idx_to_str_maps``: entity/relation string -> integer id."""
    def _to_dict(mapping, column: str) -> Dict[str, int]:
        if isinstance(mapping, dict):
            return dict(mapping)
        if hasattr(mapping, "to_dict") and not hasattr(mapping, "to_pandas"):
            return dict(zip(mapping[column], mapping.index))
        if hasattr(mapping, "to_pandas"):
            pdf = mapping.to_pandas()
            return dict(zip(pdf[column], pdf["index"]))
        raise TypeError(f"Unsupported entity/relation mapping type: {type(mapping)}")

    return _to_dict(kg.entity_to_idx, "entity"), _to_dict(kg.relation_to_idx, "relation")


def _read_triples_txt(path: Path) -> List[Tuple[str, str, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            h, r, t = parts
            rows.append((h, r, t))
    return rows


def load_materialized_sequence(scenario_dir: str, kg: KG) -> Tuple[np.ndarray, List[UpdateStep]]:
    """Load the fixed, on-disk snapshot sequence back as index arrays for training.

    Triples are read from each ``<dataset>_<step>_<scenario>/train.txt`` file
    (the literal, on-disk, cumulative ``T_i``) and mapped through the SAME
    ``kg.entity_to_idx`` / ``kg.relation_to_idx`` built from the full,
    original dataset -- not a per-snapshot vocabulary -- so entity/relation
    indices stay stable and only ever grow across snapshots, exactly as
    continual training requires.

    Returns ``(t0_array, steps)`` where ``steps[i].cumulative_triples`` is
    ``T_{i+1}`` (same shape/semantics as ``generate_update_sequence``).
    """
    meta = load_snapshot_metadata(scenario_dir)
    entity_to_idx, relation_to_idx = _str_to_idx_maps(kg)

    def _load_step_array(folder: str) -> np.ndarray:
        rows = _read_triples_txt(Path(scenario_dir) / folder / "train.txt")
        if not rows:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(
            [(entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]) for h, r, t in rows],
            dtype=np.int64,
        )

    snapshots_meta = sorted(meta["snapshots"], key=lambda s: s["step"])
    t0_meta, rest_meta = snapshots_meta[0], snapshots_meta[1:]
    t0_array = _load_step_array(t0_meta["folder"])
    prev = GraphSnapshot.from_array(t0_array)

    steps: List[UpdateStep] = []
    for s_meta in rest_meta:
        cur_array = _load_step_array(s_meta["folder"])
        current = GraphSnapshot.from_array(cur_array)
        diff = diff_snapshots(prev, current)
        steps.append(UpdateStep(
            step=s_meta["step"],
            added_triples=diff.added_array(),
            cumulative_triples=current.to_array(),
            num_added_triples=len(diff.added_triples),
            num_active_entities=len(diff.active_entities),
            num_new_entities=len(diff.new_entities),
            num_active_relations=len(diff.active_relations),
            num_new_relations=len(diff.new_relations),
            density=_density(len(current.triples), len(diff.active_entities)),
            new_entity_ids=tuple(sorted(diff.new_entities)),
            new_relation_ids=tuple(sorted(diff.new_relations)),
        ))
        prev = current

    return t0_array, steps
