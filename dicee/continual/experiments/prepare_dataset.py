"""Build fixed, on-disk continual-learning dataset snapshots for a dataset.

Takes a path to a dataset (e.g. ``KGs/UMLS``) and, for each requested
scenario, creates a sequence of literal, on-disk, KG-loadable dataset
folders -- one per snapshot -- so that training always reads the exact
same fixed data instead of generating a new random partition on the fly
every time an experiment is run.

For ``--dataset_dir KGs/UMLS --scenarios new_entities --num_steps 3`` this
creates::

    Experiments/continual_datasets/UMLS_new_entities/
        UMLS_0_new_entities/{train,valid,test}.txt
        UMLS_1_new_entities/{train,valid,test}.txt
        UMLS_2_new_entities/{train,valid,test}.txt
        UMLS_3_new_entities/{train,valid,test}.txt
        metadata.json

Each ``UMLS_<i>_new_entities`` folder is a complete, independent
``dicee.knowledge_graph.KG``-loadable dataset_dir containing the
*cumulative* snapshot ``T_i`` as ``train.txt`` (not just the delta), plus
the original, unmodified ``valid.txt`` / ``test.txt``.

Usage
-----
    # Build all three scenarios (existing_entities, new_entities, mixed)
    # for UMLS with 3 snapshots after T0:
    python -m dicee.continual.experiments.prepare_dataset \
        --dataset_dir KGs/UMLS --num_steps 3

    # Only one scenario, a different seed/fraction, forcing a rebuild:
    python -m dicee.continual.experiments.prepare_dataset \
        --dataset_dir KGs/KINSHIP --scenarios mixed --num_steps 5 \
        --seed 1 --initial_fraction 0.4 --force_regenerate

Once built, point any run at a specific snapshot directly, e.g.::

    from dicee.knowledge_graph import KG
    kg = KG(dataset_dir="Experiments/continual_datasets/UMLS_new_entities/UMLS_2_new_entities")

or let ``run_experiment.py`` walk the whole sequence automatically (it
materializes -- or reuses -- these same folders internally).
"""
from __future__ import annotations

import argparse
import logging

from dicee.continual.update_sequence import VALID_SCENARIOS
from dicee.continual.materialize import (
    DEFAULT_DATASET_ROOT,
    materialize_scenario_snapshots,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(
    dataset_dir: str,
    scenarios=VALID_SCENARIOS,
    num_steps: int = 3,
    initial_fraction: float = 0.5,
    seed: int = 0,
    output_root: str = DEFAULT_DATASET_ROOT,
    separator: str = r"\s+",
    update_batch_size=None,
    min_incident_facts: int = 1,
    force_regenerate: bool = False,
):
    """Materialize (or confirm already-materialized) dataset snapshots for every requested scenario."""
    results = {}
    for scenario in scenarios:
        scenario_dir = materialize_scenario_snapshots(
            dataset_dir=dataset_dir,
            scenario=scenario,
            num_steps=num_steps,
            initial_fraction=initial_fraction,
            seed=seed,
            output_root=output_root,
            separator=separator,
            update_batch_size=update_batch_size,
            min_incident_facts=min_incident_facts,
            force_regenerate=force_regenerate,
        )
        results[scenario] = str(scenario_dir)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build fixed, on-disk continual-KGE dataset snapshots for a dataset: one "
                    "<dataset>_<step>_<scenario> folder per snapshot, reused by every later "
                    "experiment run instead of regenerating a new random partition each time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", required=True, help="Path to a dataset_dir with train/valid/test.txt.")
    parser.add_argument("--separator", default=r"\s+", help="Column separator used to read the triple files.")
    parser.add_argument("--scenarios", nargs="+", default=list(VALID_SCENARIOS), choices=list(VALID_SCENARIOS),
                         help="Which scenario(s) to build. Defaults to all three.")
    parser.add_argument("--num_steps", type=int, default=3, help="Number of update steps (snapshots) after T0.")
    parser.add_argument("--initial_fraction", type=float, default=0.5,
                         help="Fraction of train triples placed in T0.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the partition.")
    parser.add_argument("--update_batch_size", type=int, default=None,
                         help="Fixed number of triples per update step (overrides even splitting if set).")
    parser.add_argument("--min_incident_facts", type=int, default=1,
                         help="Minimum incident facts required to keep an entity active.")
    parser.add_argument("--output_root", default=DEFAULT_DATASET_ROOT,
                         help="Root directory the per-scenario snapshot folders are written under.")
    parser.add_argument("--force_regenerate", action="store_true",
                         help="Rebuild and overwrite even if already materialized.")
    args = parser.parse_args()

    results = prepare_dataset(
        dataset_dir=args.dataset_dir,
        scenarios=args.scenarios,
        num_steps=args.num_steps,
        initial_fraction=args.initial_fraction,
        seed=args.seed,
        output_root=args.output_root,
        separator=args.separator,
        update_batch_size=args.update_batch_size,
        min_incident_facts=args.min_incident_facts,
        force_regenerate=args.force_regenerate,
    )
    for scenario, path in results.items():
        logger.info("[%s] -> %s", scenario, path)


if __name__ == "__main__":
    main()
