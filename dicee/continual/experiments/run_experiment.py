"""Configuration-driven experiment runner for continual KGE experiments.

Usage (plain CLI arguments)
---------------------------------------------------
    python -m dicee.continual.experiments.run_experiment \
        --experiment_id umls_mixed --dataset_dir KGs/UMLS --scenario mixed \
        --model DistMult --embedding_dim 32 --num_steps 3 --seed 0

    # Pick a different base model, with model-specific kwargs (e.g. Keci's
    # Clifford-algebra p/q), and only run a subset of methods:
    python -m dicee.continual.experiments.run_experiment \
        --experiment_id umls_keci --dataset_dir KGs/UMLS --model Keci \
        --model_kwargs '{"p": 0, "q": 1}' --methods uniform chebyshev

Usage (legacy JSON config file, still supported)
--------------------------------------------------
    python -m dicee.continual.experiments.run_experiment \
        --config dicee/continual/experiments/smoke_test.json

Any explicitly-passed CLI flag overrides the corresponding key when both
``--config`` and CLI flags are given, so a JSON config can be used as a
base and tweaked ad hoc from the command line.

Loads a dataset via ``dicee.knowledge_graph.KG`` (Section 5: original
train/valid/test splits, continual sequences built *only* from
``kg.train_set``), generates a reproducible update sequence, then runs each
requested baseline method (``finetune`` / ``uniform`` / ``chebyshev``) on
the exact same sequence + initial checkpoint, saving per-step metrics,
the update-sequence metadata, and the resolved config to
``output_dir / experiment_id``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from dicee.knowledge_graph import KG
from dicee.models import DistMult, ComplEx, QMult, OMult, Keci
from dicee.static_preprocess_funcs import get_er_vocab, get_re_vocab

from dicee.continual.update_sequence import save_update_sequence
from dicee.continual.materialize import materialize_scenario_snapshots, load_materialized_sequence
from dicee.continual.graph_changes import GraphSnapshot
from dicee.continual.continual_trainer import ContinualTrainer, ContinualConfig
from dicee.continual.evaluation import evaluate_snapshot, preservation_metrics
from dicee.continual.experiments.report import build_report, render_report_text, save_report_json, save_losses_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Users may pick any of these via the "model" config key; DistMult is the
# default used by every example config. All of these share the same flat
# BaseKGE args dict (see `_make_model`); models that need extra per-model
# hyperparameters (e.g. Keci's Clifford-algebra `p`/`q`) read them from the
# optional "model_kwargs" config key.
MODEL_REGISTRY = {
    "DistMult": DistMult,
    "ComplEx": ComplEx,
    "QMult": QMult,
    "OMult": OMult,
    "Keci": Keci,
}

# Default values for every experiment parameter. These are used both to
# populate argparse defaults (so every knob is adjustable straight from the
# command line) and, when ``--config`` is given, as the base that the JSON
# config and then any explicitly-passed CLI flags are layered on top of.
DEFAULT_CONFIG: Dict = {
    "experiment_id": "experiment",
    "output_dir": "Experiments/continual",
    "dataset_dir": "KGs/UMLS",
    "separator": r"\s+",
    "model": "DistMult",
    "embedding_dim": 32,
    "model_kwargs": {},
    "seed": 0,
    "scenario": "mixed",
    "initial_fraction": 0.5,
    "num_steps": 3,
    "update_batch_size": None,
    "min_incident_facts": 1,
    "cheb_K": 3,
    "cheb_rho": 0.9,
    "cheb_epsilon": 1e-8,
    "beta": 2.0,
    "lambda_pres": 1.0,
    "margin": 1.0,
    "num_negatives": 1,
    "epochs_initial": 50,
    "epochs_per_step": 10,
    "lr": 0.05,
    "num_stable_queries": 256,
    "methods": ["finetune", "uniform", "chebyshev"],
    "include_oracle": True,
    "oracle_epochs": None,
    "save_report_json": None,
    "save_losses_csv": None,
    "dataset_cache_dir": "Experiments/continual_datasets",
    "force_regenerate_sequence": False,
}


def _make_model(model_name: str, num_entities: int, num_relations: int, embedding_dim: int,
                model_kwargs: Optional[Dict] = None):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_name]
    args = {
        "model": model_name, "embedding_dim": embedding_dim, "num_entities": num_entities,
        "num_relations": num_relations, "learning_rate": 0.05, "lr": 0.05, "optim": "Adam",
        "scoring_technique": "KvsAll", "input_dropout_rate": 0.0, "hidden_dropout_rate": 0.0,
        "weight_decay": 0.0, "normalization": None, "byte_pair_encoding": False,
    }
    args.update(model_kwargs or {})
    return cls(args)


def run_experiment(config: Dict) -> Dict:
    out_dir = Path(config["output_dir"]) / config["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    kg = KG(dataset_dir=config["dataset_dir"], separator=config.get("separator", r"\s+"),
            eval_model="train_val_test")

    # Section 5: continual sequences are built ONLY from the original
    # training split; valid/test triples are never touched here.
    #
    # The sequence is materialized as FIXED, on-disk, human-readable dataset
    # folders (one <dataset>_<step>_<scenario> folder per snapshot, each a
    # complete KG-loadable train/valid/test dataset_dir) keyed by
    # (dataset, scenario, num_steps, initial_fraction, seed, ...), so
    # repeated runs with the same configuration always read the exact same
    # triples back from disk instead of generating a new random partition
    # every time. See dicee/continual/experiments/prepare_dataset.py to
    # pre-build these ahead of time; they are built lazily here otherwise.
    scenario_dir = materialize_scenario_snapshots(
        dataset_dir=config["dataset_dir"],
        scenario=config["scenario"],
        num_steps=config["num_steps"],
        initial_fraction=config["initial_fraction"],
        seed=config["seed"],
        output_root=config.get("dataset_cache_dir", "Experiments/continual_datasets"),
        separator=config.get("separator", r"\s+"),
        update_batch_size=config.get("update_batch_size"),
        min_incident_facts=config.get("min_incident_facts", 1),
        force_regenerate=config.get("force_regenerate_sequence", False),
    )
    logger.info("Using fixed, on-disk dataset snapshots from %s", scenario_dir)
    t0_array, steps = load_materialized_sequence(str(scenario_dir), kg)
    t0 = GraphSnapshot.from_array(t0_array)
    save_update_sequence(str(out_dir / "sequence"), t0, steps, config=config)

    # Filtering vocabularies for evaluation always built from the FULL
    # original train+valid+test split (standard filtered-evaluation
    # protocol), independent of which update step we're at.
    full_triples = np.concatenate(
        [kg.train_set]
        + ([kg.valid_set] if kg.valid_set is not None else [])
        + ([kg.test_set] if kg.test_set is not None else []),
        axis=0,
    )
    er_vocab = get_er_vocab([tuple(int(x) for x in row) for row in full_triples])
    re_vocab = get_re_vocab([tuple(int(x) for x in row) for row in full_triples])

    all_results: Dict[str, List[Dict]] = {}
    model_kwargs = config.get("model_kwargs", {})
    initial_model = _make_model(config["model"], kg.num_entities, kg.num_relations, config["embedding_dim"], model_kwargs)

    init_trainer = ContinualTrainer(initial_model, ContinualConfig(epochs_per_step=config["epochs_initial"]))
    t_start = time.time()
    init_train_result = init_trainer.train_initial(t0.to_array())
    init_metrics = evaluate_snapshot(initial_model, kg.test_set, er_vocab, re_vocab)
    init_metrics["training_time_seconds"] = time.time() - t_start
    init_metrics["losses"] = init_train_result.get("losses") if isinstance(init_train_result, dict) else None
    logger.info("Initial model trained on T0 (%s triples): %s", len(t0), init_metrics)

    with open(out_dir / "initial_metrics.json", "w") as f:
        json.dump(init_metrics, f, indent=2)
    torch.save(initial_model.state_dict(), out_dir / "model_t0.pt")

    rng_master = np.random.default_rng(config["seed"])

    for method in config["methods"]:
        model = _make_model(config["model"], kg.num_entities, kg.num_relations, config["embedding_dim"], model_kwargs)
        model.load_state_dict(initial_model.state_dict())

        cfg = ContinualConfig(
            method=method, lambda_pres=config["lambda_pres"], beta=config["beta"],
            cheb_K=config["cheb_K"], cheb_rho=config["cheb_rho"], cheb_epsilon=config["cheb_epsilon"],
            epochs_per_step=config["epochs_per_step"], lr=config["lr"],
            num_stable_queries=config["num_stable_queries"], seed=config["seed"],
            margin=config["margin"], num_negatives=config["num_negatives"],
        )
        trainer = ContinualTrainer(model, cfg)
        trainer.optimizer = trainer._make_optimizer()

        method_results = []
        rng = np.random.default_rng(config["seed"])
        for step in steps:
            new_num_entities = max(kg.num_entities, int(step.cumulative_triples[:, [0, 2]].max()) + 1)
            step_result = trainer.run_step(step, new_num_entities, t0.to_array(), rng)

            eval_metrics = evaluate_snapshot(model, kg.test_set, er_vocab, re_vocab)

            # Preservation metrics vs. the model *before this step*.
            stable_pool = t0.to_array()  # earliest-known facts as the stable reference pool
            pres = preservation_metrics(model, trainer._last_model_old, stable_pool[:config["num_stable_queries"]],
                                         er_vocab, re_vocab)

            step_result.update({f"test_{k}": v for k, v in eval_metrics.items()})
            step_result.update(pres)
            method_results.append(step_result)
            logger.info("[%s] step %s: test_MRR=%.4f Delta_MRR_stable=%.4f",
                        method, step.step, eval_metrics["MRR"], pres["Delta_MRR_stable"])

        all_results[method] = method_results
        with open(out_dir / f"metrics_{method}.json", "w") as f:
            json.dump(method_results, f, indent=2)
        torch.save(model.state_dict(), out_dir / f"model_final_{method}.pt")

    # ------------------------------------------------------------------
    # Oracle baseline: retrain a FRESH model from scratch on the full
    # cumulative snapshot T_i at every step (no weight carry-over, no
    # preservation loss -- this is NOT a continual-learning method, it is
    # the reference point that answers "what if we didn't care about
    # compute cost and just retrained on everything seen so far?".
    # Useful to judge whether continual methods are closing the gap to
    # what full retraining can achieve, not just beating each other.
    # ------------------------------------------------------------------
    if config.get("include_oracle", True):
        oracle_epochs = config.get("oracle_epochs") or config["epochs_initial"]
        oracle_results = []
        for step in steps:
            t_start = time.time()
            oracle_model = _make_model(config["model"], kg.num_entities, kg.num_relations, config["embedding_dim"], model_kwargs)
            oracle_trainer = ContinualTrainer(oracle_model, ContinualConfig(epochs_per_step=oracle_epochs, seed=config["seed"]))
            oracle_train_result = oracle_trainer.train_initial(step.cumulative_triples)
            train_time = time.time() - t_start

            eval_metrics = evaluate_snapshot(oracle_model, kg.test_set, er_vocab, re_vocab)
            step_result = {
                "step": step.step,
                "method": "retrain_from_scratch",
                "num_added_triples": step.num_added_triples,
                "num_new_entities": step.num_new_entities,
                "training_time_seconds": train_time,
                "num_trainable_parameters": sum(p.numel() for p in oracle_model.parameters()),
                "losses": oracle_train_result.get("losses") if isinstance(oracle_train_result, dict) else None,
            }
            step_result.update({f"test_{k}": v for k, v in eval_metrics.items()})
            oracle_results.append(step_result)
            logger.info("[retrain_from_scratch] step %s: test_MRR=%.4f (trained %s epochs on %s triples, %.1fs)",
                        step.step, eval_metrics["MRR"], oracle_epochs, len(step.cumulative_triples), train_time)

        all_results["retrain_from_scratch"] = oracle_results
        with open(out_dir / "metrics_retrain_from_scratch.json", "w") as f:
            json.dump(oracle_results, f, indent=2)

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ------------------------------------------------------------------
    # Print the per-snapshot comparison report (entities added/removed,
    # base/oracle vs. each distillation method's performance, runtime,
    # and per-epoch loss decay) so results are visible right after the run
    # finishes, without a separate invocation of report.py.
    # ------------------------------------------------------------------
    report = build_report(str(out_dir))
    print("\n" + render_report_text(report))

    if config.get("save_report_json"):
        save_report_json(report, config["save_report_json"])
        logger.info("Saved structured report to %s", config["save_report_json"])

    if config.get("save_losses_csv"):
        save_losses_csv(report, config["save_losses_csv"])
        logger.info("Saved per-epoch loss curves to %s", config["save_losses_csv"])

    return {"initial_metrics": init_metrics, "results": all_results, "output_dir": str(out_dir), "report": report}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controlled continual-KGE experiment. All parameters can be passed "
                    "directly as CLI flags; a --config JSON file is optional and, if given, is "
                    "used as the base config (CLI flags still override it).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Optional path to a base JSON experiment config.")

    parser.add_argument("--experiment_id", default=None, help="Name of this run; also the output subfolder name.")
    parser.add_argument("--output_dir", default=None, help="Root directory experiment outputs are written under.")
    parser.add_argument("--dataset_dir", default=None, help="Path to a dataset_dir with train/valid/test.txt.")
    parser.add_argument("--separator", default=None, help="Column separator used to read the triple files.")

    parser.add_argument("--model", default=None, choices=sorted(MODEL_REGISTRY),
                         help="Base KGE model to use (default: DistMult).")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension.")
    parser.add_argument("--model_kwargs", default=None,
                         help='Extra model-specific hyperparameters as a JSON object string, '
                              'e.g. \'{"p": 0, "q": 1}\' for Keci.')

    parser.add_argument("--seed", type=int, default=None, help="Random seed for the update sequence and training.")
    parser.add_argument("--scenario", default=None, choices=["existing_entities", "new_entities", "mixed"],
                         help="Update-sequence scenario.")
    parser.add_argument("--initial_fraction", type=float, default=None,
                         help="Fraction of train triples placed in T0 (the rest is split across update steps).")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of update steps after T0.")
    parser.add_argument("--update_batch_size", type=int, default=None,
                         help="Fixed number of triples per update step (overrides even splitting if set).")
    parser.add_argument("--min_incident_facts", type=int, default=None,
                         help="Minimum incident facts required to keep an entity active.")

    parser.add_argument("--cheb_K", type=int, default=None, help="Chebyshev polynomial order.")
    parser.add_argument("--cheb_rho", type=float, default=None, help="Chebyshev spectral radius estimate.")
    parser.add_argument("--cheb_epsilon", type=float, default=None, help="Chebyshev numerical stability epsilon.")
    parser.add_argument("--beta", type=float, default=None, help="Influence-score temperature/sharpness.")
    parser.add_argument("--lambda_pres", type=float, default=None, help="Preservation-loss weight.")
    parser.add_argument("--margin", type=float, default=None, help="Margin (gamma) for the L_add filtered-negative hinge loss.")
    parser.add_argument("--num_negatives", type=int, default=None, help="Filtered negatives sampled per added positive triple for L_add.")

    parser.add_argument("--epochs_initial", type=int, default=None, help="Epochs to train the initial T0 model.")
    parser.add_argument("--epochs_per_step", type=int, default=None, help="Epochs to train each update step.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for continual training steps.")
    parser.add_argument("--num_stable_queries", type=int, default=None,
                         help="Number of stable (old) queries used for preservation metrics.")

    parser.add_argument("--methods", nargs="+", default=None, choices=["finetune", "uniform", "chebyshev"],
                         help="Which continual-learning baselines to run.")
    parser.add_argument("--include_oracle", dest="include_oracle", action="store_true", default=None,
                         help="Also run the retrain-from-scratch oracle baseline at every step.")
    parser.add_argument("--no_oracle", dest="include_oracle", action="store_false",
                         help="Skip the retrain-from-scratch oracle baseline.")
    parser.add_argument("--oracle_epochs", type=int, default=None,
                         help="Epochs for the oracle baseline (default: same as --epochs_initial).")
    parser.add_argument("--save_report_json", default=None,
                         help="Optional path to also save the structured comparison report as JSON.")
    parser.add_argument("--save_losses_csv", default=None,
                         help="Optional path to also save per-epoch loss curves (all methods/steps) as CSV.")
    parser.add_argument("--dataset_cache_dir", default=None,
                         help="Root directory where pre-generated/cached update sequences are looked up and "
                              "stored (see prepare_dataset.py). Reused automatically if a matching sequence "
                              "already exists there; built and cached there if not.")
    parser.add_argument("--force_regenerate_sequence", action="store_true", default=None,
                         help="Ignore any cached update sequence and rebuild it (overwriting the cache entry).")
    return parser


def _resolve_config(args: argparse.Namespace) -> Dict:
    """Merge DEFAULT_CONFIG <- (optional) --config JSON <- explicit CLI flags."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)

    cli_overrides = {
        "experiment_id": args.experiment_id,
        "output_dir": args.output_dir,
        "dataset_dir": args.dataset_dir,
        "separator": args.separator,
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "seed": args.seed,
        "scenario": args.scenario,
        "initial_fraction": args.initial_fraction,
        "num_steps": args.num_steps,
        "update_batch_size": args.update_batch_size,
        "min_incident_facts": args.min_incident_facts,
        "cheb_K": args.cheb_K,
        "cheb_rho": args.cheb_rho,
        "cheb_epsilon": args.cheb_epsilon,
        "beta": args.beta,
        "lambda_pres": args.lambda_pres,
        "margin": args.margin,
        "num_negatives": args.num_negatives,
        "epochs_initial": args.epochs_initial,
        "epochs_per_step": args.epochs_per_step,
        "lr": args.lr,
        "num_stable_queries": args.num_stable_queries,
        "methods": args.methods,
        "include_oracle": args.include_oracle,
        "oracle_epochs": args.oracle_epochs,
        "save_report_json": args.save_report_json,
        "save_losses_csv": args.save_losses_csv,
        "dataset_cache_dir": args.dataset_cache_dir,
        "force_regenerate_sequence": args.force_regenerate_sequence,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    if args.model_kwargs is not None:
        config["model_kwargs"] = json.loads(args.model_kwargs)

    missing_required = [k for k in ("experiment_id", "dataset_dir") if not config.get(k)]
    if missing_required:
        raise ValueError(
            f"Missing required parameter(s): {missing_required}. Pass them as CLI flags "
            f"(e.g. --experiment_id my_run --dataset_dir KGs/UMLS) or via --config."
        )
    return config


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _resolve_config(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
