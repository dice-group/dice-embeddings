"""Run a controlled continual-KGE experiment on the FastKGE ``FB_CKGE`` benchmark.

``FastKGE/data/FB_CKGE`` ships its own pre-built continual-learning split of
FB15k-237 (5 snapshot folders, each a delta of newly-introduced facts --
see ``dicee.continual.experiments.fb_ckge_loader`` for the full format
description). This script is the ``run_experiment.py`` equivalent for that
benchmark: it loads the fixed snapshot sequence directly (no scenario
generation / materialization needed, since FB_CKGE already *is* a
materialized sequence), trains the requested baselines
(``finetune`` / ``uniform`` / ``chebyshev``, optionally plus the
retrain-from-scratch oracle) on the exact same sequence + initial
checkpoint, and writes the same artifacts ``run_experiment.py`` does
(``config.json``, ``sequence/sequence_metadata.json``,
``initial_metrics.json``, ``metrics_<method>.json``, ``all_metrics.json``)
so the existing report/analysis tooling
(``dicee.continual.experiments.report``) works unmodified.

Usage
-----
    python -m dicee.continual.experiments.run_fb_ckge \\
        --experiment_id fb_ckge_distmult --model DistMult --embedding_dim 100 \\
        --methods finetune uniform chebyshev --device cuda

    # Point at a different checkout location / only run a subset of methods:
    python -m dicee.continual.experiments.run_fb_ckge \\
        --fb_ckge_root FastKGE/data/FB_CKGE --experiment_id fb_ckge_keci \\
        --model Keci --model_kwargs '{"p": 0, "q": 1}' --methods uniform chebyshev
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from dicee.static_preprocess_funcs import get_er_vocab, get_re_vocab

from dicee.continual.update_sequence import save_update_sequence
from dicee.continual.continual_trainer import ContinualTrainer, ContinualConfig
from dicee.continual.evaluation import evaluate_snapshot, preservation_metrics
from dicee.continual.experiments.report import build_report, render_report_text, save_report_json, save_losses_csv
from dicee.continual.experiments.fb_ckge_loader import load_fb_ckge_sequence
from dicee.continual.experiments.run_experiment import MODEL_REGISTRY, _make_model, _load_or_train_initial_model
from dicee.continual.reciprocal import add_reciprocal_triples, effective_num_relations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict = {
    "experiment_id": "fb_ckge_experiment",
    "output_dir": "Experiments/continual",
    "fb_ckge_root": "FastKGE/data/FB_CKGE",
    "model": "DistMult",
    "embedding_dim": 100,
    "model_kwargs": {},
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cheb_K": 3,
    "cheb_rho": 0.9,
    "cheb_epsilon": 1e-8,
    "beta": 2.0,
    "lambda_pres": 1.0,
    "margin": 1.0,
    "num_negatives": 10,
    "epochs_initial": 50,
    "epochs_per_step": 10,
    "lr_initial": 0.05,  # T0 training AND retrain_from_scratch oracle (KvsAll/BCE).
    "lr": 0.05,          # continual update steps only (margin-ranking L_add).
    "num_stable_queries": 256,
    "methods": ["finetune", "uniform", "chebyshev"],
    "include_oracle": True,
    "oracle_epochs": None,
    "save_report_json": None,
    "save_losses_csv": None,
    "initial_checkpoint": None,
    "force_initial_train": False,
    # Safety valve against margin-ranking loss explosion during continual
    # steps -- see run_experiment.DEFAULT_CONFIG for the full rationale.
    # Disabled by default (same root-cause history as DistMult/FB_CKGE);
    # enable for models with an unbounded score (e.g. Keci) if the loss
    # diverges across update steps.
    "normalize_embeddings": False,
    "embedding_max_norm": None,
    "grad_clip_norm": 1.0,
    "use_reciprocal_relations": False,
    "early_stopping_patience": None,
    "early_stopping_min_delta": 1e-4,
    "two_stage_epochs": 0,
}


def run_fb_ckge_experiment(config: Dict) -> Dict:
    out_dir = Path(config["output_dir"]) / config["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)

    t0, steps, num_entities, num_relations, full_filter_triples, cumulative_test = \
        load_fb_ckge_sequence(config["fb_ckge_root"])

    # Optional reciprocal-relation augmentation (FastKGE/IncDE-style): every
    # triple (h, r, t) is trained together with its inverse (t, r+R, h) using
    # a doubled relation vocabulary. Only touches TRAINING data (T0 and every
    # step's added/cumulative triples); evaluation keeps using the original
    # (non-reciprocal) relation ids, see dicee.continual.reciprocal.
    use_reciprocal = config.get("use_reciprocal_relations", False)
    model_num_relations = effective_num_relations(num_relations, use_reciprocal)
    if use_reciprocal:
        t0_array = add_reciprocal_triples(t0.to_array(), num_relations)
        from dicee.continual.graph_changes import GraphSnapshot as _GraphSnapshot
        t0 = _GraphSnapshot.from_array(t0_array)
        for step in steps:
            step.added_triples = add_reciprocal_triples(step.added_triples, num_relations)
            step.cumulative_triples = add_reciprocal_triples(step.cumulative_triples, num_relations)
        logger.info("Reciprocal relations enabled: num_relations %s -> %s", num_relations, model_num_relations)

    save_update_sequence(str(out_dir / "sequence"), t0, steps, config=config)

    # Standard filtered-evaluation protocol: filtering vocab built from
    # EVERY triple ever seen across the whole benchmark (all train/valid/test
    # splits of all snapshots), independent of which step is being evaluated.
    er_vocab = get_er_vocab([tuple(int(x) for x in row) for row in full_filter_triples])
    re_vocab = get_re_vocab([tuple(int(x) for x in row) for row in full_filter_triples])

    all_results: Dict[str, List[Dict]] = {}
    model_kwargs = config.get("model_kwargs", {})
    initial_model, init_metrics = _load_or_train_initial_model(
        config=config, out_dir=out_dir, model_name=config["model"], num_entities=num_entities,
        num_relations=model_num_relations, embedding_dim=config["embedding_dim"], model_kwargs=model_kwargs,
        device=device, t0_triples=t0.to_array(), test_triples=cumulative_test[0], er_vocab=er_vocab,
        re_vocab=re_vocab, num_triples_for_log=len(t0), val_triples=cumulative_test[0],
    )

    rng_master = np.random.default_rng(config["seed"])

    for method in tqdm(config["methods"], desc="Methods", unit="method"):
        model = _make_model(config["model"], num_entities, model_num_relations, config["embedding_dim"],
                             model_kwargs, device=device, lr=config["lr"])
        model.load_state_dict(initial_model.state_dict())

        cfg = ContinualConfig(
            method=method, lambda_pres=config["lambda_pres"], beta=config["beta"],
            cheb_K=config["cheb_K"], cheb_rho=config["cheb_rho"], cheb_epsilon=config["cheb_epsilon"],
            epochs_per_step=config["epochs_per_step"], lr=config["lr"],
            num_stable_queries=config["num_stable_queries"], seed=config["seed"],
            margin=config["margin"], num_negatives=config["num_negatives"], device=device,
            normalize_embeddings=config.get("normalize_embeddings", False),
            embedding_max_norm=config.get("embedding_max_norm"),
            grad_clip_norm=config.get("grad_clip_norm", 1.0),
            two_stage_epochs=config.get("two_stage_epochs", 0),
        )
        trainer = ContinualTrainer(model, cfg)
        trainer.optimizer = trainer._make_optimizer()

        method_results = []
        rng = np.random.default_rng(config["seed"])
        # The "stable"/preservation reference pool must be every fact known
        # BEFORE the current step's update -- NOT always T0. Preservation
        # loss only ever protects facts present in this pool; if it stayed
        # fixed at T0 forever, every fact learned at steps 1..i-1 would get
        # zero preservation signal once we're training step i, causing rapid
        # catastrophic forgetting of everything except the original T0
        # facts (exactly the large finetune/uniform/chebyshev vs. retrain gap
        # observed on FB_CKGE/WN_CKGE). This pool grows after every step to
        # that step's full cumulative snapshot.
        known_pool = t0.to_array()
        step_pbar = tqdm(steps, desc=f"[{method}] steps", unit="step", leave=False)
        for step in step_pbar:
            new_num_entities = max(num_entities, int(step.cumulative_triples[:, [0, 2]].max()) + 1)
            step_result = trainer.run_step(step, new_num_entities, known_pool, rng)

            eval_metrics = evaluate_snapshot(model, cumulative_test[step.step], er_vocab, re_vocab)

            # Preservation metrics vs. the model *before this step*, sampled
            # from the same growing known-facts pool used for training.
            n_pres = min(config["num_stable_queries"], len(known_pool))
            pres_idx = rng.choice(len(known_pool), size=n_pres, replace=False) if n_pres > 0 else np.array([], dtype=np.int64)
            pres = preservation_metrics(model, trainer._last_model_old, known_pool[pres_idx],
                                         er_vocab, re_vocab)

            step_result.update({f"test_{k}": v for k, v in eval_metrics.items()})
            step_result.update(pres)
            method_results.append(step_result)
            step_pbar.set_postfix(MRR=f"{eval_metrics['MRR']:.4f}", d_MRR=f"{pres['Delta_MRR_stable']:.4f}")
            logger.info("[%s] step %s: test_MRR=%.4f Delta_MRR_stable=%.4f",
                        method, step.step, eval_metrics["MRR"], pres["Delta_MRR_stable"])
            known_pool = step.cumulative_triples  # everything known going into the NEXT step

        all_results[method] = method_results
        with open(out_dir / f"metrics_{method}.json", "w") as f:
            json.dump(method_results, f, indent=2)
        torch.save(model.state_dict(), out_dir / f"model_final_{method}.pt")

    if config.get("include_oracle", True):
        oracle_epochs = config.get("oracle_epochs") or config["epochs_initial"]
        oracle_results = []
        oracle_pbar = tqdm(steps, desc="[retrain_from_scratch] steps", unit="step", leave=False)
        for step in oracle_pbar:
            t_start = time.time()
            oracle_model = _make_model(config["model"], num_entities, model_num_relations, config["embedding_dim"],
                                        model_kwargs, device=device, lr=config["lr_initial"])
            oracle_trainer = ContinualTrainer(oracle_model, ContinualConfig(epochs_per_step=oracle_epochs,
                                                                             lr=config["lr_initial"],
                                                                             seed=config["seed"], device=device))
            oracle_train_result = oracle_trainer.train_initial(step.cumulative_triples)
            train_time = time.time() - t_start

            eval_metrics = evaluate_snapshot(oracle_model, cumulative_test[step.step], er_vocab, re_vocab)
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
            oracle_pbar.set_postfix(MRR=f"{eval_metrics['MRR']:.4f}")
            logger.info("[retrain_from_scratch] step %s: test_MRR=%.4f (trained %s epochs on %s triples, %.1fs)",
                        step.step, eval_metrics["MRR"], oracle_epochs, len(step.cumulative_triples), train_time)

        all_results["retrain_from_scratch"] = oracle_results
        with open(out_dir / "metrics_retrain_from_scratch.json", "w") as f:
            json.dump(oracle_results, f, indent=2)

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

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
        description="Run a controlled continual-KGE experiment on the FastKGE FB_CKGE benchmark "
                    "(its own pre-built 5-snapshot FB15k-237 continual split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Optional path to a base JSON experiment config.")
    parser.add_argument("--experiment_id", default=None, help="Name of this run; also the output subfolder name.")
    parser.add_argument("--output_dir", default=None, help="Root directory experiment outputs are written under.")
    parser.add_argument("--fb_ckge_root", default=None,
                         help="Path to the FB_CKGE directory (containing numbered subfolders 0, 1, 2, ...).")

    parser.add_argument("--model", default=None, choices=sorted(MODEL_REGISTRY),
                         help="Base KGE model to use (default: DistMult).")
    parser.add_argument("--embedding_dim", type=int, default=None, help="Embedding dimension.")
    parser.add_argument("--model_kwargs", default=None,
                         help='Extra model-specific hyperparameters as a JSON object string, '
                              'e.g. \'{"p": 0, "q": 1}\' for Keci.')

    parser.add_argument("--seed", type=int, default=None, help="Random seed for training.")
    parser.add_argument("--device", default=None, help="Device to train/evaluate on, e.g. 'cuda', 'cuda:0', or 'cpu' "
                         "(default: 'cuda' if available, else 'cpu').")

    parser.add_argument("--cheb_K", type=int, default=None, help="Chebyshev polynomial order.")
    parser.add_argument("--cheb_rho", type=float, default=None, help="Chebyshev spectral radius estimate.")
    parser.add_argument("--cheb_epsilon", type=float, default=None, help="Chebyshev numerical stability epsilon.")
    parser.add_argument("--beta", type=float, default=None, help="Influence-score temperature/sharpness.")
    parser.add_argument("--lambda_pres", type=float, default=None, help="Preservation-loss weight.")
    parser.add_argument("--margin", type=float, default=None, help="Margin (gamma) for the L_add filtered-negative hinge loss.")
    parser.add_argument("--num_negatives", type=int, default=None, help="Filtered negatives sampled per added positive triple for L_add.")

    parser.add_argument("--epochs_initial", type=int, default=None, help="Epochs to train the initial T0 model.")
    parser.add_argument("--epochs_per_step", type=int, default=None, help="Epochs to train each update step.")
    parser.add_argument("--lr_initial", type=float, default=None,
                         help="Learning rate for T0 training AND the retrain_from_scratch oracle "
                              "(KvsAll/BCE objective; deliberately separate from --lr, see DEFAULT_CONFIG).")
    parser.add_argument("--lr", type=float, default=None,
                         help="Learning rate for continual update steps only (margin-ranking L_add objective).")
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
    parser.add_argument("--initial_checkpoint", default=None,
                         help="Path to the T0 model checkpoint to load/save (default: "
                              "<output_dir>/<experiment_id>/model_t0.pt). If it exists and its metadata "
                              "(model/shapes/model_kwargs) matches the current config, T0 training is "
                              "skipped entirely and the checkpoint is loaded directly.")
    parser.add_argument("--force_initial_train", action="store_true", default=None,
                         help="Always retrain the T0 model from scratch, even if a matching checkpoint "
                              "already exists at --initial_checkpoint (overwrites it afterwards).")
    parser.add_argument("--normalize_embeddings", action="store_true", default=None,
                         help="Enable the adaptive embedding-norm safety valve during continual steps "
                              "(clips rows exceeding 4x the table's own current max row norm after every "
                              "optimizer step). Off by default; turn on for models whose score is not "
                              "naturally bounded (e.g. Keci) if you see the margin-ranking loss diverge "
                              "across update steps instead of decaying.")
    parser.add_argument("--embedding_max_norm", type=float, default=None,
                         help="Fixed max row norm for the safety valve above, overriding the adaptive "
                              "4x-current-max-norm default. Only used when --normalize_embeddings is set.")
    parser.add_argument("--grad_clip_norm", type=float, default=None,
                         help="Gradient norm clipping threshold applied during every continual step "
                              "(default: 1.0). Set to a negative value or very large number to disable.")
    parser.add_argument("--use_reciprocal_relations", action="store_true", default=None,
                         help="FastKGE/IncDE-style training augmentation: train every triple (h,r,t) "
                              "together with its inverse (t, r+num_relations, h) using a doubled relation "
                              "vocabulary. Only affects training data; evaluation is unchanged. Off by default.")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                         help="FastKGE/IncDE-style patience-based early stopping for T0 training (and the "
                              "retrain-from-scratch oracle): stop once this many consecutive epochs fail to "
                              "improve validation MRR, restoring the best-val-MRR epoch's weights. "
                              "--epochs_initial then acts as a maximum-epoch cap. None (default) disables "
                              "early stopping (train for exactly --epochs_initial epochs).")
    parser.add_argument("--early_stopping_min_delta", type=float, default=None,
                         help="Minimum validation-MRR improvement to reset the early-stopping patience counter "
                              "(default: 1e-4).")
    parser.add_argument("--two_stage_epochs", type=int, default=None,
                         help="IncDE-style two-stage training: for this many epochs at the START of EACH "
                              "continual update step, freeze gradients for entity rows that already existed "
                              "before the step (only new entity rows + all relations update); joint "
                              "fine-tuning of old+new rows resumes afterwards. 0 (default) disables this.")
    return parser


def _resolve_config(args: argparse.Namespace) -> Dict:
    config = copy.deepcopy(DEFAULT_CONFIG)

    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)

    cli_overrides = {
        "experiment_id": args.experiment_id,
        "output_dir": args.output_dir,
        "fb_ckge_root": args.fb_ckge_root,
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "seed": args.seed,
        "device": args.device,
        "cheb_K": args.cheb_K,
        "cheb_rho": args.cheb_rho,
        "cheb_epsilon": args.cheb_epsilon,
        "beta": args.beta,
        "lambda_pres": args.lambda_pres,
        "margin": args.margin,
        "num_negatives": args.num_negatives,
        "epochs_initial": args.epochs_initial,
        "epochs_per_step": args.epochs_per_step,
        "lr_initial": args.lr_initial,
        "lr": args.lr,
        "num_stable_queries": args.num_stable_queries,
        "methods": args.methods,
        "include_oracle": args.include_oracle,
        "oracle_epochs": args.oracle_epochs,
        "save_report_json": args.save_report_json,
        "save_losses_csv": args.save_losses_csv,
        "initial_checkpoint": args.initial_checkpoint,
        "force_initial_train": args.force_initial_train,
        "normalize_embeddings": args.normalize_embeddings,
        "embedding_max_norm": args.embedding_max_norm,
        "grad_clip_norm": args.grad_clip_norm,
        "use_reciprocal_relations": args.use_reciprocal_relations,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "two_stage_epochs": args.two_stage_epochs,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    if args.model_kwargs is not None:
        config["model_kwargs"] = json.loads(args.model_kwargs)

    missing_required = [k for k in ("experiment_id", "fb_ckge_root") if not config.get(k)]
    if missing_required:
        raise ValueError(
            f"Missing required parameter(s): {missing_required}. Pass them as CLI flags "
            f"(e.g. --experiment_id fb_ckge_run --fb_ckge_root FastKGE/data/FB_CKGE) or via --config."
        )
    return config


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = _resolve_config(args)
    run_fb_ckge_experiment(config)


if __name__ == "__main__":
    main()
