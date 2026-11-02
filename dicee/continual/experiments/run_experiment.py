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
from tqdm.auto import tqdm

from dicee.knowledge_graph import KG
from dicee.models import DistMult, ComplEx, QMult, OMult, Keci, TransE
from dicee.static_preprocess_funcs import get_er_vocab, get_re_vocab

from dicee.continual.update_sequence import save_update_sequence
from dicee.continual.materialize import materialize_scenario_snapshots, load_materialized_sequence
from dicee.continual.graph_changes import GraphSnapshot
from dicee.continual.continual_trainer import ContinualTrainer, ContinualConfig
from dicee.continual.evaluation import evaluate_snapshot, preservation_metrics
from dicee.continual.reciprocal import add_reciprocal_triples, effective_num_relations
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
    "TransE": TransE,
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
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "scenario": "mixed",
    "initial_fraction": 0.5,
    "num_steps": 3,
    "update_batch_size": None,
    "min_incident_facts": 1,
    "cheb_K": 3,
    "cheb_rho": 0.9,
    "cheb_epsilon": 0.1,#1e-8,
    "beta": 2.0,
    "lambda_pres": 1.0,
    "margin": 1.0,
    "num_negatives": 10,
    "epochs_initial": 50,
    "epochs_per_step": 10,
    # steps, which generally need different learning rates).
    "lr_initial": 0.05,  # T0 training AND retrain_from_scratch oracle.
    "lr": 0.05,          # continual update steps only.
    "num_stable_queries": 256,
    "methods": ["finetune", "uniform", "chebyshev"],
    "include_oracle": True,
    "oracle_epochs": None,
    "save_report_json": None,
    "save_losses_csv": None,
    "dataset_cache_dir": "Experiments/continual_datasets",
    "force_regenerate_sequence": False,
    "initial_checkpoint": None,
    "force_initial_train": False,
    "normalize_embeddings": False,
    "embedding_max_norm": None,
    "grad_clip_norm": 1.0,
    "use_reciprocal_relations": False,
    "early_stopping_patience": None,
    "early_stopping_min_delta": 1e-4,
    "two_stage_epochs": 0,
}


def _make_model(model_name: str, num_entities: int, num_relations: int, embedding_dim: int,
                model_kwargs: Optional[Dict] = None, device: str = "cpu", lr: float = 0.05):
    """Build a fresh ``BaseKGE`` instance.

    ``lr`` is embedded into ``model.args['lr']``, which ``ContinualTrainer``
    falls back to whenever a ``ContinualConfig`` is built without an
    explicit ``lr`` (this is exactly what ``train_initial`` does for T0 and
    for the retrain-from-scratch oracle -- see ``ContinualTrainer._make_optimizer``).

    IMPORTANT (confirmed FB_CKGE root-cause fix): this previously hardcoded
    ``lr=0.05`` unconditionally, regardless of the experiment config's own
    ``"lr"`` field. Every continual-step trainer (finetune/uniform/chebyshev)
    explicitly passes ``config["lr"]`` into its ``ContinualConfig`` instead,
    so T0/retrain and the continual steps were silently trained with
    *different* learning rates (e.g. 0.05 vs the configured 0.1) -- on top
    of training under a different objective (KvsAll/BCE vs. margin-ranking
    ``L_add``, whose loss/gradient scale is much larger). Because Adam's
    per-parameter update magnitude is approximately ``lr`` regardless of
    the raw gradient scale, this mismatch caused the continual steps to
    apply far larger, repeated perturbations to the already-converged T0
    embeddings than retraining ever did, which was confirmed experimentally
    to be the dominant driver of the observed MRR collapse (lowering the
    continual-step lr alone, with everything else unchanged, recovered most
    of the lost MRR). Callers must now pass the SAME, single, explicitly
    configured ``lr`` to every ``_make_model`` call (T0, retrain/oracle,
    and every continual-method model) so the config's ``"lr"`` field is the
    one source of truth everywhere.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}; choose from {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_name]
    args = {
        "model": model_name, "embedding_dim": embedding_dim, "num_entities": num_entities,
        "num_relations": num_relations, "learning_rate": lr, "lr": lr, "optim": "Adam",
        "scoring_technique": "KvsAll", "input_dropout_rate": 0.0, "hidden_dropout_rate": 0.0,
        "weight_decay": 0.0, "normalization": None, "byte_pair_encoding": False,
    }
    args.update(model_kwargs or {})
    return cls(args).to(device)


def _initial_checkpoint_meta(model_name: str, num_entities: int, num_relations: int, embedding_dim: int,
                              model_kwargs: Optional[Dict]) -> Dict:
    """Fingerprint identifying which (model, shapes, kwargs) a T0 checkpoint was trained for.

    Saved as a JSON sidecar next to the checkpoint (``<checkpoint>.meta.json``)
    so a later run can cheaply verify the checkpoint is actually compatible
    with the CURRENT config before loading it (mismatched
    ``num_entities``/``num_relations`` would otherwise fail obscurely inside
    ``load_state_dict``, or silently load into the wrong-shaped embedding
    table if sizes happen to coincide but the dataset doesn't).
    """
    return {
        "model": model_name,
        "num_entities": int(num_entities),
        "num_relations": int(num_relations),
        "embedding_dim": int(embedding_dim),
        "model_kwargs": model_kwargs or {},
    }


def _load_or_train_initial_model(
    *, config: Dict, out_dir: Path, model_name: str, num_entities: int, num_relations: int,
    embedding_dim: int, model_kwargs: Optional[Dict], device: str, t0_triples: np.ndarray,
    test_triples: np.ndarray, er_vocab: Dict, re_vocab: Dict, num_triples_for_log: int,
    val_triples: Optional[np.ndarray] = None,
):
    """Return a (possibly cached) trained T0 model, training it only if needed.

    Checkpoint resolution order:

    1. If ``config["force_initial_train"]`` is set, always retrain (ignores
       any existing checkpoint, and overwrites it afterwards).
    2. Otherwise, if a checkpoint exists at ``config["initial_checkpoint"]``
       (default: ``<out_dir>/model_t0.pt``) with a matching metadata sidecar
       (same model/shapes/model_kwargs, see :func:`_initial_checkpoint_meta`),
       load it directly and skip T0 training entirely. Cached
       ``initial_metrics.json`` is reused if present (re-evaluating from a
       loaded checkpoint is otherwise pure wasted work); it is recomputed
       only if missing.
    3. Otherwise (no checkpoint, or checkpoint metadata mismatch), train a
       fresh T0 model and save both the checkpoint and its metadata sidecar.

    Returns
    -------
    Tuple[torch.nn.Module, Dict]
        ``(initial_model, init_metrics)``.
    """
    checkpoint_path = Path(config.get("initial_checkpoint") or (out_dir / "model_t0.pt"))
    meta_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
    force_retrain = bool(config.get("force_initial_train", False))
    expected_meta = _initial_checkpoint_meta(model_name, num_entities, num_relations, embedding_dim, model_kwargs)

    if not force_retrain and checkpoint_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            saved_meta = json.load(f)
        if saved_meta == expected_meta:
            initial_model = _make_model(model_name, num_entities, num_relations, embedding_dim,
                                         model_kwargs, device=device, lr=config["lr_initial"])
            initial_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info("Loaded cached T0 checkpoint from %s (skipping T0 training; pass "
                        "--force_initial_train to retrain from scratch).", checkpoint_path)
            init_metrics_path = out_dir / "initial_metrics.json"
            if init_metrics_path.exists():
                with open(init_metrics_path) as f:
                    init_metrics = json.load(f)
            else:
                init_metrics = evaluate_snapshot(initial_model, test_triples, er_vocab, re_vocab)
                init_metrics["training_time_seconds"] = 0.0
                init_metrics["losses"] = None
                with open(init_metrics_path, "w") as f:
                    json.dump(init_metrics, f, indent=2)
            return initial_model, init_metrics
        logger.info("Found T0 checkpoint at %s but its metadata does not match the current "
                    "config (model/shapes/model_kwargs changed) -- retraining from scratch. "
                    "saved=%s expected=%s", checkpoint_path, saved_meta, expected_meta)
    elif not force_retrain and checkpoint_path.exists() and not meta_path.exists():
        logger.info("Found T0 checkpoint at %s but no metadata sidecar (%s) to verify "
                    "compatibility -- retraining from scratch to be safe. Checkpoints saved by "
                    "this version of the script always include a sidecar; this likely means the "
                    "checkpoint predates this feature.", checkpoint_path, meta_path)

    initial_model = _make_model(model_name, num_entities, num_relations, embedding_dim,
                                 model_kwargs, device=device, lr=config["lr_initial"])
    init_trainer = ContinualTrainer(initial_model, ContinualConfig(
        epochs_per_step=config["epochs_initial"], lr=config["lr_initial"], device=device,
        early_stopping_patience=config.get("early_stopping_patience"),
        early_stopping_min_delta=config.get("early_stopping_min_delta", 1e-4),
    ))
    t_start = time.time()
    init_train_result = init_trainer.train_initial(
        t0_triples, val_triples=val_triples, val_er_vocab=er_vocab, val_re_vocab=re_vocab,
    )
    init_metrics = evaluate_snapshot(initial_model, test_triples, er_vocab, re_vocab)
    init_metrics["training_time_seconds"] = time.time() - t_start
    init_metrics["losses"] = init_train_result.get("losses") if isinstance(init_train_result, dict) else None
    logger.info("Initial model trained on T0 (%s triples): %s", num_triples_for_log, init_metrics)

    with open(out_dir / "initial_metrics.json", "w") as f:
        json.dump(init_metrics, f, indent=2)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(initial_model.state_dict(), checkpoint_path)
    with open(meta_path, "w") as f:
        json.dump(expected_meta, f, indent=2)
    return initial_model, init_metrics


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

    # Optional reciprocal-relation augmentation (FastKGE/IncDE-style): every
    # triple (h, r, t) is trained together with its inverse (t, r+R, h) using
    # a doubled relation vocabulary. This only touches TRAINING data (T0 and
    # every step's added/cumulative triples); evaluation keeps using the
    # original (non-reciprocal) relation ids, see dicee.continual.reciprocal.
    use_reciprocal = config.get("use_reciprocal_relations", False)
    model_num_relations = effective_num_relations(kg.num_relations, use_reciprocal)
    if use_reciprocal:
        t0_array = add_reciprocal_triples(t0_array, kg.num_relations)
        for step in steps:
            step.added_triples = add_reciprocal_triples(step.added_triples, kg.num_relations)
            step.cumulative_triples = add_reciprocal_triples(step.cumulative_triples, kg.num_relations)
        logger.info("Reciprocal relations enabled: num_relations %s -> %s", kg.num_relations, model_num_relations)

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
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)
    initial_model, init_metrics = _load_or_train_initial_model(
        config=config, out_dir=out_dir, model_name=config["model"], num_entities=kg.num_entities,
        num_relations=model_num_relations, embedding_dim=config["embedding_dim"], model_kwargs=model_kwargs,
        device=device, t0_triples=t0.to_array(), test_triples=kg.test_set, er_vocab=er_vocab, re_vocab=re_vocab,
        num_triples_for_log=len(t0), val_triples=kg.valid_set,
    )

    rng_master = np.random.default_rng(config["seed"])

    for method in tqdm(config["methods"], desc="Methods", unit="method"):
        model = _make_model(config["model"], kg.num_entities, model_num_relations, config["embedding_dim"],
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
            new_num_entities = max(kg.num_entities, int(step.cumulative_triples[:, [0, 2]].max()) + 1)
            step_result = trainer.run_step(step, new_num_entities, known_pool, rng)

            eval_metrics = evaluate_snapshot(model, kg.test_set, er_vocab, re_vocab)

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
        oracle_pbar = tqdm(steps, desc="[retrain_from_scratch] steps", unit="step", leave=False)
        for step in oracle_pbar:
            t_start = time.time()
            oracle_model = _make_model(config["model"], kg.num_entities, model_num_relations, config["embedding_dim"],
                                        model_kwargs, device=device, lr=config["lr_initial"])
            oracle_trainer = ContinualTrainer(oracle_model, ContinualConfig(epochs_per_step=oracle_epochs,
                                                                             lr=config["lr_initial"], seed=config["seed"], device=device))
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
            oracle_pbar.set_postfix(MRR=f"{eval_metrics['MRR']:.4f}")
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
    parser.add_argument("--device", default=None, help="Device to train/evaluate on, e.g. 'cuda', 'cuda:0', or 'cpu' "
                         "(default: 'cuda' if available, else 'cpu').")
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
    parser.add_argument("--dataset_cache_dir", default=None,
                         help="Root directory where pre-generated/cached update sequences are looked up and "
                              "stored (see prepare_dataset.py). Reused automatically if a matching sequence "
                              "already exists there; built and cached there if not.")
    parser.add_argument("--force_regenerate_sequence", action="store_true", default=None,
                         help="Ignore any cached update sequence and rebuild it (overwriting the cache entry).")
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
        "device": args.device,
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
        "lr_initial": args.lr_initial,
        "lr": args.lr,
        "num_stable_queries": args.num_stable_queries,
        "methods": args.methods,
        "include_oracle": args.include_oracle,
        "oracle_epochs": args.oracle_epochs,
        "save_report_json": args.save_report_json,
        "save_losses_csv": args.save_losses_csv,
        "dataset_cache_dir": args.dataset_cache_dir,
        "force_regenerate_sequence": args.force_regenerate_sequence,
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
