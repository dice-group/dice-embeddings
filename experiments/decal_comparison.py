"""
Systematic comparison of DeCaL vs FullDeCaL (fixed) vs FullDeCaL (auto).

Two research questions
----------------------
Q1 – Full Clifford vs DeCaL approximation  (embedding_dim=64 fixed)
     Does using the *complete* 2^n-blade product help over DeCaL's 1+n
     component approximation, at iso-parameter count?

Q2 – Auto-signature discovery across embedding dimensions
     Does FullDeCaL-auto find a good algebra regardless of how
     embedding_dim (and therefore n) changes?

Usage
-----
    cd dice-embeddings
    python experiments/decal_comparison.py               # full suite
    python experiments/decal_comparison.py --q1_only
    python experiments/decal_comparison.py --q2_only
    python experiments/decal_comparison.py --dry_run     # plan only
    python experiments/decal_comparison.py --skip_existing  # resume

Results are appended to  experiments/results/comparison_results.csv.
"""

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd

# Logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S", level=logging.WARNING)
logger = logging.getLogger("comparison")
logger.setLevel(logging.INFO)

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR    = RESULTS_DIR / "runs"
RESULTS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "comparison_results.csv"

DATASETS = {
    "UMLS":    str(ROOT / "KGs" / "UMLS"),
    "KINSHIP": str(ROOT / "KGs" / "KINSHIP"),
}

BASE = dict(
    trainer="torchCPUTrainer", scoring_technique="KvsAll",
    num_epochs=100, batch_size=128, lr=0.1, optim="Adopt",
    random_seed=42, eval_model="train_val_test",
    neg_ratio=2, weight_decay=0.0, input_dropout_rate=0.0,
    hidden_dropout_rate=0.0, feature_map_dropout_rate=0.0,
    normalization="None", init_param=None, gradient_accumulation_steps=0,
    label_smoothing_rate=0.0, kernel_size=3, num_of_output_channels=2,
    num_core=0, byte_pair_encoding=False, adaptive_swa=False,
    swa=False, swag=False, ema=False, twa=False,
    swa_start_epoch=None, swa_c_epochs=1, eval_every_n_epochs=0,
    save_every_n_epochs=False, eval_at_epochs=None, n_epochs_eval_model="val_test",
    auto_batch_finding=False, block_size=8, degree=0,
    pykeen_model_kwargs={}, pl_trainer_kwargs={}, fsdp_trainer_kwargs={},
    save_embeddings_as_csv=False, sample_triples_ratio=None,
    read_only_few=None, add_noise_rate=0.0, backend="pandas",
    separator=r"\s+", reuse_existing_run_dir=False, sparql_endpoint=None,
    path_single_kg=None, num_folds_for_cv=0, save_model_at_every_epoch=None,
    continual_learning=None, adaptive_lr={}, log_level="WARNING",
    callbacks={}, path_to_store_single_run=None,
)


@dataclass
class Experiment:
    label: str
    question: str
    dataset: str
    model: str
    embedding_dim: int
    p: int = 0
    q: int = 0
    r: int = 0
    auto_signature: bool = False

    @property
    def signature(self):
        return "auto" if self.auto_signature else f"Cl({self.p},{self.q},{self.r})"

    @property
    def run_id(self):
        sig = "auto" if self.auto_signature else f"p{self.p}q{self.q}r{self.r}"
        return f"{self.dataset}__{self.model}__{sig}__d{self.embedding_dim}"

    @property
    def run_dir(self):
        return RUNS_DIR / self.run_id


def make_experiments():
    exps = []

    # Q1: Full Clifford vs DeCaL at dim=64, 5 signatures
    # Parameter count parity:
    #   DeCaL  n=3: 4 component vectors x re=16 = 64 params / embedding
    #   FullDeCaL n=3: 8 blade vectors x re=8 = 64 params / embedding
    DIM_Q1 = 64
    Q1_SIGS = [(3,0,0), (0,3,0), (1,1,1), (2,1,0), (0,0,3)]

    for ds in DATASETS:
        for (p,q,r) in Q1_SIGS:
            exps.append(Experiment(
                f"DeCaL Cl({p},{q},{r})", "Q1", ds, "DeCaL", DIM_Q1, p, q, r))
            exps.append(Experiment(
                f"FullDeCaL-fixed Cl({p},{q},{r})", "Q1", ds, "FullDeCaL", DIM_Q1,
                p, q, r, auto_signature=False))
        exps.append(Experiment(
            "FullDeCaL-auto", "Q1", ds, "FullDeCaL", DIM_Q1, auto_signature=True))

    # Q2: Auto-signature across dims 32,64,128,256
    # Auto n: dim=32->n=2,d=4,re=8 | dim=64->n=3,d=8,re=8
    #         dim=128->n=3,d=8,re=16 | dim=256->n=4,d=16,re=16
    # Baselines: FullDeCaL-fixed Cl(1,1,1) and DeCaL Cl(1,1,1)
    for ds in DATASETS:
        for dim in [32, 64, 128, 256]:
            exps.append(Experiment(
                f"FullDeCaL-auto d{dim}", "Q2", ds, "FullDeCaL", dim, auto_signature=True))
            exps.append(Experiment(
                f"FullDeCaL-fixed Cl(1,1,1) d{dim}", "Q2", ds, "FullDeCaL", dim,
                p=1, q=1, r=1, auto_signature=False))
            exps.append(Experiment(
                f"DeCaL Cl(1,1,1) d{dim}", "Q2", ds, "DeCaL", dim, p=1, q=1, r=1))

    return exps


def run_one(exp: Experiment, num_epochs: int) -> Optional[dict]:
    from dicee.executer import Execute

    if exp.run_dir.exists():
        shutil.rmtree(exp.run_dir)

    kwargs = {**BASE}
    kwargs.update(
        dataset_dir=DATASETS[exp.dataset], model=exp.model,
        embedding_dim=exp.embedding_dim, p=exp.p, q=exp.q, r=exp.r,
        auto_signature=exp.auto_signature, num_epochs=num_epochs,
        path_to_store_single_run=str(exp.run_dir),
    )
    ns = SimpleNamespace(**kwargs)

    t0 = time.time()
    try:
        Execute(ns).start()
    except Exception as exc:
        logger.error(f"  FAILED: {exc}")
        return None
    elapsed = time.time() - t0

    report_path = exp.run_dir / "eval_report.json"
    if not report_path.exists():
        logger.error(f"  No eval_report.json in {exp.run_dir}")
        return None

    with open(report_path) as fh:
        report = json.load(fh)

    # Also read inferred signature for FullDeCaL auto
    inferred_sig = None
    if exp.auto_signature:
        config_path = exp.run_dir / "configuration.json"
        if config_path.exists():
            cfg = json.load(open(config_path))
            inferred_sig = cfg.get("learned_signature")

    row = dict(
        run_id=exp.run_id, question=exp.question, dataset=exp.dataset,
        model=exp.model, signature=exp.signature, embedding_dim=exp.embedding_dim,
        label=exp.label, elapsed_s=round(elapsed, 1),
        inferred_signature=inferred_sig,
    )
    for split, metrics in report.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                row[f"{split}_{metric}"] = round(float(value), 5)
    return row


def main():
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--q1_only",       action="store_true")
    cli.add_argument("--q2_only",       action="store_true")
    cli.add_argument("--dry_run",       action="store_true")
    cli.add_argument("--epochs",        type=int, default=100)
    cli.add_argument("--datasets",      nargs="+", choices=list(DATASETS))
    cli.add_argument("--skip_existing", action="store_true")
    args = cli.parse_args()

    exps = make_experiments()
    if args.q1_only:  exps = [e for e in exps if e.question == "Q1"]
    if args.q2_only:  exps = [e for e in exps if e.question == "Q2"]
    if args.datasets: exps = [e for e in exps if e.dataset in args.datasets]

    done: set = set()
    if args.skip_existing and RESULTS_CSV.exists():
        done = set(pd.read_csv(RESULTS_CSV)["run_id"].tolist())

    total = len(exps)
    logger.info("=" * 65)
    logger.info(f"Planned: {total}  |  skip: {sum(1 for e in exps if e.run_id in done)}  |  epochs: {args.epochs}")
    logger.info("=" * 65)
    for i, e in enumerate(exps, 1):
        tag = "  [skip]" if e.run_id in done else ""
        logger.info(f"  {i:3d}. [{e.question}] {e.dataset:8s}  {e.label}{tag}")
    logger.info("=" * 65)

    if args.dry_run:
        return

    for i, exp in enumerate(exps, 1):
        if exp.run_id in done:
            logger.info(f"[{i}/{total}] skip {exp.run_id}")
            continue
        logger.info(f"\n[{i}/{total}] {exp.dataset} | {exp.label}  (dim={exp.embedding_dim})")
        row = run_one(exp, args.epochs)
        if row is None:
            continue
        df_row = pd.DataFrame([row])
        df_row.to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)
        logger.info(
            f"  ✓  Test MRR={row.get('Test_MRR','?'):.4f}  "
            f"H@1={row.get('Test_H@1','?'):.4f}  ({row['elapsed_s']:.0f}s)")

    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        if "Test_MRR" in df.columns:
            summary = (df.groupby(["dataset","label"])["Test_MRR"]
                       .mean().reset_index()
                       .sort_values(["dataset","Test_MRR"], ascending=[True,False]))
            logger.info(f"\nTest MRR summary:\n{summary.to_string(index=False)}")
    logger.info(f"\nResults: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
