"""Sweep loss / label-smoothing / weight-averaging settings over datasets and models.

Each experiment type combines a loss (BCE, label smoothing, ACLS) with an optional
weight-averaging scheme (none, ASWA, AMWA). One CSV row per run is appended to --out.

Example: python benchmarks/regularization_sweep.py --datasets UMLS KINSHIP --models DistMult --gpu 0
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Dict

DATASETS = ["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR", "CoDEx-M"]
MODELS = ["ComplEx", "DistMult", "Pykeen_RotatE", "Pykeen_MuRE", "DeCaL", "Pykeen_TransH"]

LOSSES = {
    "BCE": {"loss_fn": "BCELoss", "label_smoothing_rate": 0.0},
    "LS": {"loss_fn": "BCELoss", "label_smoothing_rate": 0.1},
    "ACLS": {"loss_fn": "ACLS", "label_smoothing_rate": 0.0},
}
AVERAGING = {
    "": {"adaptive_swa": False},
    "ASWA": {"adaptive_swa": True},
    "AMWA": {
        "adaptive_swa": False,
        "amwa": True,
        "amwa_start_epoch": 20,
        "amwa_c_epochs": 10,
        "amwa_monitor": "MRR",
        "amwa_maximize": True,
        "amwa_beta": None,
        "amwa_beta_window": 10,
        "amwa_beta_init": 1.0,
        "amwa_beta_floor": 1e-12,
    },
}
EXP_TYPES = [f"{loss}_{avg}" if avg else loss for avg in AVERAGING for loss in LOSSES]

FIELDNAMES = ["DB", "Model", "Type", "Seed", "Test-Hit@10", "Test-Hit@3", "Test-Hit@1", "Test-MRR"]


def exp_settings(exp_type: str) -> Dict:
    """Return the Namespace overrides for an experiment type such as ``LS_ASWA``."""
    loss, _, avg = exp_type.partition("_")
    return {**LOSSES[loss], **AVERAGING[avg]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--exp_types", nargs="+", default=EXP_TYPES, choices=EXP_TYPES)
    p.add_argument("--kg_root", type=Path, default=Path("KGs"), help="Directory containing the dataset folders.")
    p.add_argument("--out", type=Path, default=Path("performance.csv"))
    p.add_argument("--experiments_dir", type=Path, default=Path("Experiments"))
    p.add_argument("--gpu", default="0", help="Value for CUDA_VISIBLE_DEVICES.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu
    # Imported after CUDA_VISIBLE_DEVICES is set so torch sees only the chosen GPU.
    from dicee.config import Namespace
    from dicee.executer import Execute

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not cli.out.is_file()
    with open(cli.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for db in cli.datasets:
            for model in cli.models:
                for exp_type in cli.exp_types:
                    args = Namespace()
                    for key, value in exp_settings(exp_type).items():
                        setattr(args, key, value)
                    args.path_to_store_single_run = str(cli.experiments_dir / exp_type / db / model)
                    args.save_embeddings_as_csv = True
                    args.model = model
                    args.p, args.q = 0, 1
                    args.optim = "Adam"
                    args.scoring_technique = "KvsAll"
                    args.dataset_dir = str(cli.kg_root / db)
                    args.num_epochs = 100
                    args.embedding_dim = 32
                    args.batch_size = 1024
                    args.lr = 0.1
                    args.eval_model = "test"
                    args.trainer = "PL"
                    args.random_seed = cli.seed

                    test = Execute(args).start()["Test"]
                    writer.writerow({
                        "DB": db, "Model": model, "Type": exp_type, "Seed": cli.seed,
                        "Test-Hit@10": test["H@10"], "Test-Hit@3": test["H@3"],
                        "Test-Hit@1": test["H@1"], "Test-MRR": test["MRR"],
                    })
                    f.flush()


if __name__ == "__main__":
    main()
