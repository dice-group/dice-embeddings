from executer import run_dicee_eval
from pathlib import Path
import os, csv

from config import (
    DBS, MODELS, RECIPRIOCAL,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EMB_DIM, SCORING_TECH, OPTIM
)

DATA_ROOT = Path("./saved_perturbed_datasets/without_recipriocal")

EXP_TYPE = "combinations"   

def is_dir(p: Path) -> bool:
    return p.exists() and p.is_dir()

def sorted_dirs(parent: Path):
    return sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda x: x.name)

for DB in DBS:
    for run_model in MODELS:
        out_csv = Path(f"{EXP_TYPE}/{DB}/performance.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        file_exists = out_csv.is_file()
        fieldnames = ["DB", "RunModel", "DataModel", "Ratio", "Seed", "MRR"]

        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()

            # loop over *data* models (including itself)
            for data_model in MODELS:
                data_model_dir = DATA_ROOT / DB / "delete" / "score" / data_model

                if not is_dir(data_model_dir):
                    print(f"[skip] missing data folder: {data_model_dir}")
                    continue

                for ratio_dir in sorted_dirs(data_model_dir):
                    for seed_dir in sorted_dirs(ratio_dir):
                        # expect train/val/test.txt inside each seed folder
                        train_path = seed_dir / "train.txt"
                        val_path   = seed_dir / "val.txt"
                        test_path  = seed_dir / "test.txt"

                        print(train_path)

                        

                        dataset_folder = str(seed_dir)
                        seed = int(seed_dir.name) if seed_dir.name.isdigit() else 45
                        ratio = ratio_dir.name

                        print(f"DB={DB} run_model={run_model} data_model={data_model} ratio={ratio} seed={seed}")

                        res = run_dicee_eval(
                            dataset_folder=dataset_folder,
                            model=run_model,
                            num_epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            learning_rate=LEARNING_RATE,
                            embedding_dim=EMB_DIM,
                            path_to_store_single_run=f"saved_models_4_db/{RECIPRIOCAL}/{DB}/{run_model}/",
                            scoring_technique=SCORING_TECH,
                            optim=OPTIM,
                            seed=seed,
                        )

                        w.writerow({
                            "DB": DB,
                            "RunModel": run_model,
                            "DataModel": data_model,
                            "Ratio": ratio,
                            "Seed": seed_dir.name,
                            "MRR": res["Test"]["MRR"],
                        })

                        f.flush()

