from executer import run_dicee_eval
from utils import set_seeds
from pathlib import Path
import json

#import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 2655772424
set_seeds(seed)

batch_size = "256"
learning_rate = "0.01"

DBS = ["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR"]
MODELS = [ "DistMult", "ComplEx", 'Pykeen_TransE', 'Pykeen_TransH' ]

report_path = Path(f"./oracle_results.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)

for DB in DBS:
    for MODEL in MODELS:
        result_random_poisoned = run_dicee_eval(
            dataset_folder=f"./KGs/{DB}",
            model=MODEL,
            num_epochs="100",
            batch_size=batch_size,
            learning_rate=learning_rate,
            embedding_dim="32",
            loss_function="BCELoss",
            path_to_store_single_run=f"saved_models/wo/{DB}/{MODEL}",
            scoring_technique="KvsAll",
            optim="Adam",
            seed=seed,
        )

        report = {
            "model": MODEL,
            "db": DB,
            "train" : result_random_poisoned["Train"],
            "val" : result_random_poisoned["Val"],
            "test" : result_random_poisoned["Test"],
        }
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"{json.dumps(report)}\n")


