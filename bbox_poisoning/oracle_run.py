from executer import run_dicee_eval
from utils import set_seeds
from pathlib import Path
import json
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DBS = ["WN18RR"] #["UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237"] #, ["WN18RR", "YAGO3-10"]
MODELS = ["DistMult", "ComplEx", "Pykeen_TransE", "Pykeen_TransH", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL", "Keci"]

batch_size = "256"
learning_rate = "0.01"

recipriocal = "without_recipriocal"

for DB in DBS:
    for MODEL in MODELS:

        num_experiments = 3
        MASTER_SEED = 12345
        seed_src = random.Random(MASTER_SEED)
        experiment_seeds = [seed_src.randrange(2 ** 32) for _ in range(num_experiments)]

        for experiment, experiment_seed in enumerate(experiment_seeds):

            print(experiment_seed)

            result_random_poisoned = run_dicee_eval(
                dataset_folder=f"./KGs/{DB}",
                model=MODEL,
                num_epochs="100",
                batch_size=batch_size,
                learning_rate=learning_rate,
                embedding_dim="32",
                loss_function="BCELoss",
                path_to_store_single_run=f"saved_models/{recipriocal}/{DB}/{MODEL}/{experiment_seed}/",
                scoring_technique="KvsAll",
                optim="Adam",
                seed=experiment_seed,
            )

            report = {
                "seed": experiment_seed,
                "experiment": experiment,
                "model": MODEL,
                "db": DB,
                "train" : result_random_poisoned["Train"],
                "val" : result_random_poisoned["Val"],
                "test" : result_random_poisoned["Test"],
            }


            report_path = Path(f"./reports_for_datasets_without_corruptions/{recipriocal}/{DB}/{MODEL}/{experiment_seed}/oracle_results.txt")
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with report_path.open("a", encoding="utf-8") as f:
                f.write(f"{json.dumps(report)}\n")


