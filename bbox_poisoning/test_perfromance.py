from executer import run_dicee_eval
from utils import set_seeds
from pathlib import Path
import json
import random
import os
from config import (DBS, 
                    MODELS, 
                    RECIPRIOCAL, 
                    PERCENTAGES, 
                    BATCH_SIZE, 
                    LEARNING_RATE, 
                    NUM_EXPERIMENTS, 
                    NUM_EPOCHS, 
                    EMB_DIM, 
                    LOSS_FN, 
                    SCORING_TECH, 
                    OPTIM 
                    )
import csv
import os

DBS =  [ "UMLS" ] #[ "NELL-995-h100" ] #] #] "KINSHIP" UMLS
MODELS = [ "DistMult" ] #"DistMult", "ComplEx", "QMult", "DualE" -------"Keci", "DeCaL" ---- "Pykeen_MuRE", "Pykeen_RotatE",

for DB in DBS:
    for MODEL in MODELS:

        res = run_dicee_eval(
            #dataset_folder=f"./modu/{DB}/", 
            dataset_folder=f"./KGs/{DB}/",
            #dataset_folder= "./saved_datasets/without_recipriocal/UMLS/delete/simple_closeness/DistMult/noise/104/2430986565",
            #dataset_folder= "./prune",
            #dataset_folder= "./splits",
            #dataset_folder= "./rand_split",

            model=MODEL,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            embedding_dim=EMB_DIM,
            loss_function=LOSS_FN,
            path_to_store_single_run=f"saved_models/{RECIPRIOCAL}/{DB}/{MODEL}/",
            scoring_technique=SCORING_TECH,
            optim=OPTIM,
            seed=45,
        )

        #exp_type = "modu"
        #exp_type = "clean"
        exp_type = "noisy"

        out_csv = f"{exp_type}/{DB}/performance.csv"
        Path(f"{exp_type}/{DB}/").mkdir(parents=True, exist_ok=True)   


        fieldnames = ["Dataset", "Model", "MRR"]

        file_exists = os.path.isfile(out_csv)

        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header only the first time
            if not file_exists:
                w.writeheader()

            w.writerow({
                "Dataset": DB,
                "Model": MODEL,
                "MRR": res["Test"]["MRR"],
    })
