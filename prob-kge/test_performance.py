from executer import run_dicee_eval
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
                    SCORING_TECH, 
                    OPTIM 
                    )
import csv
import os


#DBS = [ "UMLS"  ] 
#MODELS =  [ "DistMult", "ComplEx", "QMult", "DeCaL"  ]

for DB in DBS:
    for MODEL in MODELS:

        res = run_dicee_eval(
            #dataset_folder=f"../KGs/{DB}/",
            #dataset_folder= "/home/adel/Documents/new_dice_embeddings/dice-embeddings/prob-kge/saved_datasets/without_recipriocal/UMLS/delete/random/DistMult/104/831769172",
            #dataset_folder= "/home/adel/Documents/new_dice_embeddings/dice-embeddings/prob-kge/saved_datasets/without_recipriocal/UMLS/delete/score/QMult/104/831769172",
            #dataset_folder= "/home/adel/Documents/new_dice_embeddings/dice-embeddings/prob-kge/saved_datasets/without_recipriocal/UMLS/delete/score/DistMult/104/831769172",
            dataset_folder= "/home/adel/Documents/new_dice_embeddings/dice-embeddings/prob-kge/saved_datasets/without_recipriocal/UMLS/delete/score/DeCaL/104/831769172",


            model=MODEL,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            embedding_dim=EMB_DIM,
            path_to_store_single_run=f"saved_models/{RECIPRIOCAL}/{DB}/{MODEL}/",
            scoring_technique=SCORING_TECH,
            optim=OPTIM,
            seed=45,
        )

        exp_type = "clean"
        #exp_type = "noisy"

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
