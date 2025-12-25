from executer import run_dicee_eval
from utils import set_seeds
from pathlib import Path
import json
import random
import os
from config import DBS, MODELS, RECIPRIOCAL, PERCENTAGES, BATCH_SIZE, LEARNING_RATE
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
import random

for DB in DBS:
    for MODEL in MODELS:

        
        MASTER_SEED = 12345
        seed_src = random.Random(MASTER_SEED)
        EXPERIMENT_SEEDS = [seed_src.randrange(2 ** 32) for _ in range(NUM_EXPERIMENTS)]


        for experiment, experiment_seed in enumerate(EXPERIMENT_SEEDS):

            result_random_poisoned = run_dicee_eval(
                dataset_folder=f"../KGs/{DB}",
                model=MODEL,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                embedding_dim=EMB_DIM,
                path_to_store_single_run=f"saved_models/{RECIPRIOCAL}/{DB}/{MODEL}/{experiment_seed}/",
                scoring_technique=SCORING_TECH,
                optim=OPTIM,
                seed=experiment_seed,
            )
            