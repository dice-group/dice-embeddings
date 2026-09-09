from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
from pathlib import Path
import csv
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#DATASETS = [ "UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR", "CoDEx-M" ]

DATASETS = [ "FB15k-237", "WN18RR", "CoDEx-M" ]

MODELS = [ "ComplEx", "DistMult", "Pykeen_RotatE", "Pykeen_MuRE", "DeCaL", "Pykeen_TransH" ]

EXP_TYPES = [   "BCE", 
                "LS", 
                #"LR", 
                "ACLS", 
                #"ACLS_S", 
                #"ALR",
                "BCE_ASWA", 
                "LS_ASWA", 
                #"LR_ASWA", 
                "ACLS_ASWA", 
                #"ACLS_S_ASWA", 
                #"ALR_ASWA",
                "BCE_AMWA", 
                "LS_AMWA", 
                #"LR_AMWA", 
                "ACLS_AMWA", 
                #"ACLS_S_AMWA", 
                #"ALR_AMWA"
            ]


NUM_EXPERIMENTS = 6

MASTER_SEED = 13558 #12345 #13558
seed_src = random.Random(MASTER_SEED)

EXPERIMENT_SEEDS = [seed_src.randrange(2 ** 32) for _ in range(NUM_EXPERIMENTS)]


OUT_CSV = Path(f"./performance_Aug22.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

file_exists = OUT_CSV.is_file()

fieldnames = ["DB", "Model", "Type", "Seed", "Test-Hit@10", "Test-Hit@3", "Test-Hit@1", "Test-MRR"]

with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        w.writeheader()

    for DB in DATASETS:
        for MODEL in MODELS:
            for EXP_TYPE in EXP_TYPES:

                for EXPERIMENT_SEED in EXPERIMENT_SEEDS:

                    args = Namespace()

                    if EXP_TYPE == "BCE":
                        args.loss_fn = "BCELoss"
                        args.label_smoothing_rate = 0.0
                        args.adaptive_swa = False

                    if EXP_TYPE == "LS":
                        args.loss_fn = "BCELoss"
                        args.label_smoothing_rate = 0.1
                        args.adaptive_swa = False

                    if EXP_TYPE == "LR":
                        args.loss_fn = "LRLoss"
                        args.label_relaxation_alpha = 0.1
                        args.adaptive_swa = False

                    if EXP_TYPE == "ACLS":
                        args.loss_fn = "ACLS"
                        args.label_smoothing_rate = 0.0
                        args.adaptive_swa = False

                    if EXP_TYPE == "ALR":
                        args.loss_fn = "AdaptiveLabelRelaxationLoss"
                        args.label_relaxation_alpha = 0.0
                        args.adaptive_swa = False

                    if EXP_TYPE == "ALR_ASWA":
                        args.loss_fn = "AdaptiveLabelRelaxationLoss"
                        args.adaptive_swa = True

                    if EXP_TYPE == "BCE_ASWA":
                        args.loss_fn = "BCELoss"
                        args.label_smoothing_rate = 0.0
                        args.adaptive_swa = True

                    if EXP_TYPE == "LS_ASWA":
                        args.loss_fn = "BCELoss"
                        args.label_smoothing_rate = 0.1
                        args.adaptive_swa = True

                    if EXP_TYPE == "LR_ASWA":
                        args.loss_fn = "LRLoss"
                        args.label_relaxation_alpha = 0.1
                        args.adaptive_swa = True

                    if EXP_TYPE == "ACLS_ASWA":
                        args.loss_fn = "ACLS"
                        args.label_smoothing_rate = 0.0
                        args.adaptive_swa = True

                    if EXP_TYPE == "ACLS_S":
                        args.loss_fn = "ACLS"
                        args.label_smoothing_rate = 0.1
                        args.adaptive_swa = False

                    if EXP_TYPE == "ACLS_S_ASWA":
                        args.loss_fn = "ACLS"
                        args.label_smoothing_rate = 0.1
                        args.adaptive_swa = True

                    if EXP_TYPE == "BCE_AMWA":
                        args.loss_fn = "BCELoss"
                        args.adaptive_swa = False
                        args.amwa = True
                        args.amwa_start_epoch = 20          
                        args.amwa_c_epochs = 10             
                        args.amwa_monitor = "MRR"           
                        args.amwa_maximize = True          
                        args.amwa_beta = None            
                        args.amwa_beta_window = 10
                        args.amwa_beta_init = 1.0
                        args.amwa_beta_floor = 1e-12

                    if EXP_TYPE == "LS_AMWA":
                        args.loss_fn = "BCELoss"
                        args.label_smoothing_rate = 0.1
                        args.adaptive_swa = False
                        args.amwa = True
                        args.amwa_start_epoch = 20          
                        args.amwa_c_epochs = 10             
                        args.amwa_monitor = "MRR"           
                        args.amwa_maximize = True          
                        args.amwa_beta = None            
                        args.amwa_beta_window = 10
                        args.amwa_beta_init = 1.0
                        args.amwa_beta_floor = 1e-12

                    if EXP_TYPE == "LR_AMWA":
                        args.loss_fn = "LRLoss"
                        args.label_relaxation_alpha = 0.1
                        args.adaptive_swa = False
                        args.amwa = True
                        args.amwa_start_epoch = 20          
                        args.amwa_c_epochs = 10             
                        args.amwa_monitor = "MRR"           
                        args.amwa_maximize = True          
                        args.amwa_beta = None            
                        args.amwa_beta_window = 10
                        args.amwa_beta_init = 1.0
                        args.amwa_beta_floor = 1e-12
                    
                    if EXP_TYPE == "ACLS_AMWA":
                        args.loss_fn = "ACLS"
                        args.label_smoothing_rate = 0.0
                        args.adaptive_swa = False
                        args.amwa = True
                        args.amwa_start_epoch = 20          
                        args.amwa_c_epochs = 10             
                        args.amwa_monitor = "MRR"           
                        args.amwa_maximize = True          
                        args.amwa_beta = None            
                        args.amwa_beta_window = 10
                        args.amwa_beta_init = 1.0
                        args.amwa_beta_floor = 1e-12

                    if EXP_TYPE == "ACLS_S_AMWA":
                        args.loss_fn = "ACLS"
                        args.adaptive_swa = False
                        args.amwa = True
                        args.amwa_start_epoch = 20          
                        args.amwa_c_epochs = 10             
                        args.amwa_monitor = "MRR"           
                        args.amwa_maximize = True          
                        args.amwa_beta = None            
                        args.amwa_beta_window = 10
                        args.amwa_beta_init = 1.0
                        args.amwa_beta_floor = 1e-12

                    if EXP_TYPE == "ALR_AMWA":
                        args.loss_fn = "AdaptiveLabelRelaxationLoss"
                        args.adaptive_swa = False
                        args.amwa = True
                        args.amwa_start_epoch = 20          
                        args.amwa_c_epochs = 10             
                        args.amwa_monitor = "MRR"           
                        args.amwa_maximize = True          
                        args.amwa_beta = None            
                        args.amwa_beta_window = 10
                        args.amwa_beta_init = 1.0
                        args.amwa_beta_floor = 1e-12

                    args.path_to_store_single_run = f"Experiments/{EXP_TYPE}/{DB}/{MODEL}/{EXPERIMENT_SEED}"

                    args.save_embeddings_as_csv = True

                    args.model = MODEL
                    args.p=0
                    args.q=1
                    args.optim = 'Adam'
                    args.scoring_technique = "KvsAll"

                    args.dataset_dir = f"../KGs/{DB}"

                    args.num_epochs = 100

                    
                    args.embedding_dim = 32
                    args.batch_size = 1024
                    args.lr = 0.1

                    args.eval_model = "test"

                    args.random_seed = EXPERIMENT_SEED

                    args.trainer = "PL"

                    result = Execute(args).start()


                    w.writerow({
                        "DB": DB,
                        "Model": MODEL,
                        "Type": EXP_TYPE,
                        "Seed": EXPERIMENT_SEED,
                        "Test-Hit@10": result["Test"]["H@10"],
                        "Test-Hit@3": result["Test"]["H@3"],
                        "Test-Hit@1": result["Test"]["H@1"],
                        "Test-MRR": result["Test"]["MRR"],
                    })
                    f.flush()
