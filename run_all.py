from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
from pathlib import Path
import csv
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATASETS = [ "UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237", "WN18RR", "CoDEx-M" ]

MODELS = [ "ComplEx", "DistMult", "Pykeen_RotatE", "Pykeen_MuRE", "DeCaL", "Pykeen_TransH" ]

EXP_TYPES = [  "BCE", "LS", "ACLS", "BCE_ASWA", "LS_ASWA", "ACLS_ASWA", "BCE_AMWA", "LS_AMWA", "ACLS_AMWA" ]


OUT_CSV = Path(f"./performance.csv")
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
                args = Namespace()

                # Baseline
                if EXP_TYPE == "BCE":
                    args.loss_fn = "BCELoss"
                    args.label_smoothing_rate = 0.0
                    args.adaptive_swa = False

                # Label Smoothing
                if EXP_TYPE == "LS":
                    args.loss_fn = "BCELoss"
                    args.label_smoothing_rate = 0.1
                    args.adaptive_swa = False

                # Adaptive and Conditional Label Smoothing
                if EXP_TYPE == "ACLS":
                    args.loss_fn = "ACLS"
                    args.label_smoothing_rate = 0.0
                    args.adaptive_swa = False

                # Baseline with ASWA
                if EXP_TYPE == "BCE_ASWA":
                    args.loss_fn = "BCELoss"
                    args.label_smoothing_rate = 0.0
                    args.adaptive_swa = True

                # Label Smoothing with ASWA
                if EXP_TYPE == "LS_ASWA":
                    args.loss_fn = "BCELoss"
                    args.label_smoothing_rate = 0.1
                    args.adaptive_swa = True

                # Adaptive and Conditional Label Smoothing with ASWA
                if EXP_TYPE == "ACLS_ASWA":
                    args.loss_fn = "ACLS"
                    args.label_smoothing_rate = 0.0
                    args.adaptive_swa = True

                # Baseline with Adaptive Momentum Weight Averaging (AMWA)
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

                # Label Smoothing with AMWA
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

                # Adaptive and Conditional Label Smoothing (ACLS) with Adaptive Momentum Weight Averaging (AMWA)
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


                args.path_to_store_single_run = f"Experiments/{EXP_TYPE}/{DB}/{MODEL}"
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
                args.trainer = "PL"

                result = Execute(args).start()

                w.writerow({
                    "DB": DB,
                    "Model": MODEL,
                    "Type": EXP_TYPE,
                    "Test-Hit@10": result["Test"]["H@10"],
                    "Test-Hit@3": result["Test"]["H@3"],
                    "Test-Hit@1": result["Test"]["H@1"],
                    "Test-MRR": result["Test"]["MRR"],
                })
                f.flush()
