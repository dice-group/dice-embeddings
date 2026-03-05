from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
from pathlib import Path
import csv
import random

HYPER_PARAMETERS = {
    "KINSHIP": {

        "ComplEx": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.05
        },

        "DistMult": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.04
        },

        "DualE": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.06
        },

        "QMult": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.06
        }

    },

    "UMLS": {

        "ComplEx": {
            "embedding_dim": 32,
            "batch_size": 512,
            "learning_rate": 0.05
        },

        "DistMult": {
            "embedding_dim": 32,
            "batch_size": 512,
            "learning_rate": 0.04
        },

        "DualE": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.04
        },

        "QMult": {
            "embedding_dim": 64,
            "batch_size": 512,
            "learning_rate": 0.03
        },
    },

    "NELL-995-h100": {

        "ComplEx": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.01
        },

        "DistMult": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.01
        },

        "DualE": {
            "embedding_dim": 32,
            "batch_size": 1024,
            "learning_rate": 0.05
        },

        "QMult": {
            "embedding_dim": 32,
            "batch_size": 512,
            "learning_rate": 0.02
        }

    },

    "FB15k-237": {

        "ComplEx": {
            "embedding_dim": 32,
            "batch_size": 256,
            "learning_rate": 0.02
        },

        "DistMult": {
            "embedding_dim": 32,
            "batch_size": 512,
            "learning_rate": 0.02
        },

        "DualE": {
            "embedding_dim": 32,
            "batch_size": 512,
            "learning_rate": 0.01
        },

        "QMult": {
            "embedding_dim": 32,
            "batch_size": 1024,
            "learning_rate": 0.02
        }

    },

}

DATASETS = [ "UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237" ] # , "WN18RR", "YAGO3-10"
MODELS = [ "DistMult", "ComplEx", "DualE", "QMult" ] # , "Pykeen_RotatE", "Pykeen_MuRE", "DeCaL", "Keci", "Pykeen_TransH"

EXP_TYPES = ["BCE", "LS", "LR", "ACLS", "ALR", "BCE_ASWA", "LS_ASWA", "ACLS_ASWA", "ACLS_S", "ACLS_S_ASWA"]

NUM_EXPERIMENTS = 6

MASTER_SEED = 12345
seed_src = random.Random(MASTER_SEED)
EXPERIMENT_SEEDS = [seed_src.randrange(2 ** 32) for _ in range(NUM_EXPERIMENTS)]


OUT_CSV = Path(f"./performance_all.csv")
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


                    args.path_to_store_single_run = f"Experiments/{EXP_TYPE}/{DB}/{MODEL}/{EXPERIMENT_SEED}"

                    args.save_embeddings_as_csv = True

                    args.model = MODEL
                    args.p=0
                    args.q=1
                    args.optim = 'Adam'
                    args.scoring_technique = "KvsAll"

                    args.dataset_dir = f"../KGs/{DB}"

                    args.num_epochs = 100

                    
                    args.embedding_dim = HYPER_PARAMETERS[DB][MODEL]["embedding_dim"]
                    args.batch_size = HYPER_PARAMETERS[DB][MODEL]["batch_size"]
                    args.lr = HYPER_PARAMETERS[DB][MODEL]["learning_rate"]

                    args.eval_model = "test"

                    args.random_seed = EXPERIMENT_SEED

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