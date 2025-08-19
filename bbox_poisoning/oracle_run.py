from executer import run_dicee_eval

"""
'Pykeen_TransE', 'Pykeen_TransF', 'Pykeen_TransH', 'Pykeen_TransR', 'Pykeen_TuckER'
"""

DBS = ["UMLS", "KINSHIP", "FB15k-237", "NELL-995-h100", "WN18RR", "YAGO3-10"]
MODELS = ["Keci", "ComplEx", "DistMult", "QMult", "Pykeen_MuRE", "Pykeen_RotatE", "Pykeen_BoxE", "DeCaL"]

for DB in DBS:
    for MODEL in MODELS:
        result_random_poisoned = run_dicee_eval(
            dataset_folder=f"./KGs/{DB}",
            model=MODEL,
            num_epochs="100",
            batch_size="1024",
            learning_rate="0.1",
            embedding_dim="32",
            loss_function="BCELoss",
            path_to_store_single_run=f"saved_models/{DB}/{MODEL}",
            scoring_technique="KvsAll",
            optim="Adam",
        )

