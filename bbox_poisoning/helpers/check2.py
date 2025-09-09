from executer import run_dicee_eval

"""
'Pykeen_TransE', 'Pykeen_TransF', 'Pykeen_TransH', 'Pykeen_TransR', 'Pykeen_TuckER'

DBS = ["UMLS", "KINSHIP", "FB15k-237", "NELL-995-h100", "WN18RR"] # , "YAGO3-10"
MODELS = ["Keci", "ComplEx", "DistMult", "QMult", "Pykeen_MuRE", "Pykeen_RotatE", "DeCaL"] #  "Pykeen_BoxE", Pykeen_TuckER
"""

DB = "UMLS"
MODEL =  "ComplEx" #"ComplEx" #"DistMult" # #"QMult", "Pykeen_MuRE" Pykeen_RotatE Keci DeCaL Pykeen_BoxE Pykeen_TuckER

batch_size = "256"
learning_rate = "0.01"

#dataset_folder = "./KGs/UMLS"
dataset_folder = "./saved_datasets/UMLS/fgsm/del/close_simple/ComplEx/104/0/"
#dataset_folder = "./saved_datasets/wo/UMLS/active_poisoning_whitebox/Pykeen_TransH/high_close_fgsm/1564/random-one/0"

#dataset_folder = "./saved_datasets/wo/UMLS/active_poisoning_whitebox/Pykeen_TransE/high_close_fgsm/1564/random-one/0"

#dataset_folder =  "./saved_datasets/wo/UMLS/random/ComplEx/1564/random-one/0"

#dataset_folder = "./saved_datasets/UMLS/fgsm/del/close/ComplEx/156/0/"

#dataset_folder = "./saved_datasets/wo/UMLS/active_poisoning_whitebox/ComplEx/high_close_fgsm/1564/random-one/0/"

#dataset_folder = "./saved_models/UMLS/active_poisoning_whitebox/DistMult/high_close_fgsm/1564/random-one/0"

#dataset_folder = "./saved_models/UMLS/active_poisoning_whitebox/ComplEx/high_gradients_fgsm_triples/1043/random-one/0/"


#dataset_folder = "./saved_models/UMLS/active_poisoning_whitebox_server/DistMult/high_close_fgsm/1564/random-one/0/"
#dataset_folder = "./saved_models/without_recipriocal/UMLS/active_poisoning_whitebox_server/ComplEx/high_gradients_fgsm_triples/1043/random-one/0"

#dataset_folder = "./saved_datasets/UMLS/active_poisoning_whitebox/ComplEx/high_gradients_fgsm_triples/1564/random-one/0/"

#dataset_folder = "./saved_models/KINSHIP/active_poisoning_whitebox_server/QMult/high_close_fgsm/1281/random-one/0/"

#dataset_folder = "./saved_models/KINSHIP/active_poisoning_whitebox_server/QMult/high_close_fgsm/1708/random-one/0/"

result_random_poisoned = run_dicee_eval(
    dataset_folder=dataset_folder,
    model=MODEL,
    num_epochs="100",
    batch_size=batch_size,
    learning_rate=learning_rate,
    embedding_dim="32",
    loss_function="BCELoss",
    seed=42,
    path_to_store_single_run=f"saved_models/checks_only/{DB}/{MODEL}",
    scoring_technique="KvsAll",
    optim="Adam",
)

print(result_random_poisoned['Test']['MRR'])