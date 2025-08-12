from executer import run_dicee_eval

"""
'ComplEx', 'Keci', 'ConEx', 'AConEx', 'ConvQ', 'AConvQ', 'ConvO', 'AConvO', 
'QMult', 'OMult', 'Shallom', 'DistMult', 'TransE', 'DualE', 'BytE', 'Pykeen_MuRE', 
'Pykeen_QuatE', 'Pykeen_DistMult', 'Pykeen_BoxE', 'Pykeen_CP', 'Pykeen_HolE', 'Pykeen_ProjE', 
'Pykeen_RotatE', 'Pykeen_TransE', 'Pykeen_TransF', 'Pykeen_TransH', 'Pykeen_TransR', 
'Pykeen_TuckER', 'Pykeen_ComplEx', 'LFMult', 'DeCaL'
"""

DB = "./KGs/UMLS"
MODEL = "Pykeen_BoxE"

result_random_poisoned = run_dicee_eval(
    dataset_folder=DB,
    model=MODEL,
    num_epochs="100",
    batch_size="1024",
    learning_rate="0.1",
    embedding_dim="32",
    loss_function="BCELoss",
    scoring_technique="KvsAll",
    optim="Adam",
)

print(f"{result_random_poisoned['Test']['MRR']:.4f}")