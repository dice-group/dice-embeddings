DBS = ["UMLS", "KINSHIP"] #, "NELL-995-h100", "FB15k-237", "WN18RR"
MODELS = [ "DistMult", "ComplEx", "Keci", "DeCaL", "Pykeen_MuRE", "Pykeen_RotatE"] #'Pykeen_TransE', 'Pykeen_TransH', "Pykeen_MuRE", "Pykeen_RotatE"


BATCH_SIZE = "1024"
LEARNING_RATE = "0.1"

NUM_EPOCHS = "100"
EMB_DIM = "32"
LOSS_FN = "BCELoss"
SCORING_TECH = "KvsAll"
OPTIM = "Adam"
EVAL_MODEL = "train_val_test"