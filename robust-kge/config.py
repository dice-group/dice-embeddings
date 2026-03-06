DBS = ["UMLS", "KINSHIP"] #, "NELL-995-h100", "FB15k-237", "WN18RR"
MODELS = ['Pykeen_TransE', 'Pykeen_TransH', "Pykeen_MuRE", "Pykeen_RotatE", "DistMult", "ComplEx", "DeCaL", "Keci", "QMult"]



BATCH_SIZE = "1024"
LEARNING_RATE = "0.1"

NUM_EPOCHS = "100"
EMB_DIM = "32"
LOSS_FN = "BCELoss"
SCORING_TECH = "KvsAll"
OPTIM = "Adam"

#for bayesian optimization, use train_val_test
EVAL_MODEL_TRAIN_VAL_TEST = "train_val_test"

#for actual experiments, use test
EVAL_MODEL_TEST = "test"