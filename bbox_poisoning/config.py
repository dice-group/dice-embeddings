DBS = [ "UMLS" ] #, "KINSHIP"  ] #, "NELL-995-h100", "FB15k-237"
MODELS =  [ "DistMult", "ComplEx", "QMult", "DeCaL" ] #[ "", "ComplEx", "QMult", "Pykeen_TransE", "Pykeen_TransH", "Pykeen_TransR" ] #, "ComplEx", "QMult", "DistMult", "Pykeen_MuRE", "Pykeen_RotatE", "Keci","DeCaL"] 

# "Pykeen_BoxE", "Pykeen_TransE", "Pykeen_TransH", "Pykeen_TransR"

RECIPRIOCAL = "with_recipriocal" 

PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  
BATCH_SIZE = "1024"
LEARNING_RATE = "0.1"

NUM_EXPERIMENTS = 6

NUM_EPOCHS = "100"
EMB_DIM = "32"
LOSS_FN = "BCELoss"
SCORING_TECH =  "KvsAll"
OPTIM = "Adam"
