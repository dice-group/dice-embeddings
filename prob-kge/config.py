DBS = [ "UMLS", "KINSHIP", "NELL-995-h100", "FB15k-237" ] #, "WN18RR", "DBpedia50", "DBpedia500"
MODELS =  [ "DistMult", "ComplEx", "QMult", "DualE" ] #, "Pykeen_TransE", "Pykeen_TransH", "Pykeen_TransR", "Pykeen_MuRE", "Pykeen_RotatE", # "Pykeen_BoxE"

RECIPRIOCAL = "without_recipriocal" 

PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]  

BATCH_SIZE = "1024"
LEARNING_RATE = "0.1"

NUM_EXPERIMENTS = 10

NUM_EPOCHS = "100"
EMB_DIM = "32"
SCORING_TECH =  "KvsAll"
OPTIM = "Adam"

QUANTIES = [
            (0.00, 0.10),
            (0.10, 0.26),
            (0.26, 0.42),
            (0.42, 0.58),
            (0.58, 0.74),
            (0.74, 0.90),
            (0.90, 1.00),
        ]