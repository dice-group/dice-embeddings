from dicee.executer import Execute
import pytest
from dicee.config import Namespace

args = Namespace()
args.model = 'Keci'
args.scoring_technique = 'KvsAll'
args.optim = 'Adam'
args.p = 0
args.q = 1
args.dataset_dir = 'KGs/UMLS'
args.num_epochs = 32
args.batch_size = 1000024
args.lr = 0.1
args.embedding_dim = 32
args.eval_model = 'train_val_test'
keci_result = Execute(args).start()
