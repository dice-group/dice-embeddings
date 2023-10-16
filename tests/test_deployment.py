from dicee.executer import Execute
import pytest
from dicee import KGE
from dicee.static_funcs import random_prediction
from dicee.config import Namespace

class TestDefaultParams:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult(self):
        args = Namespace()
        args.model = 'QMult'
        args.optim = 'Adam'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.backend = 'pandas'
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        executor = Execute(args)
        executor.start()

        pre_trained_kge = KGE(path=executor.args.full_storage_path)
        random_prediction(pre_trained_kge)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pykeen_quatE(self):
        args = Namespace()
        args.model = 'Pykeen_QuatE'
        args.optim = 'Adam'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.backend = 'pandas'
        args.sample_triples_ratio = None
        args.trainer = 'PL'
        executor = Execute(args)
        executor.start()

        pre_trained_kge = KGE(path=executor.args.full_storage_path)
        random_prediction(pre_trained_kge)
