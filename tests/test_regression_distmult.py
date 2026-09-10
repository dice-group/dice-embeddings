from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionDistMult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'DistMult'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.scoring_technique = 'KvsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.58 >= result['Val']['H@1'] >= 0.01

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = Namespace()
        args.model = 'DistMult'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.scoring_technique = '1vsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = Namespace()
        args.model = 'DistMult'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.73 >= result['Train']['H@1'] >= 0.01
        assert 0.73 >= result['Test']['H@1'] >= 0.01
        assert 0.73 >= result['Val']['H@1'] >= 0.01

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_margin_ranking_loss(self):
        args = Namespace()
        args.model = 'DistMult'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSampleMargin'
        args.neg_ratio = 1
        args.margin = 1.0
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.55 >= result['Train']['H@1'] >= 0.01
        assert 0.42 >= result['Test']['H@1'] >= 0.01
        assert 0.42 >= result['Val']['H@1'] >= 0.01

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pykeen_transh_k_vs_all(self):
        args = Namespace()
        args.dataset_dir = 'KGs/UMLS'
        args.trainer = 'PL'
        args.model = 'Pykeen_TransH'
        args.num_epochs = 20
        args.batch_size = 256
        args.lr = 0.1
        args.num_workers = 1
        args.num_core = 1
        args.scoring_technique = 'KvsAll'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.eval_model = 'train_val_test'
        result = Execute(args).start()
        assert 0.75 >= result['Train']['MRR'] >= 0.35
        assert 0.75 >= result['Test']['MRR'] >= 0.30
        assert 0.75 >= result['Val']['MRR'] >= 0.30
