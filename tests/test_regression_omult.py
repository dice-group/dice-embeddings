from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionOmult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'OMult'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.90 >= result['Train']['H@1'] >= 0.65
        assert 0.76 >= result['Val']['H@1'] >= 0.70

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = Namespace()
        args.model = 'OMult'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = '1vsAll'
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.75 >= result['Test']['H@1'] >= 0.71
        assert 0.78 >= result['Val']['H@1'] >= 0.74
        assert 0.92 >= result['Train']['H@1'] >= 0.82

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = Namespace()
        args.model = 'OMult'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
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
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.60 >= result['Test']['H@1'] >= .25
        assert 0.60 >= result['Val']['H@1'] >= .25
        assert 0.63 >= result['Train']['H@1'] >= .31
