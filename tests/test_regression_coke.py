from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionCoKE:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 512
        args.lr = 0.001
        args.embedding_dim = 64
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.scoring_technique = 'KvsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.80 >= result['Val']['H@1'] >= 0.01
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 512
        args.lr = 0.001
        args.embedding_dim = 64
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.scoring_technique = '1vsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.85 >= result['Val']['H@1'] >= 0.01
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_all_vs_all(self):
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 512
        args.lr = 0.001
        args.embedding_dim = 64
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.scoring_technique = 'AllvsAll'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.85 >= result['Val']['H@1'] >= 0.01
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = Namespace()
        args.model = 'CoKE'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 512
        args.lr = 0.001
        args.embedding_dim = 64
        args.input_dropout_rate = 0.1
        args.hidden_dropout_rate = 0.1
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.eval_model = 'train_val_test'
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.80 >= result['Train']['H@1'] >= 0.01
        assert 0.80 >= result['Test']['H@1'] >= 0.01
        assert 0.80 >= result['Val']['H@1'] >= 0.01

