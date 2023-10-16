from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionQmult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'QMult'
        args.dataset_dir = 'KGs/UMLS'
        args.optim = 'Adam'
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
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 1.00 >= result['Train']['H@1'] >= 0.83
        assert 0.80 >= result['Val']['H@1'] >= 0.71
        assert 0.80 >= result['Test']['H@1'] >= 0.73

        assert result['Train']['H@10'] >= result['Train']['H@3'] >= result['Train']['H@1']
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']
        assert result['Test']['H@10'] >= result['Test']['H@3'] >= result['Test']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_1_vs_all(self):
        args = Namespace()
        args.model = 'QMult'
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
        args.trainer = 'torchCPUTrainer'
        args.init_param = 'xavier_normal'
        args.normalization='LayerNorm'
        result = Execute(args).start()
        assert 0.92 >= result['Train']['H@1'] >= 0.83
        assert 0.82 >= result['Test']['H@1'] >= 0.75
        assert 0.82 >= result['Val']['H@1'] >= 0.75

        assert result['Train']['H@10'] >= result['Train']['H@3'] >= result['Train']['H@1']
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']
        assert result['Test']['H@10'] >= result['Test']['H@3'] >= result['Test']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = Namespace()
        args.model = 'QMult'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.1
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
        args.trainer = 'torchCPUTrainer'
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        result = Execute(args).start()
        assert 0.72 >= result['Train']['H@1'] >= .51
        assert 0.78 >= result['Test']['H@1'] >= .40
        assert 0.78 >= result['Val']['H@1'] >= .40
