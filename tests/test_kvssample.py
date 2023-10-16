from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionQmult:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_sample(self):
        args = Namespace()
        args.model = 'QMult'
        args.dataset_dir = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsSample'
        args.neg_ratio = 10
        args.num_folds_for_cv = 0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.normalization = 'LayerNorm'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()

        assert result['Train']['H@10'] >= result['Train']['H@3'] >= result['Train']['H@1']
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']
        assert result['Test']['H@10'] >= result['Test']['H@3'] >= result['Test']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_sample_regression(self):
        args = Namespace()
        args.model = 'AConEx'
        args.dataset_dir = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsSample'
        # size of entity vocabulary
        args.neg_ratio = 10
        args.weight_decay = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        args.normalization = 'LayerNorm'
        result = Execute(args).start()
        assert result['Train']['MRR'] >= 0.220
        assert result['Val']['MRR'] >= 0.220
        assert result['Test']['MRR'] >= 0.220
