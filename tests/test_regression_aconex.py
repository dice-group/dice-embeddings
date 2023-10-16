from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionAConEx:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'AConEx'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.num_of_output_channels = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.normalization = 'BatchNorm1d'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.70 >= result['Train']['MRR'] >= 0.68
        assert 0.57 >= result['Train']['H@1'] >= 0.54
