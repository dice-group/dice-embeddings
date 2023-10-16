from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestPolyak:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_polyak_qmult_k_vs_all(self):
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
        args.callbacks = {'PPE':None}
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.70 >= result['Train']['H@1'] >= 0.68
        assert 0.778 >= result['Train']['MRR'] >= 0.775
        assert 0.640 >= result['Val']['H@1'] >= 0.630
        assert 0.640 >= result['Test']['H@1'] >= 0.620
        assert result['Train']['H@10'] >= result['Train']['H@3'] >= result['Train']['H@1']
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']
        assert result['Test']['H@10'] >= result['Test']['H@3'] >= result['Test']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_polyak_qmult_k_vs_all(self):
        args = Namespace()
        args.model = 'QMult'
        args.dataset_dir = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 59
        args.batch_size = 1024
        args.lr = 0.1
        args.callbacks = {'FPPE': None}
        args.embedding_dim = 128
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.callbacks = ['PPE']
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['MRR'] >= 0.05
