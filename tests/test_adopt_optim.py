from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionAdoptOptim:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_adopt_optim(self):
        args = Namespace()
        args.model = 'Keci'
        args.optim = 'Adopt'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        args.scoring_technique = 'KvsAll'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'  # Force CPU trainer
        adopt_result = Execute(args).start()
        assert 0.99 >= adopt_result['Val']['H@1'] >= 0.74
        assert 0.99 >= adopt_result['Val']['MRR'] >= 0.80
        assert 0.99 >= adopt_result['Train']['H@1'] >= 0.90
        assert 0.99 >= adopt_result['Train']['MRR'] >= 0.95

        args = Namespace()
        args.model = 'Keci'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        args.scoring_technique = 'KvsAll'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'  # Force CPU trainer
        adam_result = Execute(args).start()
        assert 0.90 >= adam_result['Val']['H@1'] >= 0.66
        assert 0.90 >= adam_result['Val']['MRR'] >= 0.74
        assert 0.90 >= adam_result['Train']['H@1'] >= 0.71
        assert 0.90 >= adam_result['Train']['MRR'] >= 0.79

        assert adopt_result['Val']['MRR'] >= adam_result['Val']['MRR']
        assert adopt_result['Train']['MRR'] >= adam_result['Train']['MRR']