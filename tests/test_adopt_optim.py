from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionAdoptOptim:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_adpot_optim(self):
        args = Namespace()
        args.model = 'Keci'
        args.optim = 'Adopt'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 20
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        args.scoring_technique = 'KvsAll'
        adopt_result = Execute(args).start()
        assert 0.58 >= adopt_result['Val']['H@1'] >= 0.01

        args = Namespace()
        args.model = 'Keci'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 20
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        args.scoring_technique = 'KvsAll'
        adam_result = Execute(args).start()
        assert 0.58 >= adopt_result['Val']['H@1'] >= 0.01

        assert adopt_result['Val']['MRR'] >= adam_result['Val']['MRR']
        assert adopt_result['Train']['MRR'] >= adam_result['Train']['MRR']