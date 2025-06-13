from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionTapire:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'Tapire'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 256
        args.batch_size = 1024
        args.lr = 0.05
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        tapire_result = Execute(args).start()

        assert tapire_result["Test"]["MRR"]>=0.766
        assert tapire_result["Train"]["MRR"]>=0.82
        
        args = Namespace()
        args.model = 'DeCaL'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.r = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        decal_result = Execute(args).start()

        assert decal_result["Train"]["MRR"] < tapire_result["Train"]["MRR"]
        assert decal_result["Test"]["MRR"] < tapire_result["Test"]["MRR"]

