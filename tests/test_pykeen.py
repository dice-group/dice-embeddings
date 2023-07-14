from dicee.executer import Execute
import sys
import pytest
from dicee.config import Arguments


class TestDefaultParams:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_distmult(self):
        args = Arguments()
        args.path_dataset_folder = 'KGs/UMLS'
        args.trainer = 'torchCPUTrainer'
        args.model = 'Pykeen_DistMult'
        args.num_epochs = 10
        args.batch_size = 256
        args.lr = 0.1
        args.num_workers = 1
        args.num_core = 1
        args.scoring_technique = 'KvsAll'
        args.num_epochs = 10
        args.pykeen_model_kwargs = {'embedding_dim': 64}
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        assert 0.84 >= result['Train']['MRR'] >= 0.800

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_complex(self):
        args = Arguments()
        args.path_dataset_folder = 'KGs/UMLS'
        args.trainer = 'torchCPUTrainer'
        args.model = 'Pykeen_ComplEx'
        args.num_epochs = 10
        args.batch_size = 256
        args.lr = 0.1
        args.num_workers = 1
        args.num_core = 1
        args.scoring_technique = 'KvsAll'
        args.num_epochs = 10
        args.pykeen_model_kwargs = {'embedding_dim': 64}
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        assert 0.92 >= result['Train']['MRR'] >= 0.88

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_quate(self):
        args = Arguments()
        args.path_dataset_folder = 'KGs/UMLS'
        args.trainer = 'torchCPUTrainer'
        args.model = 'Pykeen_QuatE'
        args.num_epochs = 10
        args.batch_size = 256
        args.lr = 0.1
        args.num_workers = 1
        args.num_core = 1
        args.scoring_technique = 'KvsAll'
        args.num_epochs = 10
        args.pykeen_model_kwargs = {'embedding_dim': 64}
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        # num params 14528
        assert 0.999 >= result['Train']['MRR'] >= 0.94

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_mure(self):
        args = Arguments()
        args.path_dataset_folder = 'KGs/UMLS'
        args.trainer = 'torchCPUTrainer'
        args.model = 'Pykeen_MuRE'
        args.num_epochs = 10
        args.batch_size = 256
        args.lr = 0.1
        args.num_workers = 1
        args.num_core = 1
        args.scoring_technique = 'KvsAll'
        args.num_epochs = 10
        args.pykeen_model_kwargs = {'embedding_dim': 64}
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        # num params 20686
        assert 0.88 >= result['Train']['MRR'] >= 0.82
