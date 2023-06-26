from dicee.executer import Execute
import sys
import pytest
from dicee.config import Args


class TestDefaultParams:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_complex(self):
        args = Args()
        args.path_dataset_folder = 'KGs/UMLS'
        args.trainer = 'PL'
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
        assert 0.88 >= result['Train']['H@1'] >= 0.85
        assert 0.74 >= result['Val']['H@1'] >= 0.72
        assert 0.73 >= result['Test']['H@1'] >= 0.71
