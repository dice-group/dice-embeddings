from dicee.executer import Execute
import pytest
from dicee.config import Namespace


class TestRegressionAllvsAll:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_allvsall_kvsall(self):
        args = Namespace()
        args.path_dataset_folder = 'KGs/UMLS'
        args.scoring_technique = 'KvsAll'
        args.eval_model = 'train_val_test'
        result1 = Execute(args).start()

        args = Namespace()
        args.path_dataset_folder = 'KGs/UMLS'
        args.scoring_technique = 'AllvsAll'
        args.eval_model = 'train_val_test'
        result2 = Execute(args).start()

        assert result2['Val']['MRR'] >= result1['Val']['MRR']
        assert result2['Test']['MRR'] >= result1['Test']['MRR']