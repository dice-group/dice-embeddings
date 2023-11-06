import pytest
from dicee import Execute
from dicee.config import Namespace


class TestRegressionAllvsAll:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_allvsall_kvsall(self):
        args = Namespace()
        args.model = "Keci"
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.scoring_technique = 'KvsAll'
        args.eval_model = 'train_val_test'
        result1 = Execute(args).start()
        """
        Evaluate Keci on Train set: Evaluate Keci on Train set
        {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
        Evaluate Keci on Validation set: Evaluate Keci on Validation set
        {'H@1': 0.45015337423312884, 'H@3': 0.6756134969325154, 'H@10': 0.8895705521472392, 'MRR': 0.5935077148200957}
        Evaluate Keci on Test set: Evaluate Keci on Test set
        {'H@1': 0.4750378214826021, 'H@3': 0.7065052950075643, 'H@10': 0.9175491679273827, 'MRR': 0.6203722969924745}
        """
        args = Namespace()
        args.model = "Keci"
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.scoring_technique = 'AllvsAll'
        args.eval_model = 'train_val_test'
        result2 = Execute(args).start()
        assert result2['Test']['MRR'] >= result1['Test']['MRR']
        """
        Evaluate Keci on Train set: Evaluate Keci on Train set
        {'H@1': 1.0, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 1.0}
        Evaluate Keci on Validation set: Evaluate Keci on Validation set
        {'H@1': 0.5061349693251533, 'H@3': 0.7484662576687117, 'H@10': 0.9202453987730062, 'MRR': 0.6501140088909673}
        Evaluate Keci on Test set: Evaluate Keci on Test set
        {'H@1': 0.5249621785173979, 'H@3': 0.7639939485627837, 'H@10': 0.9334341906202723, 'MRR': 0.6656083495965645}
        """