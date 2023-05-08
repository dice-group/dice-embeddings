from dicee.executer import Execute
import pytest
from dicee.config import Args

class TestRegressionPyke:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Args()
        args.model = 'Pyke'
        args.path_dataset_folder = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 5
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.eval_model = None
        result = Execute(args).start()
