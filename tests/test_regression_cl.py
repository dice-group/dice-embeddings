from dicee.executer import Execute, ContinuousExecute
import pytest
from dicee.config import Namespace
class TestRegressionAConEx:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'AConEx'
        args.scoring_technique = 'KvsSample'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 5
        args.batch_size = 4096
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.normalization = 'LayerNorm'
        args.trainer = 'torchCPUTrainer'
        args.init_param = 'xavier_normal'
        result = Execute(args).start()

        args.path_experiment_folder = result['path_experiment_folder']
        cl_result = ContinuousExecute(args).continual_start()

        assert cl_result['Train']['H@10'] >= result['Train']['H@10']
        assert cl_result['Val']['H@10'] >= result['Val']['H@10']
        assert cl_result['Test']['H@10'] >= result['Test']['H@10']
