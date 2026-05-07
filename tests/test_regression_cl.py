from dicee.executer import Execute, ContinuousExecute
import pytest
from dicee.config import Namespace
import os
import json
class TestRegressionAConEx:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'AConEx'
        args.scoring_technique = '1vsSample'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.neg_ratio = 1
        args.batch_size = 256
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

        args.continual_learning = result['path_experiment_folder']
        cl_result = ContinuousExecute(args).continual_start()

        assert cl_result['Train']['H@10'] >= result['Train']['H@10']
        assert cl_result['Val']['H@10'] >= result['Val']['H@10']
        assert cl_result['Test']['H@10'] >= result['Test']['H@10']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_continual_with_periodic_eval(self):
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.optim = 'Adam'
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        args.eval_every_n_epochs = 1
        args.trainer = 'torchCPUTrainer'
        initial_result = Execute(args).start()

        periodic_eval_path = initial_result['path_experiment_folder'] + '/eval_report_n_epochs.json'
        assert os.path.isfile(periodic_eval_path)
        with open(periodic_eval_path, 'r') as f:
            initial_periodic = json.load(f)
        assert isinstance(initial_periodic, dict)
        initial_mtime = os.path.getmtime(periodic_eval_path)

        args.continual_learning = initial_result['path_experiment_folder']
        continual_result = ContinuousExecute(args).continual_start()

        assert 'Train' in continual_result
        assert 'Val' in continual_result
        assert 'Test' in continual_result
        assert os.path.isfile(periodic_eval_path)
        with open(periodic_eval_path, 'r') as f:
            continual_periodic = json.load(f)
        assert isinstance(continual_periodic, dict)
        assert len(continual_periodic) > 0
        assert os.path.getmtime(periodic_eval_path) >= initial_mtime

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_continual_with_adaptive_swa(self):
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.optim = 'Adam'
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        args.adaptive_swa = True
        args.trainer = 'torchCPUTrainer'
        initial_result = Execute(args).start()

        aswa_path = initial_result['path_experiment_folder'] + '/aswa.pt'
        assert os.path.isfile(aswa_path)
        initial_mtime = os.path.getmtime(aswa_path)

        args.continual_learning = initial_result['path_experiment_folder']
        continual_result = ContinuousExecute(args).continual_start()

        assert 'Train' in continual_result
        assert 'Val' in continual_result
        assert 'Test' in continual_result
        assert os.path.isfile(aswa_path)
        assert os.path.getmtime(aswa_path) >= initial_mtime
