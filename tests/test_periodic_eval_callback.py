from ast import arg
import os
import json
import pytest
from dicee.config import Namespace
from dicee.executer import Execute


class TestPeriodicEvalCallback:
    """Regression tests for periodic evaluation."""
    
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_eval_every_n_epochs(self):
        """Test periodic evaluation callback with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_every_n_epochs = 3
        args.trainer = 'torchCPUTrainer'
        
        result = Execute(args).start()

        eval_report_path = result['path_experiment_folder'] + '/eval_report_n_epochs.json'

        # if last epoch is in _n_epochs, its skipped as it is evaluated at training end
        eval_epochs = list(range(args.eval_every_n_epochs, args.num_epochs + 1, args.eval_every_n_epochs))
        if args.num_epochs in eval_epochs:
            eval_epochs.remove(args.num_epochs)

        assert os.path.exists(eval_report_path)
        with open(eval_report_path, 'r') as f:
            json_report = json.load(f)
        assert isinstance(json_report, dict)
        assert len(json_report) == len(eval_epochs)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_eval_at_epochs(self):
        """Test periodic evaluation callback with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_at_epochs = [3, 5, 8]
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()

        eval_report_path = result['path_experiment_folder'] + '/eval_report_n_epochs.json'

        assert os.path.exists(eval_report_path)
        with open(eval_report_path, 'r') as f:
            eval_report_n_epochs = json.load(f)

        assert isinstance(eval_report_n_epochs, dict)

        if args.num_epochs in args.eval_at_epochs:
            assert len(eval_report_n_epochs) == len(args.eval_at_epochs) - 1
        else:
            assert len(eval_report_n_epochs) == len(args.eval_at_epochs)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_eval_every_n_epochs_and_at_epochs(self):
        """Test periodic evaluation callback with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 12
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_every_n_epochs = 4
        args.eval_at_epochs = [3, 7, 10]
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()

        eval_report_path = result['path_experiment_folder'] + '/eval_report_n_epochs.json'
        assert os.path.exists(eval_report_path)
        with open(eval_report_path, 'r') as f:
            eval_report_n_epochs = json.load(f)

        # Check if the report is a dictionary
        assert isinstance(eval_report_n_epochs, dict)

        # Check if the number of epochs in the report matches the expected evaluation epochs
        expected_eval_epochs = set(range(args.eval_every_n_epochs, args.num_epochs + 1, args.eval_every_n_epochs))
        expected_eval_epochs.update(args.eval_at_epochs)

        if args.num_epochs in expected_eval_epochs:
            expected_eval_epochs.remove(args.num_epochs)

        n_step_eval_epochs = set(eval_report_n_epochs.keys())
        for epoch in expected_eval_epochs:
            assert f"epoch_{epoch}_eval" in n_step_eval_epochs

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_eval_every_n_epochs_with_save_model(self):
        """Test periodic evaluation callback with Keci model and model saving."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 12
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_every_n_epochs = 4
        args.save_every_n_epochs = True
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()

        eval_report_path = result['path_experiment_folder'] + '/eval_report_n_epochs.json'
        assert os.path.exists(eval_report_path)
        with open(eval_report_path, 'r') as f:
            eval_report_n_epochs = json.load(f)

        # Check if the report is a dictionary
        assert isinstance(eval_report_n_epochs, dict)

        checkpoints_dir = result['path_experiment_folder'] + '/models_n_epochs'
        assert os.path.exists(checkpoints_dir)

        # Check if the number of epochs in the report matches the expected evaluation epochs
        expected_eval_epochs = set(range(args.eval_every_n_epochs, args.num_epochs + 1, args.eval_every_n_epochs))
        if args.num_epochs in expected_eval_epochs:
            expected_eval_epochs.remove(args.num_epochs)

        pt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        assert len(pt_files) == len(expected_eval_epochs)

        for epoch in expected_eval_epochs:
            assert f"model_at_epoch_{epoch}.pt" in pt_files

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_eval_every_n_epochs_eval_model(self):
        """Test periodic evaluation callback with Keci model and model evaluation."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 12
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.eval_every_n_epochs = 4
        args.n_epochs_eval_model = 'test_val'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()

        eval_report_path = result['path_experiment_folder'] + '/eval_report_n_epochs.json'
        assert os.path.exists(eval_report_path)
        with open(eval_report_path, 'r') as f:
            eval_report_n_epochs = json.load(f)

        # Check if the report is a dictionary
        assert isinstance(eval_report_n_epochs, dict)

        # Check if the number of epochs in the report matches the expected evaluation epochs
        expected_eval_epochs = set(range(args.eval_every_n_epochs, args.num_epochs + 1, args.eval_every_n_epochs))

        if args.num_epochs in expected_eval_epochs:
            if all(split in args.eval_model.split('_') for split in args.n_epochs_eval_model.split('_')):
                expected_eval_epochs.remove(args.num_epochs)

        for epoch in expected_eval_epochs:
            assert f"epoch_{epoch}_eval" in eval_report_n_epochs.keys()

        for eval_epochs in eval_report_n_epochs.keys():
            eval_report_epoch = eval_report_n_epochs[eval_epochs]
            assert isinstance(eval_report_epoch, dict)
            eval_modes = eval_report_epoch.keys()
            for eval_model in args.n_epochs_eval_model.split('_'):
                assert any(eval_model.lower() == mode.lower() for mode in eval_modes)