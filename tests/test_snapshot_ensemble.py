import os
import pytest
import json
from dicee.config import Namespace
from dicee.executer import Execute

class TestSnapshotEnsemble:
    """Regression tests for snapshot ensemble."""
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_regression_snapshot_ensemble(self):
        """Test snapshot ensemble regression with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 200
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.adaptive_lr = {"scheduler_name": "deferred_cca", "lr_min": 0.01, "num_cycles": 10,
                             "n_snapshots": 5, "weighted_ensemble": True}
        args.trainer = 'PL'
        result = Execute(args).start()

        ensemble_report_path = result['path_experiment_folder'] + '/ensemble_eval_report.json'
        assert os.path.exists(ensemble_report_path)
        with open(ensemble_report_path, 'r') as f:
            ensemble_report = json.load(f)
        assert isinstance(ensemble_report, dict)
        
        ensemble_eval_report = ensemble_report.get('ensemble_eval_report')
        assert ensemble_eval_report.get('MRR') > result['Test']['MRR']
        assert ensemble_eval_report.get('H@1') > result['Test']['H@1']
        assert ensemble_eval_report.get('H@3') > result['Test']['H@3']
        # assert ensemble_eval_report.get('H@10') > result['Test']['H@10']

        assert ensemble_report.get('scheduler_name') == 'deferred_cca'

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_regression_adaptive_lr(self):
        """Test snapshot ensemble regression with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.trainer = 'PL'
        args.adaptive_lr = {"scheduler_name": "cca", "lr_min": 0.01, "num_cycles": 5,
                             "weighted_ensemble": True}
        args.trainer = 'PL'
        ensemble_result = Execute(args).start()

        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.trainer = 'PL'
        result = Execute(args).start()

        assert ensemble_result['Test']['MRR'] > result['Test']['MRR']
        assert ensemble_result['Test']['H@1'] > result['Test']['H@1']
        assert ensemble_result['Test']['H@3'] > result['Test']['H@3']
        assert ensemble_result['Test']['H@10'] > result['Test']['H@10']



    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_default_snapshot_ensemble(self):
        """Test snapshot ensemble with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.adaptive_lr = {"scheduler_name": "deferred_cca"}
        args.trainer = 'PL'
        result = Execute(args).start()

        snapshot_dir = result['path_experiment_folder'] + '/snapshots'
        ensemble_report_path = result['path_experiment_folder'] + '/ensemble_eval_report.json'
        assert os.path.exists(snapshot_dir)
        assert os.path.exists(ensemble_report_path)


    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ensemble_snapshots_eval_report(self):
        """Test ensemble evaluation callback with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.trainer = 'PL'
        args.adaptive_lr = {"scheduler_name": "mmcclr", "lr_min": 0.01,
                             "num_cycles": 10, "weighted_ensemble": False}
        result = Execute(args).start()

        snapshot_dir = result['path_experiment_folder'] + '/snapshots'
        ensemble_report_path = result['path_experiment_folder'] + '/ensemble_eval_report.json'
        assert os.path.exists(snapshot_dir)
        assert os.path.exists(ensemble_report_path)

        with open(ensemble_report_path, 'r') as f:
            ensemble_report = json.load(f)

        assert isinstance(ensemble_report, dict)
        assert ensemble_report.get('scheduler_name') == 'mmcclr'
        assert ensemble_report.get('total_epochs') == args.num_epochs
        assert ensemble_report.get('lr_max') == args.lr
        assert ensemble_report.get('lr_min') == 0.01
        assert ensemble_report.get('num_cycles') == 10
        assert ensemble_report.get('weighted_ensemble') is False
        
        snapshot_loss   = ensemble_report.get('snapshot_loss')
        assert snapshot_loss is not None
        assert isinstance(snapshot_loss, dict)
        assert len(snapshot_loss) == args.adaptive_lr['num_cycles']

        pt_files = set([f for f in os.listdir(snapshot_dir) if f.endswith('.pt')])
        # Assert the keys of snapshot_loss match the pt files
        snapshot_loss_keys = set(snapshot_loss.keys())
        assert pt_files == snapshot_loss_keys

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ensemble_deferred_cycling(self):
        """Test ensemble evaluation callback with Keci model."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.backend = "pandas"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.trainer = 'PL'
        args.adaptive_lr = {"scheduler_name": "deferred_cca", "lr_min": 0.01, "num_cycles": 10,
                            "weighted_ensemble": True, "n_snapshots": 5}
        result = Execute(args).start()

        snapshot_dir = result['path_experiment_folder'] + '/snapshots'
        ensemble_report_path = result['path_experiment_folder'] + '/ensemble_eval_report.json'
        assert os.path.exists(snapshot_dir)
        assert os.path.exists(ensemble_report_path)

        with open(ensemble_report_path, 'r') as f:
            ensemble_report = json.load(f)

        assert isinstance(ensemble_report, dict)
        assert ensemble_report.get('scheduler_name') == 'deferred_cca'
        assert ensemble_report.get('total_epochs') == args.num_epochs
        assert ensemble_report.get('lr_max') == args.lr
        assert ensemble_report.get('lr_min') == 0.01
        assert ensemble_report.get('num_cycles') == 10
        assert ensemble_report.get('n_snapshots') == 5
        assert ensemble_report.get('weighted_ensemble') is True

        snapshot_loss   = ensemble_report.get('snapshot_loss')
        assert snapshot_loss is not None
        assert isinstance(snapshot_loss, dict)
        assert len(snapshot_loss) == args.adaptive_lr['n_snapshots']

        pt_files = set([f for f in os.listdir(snapshot_dir) if f.endswith('.pt')])
        # Assert the keys of snapshot_loss match the pt files
        snapshot_loss_keys = set(snapshot_loss.keys())
        assert pt_files == snapshot_loss_keys


