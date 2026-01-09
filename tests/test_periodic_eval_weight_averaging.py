from dicee.executer import Execute
import pytest
import json
import os
from dicee.config import Namespace


class TestPeriodicEvalWeightAveraging:
    """Test class for periodic evaluation with weight averaging techniques."""

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_weight_averaging(self):
        # Test ASWA (Adaptive SWA)
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 100
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.eval_every_n_epochs = 20
        args.adaptive_swa = True
        aswa_report = Execute(args).start()
        aswa_n_epochs_file = aswa_report["path_experiment_folder"] + '/eval_report_n_epochs.json'
        aswa_n_epochs_report = json.loads(open(aswa_n_epochs_file, 'r').read())

        # Test SWA
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 100
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_every_n_epochs = 20
        args.batch_size = 1024
        args.swa = True
        args.swa_start_epoch = 50
        swa_report = Execute(args).start()
        swa_n_epochs_file = swa_report["path_experiment_folder"] + '/eval_report_n_epochs.json'
        swa_n_epochs_report = json.loads(open(swa_n_epochs_file, 'r').read())

        # Test baseline without weight averaging
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 100
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_every_n_epochs = 20
        args.batch_size = 1024
        report = Execute(args).start()

        baseline_n_epochs_file = report["path_experiment_folder"] + '/eval_report_n_epochs.json'
        baseline_n_epochs_report = json.loads(open(baseline_n_epochs_file, 'r').read())

        # Compare performance at epoch 80
        aswa_80 = aswa_n_epochs_report.get('epoch_80_eval', {})
        swa_80 = swa_n_epochs_report.get('epoch_80_eval', {})
        baseline_80 = baseline_n_epochs_report.get('epoch_80_eval', {})
        
        # Three-way comparative assertions at epoch 80
        assert aswa_80["Val"]["MRR"] > swa_80["Val"]["MRR"] > baseline_80["Val"]["MRR"] > 0.77
        assert aswa_80["Test"]["MRR"] > swa_80["Test"]["MRR"] > baseline_80["Test"]["MRR"] > 0.77
        
        # Final performance assertions
        assert aswa_report["Val"]["MRR"] > swa_report["Val"]["MRR"] > report["Val"]["MRR"] > 0.77
        assert aswa_report["Test"]["MRR"] > swa_report["Test"]["MRR"] > report["Test"]["MRR"] > 0.77