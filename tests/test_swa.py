from dicee.executer import Execute
import pytest
from dicee.config import Namespace


class TestSWA:

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_swa(self):
        """Test SWA with Keci model using PL trainer."""
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
        args.swa = True
        swa_report = Execute(args).start()

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
        args.swa = True
        args.swa_start_epoch = 50
        deferred_swa_report = Execute(args).start()

        assert deferred_swa_report["Val"]["MRR"] > swa_report["Val"]["MRR"]
        assert deferred_swa_report["Val"]["H@1"] > swa_report["Val"]["H@1"]
        assert deferred_swa_report["Test"]["MRR"] > swa_report["Test"]["MRR"]
        assert deferred_swa_report["Test"]["H@1"] > swa_report["Test"]["H@1"]

    def test_k_vs_all_swa_cpu_trainer(self):
        """Test SWA with Keci model using CPU trainer."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "TorchCPUTrainer"
        args.num_epochs = 100
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.swa = True
        swa_report = Execute(args).start()

        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "TorchCPUTrainer"
        args.num_epochs = 100
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.swa = True
        args.swa_start_epoch = 50
        deferred_swa_report = Execute(args).start()

        assert deferred_swa_report["Val"]["MRR"] > swa_report["Val"]["MRR"]
        assert deferred_swa_report["Val"]["H@1"] > swa_report["Val"]["H@1"]
        assert deferred_swa_report["Test"]["MRR"] > swa_report["Test"]["MRR"]
        assert deferred_swa_report["Test"]["H@1"] > swa_report["Test"]["H@1"]