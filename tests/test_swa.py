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

    @pytest.mark.filterwarnings('ignore::UserWarning')
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

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_swag(self):
        """Test SWAG with Keci and PL trainer."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 200
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        report = Execute(args).start()

        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 200
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.swag = True
        swag_report = Execute(args).start()

        assert swag_report["Val"]["MRR"] > report["Val"]["MRR"]
        assert swag_report["Val"]["H@1"] > report["Val"]["H@1"]
        assert swag_report["Test"]["MRR"] > report["Test"]["MRR"]
        assert swag_report["Test"]["H@1"] > report["Test"]["H@1"]

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_ema(self):
        """Test EMA with Keci model using PL trainer."""
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
        args.ema = True
        ema_report = Execute(args).start()

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
        args.ema = True
        args.swa_start_epoch = 50
        deferred_ema_report = Execute(args).start()

        assert deferred_ema_report["Val"]["MRR"] > ema_report["Val"]["MRR"]
        assert deferred_ema_report["Val"]["H@1"] > ema_report["Val"]["H@1"]
        assert deferred_ema_report["Test"]["MRR"] > ema_report["Test"]["MRR"]
        assert deferred_ema_report["Test"]["H@1"] > ema_report["Test"]["H@1"]

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_twa(self):
        """Test TWA with Keci model using PL trainer."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 200
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.twa = True
        args.swa_start_epoch = 10
        twa_report = Execute(args).start()

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
        args.twa = True
        args.swa_start_epoch = 50
        deferred_twa_report = Execute(args).start()

        assert deferred_twa_report["Val"]["MRR"] > twa_report["Val"]["MRR"]
        assert deferred_twa_report["Val"]["H@1"] > twa_report["Val"]["H@1"]
        assert deferred_twa_report["Test"]["MRR"] > twa_report["Test"]["MRR"]
        assert deferred_twa_report["Test"]["H@1"] > twa_report["Test"]["H@1"]