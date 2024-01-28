from dicee.executer import Execute
import pytest
from dicee.config import Namespace


class TestASWA:

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_lowest(self):
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "torchCPUTrainer"
        args.num_epochs = 100
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.adaptive_swa = True
        aswa_report = Execute(args).start()

        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "torchCPUTrainer"
        args.num_epochs = 100
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        # args.stochastic_weight_avg = True
        swa_report = Execute(args).start()

        assert aswa_report["Val"]["MRR"] > swa_report["Val"]["MRR"]
        assert aswa_report["Val"]["H@1"] > swa_report["Val"]["H@1"]
        assert aswa_report["Test"]["MRR"] > swa_report["Test"]["MRR"]
        assert aswa_report["Test"]["H@1"] > swa_report["Test"]["H@1"]

    """

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_low(self):
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 50
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.adaptive_swa = True
        aswa_report = Execute(args).start()

        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "PL"
        args.num_epochs = 50
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.stochastic_weight_avg = True
        swa_report = Execute(args).start()

        assert aswa_report["Test"]["MRR"] > swa_report["Test"]["MRR"]
        assert aswa_report["Test"]["H@1"] > swa_report["Test"]["H@1"]

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all_mid(self):
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
        args.adaptive_swa = True
        aswa_report = Execute(args).start()

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
        args.stochastic_weight_avg = True
        swa_report = Execute(args).start()

        assert aswa_report["Val"]["MRR"] > swa_report["Val"]["MRR"]
        assert aswa_report["Test"]["MRR"] > swa_report["Test"]["MRR"]
        assert 0.88 > aswa_report["Test"]["MRR"] > swa_report["Test"]["MRR"] > 0.75
    """
