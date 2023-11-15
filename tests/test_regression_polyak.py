from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestPolyak:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ppe_keci_k_vs_all(self):
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.num_epochs = 200
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        reports = Execute(args).start()
        assert reports["Train"]["MRR"]>=0.998
        assert reports["Val"]["MRR"]  >=0.729
        assert reports["Test"]["MRR"] >= 0.751
        """
        Evaluate Keci on Train set: Evaluate Keci on Train set
        {'H@1': 0.9966449386503068, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9983064928425357}
        Evaluate Keci on Validation set: Evaluate Keci on Validation set
        {'H@1': 0.6134969325153374, 'H@3': 0.8098159509202454, 'H@10': 0.9424846625766872, 'MRR': 0.7293869361804316}
        Evaluate Keci on Test set: Evaluate Keci on Test set
        {'H@1': 0.6437216338880484, 'H@3': 0.8275340393343419, 'H@10': 0.959909228441755, 'MRR': 0.751216359363361}
        Total Runtime: 13.259 seconds
        """
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.num_epochs = 200
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.callbacks = {"PPE": {"epoch_to_start": 100}}
        ppe_reports = Execute(args).start()
        assert ppe_reports["Train"]["MRR"]>=0.996
        assert ppe_reports["Val"]["MRR"]  >=0.731
        assert ppe_reports["Test"]["MRR"] >= 0.755

        assert ppe_reports["Test"]["MRR"]>reports["Test"]["MRR"]
        """
        Evaluate Keci on Train set: Evaluate Keci on Train set
        {'H@1': 0.9934815950920245, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9966609151329243}
        Evaluate Keci on Validation set: Evaluate Keci on Validation set
        {'H@1': 0.7001533742331288, 'H@3': 0.8696319018404908, 'H@10': 0.9585889570552147, 'MRR': 0.7946759330503159}
        Evaluate Keci on Test set: Evaluate Keci on Test set
        {'H@1': 0.710287443267776, 'H@3': 0.8789712556732224, 'H@10': 0.9780635400907716, 'MRR': 0.8082179592109334}
        Total Runtime: 12.497 seconds
        """
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.scoring_technique = "KvsAll"
        args.dataset_dir = "KGs/UMLS"
        args.num_epochs = 200
        args.lr = 0.1
        args.embedding_dim = 32
        args.batch_size = 1024
        args.adaptive_swa = True
        adaptive_swa_report = Execute(args).start()
        assert adaptive_swa_report["Train"]["MRR"]>=0.987
        assert adaptive_swa_report["Val"]["MRR"]  >=0.872
        assert adaptive_swa_report["Test"]["MRR"] >= 0.872
        assert adaptive_swa_report["Test"]["MRR"]>ppe_reports["Test"]["MRR"]

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_polyak_qmult_k_vs_all(self):
        args = Namespace()
        args.model = 'QMult'
        args.dataset_dir = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.callbacks = {'PPE':None}
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.70 >= result['Train']['H@1'] >= 0.68
        assert 0.778 >= result['Train']['MRR'] >= 0.775
        assert 0.640 >= result['Val']['H@1'] >= 0.630
        assert 0.640 >= result['Test']['H@1'] >= 0.620
        assert result['Train']['H@10'] >= result['Train']['H@3'] >= result['Train']['H@1']
        assert result['Val']['H@10'] >= result['Val']['H@3'] >= result['Val']['H@1']
        assert result['Test']['H@10'] >= result['Test']['H@3'] >= result['Test']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_polyak_qmult_k_vs_all(self):
        args = Namespace()
        args.model = 'QMult'
        args.dataset_dir = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 59
        args.batch_size = 1024
        args.lr = 0.1
        args.callbacks = {'FPPE': None}
        args.embedding_dim = 128
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.callbacks = ['PPE']
        args.normalization = 'LayerNorm'
        args.init_param = 'xavier_normal'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 1.0 >= result['Train']['MRR'] >= 0.05
