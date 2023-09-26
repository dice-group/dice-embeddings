from dicee.executer import Execute
import pytest
from dicee.config import Namespace
class TestCV_NegSample:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_NegSample(self):
        args = Namespace()
        args.model = 'ConEx'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_NegSample(self):
        args = Namespace()
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convq_NegSample(self):
        args = Namespace()
        args.model = 'ConvQ'
        args.scoring_technique = 'NegSample'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend = "rdflib"
        args.neg_ratio = 1
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_omult_NegSample(self):
        args = Namespace()
        args.model = 'OMult'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.neg_ratio = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convo_NegSample(self):
        args = Namespace()
        args.model = 'ConvO'
        args.scoring_technique = 'NegSample'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.neg_ratio = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    def test_distmult_NegSample(self):
        args = Namespace()
        args.model = 'DistMult'
        args.scoring_technique = 'NegSample'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.neg_ratio = 1
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    def test_complex_NegSample(self):
        args = Namespace()
        args.model = 'ComplEx'
        args.scoring_technique = 'NegSample'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.neg_ratio = 1
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()
