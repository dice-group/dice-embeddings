from dicee.executer import Execute
import pytest
from dicee.config import Namespace
class TestCV_1vsAll:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_shallom_1vs_all(self):
        args = Namespace()
        args.model = 'Shallom'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_1vs_all(self):
        args = Namespace()
        args.model = 'ConEx'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_1vs_all(self):
        args = Namespace()
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convq_1vs_all(self):
        args = Namespace()
        args.model = 'ConvQ'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_omult_1vs_all(self):
        args = Namespace()
        args.model = 'OMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.eval_model = 'test'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convo_1vs_all(self):
        args = Namespace()
        args.model = 'ConvO'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    def test_distmult_1vs_all(self):
        args = Namespace()
        args.model = 'DistMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    def test_complex_1vs_all(self):
        args = Namespace()
        args.model = 'ComplEx'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()
