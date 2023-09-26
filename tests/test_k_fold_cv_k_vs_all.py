from dicee.executer import Execute
import pytest
from dicee.config import Namespace
class TestCV_KvsAll:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_shallom_kvs_all(self):
        args = Namespace()
        args.model = 'Shallom'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
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
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_kvs_all(self):
        args = Namespace()
        args.model = 'ConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
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
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_kvs_all(self):
        args = Namespace()
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
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
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convq_kvs_all(self):
        args = Namespace()
        args.model = 'ConvQ'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
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
        args.eval_model = 'train'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_omult_kvs_all(self):
        args = Namespace()
        args.model = 'OMult'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
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
        args.eval_model = 'train'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convo_kvs_all(self):
        args = Namespace()
        args.model = 'ConvO'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    def test_distmult_kvs_all(self):
        args = Namespace()
        args.model = 'DistMult'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.eval_model = 'train'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    def test_complex_kvs_all(self):
        args = Namespace()
        args.model = 'ComplEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_single_kg = 'KGs/Family/family-benchmark_rich_background.owl'
        args.backend="rdflib"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()
