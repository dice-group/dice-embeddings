from dicee.executer import Execute, get_default_arguments
import sys
import pytest


class TestCV_1vsAll:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_shallom_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'Shallom'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
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
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'ConEx'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
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
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
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
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convq_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'ConvQ'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'test'
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_omult_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'OMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
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
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convo_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'ConvO'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
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
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    def test_distmult_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'DistMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
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
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    def test_complex_1vs_all(self):
        args = get_default_arguments([])
        args.model = 'ComplEx'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.path_dataset_folder = 'KGs/Family'
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
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.num_folds_for_cv = 3
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()
