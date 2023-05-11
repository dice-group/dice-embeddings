from dicee.executer import Execute
import sys
import pytest
from dicee.config import Args

class TestCV_NegSample:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_NegSample(self):
        args = Args()#get_default_arguments([])
        args.model = 'ConEx'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.path_dataset_folder = 'KGs/Family'
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.backend = 'pandas'  #  Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_NegSample(self):
        args = Args()  # get_default_arguments([])
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 1
        args.path_dataset_folder = 'KGs/Family'
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convq_NegSample(self):
        args = Args()  # get_default_arguments([])
        args.model = 'ConvQ'
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/Family'
        args.neg_ratio = 1
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_omult_NegSample(self):
        args = Args()  # get_default_arguments([])
        args.model = 'OMult'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/Family'
        args.neg_ratio = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_convo_NegSample(self):
        args = Args()  # get_default_arguments([])
        args.model = 'ConvO'
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/Family'
        args.neg_ratio = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    def test_distmult_NegSample(self):
        args = Args()  # get_default_arguments([])
        args.model = 'DistMult'
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/Family'
        args.neg_ratio = 1
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()

    def test_complex_NegSample(self):
        args = Args()  # get_default_arguments([])
        args.model = 'ComplEx'
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/Family'
        args.neg_ratio = 1
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        args.eval_model = 'train'
        args.trainer = 'torchCPUTrainer'
        args.num_folds_for_cv = 3
        Execute(args).start()
