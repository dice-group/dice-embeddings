from dicee.executer import Execute, ContinuousExecute, get_default_arguments
from dicee.knowledge_graph_embeddings import KGE
from dicee.knowledge_graph import KG
from dicee.config import Arguments
import pytest
import argparse
import os


class TestRegressionCL:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = Arguments()
        args.model = 'QMult'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.path_dataset_folder = 'KGs/Family'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path=result['path_experiment_folder'])
        kg = KG(data_dir=args.path_dataset_folder)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling_Family(self):
        args = Arguments()
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/Family'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.trainer = 'torchCPUTrainer'
        args.backend = 'pandas'  # Error with polars because sep="\s" should be a single byte character, but is 2 bytes long.
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path=result['path_experiment_folder'])
        kg = KG(args.path_dataset_folder, entity_to_idx=pre_trained_kge.entity_to_idx,
                relation_to_idx=pre_trained_kge.relation_to_idx)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)
