from continual_training import argparse_default as ct_argparse_default
from main import argparse_default as main_argparse_default
from dicee.executer import Execute, ContinuousExecute
from dicee.knowledge_graph_embeddings import KGE
from dicee.knowledge_graph import KG
import pytest
import argparse
import os


class TestRegressionCL:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = main_argparse_default([])
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
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])
        kg = KG(data_dir=args.path_dataset_folder)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling_Family(self):
        args = main_argparse_default([])
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
        args.backend = 'pandas'  #  Error with polars becasue sep="\s" should be a single byte character, but is 2 bytes long.
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])
        kg = KG(args.path_dataset_folder, entity_to_idx=pre_trained_kge.entity_to_idx,
                relation_to_idx=pre_trained_kge.relation_to_idx)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)
