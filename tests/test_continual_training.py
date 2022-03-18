from continual_training import argparse_default as ct_argparse_default
from main import argparse_default as main_argparse_default
from core.executer import Execute, ContinuousExecute
from core.knowledge_graph_embeddings import KGE
from core.knowledge_graph import KG
import pytest
import argparse
import os


class TestRegressionCL:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = main_argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = 1
        args.eval_on_train = 1
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()
        assert 0.71 >= result['Val']['H@1'] >= 0.05
        assert os.path.isdir(result['path_experiment_folder'])
        args = ct_argparse_default([])
        args.path_experiment_folder = result['path_experiment_folder']
        ct_results = ContinuousExecute(args).start()
        assert ct_results['Train']['H@1'] >= result['Train']['H@1']
        assert ct_results['Val']['H@1'] >= result['Val']['H@1']
        assert ct_results['Test']['H@1'] >= result['Test']['H@1']

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = main_argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = 0
        args.eval_on_train = 0
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])

        kg = KG(args.path_dataset_folder, entity_to_idx=pre_trained_kge.entity_to_idx,
                relation_to_idx=pre_trained_kge.relation_to_idx)
        pre_trained_kge.train(kg, epoch=1, batch_size=32)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling_Family(self):
        args = main_argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/Family'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 50
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = 0
        args.eval_on_train = 0
        args.read_only_few = None
        args.sample_triples_ratio = None
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])

        kg = KG(args.path_dataset_folder, entity_to_idx=pre_trained_kge.entity_to_idx,
                relation_to_idx=pre_trained_kge.relation_to_idx)
        pre_trained_kge.train(kg, epoch=1, batch_size=32)
