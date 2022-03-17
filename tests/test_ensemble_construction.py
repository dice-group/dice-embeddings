from continual_training import argparse_default as ct_argparse_default
from main import argparse_default as main_argparse_default
from core.executer import Execute, ContinuousExecute
from core.knowledge_graph_embeddings import KGE
from core.knowledge_graph import KG
import pytest
import argparse
import os


class TestEnsembleConstruction:
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
        args.save_model_at_every_epoch = 3
        result = Execute(args).start()
        assert 0.71 >= result['Train']['H@1'] >= 0.19
        assert 0.71 >= result['Val']['H@1'] >= 0.19
        assert 0.71 >= result['Test']['H@1'] >= 0.19
        assert os.path.isdir(result['path_experiment_folder'])
        from core import KGE
        from core.knowledge_graph import KG

        # (1) Load single model
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'], construct_ensemble=True)
