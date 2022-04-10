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
    def test_negative_sampling(self):
        args = main_argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/Family'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = False
        args.eval_on_train = False
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])
        kg = KG(args.path_dataset_folder, entity_to_idx=pre_trained_kge.entity_to_idx,
                relation_to_idx=pre_trained_kge.relation_to_idx)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling_Family(self):
        args = main_argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/Family'
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = False
        args.eval_on_train = False
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])
        kg = KG(args.path_dataset_folder, entity_to_idx=pre_trained_kge.entity_to_idx,
                relation_to_idx=pre_trained_kge.relation_to_idx)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling_Family_cbd_learning(self):
        args = main_argparse_default([])
        args.model = 'QMult'
        args.path_dataset_folder = 'KGs/Family'
        args.num_epochs = 1
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'KvsAll'
        args.eval = False
        args.eval_on_train = False
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])
        id_heads = pre_trained_kge.train_set['subject'].value_counts().nlargest(3)
        entities = {pre_trained_kge.entity_to_idx.iloc[i].name for i in id_heads.to_list()}
        for ith, entity in enumerate(entities):
            pre_trained_kge.train_cbd(head_entity=[entity], iteration=1, lr=.01)
            break
