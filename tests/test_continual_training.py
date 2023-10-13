from dicee.executer import Execute
from dicee.knowledge_graph_embeddings import KGE
from dicee.knowledge_graph import KG
from dicee.config import Namespace
import pytest
import os


class TestRegressionCL:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling(self):
        args = Namespace()
        args.model = 'QMult'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
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
        kg = KG(dataset_dir=args.dataset_dir)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_negative_sampling_Family(self):
        args = Namespace()
        args.model = 'QMult'
        args.dataset_dir = 'KGs/UMLS'
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
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path=result['path_experiment_folder'])
        kg = KG(args.dataset_dir, entity_to_idx=pre_trained_kge.entity_to_idx,
                relation_to_idx=pre_trained_kge.relation_to_idx)
        pre_trained_kge.train(kg, epoch=1, batch_size=args.batch_size)
