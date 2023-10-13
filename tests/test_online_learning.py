from dicee.executer import Execute
from dicee.knowledge_graph_embeddings import KGE
import torch
import pytest
import os
from dicee.config import Namespace

class TestRegressionOnlineLearning:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_umls(self):
        args = Namespace()
        args.model = 'AConEx'
        args.scoring_technique = 'KvsSample'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 0
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.backend = 'polars'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        # Load the model
        pre_trained_kge = KGE(path=result['path_experiment_folder'])
        # (1) Assume that acquired_abnormality,location_of,acquired_abnormality is a false triple
        first = pre_trained_kge.triple_score(h=["acquired_abnormality"],
                                             r=['location_of'],
                                             t=["acquired_abnormality"])

        # (2) Train the model on (1) with a negative label.
        pre_trained_kge.train_triples(h=["acquired_abnormality"],
                                      r=['location_of'],
                                      t=["acquired_abnormality"],
                                      iteration=1,
                                      optimizer=torch.optim.Adam(params=pre_trained_kge.parameters(), lr=0.01),
                                      labels=[0.0])
        # (3)
        second = pre_trained_kge.triple_score(h=["acquired_abnormality"],
                                              r=['location_of'],
                                              t=["acquired_abnormality"])
        assert second < first
