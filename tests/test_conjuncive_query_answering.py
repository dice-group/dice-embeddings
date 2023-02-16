from dicee.executer import Execute, ContinuousExecute,get_default_arguments
from dicee.knowledge_graph_embeddings import KGE
from dicee.knowledge_graph import KG
import pytest
import argparse
import os


class TestKGEInteractive:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_missing_triples_and_conjunctive_query_answering(self):
        args = get_default_arguments([])
        args.model = 'AConEx'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.path_dataset_folder = 'KGs/UMLS'
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
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path_of_pretrained_model_dir=result['path_experiment_folder'])
        m = pre_trained_kge.find_missing_triples(confidence=0.999, topk=1, at_most=10)  # tensor([0.9309])
        assert len(m) <= 10
        x = pre_trained_kge.predict_conjunctive_query(entity='alga',
                                                      relations=['isa',
                                                                 'causes'], topk=3)

        assert len(x) > 2
