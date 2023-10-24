from dicee.executer import Execute
import pytest
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
import os



class TestIndictiveLP:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_inductive(self):
        args = Namespace()
        args.model = 'Keci'
        args.dataset_dir = 'KGs/UMLS'
        args.optim = 'Adam'
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 4
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.scoring_technique = 'NegSample'
        args.neg_ratio = 10
        args.byte_pair_encoding = True
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        assert result['Train']['MRR'] >= 0.25
        assert result['Val']['MRR'] >= 0.23
        assert result['Val']['MRR'] >= 0.23

        pre_trained_kge = KGE(path=result['path_experiment_folder'])

        assert pre_trained_kge.predict(h="alga", r="isa", t="entity") >= 0.52
        assert pre_trained_kge.predict(h="Demir", r="loves", t="Embeddings") > 0.49
