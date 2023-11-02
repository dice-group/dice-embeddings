from dicee.executer import Execute
import pytest
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
import os


class TestIndictiveLP:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_inductive(self):
        args = Namespace()
        args.dataset_dir = 'KGs/UMLS'
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.embedding_dim = 32
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.num_epochs = 500
        args.batch_size = 1024
        args.lr = 0.001
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.byte_pair_encoding = True
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.trainer = 'PL'
        result = Execute(args).start()
        assert result['Train']['MRR'] >= 0.88
        assert result['Val']['MRR'] >= 0.78
        assert result['Test']['MRR'] >= 0.78
        pre_trained_kge = KGE(path=result['path_experiment_folder'])
        assert (pre_trained_kge.predict(h="alga", r="isa", t="entity", logits=False) >
                pre_trained_kge.predict(h="Demir", r="loves", t="Embeddings", logits=False))
