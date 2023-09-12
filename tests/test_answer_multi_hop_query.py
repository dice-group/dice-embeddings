from dicee.executer import Execute
from dicee.config import Namespace
import os
from dicee.knowledge_graph_embeddings import KGE
import pytest


class TestQueryAnswering:

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_3p_query_answering(self):
        # Train a KGE model
        args = Namespace()
        args.model = 'Keci'
        args.optim = 'Adam'
        args.scoring_technique = "AllvsAll"
        args.path_single_kg = "KGs/Family/train.txt"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 512
        result = Execute(args).start()
        assert result["Train"]["MRR"] >= 0.99
        assert os.path.isdir(result['path_experiment_folder'])
        # Load the Model
        pre_trained_kge = KGE(path=result['path_experiment_folder'])
        # Query: ?P : \exist Married(P,E) \land hasSibling(E, F9M167)
        # Natural Language Question: To whom a sibling of F9M167 is married to?
        """
        # (1) Who are the siblings of F9M167 ? => F9M167 hasSibling [F9M157, F9F141]
        # (2) Whom are (1) married to ? [ (F9M157 #married F9F158), (F9F141 #married F9M142) ] 
        # Hence, the answer set is  {F9F158, F9M142}
        """
        # Prediciton => [('<http://www.benchmark.org/family#F9M142>', tensor(0.9999)),
        # ('<http://www.benchmark.org/family#F9F158>', tensor(0.9997)),
        # ('<http://www.benchmark.org/family#F9M167>', tensor(0.0011))]
        pred = pre_trained_kge.answer_multi_hop_query(query_type="2p",
                                                      query=('<http://www.benchmark.org/family#F9M167>',
                                                             (
                                                                 '<http://www.benchmark.org/family#hasSibling>',
                                                                 '<http://www.benchmark.org/family#married>')),
                                                      tnorm="prod", k=10)[:3]
        top_entities = [ent for ent, s in pred]
        assert "<http://www.benchmark.org/family#F9M142>" in top_entities
        assert "<http://www.benchmark.org/family#F9F158>" in top_entities
        # Let's make it even more difficult
        # Query: ?T : \exist type(T,P) \land Married(P,E) \land hasSibling(E, F9M167)
        # Natural Language Question: What are the type of people who are married to a sibling of F9M167?
        """
        # (3) Third hop info.
        #F9M157 is [Brother Father Grandfather Male]
        #F9M142 is [Male Grandfather Father]
        """
        # Prediction => [('<http://www.benchmark.org/family#Person>', tensor(0.9999)),
        # ('<http://www.benchmark.org/family#Male>', tensor(0.9999)),
        # ('<http://www.benchmark.org/family#Father>', tensor(0.9999))]
        pred=pre_trained_kge.answer_multi_hop_query(query_type="3p", query=("<http://www.benchmark.org/family#F9M167>",
                                                                             (
                                                                                 "<http://www.benchmark.org/family#hasSibling>",
                                                                                 "<http://www.benchmark.org/family#married>",
                                                                                 "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")),
                                                     tnorm="prod", k=10)[:3]
        top_entities = [ent for ent, s in pred]
        assert "<http://www.benchmark.org/family#Male>" in top_entities
        assert "<http://www.benchmark.org/family#Father>" in top_entities

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_multi_hop_epfo_query_answering(self):
        args = Namespace()
        args.model = 'Keci'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.path_dataset_folder = "KGs/UMLS"
        args.num_epochs = 100
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.num_core = 0
        args.num_of_output_channels = 10
        result = Execute(args).start()
        assert os.path.isdir(result['path_experiment_folder'])
        pre_trained_kge = KGE(path=result['path_experiment_folder'])
        # test conjunctions
        scores = pre_trained_kge.answer_multi_hop_query(query_type="2p", query=('vitamin', ('isa', 'interacts_with')),
                                                        tnorm="prod", neg_norm="standard", lambda_=0.0, k=4)
        assert len(scores) == 135
        scores = pre_trained_kge.answer_multi_hop_query(query_type="3p",
                                                        query=('vitamin', ('isa', 'interacts_with', 'treats')),
                                                        tnorm="min", neg_norm="standard", lambda_=0.0, k=4)
        assert len(scores) == 135
        scores = pre_trained_kge.answer_multi_hop_query(query_type="pi", query=(
        ('vitamin', ('disrupts', 'part_of')), ('plant', ('interacts_with',))),
                                                        tnorm="prod", neg_norm="standard", lambda_=0.0, k=4)
        assert len(scores) == 135
        scores = pre_trained_kge.answer_multi_hop_query(query_type="ip", query=(
        (('virus', ('causes',)), ('laboratory_procedure', ('assesses_effect_of',))), ('affects',)),
                                                        tnorm="prod", neg_norm="standard", lambda_=0.0, k=4)
        assert len(scores) == 135
        scores = pre_trained_kge.answer_multi_hop_query(query_type="2i", query=(
        ('tissue', ('produces',)), ('laboratory_procedure', ('assesses_effect_of',))),
                                                        tnorm="min", neg_norm="standard", lambda_=0.0, k=4)
        assert len(scores) == 135
        scores = pre_trained_kge.answer_multi_hop_query(query_type="3i", query=(
        ('sign_or_symptom', ('manifestation_of',)), ('diagnostic_procedure', ('associated_with',)),
        ('laboratory_or_test_result', ('indicates',))),
                                                        tnorm="prod", neg_norm="standard", lambda_=0.0, k=4)
        assert len(scores) == 135
        # test negations
        # should negations be treated as new test function
        # scores = pre_trained_kge.answer_multi_hop_query(query_type="2in", query=('vitamin', ('isa', 'interacts_with')),
        #                                                 tnorm="prod", neg_norm="sugeno", lambda_=10.0, k_=4)
        # assert len(scores) == 135
        # scores = pre_trained_kge.answer_multi_hop_query(query_type="3in", query=('vitamin', ('isa', 'interacts_with')),
        #                                                 tnorm="prod", neg_norm="sugeno", lambda_=100.0, k_=4)
        # assert len(scores) == 135
        # scores = pre_trained_kge.answer_multi_hop_query(query_type="inp", query=('vitamin', ('isa', 'interacts_with')),
        #                                                 tnorm="prod", neg_norm="yager", lambda_=0.2, k_=4)
        # assert len(scores) == 135
        # scores = pre_trained_kge.answer_multi_hop_query(query_type="pni", query=('vitamin', ('isa', 'interacts_with')),
        #                                                 tnorm="prod", neg_norm="yager", lambda_=0.3, k_=4)
        # assert len(scores) == 135
        # scores = pre_trained_kge.answer_multi_hop_query(query_type="pin", query=('vitamin', ('isa', 'interacts_with')),
        #                                                 tnorm="prod", neg_norm="yager", lambda_=0.4, k_=4)
        # assert len(scores) == 135
