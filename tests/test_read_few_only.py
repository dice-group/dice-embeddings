from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestReadFewOnly:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_kvsall(self):
        args = Namespace()
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train'
        args.sample_triples_ratio = None
        args.read_only_few = 10
        args.trainer = 'torchCPUTrainer'
        report = Execute(args).start()
        # as we add negative triples
        assert report['num_train_triples'] == int(args.read_only_few * 2)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_1vsall(self):
        args = Namespace()  # get_default_arguments([])
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = '1vsAll'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train'
        args.read_only_few = 10
        args.sample_triples_ratio = None
        args.trainer = 'torchCPUTrainer'
        report = Execute(args).start()
        # as we add negative triples
        assert report['num_train_triples'] == int(args.read_only_few * 2)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_neg_sampling(self):
        args = Namespace()
        args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train'
        args.read_only_few = 10
        args.sample_triples_ratio = None
        args.neg_ratio = 1
        args.trainer = 'torchCPUTrainer'
        report = Execute(args).start()
        # as we add negative triples
        assert report['num_train_triples'] == args.read_only_few
