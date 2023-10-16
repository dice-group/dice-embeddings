from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestDefaultParams:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_shallom(self):
        args = Namespace()
        args.model = 'Shallom'
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
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.gradient_accumulation_steps = 5
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()
