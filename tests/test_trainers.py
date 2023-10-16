from dicee.executer import Execute
from dicee.config import Namespace
import pytest

class TestCallback:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_torch_cpu_trainer(self):
        args = Namespace()
        args.model = 'AConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_aconex_pl_trainer(self):
        args = Namespace()
        args.model = 'AConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'PL'
        Execute(args).start()
