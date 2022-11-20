from main import argparse_default
from core.executer import Execute
import sys
import pytest

class TestPykeenModel:

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult(self):
        args = argparse_default([])
        args.model = 'Pykeen_DistMult'
        # args.model = 'QMult'
        args.num_epochs = 1
        args.scoring_technique = 'NegSample'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.torch_trainer = 'None'
        
        Execute(args).start()