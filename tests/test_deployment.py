from main import argparse_default
from core.executer import Execute
import sys
import pytest
from core import KGE
from core.static_funcs import random_prediction

class TestDefaultParams:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult(self):
        args = argparse_default([])
        args.model = 'QMult'
        args.optim = 'Adam'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
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
        executor = Execute(args)
        executor.start()

        pre_trained_kge = KGE(path_of_pretrained_model_dir=executor.args.full_storage_path)
        random_prediction(pre_trained_kge)
