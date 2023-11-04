from dicee.executer import Execute
from dicee.config import Namespace
import pytest

from dicee import KGE
from dicee.read_preprocess_save_load_kg.util import get_er_vocab, get_re_vocab
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import torch
from tqdm import tqdm

from dicee.eval_static_funcs import evaluate_link_prediction_performance, evaluate_link_prediction_performance_with_bpe


class TestDefaultParams:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_distmult(self):
        args = Namespace()
        args.model = 'DistMult'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 5
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.byte_pair_encoding = False
        args.scoring_technique = 'NegSample'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()

        train_triples = pd.read_csv("KGs/UMLS/train.txt",
                                    sep="\s+",
                                    header=None, usecols=[0, 1, 2],
                                    names=['subject', 'relation', 'object'],
                                    dtype=str).values.tolist()
        valid_triples = pd.read_csv("KGs/UMLS/valid.txt",
                                    sep="\s+",
                                    header=None, usecols=[0, 1, 2],
                                    names=['subject', 'relation', 'object'],
                                    dtype=str).values.tolist()
        test_triples = pd.read_csv("KGs/UMLS/test.txt",
                                   sep="\s+",
                                   header=None, usecols=[0, 1, 2],
                                   names=['subject', 'relation', 'object'],
                                   dtype=str).values.tolist()
        all_triples = train_triples + valid_triples + test_triples
        model = KGE(result["path_experiment_folder"])
        assert result["Train"] == evaluate_link_prediction_performance(model, triples=train_triples,
                                                                       er_vocab=get_er_vocab(all_triples),
                                                                       re_vocab=get_re_vocab(all_triples))
        assert result["Val"] == evaluate_link_prediction_performance(model, triples=valid_triples,
                                                                     er_vocab=get_er_vocab(all_triples),
                                                                     re_vocab=get_re_vocab(all_triples))
        assert result["Test"] == evaluate_link_prediction_performance(model, triples=test_triples,
                                                                      er_vocab=get_er_vocab(all_triples),
                                                                      re_vocab=get_re_vocab(all_triples))
