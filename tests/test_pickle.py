from main import argparse_default
from dicee.executer import Execute
import sys
import os
import pickle
import pytest


class TestPickle:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_dismult_pickle(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.model = 'DistMult'
        executor = Execute(args)
        args.scoring_technique = 'NegSample'
        args.trainer = 'torchCPUTrainer'
        executor.start()
        pickle.dump(executor.trained_model, open("trained_model.p", "wb"))
        pickled_trained_model = pickle.load(open("trained_model.p", "rb"))
        os.remove('trained_model.p')

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_transe_pickle(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.model = 'TransE'
        args.scoring_technique = 'NegSample'
        executor = Execute(args)
        args.trainer = 'torchCPUTrainer'
        executor.start()
        pickle.dump(executor.trained_model, open("trained_model.p", "wb"))
        pickled_trained_model = pickle.load(open("trained_model.p", "rb"))
        os.remove('trained_model.p')

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_qmult_pickle(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.model = 'QMult'
        args.embedding_dim=16
        args.scoring_technique = 'NegSample'
        executor = Execute(args)
        args.trainer = 'torchCPUTrainer'
        executor.start()
        pickle.dump(executor.trained_model, open("trained_model.p", "wb"))
        pickled_trained_model = pickle.load(open("trained_model.p", "rb"))
        os.remove('trained_model.p')

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_complex_pickle(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.model = 'ComplEx'
        args.scoring_technique = 'NegSample'
        executor = Execute(args)
        args.trainer = 'torchCPUTrainer'
        executor.start()
        pickle.dump(executor.trained_model, open("trained_model.p", "wb"))
        pickled_trained_model = pickle.load(open("trained_model.p", "rb"))
        os.remove('trained_model.p')

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_pickle(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.model = 'ConEx'
        args.scoring_technique = 'NegSample'
        executor = Execute(args)
        args.trainer = 'torchCPUTrainer'
        executor.start()
        pickle.dump(executor.trained_model, open("trained_model.p", "wb"))
        pickled_trained_model = pickle.load(open("trained_model.p", "rb"))
        os.remove('trained_model.p')
