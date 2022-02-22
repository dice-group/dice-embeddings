from main import argparse_default
from core.executer import Execute
import sys


class TestDefaultParams:
    def test_shallom(self):
        args = argparse_default([])
        args.num_epochs = 1
        args.model = 'Shallom'
        args.scoring_technique = 'KvsAll'
        Execute(args).start()

    def test_conex(self):
        args = argparse_default([])
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.model = 'ConEx'
        Execute(args).start()

    def test_qmult(self):
        args = argparse_default([])
        args.num_epochs = 1
        args.model = 'QMult'
        args.scoring_technique = 'KvsAll'
        Execute(args).start()

    def test_convq(self):
        args = argparse_default([])
        args.num_epochs = 1
        args.model = 'ConvQ'
        args.scoring_technique = 'KvsAll'
        Execute(args).start()

    def test_omult(self):
        args = argparse_default([])
        args.num_epochs = 1
        args.model = 'OMult'
        args.scoring_technique = 'KvsAll'
        Execute(args).start()

    def test_convo(self):
        args = argparse_default([])
        args.num_epochs = 1
        args.model = 'ConvO'
        args.scoring_technique = 'KvsAll'
        Execute(args).start()

    def test_distmult(self):
        args = argparse_default([])
        args.num_epochs = 1
        args.model = 'DistMult'
        args.scoring_technique = 'KvsAll'
        Execute(args).start()
