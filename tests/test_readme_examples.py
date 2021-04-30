from static_funcs import preprocesses_input_args
from main import argparse_default
from executer import Execute
import sys


class TestDefaultParams:
    def test_shallom(self):
        args = argparse_default([])
        args.model = 'Shallom'
        Execute(args).start()

    def test_conex(self):
        args = argparse_default([])
        args.model = 'ConEx'
        Execute(args).start()

    def test_qmult(self):
        args = argparse_default([])
        args.model = 'QMult'
        Execute(args).start()

    def test_convq(self):
        args = argparse_default([])
        args.model = 'ConvQ'
        Execute(args).start()

    def test_omult(self):
        args = argparse_default([])
        args.model = 'OMult'
        Execute(args).start()

    def test_convo(self):
        args = argparse_default([])
        args.model = 'ConvO'
        Execute(args).start()

    def test_distmult(self):
        args = argparse_default([])
        args.model = 'DistMult'
        Execute(args).start()
