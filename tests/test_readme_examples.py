from static_funcs import argparse_default, preprocesses_input_args
from executer import Execute
import sys

class TestDefaultParams:
    def test_shallom(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'Shallom'
        Execute(args).start()

    def test_distmult(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'DistMult'
        Execute(args).start()

    def test_complex(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'ComplEx'
        Execute(args).start()

    def test_qmult(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'QMult'
        Execute(args).start()

    def test_omult(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'OMult'
        Execute(args).start()

    def test_conex(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'ConEx'
        Execute(args).start()

    def test_convq(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'ConvQ'
        Execute(args).start()

    def test_convo(self):
        args = preprocesses_input_args(argparse_default([]))
        args.model = 'ConvO'
        Execute(args).start()
