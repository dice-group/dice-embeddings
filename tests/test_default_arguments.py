from core.static_funcs import argparse_default
from core.executer import Execute
import sys


class TestDefaultParams:
    def test_shallom(self):
        args = argparse_default([])
        args.num_epochs=1
        args.model = 'Shallom'
        Execute(args).start()

    def test_conex(self):
        args = argparse_default([])
        args.num_epochs=1
        args.model = 'ConEx'
        Execute(args).start()

    def test_qmult(self):
        args = argparse_default([])
        args.num_epochs=1
        args.model = 'QMult'
        Execute(args).start()

    def test_convq(self):
        args = argparse_default([])
        args.num_epochs=1
        args.model = 'ConvQ'
        Execute(args).start()

    def test_omult(self):
        args = argparse_default([])
        args.num_epochs=1
        args.model = 'OMult'
        Execute(args).start()

    def test_convo(self):
        args = argparse_default([])
        args.num_epochs=1
        args.model = 'ConvO'
        Execute(args).start()

    def test_distmult(self):
        args = argparse_default([])
        args.num_epochs=1
        args.model = 'DistMult'
        Execute(args).start()
