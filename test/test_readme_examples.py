from main import Execute, argparse_default, preprocesses_input_args
import argparse
import pytorch_lightning as pl


class TestDefaultParams:
    def test_shallom(self):
        parser = argparse_default()
        arg = preprocesses_input_args(parser.parse_args())
        arg.model = 'Shallom'
        Execute(arg).start()

    def test_conex(self):
        parser = argparse_default()
        arg = preprocesses_input_args(parser.parse_args())
        arg.model = 'ConEx'
        Execute(arg).start()
