from executer import Execute
from static_funcs import argparse_default

if __name__ == '__main__':
    exc = Execute(argparse_default())
    exc.start()
