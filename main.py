from static_funcs import preprocesses_input_args,argparse_default
from executer import Execute
if __name__ == '__main__':
    Execute(preprocesses_input_args(argparse_default())).start()
