""" A benchmark for Pandas, Modin, Vaex, and Polar at reading large csv files"""
import time
import argparse


def read_by_pandas(path_csv: str):
    import pandas as pd
    print(f'Reading with Pandas {pd.__version__}...', end='\t')
    start_time = time.time()
    df = pd.read_csv(path_csv,
                     delim_whitespace=True,
                     header=None,
                     usecols=[0, 1, 2],
                     names=['subject', 'relation', 'object'],
                     dtype=str)
    print(f'Took {time.time() - start_time} seconds')
    print(f'Shape: {df.shape}')
    print(f'Type:{type(df)}')
    print(df.head())


def read_by_modin_pandas(path_csv: str):
    import modin.pandas as pd
    print(f'Reading with Modin {pd.__version__}...', end='\t')
    start_time = time.time()
    df = pd.read_csv(path_csv,
                     delim_whitespace=True,
                     header=None,
                     usecols=[0, 1, 2],
                     names=['subject', 'relation', 'object'],
                     dtype=str)
    print(f'Took {time.time() - start_time} seconds')
    print(f'Shape: {df.shape}')
    print(f'Type:{type(df)}')
    print(df.head())


def read_by_vaex(path_csv: str):
    import vaex
    print(f'Reading with Vaex {vaex.__version__}...', end='\t')
    start_time = time.time()
    df = vaex.read_csv(path_csv,
                       delim_whitespace=True,
                       header=None,
                       usecols=[0, 1, 2],
                       names=['subject', 'relation', 'object'],
                       dtype=str)
    print(f'Took {time.time() - start_time} seconds')
    print(f'Shape: {df.shape}')
    print(f'Type:{type(df)}')
    print(df.head())


def read_by_polar(path_csv: str):
    import polars
    print(f'Reading with Polar {polars.__version__}...', end='\t')
    start_time = time.time()
    df = polars.read_csv(path_csv, new_columns=['subject', 'relation', 'object'],sep="\t").to_pandas()
    print(f'Took {time.time() - start_time} seconds')
    print(f'Shape: {df.shape}')
    print(f'Type:{type(df)}')
    print(df.head())


parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True)

args = parser.parse_args()
read_by_pandas(args.path)
read_by_modin_pandas(args.path)
read_by_vaex(args.path)
read_by_polar(args.path)
