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

"""
## Results of Benchmark on CSV with Shape: (686412284, 3)
#### Pandas
Reading with Pandas 1.5.1...	Took 745.3680183887482 seconds
Shape: (686412284, 3)
Type:<class 'pandas.core.frame.DataFrame'>
                            subject  ...                                 object
0  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q1088364>
1  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q3247091>
2  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q1308013>
3  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q7112200>
4  <http://embedding.cc/entity/Q31>  ...     <http://embedding.cc/entity/Q4916>

[5 rows x 3 columns]

#### Modin
Reading with Modin 0.16.2...	UserWarning: Ray execution environment not yet initialized. Initializing...
To remove this warning, run the following python code before doing dataframe operations:

    import ray
    ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})

2022-11-08 13:26:36,701	INFO worker.py:1518 -- Started a local Ray instance.
Took 91.36292886734009 seconds
Shape: (686412284, 3)
Type:<class 'modin.pandas.dataframe.DataFrame'>
                            subject  ...                                 object
0  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q1088364>
1  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q3247091>
2  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q1308013>
3  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q7112200>
4  <http://embedding.cc/entity/Q31>  ...     <http://embedding.cc/entity/Q4916>

[5 rows x 3 columns]

#### Vaex
Reading with Vaex {'vaex': '4.14.0', 'vaex-core': '4.14.0', 'vaex-viz': '0.5.4', 'vaex-hdf5': '0.13.0', 'vaex-server': '0.8.1', 'vaex-astro': '0.9.2', 'vaex-jupyter': '0.8.0', 'vaex-ml': '0.18.0'}...	
Took 1059.6762919425964 seconds
Shape: (686412284, 3)
Type:<class 'vaex.dataframe.DataFrameLocal'>
  #  subject                           relation                                 object
  0  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P1344>  <http://embedding.cc/entity/Q1088364>
  1  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P1151>  <http://embedding.cc/entity/Q3247091>
  2  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P1546>  <http://embedding.cc/entity/Q1308013>
  3  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P5125>  <http://embedding.cc/entity/Q7112200>
  4  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P38>    <http://embedding.cc/entity/Q4916>
  5  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P1792>  <http://embedding.cc/entity/Q7021332>
  6  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P2852>  <http://embedding.cc/entity/Q1061257>
  7  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P2852>  <http://embedding.cc/entity/Q25648793>
  8  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P2852>  <http://embedding.cc/entity/Q25648794>
  9  <http://embedding.cc/entity/Q31>  <http://embedding.cc/prop/direct/P2852>  <http://embedding.cc/entity/Q25648798>

#### Polar
# Most computation takes place at dataframe conversion:
# Polar is fast see https://www.pola.rs/benchmarks.html
Reading with Polar 0.14.26...	Took 143.62675404548645 seconds
Shape: (686412283, 3)
Type:<class 'pandas.core.frame.DataFrame'>
                            subject  ...                                   object
0  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q3247091> .
1  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q1308013> .
2  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q7112200> .
3  <http://embedding.cc/entity/Q31>  ...     <http://embedding.cc/entity/Q4916> .
4  <http://embedding.cc/entity/Q31>  ...  <http://embedding.cc/entity/Q7021332> .

[5 rows x 3 columns]
"""