# Datasets

# Small datasets
1. A dataset must be located in a folder, e.g. 'KGs/YAGO3-10'.
2. A folder must contain **train** file. If the validation and test splits are available, then they must named as **valid** and **test**, respectively.
3. **train**, **valid** and **test** must be in either [N-triples](https://www.w3.org/2001/sw/RDFCore/ntriples/) format or standard link prediction dataset format (see KGs folder).


# Large datasets
1. Larger **train**, **valid**, and **test** can be stored in any of the following compression techniques [.gz, .bz2, or .zip]
```
$ python
Python 3.10.11 (main, Apr 20 2023, 19:02:41) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> df = pd.read_csv('KGs/YAGO3-10/train.txt',sep="\s+",header=None,usecols=[0, 1, 2],names=['subject', 'relation', 'object'],dtype=str)
>>> df.shape
(1079040, 3)
>>> df.to_parquet('train.parquet')
>>> exit(1)
$ mkdir LargeKGE && mv large_kg.parquet LargeKGE
$ python main.py --path_dataset_folder LargeKGE
```