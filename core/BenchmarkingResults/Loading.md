# Loading circa 1 Billion Rows 
## Hardware Setup
qilin
GIGABYTE Server R262-ZA2-00
2x NVIDIA GeForce RTX A5000 24GB
AMD EPYC 7713 64-Core Processor	64	1024 GB	KCD61LUL15T3 13.97 TiB CS3040 4TB SSD
Debian IRB managed 11 (bullseye)
5.10.0-14-amd64 x86_64

## Loading to Memory via Pandas
# Loading to Memory: Pandas vs DASK
In [1]: from dask import dataframe as ddf
In [2]: import pandas as pd
In [3]: %timeit -n 3 df = pd.read_parquet('cleaned_well_partitioned_dbpedia_03_2022.parquet',engine='pyarrow')
3min 5s ± 2.63 s per loop (mean ± std. dev. of 7 runs, 3 loops each)

