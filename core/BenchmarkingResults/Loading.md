# Loading circa 1 Billion Rows 
## Hardware Setup
qilin
GIGABYTE Server R262-ZA2-00
2x NVIDIA GeForce RTX A5000 24GB
AMD EPYC 7713 64-Core Processor	64	1024 GB	KCD61LUL15T3 13.97 TiB CS3040 4TB SSD
Debian IRB managed 11 (bullseye)
5.10.0-14-amd64 x86_64

## Loading to Memory via Pandas
TODO: Add results
# Loading to Memory: Pandas vs DASK
In [1]: from dask import dataframe as ddf
In [2]: import pandas as pd
In [3]: %timeit -n 3 df = pd.read_parquet('cleaned_well_partitioned_dbpedia_03_2022.parquet',engine='pyarrow')
3min 5s ± 2.63 s per loop (mean ± std. dev. of 7 runs, 3 loops each)


## Preprocessing
TODO: Add results

# Outcome
Pandas is faster at reading parquet formatted data as well as preprocessing the data than DASK provided that the data fits in the memory.
[MRocklin (creator of DASK): "It's quite common for Dask DataFrame to not provide a speed up over Pandas, especially for datasets that fit comfortably into memory.](https://stackoverflow.com/a/57104255/5363103)