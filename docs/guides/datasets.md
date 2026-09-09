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

## DataLoader workers and memory

`--num_core` selects the number of DataLoader workers; its default is `0`.
The indexed and BPE training datasets store variable-length targets in flat
integer tensors with row offsets. Strict negative sampling and FSDP use sorted
numeric pair indices. They retain no per-query Python lists, dictionaries, or
sets for workers to access, avoiding the copy-on-write growth described in
[PyTorch issue #13246](https://github.com/pytorch/pytorch/issues/13246).

With `spawn` or `forkserver`, PyTorch shares tensor storage between workers.
Ordinary NumPy dataset arrays use a cached tensor wrapper during serialization;
contiguous file-backed arrays are reopened from their filename, shape, dtype,
and byte offset. Keep these files available and dataset buffers unchanged while
workers run. FSDP pads positive indices only for each batch. Its negative sampler
counts distinct positives and rejects queries with no available negative entity.

Preprocessing builds all three evaluation filter vocabularies in one pass over
the available splits, without concatenating them or sending copies to processes.
The existing filter dictionaries and pickle filenames are preserved. Evaluation
accepts both the ready dictionaries and futures from older preprocessing code.

To inspect private worker memory without training or rerunning link-prediction
benchmarks:

```bash
python benchmarks/dataloader_memory.py --dataset KvsAll --start-method fork \
  --pairs 200000 --workers 2 --output Experiments/dataloader-memory.json
```

The probe also supports `AllvsAll`, `KvsSample`, `BPE`, `Strict`, and `FSDP`, and
available multiprocessing start methods. It reports USS (private memory);
summing worker RSS would count shared pages repeatedly. Worker processes,
prefetched batches, and per-batch calculations still require memory.
