import functools
import polars
import pandas as pd
import numpy as np
from typing import Tuple
import glob
import time
from collections import defaultdict
from .sanity_checkers import sanity_checking_with_arguments
import os
import multiprocessing
import concurrent

enable_log = False


def timeit(func):
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if enable_log:
            if args is not None:
                s_args = [type(i) for i in args]
            else:
                s_args = args
            if kwargs is not None:
                s_kwargs = {k: type(v) for k, v in kwargs.items()}
            else:
                s_kwargs = kwargs
            print(f'Function {func.__name__} with  Args:{s_args} | Kwargs:{s_kwargs} took {total_time:.4f} seconds')
        else:
            print(f'Took {total_time:.4f} seconds')

        return result

    return timeit_wrapper


@timeit
def read_process_modin(data_path, read_only_few: int = None, sample_triples_ratio: float = None):
    print(f'*** Reading {data_path} with Modin ***')

    import modin.pandas as pd
    if data_path[-3:] in ['txt', 'csv']:
        print('Reading with modin.read_csv with sep ** s+ ** ...')
        df = pd.read_csv(data_path,
                         sep='\s+',
                         header=None,
                         usecols=[0, 1, 2],
                         names=['subject', 'relation', 'object'],
                         dtype=str)
    else:
        df = pd.read_parquet(data_path, engine='pyarrow')

    # df <class 'modin.pandas.dataframe.DataFrame'>
    # return pandas DataFrame
    # (2)a Read only few if it is asked.
    if isinstance(read_only_few, int):
        if read_only_few > 0:
            print(f'Reading only few input data {read_only_few}...')
            df = df.head(read_only_few)
            print('Done !\n')
    # (3) Read only sample
    if sample_triples_ratio:
        print(f'Subsampling {sample_triples_ratio} of input data...')
        df = df.sample(frac=sample_triples_ratio)
        print('Done !\n')
    if sum(df.head()["subject"].str.startswith('<')) + sum(df.head()["relation"].str.startswith('<')) > 2:
        # (4) Drop Rows/triples with double or boolean: Example preprocessing
        # Drop of object does not start with **<**.
        # Specifying na to be False instead of NaN.
        print('Removing triples with literal values...')
        df = df[df["object"].str.startswith('<', na=False)]
        print('Done !\n')
        # (5) Remove **<** and **>**
        print('Removing brackets **<** and **>**...')
        df = df.apply(lambda x: x.str.removeprefix("<").str.removesuffix(">"), axis=1)
        print('Done !\n')
    return df._to_pandas()


@timeit
def read_process_polars(data_path, read_only_few: int = None, sample_triples_ratio: float = None) -> polars.DataFrame:
    """ Load and Preprocess via Polars """
    print(f'*** Reading {data_path} with Polars ***')
    # (1) Load the data
    if data_path[-3:] in ['txt', 'csv']:
        print('Reading with polars.read_csv with sep **t** ...')
        df = polars.read_csv(data_path,
                             has_header=False,
                             low_memory=False,
                             n_rows=None if read_only_few is None else read_only_few,
                             columns=[0, 1, 2],
                             dtypes=[polars.Utf8],  # str
                             new_columns=['subject', 'relation', 'object'],
                             sep="\t")  # \s+ doesn't work for polars
    else:
        df = polars.read_parquet(data_path, n_rows=None if read_only_few is None else read_only_few)

    # (2) Sample from (1)
    if sample_triples_ratio:
        print(f'Subsampling {sample_triples_ratio} of input data {df.shape}...')
        df = df.sample(frac=sample_triples_ratio)
        print(df.shape)
        print('Done !\n')

    # (3) Type heuristic prediction: If KG is an RDF KG, remove all triples where subject is not <?>.
    h = df.head().to_pandas()
    if sum(h["subject"].str.startswith('<')) + sum(h["relation"].str.startswith('<')) > 2:
        print('Removing triples with literal values...')
        df = df.filter(polars.col("object").str.starts_with('<'))
        print('Done !\n')
    return df


@timeit
def read_process_pandas(data_path, read_only_few: int = None, sample_triples_ratio: float = None):
    print(f'*** Reading {data_path} with Pandas ***')
    if data_path[-3:] in ['txt', 'csv']:
        print('Reading with pandas.read_csv with sep ** s+ ** ...')
        df = pd.read_csv(data_path,
                         sep="\s+",
                         header=None,
                         usecols=[0, 1, 2],
                         names=['subject', 'relation', 'object'],
                         dtype=str)
    else:
        df = pd.read_parquet(data_path, engine='pyarrow')
    # (2)a Read only few if it is asked.
    if isinstance(read_only_few, int):
        if read_only_few > 0:
            print(f'Reading only few input data {read_only_few}...')
            df = df.head(read_only_few)
            print('Done !\n')
    # (3) Read only sample
    if sample_triples_ratio:
        print(f'Subsampling {sample_triples_ratio} of input data...')
        df = df.sample(frac=sample_triples_ratio)
        print('Done !\n')
    if sum(df.head()["subject"].str.startswith('<')) + sum(df.head()["relation"].str.startswith('<')) > 2:
        # (4) Drop Rows/triples with double or boolean: Example preprocessing
        # Drop of object does not start with **<**.
        # Specifying na to be False instead of NaN.
        print('Removing triples with literal values...')
        df = df[df["object"].str.startswith('<', na=False)]
        print('Done !\n')
    return df


def load_data(data_path, read_only_few: int = None,
              sample_triples_ratio: float = None, backend=None):
    assert backend
    # If path exits
    if glob.glob(data_path):
        if backend == 'modin':
            return read_process_modin(data_path, read_only_few, sample_triples_ratio)
        elif backend == 'pandas':
            return read_process_pandas(data_path, read_only_few, sample_triples_ratio)
        elif backend == 'polars':
            return read_process_polars(data_path, read_only_few, sample_triples_ratio)
        else:
            raise NotImplementedError(f'{backend} not found')
    else:
        print(f'{data_path} could not found!')
        return None


def index_triples(train_set, entity_to_idx: dict, relation_to_idx: dict) -> pd.core.frame.DataFrame:
    """
    :param train_set: pandas dataframe
    :param entity_to_idx: a mapping from str to integer index
    :param relation_to_idx: a mapping from str to integer index
    :param num_core: number of cores to be used
    :return: indexed triples, i.e., pandas dataframe
    """
    n, d = train_set.shape
    train_set['subject'] = train_set['subject'].apply(lambda x: entity_to_idx.get(x))
    train_set['relation'] = train_set['relation'].apply(lambda x: relation_to_idx.get(x))
    train_set['object'] = train_set['object'].apply(lambda x: entity_to_idx.get(x))
    # train_set = train_set.dropna(inplace=True)
    if isinstance(train_set, pd.core.frame.DataFrame):
        assert (n, d) == train_set.shape
    elif isinstance(train_set, dask.dataframe.core.DataFrame):
        nn, dd = train_set.shape
        assert isinstance(dd, int)
        if isinstance(nn, int):
            assert n == nn and d == dd
    else:
        raise KeyError('Wrong type training data')
    return train_set


def preprocesses_input_args(arg):
    """ Sanity Checking in input arguments """
    # To update the default value of Trainer in pytorch-lightnings
    arg.max_epochs = arg.num_epochs
    arg.min_epochs = arg.num_epochs
    assert arg.weight_decay >= 0.0
    arg.learning_rate = arg.lr
    arg.deterministic = True
    if arg.num_core < 0:
        arg.num_core = 0

    # Below part will be investigated
    arg.check_val_every_n_epoch = 10 ** 6  # ,i.e., no eval
    arg.logger = False
    try:
        assert arg.eval_model in [None, 'None', 'train', 'val', 'test', 'train_val', 'train_test', 'val_test',
                                  'train_val_test']
    except KeyError:
        print(arg.eval_model)
        exit(1)

    if arg.eval_model == 'None':
        arg.eval_model = None

    # reciprocal checking
    # @TODO We need better way for using apply_reciprical_or_noise.
    if arg.scoring_technique in ['KvsSample', 'PvsAll', 'CCvsAll', 'KvsAll', '1vsAll', 'BatchRelaxed1vsAll',
                                 'BatchRelaxedKvsAll']:
        arg.apply_reciprical_or_noise = True
    elif arg.scoring_technique == 'NegSample':
        arg.apply_reciprical_or_noise = False
    else:
        raise KeyError(f'Unexpected input for scoring_technique.\t{arg.scoring_technique}')

    if arg.sample_triples_ratio is not None:
        assert 1.0 >= arg.sample_triples_ratio >= 0.0

    assert arg.backend in ["modin", "pandas", "vaex", "polars"]
    sanity_checking_with_arguments(arg)
    if arg.model == 'Shallom':
        arg.scoring_technique = 'KvsAll'
    assert arg.normalization in [None, 'LayerNorm', 'BatchNorm1d']
    return arg


def create_constraints(triples: np.ndarray) -> Tuple[dict, dict]:
    """
    (1) Extract domains and ranges of relations
    (2) Store a mapping from relations to entities that are outside of the domain and range.
    Crete constrainted entities based on the range of relations
    :param triples:
    :return:
    """
    assert isinstance(triples, np.ndarray)
    assert triples.shape[1] == 3

    # (1) Compute the range and domain of each relation
    range_constraints_per_rel = dict()
    domain_constraints_per_rel = dict()
    set_of_entities = set()
    set_of_relations = set()
    for (e1, p, e2) in triples:
        range_constraints_per_rel.setdefault(p, set()).add(e2)
        domain_constraints_per_rel.setdefault(p, set()).add(e1)
        set_of_entities.add(e1)
        set_of_relations.add(p)
        set_of_entities.add(e2)

    for rel in set_of_relations:
        range_constraints_per_rel[rel] = list(set_of_entities - range_constraints_per_rel[rel])
        domain_constraints_per_rel[rel] = list(set_of_entities - domain_constraints_per_rel[rel])

    return domain_constraints_per_rel, range_constraints_per_rel


def get_er_vocab(data):
    # head entity and relation
    er_vocab = defaultdict(list)
    for triple in data:
        er_vocab[(triple[0], triple[1])].append(triple[2])
    return er_vocab


def get_re_vocab(data):
    # head entity and relation
    re_vocab = defaultdict(list)
    for triple in data:
        re_vocab[(triple[1], triple[2])].append(triple[0])
    return re_vocab


def get_ee_vocab(data):
    # head entity and relation
    ee_vocab = defaultdict(list)
    for triple in data:
        ee_vocab[(triple[0], triple[2])].append(triple[1])
    return ee_vocab


demir = None


def f(start, stop):
    store = dict()
    for s_idx, p_idx, o_idx in demir[start:stop]:
        store.setdefault((s_idx, p_idx), list()).append(o_idx)
    return store


@timeit
def parallel_mapping_from_first_two_cols_to_third(train_set_idx) -> dict:
    global demir
    demir = train_set_idx
    NUM_WORKERS = os.cpu_count()
    chunk_size = int(len(train_set_idx) / NUM_WORKERS)
    futures = []
    with concurrent.futures.process.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i in range(0, NUM_WORKERS):
            start = i + chunk_size if i == 0 else 0
            futures.append(executor.submit(f, start, i + chunk_size))
    futures, _ = concurrent.futures.wait(futures)
    result = dict()
    for i in futures:
        d = i.result()
        result = result | d
    del demir
    return result


@timeit
def mapping_from_first_two_cols_to_third(train_set_idx):
    store = dict()
    for s_idx, p_idx, o_idx in train_set_idx:
        store.setdefault((s_idx, p_idx), list()).append(o_idx)
    return store
