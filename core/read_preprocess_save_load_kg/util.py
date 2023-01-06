from collections import defaultdict
import numpy as np
import polars
import glob
import time
import functools
import pandas as pd
from core.static_funcs import numpy_data_type_changer
import concurrent
import pickle
import sys
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
def read_with_modin(data_path, read_only_few: int = None, sample_triples_ratio: float = None):
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
def read_with_polars(data_path, read_only_few: int = None, sample_triples_ratio: float = None) -> polars.DataFrame:
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
        df = polars.read_parquet(data_path, use_pyarrow=True)

    print(f'Estimated size of the Polars Dataframe: {df.estimated_size()/1000000} in MB')
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
def read_with_pandas(data_path, read_only_few: int = None, sample_triples_ratio: float = None):
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


def read_from_disk(data_path, read_only_few: int = None,
                   sample_triples_ratio: float = None, backend=None):
    assert backend
    # If path exits
    if glob.glob(data_path):
        if backend == 'modin':
            return read_with_modin(data_path, read_only_few, sample_triples_ratio)
        elif backend == 'pandas':
            return read_with_pandas(data_path, read_only_few, sample_triples_ratio)
        elif backend == 'polars':
            return read_with_polars(data_path, read_only_few, sample_triples_ratio)
        else:
            raise NotImplementedError(f'{backend} not found')
    else:
        print(f'{data_path} could not found!')
        return None


def get_er_vocab(data, file_path: str = None):
    # head entity and relation
    er_vocab = defaultdict(list)
    for triple in data:
        er_vocab[(triple[0], triple[1])].append(triple[2])
    if file_path:
        save_pickle(data=er_vocab, file_path=file_path)
    return er_vocab


def get_re_vocab(data, file_path: str = None):
    # head entity and relation
    re_vocab = defaultdict(list)
    for triple in data:
        re_vocab[(triple[1], triple[2])].append(triple[0])
    if file_path:
        save_pickle(data=re_vocab, file_path=file_path)
    return re_vocab


def get_ee_vocab(data, file_path: str = None):
    # head entity and relation
    ee_vocab = defaultdict(list)
    for triple in data:
        ee_vocab[(triple[0], triple[2])].append(triple[1])
    if file_path:
        save_pickle(data=ee_vocab, file_path=file_path)
    return ee_vocab


def create_constraints(triples,file_path: str = None):
    """
    (1) Extract domains and ranges of relations
    (2) Store a mapping from relations to entities that are outside of the domain and range.
    Crete constrainted entities based on the range of relations
    :param triples:
    :return:
    Tuple[dict, dict]
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

    if file_path:
        save_pickle(data=(domain_constraints_per_rel, range_constraints_per_rel), file_path=file_path)
    return domain_constraints_per_rel, range_constraints_per_rel


@timeit
def load_with_pandas(self) -> None:
    """ Deserialize data """
    print(f'Deserialization Path: {self.kg.deserialize_flag}\n')
    start_time = time.time()
    print('[1 / 4] Deserializing compressed entity integer mapping...')
    self.kg.entity_to_idx = pd.read_parquet(self.kg.deserialize_flag + '/entity_to_idx.gzip')
    print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
    self.kg.num_entities = len(self.kg.entity_to_idx)

    print('[2 / ] Deserializing compressed relation integer mapping...')
    start_time = time.time()
    self.kg.relation_to_idx = pd.read_parquet(self.kg.deserialize_flag + '/relation_to_idx.gzip')
    print(f'Done !\t{time.time() - start_time:.3f} seconds\n')

    self.kg.num_relations = len(self.kg.relation_to_idx)
    print(
        '[3 / 4] Converting integer and relation mappings from from pandas dataframe to dictionaries for an easy access...',
    )
    start_time = time.time()
    self.kg.entity_to_idx = self.kg.entity_to_idx.to_dict()['entity']
    self.kg.relation_to_idx = self.kg.relation_to_idx.to_dict()['relation']
    print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
    # 10. Serialize (9).
    print('[4 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
    start_time = time.time()
    self.kg.train_set = pd.read_parquet(self.kg.deserialize_flag + '/idx_train_df.gzip').values
    print(f'Done !\t{time.time() - start_time:.3f} seconds\n')
    try:
        print('[5 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
        self.kg.valid_set = pd.read_parquet(self.kg.deserialize_flag + '/idx_valid_df.gzip').values
        print('Done!\n')
    except FileNotFoundError:
        print('No valid data found!\n')
        self.kg.valid_set = None  # pd.DataFrame()

    try:
        print('[6 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
        self.kg.test_set = pd.read_parquet(self.kg.deserialize_flag + '/idx_test_df.gzip').values
        print('Done!\n')
    except FileNotFoundError:
        print('No test data found\n')
        self.kg.test_set = None

    if self.kg.eval_model:
        if self.kg.valid_set is not None and self.kg.test_set is not None:
            # 16. Create a bijection mapping from subject-relation pairs to tail entities.
            data = np.concatenate([self.kg.train_set, self.kg.valid_set, self.kg.test_set])
        else:
            data = self.kg.train_set
        print('[7 / 4] Creating er,re, and ee type vocabulary for evaluation...')
        start_time = time.time()
        self.kg.er_vocab = get_er_vocab(data)
        self.kg.re_vocab = get_re_vocab(data)
        # 17. Create a bijection mapping from subject-object pairs to relations.
        self.kg.ee_vocab = get_ee_vocab(data)
        self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = create_constraints(
            self.kg.train_set)
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')


@timeit
def load_with_polars(deserialize_flag):
    print(f'Deserialization Path: {deserialize_flag}\n')
    entity_to_idx = polars.read_parquet(deserialize_flag + '/entity_to_idx')
    relation_to_idx = polars.read_parquet(deserialize_flag + '/relation_to_idx')

    print(entity_to_idx)
    exit(1)
    self.kg.entity_to_idx = dict(
        zip(self.kg.entity_to_idx['entity'].to_list(), list(range(len(self.kg.entity_to_idx)))))
    self.kg.relation_to_idx = dict(
        zip(self.kg.relation_to_idx['relation'].to_list(), list(range(len(self.kg.relation_to_idx)))))

    self.kg.train_set = polars.read_parquet(self.kg.deserialize_flag + '/idx_train_df').to_numpy()
    self.kg.train_set = numpy_data_type_changer(self.kg.train_set,
                                                num=max(self.kg.num_entities, self.kg.num_relations))

    try:
        print('[5 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
        self.kg.valid_set = polars.read_parquet(self.kg.deserialize_flag + '/idx_valid_df').to_numpy()
        self.kg.valid_set = numpy_data_type_changer(self.kg.valid_set,
                                                    num=max(self.kg.num_entities, self.kg.num_relations))

        print('Done!\n')
    except FileNotFoundError:
        print('No valid data found!\n')
        self.kg.valid_set = None

    try:
        print('[6 / 4] Deserializing integer mapped data and mapping it to numpy ndarray...')
        self.kg.test_set = polars.read_parquet(self.kg.deserialize_flag + '/idx_test_df').to_numpy()
        self.kg.test_set = numpy_data_type_changer(self.kg.test_set,
                                                   num=max(self.kg.num_entities, self.kg.num_relations))
        print('Done!\n')
    except FileNotFoundError:
        print('No test data found\n')
        self.kg.test_set = None

    if self.kg.eval_model:
        if self.kg.valid_set is not None and self.kg.test_set is not None:
            # 16. Create a bijection mapping from subject-relation pairs to tail entities.
            data = np.concatenate([self.kg.train_set, self.kg.valid_set, self.kg.test_set])
        else:
            data = self.kg.train_set
        print('[7 / 4] Creating er,re, and ee type vocabulary for evaluation...')
        start_time = time.time()
        self.kg.er_vocab = get_er_vocab(data)
        self.kg.re_vocab = get_re_vocab(data)
        # 17. Create a bijection mapping from subject-object pairs to relations.
        self.kg.ee_vocab = get_ee_vocab(data)
        self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = create_constraints(
            self.kg.train_set)
        print(f'Done !\t{time.time() - start_time:.3f} seconds\n')


def save_numpy_ndarray(*, data: np.ndarray, file_path: str):
    n, d = data.shape
    assert n > 0
    assert d == 3
    with open(file_path, 'wb') as f:
        np.save(f, data)


def load_numpy_ndarray(*, file_path: str):
    with open(file_path, 'rb') as f:
        return np.load(f)


def save_pickle(*, data: object, file_path=str):
    pickle.dump(data, open(file_path, "wb"))


def load_pickle(*, file_path=str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_recipriocal_triples(x):
    """
    Add inverse triples into dask dataframe
    :param x:
    :return:
    """
    return pd.concat([x, x['object'].to_frame(name='subject').join(
        x['relation'].map(lambda x: x + '_inverse').to_frame(name='relation')).join(
        x['subject'].to_frame(name='object'))], ignore_index=True)



def index_triples_with_pandas(train_set, entity_to_idx: dict, relation_to_idx: dict) -> pd.core.frame.DataFrame:
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


def dataset_sanity_checking(train_set: np.ndarray, num_entities: int, num_relations: int) -> None:
    """

    :param train_set:
    :param num_entities:
    :param num_relations:
    :return:
    """
    assert isinstance(train_set, np.ndarray)
    n, d = train_set.shape
    assert d == 3
    try:
        assert n > 0
    except AssertionError:
        print('Size of the training dataset must be greater than 0.')
        exit(1)
    try:
        assert num_entities >= max(train_set[:, 0]) and num_entities >= max(train_set[:, 2])
    except AssertionError:
        print(
            f'Entity Indexing Error:\nMax ID of a subject or object entity in train set:{max(train_set[:, 0])} or {max(train_set[:, 2])} is greater than num_entities:{num_entities}')
        print('Exiting...')
        exit(1)
    try:
        assert num_relations >= max(train_set[:, 1])
    except AssertionError:
        print(
            f'Relation Indexing Error:\nMax ID of a relation in train set:{max(train_set[:, 1])} is greater than num_entities:{num_relations}')
        print('Exiting...')
        exit(1)
    # 13. Sanity checking: data types
    assert isinstance(train_set[0], np.ndarray)
    # assert isinstance(train_set[0][0], np.int64) and isinstance(train_set[0][1], np.int64)
    # assert isinstance(train_set[0][2], np.int64)
