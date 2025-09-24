from collections import defaultdict
import numpy as np
import polars
import glob
import time
import functools
import pandas as pd
import pickle
import os
import psutil
import requests
from typing import Tuple
import polars as pl
from multiprocessing import Process, cpu_count
from tqdm import tqdm

def polars_dataframe_indexer(df_polars:polars.DataFrame, idx_entity:polars.DataFrame, idx_relation:polars.DataFrame)->polars.DataFrame:
    """
     Replaces 'subject', 'relation', and 'object' columns in the input Polars DataFrame with their corresponding index values
     from the entity and relation index DataFrames.

     This function processes the DataFrame in three main steps:
     1. Replace the 'relation' values with the corresponding index from `idx_relation`.
     2. Replace the 'subject' values with the corresponding index from `idx_entity`.
     3. Replace the 'object' values with the corresponding index from `idx_entity`.

     Parameters:
     -----------
     df_polars : polars.DataFrame
         The input Polars DataFrame containing columns: 'subject', 'relation', and 'object'.

     idx_entity : polars.DataFrame
         A Polars DataFrame that contains the mapping between entity names and their corresponding indices.
         Must have columns: 'entity' and 'index'.

     idx_relation : polars.DataFrame
         A Polars DataFrame that contains the mapping between relation names and their corresponding indices.
         Must have columns: 'relation' and 'index'.

     Returns:
     --------
     polars.DataFrame
         A DataFrame with the 'subject', 'relation', and 'object' columns replaced by their corresponding indices.

     Example Usage:
     --------------
     >>> df_polars = pl.DataFrame({
             "subject": ["Alice", "Bob", "Charlie"],
             "relation": ["knows", "works_with", "lives_in"],
             "object": ["Dave", "Eve", "Frank"]
         })
     >>> idx_entity = pl.DataFrame({
             "entity": ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"],
             "index": [0, 1, 2, 3, 4, 5]
         })
     >>> idx_relation = pl.DataFrame({
             "relation": ["knows", "works_with", "lives_in"],
             "index": [0, 1, 2]
         })
     >>> polars_dataframe_indexer(df_polars, idx_entity, idx_relation)

     Steps:
     ------
     1. Join the input DataFrame `df_polars` on the 'relation' column with `idx_relation` to replace the relations with their indices.
     2. Join on 'subject' to replace it with the corresponding entity index using a left join on `idx_entity`.
     3. Join on 'object' to replace it with the corresponding entity index using a left join on `idx_entity`.
     4. Select only the 'subject', 'relation', and 'object' columns to return the final result.
     """
    assert isinstance(df_polars, polars.DataFrame)
    assert isinstance(idx_entity, polars.DataFrame)
    assert isinstance(idx_relation, polars.DataFrame)

    # Step : Join on 'relation' to replace relation with its index
    df_merged = df_polars.join(idx_relation, on="relation", how="left")
    df_merged = df_merged.select([polars.col("subject"), polars.col("index").alias("relation"), polars.col("object")])
    # Step :  Consider Left Table on subject and Right Table on entity with the left join
    # Returns all rows from the left table, and the matched rows from the right table
    df_merged = df_merged.join(idx_entity, left_on="subject", right_on="entity", how="left")
    df_merged = df_merged.drop("subject").rename({"index": "subject"})
    # Step 3: Join on 'object' to replace object with its index
    df_final = df_merged.join(idx_entity, left_on="object", right_on="entity", how="left")
    df_final = df_final.drop("object").rename({"index": "object"})
    # Step 4: Select the desired columns
    df_final = df_final.select([polars.col("subject"), polars.col("relation"), polars.col("object")])
    return df_final


def pandas_dataframe_indexer(df_pandas: pd.DataFrame, idx_entity: pd.DataFrame, idx_relation: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces 'subject', 'relation', and 'object' columns in the input Pandas DataFrame with their corresponding index values
    from the entity and relation index DataFrames.

    Parameters:
    -----------
    df_pandas : pd.DataFrame
        The input Pandas DataFrame containing columns: 'subject', 'relation', and 'object'.

    idx_entity : pd.DataFrame
        A Pandas DataFrame that contains the mapping between entity names and their corresponding indices.
        Must have columns: 'entity' and 'index'.

    idx_relation : pd.DataFrame
        A Pandas DataFrame that contains the mapping between relation names and their corresponding indices.
        Must have columns: 'relation' and 'index'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the 'subject', 'relation', and 'object' columns replaced by their corresponding indices.
    """
    assert isinstance(df_pandas, pd.DataFrame)
    assert isinstance(idx_entity, pd.DataFrame)
    assert isinstance(idx_relation, pd.DataFrame)

    # Create a dictionary that maps entities to their indices
    entity_to_index = pd.Series(idx_entity.index, index=idx_entity['entity']).to_dict()
    df_pandas['subject'] = df_pandas['subject'].map(entity_to_index)
    df_pandas['object'] = df_pandas['object'].map(entity_to_index)
    del entity_to_index
    relation_to_index = pd.Series(idx_relation.index, index=idx_relation['relation']).to_dict()
    df_pandas['relation'] = df_pandas['relation'].map(relation_to_index)
    del relation_to_index
    return df_pandas


def apply_reciprical_or_noise(add_reciprical: bool, eval_model: str, df: object = None, info: str = None):
    """ (1) Add reciprocal triples (2) Add noisy triples """
    # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
    if add_reciprical and eval_model:
        if df is not None:
            print(f'Adding reciprocal triples to {info}, e.g. KG:= (s, p, o) union (o, p_inverse, s)')
            return create_recipriocal_triples(df)
        else:
            return None
    else:
        return df


def timeit(func):
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'{func.__name__} took {total_time:.4f} seconds '
            f'| Current Memory Usage {psutil.Process(os.getpid()).memory_info().rss / 1000000: .5} in MB')
        return result

    return timeit_wrapper


@timeit
def read_with_polars(data_path, read_only_few: int = None, sample_triples_ratio: float = None, separator:str=None) -> polars.DataFrame:
    """ Load and Preprocess via Polars """
    assert separator is not None, "separator cannot be None"
    print(f'*** Reading {data_path} with Polars ***')
    # (1) Load the data.
    #try:
    if ".zst" in data_path:
        df= polars.read_csv(data_path,n_rows=None if read_only_few is None else read_only_few)
    else:
        df = polars.read_csv(data_path,
                             has_header=False,
                             low_memory=False,
                             n_rows=None if read_only_few is None else read_only_few,
                             columns=[0, 1, 2],
                             dtypes=[polars.String],
                             new_columns=['subject', 'relation', 'object'],
                             separator=separator)
    #except ValueError as err:
    #    raise ValueError(f"{err}\nYou may want to use a different separator.")
    # (2) Sample from (1).
    if sample_triples_ratio:
        print(f'Subsampling {sample_triples_ratio} of input data {df.shape}...')
        df = df.sample(frac=sample_triples_ratio)
        print(df.shape)
    # (3) Type heuristic prediction: If KG is an RDF KG, remove all triples where subject is not <?>.
    h = df.head().to_pandas()
    if sum(h["subject"].str.startswith('<')) + sum(h["relation"].str.startswith('<')) > 2:
        print('Removing triples with literal values...')
        df = df.filter(polars.col("object").str.starts_with('<'))
    return df


@timeit
def read_with_pandas(data_path, read_only_few: int = None, sample_triples_ratio: float = None,separator:str=None):
    assert separator is not None, "separator cannot be None"
    print(f'*** Reading {data_path} with Pandas ***')
    if data_path[-3:] in [".nt","ttl", 'txt', 'csv', 'zst']:
        print('Reading with pandas.read_csv with sep ** s+ ** ...')
        df = pd.read_csv(data_path,
                         sep=separator,#"\s+",
                         header=None,
                         nrows=None if read_only_few is None else read_only_few,
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


def read_from_disk(data_path: str, read_only_few: int = None,
                   sample_triples_ratio: float = None, backend:str=None,separator:str=None)\
        ->Tuple[polars.DataFrame,pd.DataFrame]:
    assert backend is not None, "backend cannot be None"
    assert separator is not None, f"separator cannot be None. Currently {separator}"
    # If path exits
    if glob.glob(data_path):
        # (1) Detect data format
        dformat = data_path[data_path.find(".") + 1:]
        if dformat in ["ttl", "owl", "turtle", "rdf/xml"] and backend != "rdflib":
            raise RuntimeError(
                f"Data with **{dformat}** format cannot be read via --backend pandas or polars. Use --backend rdflib")
        if backend == 'pandas':
            return read_with_pandas(data_path, read_only_few, sample_triples_ratio, separator)
        elif backend == 'polars':
            return read_with_polars(data_path, read_only_few, sample_triples_ratio, separator)
        elif backend == "rdflib":
            # Lazy import
            from rdflib import Graph
            assert dformat in ["ttl", "owl", "nt", "turtle", "rdf/xml", "n3", " n-triples"],\
                f"--backend {backend} and dataformat **{dformat}** is not matching. Use --backend pandas"
            return pd.DataFrame(data=[(str(s), str(p), str(o)) for s, p, o in Graph().parse(data_path)],
                                columns=['subject', 'relation', 'object'], dtype=str)
        else:
            raise RuntimeError(f'--backend {backend} and {data_path} is not matching')
    else:
        print(f'{data_path} could not found!')
        return None


def count_triples(endpoint: str) -> int:
    """Returns the total number of triples in the triple store."""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/sparql-results+json"
    }
    query = """
    SELECT (COUNT(*) AS ?count)
    WHERE { ?s ?p ?o . }
    """
    response = requests.post(endpoint, data={"query": query}, headers=headers)
    response.raise_for_status()
    count = int(response.json()["results"]["bindings"][0]["count"]["value"])
    return count


def fetch_worker(endpoint: str, offsets: list[int], chunk_size: int, output_dir: str, worker_id: int):
    """Worker process: fetch assigned chunks and save to disk with per-worker tqdm."""
    os.makedirs(output_dir, exist_ok=True)
    for offset in tqdm(offsets, desc=f"Read Triple Store. Worker {worker_id}", position=worker_id, leave=True):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/sparql-results+json"
        }
        query = f"""
        SELECT ?subject ?predicate ?object
        WHERE {{
            ?subject ?predicate ?object .
        }}
        LIMIT {chunk_size} OFFSET {offset}
        """
        response = requests.post(endpoint, data={"query": query}, headers=headers)
        if not response.ok:
            print(f"[Worker {worker_id}] Query failed at offset {offset}: {response.status_code}")
            continue

        bindings = response.json()["results"]["bindings"]
        if not bindings:
            continue

        triples = [[b["subject"]["value"], b["predicate"]["value"], b["object"]["value"]] for b in bindings]
        df = pd.DataFrame(triples, columns=["subject", "relation", "object"], dtype=str)

        filename = os.path.join(output_dir, f"chunk_{offset}.parquet")
        df.to_parquet(filename, index=False)


def read_from_triple_store_with_polars(endpoint: str, chunk_size: int = 500000, output_dir: str = "triples_parquet"):
    """Main function to read all triples in parallel, save as Parquet, and load into Polars dataframe."""
    if os.path.exists(output_dir):
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".parquet")]
        if files:
            print(f"\n*** Found parquet files in folder `{output_dir}`, will read from those. Otherwise, delete the folder.***\n")
            parquet_files = sorted(files)
            df_polars = pl.read_parquet(parquet_files)
            return df_polars
    
    total_triples = count_triples(endpoint)
    total_chunks = (total_triples + chunk_size - 1) // chunk_size
    print(f"Total triples: {total_triples}, total chunks: {total_chunks}")

    # Determine number of workers
    num_workers = max(1, cpu_count())
    print(f"Using {num_workers} worker processes.")

    # Generate all offsets
    all_offsets = [i * chunk_size for i in range(total_chunks)]

    # Assign chunks deterministically to each worker
    worker_offsets = [[] for _ in range(num_workers)]
    for i, offset in enumerate(all_offsets):
        worker_offsets[i % num_workers].append(offset)

    # Launch worker processes
    processes = []
    for worker_id in range(num_workers):
        p = Process(target=fetch_worker, args=(endpoint, worker_offsets[worker_id], chunk_size, output_dir, worker_id))
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Read all Parquet chunks into a single Polars dataframe
    parquet_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".parquet")])
    df_polars = pl.read_parquet(parquet_files)
    return df_polars

def read_from_triple_store_with_pandas(endpoint: str = None):
    """ Read triples from triple store into pandas dataframe """
    assert endpoint is not None
    assert isinstance(endpoint, str)
    headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/sparql-results+json"
        }
    query = f"""
    SELECT ?subject ?predicate ?object
    WHERE {{
        ?subject ?predicate ?object .
    }}
    """
    query = """SELECT ?subject ?predicate ?object WHERE {  ?subject ?predicate ?object}"""
    response = requests.post(endpoint, data={"query": query}, headers=headers)
    assert response.ok
    # Generator
    triples = ([triple['subject']['value'], triple['predicate']['value'], triple['object']['value']] for triple in
               response.json()['results']['bindings'])
    return pd.DataFrame(data=triples, index=None, columns=["subject", "relation", "object"], dtype=str)


def get_er_vocab(data, file_path: str = None):
    # head entity and relation
    er_vocab = defaultdict(list)
    for triple in data:
        h, r, t = triple
        er_vocab[(h, r)].append(t)
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


def create_constraints(triples, file_path: str = None):
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
        '[3 / 4] Converting integer and relation mappings '
        'from from pandas dataframe to dictionaries for an easy access...',
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
        raise AssertionError('Size of the training dataset must be greater than 0.')

    try:
        assert num_entities >= max(train_set[:, 0]) and num_entities >= max(train_set[:, 2])
    except AssertionError:
        raise AssertionError(
            f'Entity Indexing Error:\n'
            f'Max ID of a subject or object entity in train set:'
            f'{max(train_set[:, 0])} or {max(train_set[:, 2])} is greater than num_entities:{num_entities}')
    try:
        assert num_relations >= max(train_set[:, 1])
    except AssertionError:
        print(
            f'Relation Indexing Error:\n'
            f'Max ID of a relation in train set:{max(train_set[:, 1])} is greater than num_entities:{num_relations}')
    # 13. Sanity checking: data types
    assert isinstance(train_set[0], np.ndarray)
