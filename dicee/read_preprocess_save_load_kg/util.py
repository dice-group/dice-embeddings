from collections import defaultdict
from typing import Any, Callable, Iterable, Tuple, Union
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


def apply_reciprical_or_noise(
    add_reciprical: bool, eval_model: str, df: pd.DataFrame = None, info: str = None
) -> Union[pd.DataFrame, None]:
    """
    Add reciprocal triples to the knowledge graph dataset.

    This function augments a dataset by adding reciprocal triples. For each triple (s, p, o) in the dataset,
    it adds a reciprocal triple (o, p_inverse, s). This augmentation is often used in knowledge graph embedding
    models to improve the learning of relation patterns.

    Parameters
    ----------
    add_reciprical : bool
        A flag indicating whether to add reciprocal triples.
    eval_model : str
        The name of the evaluation model being used, which determines whether the reciprocal triples are required.
    df : pd.DataFrame, optional
        A pandas DataFrame containing the original triples of the knowledge graph. Each row should represent
        a triple (subject, predicate, object).
    info : str, optional
        An informational string describing the dataset being processed (e.g., 'train', 'test').

    Returns
    -------
    Union[pd.DataFrame, None]
        The augmented dataset with reciprocal triples added if the conditions are met.
        Returns the original DataFrame if conditions are not met, or None if the input DataFrame is None.

    Notes
    -----
    - The function checks if both 'add_reciprical' and 'eval_model' are set to truthy values before proceeding
      with the addition of reciprocal triples.
    - If 'df' is None, the function returns None, indicating that no dataset was provided for processing.
    - The reciprocal triples are created using a custom function 'create_reciprocal_triples'.
    """
    # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
    if add_reciprical and eval_model:
        if df is not None:
            print(
                f"Adding reciprocal triples to {info}, e.g. KG:= (s, p, o) union (o, p_inverse, s)"
            )
            return create_reciprocal_triples(df)
        else:
            return None
    else:
        return df


def timeit(func) -> Callable:
    """
    A decorator to measure the execution time of a function.

    This decorator, when applied to a function, logs the time taken by the function to execute.
    It uses `time.perf_counter()` for precise time measurement. The decorator also reports the
    memory usage of the process at the time of the function's execution completion.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function with added execution time and memory usage logging.

    Notes
    -----
    - The decorator uses `functools.wraps` to preserve the metadata of the original function.
    - Time is measured using `time.perf_counter()`, which provides a higher resolution time measurement.
    - Memory usage is obtained using `psutil.Process(os.getpid()).memory_info().rss`, which gives the resident set size.
    - This decorator is useful for performance profiling and debugging.
    """

    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        """
        Wrapper function used by the timeit decorator to measure execution time.

        This function wraps around the original function to be decorated. It measures the
        execution time by noting the time before and after the function call. Additionally,
        it prints the execution time and the current memory usage of the process.

        Parameters
        ----------
        *args
            Variable length argument list for the decorated function.
        **kwargs
            Arbitrary keyword arguments for the decorated function.

        Returns
        -------
        Any
            The result of the decorated function.

        Notes
        -----
        - The execution time is calculated using `time.perf_counter()` for high precision.
        - Memory usage is reported using `psutil.Process(os.getpid()).memory_info().rss`,
        which indicates the amount of memory used by the process.
        - This wrapper is an internal component of the `timeit` decorator and is not meant to be
        used independently.
        - The function `func` referenced inside this wrapper is the function that is being decorated.
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f"{func.__name__} took {total_time:.4f} seconds "
            f"| Current Memory Usage {psutil.Process(os.getpid()).memory_info().rss / 1000000: .5} in MB"
        )
        return result

    return timeit_wrapper


@timeit
def read_with_polars(
    data_path: str, read_only_few: int = None, sample_triples_ratio: float = None
) -> polars.DataFrame:
    """
    Load and preprocess a dataset using Polars.

    This function reads a dataset from a specified file path using the Polars library. It can handle CSV, TXT,
    and Parquet file formats. The function also provides options to read only a subset of the data and to sample
    a fraction of the data. Additionally, it applies a heuristic to filter out triples with literal values in RDF
    knowledge graphs.

    Parameters
    ----------
    data_path : str
        The file path to the dataset. Supported formats include CSV, TXT, and Parquet.
    read_only_few : int, optional
        If specified, only this number of rows will be read from the dataset. Defaults to reading the entire dataset.
    sample_triples_ratio : float, optional
        If specified, a fraction of the dataset will be randomly sampled. For example, a value of 0.1 samples 10% of the data.

    Returns
    -------
    polars.DataFrame
        The loaded and optionally sampled dataset as a Polars DataFrame.

    Notes
    -----
    - The function determines the file format based on the file extension.
    - If 'sample_triples_ratio' is provided, the dataset is subsampled accordingly.
    - A heuristic is applied to remove triples where the subject or object does not start with '<',
      which is common in RDF knowledge graphs to indicate entities.
    - This function uses Polars for efficient data loading and manipulation, especially useful for large datasets.
    """
    print(f"*** Reading {data_path} with Polars ***")
    # (1) Load the data.
    if True:  # data_path[-3:] in [".tar.gz",'txt', 'csv']:
        print("Reading with polars.read_csv with sep **t** ...")
        # TODO: if byte_pair_encoding=True, we should not use "\s+" as seperator I guess
        df = polars.read_csv(
            data_path,
            has_header=False,
            low_memory=False,
            n_rows=None if read_only_few is None else read_only_few,
            columns=[0, 1, 2],
            dtypes=[polars.Utf8],  # str
            new_columns=["subject", "relation", "object"],
            separator="\t",
        )  # \s+ doesn't work for polars
    else:
        if read_only_few is None:
            df = polars.read_parquet(data_path, use_pyarrow=True)
        else:
            df = polars.read_parquet(data_path, n_rows=read_only_few)
    # (2) Sample from (1).
    if sample_triples_ratio:
        print(f"Subsampling {sample_triples_ratio} of input data {df.shape}...")
        df = df.sample(frac=sample_triples_ratio)
        print(df.shape)

    # (3) Type heuristic prediction: If KG is an RDF KG, remove all triples where subject is not <?>.
    h = df.head().to_pandas()
    if (
        sum(h["subject"].str.startswith("<")) + sum(h["relation"].str.startswith("<"))
        > 2
    ):
        print("Removing triples with literal values...")
        df = df.filter(polars.col("object").str.starts_with("<"))
    return df


@timeit
def read_with_pandas(
    data_path, read_only_few: int = None, sample_triples_ratio: float = None
):
    print(f"*** Reading {data_path} with Pandas ***")
    if data_path[-3:] in ["ttl", "txt", "csv", "zst"]:
        print("Reading with pandas.read_csv with sep ** s+ ** ...")
        # TODO: if byte_pair_encoding=True, we should not use "\s+" as seperator I guess
        df = pd.read_csv(
            data_path,
            sep="\s+",
            header=None,
            nrows=None if read_only_few is None else read_only_few,
            usecols=[0, 1, 2],
            names=["subject", "relation", "object"],
            dtype=str,
        )
    else:
        df = pd.read_parquet(data_path, engine="pyarrow")
        # (2)a Read only few if it is asked.
        if isinstance(read_only_few, int):
            if read_only_few > 0:
                print(f"Reading only few input data {read_only_few}...")
                df = df.head(read_only_few)
                print("Done !\n")
    # (3) Read only sample
    if sample_triples_ratio:
        print(f"Subsampling {sample_triples_ratio} of input data...")
        df = df.sample(frac=sample_triples_ratio)
        print("Done !\n")
    if (
        sum(df.head()["subject"].str.startswith("<"))
        + sum(df.head()["relation"].str.startswith("<"))
        > 2
    ):
        # (4) Drop Rows/triples with double or boolean: Example preprocessing
        # Drop of object does not start with **<**.
        # Specifying na to be False instead of NaN.
        print("Removing triples with literal values...")
        df = df[df["object"].str.startswith("<", na=False)]
        print("Done !\n")
    return df


def read_from_disk(
    data_path: str,
    read_only_few: int = None,
    sample_triples_ratio: float = None,
    backend: str = None,
) -> Union[pd.DataFrame, polars.DataFrame, None]:
    """
    Load and preprocess a dataset from disk using specified backend.

    This function reads a dataset from a specified file path, supporting different backends such as pandas, polars,
    and rdflib. It can handle various file formats including TTL, OWL, RDF/XML, and others. The function provides
    options to read only a subset of the data and to sample a fraction of the data.

    Parameters
    ----------
    data_path : str
        The file path to the dataset.
    read_only_few : int, optional
        If specified, only this number of rows will be read from the dataset. Defaults to reading the entire dataset.
    sample_triples_ratio : float, optional
        If specified, a fraction of the dataset will be randomly sampled.
    backend : str
        The backend to use for reading the dataset. Supported values are 'pandas', 'polars', and 'rdflib'.

    Returns
    -------
    Union[pd.DataFrame, polars.DataFrame, None]
        The loaded dataset as a DataFrame, depending on the specified backend. Returns None if the file is not found.

    Raises
    ------
    RuntimeError
        If the data format is not compatible with the specified backend, or if the backend is not recognized.
    AssertionError
        If the backend is not provided.

    Notes
    -----
    - The function automatically detects the data format based on the file extension.
    - For RDF/XML, TTL, OWL, and similar formats, the 'rdflib' backend is required.
    - This function is a general interface for loading datasets, allowing flexibility in choosing the backend
      based on data format and processing needs.
    """
    assert backend
    # If path exits
    if glob.glob(data_path):
        # (1) Detect data format
        dformat = data_path[data_path.find(".") + 1 :]
        if dformat in ["ttl", "owl", "turtle", "rdf/xml"] and backend != "rdflib":
            raise RuntimeError(
                f"Data with **{dformat}** format cannot be read via --backend pandas or polars. Use --backend rdflib"
            )

        if backend == "pandas":
            return read_with_pandas(data_path, read_only_few, sample_triples_ratio)
        elif backend == "polars":
            return read_with_polars(data_path, read_only_few, sample_triples_ratio)
        elif backend == "rdflib":
            # Lazy import
            from rdflib import Graph

            try:
                assert dformat in [
                    "ttl",
                    "owl",
                    "nt",
                    "turtle",
                    "rdf/xml",
                    "n3",
                    " n-triples",
                ]
            except AssertionError:
                raise AssertionError(
                    f"--backend {backend} and dataformat **{dformat}** is not matching. "
                    f"Use --backend pandas"
                )
            return pd.DataFrame(
                data=[(str(s), str(p), str(o)) for s, p, o in Graph().parse(data_path)],
                columns=["subject", "relation", "object"],
                dtype=str,
            )
        else:
            raise RuntimeError(f"--backend {backend} and {data_path} is not matching")
    else:
        print(f"{data_path} could not found!")
        return None


def read_from_triple_store(endpoint: str = None) -> pd.DataFrame:
    """
    Read triples from a SPARQL endpoint (triple store) and load them into a Pandas DataFrame.

    This function executes a SPARQL query against a specified SPARQL endpoint to retrieve all triples in the store.
    The result is then formatted into a Pandas DataFrame for further processing or analysis.

    Parameters
    ----------
    endpoint : str
        The URL of the SPARQL endpoint from which to retrieve triples.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the triples retrieved from the triple store, with columns 'subject', 'relation',
        and 'object'.

    Raises
    ------
    AssertionError
        If the 'endpoint' parameter is None or not a string, or if the response from the endpoint is not successful.

    Notes
    -----
    - The function sends a SPARQL query to the provided endpoint to retrieve all triples in the format
      {?subject ?predicate ?object}.
    - The response is expected in JSON format, conforming to the SPARQL query results JSON format.
    - This function is specifically designed for reading data from a SPARQL endpoint and requires an endpoint
      that responds to POST requests with SPARQL queries.
    """
    assert endpoint is not None
    assert isinstance(endpoint, str)
    query = (
        """SELECT ?subject ?predicate ?object WHERE {  ?subject ?predicate ?object}"""
    )
    response = requests.post(endpoint, data={"query": query})
    assert response.ok
    # Generator
    triples = (
        [
            triple["subject"]["value"],
            triple["predicate"]["value"],
            triple["object"]["value"],
        ]
        for triple in response.json()["results"]["bindings"]
    )
    return pd.DataFrame(
        data=triples, index=None, columns=["subject", "relation", "object"], dtype=str
    )


def get_er_vocab(
    data: Iterable[Tuple[Any, Any, Any]], file_path: str = None
) -> defaultdict:
    """
    Create a vocabulary mapping from (entity, relation) pairs to lists of tail entities.

    This function processes a dataset of triples and constructs a mapping where each key is a tuple of
    (head entity, relation) and the corresponding value is a list of all tail entities associated with
    that (head entity, relation) pair. Optionally, this vocabulary can be saved to a file.

    Parameters
    ----------
    data : Iterable[Tuple[Any, Any, Any]]
        An iterable of triples, where each triple is a tuple (head entity, relation, tail entity).
    file_path : str, optional
        The file path where the vocabulary should be saved as a pickle file. If not provided, the vocabulary
        is not saved to disk.

    Returns
    -------
    defaultdict
        A default dictionary where keys are (entity, relation) tuples and values are lists of tail entities.

    Notes
    -----
    - The function uses a `defaultdict` to handle keys that may not exist in the dictionary.
    - It is useful for creating a quick lookup of all possible tail entities for given (entity, relation) pairs,
      which can be used in various knowledge graph tasks like link prediction.
    - If 'file_path' is provided, the vocabulary is saved using the `save_pickle` function.
    """
    # head entity and relation
    er_vocab = defaultdict(list)
    for triple in data:
        h, r, t = triple
        er_vocab[(h, r)].append(t)
    if file_path:
        save_pickle(data=er_vocab, file_path=file_path)
    return er_vocab


def get_re_vocab(
    data: Iterable[Tuple[Any, Any, Any]], file_path: str = None
) -> defaultdict:
    """
    Create a vocabulary mapping from (relation, tail entity) pairs to lists of head entities.

    This function processes a dataset of triples and constructs a mapping where each key is a tuple of
    (relation, tail entity) and the corresponding value is a list of all head entities associated with
    that (relation, tail entity) pair. Optionally, this vocabulary can be saved to a file.

    Parameters
    ----------
    data : Iterable[Tuple[Any, Any, Any]]
        An iterable of triples, where each triple is a tuple (head entity, relation, tail entity).
    file_path : str, optional
        The file path where the vocabulary should be saved as a pickle file. If not provided, the vocabulary
        is not saved to disk.

    Returns
    -------
    defaultdict
        A default dictionary where keys are (relation, tail entity) tuples and values are lists of head entities.

    Notes
    -----
    - The function uses a `defaultdict` to handle keys that may not exist in the dictionary.
    - It is useful for creating a quick lookup of all possible head entities for given (relation, tail entity) pairs,
      which can be used in various knowledge graph tasks like link prediction.
    - If 'file_path' is provided, the vocabulary is saved using the `save_pickle` function.
    """
    # Function implementation...

    # head entity and relation
    re_vocab = defaultdict(list)
    for triple in data:
        re_vocab[(triple[1], triple[2])].append(triple[0])
    if file_path:
        save_pickle(data=re_vocab, file_path=file_path)
    return re_vocab


def get_ee_vocab(
    data: Iterable[Tuple[Any, Any, Any]], file_path: str = None
) -> defaultdict:
    """
    Create a vocabulary mapping from (head entity, tail entity) pairs to lists of relations.

    This function processes a dataset of triples and constructs a mapping where each key is a tuple of
    (head entity, tail entity) and the corresponding value is a list of all relations that connect these
    two entities. Optionally, this vocabulary can be saved to a file.

    Parameters
    ----------
    data : Iterable[Tuple[Any, Any, Any]]
        An iterable of triples, where each triple is a tuple (head entity, relation, tail entity).
    file_path : str, optional
        The file path where the vocabulary should be saved as a pickle file. If not provided, the vocabulary
        is not saved to disk.

    Returns
    -------
    defaultdict
        A default dictionary where keys are (head entity, tail entity) tuples and values are lists of relations.

    Notes
    -----
    - The function uses a `defaultdict` to handle keys that may not exist in the dictionary.
    - This vocabulary is useful for tasks that require knowledge of all relations between specific pairs of entities,
      such as in certain types of link prediction or relation extraction tasks.
    - If 'file_path' is provided, the vocabulary is saved using the `save_pickle` function.
    """
    # head entity and relation
    ee_vocab = defaultdict(list)
    for triple in data:
        ee_vocab[(triple[0], triple[2])].append(triple[1])
    if file_path:
        save_pickle(data=ee_vocab, file_path=file_path)
    return ee_vocab


def create_constraints(triples: np.ndarray, file_path: str = None) -> Tuple[dict, dict]:
    """
    Create domain and range constraints for each relation in a set of triples.

    This function processes a dataset of triples and constructs domain and range constraints for each relation.
    The domain of a relation is defined as the set of all head entities that appear with that relation, and the range
    is defined as the set of all tail entities. The constraints are formed by finding entities that are not in the
    domain or range of each relation.

    Parameters
    ----------
    triples : np.ndarray
        A numpy array of triples, where each row is a triple (head entity, relation, tail entity).
    file_path : str, optional
        The file path where the constraints should be saved as a pickle file. If not provided, the constraints
        are not saved to disk.

    Returns
    -------
    Tuple[dict, dict]
        A tuple containing two dictionaries. The first dictionary maps each relation to a list of entities
        not in its domain, and the second maps each relation to a list of entities not in its range.

    Notes
    -----
    - The function assumes that the input triples are in the form of a numpy array with three columns.
    - The domain and range constraints are useful in tasks that require understanding the valid head and tail
      entities for each relation, such as in link prediction.
    - If 'file_path' is provided, the constraints are saved using the `save_pickle` function.
    """
    assert isinstance(triples, np.ndarray)
    assert triples.shape[1] == 3

    # (1) Compute the range and domain of each relation
    range_constraints_per_rel = dict()
    domain_constraints_per_rel = dict()
    set_of_entities = set()
    set_of_relations = set()
    for e1, p, e2 in triples:
        range_constraints_per_rel.setdefault(p, set()).add(e2)
        domain_constraints_per_rel.setdefault(p, set()).add(e1)
        set_of_entities.add(e1)
        set_of_relations.add(p)
        set_of_entities.add(e2)

    for rel in set_of_relations:
        range_constraints_per_rel[rel] = list(
            set_of_entities - range_constraints_per_rel[rel]
        )
        domain_constraints_per_rel[rel] = list(
            set_of_entities - domain_constraints_per_rel[rel]
        )

    if file_path:
        save_pickle(
            data=(domain_constraints_per_rel, range_constraints_per_rel),
            file_path=file_path,
        )
    return domain_constraints_per_rel, range_constraints_per_rel


@timeit
def load_with_pandas(self) -> None:
    """
    Deserialize data and load it into the knowledge graph instance using Pandas.

    This method loads serialized data from disk, converting it into the appropriate data structures for
    use in the knowledge graph instance. It deserializes entity and relation mappings, training, validation,
    and test datasets, and constructs vocabularies and constraints necessary for the evaluation of the model.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - This method reads serialized data stored in Parquet format with gzip compression.
    - It deserializes mappings for entities and relations into dictionaries for efficient access.
    - Training, validation, and test sets are loaded into numpy arrays.
    - If evaluation is enabled, vocabularies for entity-relation, relation-entity, and entity-entity pairs
      are created along with domain and range constraints for relations.
    - This method handles the absence of validation or test sets gracefully, setting the corresponding
      attributes to None if the files are not found.
    - Deserialization paths and progress are logged, including time taken for each step.
    """
    print(f"Deserialization Path: {self.kg.deserialize_flag}\n")
    start_time = time.time()
    print("[1 / 4] Deserializing compressed entity integer mapping...")
    self.kg.entity_to_idx = pd.read_parquet(
        self.kg.deserialize_flag + "/entity_to_idx.gzip"
    )
    print(f"Done !\t{time.time() - start_time:.3f} seconds\n")
    self.kg.num_entities = len(self.kg.entity_to_idx)

    print("[2 / ] Deserializing compressed relation integer mapping...")
    start_time = time.time()
    self.kg.relation_to_idx = pd.read_parquet(
        self.kg.deserialize_flag + "/relation_to_idx.gzip"
    )
    print(f"Done !\t{time.time() - start_time:.3f} seconds\n")

    self.kg.num_relations = len(self.kg.relation_to_idx)
    print(
        "[3 / 4] Converting integer and relation mappings "
        "from from pandas dataframe to dictionaries for an easy access...",
    )
    start_time = time.time()
    self.kg.entity_to_idx = self.kg.entity_to_idx.to_dict()["entity"]
    self.kg.relation_to_idx = self.kg.relation_to_idx.to_dict()["relation"]
    print(f"Done !\t{time.time() - start_time:.3f} seconds\n")
    # 10. Serialize (9).
    print(
        "[4 / 4] Deserializing integer mapped data and mapping it to numpy ndarray..."
    )
    start_time = time.time()
    self.kg.train_set = pd.read_parquet(
        self.kg.deserialize_flag + "/idx_train_df.gzip"
    ).values
    print(f"Done !\t{time.time() - start_time:.3f} seconds\n")
    try:
        print(
            "[5 / 4] Deserializing integer mapped data and mapping it to numpy ndarray..."
        )
        self.kg.valid_set = pd.read_parquet(
            self.kg.deserialize_flag + "/idx_valid_df.gzip"
        ).values
        print("Done!\n")
    except FileNotFoundError:
        print("No valid data found!\n")
        self.kg.valid_set = None  # pd.DataFrame()

    try:
        print(
            "[6 / 4] Deserializing integer mapped data and mapping it to numpy ndarray..."
        )
        self.kg.test_set = pd.read_parquet(
            self.kg.deserialize_flag + "/idx_test_df.gzip"
        ).values
        print("Done!\n")
    except FileNotFoundError:
        print("No test data found\n")
        self.kg.test_set = None

    if self.kg.eval_model:
        if self.kg.valid_set is not None and self.kg.test_set is not None:
            # 16. Create a bijection mapping from subject-relation pairs to tail entities.
            data = np.concatenate(
                [self.kg.train_set, self.kg.valid_set, self.kg.test_set]
            )
        else:
            data = self.kg.train_set
        print("[7 / 4] Creating er,re, and ee type vocabulary for evaluation...")
        start_time = time.time()
        self.kg.er_vocab = get_er_vocab(data)
        self.kg.re_vocab = get_re_vocab(data)
        # 17. Create a bijection mapping from subject-object pairs to relations.
        self.kg.ee_vocab = get_ee_vocab(data)
        (
            self.kg.domain_constraints_per_rel,
            self.kg.range_constraints_per_rel,
        ) = create_constraints(self.kg.train_set)
        print(f"Done !\t{time.time() - start_time:.3f} seconds\n")


def save_numpy_ndarray(*, data: np.ndarray, file_path: str):
    """
    Save a numpy ndarray to disk.

    This function saves a given numpy ndarray to a specified file path using NumPy's binary format.
    The function is specifically designed to handle arrays with a shape (n, 3), typically representing
    triples in knowledge graphs.

    Parameters
    ----------
    data : np.ndarray
        A numpy ndarray to be saved, expected to have the shape (n, 3) where 'n' is the number of rows
        and 'd' is the number of columns (specifically 3).
    file_path : str
        The file path where the ndarray will be saved.

    Raises
    ------
    AssertionError
        If the number of rows 'n' in 'data' is not positive or the number of columns 'd' is not equal to 3.

    Notes
    -----
    - The ndarray is saved in NumPy's binary format (.npy file).
    - This function is particularly useful for saving datasets of triples in knowledge graph applications.
    - The file is opened in binary write mode and the data is saved using NumPy's `save` function.
    """
    n, d = data.shape
    assert n > 0
    assert d == 3
    with open(file_path, "wb") as f:
        np.save(f, data)


def load_numpy_ndarray(*, file_path: str) -> np.ndarray:
    """
    Load a numpy ndarray from a file.

    This function reads a numpy ndarray from a specified file path. The file is expected to be in
    NumPy's binary format (.npy file). It's commonly used to load datasets, especially in knowledge
    graph contexts.

    Parameters
    ----------
    file_path : str
        The path of the file from which the ndarray will be loaded.

    Returns
    -------
    np.ndarray
        The numpy ndarray loaded from the specified file.

    Notes
    -----
    - The function opens the file in binary read mode and loads the data using NumPy's `load` function.
    - This function is particularly useful for loading datasets of triples in knowledge graph applications
      or other numerical data saved in NumPy's binary format.
    - It's important to ensure that the file at 'file_path' exists and is a valid NumPy binary file to avoid
      runtime errors.
    """
    with open(file_path, "rb") as f:
        return np.load(f)


def save_pickle(*, data: object, file_path: str):
    """
    Serialize an object and save it to a file using pickle.

    This function serializes a given Python object using the pickle protocol and saves it to the specified file path.
    It's a general-purpose function that can be used to persist a wide range of Python objects.

    Parameters
    ----------
    data : object
        The Python object to be serialized and saved. This can be any object that is serializable by the pickle module.
    file_path : str
        The path of the file where the serialized object will be saved. The file will be created if it does not exist.
    """
    pickle.dump(data, open(file_path, "wb"))


def load_pickle(file_path: str) -> Any:
    """
    Load data from a pickle file.

    Parameters
    ----------
    file_path : str
        The file path to the pickle file to be loaded.

    Returns
    -------
    Any
        The data loaded from the pickle file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def create_reciprocal_triples(x: pd.DataFrame) -> pd.DataFrame:
    """
    Add inverse triples to a DataFrame of knowledge graph triples.

    Parameters
    ----------
    x : pd.DataFrame
        The DataFrame containing knowledge graph triples with columns "subject," "relation," and "object."

    Returns
    -------
    pd.DataFrame
        A new DataFrame that includes the original triples and their inverse counterparts.

    Notes
    -----
    This function takes a DataFrame of knowledge graph triples and adds their inverse triples to it.
    For each triple (s, r, o) in the input DataFrame, an inverse triple (o, r_inverse, s) is added to the output.
    The "relation" column of the inverse triples is created by appending "_inverse" to the original relation.

    """
    return pd.concat(
        [
            x,
            x["object"]
            .to_frame(name="subject")
            .join(x["relation"].map(lambda x: x + "_inverse").to_frame(name="relation"))
            .join(x["subject"].to_frame(name="object")),
        ],
        ignore_index=True,
    )


def index_triples_with_pandas(
    train_set: pd.DataFrame, entity_to_idx: dict, relation_to_idx: dict
) -> pd.DataFrame:
    """
    Index knowledge graph triples in a pandas DataFrame using provided entity and relation mappings.

    Parameters
    ----------
    train_set : pd.DataFrame
        A pandas DataFrame containing knowledge graph triples with columns "subject," "relation," and "object."

    entity_to_idx : dict
        A mapping from entity names (str) to integer indices.

    relation_to_idx : dict
        A mapping from relation names (str) to integer indices.

    Returns
    -------
    pd.DataFrame
        A new pandas DataFrame where the entities and relations in the original triples are replaced with their corresponding integer indices.

    Notes
    -----
    This function takes a pandas DataFrame of knowledge graph triples, along with mappings from entity and relation names to integer indices.
    It replaces the entity and relation names in the DataFrame with their corresponding integer indices, effectively indexing the triples.
    The resulting DataFrame has the same structure as the input, with integer indices replacing entity and relation names.
    """
    n, d = train_set.shape
    train_set["subject"] = train_set["subject"].apply(lambda x: entity_to_idx.get(x))
    train_set["relation"] = train_set["relation"].apply(
        lambda x: relation_to_idx.get(x)
    )
    train_set["object"] = train_set["object"].apply(lambda x: entity_to_idx.get(x))
    # train_set = train_set.dropna(inplace=True)
    if isinstance(train_set, pd.core.frame.DataFrame):
        assert (n, d) == train_set.shape
    else:
        raise KeyError("Wrong type training data")
    return train_set


def dataset_sanity_checking(
    train_set: np.ndarray, num_entities: int, num_relations: int
) -> None:
    """
    Perform sanity checks on a knowledge graph dataset.

    Parameters
    ----------
    train_set : np.ndarray
        The training dataset represented as a NumPy array. Each row represents a triple with columns "subject," "relation," and "object."

    num_entities : int
        The total number of entities in the knowledge graph.

    num_relations : int
        The total number of relations in the knowledge graph.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any of the sanity checks fail, assertions are raised to indicate potential issues in the dataset.

    Notes
    -----
    This function performs a series of sanity checks on a knowledge graph dataset to ensure its integrity and consistency.
    It checks the data type of the dataset, the number of columns, the size of the dataset, and the validity of entity and relation indices.
    If any of the checks fail, assertions are raised to signal potential problems in the dataset.

    The checks performed include:
    - Verifying that the input dataset is a NumPy array.
    - Checking that the dataset has the correct number of columns (3 for subject, relation, and object).
    - Ensuring that the dataset size is greater than 0.
    - Validating that the maximum entity indices in the dataset do not exceed the specified number of entities.
    - Validating that the maximum relation index in the dataset does not exceed the specified number of relations.
    """
    assert isinstance(train_set, np.ndarray)
    n, d = train_set.shape
    assert d == 3
    try:
        assert n > 0
    except AssertionError:
        raise AssertionError("Size of the training dataset must be greater than 0.")

    try:
        assert num_entities >= max(train_set[:, 0]) and num_entities >= max(
            train_set[:, 2]
        )
    except AssertionError:
        raise AssertionError(
            f"Entity Indexing Error:\n"
            f"Max ID of a subject or object entity in train set:"
            f"{max(train_set[:, 0])} or {max(train_set[:, 2])} is greater than num_entities:{num_entities}"
        )
    try:
        assert num_relations >= max(train_set[:, 1])
    except AssertionError:
        print(
            f"Relation Indexing Error:\n"
            f"Max ID of a relation in train set:{max(train_set[:, 1])} is greater than num_entities:{num_relations}"
        )
    # 13. Sanity checking: data types
    assert isinstance(train_set[0], np.ndarray)
