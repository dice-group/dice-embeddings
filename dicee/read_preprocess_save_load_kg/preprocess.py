import pandas as pd
import polars as pl
from .util import timeit, index_triples_with_pandas, dataset_sanity_checking
from dicee.static_funcs import numpy_data_type_changer
from .util import (
    get_er_vocab,
    get_re_vocab,
    get_ee_vocab,
    create_constraints,
    apply_reciprical_or_noise,
)
import numpy as np
import concurrent
from typing import Callable, Dict, List, Tuple
from typing import Union


class PreprocessKG:
    """
    Preprocess the data in memory for a knowledge graph.

    This class handles the preprocessing of the knowledge graph data which includes
    reading the data, adding noise or reciprocal triples, constructing vocabularies,
    and indexing datasets based on the backend being used.

    Attributes
    ----------
    kg : object
        An instance representing the knowledge graph.

    Methods
    -------
    start() -> None
        Preprocess train, valid, and test datasets stored in the knowledge graph instance.

    preprocess_with_byte_pair_encoding() -> None
        Preprocess the datasets using byte-pair encoding.

    preprocess_with_pandas() -> None
        Preprocess the datasets using pandas.

    preprocess_with_polars() -> None
        Preprocess the datasets using polars.

    sequential_vocabulary_construction() -> None
        Construct integer indexing for entities and relations.
    """

    def __init__(self, kg):
        self.kg = kg

    def start(self) -> None:
        """
        Preprocess train, valid, and test datasets stored in the knowledge graph instance.

        This method applies the appropriate preprocessing technique based on the backend
        specified in the knowledge graph instance.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the specified backend is not supported.
        """
        # Process
        if self.kg.byte_pair_encoding:
            self.preprocess_with_byte_pair_encoding()
        elif self.kg.backend == "polars":
            self.preprocess_with_polars()
        elif self.kg.backend in ["pandas", "rdflib"]:
            self.preprocess_with_pandas()
        else:
            raise KeyError(f"{self.kg.backend} not found")

        if self.kg.eval_model:
            if self.kg.byte_pair_encoding:
                data = []
                data.extend(self.kg.raw_train_set.values.tolist())
                if self.kg.raw_valid_set is not None:
                    data.extend(self.kg.raw_valid_set.values.tolist())
                if self.kg.raw_test_set is not None:
                    data.extend(self.kg.raw_test_set.values.tolist())
            else:
                if isinstance(self.kg.valid_set, np.ndarray) and isinstance(
                    self.kg.test_set, np.ndarray
                ):
                    data = np.concatenate(
                        [self.kg.train_set, self.kg.valid_set, self.kg.test_set]
                    )
                else:
                    data = self.kg.train_set
            print("Submit er-vocab, re-vocab, and ee-vocab via  ProcessPoolExecutor...")
            # We need to benchmark the benefits of using futures  ?
            executor = concurrent.futures.ProcessPoolExecutor()
            self.kg.er_vocab = executor.submit(
                get_er_vocab, data, self.kg.path_for_serialization + "/er_vocab.p"
            )
            self.kg.re_vocab = executor.submit(
                get_re_vocab, data, self.kg.path_for_serialization + "/re_vocab.p"
            )
            self.kg.ee_vocab = executor.submit(
                get_ee_vocab, data, self.kg.path_for_serialization + "/ee_vocab.p"
            )

            self.kg.constraints = executor.submit(
                create_constraints,
                self.kg.train_set,
                self.kg.path_for_serialization + "/constraints.p",
            )
            self.kg.domain_constraints_per_rel, self.kg.range_constraints_per_rel = (
                None,
                None,
            )

        # string containing
        assert isinstance(self.kg.raw_train_set, pd.DataFrame) or isinstance(
            self.kg.raw_train_set, pl.DataFrame
        )

        print("Creating dataset...")
        if self.kg.byte_pair_encoding:
            assert isinstance(self.kg.train_set, list)
            assert isinstance(self.kg.train_set[0], tuple)
            assert isinstance(self.kg.train_set[0][0], tuple)
            assert isinstance(self.kg.train_set[0][1], tuple)
            assert isinstance(self.kg.train_set[0][2], tuple)
            if self.kg.training_technique == "NegSample":
                """No need to do anything"""
            elif self.kg.training_technique == "KvsAll":
                # Construct the training data: A single data point is a unique pair of
                # a sequence of sub-words representing an entity
                # a sequence of sub-words representing a relation
                # Mapping from a sequence of bpe entity to its unique integer index
                entity_to_idx = {
                    shaped_bpe_ent: idx
                    for idx, (str_ent, bpe_ent, shaped_bpe_ent) in enumerate(
                        self.kg.ordered_bpe_entities
                    )
                }
                er_tails = dict()
                # Iterate over bpe encoded triples to obtain a mapping from pair of bpe entity and relation to
                # indices of bpe entities
                bpe_entities = []
                for h, r, t in self.kg.train_set:
                    er_tails.setdefault((h, r), list()).append(entity_to_idx[t])
                    bpe_entities.append(t)
                # Generate a training data
                self.kg.train_set = []
                self.kg.train_target_indices = []
                for (
                    shaped_bpe_h,
                    shaped_bpe_r,
                ), list_of_indices_of_tails in er_tails.items():
                    self.kg.train_set.append((shaped_bpe_h, shaped_bpe_r))
                    # List of integers denoting the index of shaped_bpe_entities
                    self.kg.train_target_indices.append(list_of_indices_of_tails)
                self.kg.train_set = np.array(self.kg.train_set)
                self.kg.target_dim = len(self.kg.ordered_bpe_entities)
            elif self.kg.training_technique == "AllvsAll":
                entity_to_idx = {
                    shaped_bpe_ent: idx
                    for idx, (str_ent, bpe_ent, shaped_bpe_ent) in enumerate(
                        self.kg.ordered_bpe_entities
                    )
                }
                er_tails = dict()
                bpe_entities = []
                for h, r, t in self.kg.train_set:
                    er_tails.setdefault((h, r), list()).append(entity_to_idx[t])
                    bpe_entities.append(t)

                bpe_tokens = {
                    shaped_bpe_token
                    for (
                        str_entity,
                        bpe_entity,
                        shaped_bpe_token,
                    ) in self.kg.ordered_bpe_entities
                    + self.kg.ordered_bpe_relations
                }
                # Iterate over all
                for i in bpe_tokens:
                    for j in bpe_tokens:
                        if er_tails.get((i, j), None) is None:
                            er_tails[(i, j)] = list()

                # Generate a training data
                self.kg.train_set = []
                self.kg.train_target_indices = []
                for (
                    shaped_bpe_h,
                    shaped_bpe_r,
                ), list_of_indices_of_tails in er_tails.items():
                    self.kg.train_set.append((shaped_bpe_h, shaped_bpe_r))
                    # List of integers denoting the index of shaped_bpe_entities
                    self.kg.train_target_indices.append(list_of_indices_of_tails)
                self.kg.train_set = np.array(self.kg.train_set)
                self.kg.target_dim = len(self.kg.ordered_bpe_entities)
            else:
                raise NotImplementedError(
                    f" Scoring technique {self.self.kg.training_technique} with BPE not implemented"
                )
            if self.kg.max_length_subword_tokens is None and self.kg.byte_pair_encoding:
                self.kg.max_length_subword_tokens = len(self.kg.train_set[0][0])
        else:
            """No need to do anything. We create datasets for other models in the pyorch dataset construction"""
            # @TODO: Either we should move the all pytorch dataset construciton into here
            # Or we should move the byte pair encoding data into
            print("Finding suitable integer type for the index...")
            self.kg.train_set = numpy_data_type_changer(
                self.kg.train_set, num=max(self.kg.num_entities, self.kg.num_relations)
            )
            if self.kg.valid_set is not None:
                self.kg.valid_set = numpy_data_type_changer(
                    self.kg.valid_set,
                    num=max(self.kg.num_entities, self.kg.num_relations),
                )
            if self.kg.test_set is not None:
                self.kg.test_set = numpy_data_type_changer(
                    self.kg.test_set,
                    num=max(self.kg.num_entities, self.kg.num_relations),
                )

    @staticmethod
    def __replace_values_df(
        df: pd.DataFrame = None, f: Callable = None
    ) -> Union[None, List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]]:
        """
        Map a DataFrame containing triples to a list of tuples with encoded entities and relations.

        This method takes a DataFrame where each row represents a triple in the knowledge graph
        (subject, relation, object) and applies an encoding function to each element. The result
        is a list of triples where each triple is represented as a tuple of tuples, with each
        tuple containing the encoded representation of the subject, relation, and object.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing triples in the format (subject, relation, object).
        f : Callable
            An encoding function that is applied to each element of the triples.

        Returns
        -------
        Union[None, List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]]
            A list of triples, where each triple is represented as a tuple of tuples, each containing
            the encoded representation of the subject, relation, and object. Returns None if the input DataFrame is None.

        Notes
        -----
        The encoding function 'f' is expected to return a tuple of integers representing the
        encoded form of the input entity or relation. This method is typically used to convert
        string representations of entities and relations into a numerical format suitable for
        machine learning models.
        """
        if df is None:
            return []
        else:
            bpe_triples = list(
                df.map(lambda x: tuple(f(x))).itertuples(index=False, name=None)
            )
            assert isinstance(bpe_triples, list)
            assert isinstance(bpe_triples[0], tuple)
            assert len(bpe_triples[0]) == 3
            assert isinstance(bpe_triples[0][0], tuple)
            assert isinstance(bpe_triples[0][0][0], int)
            return bpe_triples

    def __finding_max_token(
        self, concat_of_train_val_test: List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]
    ) -> int:
        """
        Find the maximum length of subword tokens in the dataset.

        This method iterates over the concatenated training, validation, and testing datasets
        to determine the maximum length of subword tokens present in the triples. This information
        is crucial for padding sequences to a uniform length before feeding them into a model.

        Parameters
        ----------
        concat_of_train_val_test : List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]
            A list containing triples from the training, validation, and testing datasets, where each triple
            is represented as a tuple of tuples. Each inner tuple contains the encoded representation
            (as a sequence of integers) of the subject, relation, and object.

        Returns
        -------
        int
            The maximum length of subword tokens found in the triples of the concatenated datasets.

        Notes
        -----
        The method assumes that the input list consists of triples, each represented as a tuple of tuples.
        Each inner tuple is expected to be a sequence of integers representing the encoded form of
        the subject, relation, or object in a triple.
        """
        max_length_subword_tokens = 0
        for i in concat_of_train_val_test:
            max_token_length_per_triple = max(len(i[0]), len(i[1]), len(i[2]))
            if max_token_length_per_triple > max_length_subword_tokens:
                max_length_subword_tokens = max_token_length_per_triple
        return max_length_subword_tokens

    def __padding_in_place(
        self,
        x: List[Tuple[Tuple[int], Tuple[int], Tuple[int]]],
        max_length_subword_tokens: int,
        bpe_subwords_to_shaped_bpe_entities: Dict[Tuple[int], Tuple[int]],
        bpe_subwords_to_shaped_bpe_relations: Dict[Tuple[int], Tuple[int]],
    ) -> List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]:
        """
        Apply padding in place to the sequences of subword tokens to ensure uniform length.

        This method iterates over a list of triples, where each triple consists of tuples of integers
        representing the encoded subword tokens for the subject, relation, and object. It applies padding
        to these tuples to ensure that they all have the same length, which is specified by the
        'max_length_subword_tokens' parameter.

        Parameters
        ----------
        x : List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]
            A list of triples, where each triple is represented as a tuple of tuples. Each inner tuple
            contains the encoded representation (as a sequence of integers) of the subject, relation, and object.
        max_length_subword_tokens : int
            The maximum length to which the sequences of subword tokens should be padded.
        bpe_subwords_to_shaped_bpe_entities : Dict[Tuple[int], Tuple[int]]
            A dictionary mapping the original subword token sequences of entities to their padded versions.
        bpe_subwords_to_shaped_bpe_relations : Dict[Tuple[int], Tuple[int]]
            A dictionary mapping the original subword token sequences of relations to their padded versions.

        Returns
        -------
        List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]
            The list of triples with padded subword token sequences.

        Notes
        -----
        Padding is applied by appending the necessary number of dummy identifiers (typically 0) to each
        sequence of subword tokens until it reaches the specified maximum length. This method modifies the
        input list 'x' in place and also updates the dictionaries 'bpe_subwords_to_shaped_bpe_entities' and
        'bpe_subwords_to_shaped_bpe_relations' with the padded sequences.
        """

        for i, (s, p, o) in enumerate(x):
            if len(s) < max_length_subword_tokens:
                s_encoded = s + tuple(
                    self.kg.dummy_id for _ in range(max_length_subword_tokens - len(s))
                )
            else:
                s_encoded = s

            if len(p) < max_length_subword_tokens:
                p_encoded = p + tuple(
                    self.kg.dummy_id for _ in range(max_length_subword_tokens - len(p))
                )
            else:
                p_encoded = p

            if len(o) < max_length_subword_tokens:
                o_encoded = o + tuple(
                    self.kg.dummy_id for _ in range(max_length_subword_tokens - len(o))
                )
            else:
                o_encoded = o

            bpe_subwords_to_shaped_bpe_entities[o] = o_encoded
            bpe_subwords_to_shaped_bpe_entities[s] = s_encoded

            bpe_subwords_to_shaped_bpe_relations[p] = p_encoded
            x[i] = (s_encoded, p_encoded, o_encoded)
        return x

    @timeit
    def preprocess_with_byte_pair_encoding(self) -> None:
        """
        Preprocess the datasets using byte-pair encoding (BPE).

        This method applies byte-pair encoding to the raw training, validation, and test sets of
        the knowledge graph. It transforms string representations of entities and relations into
        sequences of subword tokens. The method also handles padding of these sequences and
        constructs the necessary mappings for entities and relations.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - Byte-pair encoding is used to handle the out-of-vocabulary problem in natural language
        processing by splitting words into more frequently occurring subword units.
        - This method modifies the knowledge graph instance in place by setting various attributes
        related to the byte-pair encoding such as padded sequences, mappings, and the maximum
        length of subword tokens.
        - The method assumes that the raw datasets are available as Pandas DataFrames within the
        knowledge graph instance.
        - If the 'add_reciprical' flag is set in the knowledge graph instance, reciprocal triples are
        added to the datasets.
        - After encoding and padding, the method also constructs mappings from the subword token
        sequences to their corresponding integer indices.
        """
        # n b
        assert isinstance(self.kg.raw_train_set, pd.DataFrame)
        assert self.kg.raw_train_set.columns.tolist() == [
            "subject",
            "relation",
            "object",
        ]
        # (1)  Add recipriocal or noisy triples into raw_train_set, raw_valid_set, raw_test_set
        self.kg.raw_train_set = apply_reciprical_or_noise(
            add_reciprical=self.kg.add_reciprical,
            eval_model=self.kg.eval_model,
            df=self.kg.raw_train_set,
            info="Train",
        )
        self.kg.raw_valid_set = apply_reciprical_or_noise(
            add_reciprical=self.kg.add_reciprical,
            eval_model=self.kg.eval_model,
            df=self.kg.raw_valid_set,
            info="Validation",
        )
        self.kg.raw_test_set = apply_reciprical_or_noise(
            add_reciprical=self.kg.add_reciprical,
            eval_model=self.kg.eval_model,
            df=self.kg.raw_test_set,
            info="Test",
        )

        # (2) Transformation from DataFrame to list of tuples.
        # self.kg.train_set: List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]
        # valid_set: Union[List, List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]]
        # test_set: Union[List, List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]]
        # self.kg.train_set[0] => (bpe_h,bpe_r, bpe_t)
        # bpe_* denotes a tuple of positive integer numbers
        self.kg.train_set = self.__replace_values_df(
            df=self.kg.raw_train_set, f=self.kg.enc.encode
        )
        self.kg.valid_set = self.__replace_values_df(
            df=self.kg.raw_valid_set, f=self.kg.enc.encode
        )
        self.kg.test_set = self.__replace_values_df(
            df=self.kg.raw_test_set, f=self.kg.enc.encode
        )

        self.kg.max_length_subword_tokens = self.__finding_max_token(
            self.kg.train_set + self.kg.valid_set + self.kg.test_set
        )

        # Store padded bpe entities and relations
        bpe_subwords_to_shaped_bpe_entities = dict()
        bpe_subwords_to_shaped_bpe_relations = dict()

        print(
            "The longest sequence of sub-word units of entities and relations is ",
            self.kg.max_length_subword_tokens,
        )
        # Padding
        self.kg.train_set = self.__padding_in_place(
            self.kg.train_set,
            self.kg.max_length_subword_tokens,
            bpe_subwords_to_shaped_bpe_entities,
            bpe_subwords_to_shaped_bpe_relations,
        )
        if self.kg.valid_set is not None:
            self.kg.valid_set = self.__padding_in_place(
                self.kg.valid_set,
                self.kg.max_length_subword_tokens,
                bpe_subwords_to_shaped_bpe_entities,
                bpe_subwords_to_shaped_bpe_relations,
            )
        if self.kg.test_set is not None:
            self.kg.test_set = self.__padding_in_place(
                self.kg.test_set,
                self.kg.max_length_subword_tokens,
                bpe_subwords_to_shaped_bpe_entities,
                bpe_subwords_to_shaped_bpe_relations,
            )
        # Store str_entity, bpe_entity, padded_bpe_entity
        self.kg.ordered_bpe_entities = sorted(
            [
                (self.kg.enc.decode(k), k, v)
                for k, v in bpe_subwords_to_shaped_bpe_entities.items()
            ],
            key=lambda x: x[0],
        )
        self.kg.ordered_bpe_relations = sorted(
            [
                (self.kg.enc.decode(k), k, v)
                for k, v in bpe_subwords_to_shaped_bpe_relations.items()
            ],
            key=lambda x: x[0],
        )
        del bpe_subwords_to_shaped_bpe_entities
        del bpe_subwords_to_shaped_bpe_relations

    @timeit
    def preprocess_with_pandas(self) -> None:
        """
        Preprocess train, valid, and test datasets stored in the knowledge graph instance using pandas.

        This method involves adding reciprocal or noisy triples, constructing vocabularies for entities and relations,
        and indexing the datasets. The preprocessing is performed using the pandas library, which facilitates the handling
        and transformation of the data.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method begins by optionally adding reciprocal or noisy triples to the raw training, validation, and test sets.
        - Sequential vocabulary construction is performed to create a bijection mapping of entities and relations to integer indices.
        - The datasets (train, valid, test) are then indexed based on these mappings.
        - The method modifies the knowledge graph instance in place by setting various attributes such as the indexed datasets,
        the number of entities, and the number of relations.
        - The method assumes that the raw datasets are available as pandas DataFrames within the knowledge graph instance.
        - This preprocessing is crucial for converting the raw string-based datasets into a numerical format suitable for
        training machine learning models.
        """
        # (1)  Add recipriocal or noisy triples.
        self.kg.raw_train_set = apply_reciprical_or_noise(
            add_reciprical=self.kg.add_reciprical,
            eval_model=self.kg.eval_model,
            df=self.kg.raw_train_set,
            info="Train",
        )
        self.kg.raw_valid_set = apply_reciprical_or_noise(
            add_reciprical=self.kg.add_reciprical,
            eval_model=self.kg.eval_model,
            df=self.kg.raw_valid_set,
            info="Validation",
        )
        self.kg.raw_test_set = apply_reciprical_or_noise(
            add_reciprical=self.kg.add_reciprical,
            eval_model=self.kg.eval_model,
            df=self.kg.raw_test_set,
            info="Test",
        )

        # (2) Construct integer indexing for entities and relations.
        self.sequential_vocabulary_construction()
        self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(
            self.kg.relation_to_idx
        )

        # (3) Index datasets
        self.kg.train_set = index_triples_with_pandas(
            self.kg.raw_train_set, self.kg.entity_to_idx, self.kg.relation_to_idx
        )
        assert isinstance(self.kg.train_set, pd.core.frame.DataFrame)
        self.kg.train_set = self.kg.train_set.values
        self.kg.train_set = numpy_data_type_changer(
            self.kg.train_set, num=max(self.kg.num_entities, self.kg.num_relations)
        )
        dataset_sanity_checking(
            self.kg.train_set, self.kg.num_entities, self.kg.num_relations
        )
        if self.kg.raw_valid_set is not None:
            self.kg.valid_set = index_triples_with_pandas(
                self.kg.raw_valid_set, self.kg.entity_to_idx, self.kg.relation_to_idx
            )
            self.kg.valid_set = self.kg.valid_set.values
            dataset_sanity_checking(
                self.kg.valid_set, self.kg.num_entities, self.kg.num_relations
            )
            self.kg.valid_set = numpy_data_type_changer(
                self.kg.valid_set, num=max(self.kg.num_entities, self.kg.num_relations)
            )

        if self.kg.raw_test_set is not None:
            self.kg.test_set = index_triples_with_pandas(
                self.kg.raw_test_set, self.kg.entity_to_idx, self.kg.relation_to_idx
            )
            # To numpy
            self.kg.test_set = self.kg.test_set.values
            dataset_sanity_checking(
                self.kg.test_set, self.kg.num_entities, self.kg.num_relations
            )
            self.kg.test_set = numpy_data_type_changer(
                self.kg.test_set, num=max(self.kg.num_entities, self.kg.num_relations)
            )

    @timeit
    def preprocess_with_polars(self) -> None:
        """
        Preprocess train, valid, and test datasets stored in the knowledge graph instance using Polars.

        This method involves preprocessing the datasets with the Polars library, which is designed for efficient data
        manipulation and indexing. The process includes adding reciprocal triples, indexing entities and relations,
        and transforming the datasets from string-based to integer-based formats.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method begins by adding reciprocal triples to the raw datasets if the 'add_reciprical' flag is set
        in the knowledge graph instance.
        - It then constructs a bijection mapping from entities and relations to integer indices, using the unique
        entities and relations found in the concatenated datasets.
        - The datasets (train, valid, test) are indexed based on these mappings and converted to NumPy arrays.
        - The method updates the knowledge graph instance by setting attributes such as the number of entities,
        the number of relations, and the indexed datasets.
        - Polars is used for its performance advantages in handling large datasets and its efficient data manipulation capabilities.
        - This preprocessing step is crucial for converting the raw string-based datasets into a numerical format suitable
        for training machine learning models.
        """
        print(
            f"*** Preprocessing Train Data:{self.kg.raw_train_set.shape} with Polars ***"
        )

        # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
        if self.kg.add_reciprical and self.kg.eval_model:

            def adding_reciprocal_triples():
                """Add reciprocal triples"""
                # (1.1) Add reciprocal triples into training set
                self.kg.raw_train_set.extend(
                    self.kg.raw_train_set.select(
                        [
                            pl.col("object").alias("subject"),
                            pl.col("relation").apply(lambda x: x + "_inverse"),
                            pl.col("subject").alias("object"),
                        ]
                    )
                )
                if self.kg.raw_valid_set is not None:
                    # (1.2) Add reciprocal triples into valid_set set.
                    self.kg.raw_valid_set.extend(
                        self.kg.raw_valid_set.select(
                            [
                                pl.col("object").alias("subject"),
                                pl.col("relation").apply(lambda x: x + "_inverse"),
                                pl.col("subject").alias("object"),
                            ]
                        )
                    )
                if self.kg.raw_test_set is not None:
                    # (1.2) Add reciprocal triples into test set.
                    self.kg.raw_test_set.extend(
                        self.kg.raw_test_set.select(
                            [
                                pl.col("object").alias("subject"),
                                pl.col("relation").apply(lambda x: x + "_inverse"),
                                pl.col("subject").alias("object"),
                            ]
                        )
                    )

            print("Adding Reciprocal Triples...")
            adding_reciprocal_triples()

        # (2) Type checking
        try:
            assert isinstance(self.kg.raw_train_set, pl.DataFrame)
        except TypeError:
            raise TypeError(f"{type(self.kg.raw_train_set)}")
        assert (
            isinstance(self.kg.raw_valid_set, pl.DataFrame)
            or self.kg.raw_valid_set is None
        )
        assert (
            isinstance(self.kg.raw_test_set, pl.DataFrame)
            or self.kg.raw_test_set is None
        )

        def concat_splits(train, val, test):
            x = [train]
            if val is not None:
                x.append(val)
            if test is not None:
                x.append(test)
            return pl.concat(x)

        print("Concat Splits...")
        df_str_kg = concat_splits(
            self.kg.raw_train_set, self.kg.raw_valid_set, self.kg.raw_test_set
        )

        print("Entity Indexing...")
        self.kg.entity_to_idx = (
            pl.concat((df_str_kg["subject"], df_str_kg["object"]))
            .unique(maintain_order=True)
            .rename("entity")
        )
        print("Relation Indexing...")
        self.kg.relation_to_idx = df_str_kg["relation"].unique(maintain_order=True)
        print("Creating index for entities...")
        self.kg.entity_to_idx = {
            ent: idx for idx, ent in enumerate(self.kg.entity_to_idx.to_list())
        }
        print("Creating index for relations...")
        self.kg.relation_to_idx = {
            rel: idx for idx, rel in enumerate(self.kg.relation_to_idx.to_list())
        }
        self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(
            self.kg.relation_to_idx
        )

        print(f"Indexing Training Data {self.kg.raw_train_set.shape}...")
        self.kg.train_set = self.kg.raw_train_set.with_columns(
            pl.col("subject").map_dict(self.kg.entity_to_idx).alias("subject"),
            pl.col("relation").map_dict(self.kg.relation_to_idx).alias("relation"),
            pl.col("object").map_dict(self.kg.entity_to_idx).alias("object"),
        ).to_numpy()
        if self.kg.raw_valid_set is not None:
            print(f"Indexing Val Data {self.kg.raw_valid_set.shape}...")
            self.kg.valid_set = self.kg.raw_valid_set.with_columns(
                pl.col("subject").map_dict(self.kg.entity_to_idx).alias("subject"),
                pl.col("relation").map_dict(self.kg.relation_to_idx).alias("relation"),
                pl.col("object").map_dict(self.kg.entity_to_idx).alias("object"),
            ).to_numpy()
        if self.kg.raw_test_set is not None:
            print(f"Indexing Test Data {self.kg.raw_test_set.shape}...")
            self.kg.test_set = self.kg.raw_test_set.with_columns(
                pl.col("subject").map_dict(self.kg.entity_to_idx).alias("subject"),
                pl.col("relation").map_dict(self.kg.relation_to_idx).alias("relation"),
                pl.col("object").map_dict(self.kg.entity_to_idx).alias("object"),
            ).to_numpy()
        print(
            f"*** Preprocessing Train Data:{self.kg.train_set.shape} with Polars DONE ***"
        )

    def sequential_vocabulary_construction(self) -> None:
        """
        Construct sequential vocabularies for entities and relations in the knowledge graph.

        This method processes the raw training, validation, and test sets to create sequential mappings (bijection)
        of entities and relations to integer indices. These mappings are essential for converting the string-based
        representations of entities and relations to numerical formats that can be processed by machine learning models.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method first concatenates the raw datasets and then creates unique lists of all entities and relations.
        - It then assigns a unique integer index to each entity and relation, creating two dictionaries:
        'entity_to_idx' and 'relation_to_idx'.
        - These dictionaries are used to index entities and relations in the knowledge graph.
        - The method updates the knowledge graph instance by setting attributes such as 'entity_to_idx',
        'relation_to_idx', 'num_entities', and 'num_relations'.
        - This method is a crucial preprocessing step for transforming knowledge graph data into a format suitable
        for training and evaluating machine learning models.
        - The method assumes that the raw datasets are available as Pandas DataFrames within the knowledge graph instance.
        """
        try:
            assert isinstance(self.kg.raw_train_set, pd.DataFrame)
        except AssertionError:
            raise AssertionError
            print(type(self.kg.raw_train_set))
            print("HEREE")
            exit(1)
        assert (
            isinstance(self.kg.raw_valid_set, pd.DataFrame)
            or self.kg.raw_valid_set is None
        )
        assert (
            isinstance(self.kg.raw_test_set, pd.DataFrame)
            or self.kg.raw_test_set is None
        )

        # (1) Read input data into memory
        # (2) Remove triples with a condition
        # (3) Serialize vocabularies in a pandas dataframe where
        #             => the index is integer and
        #             => a single column is string (e.g. URI)
        # (4) Remove triples from (1).
        self.remove_triples_from_train_with_condition()
        # Concatenate dataframes.
        print("Concatenating data to obtain index...")
        x = [self.kg.raw_train_set]
        if self.kg.raw_valid_set is not None:
            x.append(self.kg.raw_valid_set)
        if self.kg.raw_test_set is not None:
            x.append(self.kg.raw_test_set)
        df_str_kg = pd.concat(x, ignore_index=True)
        del x
        print("Creating a mapping from entities to integer indexes...")
        # (5) Create a bijection mapping from entities of (2) to integer indexes.
        # ravel('K') => Return a contiguous flattened array.
        # ‘K’ means to read the elements in the order they occur in memory,
        # except for reversing the data when strides are negative.
        ordered_list = pd.unique(
            df_str_kg[["subject", "object"]].values.ravel("K")
        ).tolist()
        self.kg.entity_to_idx = {k: i for i, k in enumerate(ordered_list)}
        # 5. Create a bijection mapping  from relations to integer indexes.
        ordered_list = pd.unique(df_str_kg["relation"].values.ravel("K")).tolist()
        self.kg.relation_to_idx = {k: i for i, k in enumerate(ordered_list)}
        del ordered_list

    def dept_remove_triples_from_train_with_condition(self):
        """
        Remove specific triples from the training set based on a predefined condition.

        This method filters out triples from the raw training dataset of the knowledge graph based on
        a condition, such as the frequency of entities or relations. This is often used to refine the
        training data, for instance, by removing infrequent entities or relations that may not be
        significant for the model's training.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method specifically targets the removal of triples that contain entities or relations
        occurring below a certain frequency threshold.
        - The frequency threshold is determined by the 'min_freq_for_vocab' attribute of the knowledge graph instance.
        - The method updates the knowledge graph instance by modifying the 'raw_train_set' attribute,
        which holds the raw training dataset.
        - This preprocessing step is crucial for ensuring the quality of the training data and can impact
        the performance and generalization ability of the resulting machine learning models.
        - The method assumes that the raw training dataset is available as a Pandas DataFrame within the
        knowledge graph instance.
        """
        if None:
            # self.kg.min_freq_for_vocab is not
            assert isinstance(self.kg.min_freq_for_vocab, int)
            assert self.kg.min_freq_for_vocab > 0
            print(
                f"[5 / 14] Dropping triples having infrequent entities or relations (>{self.kg.min_freq_for_vocab})...",
                end=" ",
            )
            num_triples = self.kg.raw_train_set.size
            print("Total num triples:", num_triples, end=" ")
            # Compute entity frequency: index is URI, val is number of occurrences.
            entity_frequency = pd.concat(
                [self.kg.raw_train_set["subject"], self.kg.raw_train_set["object"]]
            ).value_counts()
            relation_frequency = self.kg.raw_train_set["relation"].value_counts()

            # low_frequency_entities index and values are the same URIs: dask.dataframe.core.DataFrame
            low_frequency_entities = entity_frequency[
                entity_frequency <= self.kg.min_freq_for_vocab
            ].index.values
            low_frequency_relation = relation_frequency[
                relation_frequency <= self.kg.min_freq_for_vocab
            ].index.values
            # If triple contains subject that is in low_freq, set False do not select
            self.kg.raw_train_set = self.kg.raw_train_set[
                ~self.kg.raw_train_set["subject"].isin(low_frequency_entities)
            ]
            # If triple contains object that is in low_freq, set False do not select
            self.kg.raw_train_set = self.kg.raw_train_set[
                ~self.kg.raw_train_set["object"].isin(low_frequency_entities)
            ]
            # If triple contains relation that is in low_freq, set False do not select
            self.kg.raw_train_set = self.kg.raw_train_set[
                ~self.kg.raw_train_set["relation"].isin(low_frequency_relation)
            ]
            # print('\t after dropping:', df_str_kg.size.compute(scheduler=scheduler_flag))
            print(
                "\t after dropping:", self.kg.raw_train_set.size
            )  # .compute(scheduler=scheduler_flag))
            del low_frequency_entities
