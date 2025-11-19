import pandas as pd
import polars as pl
from .util import (timeit, pandas_dataframe_indexer, dataset_sanity_checking, 
                   get_er_vocab, get_re_vocab, get_ee_vocab, apply_reciprocal_or_noise, 
                   polars_dataframe_indexer)
from dicee.static_funcs import numpy_data_type_changer
import numpy as np
import concurrent
from typing import List, Tuple
from typing import Union


class PreprocessKG:
    """ Preprocess the data in memory """

    def __init__(self, kg):
        self.kg = kg

    def start(self) -> None:
        """
        Preprocess train, valid and test datasets stored in knowledge graph instance

        Parameter
        ---------

        Returns
        -------
        None
        """
        # Process
        if self.kg.byte_pair_encoding and self.kg.padding:
            self.preprocess_with_byte_pair_encoding_with_padding()
        elif self.kg.byte_pair_encoding:
            self.preprocess_with_byte_pair_encoding()
        elif self.kg.backend == "polars":
            self.preprocess_with_polars()
        elif self.kg.backend in ["pandas", "rdflib"]:
            self.preprocess_with_pandas()
        else:
            raise KeyError(f'{self.kg.backend} not found')

        if self.kg.eval_model:
            if self.kg.byte_pair_encoding:
                data = []
                data.extend(self.kg.raw_train_set.values.tolist())
                if self.kg.raw_valid_set is not None:
                    data.extend(self.kg.raw_valid_set.values.tolist())
                if self.kg.raw_test_set is not None:
                    data.extend(self.kg.raw_test_set.values.tolist())
            else:
                if isinstance(self.kg.valid_set, np.ndarray) and isinstance(self.kg.test_set, np.ndarray):
                    data = np.concatenate([self.kg.train_set, self.kg.valid_set, self.kg.test_set])
                else:
                    data = self.kg.train_set
            print('Submit er-vocab, re-vocab, and ee-vocab via  ProcessPoolExecutor...')
            # We need to benchmark the benefits of using futures  ?
            executor = concurrent.futures.ProcessPoolExecutor()
            self.kg.er_vocab = executor.submit(get_er_vocab, data, self.kg.path_for_serialization + '/er_vocab.p')
            self.kg.re_vocab = executor.submit(get_re_vocab, data, self.kg.path_for_serialization + '/re_vocab.p')
            self.kg.ee_vocab = executor.submit(get_ee_vocab, data, self.kg.path_for_serialization + '/ee_vocab.p')

        assert isinstance(self.kg.raw_train_set, (pd.DataFrame, pl.DataFrame))

        if self.kg.byte_pair_encoding and self.kg.padding:
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
                entity_to_idx = {shaped_bpe_ent: idx for idx, (str_ent, bpe_ent, shaped_bpe_ent) in
                                 enumerate(self.kg.ordered_bpe_entities)}
                er_tails = dict()
                # Iterate over bpe encoded triples to obtain a mapping from pair of bpe entity and relation to
                # indices of bpe entities
                bpe_entities = []
                for (h, r, t) in self.kg.train_set:
                    er_tails.setdefault((h, r), list()).append(entity_to_idx[t])
                    bpe_entities.append(t)
                # Generate a training data
                self.kg.train_set = []
                self.kg.train_target_indices = []
                for (shaped_bpe_h, shaped_bpe_r), list_of_indices_of_tails in er_tails.items():
                    self.kg.train_set.append((shaped_bpe_h, shaped_bpe_r))
                    # List of integers denoting the index of shaped_bpe_entities
                    self.kg.train_target_indices.append(list_of_indices_of_tails)
                self.kg.train_set = np.array(self.kg.train_set)
                self.kg.target_dim = len(self.kg.ordered_bpe_entities)
            elif self.kg.training_technique == "AllvsAll":
                entity_to_idx = {shaped_bpe_ent: idx for idx, (str_ent, bpe_ent, shaped_bpe_ent) in
                                 enumerate(self.kg.ordered_bpe_entities)}
                er_tails = dict()
                bpe_entities = []
                for (h, r, t) in self.kg.train_set:
                    er_tails.setdefault((h, r), list()).append(entity_to_idx[t])
                    bpe_entities.append(t)

                bpe_tokens = {shaped_bpe_token for (str_entity, bpe_entity, shaped_bpe_token) in
                              self.kg.ordered_bpe_entities + self.kg.ordered_bpe_relations}
                # Iterate over all
                for i in bpe_tokens:
                    for j in bpe_tokens:
                        if er_tails.get((i, j), None) is None:
                            er_tails[(i, j)] = list()

                # Generate a training data
                self.kg.train_set = []
                self.kg.train_target_indices = []
                for (shaped_bpe_h, shaped_bpe_r), list_of_indices_of_tails in er_tails.items():
                    self.kg.train_set.append((shaped_bpe_h, shaped_bpe_r))
                    # List of integers denoting the index of shaped_bpe_entities
                    self.kg.train_target_indices.append(list_of_indices_of_tails)
                self.kg.train_set = np.array(self.kg.train_set)
                self.kg.target_dim = len(self.kg.ordered_bpe_entities)
            else:

                raise NotImplementedError(
                    f" Scoring technique {self.self.kg.training_technique} with BPE not implemented")
            if self.kg.max_length_subword_tokens is None and self.kg.byte_pair_encoding:
                self.kg.max_length_subword_tokens = len(self.kg.train_set[0][0])
        elif self.kg.byte_pair_encoding:
            space_token = self.kg.enc.encode(" ")[0]
            end_token = self.kg.enc.encode(".")[0]
            triples = []
            for (h, r, t) in self.kg.train_set:
                x = [*h, space_token, *r, space_token, *t, end_token]
                triples.extend(x)
            self.kg.train_set = np.array(triples)
        else:
            # Clear raw data to free memory
            self.kg.raw_train_set = None
            self.kg.raw_valid_set = None
            self.kg.raw_test_set = None


    @staticmethod
    def __replace_values_df(df: pd.DataFrame = None, f=None) -> Union[
        None, List[Tuple[Tuple[int], Tuple[int], Tuple[int]]]]:
        """
        Map a n by 3 pandas dataframe containing n triples into a list of n tuples
        where each tuple contains three tuples corresdoing to sequence of sub-word list representing head entity
        relation, and tail entity respectivly.
        Parameters
        ----------
        df: pandas.Dataframe
        f: an encoder function.

        Returns
        -------

        """
        if df is None:
            return []
        else:
            bpe_triples = list(df.map(lambda x: tuple(f(x))).itertuples(index=False, name=None))
            assert isinstance(bpe_triples, list)
            assert isinstance(bpe_triples[0], tuple)
            assert len(bpe_triples[0]) == 3
            assert isinstance(bpe_triples[0][0], tuple)
            assert isinstance(bpe_triples[0][0][0], int)
            return bpe_triples

    def __finding_max_token(self, concat_of_train_val_test) -> int:
        max_length_subword_tokens = 0
        for i in concat_of_train_val_test:
            max_token_length_per_triple = max(len(i[0]), len(i[1]), len(i[2]))
            if max_token_length_per_triple > max_length_subword_tokens:
                max_length_subword_tokens = max_token_length_per_triple
        return max_length_subword_tokens

    def __padding_in_place(self, x, max_length_subword_tokens, bpe_subwords_to_shaped_bpe_entities,
                           bpe_subwords_to_shaped_bpe_relations):

        for i, (s, p, o) in enumerate(x):
            if len(s) < max_length_subword_tokens:
                s_encoded = s + tuple(self.kg.dummy_id for _ in range(max_length_subword_tokens - len(s)))
            else:
                s_encoded = s

            if len(p) < max_length_subword_tokens:
                p_encoded = p + tuple(self.kg.dummy_id for _ in range(max_length_subword_tokens - len(p)))
            else:
                p_encoded = p

            if len(o) < max_length_subword_tokens:
                o_encoded = o + tuple(self.kg.dummy_id for _ in range(max_length_subword_tokens - len(o)))
            else:
                o_encoded = o

            bpe_subwords_to_shaped_bpe_entities[o] = o_encoded
            bpe_subwords_to_shaped_bpe_entities[s] = s_encoded

            bpe_subwords_to_shaped_bpe_relations[p] = p_encoded
            x[i] = (s_encoded, p_encoded, o_encoded)
        return x

    def preprocess_with_byte_pair_encoding(self):
        assert isinstance(self.kg.raw_train_set, pd.DataFrame)
        assert self.kg.raw_train_set.columns.tolist() == ['subject', 'relation', 'object']
        
        # Add reciprocal or noisy triples
        self.kg.raw_train_set = apply_reciprocal_or_noise(add_reciprocal=self.kg.add_reciprocal,
                                                          eval_model=self.kg.eval_model,
                                                          df=self.kg.raw_train_set, info="Train")
        self.kg.raw_valid_set = apply_reciprocal_or_noise(add_reciprocal=self.kg.add_reciprocal,
                                                          eval_model=self.kg.eval_model,
                                                          df=self.kg.raw_valid_set, info="Validation")
        self.kg.raw_test_set = apply_reciprocal_or_noise(add_reciprocal=self.kg.add_reciprocal,
                                                         eval_model=self.kg.eval_model,
                                                         df=self.kg.raw_test_set, info="Test")
        
        # Transform DataFrames to list of tuples with BPE encoding
        self.kg.train_set = self.__replace_values_df(df=self.kg.raw_train_set, f=self.kg.enc.encode)
        self.kg.valid_set = self.__replace_values_df(df=self.kg.raw_valid_set, f=self.kg.enc.encode)
        self.kg.test_set = self.__replace_values_df(df=self.kg.raw_test_set, f=self.kg.enc.encode)

    @timeit
    def preprocess_with_byte_pair_encoding_with_padding(self) -> None:
        """Preprocess with byte pair encoding and add padding"""
        self.preprocess_with_byte_pair_encoding()

        self.kg.max_length_subword_tokens = self.__finding_max_token(
            self.kg.train_set + self.kg.valid_set + self.kg.test_set)

        # Store padded bpe entities and relations
        bpe_subwords_to_shaped_bpe_entities = dict()
        bpe_subwords_to_shaped_bpe_relations = dict()

        print("The longest sequence of sub-word units of entities and relations is ",
              self.kg.max_length_subword_tokens)
        # Padding
        self.kg.train_set = self.__padding_in_place(self.kg.train_set, self.kg.max_length_subword_tokens,
                                                    bpe_subwords_to_shaped_bpe_entities,
                                                    bpe_subwords_to_shaped_bpe_relations)
        if self.kg.valid_set is not None:
            self.kg.valid_set = self.__padding_in_place(self.kg.valid_set, self.kg.max_length_subword_tokens,
                                                        bpe_subwords_to_shaped_bpe_entities,
                                                        bpe_subwords_to_shaped_bpe_relations)
        if self.kg.test_set is not None:
            self.kg.test_set = self.__padding_in_place(self.kg.test_set, self.kg.max_length_subword_tokens,
                                                       bpe_subwords_to_shaped_bpe_entities,
                                                       bpe_subwords_to_shaped_bpe_relations)
        # Store str_entity, bpe_entity, padded_bpe_entity
        self.kg.ordered_bpe_entities = sorted([(self.kg.enc.decode(k), k, v) for k, v in
                                               bpe_subwords_to_shaped_bpe_entities.items()], key=lambda x: x[0])
        self.kg.ordered_bpe_relations = sorted([(self.kg.enc.decode(k), k, v) for k, v in
                                                bpe_subwords_to_shaped_bpe_relations.items()], key=lambda x: x[0])
        del bpe_subwords_to_shaped_bpe_entities
        del bpe_subwords_to_shaped_bpe_relations

    @timeit
    def preprocess_with_pandas(self) -> None:
        """Preprocess with pandas: add reciprocal triples, construct vocabulary, and index datasets"""
        # Add reciprocal or noisy triples
        self.kg.raw_train_set = apply_reciprocal_or_noise(add_reciprocal=self.kg.add_reciprocal,
                                                          eval_model=self.kg.eval_model,
                                                          df=self.kg.raw_train_set, info="Train")
        self.kg.raw_valid_set = apply_reciprocal_or_noise(add_reciprocal=self.kg.add_reciprocal,
                                                          eval_model=self.kg.eval_model,
                                                          df=self.kg.raw_valid_set, info="Validation")
        self.kg.raw_test_set = apply_reciprocal_or_noise(add_reciprocal=self.kg.add_reciprocal,
                                                         eval_model=self.kg.eval_model,
                                                         df=self.kg.raw_test_set, info="Test")

        # Construct vocabulary
        self.sequential_vocabulary_construction()
        self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(self.kg.relation_to_idx)
        
        max_idx = max(self.kg.num_entities, self.kg.num_relations)

        # Index and convert datasets
        def index_and_convert(df, name):
            print(f'Indexing {name} data with shape {df.shape}...')
            indexed = pandas_dataframe_indexer(df, self.kg.entity_to_idx, self.kg.relation_to_idx).values
            dataset_sanity_checking(indexed, self.kg.num_entities, self.kg.num_relations)
            return numpy_data_type_changer(indexed, num=max_idx)

        self.kg.train_set = index_and_convert(self.kg.raw_train_set, "train")
        
        if self.kg.raw_valid_set is not None:
            self.kg.valid_set = index_and_convert(self.kg.raw_valid_set, "valid")

        if self.kg.raw_test_set is not None:
            self.kg.test_set = index_and_convert(self.kg.raw_test_set, "test")

    @timeit
    def preprocess_with_polars(self) -> None:
        """Preprocess with polars: add reciprocal triples and create indexed datasets"""
        print(f'*** Preprocessing Train Data:{self.kg.raw_train_set.shape} with Polars ***')

        # Add reciprocal triples
        if self.kg.add_reciprocal and self.kg.eval_model:
            def add_reciprocal(df):
                if df is not None:
                    return df.extend(df.select([
                        pl.col("object").alias('subject'),
                        pl.col("relation") + '_inverse',
                        pl.col("subject").alias('object')
                    ]))
                return df

            print('Adding Reciprocal Triples...')
            self.kg.raw_train_set = add_reciprocal(self.kg.raw_train_set)
            self.kg.raw_valid_set = add_reciprocal(self.kg.raw_valid_set)
            self.kg.raw_test_set = add_reciprocal(self.kg.raw_test_set)

        # Type checking
        assert isinstance(self.kg.raw_train_set, pl.DataFrame)
        assert self.kg.raw_valid_set is None or isinstance(self.kg.raw_valid_set, pl.DataFrame)
        assert self.kg.raw_test_set is None or isinstance(self.kg.raw_test_set, pl.DataFrame)
        # Concatenate all splits for vocabulary construction
        print('Concat Splits...')
        splits = [self.kg.raw_train_set]
        if self.kg.raw_valid_set is not None:
            splits.append(self.kg.raw_valid_set)
        if self.kg.raw_test_set is not None:
            splits.append(self.kg.raw_test_set)
        df_str_kg = pl.concat(splits)
        
        # Build entity vocabulary
        print("Collecting entities...")
        subjects = df_str_kg.select(pl.col("subject").unique(maintain_order=True).alias("entity"))
        objects = df_str_kg.select(pl.col("object").unique(maintain_order=True).alias("entity"))
        self.kg.entity_to_idx = pl.concat([subjects, objects], how="vertical").unique(maintain_order=True)
        self.kg.entity_to_idx = self.kg.entity_to_idx.with_row_index("index").select(["index", "entity"])
        print(f"Unique entities: {len(self.kg.entity_to_idx)}")
        
        # Build relation vocabulary
        print('Relation Indexing...')
        self.kg.relation_to_idx = df_str_kg.select(pl.col("relation").unique(maintain_order=True))
        self.kg.relation_to_idx = self.kg.relation_to_idx.with_row_index("index").select(["index", "relation"])
        del df_str_kg
        # Index datasets
        print(f'Indexing Training Data {self.kg.raw_train_set.shape}...')
        self.kg.train_set = polars_dataframe_indexer(self.kg.raw_train_set, self.kg.entity_to_idx, 
                                                      self.kg.relation_to_idx).to_numpy()

        if self.kg.raw_valid_set is not None:
            print(f'Indexing Val Data {self.kg.raw_valid_set.shape}...')
            self.kg.valid_set = polars_dataframe_indexer(self.kg.raw_valid_set, self.kg.entity_to_idx, 
                                                          self.kg.relation_to_idx).to_numpy()

        if self.kg.raw_test_set is not None:
            print(f'Indexing Test Data {self.kg.raw_test_set.shape}...')
            self.kg.test_set = polars_dataframe_indexer(self.kg.raw_test_set, self.kg.entity_to_idx, 
                                                         self.kg.relation_to_idx).to_numpy()

        self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(self.kg.relation_to_idx)
        print(f'*** Preprocessing Train Data:{self.kg.train_set.shape} with Polars DONE ***')

    def sequential_vocabulary_construction(self) -> None:
        """
        (1) Read input data into memory
        (2) Remove triples with a condition
        (3) Serialize vocabularies in a pandas dataframe where
                    => the index is integer and
                    => a single column is string (e.g. URI)
        """
        assert isinstance(self.kg.raw_train_set, pd.DataFrame)
        assert self.kg.raw_valid_set is None or isinstance(self.kg.raw_valid_set, pd.DataFrame)
        assert self.kg.raw_test_set is None or isinstance(self.kg.raw_test_set, pd.DataFrame)

        # Concatenate all data splits
        print('Concatenating data to build vocabulary...')
        splits = [self.kg.raw_train_set]
        if self.kg.raw_valid_set is not None:
            splits.append(self.kg.raw_valid_set)
        if self.kg.raw_test_set is not None:
            splits.append(self.kg.raw_test_set)
        df_str_kg = pd.concat(splits, ignore_index=True)
        print('Creating a mapping from entities to integer indexes...')
        # (5) Create a bijection mapping from entities of (2) to integer indexes.
        # ravel('K') => Return a contiguous flattened array.
        # ‘K’ means to read the elements in the order they occur in memory,
        # except for reversing the data when strides are negative.
        # ordered_list = pd.unique(df_str_kg[['subject', 'object']].values.ravel('K')).tolist()
        # self.kg.entity_to_idx = {k: i for i, k in enumerate(ordered_list)}
        # Instead of dict, storing it in a pandas dataframe
        self.kg.entity_to_idx = pd.concat((df_str_kg['subject'],df_str_kg['object'])).to_frame("entity").drop_duplicates(keep="first",ignore_index=True)
        # 5. Create a bijection mapping  from relations to integer indexes.
        # ordered_list = pd.unique(df_str_kg['relation'].values.ravel('K')).tolist()
        # self.kg.relation_to_idx = {k: i for i, k in enumerate(ordered_list)}
        self.kg.relation_to_idx = df_str_kg['relation'].to_frame("relation").drop_duplicates(keep="first", ignore_index=True)

        # del ordered_list