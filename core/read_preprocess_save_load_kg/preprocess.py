from .util import *


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
        if self.kg.backend == 'polars':
            self.preprocess_with_polars()
        elif self.kg.backend in ['pandas', 'modin']:
            self.preprocess_with_pandas()
        else:
            raise KeyError(f'{self.kg.backend} not found')

        print('Data Type conversion...')
        self.kg.train_set = numpy_data_type_changer(self.kg.train_set,
                                                    num=max(self.kg.num_entities, self.kg.num_relations))
        if self.kg.valid_set is not None:
            self.kg.valid_set = numpy_data_type_changer(self.kg.valid_set,
                                                        num=max(self.kg.num_entities, self.kg.num_relations))

        if self.kg.test_set is not None:
            self.kg.test_set = numpy_data_type_changer(self.kg.test_set,
                                                       num=max(self.kg.num_entities, self.kg.num_relations))

    @timeit
    def preprocess_with_pandas(self) -> None:
        """
        Preprocess train, valid and test datasets stored in knowledge graph instance with pandas

        (1) Add recipriocal or noisy triples
        (2) Construct vocabulary
        (3) Index datasets

        Parameter
        ---------

        Returns
        -------
        None
        """

        assert self.kg.backend in ['pandas', 'modin']
        # (1)  Add recipriocal or noisy triples.
        self.apply_reciprical_or_noise()
        # (2) Construct integer indexing for entities and relations.
        self.sequential_vocabulary_construction()
        self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(self.kg.relation_to_idx)

        start_time = time.time()
        # (3) Index datasets
        self.kg.train_set = index_triples_with_pandas(self.kg.train_set,
                                                      self.kg.entity_to_idx,
                                                      self.kg.relation_to_idx)
        print(f'Done ! {time.time() - start_time:.3f} seconds\n')
        assert isinstance(self.kg.train_set, pd.core.frame.DataFrame)
        self.kg.train_set = self.kg.train_set.values
        self.kg.train_set = numpy_data_type_changer(self.kg.train_set,
                                                    num=max(self.kg.num_entities, self.kg.num_relations))
        dataset_sanity_checking(self.kg.train_set, self.kg.num_entities, self.kg.num_relations)
        print('Done !\n')
        if self.kg.valid_set is not None:
            self.kg.valid_set = index_triples_with_pandas(self.kg.valid_set, self.kg.entity_to_idx,
                                                          self.kg.relation_to_idx)
            self.kg.valid_set = self.kg.valid_set.values
            dataset_sanity_checking(self.kg.valid_set, self.kg.num_entities, self.kg.num_relations)
            self.kg.valid_set = numpy_data_type_changer(self.kg.valid_set,
                                                        num=max(self.kg.num_entities, self.kg.num_relations))

        if self.kg.test_set is not None:
            self.kg.test_set = index_triples_with_pandas(self.kg.test_set, self.kg.entity_to_idx,
                                                         self.kg.relation_to_idx)
            # To numpy
            self.kg.test_set = self.kg.test_set.values
            dataset_sanity_checking(self.kg.test_set, self.kg.num_entities, self.kg.num_relations)
            self.kg.test_set = numpy_data_type_changer(self.kg.test_set,
                                                       num=max(self.kg.num_entities, self.kg.num_relations))
            print('Done !\n')

    @timeit
    def preprocess_with_polars(self) -> None:
        print(f'*** Preprocessing Train Data:{self.kg.train_set.shape} with Polars ***')
        # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
        if self.kg.add_reciprical and self.kg.eval_model:
            def adding_reciprocal_triples():
                """ Add reciprocal triples """
                # (1.1) Add reciprocal triples into training set
                self.kg.train_set.extend(self.kg.train_set.select([
                    polars.col("object").alias('subject'),
                    polars.col("relation").apply(lambda x: x + '_inverse'),
                    polars.col("subject").alias('object')
                ]))
                if self.kg.valid_set is not None:
                    # (1.2) Add reciprocal triples into valid_set set.
                    self.kg.valid_set.extend(self.kg.valid_set.select([
                        polars.col("object").alias('subject'),
                        polars.col("relation").apply(lambda x: x + '_inverse'),
                        polars.col("subject").alias('object')
                    ]))
                if self.kg.test_set is not None:
                    # (1.2) Add reciprocal triples into test set.
                    self.kg.test_set.extend(self.kg.test_set.select([
                        polars.col("object").alias('subject'),
                        polars.col("relation").apply(lambda x: x + '_inverse'),
                        polars.col("subject").alias('object')
                    ]))

            print('Adding Reciprocal Triples...', end=' ')
            adding_reciprocal_triples()

        # (2) Type checking
        try:
            assert isinstance(self.kg.train_set, polars.DataFrame)
        except TypeError:
            raise TypeError(f"{type(kg.train_set)}")
        assert isinstance(self.kg.valid_set, polars.DataFrame) or self.kg.valid_set is None
        assert isinstance(self.kg.test_set, polars.DataFrame) or self.kg.test_set is None
        if self.kg.min_freq_for_vocab is not None:
            raise NotImplementedError('With using Polars')

        def concat_splits(train, val, test):
            x = [train]
            if val is not None:
                x.append(val)
            if test is not None:
                x.append(test)
            return polars.concat(x)

        print('Concat Splits...', end=' ')
        df_str_kg = concat_splits(self.kg.train_set, self.kg.valid_set, self.kg.test_set)

        @timeit
        def entity_index():
            """ Create a mapping from str representation of entities/nodes to integers"""
            # Entity Index: {'a':1, 'b':2} :
            return polars.concat((df_str_kg['subject'], df_str_kg['object'])).unique(maintain_order=True).rename(
                'entity')

        print('Entity Indexing...', end=' ')
        self.kg.entity_to_idx = entity_index()

        @timeit
        def relation_index():
            """ Create a mapping from str representation of relations/edges to integers"""
            # Relation Index: {'r1':1, 'r2:'2}
            return df_str_kg['relation'].unique(maintain_order=True)

        print('Relation Indexing...', end=' ')
        self.kg.relation_to_idx = relation_index()
        # On YAGO3-10     # 2.90427 and 0.00065 MB,
        # print(f'Est. size of entity_to_idx in Polars:{self.kg.entity_to_idx.estimated_size(unit="mb"):.5f} in MB')
        # print(f'Est. of relation_to_idx in Polars:{self.kg.relation_to_idx.estimated_size(unit="mb"):.5f} in MB')
        self.kg.entity_to_idx = dict(zip(self.kg.entity_to_idx.to_list(), list(range(len(self.kg.entity_to_idx)))))
        self.kg.relation_to_idx = dict(
            zip(self.kg.relation_to_idx.to_list(), list(range(len(self.kg.relation_to_idx)))))
        # On YAGO3-10, 5.24297 in MB and 0.00118 in MB
        # print(f'Estimated size of entity_to_idx in Python dict:{sys.getsizeof(self.kg.entity_to_idx) / 1000000 :.5f} in MB')
        # print(f'Estimated size of relation_to_idx in Python dict:{sys.getsizeof(self.kg.relation_to_idx) / 1000000 :.5f} in MB')
        self.kg.num_entities, self.kg.num_relations = len(self.kg.entity_to_idx), len(self.kg.relation_to_idx)

        def indexer(data):
            """ Apply str to int mapping on an input data"""
            # These column assignments are executed in parallel
            # with_colums allows you to create new columns for you analyses.
            # https://pola-rs.github.io/polars-book/user-guide/quickstart/quick-exploration-guide.html#with_columns
            return data.with_columns([polars.col("subject").apply(lambda x: self.kg.entity_to_idx[x]).alias("subject"),
                                      polars.col("relation").apply(lambda x: self.kg.relation_to_idx[x]).alias(
                                          "relation"),
                                      polars.col("object").apply(lambda x: self.kg.entity_to_idx[x]).alias("object")])

        @timeit
        def index_datasets():
            """ Map str stored in a polars Dataframe to int"""
            self.kg.train_set = self.kg.train_set.select(
                [polars.col("subject").apply(lambda x: self.kg.entity_to_idx[x]),
                 polars.col("relation").apply(lambda x: self.kg.relation_to_idx[x]),
                 polars.col("object").apply(lambda x: self.kg.entity_to_idx[x])]).to_numpy()

        @timeit
        def from_pandas_to_numpy():
            # Index pandas dataframe?
            print(f'Convering data to Pandas {self.kg.train_set.shape}...')
            self.kg.train_set = self.kg.train_set.to_pandas()
            # Index pandas dataframe?
            print(f'Indexing Training Data {self.kg.train_set.shape}...')
            self.kg.train_set = index_triples_with_pandas(self.kg.train_set, entity_to_idx=self.kg.entity_to_idx,
                                                          relation_to_idx=self.kg.relation_to_idx).to_numpy()

        print(f'Indexing Training Data {self.kg.train_set.shape}...')
        from_pandas_to_numpy()
        # index_datasets(df=self.kg.train_set)

        print(f'Estimated size of train_set in Numpy: {self.kg.train_set.nbytes / 1000000 :.5f} in MB')
        if self.kg.valid_set is not None:
            print(f'Indexing Val Data {self.kg.valid_set.shape}...')
            self.kg.valid_set = index_datasets(df=self.kg.valid_set).to_numpy()
            print(f'Estimated size of valid_set in Numpy: {self.kg.valid_set.nbytes / 1000000:.5f} in MB')
        if self.kg.test_set is not None:
            print(f'Indexing Test Data {self.kg.test_set.shape}...')
            self.kg.test_set = index_datasets(df=self.kg.test_set).to_numpy()
            print(f'Estimated size of test_set in Numpy: {self.kg.test_set.nbytes / 1000000:.5f} in MB')
        print(f'*** Preprocessing Train Data:{self.kg.train_set.shape} with Polars DONE ***')

    def sequential_vocabulary_construction(self) -> None:
        """
        (1) Read input data into memory
        (2) Remove triples with a condition
        (3) Serialize vocabularies in a pandas dataframe where
                    => the index is integer and
                    => a single column is string (e.g. URI)
        """
        try:
            assert isinstance(self.kg.train_set, pd.DataFrame)
        except AssertionError:
            print(type(self.kg.train_set))
            print('HEREE')
            exit(1)
        assert isinstance(self.kg.valid_set, pd.DataFrame) or self.kg.valid_set is None
        assert isinstance(self.kg.test_set, pd.DataFrame) or self.kg.test_set is None

        # (4) Remove triples from (1).
        self.remove_triples_from_train_with_condition()
        # Concatenate dataframes.
        print('\nConcatenating data to obtain index...')
        x = [self.kg.train_set]
        if self.kg.valid_set is not None:
            x.append(self.kg.valid_set)
        if self.kg.test_set is not None:
            x.append(self.kg.test_set)
        df_str_kg = pd.concat(x, ignore_index=True)
        del x
        print('Done !\n')

        print('Creating a mapping from entities to integer indexes...')
        # (5) Create a bijection mapping from entities of (2) to integer indexes.
        # ravel('K') => Return a contiguous flattened array.
        # ‘K’ means to read the elements in the order they occur in memory, except for reversing the data when strides are negative.
        ordered_list = pd.unique(df_str_kg[['subject', 'object']].values.ravel('K')).tolist()
        self.kg.entity_to_idx = {k: i for i, k in enumerate(ordered_list)}
        # 5. Create a bijection mapping  from relations to integer indexes.
        ordered_list = pd.unique(df_str_kg['relation'].values.ravel('K')).tolist()
        self.kg.relation_to_idx = {k: i for i, k in enumerate(ordered_list)}
        print('Done !\n')
        del ordered_list

    def remove_triples_from_train_with_condition(self):
        if self.kg.min_freq_for_vocab is not None:
            assert isinstance(self.kg.min_freq_for_vocab, int)
            assert self.kg.min_freq_for_vocab > 0
            print(
                f'[5 / 14] Dropping triples having infrequent entities or relations (>{self.kg.min_freq_for_vocab})...',
                end=' ')
            num_triples = self.kg.train_set.size
            print('Total num triples:', num_triples, end=' ')
            # Compute entity frequency: index is URI, val is number of occurrences.
            entity_frequency = pd.concat([self.kg.train_set['subject'], self.kg.train_set['object']]).value_counts()
            relation_frequency = self.kg.train_set['relation'].value_counts()

            # low_frequency_entities index and values are the same URIs: dask.dataframe.core.DataFrame
            low_frequency_entities = entity_frequency[
                entity_frequency <= self.kg.min_freq_for_vocab].index.values
            low_frequency_relation = relation_frequency[
                relation_frequency <= self.kg.min_freq_for_vocab].index.values
            # If triple contains subject that is in low_freq, set False do not select
            self.kg.train_set = self.kg.train_set[~self.kg.train_set['subject'].isin(low_frequency_entities)]
            # If triple contains object that is in low_freq, set False do not select
            self.kg.train_set = self.kg.train_set[~self.kg.train_set['object'].isin(low_frequency_entities)]
            # If triple contains relation that is in low_freq, set False do not select
            self.kg.train_set = self.kg.train_set[~self.kg.train_set['relation'].isin(low_frequency_relation)]
            # print('\t after dropping:', df_str_kg.size.compute(scheduler=scheduler_flag))
            print('\t after dropping:', self.kg.train_set.size)  # .compute(scheduler=scheduler_flag))
            del low_frequency_entities
            print('Done !\n')

    def apply_reciprical_or_noise(self) -> None:
        """ (1) Add reciprocal triples (2) Add noisy triples """
        # (1) Add reciprocal triples, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}
        if self.kg.add_reciprical and self.kg.eval_model:
            print(
                '[3.1 / 14] Add reciprocal triples to train, validation, and test sets, e.g. KG:= {(s,p,o)} union {(o,p_inverse,s)}',
            )
            self.kg.train_set = create_recipriocal_triples(self.kg.train_set)
            if self.kg.valid_set is not None:
                self.kg.valid_set = create_recipriocal_triples(self.kg.valid_set)
            if self.kg.test_set is not None:
                self.kg.test_set = create_recipriocal_triples(self.kg.test_set)
            print('Done !\n')

        # (2) Extend KG with triples where entities and relations are randomly sampled.
        if self.kg.add_noise_rate is not None:
            print(f'[4 / 14] Adding noisy triples...')
            self.kg.train_set = add_noisy_triples(self.kg.train_set, self.kg.add_noise_rate)
            print('Done!\n')
