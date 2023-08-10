from .util import read_from_disk,read_from_triple_store
import glob
import pandas as pd
import numpy as np

class ReadFromDisk:
    """Read the data from disk into memory"""

    def __init__(self, kg):
        self.kg = kg

    def start(self) -> None:
        """
        Read a knowledge graph from disk into memory

        Data will be available at the train_set, test_set, valid_set attributes of kg object.

        Parameter
        ---------


        Returns
        -------
        None
        """
        if self.kg.path_single_kg is not None:
            self.kg.train_set = read_from_disk(self.kg.path_single_kg,
                                               self.kg.read_only_few,
                                               self.kg.sample_triples_ratio,
                                               backend=self.kg.backend)
            self.kg.valid_set = None
            self.kg.test_set = None
        elif self.kg.sparql_endpoint:
            self.kg.train_set=read_from_triple_store(endpoint=self.kg.sparql_endpoint)
            self.kg.valid_set = None
            self.kg.test_set = None

        else:
            for i in glob.glob(self.kg.data_dir + '/*'):
                if 'train' in i:
                    self.kg.train_set = read_from_disk(i, self.kg.read_only_few, self.kg.sample_triples_ratio,
                                                       backend=self.kg.backend)
                    if self.kg.add_noise_rate:
                        self.add_noisy_triples()

                elif 'test' in i and self.kg.eval_model is not None:
                    self.kg.test_set = read_from_disk(i, backend=self.kg.backend)
                elif 'valid' in i and self.kg.eval_model is not None:
                    self.kg.valid_set = read_from_disk(i, backend=self.kg.backend)
                else:
                    print(f'Unrecognized data {i}')

    def add_noisy_triples(self):
        num_noisy_triples = int(len(self.kg.train_set) * self.kg.add_noise_rate)
        s = len(self.kg.train_set)
        # @TODO: Can we use polars here ?
        list_of_entities = pd.unique(self.kg.train_set[['subject', 'object']].values.ravel('K'))
        self.kg.train_set = pd.concat([self.kg.train_set,
                                    # Noisy triples
                                    pd.DataFrame(
                                        {'subject': np.random.choice(list_of_entities, num_noisy_triples),
                                         'relation': np.random.choice(
                                             pd.unique(self.kg.train_set[['relation']].values.ravel('K')),
                                             num_noisy_triples),
                                         'object': np.random.choice(list_of_entities, num_noisy_triples)}
                                    )
                                    ], ignore_index=True)

        assert s + num_noisy_triples == len(self.kg.train_set)

