from .util import read_from_disk, read_from_triple_store
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

        Data will be available at the train_set, test_set, valid_set attributes.

        Parameter
        ---------
        None

        Returns
        -------
        None
        """
        if self.kg.path_single_kg is not None:
            self.kg.raw_train_set = read_from_disk(self.kg.path_single_kg,
                                                   self.kg.read_only_few,
                                                   self.kg.sample_triples_ratio,
                                                   backend=self.kg.backend)
            if self.kg.add_noise_rate:
                self.add_noisy_triples_into_training()

            self.kg.raw_valid_set = None
            self.kg.raw_test_set = None
        elif self.kg.sparql_endpoint is not None:
            self.kg.raw_train_set = read_from_triple_store(endpoint=self.kg.sparql_endpoint)
            self.kg.raw_valid_set = None
            self.kg.raw_test_set = None
        elif self.kg.dataset_dir:
            for i in glob.glob(self.kg.dataset_dir + '/*'):
                if 'train' in i:
                    self.kg.raw_train_set = read_from_disk(i, self.kg.read_only_few, self.kg.sample_triples_ratio,
                                                           backend=self.kg.backend)
                    if self.kg.add_noise_rate:
                        self.add_noisy_triples_into_training()
                elif 'test' in i and self.kg.eval_model is not None:
                    self.kg.raw_test_set = read_from_disk(i, backend=self.kg.backend)
                elif 'valid' in i and self.kg.eval_model is not None:
                    self.kg.raw_valid_set = read_from_disk(i, backend=self.kg.backend)
                else:
                    print(f'Not processed data: {i}')
        else:
            raise RuntimeError(f"Invalid data:{self.kg.data_dir}\t{self.kg.sparql_endpoint}\t{self.kg.path_single_kg}")

    def add_noisy_triples_into_training(self):
        num_noisy_triples = int(len(self.kg.train_set) * self.kg.add_noise_rate)
        s = len(self.kg.train_set)
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
