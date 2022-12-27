from .util import *

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
        for i in glob.glob(self.kg.data_dir + '/*'):
            if 'train' in i:
                self.kg.train_set = read_from_disk(i, self.kg.read_only_few, self.kg.sample_triples_ratio,
                                                   backend=self.kg.backend)
            elif 'test' in i:
                self.kg.test_set = read_from_disk(i, backend=self.kg.backend)
            elif 'valid' in i:
                self.kg.valid_set = read_from_disk(i, backend=self.kg.backend)
            else:
                print(f'Unrecognized data {i}')

