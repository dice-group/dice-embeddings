from .util import read_from_disk
import glob
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
            self.kg.train_set = read_from_disk(self.kg.path_single_kg, self.kg.read_only_few, self.kg.sample_triples_ratio,
                                               backend=self.kg.backend)
            self.kg.valid_set = None
            self.kg.test_set=None
        else:
            for i in glob.glob(self.kg.data_dir + '/*'):
                if 'train' in i:
                    self.kg.train_set = read_from_disk(i, self.kg.read_only_few, self.kg.sample_triples_ratio,
                                                       backend=self.kg.backend)
                elif 'test' in i and self.kg.eval_model is not None:
                    self.kg.test_set = read_from_disk(i, backend=self.kg.backend)
                elif 'valid' in i and self.kg.eval_model is not None:
                    self.kg.valid_set = read_from_disk(i, backend=self.kg.backend)
                else:
                    print(f'Unrecognized data {i}')
