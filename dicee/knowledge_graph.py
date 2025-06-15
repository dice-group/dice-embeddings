from typing import List
from .read_preprocess_save_load_kg import ReadFromDisk, PreprocessKG, LoadSaveToDisk
import sys
from .config import Namespace

class KG:
    """ Knowledge Graph """

    def __init__(self, configs: Namespace, dataset_dir: str = None,
                 byte_pair_encoding: bool = False,
                 padding: bool = False,
                 add_noise_rate: float = None,
                 sparql_endpoint: str = None,
                 path_single_kg: str = None,
                 path_for_deserialization: str = None,
                 add_reciprical: bool = None, eval_model: str = None,
                 read_only_few: int = None, sample_triples_ratio: float = None,
                 path_for_serialization: str = None,
                 entity_to_idx=None, relation_to_idx=None, backend=None, training_technique: str = None,
                 ):
        """
        :param dataset_dir: A path of a folder containing train.txt, valid.txt, test.text
        :param byte_pair_encoding: Apply Byte pair encoding.
        :param padding: Add empty string into byte-pair encoded subword units representing triples
        :param add_noise_rate: Noisy triples added into the training adataset by x % of its size.
        :param sparql_endpoint: An endpoint of a triple store
        :param path_single_kg: The path of a single file containing the input knowledge graph
        :param path_for_deserialization: A path of a folder containing previously parsed data
        :param num_core: Number of subprocesses used for data loading
        :param add_reciprical: A flag for applying reciprocal data augmentation technique
        :param eval_model: A flag indicating whether evaluation will be applied.
        If no eval, then entity relation mappings will be deleted to free memory.
        :param add_noise_rate: Add say 10% noise in the input data
        sample_triples_ratio
        :param training_technique
        """
        if not isinstance(configs, Namespace):
            raise TypeError("KG needs a Namespace")
        self.configs             = configs
        self.dataset_dir = dataset_dir
        self.byte_pair_encoding = byte_pair_encoding
        self.ordered_shaped_bpe_tokens = None
        self.sparql_endpoint = sparql_endpoint
        self.add_noise_rate = add_noise_rate
        self.num_entities = None
        self.num_relations = None
        self.path_single_kg = path_single_kg
        self.path_for_deserialization = path_for_deserialization
        self.add_reciprical = add_reciprical
        self.eval_model = eval_model
        #####################################################################
        self.use_custom_tokenizer = configs.use_custom_tokenizer
        self.tokenizer_path     = configs.tokenizer_path
        self.use_transformer     = configs.use_transformer
        #####################################################################        

        self.read_only_few = read_only_few
        self.sample_triples_ratio = sample_triples_ratio
        self.path_for_serialization = path_for_serialization
        # dicts of str to int
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.backend = 'pandas' if backend is None else backend
        self.training_technique = training_technique
        self.raw_train_set, self.raw_valid_set, self.raw_test_set = None, None, None
        self.train_set, self.valid_set, self.test_set = None, None, None
        self.idx_entity_to_bpe_shaped = dict()

        ####################################################################################################
        if configs.use_custom_tokenizer:
            from tokenizers import Tokenizer
            self.enc = Tokenizer.from_file("C:\\Users\\Harshit Purohit\\Byte\\myenv6\\Lib\\site-packages\\dicee\\Tokenizer\\Tokenizer_Path\\tokenizer.json")
            self.num_tokens = self.enc.get_vocab_size()
            if configs.tokenizer_path:
                from tokenizers import Tokenizer
                self.enc = Tokenizer.from_file(configs.tokenizer_path)
                self.num_tokens = self.enc.get_vocab_size()
        else:
            # WIP:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
            self.num_tokens = self.enc.n_vocab  # ~ 50
        ####################################################################################################


        self.num_bpe_entities = None
        self.padding = padding
        # TODO: Find a unique token later
        #####################################################
        if configs.use_custom_tokenizer:
            self.dummy_id = self.enc.encode(" ").ids[0]
            if configs.tokenizer_path:
                self.dummy_id = self.enc.encode(" ").ids[0]
        else:
            self.dummy_id = self.enc.encode(" ")[0]
        #####################################################
        self.max_length_subword_tokens = None
        self.train_set_target = None
        self.target_dim = None
        self.train_target_indices = None
        self.ordered_bpe_entities = None

        if self.path_for_deserialization is None:
            ReadFromDisk(kg=self).start()
            PreprocessKG(kg=self).start()
            LoadSaveToDisk(kg=self).save()

        else:
            LoadSaveToDisk(kg=self).load()

        assert len(self.train_set) > 0

        self._describe()

    def _describe(self) -> None:
        self.description_of_input = f'\n------------------- Description of Dataset {self.dataset_dir} -------------------'
        if self.byte_pair_encoding:
            self.description_of_input += f'\nNumber of tokens:{self.num_tokens}' \
                                         f'\nNumber of max sequence of sub-words: {self.max_length_subword_tokens}' \
                                         f'\nNumber of triples on train set:' \
                                         f'{len(self.train_set)}' \
                                         f'\nNumber of triples on valid set:' \
                                         f'{len(self.valid_set) if self.valid_set is not None else 0}' \
                                         f'\nNumber of triples on test set:' \
                                         f'{len(self.test_set) if self.test_set is not None else 0}\n'
        else:
            self.description_of_input += f'\nNumber of entities:{self.num_entities}' \
                                         f'\nNumber of relations:{self.num_relations}' \
                                         f'\nNumber of triples on train set:' \
                                         f'{len(self.train_set)}' \
                                         f'\nNumber of triples on valid set:' \
                                         f'{len(self.valid_set) if self.valid_set is not None else 0}' \
                                         f'\nNumber of triples on test set:' \
                                         f'{len(self.test_set) if self.test_set is not None else 0}\n'
            self.description_of_input += f"Entity Index:{sys.getsizeof(self.entity_to_idx) / 1_000_000_000:.5f} in GB\n"
            self.description_of_input += f"Relation Index:{sys.getsizeof(self.relation_to_idx) / 1_000_000_000:.5f} in GB\n"

    @property
    def entities_str(self) -> List:
        return list(self.entity_to_idx.keys())

    @property
    def relations_str(self) -> List:
        return list(self.relation_to_idx.keys())

    def func_triple_to_bpe_representation(self, triple: List[str]):
        result = []
        use_custom_tokenizer = self.use_custom_tokenizer
        tokenizer_path = self.tokenizer_path
        use_transformer = self.use_transformer

        for x in triple:
            ###################################################################################
            if use_custom_tokenizer:
                unshaped_bpe_repr = self.enc.encode(x).ids
                if tokenizer_path:
                    unshaped_bpe_repr = self.enc.encode(x).ids
            else:
                unshaped_bpe_repr = self.enc.encode(x)
            ###################################################################################
            if len(unshaped_bpe_repr) < self.max_length_subword_tokens:
                unshaped_bpe_repr.extend([self.dummy_id for _ in
                                          range(self.max_length_subword_tokens - len(unshaped_bpe_repr))])
            else:
                pass
            result.append(unshaped_bpe_repr)
        return result
