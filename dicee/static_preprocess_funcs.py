import functools
import numpy as np
from typing import Tuple
import time
from collections import defaultdict
from .sanity_checkers import sanity_checking_with_arguments

enable_log = False
def timeit(func):
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if enable_log:
            if args is not None:
                s_args = [type(i) for i in args]
            else:
                s_args = args
            if kwargs is not None:
                s_kwargs = {k: type(v) for k, v in kwargs.items()}
            else:
                s_kwargs = kwargs
            print(f'Function {func.__name__} with  Args:{s_args} | Kwargs:{s_kwargs} took {total_time:.4f} seconds')
        else:
            print(f'Took {total_time:.4f} seconds')

        return result

    return timeit_wrapper


def preprocesses_input_args(args):
    """ Sanity Checking in input arguments """
    # To update the default value of Trainer in pytorch-lightnings
    args.max_epochs = args.num_epochs
    args.min_epochs = args.num_epochs
    assert args.weight_decay >= 0.0
    args.learning_rate = args.lr
    args.deterministic = True

    assert args.init_param in ['xavier_normal', None]

    # No need to eval. Investigate runtime performance
    args.check_val_every_n_epoch = 10 ** 6  # ,i.e., no eval
    assert args.add_noise_rate is None or isinstance(args.add_noise_rate, float)
    args.logger = False
    try:
        assert args.eval_model in [None, 'None', 'train', 'val', 'test', 'train_val', 'train_test', 'val_test',
                                   'train_val_test']
    except AssertionError:
        raise AssertionError(f'Unexpected input for eval_model ***\t{args.eval_model}\t***')

    if args.eval_model == 'None':
        args.eval_model = None

    # reciprocal checking
    if args.scoring_technique in ["AllvsAll", "KvsSample", "KvsAll", "1vsAll"]:
        args.apply_reciprical_or_noise = True
    elif args.scoring_technique in ["NegSample", "Sentence"]:
        args.apply_reciprical_or_noise = False
    else:
        raise KeyError(f'Unexpected input for scoring_technique \t{args.scoring_technique}')

    if args.sample_triples_ratio is not None:
        assert 1.0 >= args.sample_triples_ratio >= 0.0
    assert args.backend in ["pandas", "polars", "rdflib"]
    sanity_checking_with_arguments(args)
    if args.model == 'Shallom':
        args.scoring_technique = 'KvsAll'
    if args.normalization == 'None':
        args.normalization = None
    assert args.normalization in [None, 'LayerNorm', 'BatchNorm1d']
    """
    if args.byte_pair_encoding:
        args.scoring_technique="BytePairEncodedTriplesNegSample"
        try:
            assert args.neg_ratio>0
        except AssertionError:
            raise AssertionError(f"--byte_pair_encoding {args.byte_pair_encoding} currently only works with "
                                 f"positive --neg_ratio {args.neg_ratio}")
    """
    return args


@timeit
def create_constraints(triples: np.ndarray) -> Tuple[dict, dict, dict, dict]:
    """
    (1) Extract domains and ranges of relations
    (2) Store a mapping from relations to entities that are outside of the domain and range.
    Create constraints entities based on the range of relations
    :param triples:
    :return:
    """
    assert isinstance(triples, np.ndarray)
    assert triples.shape[1] == 3

    # (1) Compute the range and domain of each relation
    domain_per_rel = dict()
    range_per_rel = dict()

    range_constraints_per_rel = dict()
    domain_constraints_per_rel = dict()
    set_of_entities = set()
    set_of_relations = set()
    print(f'Constructing domain and range information by iterating over {len(triples)} triples...', end='\t')
    for (e1, p, e2) in triples:
        # e1, p, e2 have numpy.int16 or else types.
        domain_per_rel.setdefault(p, set()).add(e1)
        range_per_rel.setdefault(p, set()).add(e2)
        set_of_entities.add(e1)
        set_of_relations.add(p)
        set_of_entities.add(e2)
    print(f'Creating constraints based on {len(set_of_relations)} relations and {len(set_of_entities)} entities...',
          end='\t')
    for rel in set_of_relations:
        range_constraints_per_rel[rel] = list(set_of_entities - range_per_rel[rel])
        domain_constraints_per_rel[rel] = list(set_of_entities - domain_per_rel[rel])
    return domain_constraints_per_rel, range_constraints_per_rel, domain_per_rel, range_per_rel


def get_er_vocab(data):
    # head entity and relation
    er_vocab = defaultdict(list)
    for triple in data:
        er_vocab[(triple[0], triple[1])].append(triple[2])
    return er_vocab


def get_re_vocab(data):
    # head entity and relation
    re_vocab = defaultdict(list)
    for triple in data:
        re_vocab[(triple[1], triple[2])].append(triple[0])
    return re_vocab


def get_ee_vocab(data):
    # head entity and relation
    ee_vocab = defaultdict(list)
    for triple in data:
        ee_vocab[(triple[0], triple[2])].append(triple[1])
    return ee_vocab


@timeit
def mapping_from_first_two_cols_to_third(train_set_idx):
    store = dict()
    for s_idx, p_idx, o_idx in train_set_idx:
        store.setdefault((s_idx, p_idx), list()).append(o_idx)
    return store
