import os

import core
from core.typings import *
import numpy as np
import torch
import datetime
import logging
from collections import defaultdict
import pytorch_lightning as pl
import sys
from .helper_classes import CustomArg
from .models import *
import time
import pandas as pd
import json
import glob
import dask.dataframe as dd
from dask import dataframe as ddf
import dask
from .sanity_checkers import sanity_checking_with_arguments, config_kge_sanity_checking
import swifter
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin


# @TODO: Could these funcs can be merged?
def select_model(args: dict, is_continual_training: bool = None, storage_path: str = None):
    isinstance(args, dict)
    assert len(args) > 0
    assert isinstance(is_continual_training, bool)
    assert isinstance(storage_path, str)
    if is_continual_training:
        print('Loading pre-trained model...')
        model, _ = intialize_model(args)
        try:
            weights = torch.load(storage_path + '/model.pt', torch.device('cpu'))
            model.load_state_dict(weights)
            for parameter in model.parameters():
                parameter.requires_grad = True
            model.train()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{storage_path}/model.pt is not found. The model will be trained with random weights")
        return model, _
    else:
        return intialize_model(args)


def load_model(path_of_experiment_folder, model_name='model.pt') -> Tuple[BaseKGE, pd.DataFrame, pd.DataFrame]:
    """ Load weights and initialize pytorch module from namespace arguments"""
    print(f'Loading model {model_name}...', end=' ')
    start_time = time.time()
    # (1) Load weights..
    weights = torch.load(path_of_experiment_folder + f'/{model_name}', torch.device('cpu'))
    # (2) Loading input configuration..
    configs = load_json(path_of_experiment_folder + '/configuration.json')
    # (3) Loading the report of a training process.
    report = load_json(path_of_experiment_folder + '/report.json')
    configs["num_entities"] = report["num_entities"]
    configs["num_relations"] = report["num_relations"]
    print(f'Done! It took {time.time() - start_time:.3f}')
    # (4) Select the model
    model, _ = intialize_model(configs)
    # (5) Put (1) into (4)
    model.load_state_dict(weights)
    # (6) Set it into eval model.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    start_time = time.time()
    print('Loading entity and relation indexes...', end=' ')
    entity_to_idx = pd.read_parquet(path_of_experiment_folder + '/entity_to_idx.gzip')
    relation_to_idx = pd.read_parquet(path_of_experiment_folder + '/relation_to_idx.gzip')
    print(f'Done! It took {time.time() - start_time:.4f}')
    return model, entity_to_idx, relation_to_idx


def load_model_ensemble(path_of_experiment_folder: str) -> Tuple[BaseKGE, pd.DataFrame, pd.DataFrame]:
    """ Construct Ensemble Of weights and initialize pytorch module from namespace arguments

    (1) Detect models under given path
    (2) Accumulate parameters of detected models
    (3) Normalize parameters
    (4) Insert (3) into model.
    """
    print('Constructing Ensemble of ', end=' ')
    start_time = time.time()
    # (1) Detect models under given path.
    paths_for_loading = glob.glob(path_of_experiment_folder + '/model*')
    print(f'{len(paths_for_loading)} models...')
    assert len(paths_for_loading) > 0
    num_of_models = len(paths_for_loading)
    weights = None
    # (2) Accumulate parameters of detected models.
    while len(paths_for_loading):
        p = paths_for_loading.pop()
        print(f'Model: {p}...')
        if weights is None:
            weights = torch.load(p, torch.device('cpu'))
        else:
            five_weights = torch.load(p, torch.device('cpu'))
            # (2.1) Accumulate model parameters
            for k, _ in weights.items():
                if 'weight' in k:
                    weights[k] = (weights[k] + five_weights[k])
    # (3) Normalize parameters.
    for k, _ in weights.items():
        if 'weight' in k:
            weights[k] /= num_of_models
    # (4) Insert (3) into model
    # (4.1) Load report and configuration to initialize model.
    configs = load_json(path_of_experiment_folder + '/configuration.json')
    report = load_json(path_of_experiment_folder + '/report.json')
    configs["num_entities"] = report["num_entities"]
    configs["num_relations"] = report["num_relations"]
    print(f'Done! It took {time.time() - start_time:.2f} seconds.')
    # (4.2) Select the model
    model, _ = intialize_model(configs)
    # (4.3) Put (3) into their places
    model.load_state_dict(weights, strict=True)
    # (6) Set it into eval model.
    print('Setting Eval mode & requires_grad params to False')
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    start_time = time.time()
    print('Loading entity and relation indexes...', end=' ')
    entity_to_idx = pd.read_parquet(path_of_experiment_folder + '/entity_to_idx.gzip')
    relation_to_idx = pd.read_parquet(path_of_experiment_folder + '/relation_to_idx.gzip')
    print(f'Done! It took {time.time() - start_time:.4f}')
    return model, entity_to_idx, relation_to_idx


def numpy_data_type_changer(train_set: np.ndarray, num: int) -> np.ndarray:
    """
    Detect most efficient data type for a given triples
    :param train_set:
    :param num:
    :return:
    """
    assert isinstance(num, int)
    # train_set = train_set.astype(np.int32)
    if np.iinfo(np.int8).max > num:
        print(f'Setting int8,\t {np.iinfo(np.int8).max}')
        train_set = train_set.astype(np.int8)
    elif np.iinfo(np.int16).max > num:
        print(f'Setting int16,\t {np.iinfo(np.int16).max}')
        train_set = train_set.astype(np.int16)
    elif np.iinfo(np.int32).max > num:
        print(f'Setting int32,\t {np.iinfo(np.int32).max}')
        train_set = train_set.astype(np.int32)
    else:
        pass
    return train_set


def model_fitting(trainer, model, train_dataloaders) -> None:
    """ Standard Pytorch Lightning model fitting """
    assert trainer.max_epochs == trainer.min_epochs
    print(f'Number of epochs:{trainer.max_epochs}')
    print(f'Number of mini-batches to compute for a single epoch: {len(train_dataloaders)}')
    print(f'Learning rate:{model.learning_rate}\n')
    trainer.fit(model, train_dataloaders=train_dataloaders)


def initialize_pl_trainer(args, callbacks: List, plugins: List) -> pl.Trainer:
    """ Initialize pl.Traner from input arguments """
    if args.gpus:
        plugins.append(DDPPlugin(find_unused_parameters=False))
        plugins.append(DeepSpeedPlugin(stage=3))  # experiment with it when we use GPUs
        return pl.Trainer.from_argparse_args(args, plugins=plugins,
                                             callbacks=callbacks)
    else:
        return pl.Trainer.from_argparse_args(args, plugins=plugins,
                                             callbacks=callbacks)


def load_data_parallel(data_path, read_only_few: int = None,
                       sample_triples_ratio: float = None) -> dask.dataframe.core.DataFrame:
    """
    Parse KG via DASK.
    :param read_only_few:
    :param data_path:
    :param sample_triples_ratio:
    :return:
    """
    # (1) Check file exists, .e.g, ../../train.* exists
    if glob.glob(data_path + '*'):
        # (1) Read knowledge graph  via
        # (1.1) Using the whitespace as a deliminator
        # (1.2) Taking first three columns detected in (1.1.)
        #  Delayed Read operation
        df = ddf.read_csv(data_path + '*',
                          delim_whitespace=True,
                          header=None,
                          usecols=[0, 1, 2],
                          names=['subject', 'relation', 'object'],
                          dtype=str)
        # (2)a Read only few if it is asked.
        if isinstance(read_only_few, int):
            if read_only_few > 0:
                df = df.loc[:read_only_few]
        # (3) Read only sample
        if sample_triples_ratio:
            print(f'Subsampling {sample_triples_ratio} of input data...')
            df = df.sample(frac=sample_triples_ratio)

        # (4) Drop Rows/triples with double or boolean: Example preprocessing
        # if the first character of object is **"*, then drop it.
        # Drop rows having ^^
        # df = df[df["object"].str.contains('"') == False]
        df = df[df["object"].str.contains("<http://www.w3.org/2001/XMLSchema#double>") == False]
        df = df[df["object"].str.contains("<http://www.w3.org/2001/XMLSchema#boolean>") == False]
        df['subject'] = df['subject'].str.removeprefix("<").str.removesuffix(">")
        df['relation'] = df['relation'].str.removeprefix("<").str.removesuffix(">")
        df['object'] = df['object'].str.removeprefix("<").str.removesuffix(">")
        return df
    else:
        print(f'{data_path} could not found!')
        return None


def store_kge(trained_model, path: str) -> None:
    """
    Save parameters of model into path via torch
    :param trained_model: an instance of BaseKGE(pl.LightningModule) see core.models.base_model .
    :param path:
    :return:
    """
    try:
        torch.save(trained_model.state_dict(), path)
    except ReferenceError as e:
        print(e)
        print(trained_model.name)
        print('Could not save the model correctly')


def store(trained_model, model_name: str = 'model', full_storage_path: str = None,
          dataset=None, save_as_csv=False) -> None:
    """
    Store trained_model model and save embeddings into csv file.

    :param dataset: an instance of KG see core.knowledge_graph.
    :param full_storage_path: path to save parameters.
    :param model_name: string representation of the name of the model.
    :param trained_model: an instance of BaseKGE(pl.LightningModule) see core.models.base_model .
    :param save_as_csv: for easy access of embeddings.
    :return:
    """
    print('------------------- Store -------------------')
    assert full_storage_path is not None
    assert dataset is not None
    assert isinstance(model_name, str)
    assert len(model_name) > 1

    # (1) Save pytorch model in trained_model .
    store_kge(trained_model, path=full_storage_path + f'/{model_name}.pt')
    if save_as_csv:
        # (2.1) Get embeddings.
        entity_emb, relation_ebm = trained_model.get_embeddings()
        save_embeddings(entity_emb.numpy(), indexes=dataset.entities_str,
                        path=full_storage_path + '/' + trained_model.name + '_entity_embeddings.csv')
        del entity_emb
        if relation_ebm is not None:
            save_embeddings(relation_ebm.numpy(), indexes=dataset.relations_str,
                            path=full_storage_path + '/' + trained_model.name + '_relation_embeddings.csv')
            del relation_ebm
        else:
            pass
    else:
        # (2) See available memory and decide whether embeddings are stored separately or not.
        available_memory = [i.split() for i in os.popen('free -h').read().splitlines()][1][-1]  # ,e.g., 10Gi
        available_memory_mb = float(available_memory[:-2]) * 1000
        # Decision: model size in MB should be at most 1 percent of the available memory.
        if available_memory_mb * .01 > extract_model_summary(trained_model.summarize())['EstimatedSizeMB']:
            # (2.1) Get embeddings.
            entity_emb, relation_ebm = trained_model.get_embeddings()
            # (2.2) If we have less than 1000 rows total save it as csv.
            if len(entity_emb) < 1000:
                save_embeddings(entity_emb.numpy(), indexes=dataset.entities_str,
                                path=full_storage_path + '/' + trained_model.name + '_entity_embeddings.csv')
                del entity_emb
                if relation_ebm is not None:
                    save_embeddings(relation_ebm.numpy(), indexes=dataset.relations_str,
                                    path=full_storage_path + '/' + trained_model.name + '_relation_embeddings.csv')
                    del relation_ebm
            else:
                torch.save(entity_emb, full_storage_path + '/' + trained_model.name + '_entity_embeddings.pt')
                if relation_ebm is not None:
                    torch.save(relation_ebm, full_storage_path + '/' + trained_model.name + '_relation_embeddings.pt')
        else:
            print('There is not enough memory to store embeddings separately.')


def index_triples(train_set, entity_to_idx: dict, relation_to_idx: dict, num_core=False):
    """
    :param num_core:
    :param train_set: pandas dataframe or dask dataframe
    :param entity_to_idx:
    :param relation_to_idx:
    :return:
    """
    def entity_look_up(x):
        try:
            return entity_to_idx[x]
        except KeyError:
            return None

    def relation_look_up(x):
        try:
            return relation_to_idx[x]
        except KeyError:
            return None

    if num_core > 1:
        assert isinstance(train_set, pd.core.frame.DataFrame)
        train_set['subject'] = train_set['subject'].swifter.apply(lambda x: entity_look_up(x))
        train_set['relation'] = train_set['relation'].swifter.apply(lambda x: relation_look_up(x))
        train_set['object'] = train_set['object'].swifter.apply(lambda x: entity_look_up(x))
    else:
        train_set['subject'] = train_set['subject'].apply(lambda x: entity_look_up(x))
        train_set['relation'] = train_set['relation'].apply(lambda x: relation_look_up(x))
        train_set['object'] = train_set['object'].apply(lambda x: entity_look_up(x))

    train_set = train_set.dropna()
    return train_set


def add_noisy_triples(train_set, add_noise_rate: float) -> pd.DataFrame:
    """
    Add randomly constructed triples
    :param train_set:
    :param add_noise_rate:
    :return:
    """
    # Can not be applied on large
    train_set = train_set.compute()

    num_triples = len(train_set)
    num_noisy_triples = int(num_triples * add_noise_rate)
    print(f'[4 / 14] Generating {num_noisy_triples} noisy triples for training data...')

    list_of_entities = pd.unique(train_set[['subject', 'object']].values.ravel())

    train_set = pd.concat([train_set,
                           # Noisy triples
                           pd.DataFrame(
                               {'subject': np.random.choice(list_of_entities, num_noisy_triples),
                                'relation': np.random.choice(
                                    pd.unique(train_set[['relation']].values.ravel()),
                                    num_noisy_triples),
                                'object': np.random.choice(list_of_entities, num_noisy_triples)}
                           )
                           ], ignore_index=True)

    del list_of_entities

    assert num_triples + num_noisy_triples == len(train_set)
    return train_set


def create_recipriocal_triples_from_dask(x):
    """
    Add inverse triples into dask dataframe
    :param x:
    :return:
    """
    # x dask dataframe
    return dd.concat([x, x['object'].to_frame(name='subject').join(
        x['relation'].map(lambda x: x + '_inverse').to_frame(name='relation')).join(
        x['subject'].to_frame(name='object'))], ignore_index=True)


def read_preprocess_index_serialize_kg(args, cls):
    """ Read & Parse input data for training and testing"""
    print('*** Read, Parse, and Serialize Knowledge Graph  ***')
    start_time = time.time()
    # 1. Read & Parse input data
    kg = cls(data_dir=args.path_dataset_folder,
             num_core=args.num_core,
             dashboard=args.dashboard,
             add_reciprical=args.apply_reciprical_or_noise,
             eval_model=args.eval,
             read_only_few=args.read_only_few,
             sample_triples_ratio=args.sample_triples_ratio,
             path_for_serialization=args.full_storage_path,
             add_noise_rate=args.add_noise_rate,
             min_freq_for_vocab=args.min_freq_for_vocab
             )
    print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
    print(kg.description_of_input)
    return kg


def reload_input_data(storage_path: str = None, cls=None):
    assert isinstance(storage_path, str)
    assert len(storage_path)
    print('*** Reload Knowledge Graph  ***')
    start_time = time.time()
    kg = cls(deserialize_flag=storage_path)
    print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
    print(kg.description_of_input)
    return kg


def performance_debugger(func_name):
    def func_decorator(func):
        def debug(*args, **kwargs):
            starT = time.time()
            print('\n######', func_name, ' ', end='')
            r = func(*args, **kwargs)
            print(f' took  {time.time() - starT:.3f}  seconds')
            return r

        return debug

    return func_decorator


def preprocesses_input_args(arg):
    """ Sanity Checking in input arguments """
    # To update the default value of Trainer in pytorch-lightnings
    arg.max_epochs = arg.num_epochs
    arg.min_epochs = arg.num_epochs
    if arg.add_noise_rate is not None:
        assert 1. >= arg.add_noise_rate > 0.

    assert arg.weight_decay >= 0.0
    arg.learning_rate = arg.lr
    arg.deterministic = True
    assert arg.num_core >= 0

    # Below part will be investigated
    arg.check_val_every_n_epoch = 10 ** 6
    # del arg.check_val_every_n_epochs
    arg.checkpoint_callback = False
    arg.logger = False
    arg.eval = True if arg.eval == 1 else False
    arg.eval_on_train = True if arg.eval_on_train == 1 else False
    # reciprocal checking
    # @TODO We need better way for using apply_reciprical_or_noise.
    if arg.scoring_technique in ['PvsAll', 'CCvsAll', 'KvsAll', '1vsAll', 'BatchRelaxed1vsAll', 'BatchRelaxedKvsAll']:
        arg.apply_reciprical_or_noise = True
    elif arg.scoring_technique == 'NegSample':
        arg.apply_reciprical_or_noise = False
    else:
        raise KeyError(f'Unexpected input for scoring_technique.\t{arg.scoring_technique}')

    if arg.sample_triples_ratio is not None:
        assert 1.0 >= arg.sample_triples_ratio >= 0.0
    sanity_checking_with_arguments(arg)
    if arg.num_folds_for_cv > 0:
        arg.eval = True
    if arg.model == 'Shallom':
        arg.scoring_technique = 'KvsAll'
    assert arg.normalization in ['LayerNorm', 'BatchNorm1d']
    return arg


def create_constraints(triples: np.ndarray) -> Tuple[dict, dict]:
    """
    (1) Extract domains and ranges of relations
    (2) Store a mapping from relations to entities that are outside of the domain and range.
    Crete constrainted entities based on the range of relations
    :param triples:
    :return:
    """
    assert isinstance(triples, np.ndarray)
    assert triples.shape[1] == 3

    # (1) Compute the range and domain of each relation
    range_constraints_per_rel = dict()
    domain_constraints_per_rel = dict()
    set_of_entities = set()
    set_of_relations = set()
    for (e1, p, e2) in triples:
        range_constraints_per_rel.setdefault(p, set()).add(e2)
        domain_constraints_per_rel.setdefault(p, set()).add(e1)
        set_of_entities.add(e1)
        set_of_relations.add(p)
        set_of_entities.add(e2)

    for rel in set_of_relations:
        range_constraints_per_rel[rel] = list(set_of_entities - range_constraints_per_rel[rel])
        domain_constraints_per_rel[rel] = list(set_of_entities - domain_constraints_per_rel[rel])

    return domain_constraints_per_rel, range_constraints_per_rel


def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def create_experiment_folder(folder_name='Experiments'):
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder


def intialize_model(args: dict) -> Tuple[pl.LightningModule, AnyStr]:
    print('Initializing the selected model...', end=' ')
    start_time = time.time()
    model_name = args['model']
    if model_name == 'KronELinear':
        model = KronELinear(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KPDistMult':
        model = KPDistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KPFullDistMult':
        # Full compression of entities and relations.
        model = KPFullDistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KronE':
        model = KronE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'KronE_wo_f':
        model = KronE_wo_f(args=args)
        form_of_labelling = 'EntityPrediction'
    # elif model_name == 'BaseKronE':
    #    model = BaseKronE(args=args)
    #    form_of_labelling = 'EntityPrediction'
    elif model_name == 'Shallom':
        model = Shallom(args=args)
        form_of_labelling = 'RelationPrediction'
    elif model_name == 'ConEx':
        model = ConEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'QMult':
        model = QMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'OMult':
        model = OMult(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'ConvQ':
        model = ConvQ(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'ConvO':
        model = ConvO(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'ComplEx':
        model = ComplEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'DistMult':
        model = DistMult(args=args)
        form_of_labelling = 'EntityPrediction'
    else:
        raise ValueError
    print(f'Done! {time.time() - start_time:.3f}')
    return model, form_of_labelling


def extract_model_summary(s):
    return {'NumParam': s.total_parameters, 'EstimatedSizeMB': s.model_size}


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


def load_json(p: str) -> dict:
    assert os.path.isfile(p)
    with open(p, 'r') as r:
        args = json.load(r)
    return args


def save_embeddings(embeddings: np.ndarray, indexes, path: str) -> None:
    """
    Save it as CSV if memory allows.
    :param embeddings:
    :param indexes:
    :param path:
    :return:
    """
    try:
        df = pd.DataFrame(embeddings, index=indexes)
        del embeddings
        num_mb = df.memory_usage(index=True, deep=True).sum() / (10 ** 6)
        if num_mb > 10 ** 6:
            df = dd.from_pandas(df, npartitions=len(df) / 100)
            # PARQUET wants columns to be stn
            df.columns = df.columns.astype(str)
            df.to_parquet(path)
        else:
            df.to_csv(path)
    except KeyError or AttributeError as e:
        print('Exception occurred at saving entity embeddings. Computation will continue')
        print(e)
    del df


def random_prediction(pre_trained_kge):
    head_entity: List[str]
    relation: List[str]
    tail_entity: List[str]
    head_entity = pre_trained_kge.sample_entity(1)
    relation = pre_trained_kge.sample_relation(1)
    tail_entity = pre_trained_kge.sample_entity(1)
    triple_score = pre_trained_kge.predict_topk(head_entity=head_entity,
                                                relation=relation,
                                                tail_entity=tail_entity)
    return f'( {head_entity[0]},{relation[0]}, {tail_entity[0]} )', pd.DataFrame({'Score': triple_score})


def deploy_triple_prediction(pre_trained_kge, str_subject, str_predicate, str_object):
    triple_score = pre_trained_kge.predict_topk(head_entity=[str_subject],
                                                relation=[str_predicate],
                                                tail_entity=[str_object])
    return f'( {str_subject}, {str_predicate}, {str_object} )', pd.DataFrame({'Score': triple_score})


def deploy_tail_entity_prediction(pre_trained_kge, str_subject, str_predicate, top_k):
    if pre_trained_kge.model.name == 'Shallom':
        print('Tail entity prediction is not available for Shallom')
        raise NotImplementedError
    scores, entity = pre_trained_kge.predict_topk(head_entity=[str_subject], relation=[str_predicate], k=top_k)
    return f'(  {str_subject},  {str_predicate}, ? )', pd.DataFrame({'Entity': entity, 'Score': scores})


def deploy_head_entity_prediction(pre_trained_kge, str_object, str_predicate, top_k):
    if pre_trained_kge.model.name == 'Shallom':
        print('Head entity prediction is not available for Shallom')
        raise NotImplementedError

    scores, entity = pre_trained_kge.predict_topk(tail_entity=[str_object], relation=[str_predicate], k=top_k)
    return f'(  ?,  {str_predicate}, {str_object} )', pd.DataFrame({'Entity': entity, 'Score': scores})


def deploy_relation_prediction(pre_trained_kge, str_subject, str_object, top_k):
    scores, relations = pre_trained_kge.predict_topk(head_entity=[str_subject], tail_entity=[str_object], k=top_k)
    return f'(  {str_subject}, ?, {str_object} )', pd.DataFrame({'Relations': relations, 'Score': scores})


def semi_supervised_split(train_set: np.ndarray, train_split_ratio=None, calibration_split_ratio=None):
    """
    Split input triples into three splits
    1. split corresponds to the first 10% of the input
    2. split corresponds to the second 10% of the input
    3. split corresponds to the remaining data.
    """
    # Divide train_set into
    n, d = train_set.shape
    assert d == 3
    # (1) Select X % of the first triples for the training.
    train = train_set[: int(n * train_split_ratio)]
    # (2) Select remaining first Y % of the triples for the calibration.
    calibration = train_set[len(train):len(train) + int(n * calibration_split_ratio)]
    # (3) Consider remaining triples as unlabelled.
    unlabelled = train_set[-len(train) - len(calibration):]
    print(f'Shapes:\tTrain{train.shape}\tCalib:{calibration.shape}\tUnlabelled:{unlabelled.shape}')
    return train, calibration, unlabelled


def p_value(non_conf_scores, act_score):
    if len(act_score.shape) < 2:
        act_score = act_score.unsqueeze(-1)

    # return (torch.sum(non_conf_scores >= act_score) + 1) / (len(non_conf_scores) + 1)
    return (torch.sum(non_conf_scores >= act_score, dim=-1) + 1) / (len(non_conf_scores) + 1)


def norm_p_value(p_values, variant):
    if len(p_values.shape) < 2:
        p_values = p_values.unsqueeze(0)

    if variant == 0:
        norm_p_values = p_values / (torch.max(p_values, dim=-1).values.unsqueeze(-1))
    else:
        norm_p_values = p_values.scatter_(1, torch.max(p_values, dim=-1).indices.unsqueeze(-1),
                                          torch.ones_like(p_values))
    return norm_p_values


def is_in_credal_set(p_hat, pi):
    if len(p_hat.shape) == 1:
        p_hat = p_hat.unsqueeze(0)
    if len(pi.shape) == 1:
        pi = pi.unsqueeze(0)

    c = torch.cumsum(torch.flip(p_hat, dims=[-1]), dim=-1)
    rev_pi = torch.flip(pi, dims=[-1])
    return torch.all(c <= rev_pi, dim=-1)


def gen_lr(p_hat, pi):
    if len(p_hat.shape) < 2:
        p_hat = p_hat.unsqueeze(0)
    if len(pi.shape) < 2:
        pi = pi.unsqueeze(0)

    with torch.no_grad():
        # Sort values
        sorted_pi_rt = pi.sort(descending=True)

        sorted_pi = sorted_pi_rt.values
        sorted_p_hat = torch.gather(p_hat, 1, sorted_pi_rt.indices)

        def search_fn(sorted_p_hat, sorted_pi, sorted_pi_rt_ind):
            result_probs = torch.zeros_like(sorted_p_hat)

            for i in range(sorted_p_hat.shape[0]):
                # Search for loss
                proj = torch.zeros_like(sorted_p_hat[i])

                j = sorted_p_hat[i].shape[0] - 1
                while j >= 0:
                    lookahead = det_lookahead(sorted_p_hat[i], sorted_pi[i], j, proj)
                    proj[lookahead:j + 1] = sorted_p_hat[i][lookahead:j + 1] / torch.sum(
                        sorted_p_hat[i][lookahead:j + 1]) * (
                                                    sorted_pi[i][lookahead] - torch.sum(proj[j + 1:]))

                    j = lookahead - 1

                # e-arrange projection again according to original order
                proj = proj[sorted_pi_rt_ind[i].sort().indices]

                result_probs[i] = proj
            return result_probs

        is_c_set = is_in_credal_set(sorted_p_hat, sorted_pi)

        sorted_p_hat_non_c = sorted_p_hat[~is_c_set]
        sorted_pi_non_c = sorted_pi[~is_c_set]
        sorted_pi_ind_c = sorted_pi_rt.indices[~is_c_set]

        result_probs = torch.zeros_like(sorted_p_hat)
        result_probs[~is_c_set] = search_fn(sorted_p_hat_non_c, sorted_pi_non_c, sorted_pi_ind_c)
        result_probs[is_c_set] = p_hat[is_c_set]

    p_hat = torch.clip(p_hat, 1e-5, 1.)
    result_probs = torch.clip(result_probs, 1e-5, 1.)

    divergence = F.kl_div(p_hat.log(), result_probs, log_target=False, reduction="none")
    divergence = torch.sum(divergence, dim=-1)

    result = torch.where(is_c_set, torch.zeros_like(divergence), divergence)

    return torch.mean(result)


def det_lookahead(p_hat, pi, ref_idx, proj, precision=1e-5):
    for i in range(ref_idx):
        prop = p_hat[i:ref_idx + 1] / torch.sum(p_hat[i:ref_idx + 1])
        prop *= (pi[i] - torch.sum(proj[ref_idx + 1:]))

        # Check violation
        violates = False
        # TODO: Make this more efficient by using cumsum
        for j in range(len(prop)):
            if (torch.sum(prop[j:]) + torch.sum(proj[ref_idx + 1:])) > (torch.max(pi[i + j:]) + precision):
                violates = True
                break

        if not violates:
            return i

    return ref_idx


def construct_p_values(non_conf_scores, preds, non_conf_score_fn):
    num_class = preds.shape[1]
    tmp_non_conf = torch.zeros([preds.shape[0], num_class]).detach()
    p_values = torch.zeros([preds.shape[0], num_class]).detach()
    for clz in range(num_class):
        tmp_non_conf[:, clz] = non_conf_score_fn(preds, torch.tensor(clz).repeat(preds.shape[0]))
        p_values[:, clz] = p_value(non_conf_scores, tmp_non_conf[:, clz])
    return p_values


def non_conformity_score_prop(predictions, targets) -> torch.Tensor:
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)

    class_val = torch.gather(predictions, 1, targets.type(torch.int64))
    num_class = predictions.shape[1]

    # Exclude the target class here
    indices = torch.arange(0, num_class).view(1, -1).repeat(predictions.shape[0], 1)
    mask = torch.zeros_like(indices).bool()
    mask.scatter_(1, targets.type(torch.int64), True)

    selected_predictions = predictions[~mask].view(-1, args.num_classes - 1)

    return torch.max(selected_predictions, dim=-1).values.squeeze() / (
            class_val.squeeze() + args.non_conf_score_prop_gamma + 1e-5)


def non_conformity_score_diff(predictions, targets) -> torch.Tensor:
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)
    num_class = predictions.shape[1]
    class_val = torch.gather(predictions, 1, targets.type(torch.int64))

    # Exclude the target class here
    indices = torch.arange(0, num_class).view(1, -1).repeat(predictions.shape[0], 1)
    mask = torch.zeros_like(indices).bool()
    mask.scatter_(1, targets.type(torch.int64), True)

    selected_predictions = predictions[~mask].view(-1, num_class - 1)

    return torch.max(selected_predictions - class_val, dim=-1).values
