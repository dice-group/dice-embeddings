import os
import numpy as np
import torch
import datetime
import pytorch_lightning as pl
from .models import *
import time
import pandas as pd
import json
import glob
import pandas
import polars
import functools
import pickle

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
            print(f"{storage_path}/model.pt is not found. The model will be trained with random weights")
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
    with open(path_of_experiment_folder + '/entity_to_idx.p', 'rb') as f:
        entity_to_idx = pickle.load(f)
    with open(path_of_experiment_folder + '/relation_to_idx.p', 'rb') as f:
        relation_to_idx = pickle.load(f)
    assert isinstance(entity_to_idx, dict)
    assert isinstance(relation_to_idx, dict)
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
    with open(path_of_experiment_folder + '/entity_to_idx.p', 'rb') as f:
        entity_to_idx = pickle.load(f)
    with open(path_of_experiment_folder + '/relation_to_idx.p', 'rb') as f:
        relation_to_idx = pickle.load(f)
    assert isinstance(entity_to_idx, dict)
    assert isinstance(relation_to_idx, dict)
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
    if np.iinfo(np.int8).max > num:
        # print(f'Setting int8,\t {np.iinfo(np.int8).max}')
        train_set = train_set.astype(np.int8)
    elif np.iinfo(np.int16).max > num:
        # print(f'Setting int16,\t {np.iinfo(np.int16).max}')
        train_set = train_set.astype(np.int16)
    elif np.iinfo(np.int32).max > num:
        # print(f'Setting int32,\t {np.iinfo(np.int32).max}')
        train_set = train_set.astype(np.int32)
    else:
        raise TypeError('Int64?')
    return train_set


def save_checkpoint_model(trainer, model, path: str) -> None:
    """ Store Pytorch model into disk"""
    try:
        torch.save(model.state_dict(), path)
    except ReferenceError as e:
        print(e)
        print(model.name)
        print('Could not save the model correctly')


def store(trainer,
          trained_model, model_name: str = 'model', full_storage_path: str = None,
          dataset=None, save_as_csv=False) -> None:
    """
    Store trained_model model and save embeddings into csv file.
    :param trainer: an instance of trainer class
    :param dataset: an instance of KG see core.knowledge_graph.
    :param full_storage_path: path to save parameters.
    :param model_name: string representation of the name of the model.
    :param trained_model: an instance of BaseKGE(pl.LightningModule) see core.models.base_model .
    :param save_as_csv: for easy access of embeddings.
    :return:
    """
    assert full_storage_path is not None
    assert dataset is not None
    assert isinstance(model_name, str)
    assert len(model_name) > 1

    # (1) Save pytorch model in trained_model .
    save_checkpoint_model(trainer=trainer,
                          model=trained_model, path=full_storage_path + f'/{model_name}.pt')
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


def add_noisy_triples(train_set: pd.DataFrame, add_noise_rate: float) -> pd.DataFrame:
    """
    Add randomly constructed triples
    :param train_set:
    :param add_noise_rate:
    :return:
    """
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


def read_or_load_kg(args, cls):
    print('*** Read or Load Knowledge Graph  ***')
    start_time = time.time()
    kg = cls(data_dir=args.path_dataset_folder,
             num_core=args.num_core,
             add_reciprical=args.apply_reciprical_or_noise,
             eval_model=args.eval_model,
             read_only_few=args.read_only_few,
             sample_triples_ratio=args.sample_triples_ratio,
             path_for_serialization=args.full_storage_path,
             min_freq_for_vocab=args.min_freq_for_vocab,
             path_for_deserialization=args.path_experiment_folder if hasattr(args, 'path_experiment_folder') else None,
             backend=args.backend)
    print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
    print(kg.description_of_input)
    return kg


def intialize_model(args: dict) -> Tuple[pl.LightningModule, str]:
    # @TODO: Apply construct_krone as callback? or use KronE_QMult as a prefix.
    # @TODO: Remove form_of_labelling
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
    elif model_name == 'Shallom':
        model = Shallom(args=args)
        form_of_labelling = 'RelationPrediction'
    elif model_name == 'ConEx':
        model = ConEx(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'AConEx':
        model = AConEx(args=args)
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
    elif model_name == 'TransE':
        model = TransE(args=args)
        form_of_labelling = 'EntityPrediction'
    elif model_name == 'Pyke':
        model = Pyke(args=args)
        form_of_labelling = 'Pyke'
    elif model_name == 'CLf':
        model = CLf(args=args)
        form_of_labelling = 'EntityPrediction'
    # elif for PYKEEN https://github.com/dice-group/dice-embeddings/issues/54
    else:
        raise ValueError
    return model, form_of_labelling


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


@timeit
def vocab_to_parquet(vocab_to_idx, name, path_for_serialization, print_into):
    # @TODO: This function should take any DASK/Pandas DataFrame or Series.
    print(print_into)
    vocab_to_idx.to_parquet(path_for_serialization + f'/{name}', compression='gzip', engine='pyarrow')
    print('Done !\n')


def create_experiment_folder(folder_name='Experiments'):
    directory = os.getcwd() + '/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder


def continual_training_setup_executor(executor):
    if executor.is_continual_training:
        # (4.1) If it is continual, then store new models on previous path.
        executor.storage_path = executor.args.full_storage_path
    else:
        # (4.2) Create a folder for the experiments.
        executor.args.full_storage_path = create_experiment_folder(folder_name=executor.args.storage_path)
        executor.storage_path = executor.args.full_storage_path
        with open(executor.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(executor.args)
            json.dump(temp, file_descriptor, indent=3)
