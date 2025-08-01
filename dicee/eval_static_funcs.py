import os
import torch
from typing import Dict, Tuple, List, Callable
from .knowledge_graph_embeddings import KGE
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


@torch.no_grad()
def evaluate_link_prediction_performance(model: KGE, triples, er_vocab: Dict[Tuple, List],
                                         re_vocab: Dict[Tuple, List]) -> Dict:
    """

    Parameters
    ----------
    model
    triples
    er_vocab
    re_vocab

    Returns
    -------

    """
    assert isinstance(model, KGE)
    model.model.eval()
    hits = dict()
    reciprocal_ranks = []
    num_entities = model.num_entities

    # Iterate over test triples
    all_entities = torch.arange(0, num_entities).long()
    all_entities = all_entities.reshape(len(all_entities), )
    # Iterating one by one is not good when you are using batch norm
    for i in tqdm(range(0, len(triples))):
        # (1) Get a triple (head entity, relation, tail entity
        data_point = triples[i]
        str_h, str_r, str_t = data_point[0], data_point[1], data_point[2]

        h, r, t = model.get_entity_index(str_h), model.get_relation_index(str_r), model.get_entity_index(str_t)
        # (2) Predict missing heads and tails
        x = torch.stack((torch.tensor(h).repeat(num_entities, ),
                         torch.tensor(r).repeat(num_entities, ),
                         all_entities), dim=1)

        predictions_tails = model.model.forward_triples(x)
        x = torch.stack((all_entities,
                         torch.tensor(r).repeat(num_entities, ),
                         torch.tensor(t).repeat(num_entities)
                         ), dim=1)

        predictions_heads = model.model.forward_triples(x)
        del x

        # 3. Computed filtered ranks for missing tail entities.
        # 3.1. Compute filtered tail entity rankings
        filt_tails = [model.entity_to_idx[i] for i in er_vocab[(str_h, str_r)]]
        # 3.2 Get the predicted target's score
        target_value = predictions_tails[t].item()
        # 3.3 Filter scores of all triples containing filtered tail entities
        predictions_tails[filt_tails] = -np.Inf
        # 3.4 Reset the target's score
        predictions_tails[t] = target_value
        # 3.5. Sort the score
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == t)[0][0]

        # 4. Computed filtered ranks for missing head entities.
        # 4.1. Retrieve head entities to be filtered
        filt_heads = [model.entity_to_idx[i] for i in re_vocab[(str_r, str_t)]]
        # 4.2 Get the predicted target's score
        target_value = predictions_heads[h].item()
        # 4.3 Filter scores of all triples containing filtered head entities.
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[h] = target_value
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_head_entity_rank = np.where(sort_idxs == h)[0][0]

        # 4. Add 1 to ranks as numpy array first item has the index of 0.
        filt_head_entity_rank += 1
        filt_tail_entity_rank += 1

        rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
        # 5. Store reciprocal ranks.
        reciprocal_ranks.append(rr)
        # print(f'{i}.th triple: mean reciprical rank:{rr}')

        # 4. Compute Hit@N
        for hits_level in range(1, 11):
            res = 1 if filt_head_entity_rank <= hits_level else 0
            res += 1 if filt_tail_entity_rank <= hits_level else 0
            if res > 0:
                hits.setdefault(hits_level, []).append(res)

    mean_reciprocal_rank = sum(reciprocal_ranks) / (float(len(triples) * 2))

    if 1 in hits:
        hit_1 = sum(hits[1]) / (float(len(triples) * 2))
    else:
        hit_1 = 0

    if 3 in hits:
        hit_3 = sum(hits[3]) / (float(len(triples) * 2))
    else:
        hit_3 = 0

    if 10 in hits:
        hit_10 = sum(hits[10]) / (float(len(triples) * 2))
    else:
        hit_10 = 0
    return {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}


@torch.no_grad()
def evaluate_link_prediction_performance_with_reciprocals(model: KGE, triples,
                                                          er_vocab: Dict[Tuple, List]):
    model.model.eval()
    entity_to_idx = model.entity_to_idx
    relation_to_idx = model.relation_to_idx
    batch_size = model.model.args["batch_size"]
    num_triples = len(triples)
    ranks = []
    # Hit range
    hits_range = [i for i in range(1, 11)]
    hits = {i: [] for i in hits_range}
    # Iterate over integer indexed triples in mini batch fashion
    for i in range(0, num_triples, batch_size):
        # (1) Get a batch of data.
        str_data_batch = triples[i:i + batch_size]
        data_batch = np.array(
            [[entity_to_idx[str_triple[0]], relation_to_idx[str_triple[1]], entity_to_idx[str_triple[2]]] for
             str_triple in str_data_batch])
        # (2) Extract entities and relations.
        e1_idx_r_idx, e2_idx = torch.LongTensor(data_batch[:, [0, 1]]), torch.tensor(data_batch[:, 2])
        # (3) Predict missing entities, i.e., assign probs to all entities.
        predictions = model.model(e1_idx_r_idx)
        # (4) Filter entities except the target entity
        for j in range(data_batch.shape[0]):
            # (4.1) Get the ids of the head entity, the relation and the target tail entity in the j.th triple.
            str_h, str_r, str_t = str_data_batch[j]

            id_e, id_r, id_e_target = data_batch[j]
            # (4.2) Get all ids of all entities occurring with the head entity and relation extracted in 4.1.
            filt = [entity_to_idx[_] for _ in er_vocab[(str_h, str_r)]]
            # (4.3) Store the assigned score of the target tail entity extracted in 4.1.
            target_value = predictions[j, id_e_target].item()
            # (4.4.1) Filter all assigned scores for entities.
            predictions[j, filt] = -np.Inf
            # (4.5) Insert 4.3. after filtering.
            predictions[j, id_e_target] = target_value
        # (5) Sort predictions.
        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
        # (6) Compute the filtered ranks.
        for j in range(data_batch.shape[0]):
            # index between 0 and \inf
            rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
            ranks.append(rank)
            for hits_level in hits_range:
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
    # (7) Sanity checking: a rank for a triple
    assert len(triples) == len(ranks) == num_triples
    hit_1 = sum(hits[1]) / num_triples
    hit_3 = sum(hits[3]) / num_triples
    hit_10 = sum(hits[10]) / num_triples
    mean_reciprocal_rank = np.mean(1. / np.array(ranks))

    results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
    return results


def evaluate_link_prediction_performance_with_bpe_reciprocals(model: KGE,
                                                              within_entities: List[str],
                                                              triples: List[List[str]],
                                                              er_vocab: Dict[Tuple, List]):
    triples = np.array(triples)
    model.model.eval()
    entity_to_idx={ent: id_ for id_, ent in enumerate(within_entities)}
    padded_bpe_within_entities = model.get_bpe_token_representation(within_entities)
    padded_bpe_within_entities = torch.LongTensor(padded_bpe_within_entities)

    batch_size = model.model.args["batch_size"]
    num_triples = len(triples)
    ranks = []
    # Hit range
    hits_range = [i for i in range(1, 11)]
    hits = {i: [] for i in hits_range}
    # (!!!) Set the entities for which triple scores are computed
    model.model.ordered_bpe_entities = padded_bpe_within_entities
    # Iterate over integer indexed triples in mini batch fashion
    for i in range(0, num_triples, batch_size):
        # (1) Get a batch of data.
        str_data_batch = triples[i:i + batch_size]

        str_heads, str_rels = str_data_batch[:, 0].tolist(), str_data_batch[:, 1].tolist()

        padded_bpe_heads = torch.LongTensor(model.get_bpe_token_representation(str_heads)).unsqueeze(1)
        padded_bpe_rels = torch.LongTensor(model.get_bpe_token_representation(str_rels)).unsqueeze(1)

        e1_idx_r_idx = torch.cat((padded_bpe_heads, padded_bpe_rels), dim=1)

        # (3) Predict missing entities, i.e., assign probs to all entities.
        predictions = model.model(e1_idx_r_idx)
        # (4) Filter entities except the target entity
        for j, (str_h, str_r, str_t) in enumerate(str_data_batch):
            # (4.1) Get the ids of the head entity, the relation and the target tail entity in the j.th triple.
            id_e_target = entity_to_idx[str_t]
            # (4.2) Get all ids of all entities occurring with the head entity and relation extracted in 4.1.
            filt = [entity_to_idx[_] for _ in er_vocab[(str_h, str_r)]]
            # (4.3) Store the assigned score of the target tail entity extracted in 4.1.
            target_value = predictions[j, id_e_target].item()
            # (4.4.1) Filter all assigned scores for entities.
            predictions[j, filt] = -np.Inf
            # (4.5) Insert 4.3. after filtering.
            predictions[j, id_e_target] = target_value
        # (5) Sort predictions.
        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

        # (6) Compute the filtered ranks.
        for j,(_, __, str_t) in enumerate(str_data_batch):
            # index between 0 and \inf
            rank = torch.where(sort_idxs[j] == entity_to_idx[str_t])[0].item() + 1
            ranks.append(rank)
            for hits_level in hits_range:
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
    # (7) Sanity checking: a rank for a triple
    assert len(triples) == len(ranks) == num_triples
    hit_1 = sum(hits[1]) / num_triples
    hit_3 = sum(hits[3]) / num_triples
    hit_10 = sum(hits[10]) / num_triples
    mean_reciprocal_rank = np.mean(1. / np.array(ranks))

    results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
    return results


# @torch.no_grad()
def evaluate_link_prediction_performance_with_bpe(model: KGE,
                                                  within_entities: List[str],
                                                  triples: List[Tuple[str]],
                                                  er_vocab: Dict[Tuple, List], re_vocab: Dict[Tuple, List]):
    """

    Parameters
    ----------
    model
    triples
    within_entities
    er_vocab
    re_vocab

    Returns
    -------

    """
    assert isinstance(triples, list)
    assert len(triples[0]) == 3
    model.model.eval()
    hits = dict()
    reciprocal_ranks = []
    # Iterate over test triples
    num_entities = len(within_entities)
    bpe_entity_to_idx = dict()
    all_bpe_entities = []
    for idx, str_entity in tqdm(enumerate(within_entities)):
        shaped_bpe_entity = model.get_bpe_token_representation(str_entity)
        bpe_entity_to_idx[shaped_bpe_entity] = idx
        all_bpe_entities.append(shaped_bpe_entity)
    all_bpe_entities = torch.LongTensor(all_bpe_entities)
    for str_h, str_r, str_t in tqdm(triples):
        # (1) Indices of head and tail entities in all entities
        idx_bpe_h = bpe_entity_to_idx[model.get_bpe_token_representation(str_h)]
        idx_bpe_t = bpe_entity_to_idx[model.get_bpe_token_representation(str_t)]

        # (2) Tensor representation of sequence of sub-word representation of entities and relations
        torch_bpe_h = torch.LongTensor(model.get_bpe_token_representation(str_h)).unsqueeze(0)
        torch_bpe_r = torch.LongTensor(model.get_bpe_token_representation(str_r)).unsqueeze(0)
        torch_bpe_t = torch.LongTensor(model.get_bpe_token_representation(str_t)).unsqueeze(0)

        # (3) Missing head and tail predictions
        x = torch.stack((torch.repeat_interleave(input=torch_bpe_h, repeats=num_entities, dim=0),
                         torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
                         all_bpe_entities), dim=1)
        with torch.no_grad():
            predictions_tails = model.model(x)
        x = torch.stack((all_bpe_entities,
                         torch.repeat_interleave(input=torch_bpe_r, repeats=num_entities, dim=0),
                         torch.repeat_interleave(input=torch_bpe_t, repeats=num_entities, dim=0)), dim=1)
        with torch.no_grad():
            predictions_heads = model.model(x)
        # 3. Computed filtered ranks for missing tail entities.
        # 3.1. Compute filtered tail entity rankings
        filt_tails = [bpe_entity_to_idx[model.get_bpe_token_representation(i)] for i in er_vocab[(str_h, str_r)]]
        # 3.2 Get the predicted target's score
        target_value = predictions_tails[idx_bpe_t].item()
        # 3.3 Filter scores of all triples containing filtered tail entities
        predictions_tails[filt_tails] = -np.Inf
        # 3.4 Reset the target's score
        predictions_tails[idx_bpe_t] = target_value
        # 3.5. Sort the score
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_tail_entity_rank = np.where(sort_idxs == idx_bpe_t)[0][0]

        # 4. Computed filtered ranks for missing head entities.
        # 4.1. Retrieve head entities to be filtered
        filt_heads = [bpe_entity_to_idx[model.get_bpe_token_representation(i)] for i in re_vocab[(str_r, str_t)]]
        # 4.2 Get the predicted target's score
        target_value = predictions_heads[idx_bpe_h].item()
        # 4.3 Filter scores of all triples containing filtered head entities.
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[idx_bpe_h] = target_value
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        sort_idxs = sort_idxs.detach()
        filt_head_entity_rank = np.where(sort_idxs == idx_bpe_h)[0][0]

        # 4. Add 1 to ranks as numpy array first item has the index of 0.
        filt_head_entity_rank += 1
        filt_tail_entity_rank += 1

        rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
        # 5. Store reciprocal ranks.
        reciprocal_ranks.append(rr)
        # print(f'{i}.th triple: mean reciprical rank:{rr}')

        # 4. Compute Hit@N
        for hits_level in range(1, 11):
            res = 1 if filt_head_entity_rank <= hits_level else 0
            res += 1 if filt_tail_entity_rank <= hits_level else 0
            if res > 0:
                hits.setdefault(hits_level, []).append(res)

    mean_reciprocal_rank = sum(reciprocal_ranks) / (float(len(triples) * 2))

    if 1 in hits:
        hit_1 = sum(hits[1]) / (float(len(triples) * 2))
    else:
        hit_1 = 0

    if 3 in hits:
        hit_3 = sum(hits[3]) / (float(len(triples) * 2))
    else:
        hit_3 = 0

    if 10 in hits:
        hit_10 = sum(hits[10]) / (float(len(triples) * 2))
    else:
        hit_10 = 0

    results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
    return results


@torch.no_grad()
def evaluate_lp_bpe_k_vs_all(model, triples: List[List[str]], er_vocab=None, batch_size=None,
                             func_triple_to_bpe_representation: Callable = None, str_to_bpe_entity_to_idx=None):
    # (1) set model to eval model
    model.model.eval()
    num_triples = len(triples)
    ranks = []
    # Hit range
    hits_range = [i for i in range(1, 11)]
    hits = {i: [] for i in hits_range}
    # Iterate over integer indexed triples in mini batch fashion
    for i in range(0, num_triples, batch_size):
        str_data_batch = triples[i:i + batch_size]
        # (1) Get a batch of data.
        torch_batch_bpe_triple = torch.LongTensor(
            [func_triple_to_bpe_representation(i) for i in str_data_batch])

        # (2) Extract entities and relations.
        bpe_hr = torch_batch_bpe_triple[:, [0, 1], :]
        # (3) Predict missing entities, i.e., assign probs to all entities.
        predictions = model(bpe_hr)
        # (4) Filter entities except the target entity
        for j in range(len(predictions)):
            # (4.2) Get all ids of all entities occurring with the head entity and relation extracted in 4.1.
            h, r, t = str_data_batch[j]
            id_e_target = str_to_bpe_entity_to_idx[t]
            filt_idx_entities = [str_to_bpe_entity_to_idx[_] for _ in er_vocab[(h, r)]]

            # (4.3) Store the assigned score of the target tail entity extracted in 4.1.
            target_value = predictions[j, id_e_target].item()
            # (4.4.1) Filter all assigned scores for entities.
            predictions[j, filt_idx_entities] = -np.Inf
            # (4.4.2) Filter entities based on the range of a relation as well.
            # (4.5) Insert 4.3. after filtering.
            predictions[j, id_e_target] = target_value
        # (5) Sort predictions.
        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

        # (6) Compute the filtered ranks.
        for j in range(len(predictions)):
            t = str_data_batch[j][2]
            # index between 0 and \inf
            rank = torch.where(sort_idxs[j] == str_to_bpe_entity_to_idx[t])[0].item() + 1
            ranks.append(rank)
            for hits_level in hits_range:
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
    # (7) Sanity checking: a rank for a triple
    assert len(triples) == len(ranks) == num_triples
    hit_1 = sum(hits[1]) / num_triples
    hit_3 = sum(hits[3]) / num_triples
    hit_10 = sum(hits[10]) / num_triples
    mean_reciprocal_rank = np.mean(1. / np.array(ranks))

    results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
    return results

def evaluate_literal_prediction(
        kge_model: KGE,
        eval_file_path: str = None,
        store_lit_preds: bool = True,
        eval_literals: bool = True,
        loader_backend: str = "pandas",
        return_attr_error_metrics: bool = False,
    ):
        """
        Evaluates the trained literal prediction model on a test file.

        Args:
            eval_file_path (str): Path to the evaluation file.
            store_lit_preds (bool): If True, stores the predictions in a CSV file.
            eval_literals (bool): If True, evaluates the literal predictions and prints error metrics.
            loader_backend (str): Backend for loading the dataset ('pandas' or 'rdflib').

        Returns:
            pd.DataFrame: DataFrame containing error metrics for each attribute if return_attr_error_metrics is True.
        Raises:
            RuntimeError: If the kGE model does not have a trained literal model.
            AssertionError: If the kGE model is not an instance of KGE or if the test set has no valid entities or attributes.
        """
        # kGE Literal model sanity checking
        assert isinstance(kge_model, KGE), "kge_model must be an instance of KGE."
        if not hasattr(kge_model, "literal_model") or kge_model.literal_model is None:
            raise RuntimeError("Literal model is not trained or loaded.")
        
        # sanity checking done in load_and_validate_literal_data
        test_df_unfiltered = kge_model.literal_dataset.load_and_validate_literal_data(
            file_path=eval_file_path,loader_backend=loader_backend
        )
        test_df = test_df_unfiltered[
            test_df_unfiltered["head"].isin(kge_model.entity_to_idx.keys()) &
            test_df_unfiltered["attribute"].isin(kge_model.data_property_to_idx.keys())
            ]

        entities = test_df["head"].to_list()
        attributes = test_df["attribute"].to_list()
        
        assert len(entities) > 0, "No valid entities in test set — check entity_to_idx."
        assert len(attributes) > 0, "No valid attributes in test set — check data_property_to_idx."
        
        test_df["predictions"] = kge_model.predict_literals(
            entity=entities, attribute=attributes
        )

        # If store_lit_preds is True, save the predictions to a CSV file
        if store_lit_preds:
            prediction_df = test_df[["head", "attribute", "predictions"]]
            prediction_path = os.path.join(kge_model.path, "lit_predictions.csv")
            prediction_df.to_csv(prediction_path, index=False)
            print(f"Literal predictions saved to {prediction_path}")

        # Calculate,print and store error metrics
        if eval_literals:
            attr_error_metrics = test_df.groupby("attribute").agg(
                MAE=("value", lambda x: mean_absolute_error(x, test_df.loc[x.index, "predictions"])),
                RMSE=("value", lambda x: root_mean_squared_error(x, test_df.loc[x.index, "predictions"]))
            ).reset_index()

            pd.options.display.float_format = "{:.6f}".format
            print("Literal-Prediction evaluation results  on Test Set")
            print(attr_error_metrics)
            results_path = os.path.join(kge_model.path, "lit_eval_results.csv")
            attr_error_metrics.to_csv(results_path, index=False)
            print(f"Literal-Prediction evaluation results saved to {results_path}")

            if return_attr_error_metrics:
                return attr_error_metrics

@torch.no_grad()
def evaluate_ensemble_link_prediction_performance(models, triples, er_vocab: Dict[Tuple, List],
    weights: List[float] = None, batch_size: int = 512, weighted_averaging : bool = True ) -> Dict:
    """
    Evaluates link prediction performance of an ensemble of KGE models.
    Args:
        models : List of KGE models (snapshots)
        triples : np.ndarray or list of lists, shape (N,3), all integer indices (head, rel, tail)
        er_vocab : Dict[Tuple, List]
            Mapping (head_idx, rel_idx) → list of tail_idx to filter (incl. target).
        weights : Optional[List[float]]
            Weights for model averaging. If None, use uniform (=simple mean).
        batch_size : int
    Returns:
        dict of link prediction metrics (H@1, H@3, H@10, MRR)
    """
    num_triples = len(triples)
    ranks = []
    hits_range = list(range(1, 11))
    hits = {i: [] for i in hits_range}
    n_models = len(models)

    if  weighted_averaging:
        assert weights is not None, "Weights must be provided for weighted averaging."
        assert len(weights) == n_models, "Number of weights must match number of models."
        weights_tensor = torch.FloatTensor(weights)

    for i in range(0, num_triples, batch_size):
        data_batch = np.array(triples[i:i + batch_size])  # Ensure ndarray
        e1_idx_r_idx = torch.LongTensor(data_batch[:, [0, 1]])
        e2_idx = torch.LongTensor(data_batch[:, 2])

        preds_list = []
        for model in models:
            model.eval()
            preds_raw = model(e1_idx_r_idx) # [batch, n_entities]
            preds_min = preds_raw.min(dim=1, keepdim=True)[0]
            preds_max = preds_raw.max(dim=1, keepdim=True)[0]
            preds = (preds_raw - preds_min) / (preds_max - preds_min + 1e-8)

            preds_list.append(preds)
        preds_stack = torch.stack(preds_list, dim=0)  # [n_models, batch, n_entities]

        if weighted_averaging:
            # Weighted mean aggregation
            avg_preds = torch.sum(preds_stack * weights_tensor.view(-1, 1, 1), dim=0)
        else:
            # Simple mean aggregation
            avg_preds = torch.mean(preds_stack, dim=0)
        

        # Filtering and scoring
        for j in range(data_batch.shape[0]):
            id_e, id_r, id_e_target = data_batch[j]
            filt = er_vocab.get((id_e, id_r), [])
            target_value = avg_preds[j, id_e_target].item()
            if len(filt) > 0:
                avg_preds[j, filt] = -np.Inf
            avg_preds[j, id_e_target] = target_value

        # Ranking and metrics
        _, sort_idxs = torch.sort(avg_preds, dim=1, descending=True)
        for j in range(data_batch.shape[0]):
            rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
            ranks.append(rank)
            for hits_level in hits_range:
                if rank <= hits_level:
                    hits[hits_level].append(1.0)

    assert len(triples) == len(ranks) == num_triples
    hit_1 = sum(hits[1]) / num_triples
    hit_3 = sum(hits[3]) / num_triples
    hit_10 = sum(hits[10]) / num_triples
    mean_reciprocal_rank = np.mean(1. / np.array(ranks))
    return {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}