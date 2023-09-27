# Deprecated
from dicee import KGE
import numpy as np
import pickle
import os
import torch
import click
import logging
from rich.table import Table
from rich.console import Console

def load_data(data_path: str, tasks):

        #load queries from the datapath
        queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        answers_hard = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        answers_easy = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))

        for task in list(queries.keys()):
            if task not in query_name_dict or query_name_dict[task] not in tasks:
                del queries[task]

        return queries, answers_easy, answers_hard

query_name_dict = {

        ("e", ("r",)): "1p",
        ("e", ("r", "r")): "2p",
        ("e", ("r", "r", "r",),): "3p",
        (("e", ("r",)), ("e", ("r",))): "2i",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
        ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
        (("e", ("r", "r")), ("e", ("r",))): "pi",
        # negation
        (("e", ("r",)), ("e", ("r", "n"))): "2in",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
        ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
        (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
        (("e", ("r", "r", "n")), ("e", ("r",))): "pni",

        # union
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2u",
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up",

    }
name_query_dict = {value: key for key, value in query_name_dict.items()}


def t_norm(tens_1: torch.Tensor, tens_2: torch.Tensor, t_norm: str = 'min')-> torch.Tensor:
    if 'min' in t_norm:
        return torch.min(tens_1, tens_2)
    elif 'prod' in t_norm:
        return tens_1 * tens_2


def t_conorm(tens_1: torch.Tensor, tens_2: torch.Tensor, t_conorm: str = 'min') -> torch.Tensor:
    if 'min' in t_conorm:
        return torch.max(tens_1, tens_2)
    elif 'prod' in t_conorm:
        return (tens_1 + tens_2) - (tens_1 * tens_2)

def negnorm(tens_1: torch.Tensor,lambda_: float,neg_norm: str = 'standard') -> torch.Tensor:
    if 'standard' in neg_norm:
        return 1-tens_1
    elif 'sugeno' in neg_norm:
        return (1 - tens_1) / (1 + lambda_ * tens_1)
    elif 'yager' in neg_norm:
        return (1 - torch.pow(tens_1, lambda_)) ** (1 / lambda_)

def scores_1p(model, queries):

    entity_scores={}
    for query in queries:
        head, relation = query

        # Get scores for the  atom
        atom1_scores = model.predict(h=[head], r=[relation[0]]).squeeze()
        entity_scores[query]=atom1_scores

    return entity_scores
def scores_2p(model, queries, tnorm: str, k_: int):
    # Function to calculate entity scores for type 2p structure



    entity_scores = {}

    for query in queries:
        head1, (relation1,relation2) = query


        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        k=min(k_,len(model.entity_to_idx))

        # sort atom1_scores in descending order and get the top k entities indices
        top_k_scores1,top_k_indices=torch.topk(atom1_scores,k)

        #using model.entity_to_idx.keys() take the name of entities from topk heads 2
        entity_to_idx_keys = list(model.entity_to_idx.keys())
        top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

        # Initialize an empty tensor
        atom2_scores = torch.empty(0, len(model.entity_to_idx)).to(atom1_scores.device)

        # Get scores for the second atom
        for head2 in top_k_heads:
            # The score tensor for the current head2
            atom2_score = model.predict(h=[head2], r=[relation2])
            # Concatenate the score tensor for the current head2 with the previous scores
            atom2_scores = torch.cat([atom2_scores, atom2_score], dim=0)

        topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom2_scores.shape[1])

        combined_scores = t_norm(topk_scores1_expanded, atom2_scores,tnorm)

        res, _ = torch.max(combined_scores, dim=0)

        entity_scores[query] = res

    return entity_scores

def scores_3p(model, queries, tnorm: str, k_: int):
        # Function to calculate entity scores for type 3p structure

        entity_scores = {}

        for query in queries:
            head1, (relation1, relation2, relation3) = query

            # Get scores for the first atom
            atom1_scores = model.predict(h=[head1], r=[relation1]).squeeze()
            k = min(k_, len(model.entity_to_idx))

            # Get the top k entities indices for the first atom
            top_k_scores1, top_k_indices1 = torch.topk(atom1_scores, k)

            # Get the name of entities from top k heads for the first atom
            entity_to_idx_keys = list(model.entity_to_idx.keys())
            top_k_heads1 = [entity_to_idx_keys[idx.item()] for idx in top_k_indices1]

            # Initialize an empty tensor for the second atom scores
            atom2_scores = torch.empty(0, len(model.entity_to_idx)).to(atom1_scores.device)

            # Get scores for the second atom
            for head2 in top_k_heads1:
                atom2_score = model.predict(h=[head2], r=[relation2])
                atom2_scores = torch.cat([atom2_scores, atom2_score], dim=0)

            # Get the top k entities indices for each head of the second atom
            top_k_scores2, top_k_indices2 = torch.topk(atom2_scores, k, dim=1)

            # Get the name of entities from top k heads for each head of the second atom
            top_k_heads2 = [[entity_to_idx_keys[idx.item()] for idx in row] for row in top_k_indices2]

            # Initialize an empty tensor for the third atom scores
            atom3_scores = torch.empty(0, len(model.entity_to_idx)).to(atom1_scores.device)

            # Get scores for the third atom
            for row in top_k_heads2:
                for head3 in row:
                    atom3_score = model.predict(h=[head3], r=[relation3])
                    atom3_scores = torch.cat([atom3_scores, atom3_score], dim=0)

            topk_scores1_2d = top_k_scores1.unsqueeze(-1).repeat(1,top_k_scores2.shape[1])
            topk_scores1_expanded = topk_scores1_2d.view(-1, 1).repeat(1,atom3_scores.shape[1])
            topk_scores2_expanded = top_k_scores2.view(-1, 1).repeat(1, atom3_scores.shape[1])

            inter_scores = t_norm(topk_scores1_expanded, topk_scores2_expanded, tnorm)
            # atom3_scores_flattened = atom3_scores.view(-1)

            combined_scores = t_norm(inter_scores, atom3_scores, tnorm)

            res, _ = torch.max(combined_scores, dim=0)

            entity_scores[query] = res

        return entity_scores


def scores_2i(model, queries, tnorm: str):
    # Function to calculate entity scores for type 2i structure

    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0]
        head2, relation2 = query[1]

        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()
        # Get scores for the second atom
        atom2_scores = model.predict(h=[head2], r=[relation2[0]]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []


        combined_scores=t_norm(atom1_scores,atom2_scores,tnorm)

        entity_scores[query] = combined_scores

    return entity_scores


def scores_3i(model, queries, tnorm: str):
    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0]
        head2, relation2 = query[1]
        head3, relation3 = query[2]
        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()
        # Get scores for the second atom
        atom2_scores = model.predict(h=[head2], r=[relation2[0]]).squeeze()
        # Get scores for the third atom
        atom3_scores = model.predict(h=[head3], r=[relation3[0]]).squeeze()


        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []


        inter_scores=t_norm(atom1_scores,atom2_scores,tnorm)
        combined_scores=t_norm(inter_scores,atom3_scores,tnorm)

        entity_scores[query] = combined_scores

    return entity_scores


def scores_pi(model, queries, tnorm: str, k_: int):

        # Combine the scores using the t-norm
        entity_scores = {}

        for query in queries:
            head1, (relation1, relation2) = query[0]
            head3, relation3=query[1]
            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = model.predict(h=[head1], r=[relation1]).squeeze()

            assert len(atom1_scores) == len(model.entity_to_idx)
            k = min(k_, len(model.entity_to_idx))

            # sort atom1_scores in descending order and get the top k entities indices
            top_k_scores1, top_k_indices = torch.topk(atom1_scores, k)

            # using model.entity_to_idx.keys() take the name of entities from topk heads 2
            entity_to_idx_keys = list(model.entity_to_idx.keys())
            top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

            # Initialize an empty tensor
            atom2_scores = torch.empty(0, len(model.entity_to_idx)).to(atom1_scores.device)

            # Get scores for the second atom
            for head2 in top_k_heads:
                # The score tensor for the current head2
                atom2_score = model.predict(h=[head2], r=[relation2])
                # Concatenate the score tensor for the current head2 with the previous scores
                atom2_scores = torch.cat([atom2_scores, atom2_score], dim=0)

            topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom2_scores.shape[1])

            inter_scores = t_norm(topk_scores1_expanded, atom2_scores, tnorm)


            scores_2p_query, _ = torch.max(inter_scores, dim=0)

            scores_1p_query = model.predict(h=[head3], r=[relation3[0]]).squeeze()


            combined_scores = t_norm(scores_2p_query, scores_1p_query,tnorm)
            entity_scores[query] = combined_scores

        return entity_scores


def scores_ip(model, queries, tnorm: str, k_: int):


    # Initialize the entity scores dictionary
    entity_scores = {}
    for query in queries:
        head1, relation1 = query[0][0]
        head2, relation2 = query[0][1]
        relation_1p=query[1]
        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()
        # Get scores for the second atom
        atom2_scores = model.predict(h=[head2], r=[relation2[0]]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []

        scores_2i_query=t_norm(atom1_scores,atom2_scores,tnorm)
        # Get the top k entities from the 2i query

        k = min(k_, len(model.entity_to_idx))

        # sort atom1_scores in descending order and get the top k entities indices
        top_k_scores1, top_k_indices = torch.topk(scores_2i_query, k)

        # using model.entity_to_idx.keys() take the name of entities from topk heads
        entity_to_idx_keys = list(model.entity_to_idx.keys())
        top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

        # Get scores for the second atom
        # Initialize an empty tensor
        atom3_scores = torch.empty(0, len(model.entity_to_idx)).to(scores_2i_query.device)

        # Get scores for the second atom
        for head3 in top_k_heads:
            # The score tensor for the current head2
            atom3_score = model.predict(h=[head3], r=[relation_1p[0]])
            # Concatenate the score tensor for the current head2 with the previous scores
            atom3_scores = torch.cat([atom3_scores, atom3_score], dim=0)

        topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom3_scores.shape[1])

        combined_scores = t_norm(topk_scores1_expanded, atom3_scores, tnorm)

        res, _ = torch.max(combined_scores, dim=0)

        entity_scores[query] = res

    return entity_scores


def scores_2u(model, queries, tconorm: str):
    # Function to calculate entity scores for type 2u structure

    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0]
        head2, relation2 = query[1]

        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()
        # Get scores for the second atom
        atom2_scores = model.predict(h=[head2], r=[relation2[0]]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []


        combined_scores=t_conorm(atom1_scores,atom2_scores,tconorm)

        entity_scores[query] = combined_scores

    return entity_scores


def scores_up(model, queries, tnorm: str, tconorm: str, k_: int):


    # Initialize the entity scores dictionary
    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0][0]
        head2, relation2 = query[0][1]
        relation_1p = query[1]

        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()

        # Get scores for the second atom
        atom2_scores = model.predict(h=[head2], r=[relation2[0]]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)

        scores_2u_query = t_conorm(atom1_scores, atom2_scores, tconorm)

        # Get the top k entities from the 2i query
        k = min(k_, len(model.entity_to_idx))

        # Sort atom1_scores in descending order and get the top k entities indices
        top_k_scores1, top_k_indices = torch.topk(scores_2u_query, k)

        # Using model.entity_to_idx.keys() take the name of entities from topk heads
        entity_to_idx_keys = list(model.entity_to_idx.keys())
        top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

        # Initialize an empty tensor
        atom3_scores = torch.empty(0, len(model.entity_to_idx)).to(scores_2u_query.device)

        for head3 in top_k_heads:
            # The score tensor for the current head3
            atom3_score = model.predict(h=[head3], r=[relation_1p[0]])

            # Concatenate the score tensor for the current head3 with the previous scores
            atom3_scores = torch.cat([atom3_scores, atom3_score], dim=0)

        topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom3_scores.shape[1])
        combined_scores = t_norm(topk_scores1_expanded, atom3_scores, tnorm)

        res, _ = torch.max(combined_scores, dim=0)
        entity_scores[query] = res

    return entity_scores


def scores_2in(model,queries,tnorm: str,neg_norm: str,lambda_: float):
    # Function to calculate entity scores for type 2in structure

    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0]
        head2, relation2 = query[1]

        # Calculate entity scores for each query
        # Get scores for the first atom (positive)
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()
        # Get scores for the second atom (negative)
        #if neg_norm == "standard":
        predictions = model.predict(h=[head2], r=[relation2[0]]).squeeze()
        atom2_scores =negnorm(predictions,lambda_,neg_norm)


        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []


        combined_scores = t_norm(atom1_scores,atom2_scores,tnorm)
        entity_scores[query] = combined_scores

    return entity_scores


def scores_3in(model,queries,tnorm: str,neg_norm: str,lambda_: float):
    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0]
        head2, relation2 = query[1]
        head3, relation3 = query[2]

        # Calculate entity scores for each query
        # Get scores for the first atom (positive)
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()
        # Get scores for the second atom (negative)
        # modelling standard negation (1-x)
        atom2_scores = model.predict(h=[head2], r=[relation2[0]]).squeeze()
        # Get scores for the third atom
        # if neg_norm == "standard":
        predictions = model.predict(h=[head3], r=[relation3[0]]).squeeze()
        atom3_scores =negnorm(predictions,lambda_,neg_norm)

        assert len(atom1_scores) == len(model.entity_to_idx)
        combined_scores = []

        inter_scores = t_norm(atom1_scores, atom2_scores, tnorm)
        combined_scores = t_norm(inter_scores, atom3_scores, tnorm)
        entity_scores[query] = combined_scores

    return entity_scores

def scores_inp(model,queries,tnorm: str,neg_norm: str,lambda_: float,k_: int):
    """((("e", ("r",)), ("e", ("r", "n"))), ("r",))"""
    entity_scores = {}

    for query in queries:
        head1, relation1 = query[0][0]
        head2, relation2 = query[0][1]
        relation_1p = query[1]

        # Calculate entity scores for each query
        # Get scores for the first atom (positive)
        atom1_scores = model.predict(h=[head1], r=[relation1[0]]).squeeze()
        # Get scores for the second atom (negative)
        # if neg_norm == "standard":
        predictions = model.predict(h=[head2], r=[relation2[0]]).squeeze()
        atom2_scores = negnorm(predictions, lambda_, neg_norm)

        assert len(atom1_scores) == len(model.entity_to_idx)

        scores_2in_query = t_norm(atom1_scores, atom2_scores, tnorm)
        k = min(k_, len(model.entity_to_idx))

        # sort atom1_scores in descending order and get the top k entities indices
        top_k_scores1, top_k_indices = torch.topk(scores_2in_query, k)

        # using model.entity_to_idx.keys() take the name of entities from topk heads 2
        entity_to_idx_keys = list(model.entity_to_idx.keys())
        top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

        # Get scores for the second atom
        # Initialize an empty tensor
        atom3_scores = torch.empty(0, len(model.entity_to_idx)).to(scores_2in_query.device)

        # Get scores for the second atom
        for head3 in top_k_heads:
            # The score tensor for the current head2
            atom3_score = model.predict(h=[head3], r=[relation_1p[0]])
            # Concatenate the score tensor for the current head2 with the previous scores
            atom3_scores = torch.cat([atom3_scores, atom3_score], dim=0)

        topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom3_scores.shape[1])

        combined_scores = t_norm(topk_scores1_expanded, atom3_scores, tnorm)

        res, _ = torch.max(combined_scores, dim=0)

        entity_scores[query] = res

    return entity_scores

def scores_pin(model, queries, tnorm: str, neg_norm: str,lambda_: float,k_: int):
    """(("e", ("r", "r")), ("e", ("r", "n")))"""
    entity_scores = {}

    for query in queries:
        head1, (relation1, relation2) = query[0]
        head3, relation3 = query[1]
        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        k = min(k_, len(model.entity_to_idx))

        # sort atom1_scores in descending order and get the top k entities indices
        top_k_scores1, top_k_indices = torch.topk(atom1_scores, k)

        # using model.entity_to_idx.keys() take the name of entities from topk heads 2
        entity_to_idx_keys = list(model.entity_to_idx.keys())
        top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

        # Initialize an empty tensor
        atom2_scores = torch.empty(0, len(model.entity_to_idx)).to(atom1_scores.device)

        # Get scores for the second atom
        for head2 in top_k_heads:
            # The score tensor for the current head2
            atom2_score = model.predict(h=[head2], r=[relation2])
            # Concatenate the score tensor for the current head2 with the previous scores
            atom2_scores = torch.cat([atom2_scores, atom2_score], dim=0)

        topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom2_scores.shape[1])

        inter_scores = t_norm(topk_scores1_expanded, atom2_scores, tnorm)

        scores_2p_query, _ = torch.max(inter_scores, dim=0)

        scores_1p_query = model.predict(h=[head3], r=[relation3[0]]).squeeze()
        #taking negation for the e,(r,n) part of query
        neg_scores_1p_query = negnorm(scores_1p_query, lambda_, neg_norm)
        combined_scores = t_norm(scores_2p_query, neg_scores_1p_query, tnorm)

        entity_scores[query] = combined_scores

    return entity_scores

def scores_pni(model,queries,tnorm: str,neg_norm: str,lambda_: float,k_: int):


    entity_scores = {}

    for query in queries:
        head1, (relation1, relation2, _) = query[0]
        head3, relation3 = query[1]
        # Calculate entity scores for each query
        # Get scores for the first atom
        atom1_scores = model.predict(h=[head1], r=[relation1]).squeeze()

        assert len(atom1_scores) == len(model.entity_to_idx)
        k = min(k_, len(model.entity_to_idx))

        # sort atom1_scores in descending order and get the top k entities indices
        top_k_scores1, top_k_indices = torch.topk(atom1_scores, k)

        # using model.entity_to_idx.keys() take the name of entities from topk heads 2
        entity_to_idx_keys = list(model.entity_to_idx.keys())
        top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

        # Get scores for the second atom
        # Initialize an empty tensor
        atom2_scores = torch.empty(0, len(model.entity_to_idx)).to(atom1_scores.device)

        # Get scores for the second atom
        for head2 in top_k_heads:
            # The score tensor for the current head2
            atom2_score = model.predict(h=[head2], r=[relation2])
            neg_atom2_score = negnorm(atom2_score, lambda_, neg_norm)
            # Concatenate the score tensor for the current head2 with the previous scores
            atom2_scores = torch.cat([atom2_scores,neg_atom2_score], dim=0)

        topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom2_scores.shape[1])

        inter_scores = t_norm(topk_scores1_expanded, atom2_scores, tnorm)

        scores_2pn_query, _ = torch.max(inter_scores, dim=0)
        scores_1p_query = model.predict(h=[head3], r=[relation3[0]]).squeeze()

        combined_scores = t_norm(scores_2pn_query, scores_1p_query, tnorm)

        entity_scores[query] = combined_scores

    return entity_scores

def evaluate(model, scores, easy_answers, hard_answers):
    # Calculate MRR considering the hard and easy answers
    total_mrr = 0
    total_h1 = 0
    total_h3 = 0
    total_h10 = 0
    num_queries = len(scores)

    for query,entity_score in scores.items():
        assert len(entity_score) == len(model.entity_to_idx)
        entity_scores = [(ei, s) for ei, s in zip(model.entity_to_idx.keys(), entity_score)]
        entity_scores = sorted(entity_scores, key=lambda x: x[1], reverse=True)

        # Extract corresponding easy and hard answers
        easy_ans=easy_answers[query]
        hard_ans=hard_answers[query]
        easy_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in easy_ans]
        hard_answer_indices = [idx for idx, (entity, _) in enumerate(entity_scores) if entity in hard_ans]

        answer_indices = easy_answer_indices + hard_answer_indices


        cur_ranking = np.array(answer_indices)

        # Sort by position in the ranking; indices for (easy + hard) answers
        cur_ranking, indices = np.sort(cur_ranking), np.argsort(cur_ranking)
        num_easy = len(easy_ans)
        num_hard = len(hard_ans)

        # Indices with hard answers only
        masks = indices >= num_easy

        # Reduce ranking for each answer entity by the amount of (easy+hard) answers appearing before it
        answer_list = np.arange(num_hard + num_easy, dtype=float)
        cur_ranking = cur_ranking - answer_list + 1

        # Only take indices that belong to the hard answers
        cur_ranking = cur_ranking[masks]
        # print(cur_ranking)
        mrr = np.mean(1.0 / cur_ranking)
        h1 = np.mean((cur_ranking <= 1).astype(float))
        h3 = np.mean((cur_ranking <= 3).astype(float))
        h10 = np.mean((cur_ranking <= 10).astype(float))
        total_mrr += mrr
        total_h1 += h1
        total_h3 += h3
        total_h10 += h10
    # average for all queries of a type
    avg_mrr = total_mrr / num_queries
    avg_h1 = total_h1 / num_queries
    avg_h3 = total_h3 / num_queries
    avg_h10 = total_h10 / num_queries

    return avg_mrr, avg_h1, avg_h3, avg_h10

@click.command()
@click.option('--datapath',type=str, default="/Users/sourabh/dice-embeddings/KGs/UMLS")
@click.option('--experiment', default="/Users/sourabh/dice-embeddings/Experiments/2023-08-06 01-51-31.514483",help='pre trained model experiment')
@click.option('--tnorm', type=click.Choice(['min', 'prod']), default='prod',
              help='triangular norm and conorm to be used')
@click.option('--neg_norm', type=click.Choice(['standard', 'sugeno','yager']), default='yager',
              help='negation norm to be used')
@click.option('--k_', default=4, type= int,help='top k candidates for each query happen')
@click.option('--lambda_', default=0.4,type=float, help='lambda value for sugeno or yager negation, for sugeno norm lamda cannot be equal to 1')


# Add more functions for other types of query structures
def main(datapath: str, experiment, tnorm: str, neg_norm: str,lambda_: float,k_: int):
    logging.basicConfig(filename="complex-query-answering.log", filemode='a',  format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    #load pre trained neural link predictor
    model = KGE(experiment)
    logging.info(f"Model name: {model.name}")
    logging.info(f"Candidates for beam search: k = {k_}")
    logging.info(f"T-norm: {tnorm}")
    logging.info(f"Negation Norm: {neg_norm}")

    if neg_norm.lower() in ['sugeno', 'yager']:
        logging.info(f"Lambda: {lambda_}")

    tasks = (
                    # "1p",
                    "2p",
                    "3p",
                    "2i",
                    "3i",
                    "ip",
                    "pi",
                    "2in",
                    "3in",
                    "pin",
                    "pni",
                    "inp",
                    "2u",
                    "up",
                )

    queries, easy_answers, hard_answers = load_data(datapath, tasks)
    # Print model name, k, t-norm and neg-norm
    print(f"Model Name: {model.name}")
    print(f"K: {k_}")
    print(f"T-Norm: {tnorm}")
    print(f"Negation Norm: {neg_norm}")

    if neg_norm.lower() in ['sugeno', 'yager']:
        print(f"Lambda: {lambda_}")

    """Answering Complex Query types including negation following the CQD beam search"""
    #results = OrderedDict((name, {'Query Name': name}) for name in query_name_dict.values())
    metrics = {task: {"MRR": 0, "H1": 0, "H3": 0, "H10": 0} for task in tasks}

    for query_structure, query in queries.items():

        #negation query types
        #2in
        if query_structure == (("e", ("r",)), ("e", ("r", "n"))):
            entity_scores = scores_2in(model, query, tnorm, neg_norm, lambda_)

        #3in
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r","n"))):
            entity_scores = scores_3in(model, query, tnorm, neg_norm, lambda_)


        #pni
        elif query_structure == (("e", ("r", "r", "n")), ("e", ("r",))):
            entity_scores = scores_pni(model, query, tnorm, neg_norm, lambda_, k_)

        #pin
        elif query_structure == (("e", ("r", "r")), ("e", ("r", "n"))):
            entity_scores = scores_pin(model, query, tnorm, neg_norm, lambda_, k_)

        #inp
        elif query_structure == ((("e", ("r",)), ("e", ("r", "n"))), ("r",)):
            entity_scores = scores_inp(model, query, tnorm, neg_norm, lambda_, k_)


        # complex positive query types conjunction

        #2p
        elif query_structure == ("e", ("r", "r")):
            entity_scores = scores_2p(model, query, tnorm, k_)

        #3p
        elif query_structure == ("e", ("r", "r", "r",)):
            entity_scores = scores_3p(model, query, tnorm, k_)

        #2i
        elif query_structure == (("e", ("r",)), ("e", ("r",))):
            entity_scores = scores_2i(model, query, tnorm)

        #3i
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r",))):
            entity_scores = scores_3i(model, query, tnorm)

        #pi
        elif query_structure == (("e", ("r", "r")), ("e", ("r",))):
            entity_scores = scores_pi(model, query, tnorm, k_)

        #ip
        elif query_structure == ((("e", ("r",)), ("e", ("r",))), ("r",)):
            entity_scores = scores_ip(model, query, tnorm, k_)


        #disjunction
        #2u
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("u",)):
            entity_scores = scores_2u(model, query, tnorm)

        #up
        # here the second tnorm is for t-conorm (used in pairs)
        elif query_structure == ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)):
            entity_scores = scores_up(model, query, tnorm, tnorm, k_)

        mrr, h1, h3, h10 = evaluate(model, entity_scores, easy_answers, hard_answers)
        query_name = query_name_dict[query_structure]
        logging.info(
            f"{query_name}, {query_structure}, MRR={mrr:.5f}, H1={h1:.5f}, H3={h3:.5f}, H10={h10:.5f}"
        )




        metrics[query_name]["MRR"] = mrr
        metrics[query_name]["H1"] = h1
        metrics[query_name]["H3"] = h3
        metrics[query_name]["H10"] = h10


    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Query Name")
    table.add_column("MRR")
    table.add_column("H1")
    table.add_column("H3")
    table.add_column("H10")

    # Add rows to the table
    for task, values in metrics.items():
        table.add_row(task, *[f"{value:.5f}" for value in values.values()])

    console.print(table)

if __name__ == '__main__':
    main()