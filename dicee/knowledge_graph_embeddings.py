import logging
import sys
import traceback
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from .abstracts import BaseInteractiveKGE, BaseInteractiveTrainKGE, InteractiveQueryDecomposition
from .evaluation._filtering import evaluation_tie_options
from .evaluation.link_prediction import evaluate_lp
from .static_funcs import load_pickle

logger = logging.getLogger(__name__)


class KGE(BaseInteractiveKGE, InteractiveQueryDecomposition, BaseInteractiveTrainKGE):
    """ Knowledge Graph Embedding Class for interactive usage of pre-trained models"""

    def __init__(self, path=None, url=None, construct_ensemble=False,
                 model_name=None):
        super().__init__(path=path, url=url, construct_ensemble=construct_ensemble, model_name=model_name)
        # Only check base relations (those without "_inverse" suffix) for their inverse counterparts
        if hasattr(self, 'relation_to_idx'):
            base_relations = [rel for rel in self.relation_to_idx.keys() if not rel.endswith("_inverse")]
            self.all_have_inverse = all(f"{rel}_inverse" in self.relation_to_idx for rel in base_relations)
        else:
            # For BPE models, we don't have explicit relation mappings
            self.all_have_inverse = False
    def __str__(self):
        return "KGE | " + str(self.model)

    def to(self, device: str) -> None:
        if "cpu" not in device and "cuda" not in device:
            raise ValueError(f"Device must be either cpu or cuda, got {device!r}")
        self.clear_query_cache()
        self.model.to(device)

    def clear_query_cache(self):
        """Release retained CQD rows and their model references."""
        self._query_engine = None
        self._query_engine_key = None

    def get_transductive_entity_embeddings(self,
                                           indices: Union[torch.LongTensor, List[str]],
                                           as_pytorch=False,
                                           as_numpy=False,
                                           as_list=True) -> Union[torch.FloatTensor, np.ndarray, List[float]]:

        if isinstance(indices, torch.LongTensor):
            """ Do nothing"""
        else:
            if not isinstance(indices, list):
                raise TypeError(f"indices must be either torch.LongTensor or list of strings, got {indices}")
            indices = torch.LongTensor([self.entity_to_idx[i] for i in indices])

        if as_pytorch:
            return self.model.entity_embeddings(indices)
        elif as_numpy:
            return self.model.entity_embeddings(indices).numpy
        elif as_list:
            return self.model.entity_embeddings(indices).tolist()
        else:
            raise RuntimeError("Something went wrong with the types")

    def create_vector_database(self, collection_name: str, distance: str,
                               location: str = "localhost",
                               port: int = 6333):
        if distance not in ["cosine", "dot"]:
            raise ValueError(f"distance must be one of ['cosine', 'dot'], got {distance!r}")
        # lazy imports
        try:
            from qdrant_client import QdrantClient
        except ModuleNotFoundError:
            traceback.print_exc()
            logger.error("Please install qdrant_client: pip install qdrant_client")
            exit(1)

        from qdrant_client.http.models import Distance, PointStruct, VectorParams
        # from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        client = QdrantClient(location=location, port=port)
        # If the collection is not created, create it
        if collection_name in [i.name for i in client.get_collections().collections]:
            logger.info(f"Deleting existing collection {collection_name}")
            client.delete_collection(collection_name=collection_name)

        logger.info(f"Creating a collection {collection_name} with distance metric:Cosine")
        client.create_collection(collection_name=collection_name,
                                 vectors_config=VectorParams(size=self.model.embedding_dim, distance=Distance.COSINE))

        entities = list(self.idx_to_entity.values())
        logger.info("Fetching entity embeddings..")
        vectors = self.get_transductive_entity_embeddings(indices=entities, as_list=True)
        logger.info("Indexing....")
        points = []
        for str_ent, vec in zip(entities, vectors):
            points.append(PointStruct(id=self.entity_to_idx[str_ent],
                                      vector=vec, payload={"name": str_ent}))
        operation_info = client.upsert(collection_name=collection_name, wait=True,
                                       points=points)
        logger.info(operation_info)

    def generate(self, h="", r=""):
        if not self.configs["byte_pair_encoding"]:
            raise ValueError("generate() requires a model trained with byte_pair_encoding=True")

        h_encode = self.enc.encode(h)
        r_encode = self.enc.encode(r)

        length = self.configs["max_length_subword_tokens"]

        if len(h_encode) != length:
            h_encode.extend([self.dummy_id for _ in range(length - len(h_encode))])

        if len(r_encode) != length:
            r_encode.extend([self.dummy_id for _ in range(length - len(r_encode))])

        h_encode = torch.LongTensor(h_encode).reshape(1, length)
        r_encode = torch.LongTensor(r_encode).reshape(1, length)
        # Initialize batch as all dummy ID
        X = torch.ones(self.enc.n_vocab, length) * self.dummy_id
        X = X.long()
        h_encode = h_encode.repeat_interleave(self.enc.n_vocab, dim=0)
        r_encode = r_encode.repeat_interleave(self.enc.n_vocab, dim=0)

        counter = 0
        pointer = 0
        tokens = [self.dummy_id for _ in range(length)]
        while counter != self.max_length_subword_tokens:
            X[:, pointer] = torch.arange(0, self.enc.n_vocab, dtype=int)

            x = torch.stack((h_encode, r_encode, X), dim=1)
            score, id_next_token = torch.max(self.model(x), dim=0)
            id_next_token = int(id_next_token)
            tokens[pointer] = id_next_token
            X[:, pointer] = id_next_token
            pointer += 1
            counter += 1
            logger.info(f"{self.enc.decode(tokens)}\t {score}")

    # given a string, return is bpe encoded embeddings
    def eval_lp_performance(self, dataset=List[Tuple[str, str, str]], filtered=True,
                            *, tie_policy=None, tie_seed=None):
        """Evaluate head/tail ranks, inheriting saved tie settings unless overridden.

        ``tie_policy`` accepts sort (legacy), optimistic, random, or pessimistic.
        ``tie_seed`` seeds an independent random stream for each evaluation.
        """
        if not isinstance(dataset, list) or len(dataset) == 0:
            raise TypeError("dataset must be a non-empty list of (head, relation, tail) triples")
        tie_options = evaluation_tie_options(self.configs)
        if tie_policy is not None:
            tie_options["tie_policy"] = tie_policy
        if tie_seed is not None:
            tie_options["tie_seed"] = tie_seed
        idx_dataset = np.array(
            [(self.entity_to_idx[s], self.relation_to_idx[p], self.entity_to_idx[o]) for s, p, o in dataset])
        if filtered:
            return evaluate_lp(model=self.model, triple_idx=idx_dataset, num_entities=len(self.entity_to_idx),
                               er_vocab=load_pickle(self.path + '/er_vocab.p'),
                               re_vocab=load_pickle(self.path + '/re_vocab.p'), **tie_options)
        else:
            return evaluate_lp(model=self.model, triple_idx=idx_dataset, num_entities=len(self.entity_to_idx),
                               er_vocab={(h, r): [] for h, r, _ in idx_dataset},
                               re_vocab={(r, t): [] for _, r, t in idx_dataset}, **tie_options)

    def predict_missing_head_entity(self, relation: Union[List[str], str], tail_entity: Union[List[str], str],
                                    within=None, batch_size = 2, topk = 1, return_indices = False) -> Tuple:
        r"""
        Given a relation and a tail entity, return top k ranked head entity.

        argmax_{e \in E } f(e,r,t), where r \in R, t \in E.

        Parameter
        ---------
        relation:  Union[List[str], str]

        String representation of selected relations.

        tail_entity: Union[List[str], str]

        String representation of selected entities.


        k: int

        Highest ranked k entities.

        Returns: Tuple
        ---------

        Highest K scores and entities
        """
        if self.all_have_inverse and within is None and not hasattr(self.model, "forward_k_vs_all_heads"):
            if isinstance(relation, str):
                relation = [f"{relation}_inverse"]
            else:
                relation = [f"{rel}_inverse" for rel in relation]
            return self.predict_missing_tail_entity(tail_entity, relation, within, batch_size, topk, return_indices)
        if isinstance(relation, list):
            relation = torch.LongTensor([self.relation_to_idx[i] for i in relation])
        else:
            relation = torch.LongTensor([self.relation_to_idx[relation]])
        if isinstance(tail_entity, list):
            tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity])
        else:
            tail_entity = torch.LongTensor([self.entity_to_idx[tail_entity]])

        head_entity = (torch.arange(len(self.entity_to_idx)) if within is None
                       else torch.tensor([self.entity_to_idx[e] for e in within], dtype=torch.long))
        # Generate all (tail, relation) pairs
        tr_pairs = torch.cartesian_prod(tail_entity, relation)  # Shape: (num_tr_pairs, 2)
        num_tr_pairs = tr_pairs.size(0)
        H = head_entity.size(0)

        if return_indices:
            # For predict_topk: store only top-k scores and indices
            scores = torch.zeros(num_tr_pairs, topk)  # Pre-allocate score tensor
            indices = torch.zeros(num_tr_pairs, topk, dtype=torch.long)  # Pre-allocate indices tensor
        else:
            # For predict: store all entity scores
            scores = torch.zeros(num_tr_pairs * H)  # Pre-allocate scores

        # Process in batches of (t, r) pairs
        batch_size_tr = batch_size  # Adjust batch_size to control memory usage
        device = self.model.device

        for i in range(0, num_tr_pairs, batch_size_tr):
            batch_tr = tr_pairs[i:i + batch_size_tr]  # Current batch of (t, r)
            t_batch = batch_tr[:, 0]
            r_batch = batch_tr[:, 1]
            B = t_batch.size(0)

            # Compute scores and store
            if hasattr(self.model, "forward_k_vs_all_heads"):
                batch_scores = self.model.forward_k_vs_all_heads(
                    torch.stack((r_batch, t_batch), 1).to(device), head_entity.to(device))
            else:
                h = head_entity.repeat(B).to(device)
                r = r_batch.repeat_interleave(H).to(device)
                t = t_batch.repeat_interleave(H).to(device)
                triples = torch.stack([h, r, t], dim=1)
                batch_scores = self.model(triples).view(B, H)

            if return_indices:
                # Store top-k scores and indices
                topk_scores, topk_idxs = torch.topk(batch_scores, topk, dim=1)
                scores[i:i + batch_size_tr, :] = topk_scores
                indices[i:i + batch_size_tr, :] = head_entity[topk_idxs.cpu()]
            else:
                # Store all scores
                start_idx = i * H
                end_idx = start_idx + B * H
                scores[start_idx:end_idx] = batch_scores.flatten()

        if return_indices:
            return scores.flatten(), indices.flatten()
        else:
            return scores

    def predict_missing_relations(self, head_entity: Union[List[str], str],
                                  tail_entity: Union[List[str], str], within=None, batch_size = 2, topk = 1, return_indices = False) -> Tuple:
        r"""
        Given a head entity and a tail entity, return top k ranked relations.

        argmax_{r \in R } f(h,r,t), where h, t \in E.


        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        tail_entity: List[str]

        String representation of selected entities.


        k: int

        Highest ranked k entities.

        Returns: Tuple
        ---------

        Highest K scores and entities
        """
        relation = torch.arange(0, len(self.relation_to_idx))
        if isinstance(head_entity, list):
            head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity])
        else:
            head_entity = torch.LongTensor([self.entity_to_idx[head_entity]])
        if isinstance(tail_entity, list):
            tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity])
        else:
            tail_entity = torch.LongTensor([self.entity_to_idx[tail_entity]])

        # Generate all (head, tail) pairs
        ht_pairs = torch.cartesian_prod(head_entity, tail_entity)  # Shape: (num_ht_pairs, 2)
        num_ht_pairs = ht_pairs.size(0)
        R = relation.size(0)

        if return_indices:
            # For predict_topk: store only top-k scores and indices
            scores = torch.zeros(num_ht_pairs, topk)  # Pre-allocate score tensor
            indices = torch.zeros(num_ht_pairs, topk, dtype=torch.long)  # Pre-allocate indices tensor
        else:
            # For predict: store all relation scores
            scores = torch.zeros(num_ht_pairs * R)  # Pre-allocate score tensor

        batch_size_ht = batch_size
        device = self.model.device

        for i in range(0, num_ht_pairs, batch_size_ht):
            batch_ht = ht_pairs[i:i + batch_size_ht]
            h_batch = batch_ht[:, 0]
            t_batch = batch_ht[:, 1]
            B = h_batch.size(0)

            if hasattr(self.model, "forward_k_vs_all_relations"):
                batch_scores = self.model.forward_k_vs_all_relations(batch_ht.to(device)).cpu()
            else:
                # Generate triples (h, r, t) for triple-scoring models.
                h = h_batch.repeat_interleave(R).to(device)
                r = relation.repeat(B).to(device)
                t = t_batch.repeat_interleave(R).to(device)
                triples = torch.stack([h, r, t], dim=1)
                batch_scores = self.model(triples).view(B, R)

            if return_indices:
                # Store top-k scores and indices
                topk_scores, topk_idxs = torch.topk(batch_scores, topk, dim=1)
                scores[i:i + batch_size_ht, :] = topk_scores
                indices[i:i + batch_size_ht, :] = topk_idxs
            else:
                # Store all scores
                start_idx = i * R
                end_idx = start_idx + B * R
                scores[start_idx:end_idx] = batch_scores.flatten()

        if return_indices:
            return scores.flatten(), indices.flatten()
        else:
            return scores

    def predict_missing_tail_entity(self, head_entity: Union[List[str], str],
                                    relation: Union[List[str], str], within: Optional[List[str]] = None, batch_size = 2, topk = 1, return_indices = False) -> torch.FloatTensor:
        r"""
        Given a head entity and a relation, return top k ranked entities

        argmax_{e \in E } f(h,r,e), where h \in E and r \in R.


        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        tail_entity: List[str]

        String representation of selected entities.

        Returns: Tuple
        ---------

        scores
        """
        if within is not None:
            h_encode = self.enc.encode(head_entity[0])
            r_encode = self.enc.encode(relation[0])
            t_encode = self.enc.encode_batch(within)
            length = self.configs["max_length_subword_tokens"]

            num_entities = len(within)
            if len(h_encode) != length:
                h_encode.extend([self.dummy_id for _ in range(length - len(h_encode))])

            if len(r_encode) != length:
                r_encode.extend([self.dummy_id for _ in range(length - len(r_encode))])

            if len(t_encode) != length:
                for i in range(len(t_encode)):
                    t_encode[i].extend([self.dummy_id for _ in range(length - len(t_encode[i]))])

            h_encode = torch.LongTensor(h_encode).unsqueeze(0)
            r_encode = torch.LongTensor(r_encode).unsqueeze(0)
            t_encode = torch.LongTensor(t_encode)

            x = torch.stack((torch.repeat_interleave(input=h_encode, repeats=num_entities, dim=0),
                             torch.repeat_interleave(input=r_encode, repeats=num_entities, dim=0),
                             t_encode), dim=1)
            return self.model(x)
        else:
            tail_entity = torch.arange(0, len(self.entity_to_idx))
            if isinstance(head_entity, list):
                head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity])
            else:
                head_entity = torch.LongTensor([self.entity_to_idx[head_entity]])
            if isinstance(relation, list):
                relation = torch.LongTensor([self.relation_to_idx[i] for i in relation])
            else:
                relation = torch.LongTensor([self.relation_to_idx[relation]])

            # Generate all (head, relation) pairs
            hr_pairs = torch.cartesian_prod(head_entity, relation)  # Shape: (num_hr_pairs, 2)
            num_hr_pairs = hr_pairs.size(0)
            T = tail_entity.size(0)

            if return_indices:
                # For predict_topk: store only top-k scores and indices
                scores = torch.zeros(num_hr_pairs, topk)  # Pre-allocate score tensor
                indices = torch.zeros(num_hr_pairs, topk, dtype=torch.long)  # Pre-allocate indices tensor
            else:
                # For predict: store all entity scores
                scores = torch.zeros(num_hr_pairs * T)  # Flat tensor for all scores

            # Process in batches
            batch_size_hr = batch_size  # Adjust as needed
            device = self.model.device

            for i in range(0, num_hr_pairs, batch_size_hr):
                batch_hr = hr_pairs[i:i + batch_size_hr]  # Current batch of (h, r)
                batch_hr = batch_hr.to(device)
                B = batch_hr.size(0)


                # Compute scores and store
                batch_scores = self.model(batch_hr).view(B, T)

                if return_indices:
                    # Store top-k scores and indices
                    topk_scores, topk_idxs = torch.topk(batch_scores, topk, dim=1)
                    scores[i:i + batch_size_hr, :] = topk_scores
                    indices[i:i + batch_size_hr, :] = topk_idxs
                else:
                    # Store all scores
                    start_idx = i * T
                    end_idx = start_idx + B * T
                    scores[start_idx:end_idx] = batch_scores.flatten()

        if return_indices:
            return scores.flatten(), indices.flatten()
        else:
            return scores

    def predict(self, *, h: Optional[Union[List[str], str]] = None,
                r: Optional[Union[List[str], str]] = None,
                t: Optional[Union[List[str], str]] = None,
                within: Optional[List[str]] = None,
                logits: bool = True) -> torch.FloatTensor:
        """
        Predict scores for triples or missing triple elements.

        Args:
            h: Head entity/entities. None to predict heads.
            r: Relation/relations. None to predict relations.
            t: Tail entity/entities. None to predict tails.
            within: Optional list of entities to restrict predictions to.
            logits: If True, return raw scores. If False, return sigmoid scores (0-1).

        Returns:
            torch.FloatTensor of scores. Shape depends on the query type:
            - Single triple (h, r, t): scalar score
            - Missing element: vector of all possible scores

        Raises:
            TypeError: If inputs are not strings or lists of strings.
            ValueError: If a required argument for the query type is missing.

        Examples:
            >>> # Score a specific triple
            >>> model.predict(h="Mongolia", r="isLocatedIn", t="Asia", logits=False)
            tensor(0.9523)

            >>> # Get scores for all possible tail entities
            >>> model.predict(h="Mongolia", r="isLocatedIn", t=None)
            tensor([0.21, 0.95, 0.03, ...])  # One score per entity
        """
        # (1) Sanity checking.
        if h is not None:
            if not isinstance(h, (list, str)):
                raise TypeError(f"h must be a str or list of str, got {type(h)}")
            if not isinstance(h[0], str):
                raise TypeError(f"h must contain str entities, got {type(h[0])}")
        if r is not None:
            if not isinstance(r, (list, str)):
                raise TypeError(f"r must be a str or list of str, got {type(r)}")
            if not isinstance(r[0], str):
                raise TypeError(f"r must contain str relations, got {type(r[0])}")
        if t is not None:
            if not isinstance(t, (list, str)):
                raise TypeError(f"t must be a str or list of str, got {type(t)}")
            if not isinstance(t[0], str):
                raise TypeError(f"t must contain str entities, got {type(t[0])}")

        # (2) Predict missing head entity given a relation and a tail entity.
        if h is None:
            if r is None or t is None:
                raise ValueError("r and t must both be provided when predicting a missing head entity")
            # ? r, t
            scores = self.predict_missing_head_entity(r, t, within, batch_size=2, topk=len(self.entity_to_idx), return_indices=False)
        # (3) Predict missing relation given a head entity and a tail entity.
        elif r is None:
            if h is None or t is None:
                raise ValueError("h and t must both be provided when predicting a missing relation")
            # h ? t
            scores = self.predict_missing_relations(h, t, within, batch_size=2, topk=len(self.relation_to_idx), return_indices=False)
        # (4) Predict missing tail entity given a head entity and a relation
        elif t is None:
            if h is None or r is None:
                raise ValueError("h and r must both be provided when predicting a missing tail entity")
            # h r ?
            scores = self.predict_missing_tail_entity(h, r, within, batch_size=2, topk=len(self.entity_to_idx), return_indices=False)
        else:
            scores = self.triple_score(h, r, t, logits=True)

        if logits:
            return scores
        else:
            return torch.sigmoid(scores)

    def predict_topk(
        self,
        *,
        h: Optional[Union[str, List[str]]] = None,
        r: Optional[Union[str, List[str]]] = None,
        t: Optional[Union[str, List[str]]] = None,
        topk: int = 10,
        within: Optional[List[str]] = None,
        batch_size: int = 1024
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Predict top-k missing items in a given triple pattern.

        Args:
            h: Head entity/entities. None to predict heads.
            r: Relation/relations. None to predict relations.
            t: Tail entity/entities. None to predict tails.
            topk: Number of top predictions to return.
            within: Optional list of entities to restrict predictions to.
            batch_size: Batch size for processing multiple queries.

        Returns:
            For single query: List[(item, score), ...] of length topk.
            For batch query: List of such lists, one per query.

        Raises:
            TypeError: If h, r, or t is not a str or list of str.
            ValueError: If the required arguments for a query type are None.

        Examples:
            >>> model.predict_topk(h=["Mongolia"], r=["isLocatedIn"], topk=3)
            [('Asia', 0.99), ('Europe', 0.02), ...]

            >>> model.predict_topk(r=["isLocatedIn"], t=["Asia"], topk=5)
            [('Mongolia', 0.85), ('China', 0.82), ...]
        """

        # (1) Sanity checking
        if h is not None and not isinstance(h, (list, str)):
            raise TypeError(f"h must be a str or list of str, got {type(h)}")
        if r is not None and not isinstance(r, (list, str)):
            raise TypeError(f"r must be a str or list of str, got {type(r)}")
        if t is not None and not isinstance(t, (list, str)):
            raise TypeError(f"t must be a str or list of str, got {type(t)}")

        # --- Missing HEAD: (?, r, t) ---
        if h is None:
            if r is None or t is None:
                raise ValueError("r and t must both be provided when predicting a missing head entity")
            # Convert input to lists if they're strings
            if isinstance(r, str):
                r = [r]
            if isinstance(t, str):
                t = [t]
            flat_scores, flat_indices = self.predict_missing_head_entity(r, t, within, batch_size, topk, return_indices=True)
            num_rt_pairs = len(r) * len(t)

            # Reshape to (num_rt_pairs, topk)
            scores_2d = flat_scores.view(num_rt_pairs, topk)
            indices_2d = flat_indices.view(num_rt_pairs, topk)

            # Convert to the expected format
            topk_scores = torch.sigmoid(scores_2d).tolist()
            topk_idxs = indices_2d.tolist()
            lookup = self.idx_to_entity

            all_results = [
                [(lookup[idx], score) for idx, score in zip(row_idxs, row_scores)]
                for row_idxs, row_scores in zip(topk_idxs, topk_scores)
            ]
            return all_results

        # --- Missing RELATION: (h, ?, t) ---
        elif r is None:
            if h is None or t is None:
                raise ValueError("h and t must both be provided when predicting a missing relation")
            flat_scores, flat_indices = self.predict_missing_relations(h, t, within, batch_size, topk, return_indices=True)

            # Convert input to lists if they're strings
            if isinstance(h, str):
                h = [h]
            if isinstance(t, str):
                t = [t]

            num_ht_pairs = len(h) * len(t)

            # Reshape to (num_ht_pairs, topk)
            scores_2d = flat_scores.view(num_ht_pairs, topk)
            indices_2d = flat_indices.view(num_ht_pairs, topk)

            # Convert to the expected format
            topk_scores = torch.sigmoid(scores_2d).tolist()
            topk_idxs = indices_2d.tolist()
            lookup = self.idx_to_relations

            all_results = [
                [(lookup[idx], score) for idx, score in zip(row_idxs, row_scores)]
                for row_idxs, row_scores in zip(topk_idxs, topk_scores)
            ]
            return all_results

        # --- Missing TAIL: (h, r, ?) ---
        elif t is None:
            if h is None or r is None:
                raise ValueError("h and r must both be provided when predicting a missing tail entity")

            # predict_missing_tail_entity now returns both scores and indices
            flat_scores, flat_indices = self.predict_missing_tail_entity(h, r, within, batch_size, topk, return_indices=True)

            # Convert input to lists if they're strings
            if isinstance(h, str):
                h = [h]
            if isinstance(r, str):
                r = [r]

            num_hr_pairs = len(h) * len(r)

            # Reshape to (num_hr_pairs, topk)
            scores_2d = flat_scores.view(num_hr_pairs, topk)
            indices_2d = flat_indices.view(num_hr_pairs, topk)

            # Convert to the expected format
            topk_scores = torch.sigmoid(scores_2d).tolist()
            topk_idxs = indices_2d.tolist()
            lookup = self.idx_to_entity

            all_results = [
                [(lookup[idx], score) for idx, score in zip(row_idxs, row_scores)]
                for row_idxs, row_scores in zip(topk_idxs, topk_scores)
            ]

            return all_results
        else:
            raise AttributeError('Use triple_score method')

    def triple_score(self, h: Optional[Union[List[str], str]] = None, r: Optional[Union[List[str], str]] = None,
                     t: Optional[Union[List[str], str]] = None, logits=False) -> torch.FloatTensor:
        """
        Predict triple score

        Parameter
        ---------
        head_entity: List[str]

        String representation of selected entities.

        relation: List[str]

        String representation of selected relations.

        tail_entity: List[str]

        String representation of selected entities.

        logits: bool

        If logits is True, unnormalized score returned

        Returns: Tuple
        ---------

        pytorch tensor of triple score
        """

        if self.configs.get("byte_pair_encoding", None):
            h_encode = self.enc.encode(h)
            r_encode = self.enc.encode(r)
            t_encode = self.enc.encode(t)

            length = self.configs["max_length_subword_tokens"]

            if len(h_encode) != length:
                h_encode.extend([self.dummy_id for _ in range(length - len(h_encode))])

            if len(r_encode) != length:
                r_encode.extend([self.dummy_id for _ in range(length - len(r_encode))])

            if len(t_encode) != length:
                t_encode.extend([self.dummy_id for _ in range(length - len(t_encode))])

            h_encode = torch.LongTensor(h_encode).reshape(1, length)
            r_encode = torch.LongTensor(r_encode).reshape(1, length)
            t_encode = torch.LongTensor(t_encode).reshape(1, length)
            x = torch.cat((h_encode, r_encode, t_encode), dim=0)
            x = torch.unsqueeze(x, dim=0)
        else:
            if isinstance(h, list) and isinstance(r, list) and isinstance(t, list):
                h = torch.LongTensor([self.entity_to_idx[i] for i in h]).reshape(len(h), 1)
                r = torch.LongTensor([self.relation_to_idx[i] for i in r]).reshape(len(r), 1)
                t = torch.LongTensor([self.entity_to_idx[i] for i in t]).reshape(len(t), 1)
            else:
                h = torch.LongTensor([self.entity_to_idx[h]]).reshape(1, 1)
                r = torch.LongTensor([self.relation_to_idx[r]]).reshape(1, 1)
                t = torch.LongTensor([self.entity_to_idx[t]]).reshape(1, 1)
            x = torch.hstack((h, r, t))

        if self.apply_semantic_constraint:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                x = x.to(self.model.device)
                if logits:
                    return self.model(x)
                else:
                    return torch.sigmoid(self.model(x))

    def return_multi_hop_query_results(self, scores, k: int, only_scores):
        if only_scores:
            return scores
        order = torch.argsort(scores, descending=True, stable=True)[:k]
        names = {index: name for name, index in self.entity_to_idx.items()}
        return [(names[int(index)], scores[index]) for index in order]

    def single_hop_query_answering(self, query: tuple, only_scores: bool = True,
                                   k: Optional[int] = None, use_logits: bool = False):
        return self.answer_multi_hop_query('1p', query, k=len(self.entity_to_idx) if k is None else k,
                                          only_scores=only_scores, use_logits=use_logits)

    def answer_multi_hop_query(
        self, query_type=None, query=None, queries=None, tnorm="prod",
        neg_norm="standard", lambda_=0.0, k=10, only_scores=False,
        use_logits=False, *, beam_size=None, context=None, adapter=None,
        observed_mix=None, row_batch_size=8, cache_bytes=64 * 1024 * 1024,
        seed=0, samples=None, executor='cqd',
    ):
        """Answer any of the 14 standard positive/negated query shapes.

        All entity-prediction models use one evaluator, including ULTRA, TRIX,
        and Flock. Scores default to sigmoid memberships; ``use_logits=True``
        explicitly requests legacy raw-score algebra (without an adapter).
        ``k`` limits returned answers; ``beam_size`` controls existential search
        independently and defaults to max(1, k). ``only_scores`` returns all
        entity scores in numeric ID order. Ties use entity ID order.
        ``executor='qto'`` uses exact projections without a beam limit.

        Pass QueryContext for transductive observed-edge/feature support; graph
        models use their attached context. Pass QueryScoreAdapter for a learned
        transform. Context must never contain held-out evaluation answers.
        See docs/guides/query_adapters.md for fitting and loading adapters.
        """
        from .query_answering import QueryAnswerer
        from .query_answering._query import index_query

        if (query is None) == (queries is None):
            raise ValueError("Provide exactly one of 'query' or 'queries'")
        if type(k) is not int or k < 0:
            raise ValueError('k must be a nonnegative integer')
        key = (id(self.model), id(context), id(adapter), observed_mix, row_batch_size, seed, samples)
        if key != getattr(self, '_query_engine_key', None):
            self._query_engine = QueryAnswerer(self.model, context=context, adapter=adapter, observed_mix=observed_mix,
                                               row_batch_size=row_batch_size, cache_bytes=cache_bytes, seed=seed, samples=samples)
            self._query_engine_key = key
        engine = self._query_engine
        if cache_bytes < 0:
            raise ValueError('Cache size must be nonnegative')
        engine.cache_bytes = cache_bytes
        beam = max(1, k) if beam_size is None else beam_size
        names = {index: name for name, index in self.entity_to_idx.items()} if not only_scores else None

        def answer(item):
            indexed = index_query(query_type, item, self.entity_to_idx, self.relation_to_idx)
            scores = engine.predict(indexed, beam_size=beam, tnorm=tnorm, neg_norm=neg_norm,
                                    lambda_=lambda_, use_logits=use_logits, return_log_scores=not use_logits, executor=executor)
            if only_scores:
                return scores if use_logits else scores.exp()
            # Rank in log space before converting to memberships, avoiding underflow ties.
            order = torch.argsort(scores, descending=True, stable=True)[:k]
            values = scores if use_logits else scores.exp()
            return [(names[int(index)], values[index]) for index in order]

        return [answer(item) for item in queries] if queries is not None else answer(query)

    def find_missing_triples(self, confidence: float, entities: Optional[List[str]] = None, relations: Optional[List[str]] = None,
                             topk: int = 10,
                             at_most: int = sys.maxsize) -> Set:
        """
         Find missing triples

         Iterative over a set of entities E and a set of relation R : \forall e \\in E and \forall r \\in R f(e,r,x)
         Return (e,r,x)\not\\in G and  f(e,r,x) > confidence

        Parameter
        ---------
        confidence: float

        A threshold for an output of a sigmoid function given a triple.

        topk: int

        Highest ranked k item to select triples with f(e,r,x) > confidence .

        at_most: int

        Stop after finding at_most missing triples

        Returns: Set
        ---------

        {(e,r,x) | f(e,r,x) > confidence \\land (e,r,x) \not\\in G
        """

        if not (1.0 >= confidence >= 0.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")

        def select(items: List[str], item_mapping: Dict[str, int]) -> Iterable[Tuple[str, int]]:
            """
             Get selected entities and their indexes

            Parameter
            ---------
            items: list

            item_mapping: dict


            Returns: Iterable
            ---------

            """

            if items is None:
                return item_mapping.items()
            else:
                return ((i, item_mapping[i]) for i in items)

        extended_triples = set()
        logger.info(f'Number of entities:{len(self.entity_to_idx)} \t Number of relations:{len(self.relation_to_idx)}')

        # (5) Cartesian Product over entities and relations
        # (5.1) Iterate over entities
        logger.info('Finding missing triples..')
        for str_head_entity, idx_entity in select(entities, self.entity_to_idx):
            # (5.1) Iterate over relations
            for str_relation, idx_relation in select(relations, self.relation_to_idx):
                # (5.2) \forall e \in Entities store a tuple of scoring_func(head,relation,e) and e
                # (5.3.) Sort (5.2) and return top  tuples
                predictions = self.predict_topk(h=[str_head_entity], r=[str_relation], topk=topk)
                # (5.4) Iterate over 5.3
                for str_entity, predicted_score in predictions:
                    # (5.5) If score is less than 99% ignore it
                    if predicted_score < confidence:
                        break
                    else:
                        # (5.8) Remember it
                        extended_triples.add((str_head_entity, str_relation, str_entity))
                        logger.info(f'Number of found missing triples: {len(extended_triples)}')
                        if len(extended_triples) == at_most:
                            return extended_triples
                        # No need to store a large KG into memory
                        # /5.6) False if 0, otherwise 1
                        is_in = np.any(
                            np.all(self.train_set == [idx_entity, idx_relation, self.entity_to_idx[str_entity]],
                                   axis=1))
                        # (5.7) If (5.6) is true, ignore it
                        if is_in:
                            continue
                        else:
                            # (5.8) Remember it
                            extended_triples.add((str_head_entity, str_relation, str_entity))
                            logger.info(f'Number of found missing triples: {len(extended_triples)}')
                            if len(extended_triples) == at_most:
                                return extended_triples
        return extended_triples

    def predict_literals(
        self,
        entity: Optional[Union[List[str], str]] = None,
        attribute: Optional[Union[List[str], str]] = None,
        denormalize_preds: bool = True,
    ) -> np.ndarray:
        """Predicts literal values for given entities and attributes.

        Args:
            entity (Union[List[str], str]): Entity or list of entities to predict literals for.
            attribute (Union[List[str], str]): Attribute or list of attributes to predict literals for.
            denormalize_preds (bool): If True, denormalizes the predictions.
        Returns:

            numpy ndarray : Predictions for the given entities and attributes.
        """
        # sanity checking
        # Check if the literal model is trained or loaded
        if not hasattr(self, "literal_model") or self.literal_model is None:
            raise RuntimeError("Literal model is not trained or loaded.")

        # TODO :Should we initialize self.literal_model in __init__ ?
        # RS : Predict functions could also work with entity and attribute index

        if entity is None or attribute is None:
            raise RuntimeError("Entity and Attribute cannot be of type None")

        # Convert entity and attribute to list if they are a single string
        if isinstance(entity, str):
            entity = [entity]
        if isinstance(attribute, str):
            attribute = [attribute]

        # Validate that entity and attribute are lists of strings
        if not isinstance(entity, list) or not all(isinstance(e, str) for e in entity):
            raise TypeError(f"entity must be a str or list of str, got {entity}")
        if not isinstance(attribute, list) or not all(isinstance(a, str) for a in attribute):
            raise TypeError(f"attribute must be a str or list of str, got {attribute}")

        # Ensure entity and attribute lists are the same length
        if len(entity) != len(attribute):
            raise ValueError("Entity and attribute lists must be of equal length")

        # Convert entity and attribute names to their corresponding index tensor
        entity_idx = torch.LongTensor([self.entity_to_idx[i] for i in entity])
        attribute_idx = torch.LongTensor([self.data_property_to_idx[i] for i in attribute])


        # device allocation
        device = self.literal_model.device
        self.literal_model, entity_idx, attribute_idx = (
            self.literal_model.to(device),
            entity_idx.to(device),
            attribute_idx.to(device),
        )

        with torch.no_grad():
            predictions = self.literal_model(entity_idx, attribute_idx)

        # move predictions to cpu and convert to numpy
        predictions = predictions.cpu().numpy()
        if denormalize_preds:
            predictions = self.literal_dataset.denormalize(
                preds_norm=predictions,
                attributes=attribute,
                normalization_params=self.literal_dataset.normalization_params,
            )
        return predictions
