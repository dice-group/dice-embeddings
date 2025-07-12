from typing import List, Tuple, Set, Iterable, Dict, Union
import torch
from .abstracts import BaseInteractiveKGE, InteractiveQueryDecomposition, BaseInteractiveTrainKGE
from .static_funcs import random_prediction, deploy_triple_prediction, deploy_tail_entity_prediction, \
    deploy_relation_prediction, deploy_head_entity_prediction, load_pickle
from .static_funcs_training import evaluate_lp
import numpy as np
import sys
import traceback

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
        assert "cpu" in device or "cuda" in device, "Device must be either cpu or cuda"
        self.model.to(device)

    def get_transductive_entity_embeddings(self,
                                           indices: Union[torch.LongTensor, List[str]],
                                           as_pytorch=False,
                                           as_numpy=False,
                                           as_list=True) -> Union[torch.FloatTensor, np.ndarray, List[float]]:

        if isinstance(indices, torch.LongTensor):
            """ Do nothing"""
        else:
            assert isinstance(indices, list), f"indices must be either torch.LongTensor or list of strings{indices}"
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
        assert distance in ["cosine", "dot"]
        # lazy imports
        try:
            from qdrant_client import QdrantClient
        except ModuleNotFoundError:
            traceback.print_exc()
            print("Please install qdrant_client: pip install qdrant_client")
            exit(1)

        from qdrant_client.http.models import Distance, VectorParams
        from qdrant_client.http.models import PointStruct
        # from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        client = QdrantClient(location=location, port=port)
        # If the collection is not created, create it
        if collection_name in [i.name for i in client.get_collections().collections]:
            print("Deleting existing collection ", collection_name)
            client.delete_collection(collection_name=collection_name)

        print(f"Creating a collection {collection_name} with distance metric:Cosine")
        client.create_collection(collection_name=collection_name,
                                 vectors_config=VectorParams(size=self.model.embedding_dim, distance=Distance.COSINE))

        entities = list(self.idx_to_entity.values())
        print("Fetching entity embeddings..")
        vectors = self.get_transductive_entity_embeddings(indices=entities, as_list=True)
        print("Indexing....")
        points = []
        for str_ent, vec in zip(entities, vectors):
            points.append(PointStruct(id=self.entity_to_idx[str_ent],
                                      vector=vec, payload={"name": str_ent}))
        operation_info = client.upsert(collection_name=collection_name, wait=True,
                                       points=points)
        print(operation_info)

    def generate(self, h="", r=""):
        assert self.configs["byte_pair_encoding"]

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
            print(self.enc.decode(tokens), end=f"\t {score}\n")

    # given a string, return is bpe encoded embeddings
    def eval_lp_performance(self, dataset=List[Tuple[str, str, str]], filtered=True):
        assert isinstance(dataset, list) and len(dataset) > 0
        idx_dataset = np.array(
            [(self.entity_to_idx[s], self.relation_to_idx[p], self.entity_to_idx[o]) for s, p, o in dataset])
        if filtered:
            return evaluate_lp(model=self.model, triple_idx=idx_dataset, num_entities=len(self.entity_to_idx),
                               er_vocab=load_pickle(self.path + '/er_vocab.p'),
                               re_vocab=load_pickle(self.path + '/re_vocab.p'))
        else:
            return evaluate_lp(model=self.model, triple_idx=idx_dataset, num_entities=len(self.entity_to_idx),
                               er_vocab=None, re_vocab=None)

    def predict_missing_head_entity(self, relation: Union[List[str], str], tail_entity: Union[List[str], str],
                                    within=None, batch_size = 2, topk = 1, return_indices = False) -> Tuple:
        """
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
        if self.all_have_inverse:
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

        head_entity = torch.arange(0, len(self.entity_to_idx))
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
            
            # Generate triples (h, r, t) for this batch
            h = head_entity.repeat(B).to(device)  # h: [h0, h1..., hN, h0, h1..., ... (B times)]
            r = r_batch.repeat_interleave(H).to(device)
            t = t_batch.repeat_interleave(H).to(device)
            triples = torch.stack([h, r, t], dim=1)
            
            # Compute scores and store
            batch_scores = self.model(triples).view(B, H)
            
            if return_indices:
                # Store top-k scores and indices
                topk_scores, topk_idxs = torch.topk(batch_scores, topk, dim=1)
                scores[i:i + batch_size_tr, :] = topk_scores
                indices[i:i + batch_size_tr, :] = topk_idxs
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
        """
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

            # Generate triples (h, r, t)
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
                                    relation: Union[List[str], str], within: List[str] = None, batch_size = 2, topk = 1, return_indices = False) -> torch.FloatTensor:
        """
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

    def predict(self, *, h: Union[List[str], str] = None, r: Union[List[str], str] = None,
                t: Union[List[str], str] = None, within=None, logits=True) -> torch.FloatTensor:
        """

        Parameters
        ----------
        logits
        h
        r
        t
        within

        Returns
        -------

        """
        # (1) Sanity checking.
        if h is not None:
            assert isinstance(h, list) or isinstance(h, str)
            assert isinstance(h[0], str)
        if r is not None:
            assert isinstance(r, list) or isinstance(r, str)
            assert isinstance(r[0], str)
        if t is not None:
            assert isinstance(t, list) or isinstance(t, str)
            assert isinstance(t[0], str)

        # (2) Predict missing head entity given a relation and a tail entity.
        if h is None:
            assert r is not None
            assert t is not None
            # ? r, t
            scores = self.predict_missing_head_entity(r, t, within, batch_size=2, topk=len(self.entity_to_idx), return_indices=False)
        # (3) Predict missing relation given a head entity and a tail entity.
        elif r is None:
            assert h is not None
            assert t is not None
            # h ? t
            scores = self.predict_missing_relations(h, t, within, batch_size=2, topk=len(self.relation_to_idx), return_indices=False)
        # (4) Predict missing tail entity given a head entity and a relation
        elif t is None:
            assert h is not None
            assert r is not None
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
        h: Union[str, List[str]] = None,
        r: Union[str, List[str]] = None,
        t: Union[str, List[str]] = None,
        topk: int = 10,
        within: List[str] = None,
        batch_size: int = 1024
    ):
        """
        Predict missing item in a given triple.

        Returns:
            - If you query a single (h, r, ?) or (?, r, t) or (h, ?, t), returns List[(item, score)]
            - If you query a batch of B, returns List of B such lists.
        """

        # (1) Sanity checking
        if h is not None:
            assert isinstance(h, (list, str))
        if r is not None:
            assert isinstance(r, (list, str))
        if t is not None:
            assert isinstance(t, (list, str))

        # --- Missing HEAD: (?, r, t) ---
        if h is None:
            assert r is not None and t is not None
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
            assert h is not None and t is not None
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
            assert h is not None and r is not None
            
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

    def triple_score(self, h: Union[List[str], str] = None, r: Union[List[str], str] = None,
                     t: Union[List[str], str] = None, logits=False) -> torch.FloatTensor:
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

    def return_multi_hop_query_results(self, aggregated_query_for_all_entities, k: int, only_scores):
        # @TODO: refactor by torchargmax(aggregated_query_for_all_entities)
        if only_scores:
            return aggregated_query_for_all_entities
        # from idx obtain entity str
        return sorted([(ei, s) for ei, s in zip(self.entity_to_idx.keys(), aggregated_query_for_all_entities)],
                      key=lambda x: x[1], reverse=True)[:k]

    def single_hop_query_answering(self, query: tuple, only_scores: bool = True, k: int = None):
        h, r = query
        result = self.predict(h=h, r=r[0]).squeeze()
        if only_scores:
            """ do nothing"""
        else:
            query_score_of_all_entities = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), result)]
            result = sorted(query_score_of_all_entities, key=lambda x: x[1], reverse=True)[:k]
        return result

    def answer_multi_hop_query(self, query_type: str = None, query: Tuple[Union[str, Tuple[str, str]], ...] = None,
                               queries: List[Tuple[Union[str, Tuple[str, str]], ...]] = None, tnorm: str = "prod",
                               neg_norm: str = "standard", lambda_: float = 0.0, k: int = 10, only_scores=False) -> \
            List[Tuple[str, torch.Tensor]]:
        """
        # @TODO: Refactoring is needed
        # @TODO: Score computation for each query type should be done in a static function

        Find an answer set for EPFO queries including negation and disjunction

        Parameter
        ----------
        query_type: str
        The type of the query, e.g., "2p".

        query: Union[str, Tuple[str, Tuple[str, str]]]
        The query itself, either a string or a nested tuple.

        queries: List of Tuple[Union[str, Tuple[str, str]], ...]

        tnorm: str
        The t-norm operator.

        neg_norm: str
        The negation norm.

        lambda_: float
        lambda parameter for sugeno and yager negation norms

        k: int
        The top-k substitutions for intermediate variables.

        Returns
        -------
        List[Tuple[str, torch.Tensor]]
        Entities and corresponding scores sorted in the descening order of scores
        """

        if queries is not None:
            results = []
            for i in queries:
                assert query is None
                results.append(
                    self.answer_multi_hop_query(query_type=query_type, query=i, tnorm=tnorm, neg_norm=neg_norm,
                                                lambda_=lambda_, k=k, only_scores=only_scores))
            return results

        assert len(self.entity_to_idx) >= k >= 0

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

        # Create an inverse mapping
        inverse_query_name_dict = {v: k for k, v in query_name_dict.items()}

        # Look up the corresponding query_structure
        if query_type in inverse_query_name_dict:
            query_structure = inverse_query_name_dict[query_type]
        else:
            raise ValueError(f"Invalid query type: {query_type}")

        # 1p
        if query_structure == ("e", ("r",)):
            return self.single_hop_query_answering(query, only_scores, k)
        # 2p
        elif query_structure == ("e", ("r", "r",)):
            # ?M : \exist A. r1(e,A) \land r2(A,M)
            e, (r1, r2) = query
            top_k_scores1 = []
            atom2_scores = []
            # (1) Iterate over top k substitutes of A in the first hop query: r1(e,A) s.t. A<-a
            for top_k_entity, score_of_e_r1_a in self.answer_multi_hop_query(query_type="1p", query=(e, (r1,)),
                                                                             only_scores=False, tnorm=tnorm, k=k):
                # (1.1) Store scores of (e, r1, a) s.t. a is a substitute of A and a is a top ranked entity.
                top_k_scores1.append(score_of_e_r1_a)
                # (1.2) Compute scores for (a, r2, M): Replace predict with answer_multi_hop_query.
                atom2_scores.append(self.predict(h=top_k_entity, r=r2))
            # (2) k by E tensor
            atom2_scores = torch.vstack(atom2_scores)
            kk, E = atom2_scores.shape
            # Sanity checking
            assert k == kk
            # Top k scores for all replacement of A. torch.Size([k,1])
            top_k_scores1 = torch.FloatTensor(top_k_scores1).reshape(k, 1)
            # k x E
            top_k_scores1 = top_k_scores1.repeat(1, E)
            # E scores
            aggregated_query_for_all_entities, _ = torch.max(self.t_norm(top_k_scores1, atom2_scores, tnorm), dim=0)
            return self.return_multi_hop_query_results(aggregated_query_for_all_entities, k, only_scores)
        # 3p
        elif query_structure == ("e", ("r", "r", "r",)):
            head1, (relation1, relation2, relation3) = query
            top_k_scores1 = []
            atom_scores = []
            # (1) Iterate over top k substitutes of A in the first hop query: r1(e,A) s.t. A<-a
            for top_k_entity, score_of_e_r1_a in self.answer_multi_hop_query(query_type="2p",
                                                                             query=(head1, (relation1, relation2)),
                                                                             tnorm=tnorm,
                                                                             k=k):
                top_k_scores1.append(score_of_e_r1_a)
                # () Scores for all entities E
                atom_scores.append(self.predict(h=[top_k_entity], r=[relation3]))

            # (2) k by E tensor
            atom_scores = torch.vstack(atom_scores)
            kk, E = atom_scores.shape
            # Sanity checking
            assert k == kk
            # Top k scores for all replacement of A. torch.Size([k,1])
            top_k_scores1 = torch.FloatTensor(top_k_scores1).reshape(k, 1)
            # k x E
            top_k_scores1 = top_k_scores1.repeat(1, E)
            # E scores
            aggregated_query_for_all_entities, _ = torch.max(self.t_norm(top_k_scores1, atom_scores, tnorm), dim=0)
            return self.return_multi_hop_query_results(aggregated_query_for_all_entities, k, only_scores)

        # 2in
        elif query_structure == (("e", ("r",)), ("e", ("r", "n"))):
            # entity_scores = scores_2in(query, tnorm, neg_norm, lambda_)
            head1, relation1 = query[0]
            head2, relation2 = query[1]

            # Calculate entity scores for each query
            # Get scores for the first atom (positive)
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()
            # Get scores for the second atom (negative)
            # if neg_norm == "standard":
            predictions = self.predict(h=[head2], r=[relation2[0]]).squeeze()
            atom2_scores = self.negnorm(predictions, lambda_, neg_norm)

            assert len(atom1_scores) == len(self.entity_to_idx)

            combined_scores = self.t_norm(atom1_scores, atom2_scores, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # 3in
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))):
            # entity_scores = scores_3in(model, query, tnorm, neg_norm, lambda_)
            head1, relation1 = query[0]
            head2, relation2 = query[1]
            head3, relation3 = query[2]

            # Calculate entity scores for each query
            # Get scores for the first atom (positive)
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()
            # Get scores for the second atom (negative)
            # modelling standard negation (1-x)
            atom2_scores = self.predict(h=[head2], r=[relation2[0]]).squeeze()
            # Get scores for the third atom
            # if neg_norm == "standard":
            predictions = self.predict(h=[head3], r=[relation3[0]]).squeeze()
            atom3_scores = self.negnorm(predictions, lambda_, neg_norm)

            assert len(atom1_scores) == len(self.entity_to_idx)

            inter_scores = self.t_norm(atom1_scores, atom2_scores, tnorm)
            combined_scores = self.t_norm(inter_scores, atom3_scores, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # pni
        elif query_structure == (("e", ("r", "r", "n")), ("e", ("r",))):
            # entity_scores = scores_pni(model, query, tnorm, neg_norm, lambda_, k_)
            head1, (relation1, relation2, _) = query[0]
            head3, relation3 = query[1]
            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)
            # sort atom1_scores in descending order and get the top k entities indices
            top_k_scores1, top_k_indices = torch.topk(atom1_scores, k)

            # using model.entity_to_idx.keys() take the name of entities from topk heads 2
            entity_to_idx_keys = list(self.entity_to_idx.keys())
            top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

            # Get scores for the second atom
            # Initialize an empty tensor
            atom2_scores = torch.empty(0, len(self.entity_to_idx)).to(atom1_scores.device)

            # Get scores for the second atom
            for head2 in top_k_heads:
                # The score tensor for the current head2
                atom2_score = self.predict(h=[head2], r=[relation2])
                neg_atom2_score = self.negnorm(atom2_score, lambda_, neg_norm)
                # Concatenate the score tensor for the current head2 with the previous scores
                atom2_scores = torch.cat([atom2_scores, neg_atom2_score], dim=0)

            topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom2_scores.shape[1])

            inter_scores = self.t_norm(topk_scores1_expanded, atom2_scores, tnorm)

            scores_2pn_query, _ = torch.max(inter_scores, dim=0)
            scores_1p_query = self.predict(h=[head3], r=[relation3[0]]).squeeze()

            combined_scores = self.t_norm(scores_2pn_query, scores_1p_query, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # pin
        elif query_structure == (("e", ("r", "r")), ("e", ("r", "n"))):
            # entity_scores = scores_pin(model, query, tnorm, neg_norm, lambda_, k_)
            head1, (relation1, relation2) = query[0]
            head3, relation3 = query[1]
            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)

            # sort atom1_scores in descending order and get the top k entities indices
            top_k_scores1, top_k_indices = torch.topk(atom1_scores, k)

            # using model.entity_to_idx.keys() take the name of entities from topk heads 2
            entity_to_idx_keys = list(self.entity_to_idx.keys())
            top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

            # Initialize an empty tensor
            atom2_scores = torch.empty(0, len(self.entity_to_idx)).to(atom1_scores.device)

            # Get scores for the second atom
            for head2 in top_k_heads:
                # The score tensor for the current head2
                atom2_score = self.predict(h=[head2], r=[relation2])
                # Concatenate the score tensor for the current head2 with the previous scores
                atom2_scores = torch.cat([atom2_scores, atom2_score], dim=0)

            topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom2_scores.shape[1])

            inter_scores = self.t_norm(topk_scores1_expanded, atom2_scores, tnorm)

            scores_2p_query, _ = torch.max(inter_scores, dim=0)

            scores_1p_query = self.predict(h=[head3], r=[relation3[0]]).squeeze()
            # taking negation for the e,(r,n) part of query
            neg_scores_1p_query = self.negnorm(scores_1p_query, lambda_, neg_norm)
            combined_scores = self.t_norm(scores_2p_query, neg_scores_1p_query, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # inp
        elif query_structure == ((("e", ("r",)), ("e", ("r", "n"))), ("r",)):
            # entity_scores = scores_inp(model, query, tnorm, neg_norm, lambda_, k_)
            head1, relation1 = query[0][0]
            head2, relation2 = query[0][1]
            relation_1p = query[1]

            # Calculate entity scores for each query
            # Get scores for the first atom (positive)
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()
            # Get scores for the second atom (negative)
            # if neg_norm == "standard":
            predictions = self.predict(h=[head2], r=[relation2[0]]).squeeze()
            atom2_scores = self.negnorm(predictions, lambda_, neg_norm)

            assert len(atom1_scores) == len(self.entity_to_idx)

            scores_2in_query = self.t_norm(atom1_scores, atom2_scores, tnorm)

            # sort atom1_scores in descending order and get the top k entities indices
            top_k_scores1, top_k_indices = torch.topk(scores_2in_query, k)

            # using model.entity_to_idx.keys() take the name of entities from topk heads 2
            entity_to_idx_keys = list(self.entity_to_idx.keys())
            top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

            # Get scores for the second atom
            # Initialize an empty tensor
            atom3_scores = torch.empty(0, len(self.entity_to_idx)).to(scores_2in_query.device)

            # Get scores for the second atom
            for head3 in top_k_heads:
                # The score tensor for the current head2
                atom3_score = self.predict(h=[head3], r=[relation_1p[0]])
                # Concatenate the score tensor for the current head2 with the previous scores
                atom3_scores = torch.cat([atom3_scores, atom3_score], dim=0)

            topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom3_scores.shape[1])

            combined_scores = self.t_norm(topk_scores1_expanded, atom3_scores, tnorm)

            res, _ = torch.max(combined_scores, dim=0)
            if only_scores:
                return res
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), res)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # 2i
        elif query_structure == (("e", ("r",)), ("e", ("r",))):
            # entity_scores = scores_2i(model, query, tnorm)
            head1, relation1 = query[0]
            head2, relation2 = query[1]

            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()
            # Get scores for the second atom
            atom2_scores = self.predict(h=[head2], r=[relation2[0]]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)

            combined_scores = self.t_norm(atom1_scores, atom2_scores, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # 3i
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r",))):
            # entity_scores = scores_3i(model, query, tnorm)
            head1, relation1 = query[0]
            head2, relation2 = query[1]
            head3, relation3 = query[2]
            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()
            # Get scores for the second atom
            atom2_scores = self.predict(h=[head2], r=[relation2[0]]).squeeze()
            # Get scores for the third atom
            atom3_scores = self.predict(h=[head3], r=[relation3[0]]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)

            inter_scores = self.t_norm(atom1_scores, atom2_scores, tnorm)
            combined_scores = self.t_norm(inter_scores, atom3_scores, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # pi
        elif query_structure == (("e", ("r", "r")), ("e", ("r",))):
            # entity_scores = scores_pi(model, query, tnorm, k_)
            head1, (relation1, relation2) = query[0]
            head3, relation3 = query[1]
            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)
            # sort atom1_scores in descending order and get the top k entities indices
            top_k_scores1, top_k_indices = torch.topk(atom1_scores, k)

            # using model.entity_to_idx.keys() take the name of entities from topk heads 2
            entity_to_idx_keys = list(self.entity_to_idx.keys())
            top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

            # Initialize an empty tensor
            atom2_scores = torch.empty(0, len(self.entity_to_idx)).to(atom1_scores.device)

            # Get scores for the second atom
            for head2 in top_k_heads:
                # The score tensor for the current head2
                atom2_score = self.predict(h=[head2], r=[relation2]).unsqueeze(0)
                # Concatenate the score tensor for the current head2 with the previous scores
                atom2_scores = torch.cat([atom2_scores, atom2_score], dim=0)

            topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom2_scores.shape[1])

            inter_scores = self.t_norm(topk_scores1_expanded, atom2_scores, tnorm)

            scores_2p_query, _ = torch.max(inter_scores, dim=0)

            scores_1p_query = self.predict(h=[head3], r=[relation3[0]]).squeeze()

            combined_scores = self.t_norm(scores_2p_query, scores_1p_query, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # ip
        elif query_structure == ((("e", ("r",)), ("e", ("r",))), ("r",)):
            # entity_scores = scores_ip(model, query, tnorm, k_)
            head1, relation1 = query[0][0]
            head2, relation2 = query[0][1]
            relation_1p = query[1]
            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()
            # Get scores for the second atom
            atom2_scores = self.predict(h=[head2], r=[relation2[0]]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)

            scores_2i_query = self.t_norm(atom1_scores, atom2_scores, tnorm)
            # Get the top k entities from the 2i query

            # sort atom1_scores in descending order and get the top k entities indices
            top_k_scores1, top_k_indices = torch.topk(scores_2i_query, k)

            # using model.entity_to_idx.keys() take the name of entities from topk heads
            entity_to_idx_keys = list(self.entity_to_idx.keys())
            top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

            # Get scores for the second atom
            # Initialize an empty tensor
            atom3_scores = torch.empty(0, len(self.entity_to_idx)).to(scores_2i_query.device)

            # Get scores for the second atom
            for head3 in top_k_heads:
                # The score tensor for the current head2
                atom3_score = self.predict(h=[head3], r=[relation_1p[0]]).unsqueeze(0)

                # Concatenate the score tensor for the current head2 with the previous scores
                atom3_scores = torch.cat([atom3_scores, atom3_score], dim=0)

            topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom3_scores.shape[1])

            combined_scores = self.t_norm(topk_scores1_expanded, atom3_scores, tnorm)
            res, _ = torch.max(combined_scores, dim=0)
            if only_scores:
                return res
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), res)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        # disjunction
        # 2u
        elif query_structure == (("e", ("r",)), ("e", ("r",)), ("u",)):
            # entity_scores = scores_2u(model, query, tnorm)
            head1, relation1 = query[0]
            head2, relation2 = query[1]

            # Calculate entity scores for each query
            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()
            # Get scores for the second atom
            atom2_scores = self.predict(h=[head2], r=[relation2[0]]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)

            combined_scores = self.t_conorm(atom1_scores, atom2_scores, tnorm)
            if only_scores:
                return combined_scores
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), combined_scores)]
            entity_scores = sorted(entity_scores, key=lambda x: x[1], reverse=True)

            return entity_scores
        # up
        # here the second tnorm is for t-conorm (used in pairs)
        elif query_structure == ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)):
            # entity_scores = scores_up(model, query, tnorm, tnorm, k_)
            head1, relation1 = query[0][0]
            head2, relation2 = query[0][1]
            relation_1p = query[1]

            # Get scores for the first atom
            atom1_scores = self.predict(h=[head1], r=[relation1[0]]).squeeze()

            # Get scores for the second atom
            atom2_scores = self.predict(h=[head2], r=[relation2[0]]).squeeze()

            assert len(atom1_scores) == len(self.entity_to_idx)

            scores_2u_query = self.t_conorm(atom1_scores, atom2_scores, tnorm)

            # Sort atom1_scores in descending order and get the top k entities indices
            top_k_scores1, top_k_indices = torch.topk(scores_2u_query, k)

            # Using model.entity_to_idx.keys() take the name of entities from topk heads
            entity_to_idx_keys = list(self.entity_to_idx.keys())
            top_k_heads = [entity_to_idx_keys[idx.item()] for idx in top_k_indices]

            # Initialize an empty tensor
            atom3_scores = torch.empty(0, len(self.entity_to_idx)).to(scores_2u_query.device)

            for head3 in top_k_heads:
                # The score tensor for the current head3
                atom3_score = self.predict(h=[head3], r=[relation_1p[0]]).unsqueeze(0)

                # Concatenate the score tensor for the current head3 with the previous scores
                atom3_scores = torch.cat([atom3_scores, atom3_score], dim=0)

            topk_scores1_expanded = top_k_scores1.view(-1, 1).repeat(1, atom3_scores.shape[1])
            combined_scores = self.t_norm(topk_scores1_expanded, atom3_scores, tnorm)
            res, _ = torch.max(combined_scores, dim=0)
            if only_scores:
                return res
            entity_scores = [(ei, s) for ei, s in zip(self.entity_to_idx.keys(), res)]
            return sorted(entity_scores, key=lambda x: x[1], reverse=True)
        else:
            raise RuntimeError(f"Incorrect query_structure {query_structure}")

    def find_missing_triples(self, confidence: float, entities: List[str] = None, relations: List[str] = None,
                             topk: int = 10,
                             at_most: int = sys.maxsize) -> Set:
        """
         Find missing triples

         Iterative over a set of entities E and a set of relation R : \forall e \in E and \forall r \in R f(e,r,x)
         Return (e,r,x)\not\in G and  f(e,r,x) > confidence

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

        {(e,r,x) | f(e,r,x) > confidence \land (e,r,x) \not\in G
        """

        assert 1.0 >= confidence >= 0.0
        assert topk >= 1

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
        print(f'Number of entities:{len(self.entity_to_idx)} \t Number of relations:{len(self.relation_to_idx)}')

        # (5) Cartesian Product over entities and relations
        # (5.1) Iterate over entities
        print('Finding missing triples..')
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
                        print(f'Number of found missing triples: {len(extended_triples)}')
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
                            print(f'Number of found missing triples: {len(extended_triples)}')
                            if len(extended_triples) == at_most:
                                return extended_triples
        return extended_triples

    def deploy(self, share: bool = False, top_k: int = 10):
        # Lazy import
        import gradio as gr

        def predict(str_subject: str, str_predicate: str, str_object: str, random_examples: bool):

            if random_examples:
                return random_prediction(self)
            else:
                if self.is_seen(entity=str_subject) and self.is_seen(
                        relation=str_predicate) and self.is_seen(entity=str_object):
                    """ Triple Prediction """
                    return deploy_triple_prediction(self, str_subject, str_predicate, str_object)

                elif self.is_seen(entity=str_subject) and self.is_seen(
                        relation=str_predicate):
                    """ Tail Entity Prediction """
                    return deploy_tail_entity_prediction(self, str_subject, str_predicate, top_k)
                elif self.is_seen(entity=str_object) and self.is_seen(
                        relation=str_predicate):
                    """ Head Entity Prediction """
                    return deploy_head_entity_prediction(self, str_object, str_predicate, top_k)
                elif self.is_seen(entity=str_subject) and self.is_seen(entity=str_object):
                    """ Relation Prediction """
                    return deploy_relation_prediction(self, str_subject, str_object, top_k)
                else:
                    KeyError('Uncovered scenario')
            # If user simply select submit
            return random_prediction(self)

        gr.Interface(
            fn=predict,
            inputs=[gr.Textbox(lines=1, placeholder=None, label='Subject'),
                    gr.Textbox(lines=1, placeholder=None, label='Predicate'),
                    gr.Textbox(lines=1, placeholder=None, label='Object'), "checkbox"],
            outputs=[gr.Textbox(label='Input Triple'),
                     gr.Dataframe(label='Outputs', type='pandas')],
            title=f'{self.name} Deployment',
            description='1. Enter a triple to compute its score,\n'
                        '2. Enter a subject and predicate pair to obtain most likely top ten entities or\n'
                        '3. Checked the random examples box and click submit').launch(share=share)

    def predict_literals(
        self,
        entity: Union[List[str], str] = None,
        attribute: Union[List[str], str] = None,
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
        assert isinstance(entity, list)
        assert isinstance(attribute, list)
        assert all(isinstance(e, str) for e in entity)      # Ensure all elements in entity are strings
        assert all(isinstance(a, str) for a in attribute)   # Ensure all elements in attribute are strings

        # Ensure entity and attribute lists are the same length
        assert len(entity) == len(attribute), "Entity and attribute lists must be of equal length"

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
    