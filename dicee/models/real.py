from .base_model import BaseKGE
from typing import Tuple
import torch
import numpy as np


class DistMult(BaseKGE):
    """
    DistMult model for learning and inference in knowledge bases. It represents both entities
    and relations using embeddings and uses a simple bilinear form to compute scores for triples.

    This implementation of the DistMult model is based on the paper:
    'Embedding Entities and Relations for Learning and Inference in Knowledge Bases'
    (https://arxiv.org/abs/1412.6575).

    Attributes
    ----------
    name : str
        The name identifier for the DistMult model.

    Methods
    -------
    k_vs_all_score(emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor) -> torch.FloatTensor
        Computes scores in a K-vs-All setting using embeddings for a batch of head entities and relations.

    forward_k_vs_all(x: torch.LongTensor) -> torch.FloatTensor
        Computes scores for all entities given a batch of head entities and relations.

    forward_k_vs_sample(x: torch.LongTensor, target_entity_idx: torch.LongTensor) -> torch.FloatTensor
        Computes scores for a sampled subset of entities given a batch of head entities and relations.

    score(h: torch.FloatTensor, r: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor
        Computes the score of triples using DistMult's scoring function.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "DistMult"

    def k_vs_all_score(
        self,
        emb_h: torch.FloatTensor,
        emb_r: torch.FloatTensor,
        emb_E: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Computes scores in a K-vs-All setting using embeddings for a batch of head entities and relations.

        This method multiplies the head entity and relation embeddings, applies a dropout and a normalization,
        and then computes the dot product with all tail entity embeddings.

        Parameters
        ----------
        emb_h : torch.FloatTensor
            Embeddings of head entities.
        emb_r : torch.FloatTensor
            Embeddings of relations.
        emb_E : torch.FloatTensor
            Embeddings of all entities.

        Returns
        -------
        torch.FloatTensor
            Scores for all possible triples formed with the given head entities and relations against all entities.
        """
        return torch.mm(
            self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)),
            emb_E.transpose(1, 0),
        )

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Computes scores for all entities given a batch of head entities and relations.

        This method is used for K-vs-All scoring, where the model predicts the likelihood of each entity
        being the tail entity in a triple with each head entity and relation pair in the batch.

        Parameters
        ----------
        x : torch.LongTensor
            Tensor containing indices for head entities and relations.

        Returns
        -------
        torch.FloatTensor
            Scores for all entities for each head entity and relation pair in the batch.
        """
        emb_head, emb_rel = self.get_head_relation_representation(x)
        return self.k_vs_all_score(
            emb_h=emb_head, emb_r=emb_rel, emb_E=self.entity_embeddings.weight
        )

    def forward_k_vs_sample(
        self, x: torch.LongTensor, target_entity_idx: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Computes scores for a sampled subset of entities given a batch of head entities and relations.

        This method is particularly useful when the full set of entities is too large to score
        with every batch and only a subset of entities is required.

        Parameters
        ----------
        x : torch.LongTensor
            Tensor containing indices for head entities and relations.
        target_entity_idx : torch.LongTensor
            Indices of the target entities against which the scores are to be computed.

        Returns
        -------
        torch.FloatTensor
            Scores for each head entity and relation pair against the sampled subset of entities.
        """
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        hr = self.hidden_dropout(
            self.hidden_normalizer(emb_head_real * emb_rel_real)
        ).unsqueeze(1)
        t = self.entity_embeddings(target_entity_idx).transpose(1, 2)
        return torch.bmm(hr, t).squeeze(1)

    def score(
        self, h: torch.FloatTensor, r: torch.FloatTensor, t: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Computes the score of triples using DistMult's scoring function.

        The scoring function multiplies head entity and relation embeddings, applies dropout and normalization,
        and computes the dot product with the tail entity embeddings.

        Parameters
        ----------
        h : torch.FloatTensor
            Embedding of the head entity.
        r : torch.FloatTensor
            Embedding of the relation.
        t : torch.FloatTensor
            Embedding of the tail entity.

        Returns
        -------
        torch.FloatTensor
            The score of the triple.
        """
        return (self.hidden_dropout(self.hidden_normalizer(h * r)) * t).sum(dim=1)


class TransE(BaseKGE):
    """
    TransE model for learning embeddings in multi-relational data. It is based on the idea of translating
    embeddings for head entities by the relation vector to approach the tail entity embeddings in the embedding space.

    This implementation of TransE is based on the paper:
    'Translating Embeddings for Modeling Multi-relational Data'
    (https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf).

    Attributes
    ----------
    name : str
        The name identifier for the TransE model.
    _norm : int
        The norm used for computing pairwise distances in the embedding space.
    margin : int
        The margin value used in the scoring function.

    Methods
    -------
    score(head_ent_emb: torch.Tensor, rel_ent_emb: torch.Tensor, tail_ent_emb: torch.Tensor) -> torch.Tensor
        Computes the score of triples using the TransE scoring function.

    forward_k_vs_all(x: torch.Tensor) -> torch.FloatTensor
        Computes scores for all entities given a head entity and a relation.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "TransE"
        self._norm = 2
        self.margin = 4

    def score(
        self,
        head_ent_emb: torch.Tensor,
        rel_ent_emb: torch.Tensor,
        tail_ent_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score of triples using the TransE scoring function.

        The scoring function computes the L2 distance between the translated head entity
        and the tail entity embeddings and subtracts this distance from the margin.

        Parameters
        ----------
        head_ent_emb : torch.Tensor
            Embedding of the head entity.
        rel_ent_emb : torch.Tensor
            Embedding of the relation.
        tail_ent_emb : torch.Tensor
            Embedding of the tail entity.

        Returns
        -------
        torch.Tensor
            The score of the triple.
        """
        # Original d:=|| s+p - t||_2 \approx 0 distance, if true
        # if d =0 sigma(5-0) => 1
        # if d =5 sigma(5-5) => 0.5
        # Update: sigmoid( \gamma - d)
        return self.margin - torch.nn.functional.pairwise_distance(
            head_ent_emb + rel_ent_emb, tail_ent_emb, p=self._norm
        )

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Computes scores for all entities given a head entity and a relation.

        This method is used for K-vs-All scoring, where the model predicts the likelihood of each entity
        being the tail entity in a triple with each head entity and relation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing indices for head entities and relations.

        Returns
        -------
        torch.FloatTensor
            Scores for all entities for each head entity and relation pair.
        """
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        distance = torch.nn.functional.pairwise_distance(
            torch.unsqueeze(emb_head_real + emb_rel_real, 1),
            self.entity_embeddings.weight,
            p=self._norm,
        )
        return self.margin - distance


class Shallom(BaseKGE):
    """
    Shallom is a shallow neural model designed for relation prediction in knowledge graphs.
    The model combines entity embeddings and passes them through a neural network to predict
    the likelihood of different relations. It's based on the paper:
    'A Shallow Neural Model for Relation Prediction'
    (https://arxiv.org/abs/2101.09090).

    Attributes
    ----------
    name : str
        The name identifier for the Shallom model.
    shallom : torch.nn.Sequential
        A sequential neural network model used for predicting relations.

    Methods
    -------
    get_embeddings() -> Tuple[np.ndarray, None]
        Retrieves the entity embeddings.

    forward_k_vs_all(x) -> torch.FloatTensor
        Computes relation scores for all pairs of entities in the batch.

    forward_triples(x) -> torch.FloatTensor
        Computes relation scores for a batch of triples.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "Shallom"
        shallom_width = int(2 * self.embedding_dim)
        self.shallom = torch.nn.Sequential(
            torch.nn.Dropout(self.input_dropout_rate),
            torch.nn.Linear(self.embedding_dim * 2, shallom_width),
            self.normalizer_class(shallom_width),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.hidden_dropout_rate),
            torch.nn.Linear(shallom_width, self.num_relations),
        )

    def get_embeddings(self) -> Tuple[np.ndarray, None]:
        """
        Retrieves the entity embeddings from the model.

        Returns
        -------
        Tuple[np.ndarray, None]
            A tuple containing the entity embeddings as a NumPy array and None for the relation embeddings.
        """
        return self.entity_embeddings.weight.data.detach(), None

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Computes relation scores for all pairs of entities in the batch.

        Each pair of entities is passed through the Shallom neural network to predict
        the likelihood of various relations between them.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of entity pairs.

        Returns
        -------
        torch.FloatTensor
            A tensor of relation scores for each pair of entities in the batch.
        """
        e1_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, e2_idx = x[:, 0], x[:, 1]
        emb_s, emb_o = self.entity_embeddings(e1_idx), self.entity_embeddings(e2_idx)
        return self.shallom(torch.cat((emb_s, emb_o), 1))

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Computes relation scores for a batch of triples.

        This method first computes relation scores for all possible relations for each pair of entities
        and then selects the scores corresponding to the actual relations in the triples.

        Parameters
        ----------
        x : torch.Tensor
            A tensor containing a batch of triples.

        Returns
        -------
        torch.FloatTensor
            A flattened tensor of relation scores for the given batch of triples.
        """
        n, d = x.shape
        assert d == 3
        scores_for_all_relations = self.forward_k_vs_all(x[:, [0, 2]])
        return scores_for_all_relations[:, x[:, 1]].flatten()


class Pyke(BaseKGE):
    """
    Pyke is a physical embedding model for knowledge graphs, emphasizing the geometric relationships
    in the embedding space. The model aims to represent entities and relations in a way that captures
    the underlying structure of the knowledge graph.

    Attributes
    ----------
    name : str
        The name identifier for the Pyke model.
    dist_func : torch.nn.PairwiseDistance
        A pairwise distance function to compute distances in the embedding space.
    margin : float
        The margin value used in the scoring function.

    Methods
    -------
    forward_triples(x: torch.LongTensor) -> torch.FloatTensor
        Computes scores for a batch of triples based on the physical embedding approach.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "Pyke"
        self.dist_func = torch.nn.PairwiseDistance(p=2)
        self.margin = 1.0

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Computes scores for a batch of triples based on the physical embedding approach.

        The method calculates the Euclidean distance between the head and relation embeddings,
        and between the relation and tail embeddings. The average of these distances is subtracted
        from the margin to compute the score for each triple.

        Parameters
        ----------
        x : torch.LongTensor
            A tensor containing indices for head entities, relations, and tail entities.

        Returns
        -------
        torch.FloatTensor
            Scores for the given batch of triples. Lower scores indicate more likely triples
            according to the geometric arrangement of embeddings.
        """
        # (1) get embeddings for a batch of entities and relations
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Compute the Euclidean distance from head to relation
        dist_head_rel = self.dist_func(head_ent_emb, rel_ent_emb)
        dist_rel_tail = self.dist_func(rel_ent_emb, tail_ent_emb)
        avg_dist = (dist_head_rel + dist_rel_tail) / 2
        return self.margin - avg_dist
