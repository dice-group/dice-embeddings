import torch
import torch.utils.data
from pykeen.models import model_resolver
from .base_model import BaseKGE
from collections import namedtuple


class PykeenKGE(BaseKGE):
    """
    A class for using knowledge graph embedding models implemented in Pykeen.

    Parameters
    ----------
    args : dict
        A dictionary of arguments containing hyperparameters and settings for the model,
        such as embedding dimensions, random seed, and model-specific kwargs.

    Attributes
    ----------
    name : str
        The name identifier for the PykeenKGE model.
    model : pykeen.models.base.Model
        The Pykeen model instance.
    loss_history : list
        A list to store the training loss history.
    args : dict
        The arguments used to initialize the model.
    entity_embeddings : torch.nn.Embedding
        Entity embeddings learned by the model.
    relation_embeddings : torch.nn.Embedding
        Relation embeddings learned by the model.
    interaction : pykeen.nn.modules.Interaction
        Interaction module used by the Pykeen model.

    Methods
    -------
    forward_k_vs_all(x: torch.LongTensor) -> torch.FloatTensor
        Compute scores for all entities given a batch of head entities and relations.
    forward_triples(x: torch.LongTensor) -> torch.FloatTensor
        Compute scores for a batch of triples.
    forward_k_vs_sample(x: torch.LongTensor, target_entity_idx)
        Compute scores against a sampled subset of entities.

    Notes
    -----
    This class provides an interface for using knowledge graph embedding models implemented
    in Pykeen. It initializes Pykeen models based on the provided arguments and allows for
    scoring triples and conducting knowledge graph embedding experiments.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.model_kwargs = {
            "embedding_dim": args["embedding_dim"],
            "entity_initializer": None
            if args["init_param"] is None
            else torch.nn.init.xavier_normal_,
            "random_seed": args["random_seed"],
        }
        self.model_kwargs.update(args["pykeen_model_kwargs"])
        self.name = args["model"].split("_")[1]
        # Solving memory issue of Pykeen models caused by the regularizers
        # See https://github.com/pykeen/pykeen/issues/1297
        if self.name == "MuRE":
            "No Regularizer =>  no Memory Leakage"
            # https://pykeen.readthedocs.io/en/stable/api/pykeen.models.MuRE.html
        elif self.name == "QuatE":
            self.model_kwargs["entity_regularizer"] = None
            self.model_kwargs["relation_regularizer"] = None
        elif self.name == "DistMult":
            self.model_kwargs["regularizer"] = None
        elif self.name == "BoxE":
            pass
        elif self.name == "CP":
            # No regularizers
            pass
        elif self.name == "HolE":
            # No regularizers
            pass
        elif self.name == "ProjE":
            # Nothing
            pass
        elif self.name == "RotatE":
            pass
        elif self.name == "TransE":
            self.model_kwargs["regularizer"] = None
        else:
            print(
                "Pykeen model have a memory leak caused by their implementation of regularizers"
            )
            print(f"{self.name} does not seem to have any regularizer")

        self.model = model_resolver.make(
            self.name,
            self.model_kwargs,
            triples_factory=namedtuple(
                "triples_factory",
                ["num_entities", "num_relations", "create_inverse_triples"],
            )(self.num_entities, self.num_relations, False),
        )
        self.loss_history = []
        self.args = args
        self.entity_embeddings = None
        self.relation_embeddings = None
        for k, v in self.model.named_modules():
            if "entity_representations" == k:
                self.entity_embeddings = v[0]._embeddings
            elif "relation_representations" == k:
                self.relation_embeddings = v[0]._embeddings
            elif "interaction" == k:
                self.interaction = v
            else:
                pass

    def forward_k_vs_all(self, x: torch.LongTensor):
        """
        TODO: Format in Numpy-style documentation

        # => Explicit version by this we can apply bn and dropout

        # (1) Retrieve embeddings of heads and relations +  apply Dropout & Normalization if given.
        h, r = self.get_head_relation_representation(x)
        # (2) Reshape (1).
        if self.last_dim > 0:
            h = h.reshape(len(x), self.embedding_dim, self.last_dim)
            r = r.reshape(len(x), self.embedding_dim, self.last_dim)
        # (3) Reshape all entities.
        if self.last_dim > 0:
            t = self.entity_embeddings.weight.reshape(self.num_entities, self.embedding_dim, self.last_dim)
        else:
            t = self.entity_embeddings.weight
        # (4) Call the score_t from interactions to generate triple scores.
        return self.interaction.score_t(h=h, r=r, all_entities=t, slice_size=1)
        """

        return self.model.score_t(x)

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        TODO: Format in Numpy-style documentation
        
        # => Explicit version by this we can apply bn and dropout

        # (1) Retrieve embeddings of heads, relations and tails and apply Dropout & Normalization if given.
        h, r, t = self.get_triple_representation(x)
        # (2) Reshape (1).
        if self.last_dim > 0:
            h = h.reshape(len(x), self.embedding_dim, self.last_dim)
            r = r.reshape(len(x), self.embedding_dim, self.last_dim)
            t = t.reshape(len(x), self.embedding_dim, self.last_dim)
        # (3) Compute the triple score
        return self.interaction.score(h=h, r=r, t=t, slice_size=None, slice_dim=0)
        """
        return self.model.score_hrt(hrt_batch=x, mode=None).flatten()

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx):
        raise NotImplementedError(f"KvsSample has not yet implemented for {self.name}")
