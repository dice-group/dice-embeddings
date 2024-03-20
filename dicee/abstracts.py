import os
import datetime
from .static_funcs import (
    load_model_ensemble,
    load_model,
    save_checkpoint_model,
    load_json,
    download_pretrained_model,
)
import torch
from typing import List, Optional, Tuple, Union
import random
from abc import ABC
import lightning


class AbstractTrainer:
    """
    Abstract base class for Trainer classes used in training knowledge graph embedding models.
    Defines common functionalities and lifecycle hooks for training processes.

    Parameters
    ----------
    args : Namespace or similar
        A container for various training configurations and hyperparameters.
    callbacks : list of Callback objects
        A list of callback instances to be invoked at various stages of the training process.
    """

    def __init__(self, args, callbacks):
        self.attributes = args
        self.callbacks = callbacks
        self.is_global_zero = True
        # Set True to use Model summary callback of pl.
        torch.manual_seed(self.attributes.random_seed)
        torch.cuda.manual_seed_all(self.attributes.random_seed)
        # To be able to use pl callbacks with our trainers.
        self.strategy = None

    def on_fit_start(self, *args, **kwargs) -> None:
        """
        Invokes the `on_fit_start` method of each registered callback before the training starts.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_fit_start(*args, **kwargs)

    def on_fit_end(self, *args, **kwargs) -> None:
        """
        Invokes the `on_fit_end` method of each registered callback after the training ends.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_fit_end(*args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        """
        Invokes the `on_train_epoch_end` method of each registered callback after each epoch ends.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_train_epoch_end(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """
        Invokes the `on_train_batch_end` method of each registered callback after each training batch ends.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        for c in self.callbacks:
            c.on_train_batch_end(*args, **kwargs)

    @staticmethod
    def save_checkpoint(full_path: str, model: torch.nn.Module) -> None:
        """
        Saves the model's state dictionary to a file.

        Parameters
        ----------
        full_path : str
            The file path where the model checkpoint will be saved.
        model : torch.nn.Module
            The model instance whose parameters are to be saved.

        Returns
        -------
        None
        """
        torch.save(model.state_dict(), full_path)


class BaseInteractiveKGE:
    """
    Base class for interactively utilizing knowledge graph embedding models.
    Supports operations such as loading pretrained models, querying the model, and adding new embeddings.

    Parameters
    ----------
    path : str, optional
        Path to the directory where the pretrained model is stored. Either `path` or `url` must be provided.
    url : str, optional
        URL to download the pretrained model. If provided, `path` is ignored and the model is downloaded to a local path.
    construct_ensemble : bool, default=False
        Whether to construct an ensemble model from the pretrained models available in the specified directory.
    model_name : str, optional
        Name of the specific model to load. Required if multiple models are present and `construct_ensemble` is False.
    apply_semantic_constraint : bool, default=False
        Whether to apply semantic constraints based on domain and range information during inference.

    Attributes
    ----------
    model : torch.nn.Module
        The loaded or constructed knowledge graph embedding model.
    entity_to_idx : dict
        Mapping from entity names to their corresponding indices in the embedding matrix.
    relation_to_idx : dict
        Mapping from relation names to their corresponding indices in the embedding matrix.
    num_entities : int
        The number of unique entities in the knowledge graph.
    num_relations : int
        The number of unique relations in the knowledge graph.
    configs : dict
        Configuration settings and performance metrics of the pretrained model.
    """

    def __init__(
        self,
        path: str = None,
        url: str = None,
        construct_ensemble: bool = False,
        model_name: str = None,
        apply_semantic_constraint: bool = False,
    ):
        if url is not None:
            assert path is None
            self.path = download_pretrained_model(url)
        else:
            self.path = path
        try:
            assert os.path.isdir(self.path)
        except AssertionError:
            raise AssertionError(f"Could not find a directory {self.path}")

        # (1) Load model...
        self.construct_ensemble = construct_ensemble
        self.apply_semantic_constraint = apply_semantic_constraint
        self.configs = load_json(self.path + "/configuration.json")
        self.configs.update(load_json(self.path + "/report.json"))

        if construct_ensemble:
            self.model, tuple_of_entity_relation_idx = load_model_ensemble(self.path)
        else:
            if model_name:
                self.model, tuple_of_entity_relation_idx = load_model(
                    self.path, model_name=model_name
                )
            else:
                self.model, tuple_of_entity_relation_idx = load_model(self.path)
        if self.configs.get("byte_pair_encoding", None):
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
            self.dummy_id = tiktoken.get_encoding("gpt2").encode(" ")[0]
            self.max_length_subword_tokens = self.configs["max_length_subword_tokens"]
        else:
            assert len(tuple_of_entity_relation_idx) == 2

            self.entity_to_idx, self.relation_to_idx = tuple_of_entity_relation_idx
            self.num_entities = len(self.entity_to_idx)
            self.num_relations = len(self.relation_to_idx)
            self.entity_to_idx: dict
            self.relation_to_idx: dict
            assert list(self.entity_to_idx.values()) == list(
                range(0, len(self.entity_to_idx))
            )
            assert list(self.relation_to_idx.values()) == list(
                range(0, len(self.relation_to_idx))
            )

            self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
            self.idx_to_relations = {v: k for k, v in self.relation_to_idx.items()}

        # See https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        # @TODO: Ignore temporalryIf file exists
        # if os.path.exists(self.path + '/train_set.npy'):
        #    self.train_set = np.load(file=self.path + '/train_set.npy', mmap_mode='r')

        # if apply_semantic_constraint:
        #    (self.domain_constraints_per_rel, self.range_constraints_per_rel,
        #     self.domain_per_rel, self.range_per_rel) = create_constraints(self.train_set)

    def get_eval_report(self) -> dict:
        """
        Retrieves the evaluation report of the pretrained model.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics and their values.
        """
        return load_json(self.path + "/eval_report.json")

    def get_bpe_token_representation(
        self, str_entity_or_relation: Union[List[str], str]
    ) -> Union[List[List[int]], List[int]]:
        """
        Converts a string entity or relation name (or a list of them) to its Byte Pair Encoding (BPE) token representation.

        Parameters
        ----------
        str_entity_or_relation : Union[List[str], str]
            The entity or relation name(s) to be converted.

        Returns
        -------
        Union[List[List[int]], List[int]]
            The BPE token representation as a list of integers or a list of lists of integers.
        """

        if isinstance(str_entity_or_relation, list):
            return [
                self.get_bpe_token_representation(i) for i in str_entity_or_relation
            ]
        else:
            # (1) Map a string into its binary representation
            unshaped_bpe_repr = self.enc.encode(str_entity_or_relation)
            # (2)
            if len(unshaped_bpe_repr) <= self.max_length_subword_tokens:
                unshaped_bpe_repr.extend(
                    [
                        self.dummy_id
                        for _ in range(
                            self.max_length_subword_tokens - len(unshaped_bpe_repr)
                        )
                    ]
                )
            else:
                # @TODO: What to do ?
                # print(f'Larger length is detected from all lengths have seen:{str_entity_or_relation} | {len(unshaped_bpe_repr)}')
                pass

            return unshaped_bpe_repr

    def get_padded_bpe_triple_representation(
        self, triples: List[List[str]]
    ) -> Tuple[List, List, List]:
        """
        Converts a list of triples to their padded BPE token representations.

        Parameters
        ----------
        triples : List[List[str]]
            A list of triples, where each triple is a list of strings [head entity, relation, tail entity].

        Returns
        -------
        Tuple[List, List, List]
            Three lists corresponding to the padded BPE token representations of head entities, relations, and tail entities.
        """
        assert isinstance(triples, List)

        if isinstance(triples[0], List) is False:
            triples = [triples]

        assert len(triples[0]) == 3
        padded_bpe_h = []
        padded_bpe_r = []
        padded_bpe_t = []

        for [str_s, str_p, str_o] in triples:
            padded_bpe_h.append(self.get_bpe_token_representation(str_s))
            padded_bpe_r.append(self.get_bpe_token_representation(str_p))
            padded_bpe_t.append(self.get_bpe_token_representation(str_o))
        return padded_bpe_h, padded_bpe_r, padded_bpe_t

    def get_domain_of_relation(self, rel: str) -> List[str]:
        """
        Retrieves the domain of a given relation.

        Parameters
        ----------
        rel : str
            The relation name.

        Returns
        -------
        List[str]
            A list of entity names that constitute the domain of the specified relation.
        """
        x = [
            self.idx_to_entity[i]
            for i in self.domain_per_rel[self.relation_to_idx[rel]]
        ]
        res = set(x)
        assert len(x) == len(res)
        return res

    def get_range_of_relation(self, rel: str) -> List[str]:
        """
        Retrieves the range of a given relation.

        Parameters
        ----------
        rel : str
            The relation name.

        Returns
        -------
        List[str]
            A list of entity names that constitute the range of the specified relation.
        """
        x = [
            self.idx_to_entity[i] for i in self.range_per_rel[self.relation_to_idx[rel]]
        ]
        res = set(x)
        assert len(x) == len(res)
        return res

    def set_model_train_mode(self) -> None:
        """Sets the model to training mode. This enables gradient computation and backpropagation."""
        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def set_model_eval_mode(self) -> None:
        """Sets the model to evaluation mode. This disables gradient computation, making the model read-only and faster for inference."""
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def name(self) -> str:
        """
        Property that returns the model's name.

        Returns
        -------
        str
            The name of the model.
        """
        return self.model.name

    def sample_entity(self, n: int) -> List[str]:
        """
        Randomly samples a specified number of unique entities from the knowledge graph.

        Parameters
        ----------
        n : int
            The number of entities to sample.

        Returns
        -------
        List[str]
            A list of sampled entity names.
        """
        assert isinstance(n, int)
        assert n >= 0
        return random.sample([i for i in self.entity_to_idx.keys()], n)

    def sample_relation(self, n: int) -> List[str]:
        """
        Randomly samples a specified number of unique relations from the knowledge graph.

        Parameters
        ----------
        n : int
            The number of relations to sample.

        Returns
        -------
        List[str]
            A list of sampled relation names.
        """
        assert isinstance(n, int)
        assert n >= 0
        return random.sample([i for i in self.relation_to_idx.keys()], n)

    def is_seen(self, entity: str = None, relation: str = None) -> bool:
        """
        Checks if the specified entity or relation is known to the model.

        Parameters
        ----------
        entity : str, optional
            The entity name to check.
        relation : str, optional
            The relation name to check.

        Returns
        -------
        bool
            True if the entity or relation is known; False otherwise.
        """
        if entity is not None:
            return True if self.entity_to_idx.get(entity) else False
        if relation is not None:
            return True if self.relation_to_idx.get(relation) else False

    def save(self) -> None:
        """
        Saves the current state of the model to disk. The filename is timestamped.

        Returns
        -------
        None
        """
        t = str(datetime.datetime.now())
        if self.construct_ensemble:
            save_checkpoint_model(
                self.model, path=self.path + f"/model_ensemble_interactive_{str(t)}.pt"
            )
        else:
            save_checkpoint_model(
                self.model, path=self.path + f"/model_interactive_{str(t)}.pt"
            )

    def get_entity_index(self, x: str) -> int:
        """
        Retrieves the index of the specified entity.

        Parameters
        ----------
        x : str
            The entity name.

        Returns
        -------
        int
            The index of the entity.
        """
        return self.entity_to_idx[x]

    def get_relation_index(self, x: str) -> int:
        """
        Retrieves the index of the specified relation.

        Parameters
        ----------
        x : str
            The relation name.

        Returns
        -------
        int
            The index of the relation.
        """
        return self.relation_to_idx[x]

    def index_triple(
        self, head_entity: List[str], relation: List[str], tail_entity: List[str]
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Converts a list of triples from string representation to tensor indices.

        Parameters
        ----------
        head_entity : List[str]
            The list of head entities.
        relation : List[str]
            The list of relations.
        tail_entity : List[str]
            The list of tail entities.

        Returns
        -------
        Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]
            The tensor indices of head entities, relations, and tail entities.
        """
        n = len(head_entity)
        assert n == len(relation) == len(tail_entity)
        idx_head_entity = torch.LongTensor(
            [self.entity_to_idx[i] for i in head_entity]
        ).reshape(n, 1)
        idx_relation = torch.LongTensor(
            [self.relation_to_idx[i] for i in relation]
        ).reshape(n, 1)
        idx_tail_entity = torch.LongTensor(
            [self.entity_to_idx[i] for i in tail_entity]
        ).reshape(n, 1)
        return idx_head_entity, idx_relation, idx_tail_entity

    def add_new_entity_embeddings(
        self, entity_name: str = None, embeddings: torch.FloatTensor = None
    ) -> None:
        """
        Adds a new entity and its embeddings to the model.

        Parameters
        ----------
        entity_name : str
            The name of the new entity.
        embeddings : torch.FloatTensor
            The embedding vector of the new entity.

        Returns
        -------
        None
        """
        assert isinstance(entity_name, str) and isinstance(
            embeddings, torch.FloatTensor
        )

        if entity_name in self.entity_to_idx:
            print(f"Entity ({entity_name}) exists..")
        else:
            self.entity_to_idx[entity_name] = len(self.entity_to_idx)
            self.idx_to_entity[self.entity_to_idx[entity_name]] = entity_name
            self.num_entities += 1
            self.model.num_entities += 1
            self.model.entity_embeddings.weight.data = torch.cat(
                (
                    self.model.entity_embeddings.weight.data.detach(),
                    embeddings.unsqueeze(0),
                ),
                dim=0,
            )
            self.model.entity_embeddings.num_embeddings += 1

    def get_entity_embeddings(self, items: List[str]) -> torch.FloatTensor:
        """
        Retrieves embeddings for a list of entities.

        Parameters
        ----------
        items : List[str]
            A list of entity names.

        Returns
        -------
        torch.FloatTensor
            A tensor containing the embeddings of the specified entities.
        """
        if self.configs["byte_pair_encoding"]:
            t_encode = self.enc.encode_batch(items)
            if len(t_encode) != self.configs["max_length_subword_tokens"]:
                for i in range(len(t_encode)):
                    t_encode[i].extend(
                        [
                            self.dummy_id
                            for _ in range(
                                self.configs["max_length_subword_tokens"]
                                - len(t_encode[i])
                            )
                        ]
                    )
            return self.model.token_embeddings(torch.LongTensor(t_encode)).flatten(1)
        else:
            return self.model.entity_embeddings(
                torch.LongTensor([self.entity_to_idx[i] for i in items])
            )

    def get_relation_embeddings(self, items: List[str]) -> torch.FloatTensor:
        """
        Retrieves embeddings for a list of relations.

        Parameters
        ----------
        items : List[str]
            A list of relation names.

        Returns
        -------
        torch.FloatTensor
            A tensor containing the embeddings of the specified relations.
        """
        return self.model.relation_embeddings(
            torch.LongTensor([self.relation_to_idx[i] for i in items])
        )

    def construct_input_and_output(
        self,
        head_entity: List[str],
        relation: List[str],
        tail_entity: List[str],
        labels,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs input and output tensors for a given set of triples and labels.

        Parameters
        ----------
        head_entity : List[str]
            A list of head entities.
        relation : List[str]
            A list of relations.
        tail_entity : List[str]
            A list of tail entities.
        labels : List[int] or torch.Tensor
            The labels associated with each triple.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The input tensor consisting of indexed triples and the output tensor of labels.
        """
        idx_head_entity, idx_relation, idx_tail_entity = self.index_triple(
            head_entity, relation, tail_entity
        )
        x = torch.hstack((idx_head_entity, idx_relation, idx_tail_entity))
        # Hard Labels
        labels: object = torch.FloatTensor(labels)
        return x, labels

    def parameters(self):
        """
        Retrieves the parameters of the model.

        This method is typically used to access the parameters of the model for optimization or inspection.

        Returns
        -------
        Iterator[torch.nn.parameter.Parameter]
            An iterator over the model parameters, which are instances of torch.nn.parameter.Parameter.
        """
        return self.model.parameters()


class AbstractCallback(ABC, lightning.pytorch.callbacks.Callback):
    """
    Abstract base class for implementing custom callbacks for knowledge graph embedding models during training with PyTorch Lightning.

    This class is designed to be subclassed, with methods overridden to perform actions at various points during the training life cycle.
    """

    def __init__(self):
        pass

    def on_init_start(self, *args, **kwargs):
        """
        Called before the trainer initialization starts.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        """
        pass

    def on_init_end(self, *args, **kwargs):
        """
        Called after the trainer initialization ends.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        """
        pass

    def on_fit_start(self, trainer, model):
        """
        Called at the very beginning of fit.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        pl_module : pl.LightningModule
            The model that is being trained.
        """
        return

    def on_train_epoch_end(self, trainer, model):
        """
        Called at the end of the training epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        pl_module : pl.LightningModule
            The model that is being trained.
        """
        pass

    def on_train_batch_end(self, *args, **kwargs):
        """
        Call at the end of each mini-batch during the training.

        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pass

    def on_fit_end(self, *args, **kwargs):
        """
        Called at the end of fit.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer instance.
        pl_module : pl.LightningModule
            The model that has been trained.
        """
        pass


class AbstractPPECallback(AbstractCallback):
    """
    Abstract base class for implementing Parameter Prediction Ensemble (PPE) callbacks for knowledge graph embedding models.

    This class provides a structure for creating ensemble models by averaging model parameters over epochs,
    which can potentially improve model performance and robustness.

    Parameters
    ----------
    num_epochs : int
        Total number of epochs for training.
    path : str
        Path to save or load the ensemble model.
    epoch_to_start : Optional[int]
        The epoch number to start creating the ensemble. If None, a percentage of epochs to consider can be specified instead.
    last_percent_to_consider : Optional[float]
        The last percentage of epochs to consider for creating the ensemble. If both `epoch_to_start` and `last_percent_to_consider` are None, ensemble starts from epoch 1.
    """

    def __init__(
        self,
        num_epochs: int,
        path: str,
        epoch_to_start: Optional[int] = None,
        last_percent_to_consider: Optional[float] = None,
    ):
        super(AbstractPPECallback, self).__init__()
        self.num_epochs = num_epochs
        self.path = path
        self.sample_counter = 0
        self.epoch_count = 0
        self.alphas = None
        if epoch_to_start is not None:
            self.epoch_to_start = epoch_to_start
            try:
                assert self.epoch_to_start < self.num_epochs
            except AssertionError:
                raise AssertionError(
                    f"--epoch_to_start {self.epoch_to_start} "
                    f"must be less than --num_epochs {self.num_epochs}"
                )
            self.num_ensemble_coefficient = self.num_epochs - self.epoch_to_start + 1
        elif last_percent_to_consider is None:
            self.epoch_to_start = 1
            self.num_ensemble_coefficient = self.num_epochs - 1
        else:
            # Compute the last X % of the training
            self.epoch_to_start = self.num_epochs - int(
                self.num_epochs * last_percent_to_consider / 100
            )
            self.num_ensemble_coefficient = self.num_epochs - self.epoch_to_start

    def on_fit_start(self, trainer, model):
        """
        Called at the very beginning of fit.

        Parameters
        ----------
        trainer : Trainer instance
            The trainer instance.
        model : LightningModule
            The model that is being trained.
        """
        pass

    def on_fit_end(self, trainer, model):
        """
        Called at the end of fit. It loads the ensemble parameters if they exist.

        Parameters
        ----------
        trainer : Trainer instance
            The trainer instance.
        model : LightningModule
            The model that has been trained.
        """
        if os.path.exists(f"{self.path}/trainer_checkpoint_main.pt"):
            param_ensemble = torch.load(
                f"{self.path}/trainer_checkpoint_main.pt", torch.device("cpu")
            )
            model.load_state_dict(param_ensemble)
        else:
            print(
                f"No parameter ensemble found at {self.path}/trainer_checkpoint_main.pt"
            )

    def store_ensemble(self, param_ensemble: torch.Tensor) -> None:
        """
        Saves the updated parameter ensemble model to disk.

        Parameters
        ----------
        param_ensemble : torch.Tensor
            The ensemble of model parameters to be saved.
        """
        # (3) Save the updated parameter ensemble model.
        torch.save(param_ensemble, f=f"{self.path}/trainer_checkpoint_main.pt")
        if self.sample_counter > 1:
            self.sample_counter += 1
