import os
import datetime
from .static_funcs import load_model_ensemble, load_model, save_checkpoint_model, load_json, download_pretrained_model
import torch
from typing import List, Tuple, Union
import random
from abc import ABC
import lightning
from .models.literal import LiteralEmbeddings
from .dataset_classes import TriplePredictionDataset, LiteralDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


class AbstractTrainer:
    """Abstract base class for KGE model trainers.

    Provides the callback dispatch mechanism shared by all concrete trainer
    implementations (TorchTrainer, TorchDDPTrainer, etc.).  Sub-classes call
    the ``on_*`` hooks at the appropriate points in the training loop so that
    any registered :class:`AbstractCallback` can react.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Processed configuration object.  Must expose at least
        ``random_seed`` (int).
    callbacks : list of AbstractCallback
        Ordered list of callback instances to invoke at each lifecycle hook.
    """

    def __init__(self, args, callbacks):
        self.attributes = args
        self.callbacks = callbacks
        self.is_global_zero = True
        self.global_rank=0
        self.local_rank = 0
        # Set True to use Model summary callback of pl.
        torch.manual_seed(self.attributes.random_seed)
        torch.cuda.manual_seed_all(self.attributes.random_seed)
        # To be able to use pl callbacks with our trainers.
        self.strategy=None

    def on_fit_start(self, *args, **kwargs):
        """Dispatch ``on_fit_start`` to all registered callbacks.

        Called once before the first training epoch begins.
        """
        for c in self.callbacks:
            c.on_fit_start(*args, **kwargs)

    def on_fit_end(self, *args, **kwargs):
        """Dispatch ``on_fit_end`` to all registered callbacks.

        Called once after the last training epoch completes.
        """
        for c in self.callbacks:
            c.on_fit_end(*args, **kwargs)

    def on_train_epoch_start(self, *args, **kwargs):
        """Dispatch ``on_train_epoch_start`` to all registered callbacks.

        Called at the beginning of every epoch.
        """
        for c in self.callbacks:
            c.on_train_epoch_start(*args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        """Dispatch ``on_train_epoch_end`` to all registered callbacks.

        Called at the end of every epoch after the loss has been accumulated.
        """
        for c in self.callbacks:
            c.on_train_epoch_end(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        """Dispatch ``on_train_batch_end`` to all registered callbacks.

        Called after each mini-batch gradient update.
        """
        for c in self.callbacks:
            c.on_train_batch_end(*args, **kwargs)

    @staticmethod
    def save_checkpoint(full_path: str, model) -> None:
        """Persist model weights to disk.

        Parameters
        ----------
        full_path : str
            Absolute or relative file path (including filename) where the
            ``state_dict`` will be written, e.g.
            ``'Experiments/run1/model.pt'``.
        model : torch.nn.Module
            The model whose ``state_dict`` is to be saved.
        """
        torch.save(model.state_dict(), full_path)


class BaseInteractiveKGE:
    """Base class for interactive, post-training use of KGE models.

    Loads a pre-trained model from disk (or a remote URL) together with its
    entity/relation index mappings and exposes the prediction API used by
    :class:`~dicee.knowledge_graph_embeddings.KGE`.

    Parameters
    ----------
    path : str, optional
        Path to the experiment directory produced by :class:`Execute`.
        Must contain ``model.pt``, ``configuration.json``,
        ``entity_to_idx.csv`` and ``relation_to_idx.csv``.
    url : str, optional
        Remote URL of a pre-trained model to download.  Mutually exclusive
        with *path*.
    construct_ensemble : bool, optional
        When ``True``, load all checkpoint files in *path* and average their
        weights to form an ensemble model.  Defaults to ``False``.
    model_name : str, optional
        Filename (without extension) of the checkpoint to load when multiple
        ``.pt`` files exist in *path*.
    apply_semantic_constraint : bool, optional
        Reserved for future use.  Defaults to ``False``.
    """

    def __init__(self, path: str = None, url: str = None, construct_ensemble: bool = False, model_name: str = None,
                 apply_semantic_constraint: bool = False):
        if url is not None:
            assert path is None
            self.path = download_pretrained_model(url)
        else:
            self.path = path
        try:
            assert os.path.isdir(self.path)
        except AssertionError:
            raise AssertionError(f'Could not find a directory {self.path}')

        # (1) Load model...
        self.construct_ensemble = construct_ensemble
        self.apply_semantic_constraint = apply_semantic_constraint
        self.configs = load_json(self.path + '/configuration.json')
        self.configs.update(load_json(self.path + '/report.json'))

        if construct_ensemble:
            self.model, tuple_of_entity_relation_idx = load_model_ensemble(self.path)
        else:
            if model_name:
                self.model, tuple_of_entity_relation_idx = load_model(self.path, model_name=model_name)
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
            # 0, ....,
            assert sorted(list(self.entity_to_idx.values())) == list(range(0, len(self.entity_to_idx)))
            assert sorted(list(self.relation_to_idx.values())) == list(range(0, len(self.relation_to_idx)))

            self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}
            self.idx_to_relations = {v: k for k, v in self.relation_to_idx.items()}

    def get_eval_report(self) -> dict:
        return load_json(self.path + "/eval_report.json")

    def get_bpe_token_representation(self, str_entity_or_relation: Union[List[str], str]) -> Union[
        List[List[int]], List[int]]:
        """

        Parameters
        ----------
        str_entity_or_relation: corresponds to a str or a list of strings to be tokenized via BPE and shaped.

        Returns
        -------
        A list integer(s) or a list of lists containing integer(s)

        """

        if isinstance(str_entity_or_relation, list):
            return [self.get_bpe_token_representation(i) for i in str_entity_or_relation]
        else:
            # (1) Map a string into its binary representation
            unshaped_bpe_repr = self.enc.encode(str_entity_or_relation)
            # (2)
            if len(unshaped_bpe_repr) <= self.max_length_subword_tokens:
                unshaped_bpe_repr.extend(
                    [self.dummy_id for _ in range(self.max_length_subword_tokens - len(unshaped_bpe_repr))])
            else:
                # @TODO: What to do ?
                # print(f'Larger length is detected from all lengths have seen:{str_entity_or_relation} | {len(unshaped_bpe_repr)}')
                pass

            return unshaped_bpe_repr

    def get_padded_bpe_triple_representation(self, triples: List[List[str]]) -> Tuple[List, List, List]:
        """

        Parameters
        ----------
        triples

        Returns
        -------

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


    def set_model_train_mode(self) -> None:
        """Switch the underlying model to training mode.

        Calls ``model.train()`` and re-enables gradient computation for all
        parameters so that subsequent calls to optimisation steps work
        correctly after a period of inference.
        """
        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def set_model_eval_mode(self) -> None:
        """Switch the underlying model to evaluation mode.

        Calls ``model.eval()`` and freezes all parameters (``requires_grad =
        False``) so that dropout and batch-norm layers behave deterministically
        during inference.
        """
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def name(self):
        return self.model.name

    def sample_entity(self, n: int) -> List[str]:
        """Return *n* random entity strings without replacement.

        Parameters
        ----------
        n : int
            Number of entities to sample.  Must be non-negative and at most
            ``num_entities``.

        Returns
        -------
        List[str]
            Randomly selected entity string labels.
        """
        assert isinstance(n, int)
        assert n >= 0
        return random.sample([i for i in self.entity_to_idx.keys()], n)

    def sample_relation(self, n: int) -> List[str]:
        """Return *n* random relation strings without replacement.

        Parameters
        ----------
        n : int
            Number of relations to sample.  Must be non-negative and at most
            ``num_relations``.

        Returns
        -------
        List[str]
            Randomly selected relation string labels.
        """
        assert isinstance(n, int)
        assert n >= 0
        return random.sample([i for i in self.relation_to_idx.keys()], n)

    def is_seen(self, entity: str = None, relation: str = None) -> bool:
        """Check whether an entity or relation was present in the training set.

        Exactly one of *entity* or *relation* should be provided.

        Parameters
        ----------
        entity : str, optional
            Entity string label to look up.
        relation : str, optional
            Relation string label to look up.

        Returns
        -------
        bool
            ``True`` if the given string is in the respective index mapping,
            ``False`` otherwise.
        """
        if entity is not None:
            return True if self.entity_to_idx.get(entity) else False
        if relation is not None:
            return True if self.relation_to_idx.get(relation) else False

    def save(self) -> None:
        """Persist the current model weights to the experiment directory.

        The checkpoint filename encodes the current timestamp so successive
        calls do not overwrite each other.  Ensemble models are saved with an
        ``_ensemble_`` infix in the filename.
        """
        t = str(datetime.datetime.now())
        if self.construct_ensemble:
            save_checkpoint_model(self.model, path=self.path + f'/model_ensemble_interactive_{str(t)}.pt')
        else:
            save_checkpoint_model(self.model, path=self.path + f'/model_interactive_{str(t)}.pt')

    def get_entity_index(self, x: str) -> int:
        """Return the integer index for a given entity string.

        Parameters
        ----------
        x : str
            Entity string label (must have been seen during training).

        Returns
        -------
        int
            Corresponding row index in the entity embedding matrix.
        """
        return self.entity_to_idx[x]

    def get_relation_index(self, x: str) -> int:
        """Return the integer index for a given relation string.

        Parameters
        ----------
        x : str
            Relation string label (must have been seen during training).

        Returns
        -------
        int
            Corresponding row index in the relation embedding matrix.
        """
        return self.relation_to_idx[x]

    def index_triple(self, head_entity: List[str], relation: List[str], tail_entity: List[str]) -> Tuple[
        torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Convert string triple lists to integer index tensors.

        Parameters
        ----------
        head_entity : List[str]
            Head entity string labels.
        relation : List[str]
            Relation string labels.
        tail_entity : List[str]
            Tail entity string labels.

        Returns
        -------
        idx_head_entity, idx_relation, idx_tail_entity : torch.LongTensor
            Each has shape ``(n, 1)`` containing the integer indices for the
            corresponding strings.
        """
        n = len(head_entity)
        assert n == len(relation) == len(tail_entity)
        idx_head_entity = torch.LongTensor([self.entity_to_idx[i] for i in head_entity]).reshape(n, 1)
        idx_relation = torch.LongTensor([self.relation_to_idx[i] for i in relation]).reshape(n, 1)
        idx_tail_entity = torch.LongTensor([self.entity_to_idx[i] for i in tail_entity]).reshape(n, 1)
        return idx_head_entity, idx_relation, idx_tail_entity

    def add_new_entity_embeddings(self, entity_name: str = None, embeddings: torch.FloatTensor = None) -> None:
        """Extend the entity embedding table with a new entity at inference time.

        The new entity is appended to both ``entity_to_idx`` / ``idx_to_entity``
        mappings and the ``entity_embeddings`` weight tensor so that subsequent
        calls to prediction methods can reference it by name.

        Parameters
        ----------
        entity_name : str
            String label for the new entity.  If the entity already exists in
            the index no modification is made.
        embeddings : torch.FloatTensor
            1-D float tensor of length ``embedding_dim`` containing the
            pre-computed embedding for the new entity.
        """
        assert isinstance(entity_name, str) and isinstance(embeddings, torch.FloatTensor)

        if entity_name in self.entity_to_idx:
            print(f'Entity ({entity_name}) exists..')
        else:
            self.entity_to_idx[entity_name] = len(self.entity_to_idx)
            self.idx_to_entity[self.entity_to_idx[entity_name]] = entity_name
            self.num_entities += 1
            self.model.num_entities += 1
            self.model.entity_embeddings.weight.data = torch.cat(
                (self.model.entity_embeddings.weight.data.detach(), embeddings.unsqueeze(0)), dim=0)
            self.model.entity_embeddings.num_embeddings += 1

    def get_entity_embeddings(self, items: List[str]) -> torch.FloatTensor:
        """Return the embedding vectors for the given entity strings.

        For standard (non-BPE) models the method looks up each string in
        ``entity_to_idx`` and returns the corresponding rows of the entity
        embedding matrix.  For BPE models subword token embeddings are
        fetched and flattened into a single vector per entity.

        Parameters
        ----------
        items : List[str]
            Entity string labels to retrieve.

        Returns
        -------
        torch.FloatTensor
            Shape ``(len(items), embedding_dim)``.
        """
        if self.configs["byte_pair_encoding"]:
            t_encode = self.enc.encode_batch(items)
            if len(t_encode) != self.configs["max_length_subword_tokens"]:
                for i in range(len(t_encode)):
                    t_encode[i].extend(
                        [self.dummy_id for _ in range(self.configs["max_length_subword_tokens"] - len(t_encode[i]))])
            return self.model.token_embeddings(torch.LongTensor(t_encode)).flatten(1)
        else:
            return self.model.entity_embeddings(torch.LongTensor([self.entity_to_idx[i] for i in items]))

    def get_relation_embeddings(self, items: List[str]) -> torch.FloatTensor:
        """Return the embedding vectors for the given relation strings.

        Parameters
        ----------
        items : List[str]
            Relation string labels to retrieve.

        Returns
        -------
        torch.FloatTensor
            Shape ``(len(items), embedding_dim)``.
        """
        return self.model.relation_embeddings(torch.LongTensor([self.relation_to_idx[i] for i in items]))

    def construct_input_and_output(self, head_entity: List[str], relation: List[str], tail_entity: List[str],
                                    labels) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Build an indexed triple tensor and a label tensor from string inputs.

        Parameters
        ----------
        head_entity : List[str]
            Head entity string labels.
        relation : List[str]
            Relation string labels.
        tail_entity : List[str]
            Tail entity string labels.
        labels : list or array-like
            Binary or soft labels (one per triple) used as training targets.

        Returns
        -------
        x : torch.LongTensor
            Shape ``(n, 3)`` integer-indexed triples.
        labels : torch.FloatTensor
            Shape ``(n,)`` float label tensor.
        """
        idx_head_entity, idx_relation, idx_tail_entity = self.index_triple(head_entity, relation, tail_entity)
        x = torch.hstack((idx_head_entity, idx_relation, idx_tail_entity))
        # Hard Labels
        labels: object = torch.FloatTensor(labels)
        return x, labels

    def parameters(self):
        return self.model.parameters()

class InteractiveQueryDecomposition:
    """Mixin that provides fuzzy-logic operators for multi-hop EPFO query answering.

    The three families of operators — T-norm, T-conorm, and negation norm — are
    applied element-wise over entity score tensors to compose complex queries
    from atomic link-prediction results (e.g. 2p, 3p, 2i, ip, up).
    """

    def t_norm(self, tens_1: torch.Tensor, tens_2: torch.Tensor, tnorm: str = 'min') -> torch.Tensor:
        """Apply a T-norm to combine two entity score distributions.

        Parameters
        ----------
        tens_1, tens_2 : torch.Tensor
            Score tensors of identical shape, values in ``[0, 1]``.
        tnorm : str
            Operator to use.  ``'min'`` applies the Gödel (min) T-norm;
            ``'prod'`` applies the product T-norm.

        Returns
        -------
        torch.Tensor
            Element-wise combined scores of the same shape as the inputs.
        """
        if 'min' in tnorm:
            return torch.min(tens_1, tens_2)
        elif 'prod' in tnorm:
            return tens_1 * tens_2

    def tensor_t_norm(self, subquery_scores: torch.FloatTensor, tnorm: str = "min") -> torch.FloatTensor:
        """
        Compute T-norm over [0,1] ^{n \times d} where n denotes the number of hops and d denotes number of entities
        """
        if "min" == tnorm:
            return torch.min(subquery_scores, dim=0)
        elif "prod" == tnorm:
            raise NotImplementedError("Product T-norm is not implemented")
        else:
            raise NotImplementedError(f"{tnorm} is not implemented")

    def t_conorm(self, tens_1: torch.Tensor, tens_2: torch.Tensor, tconorm: str = 'min') -> torch.Tensor:
        """Apply a T-conorm (S-norm) to combine two score distributions (union).

        Parameters
        ----------
        tens_1, tens_2 : torch.Tensor
            Score tensors of identical shape, values in ``[0, 1]``.
        tconorm : str
            Operator to use.  ``'min'`` applies the Gödel (max) T-conorm;
            ``'prod'`` applies the probabilistic sum T-conorm.

        Returns
        -------
        torch.Tensor
            Element-wise combined scores of the same shape as the inputs.
        """
        if 'min' in tconorm:
            return torch.max(tens_1, tens_2)
        elif 'prod' in tconorm:
            return (tens_1 + tens_2) - (tens_1 * tens_2)

    def negnorm(self, tens_1: torch.Tensor, lambda_: float, neg_norm: str = 'standard') -> torch.Tensor:
        """Apply a negation norm (complement) to an entity score distribution.

        Parameters
        ----------
        tens_1 : torch.Tensor
            Input score tensor, values in ``[0, 1]``.
        lambda_ : float
            Shape parameter used by the Sugeno and Yager negation norms.
            Ignored for the standard complement.
        neg_norm : str
            Which negation to apply: ``'standard'`` (``1 - x``),
            ``'sugeno'``, or ``'yager'``.

        Returns
        -------
        torch.Tensor
            Complemented score tensor of the same shape as *tens_1*.
        """
        if 'standard' in neg_norm:
            return 1 - tens_1
        elif 'sugeno' in neg_norm:
            return (1 - tens_1) / (1 + lambda_ * tens_1)
        elif 'yager' in neg_norm:
            return (1 - torch.pow(tens_1, lambda_)) ** (1 / lambda_)

class AbstractCallback(ABC, lightning.pytorch.callbacks.Callback):
    """Abstract base class for KGE training lifecycle callbacks.

    Concrete sub-classes override one or more hook methods to perform
    custom actions at specific points during training (e.g. weight
    averaging, periodic evaluation, model checkpointing).  All hooks have
    empty default implementations so sub-classes only need to override the
    hooks they care about.

    Callbacks are registered by passing them to the trainer's *callbacks*
    list.  They are also compatible with PyTorch Lightning trainers because
    this class extends ``lightning.pytorch.callbacks.Callback``.
    """

    def __init__(self):
        pass

    def on_init_start(self, *args, **kwargs):
        """Called when the trainer is about to be constructed.

        Override to perform setup that must happen before any trainer
        state is initialised.
        """
        pass

    def on_init_end(self, *args, **kwargs):
        """Called immediately after the trainer has been constructed.

        Override to perform setup that requires a fully initialised trainer.
        """
        pass

    def on_fit_start(self, trainer, model):
        """Called once before the first training epoch.

        Parameters
        ----------
        trainer : AbstractTrainer or pl.Trainer
            The active trainer instance.
        model : BaseKGE
            The model about to be trained.
        """
        return

    def on_train_epoch_end(self, trainer, model):
        """Called at the end of each training epoch.

        Parameters
        ----------
        trainer : AbstractTrainer or pl.Trainer
            The active trainer instance.
        model : BaseKGE
            The model being trained.  ``model.loss_history`` contains the
            per-epoch average losses accumulated so far.
        """
        pass

    def on_train_batch_end(self, *args, **kwargs):
        """Called after each mini-batch gradient update.

        Override to inspect or modify the model at a finer granularity than
        epoch-level hooks.
        """
        pass

    def on_fit_end(self, *args, **kwargs):
        """Called once after the final training epoch completes.

        Override to perform post-training actions such as saving the final
        model state, computing evaluation metrics, or cleaning up resources.
        """
        pass


class AbstractPPECallback(AbstractCallback):
    """Abstract base class for Post-training Parameter Ensembling (PPE) callbacks.

    Sub-classes implement weight-averaging strategies (SWA, EMA, SWAG, …) by
    overriding :meth:`on_train_epoch_end` and :meth:`on_fit_end`.  Common
    book-keeping (epoch counter, sample counter, alpha weights) is managed
    here.

    Parameters
    ----------
    num_epochs : int
        Total number of training epochs.
    path : str
        Experiment directory where averaged checkpoints will be written.
    epoch_to_start : int
        First epoch at which the averaging procedure should begin.
    last_percent_to_consider : float
        Fraction of the total training epochs (counted from the end) whose
        checkpoints are included in the ensemble.
    """

    def __init__(self, num_epochs, path, epoch_to_start, last_percent_to_consider):
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
                raise AssertionError(f"--epoch_to_start {self.epoch_to_start} "
                                     f"must be less than --num_epochs {self.num_epochs}")
            self.num_ensemble_coefficient = self.num_epochs - self.epoch_to_start + 1
        elif last_percent_to_consider is None:
            self.epoch_to_start = 1
            self.num_ensemble_coefficient = self.num_epochs - 1
        else:
            # Compute the last X % of the training
            self.epoch_to_start = self.num_epochs - int(self.num_epochs * last_percent_to_consider / 100)
            self.num_ensemble_coefficient = self.num_epochs - self.epoch_to_start

    def on_fit_start(self, trainer, model):
        pass

    def on_fit_end(self, trainer, model):
        if os.path.exists(f"{self.path}/trainer_checkpoint_main.pt"):
            param_ensemble = torch.load(f"{self.path}/trainer_checkpoint_main.pt", torch.device("cpu"))
            model.load_state_dict(param_ensemble)
        else:
            print(f"No parameter ensemble found at {self.path}/trainer_checkpoint_main.pt")

    def store_ensemble(self, param_ensemble) -> None:
        # (3) Save the updated parameter ensemble model.
        torch.save(param_ensemble, f=f"{self.path}/trainer_checkpoint_main.pt")
        if self.sample_counter > 1:
            self.sample_counter += 1

class BaseInteractiveTrainKGE:
    """
    Abstract/base class for training knowledge graph embedding models interactively.
    This class provides methods for re-training KGE models and also Literal Embedding model.
    """

    # @TODO: Do we really need this ?!
    def train_triples(self, h: List[str], r: List[str], t: List[str], labels: List[float],
                      iteration=2, optimizer=None):
        assert len(h) == len(r) == len(t) == len(labels)
        # (1) From List of strings to TorchLongTensor.
        x = torch.LongTensor(self.index_triple(h, r, t)).reshape(1, 3)
        # (2) From List of float to Torch Tensor.
        labels = torch.FloatTensor(labels)
        # (3) Train mode.
        self.set_model_train_mode()
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        print('Iteration starts...')
        # (4) Train.
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            print(f"Iteration:{epoch}\t Loss:{loss.item()}\t Outputs:{outputs.detach().mean()}")
            loss.backward()
            optimizer.step()
        # (5) Eval
        self.set_model_eval_mode()
        with torch.no_grad():
            x = x.to(self.model.device)
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            print(f"Eval Mode:\tLoss:{loss.item()}")

    def train_k_vs_all(self, h, r, iteration=1, lr=.001):
        """
        Train k vs all
        :param head_entity:
        :param relation:
        :param iteration:
        :param lr:
        :return:
        """
        assert len(h) == 1
        # (1) Construct input and output
        out = self.construct_input_and_output_k_vs_all(h, r)
        if out is None:
            return
        x, labels, idx_tails = out
        # (2) Train mode
        self.set_model_train_mode()
        # (3) Initialize optimizer # SGD considerably faster than ADAM.
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=.00001)

        print('\nIteration starts.')
        # (3) Iterative training.
        for epoch in range(iteration):
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
            if len(idx_tails) > 0:
                print(
                    f"Iteration:{epoch}\t"
                    f"Loss:{loss.item()}\t"
                    f"Avg. Logits for correct tails: {outputs[0, idx_tails].flatten().mean().detach()}")
            else:
                print(
                    f"Iteration:{epoch}\t"
                    f"Loss:{loss.item()}\t"
                    f"Avg. Logits for all negatives: {outputs[0].flatten().mean().detach()}")

            loss.backward()
            optimizer.step()
            if loss.item() < .00001:
                print(f'loss is {loss.item():.3f}. Converged !!!')
                break
        # (4) Eval mode
        self.set_model_eval_mode()
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.model.loss(outputs, labels)
        print(f"Eval Mode:Loss:{loss.item():.4f}\t Outputs:{outputs[0, idx_tails].flatten().detach()}\n")

    def train(self, kg, lr=.1, epoch=10, batch_size=32, neg_sample_ratio=10, num_workers=1) -> None:
        """ Retrained a pretrain model on an input KG via negative sampling."""
        # (1) Create Negative Sampling Setting for training
        print('Creating Dataset...')
        train_set = TriplePredictionDataset(kg.train_set,
                                            num_entities=len(kg.entity_to_idx),
                                            num_relations=len(kg.relation_to_idx),
                                            neg_sample_ratio=neg_sample_ratio)
        num_data_point = len(train_set)
        print('Number of data points: ', num_data_point)
        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                                      #  shuffle => to have the data reshuffled at every epoc
                                      shuffle=True, num_workers=num_workers,
                                      collate_fn=train_set.collate_fn, pin_memory=True)

        # (2) Go through valid triples + corrupted triples and compute scores.
        # Average loss per triple is stored. This will be used  to indicate whether we learned something.
        print('First Eval..')
        self.set_model_eval_mode()
        first_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            first_avg_loss_per_triple += self.model.loss(pred, y)
        first_avg_loss_per_triple /= num_data_point
        print(first_avg_loss_per_triple)
        # (3) Prepare Model for Training
        self.set_model_train_mode()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        print('Training Starts...')
        for epoch in range(epoch):  # loop over the dataset multiple times
            epoch_loss = 0
            for x, y in train_dataloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(x)
                loss = self.model.loss(outputs, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Epoch={epoch}\t Avg. Loss per epoch: {epoch_loss / num_data_point:.3f}')
        # (5) Prepare For Saving
        self.set_model_eval_mode()
        print('Eval starts...')
        # (6) Eval model on training data to check how much an Improvement
        last_avg_loss_per_triple = 0
        for x, y in train_dataloader:
            pred = self.model(x)
            last_avg_loss_per_triple += self.model.loss(pred, y)
        last_avg_loss_per_triple /= len(train_set)
        print(f'On average Improvement: {first_avg_loss_per_triple - last_avg_loss_per_triple:.3f}')

    def train_literals(
        self,
        train_file_path: str = None,
        num_epochs: int = 100,
        lit_lr: float = 0.001,
        lit_normalization_type: str = "z-norm",
        batch_size: int = 1024,
        sampling_ratio: float = None,
        random_seed=1,
        loader_backend: str = "pandas",
        freeze_entity_embeddings: bool = True,
        gate_residual: bool = True,
        device: str = None,
        suffle_data: bool = True
    ):
        """
        Trains the Literal Embeddings model using literal data.

        Args:
            train_file_path (str): Path to the training data file.
            num_epochs (int): Number of training epochs.
            lit_lr (float): Learning rate for the literal model.
            norm_type (str): Normalization type to use ('z-norm', 'min-max', or None).
            batch_size (int): Batch size for training.
            sampling_ratio (float): Ratio of training triples to use.
            loader_backend (str): Backend for loading the dataset ('pandas' or 'rdflib').
            freeze_entity_embeddings (bool): If True, freeze the entity embeddings during training.
            gate_residual (bool): If True, use gate residual connections in the model.
            device (str): Device to use for training ('cuda' or 'cpu'). If None, will use available GPU or CPU.
            suffle_data (bool): If True, shuffle the dataset before training.
        """
        # Assign torch.seed to reproduice experiments
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # Set the device for training
        try:
            device = torch.device(device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
        # Prepare the dataset and DataLoader
        literal_dataset = LiteralDataset(
            file_path=train_file_path,
            ent_idx=self.entity_to_idx,
            normalization_type=lit_normalization_type,
            sampling_ratio=sampling_ratio,
            loader_backend=loader_backend,
        )

        self.data_property_to_idx = literal_dataset.data_property_to_idx

        batch_data = DataLoader(
            dataset=literal_dataset,
            shuffle=suffle_data,
            batch_size=batch_size,
        )

        # Initialize Literal Embedding model
        literal_model = LiteralEmbeddings(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=self.model.embedding_dim,
            entity_embeddings=self.model.entity_embeddings,
            freeze_entity_embeddings=freeze_entity_embeddings,
            gate_residual=gate_residual
        ).to(device)

        optimizer = optim.Adam(literal_model.parameters(), lr=lit_lr)
        loss_log = {"lit_loss": []}
        literal_model.train()

        print(
            f"Training Literal Embedding model"
            f" using pre-trained '{self.model.name}' embeddings."
        )

        # Training loop
        for epoch in (tqdm_bar := tqdm(range(num_epochs))):
            epoch_loss = 0
            for batch_x, batch_y in batch_data:
                optimizer.zero_grad()
                lit_entities = batch_x[:, 0].long().to(device)
                lit_properties = batch_x[:, 1].long().to(device)
                batch_y = batch_y.to(device)
                yhat = literal_model(lit_entities, lit_properties)
                lit_loss = F.l1_loss(yhat, batch_y)
                lit_loss.backward()
                optimizer.step()
                epoch_loss += lit_loss

            avg_epoch_loss = epoch_loss / len(batch_data)
            tqdm_bar.set_postfix_str(f"loss_lit={lit_loss:.5f}")
            loss_log["lit_loss"].append(avg_epoch_loss.item())

        self.literal_model = literal_model
        self.literal_dataset = literal_dataset
        torch.save(literal_model.state_dict(), self.path + "/literal_model.pt")
        print(f"Literal Embedding model saved to {self.path}/literal_model.pt")
        self.idx_to_data_property = {v: k for k, v in self.data_property_to_idx.items()}
        df = pd.DataFrame.from_dict(self.idx_to_data_property, orient="index", columns=["attribute"])
        df.to_csv(self.path + "/attribute_to_idx.csv")
        print(f"Literal attributes indexing saved to {self.path}/attribute_to_idx.csv")

