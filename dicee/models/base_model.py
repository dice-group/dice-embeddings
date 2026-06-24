from typing import Any, Dict, List, Tuple, Union

import lightning as pl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adopt import ADOPT


class BaseKGELightning(pl.LightningModule):
    """Thin PyTorch Lightning wrapper shared by all KGE models.

    Provides the standard Lightning training loop hooks (``training_step``,
    ``on_train_epoch_end``, ``configure_optimizers``) as well as a helper
    for reporting model size.  All concrete KGE models should extend
    :class:`BaseKGE` rather than this class directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_step_outputs = []

    def mem_of_model(self) -> Dict:
        """ Size of model in MB and number of params"""
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        # (2) Store NumParam and EstimatedSizeMB
        num_params = sum(p.numel() for p in self.parameters())
        # Not quite sure about EstimatedSizeMB ?
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return {'EstimatedSizeMB': (num_params + buffer_size) / 1024 ** 2, 'NumParam': num_params}

    def training_step(self, batch, batch_idx=None):
        """Execute one optimisation step for the given mini-batch.

        Handles two- and three-element batches produced by the different
        dataset classes (``KvsAll`` / ``NegSample`` vs. ``KvsSample``).

        Parameters
        ----------
        batch : tuple
            ``(x, y)`` for standard scoring, or ``(x, y_select, y)`` for
            sample-based labelling.
        batch_idx : int, optional
            Index of the current batch (unused, kept for Lightning API compat).

        Returns
        -------
        torch.FloatTensor
            Scalar loss value for this batch.
        """
        if len(batch)==2:
            # Default
            x_batch, y_batch = batch
            yhat_batch = self.forward(x_batch)
        elif len(batch)==3:
            # KvsSample or 1vsSample
            x_batch, y_select, y_batch = batch
            yhat_batch = self.forward((x_batch,y_select))
        else:
            raise RuntimeError("Invalid batch received.")

        loss_batch = self.loss_function(yhat_batch, y_batch)
        self.training_step_outputs.append(loss_batch.item())
        # Only log when using PyTorch Lightning trainer
        # Check private _trainer attribute to avoid RuntimeError from property getter
        if hasattr(self, '_trainer') and self._trainer is not None:
            self.log("loss",
                     value=loss_batch,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     sync_dist=True,
                     logger=False)
        return loss_batch

    def loss_function(self, yhat_batch: torch.FloatTensor, y_batch: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the loss between model predictions and targets.

        Delegates to ``self.loss`` which is configured in
        :class:`BaseKGE.__init__` based on the scoring technique
        (``BCEWithLogitsLoss`` for entity/relation prediction,
        ``CrossEntropyLoss`` for classification).

        Parameters
        ----------
        yhat_batch : torch.FloatTensor
            Model output scores, shape ``(batch_size, *)``.
        y_batch : torch.FloatTensor
            Ground-truth labels of the same shape as *yhat_batch*.

        Returns
        -------
        torch.FloatTensor
            Scalar loss value.
        """
        return self.loss(yhat_batch, y_batch)

    def on_train_epoch_end(self, *args, **kwargs):
        if len(args) >= 1:
            raise RuntimeError(f"Arguments must not be empty:{args}")
        if len(kwargs) >= 1:
            raise RuntimeError(f"Keyword Arguments must not be empty:{kwargs}")

        self.loss_history.append(sum(self.training_step_outputs) / len(self.training_step_outputs))
        self.training_step_outputs.clear()

    def test_epoch_end(self, outputs: List[Any]):
        """ """

    def test_dataloader(self) -> None:
        pass

    def val_dataloader(self) -> None:
        pass

    def predict_dataloader(self) -> None:
        pass

    def train_dataloader(self) -> None:
        pass

    def configure_optimizers(self, parameters=None):
        """Instantiate and return the optimiser for training.

        The optimiser type is taken from ``self.optimizer_name`` which is set
        in :meth:`BaseKGE.init_params_with_sanity_checking` from the
        ``--optim`` argument.  Supported values: ``'SGD'``, ``'Adam'``,
        ``'Adopt'``, ``'AdamW'``, ``'NAdam'``, ``'Adagrad'``, ``'ASGD'``,
        ``'Muon'``.

        Parameters
        ----------
        parameters : iterable, optional
            Model parameters to optimise.  Defaults to
            ``self.parameters()`` when ``None``.

        Returns
        -------
        torch.optim.Optimizer
            The configured optimiser instance.
        """
        if parameters is None:
            parameters = self.parameters()

        # default params in pytorch.
        if self.optimizer_name == 'SGD':
            self.selected_optimizer = torch.optim.SGD(params=parameters, lr=self.learning_rate,
                                                      momentum=0, dampening=0, weight_decay=self.weight_decay,
                                                      nesterov=False)

        elif self.optimizer_name == 'Adam':
            self.selected_optimizer = torch.optim.Adam(parameters, lr=self.learning_rate,
                                                       weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adopt':
            self.selected_optimizer = ADOPT(parameters, lr=self.learning_rate)
        elif self.optimizer_name == 'AdamW':
            self.selected_optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate,
                                                       weight_decay=self.weight_decay)
        elif self.optimizer_name == 'NAdam':
            self.selected_optimizer = torch.optim.NAdam(parameters, lr=self.learning_rate, betas=(0.9, 0.999),
                                                        eps=1e-08, weight_decay=self.weight_decay, momentum_decay=0.004)
        elif self.optimizer_name == 'Adagrad':
            self.selected_optimizer = torch.optim.Adagrad(parameters,
                                                          lr=self.learning_rate, eps=1e-10,
                                                          weight_decay=self.weight_decay)
        elif self.optimizer_name == 'ASGD':
            self.selected_optimizer = torch.optim.ASGD(parameters,
                                                       lr=self.learning_rate, lambd=0.0001, alpha=0.75,
                                                       weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Muon':
            self.selected_optimizer = torch.optim.Muon(parameters, lr=self.learning_rate,
                                                       weight_decay=self.weight_decay)
        else:
            raise KeyError(f"{self.optimizer_name} is not found!")
        print(self.selected_optimizer)
        return self.selected_optimizer


class BaseKGE(BaseKGELightning):
    """Base class for all Knowledge Graph Embedding models.

    Inherits the Lightning training loop from :class:`BaseKGELightning` and
    adds the embedding tables, normalisation / dropout layers, and the
    routing logic that dispatches ``forward()`` calls to the appropriate
    scoring method.

    Sub-classes must implement at minimum:

    * :meth:`forward_triples` — score a batch of ``(h, r, t)`` triples.
    * :meth:`forward_k_vs_all` — score a ``(h, r)`` batch against every entity.

    Parameters
    ----------
    args : dict
        Flat configuration dictionary produced by
        ``vars(argparse.Namespace)``.
        Required keys: ``embedding_dim``, ``num_entities``, ``num_relations``,
        ``learning_rate`` (or ``lr``), ``optim``, ``scoring_technique``.
    """

    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.embedding_dim = None
        self.num_entities = None
        self.num_relations = None
        self.num_tokens = None
        self.learning_rate = None
        self.apply_unit_norm = None
        self.input_dropout_rate = None
        self.hidden_dropout_rate = None
        self.optimizer_name = None
        self.feature_map_dropout_rate = None
        self.kernel_size = None
        self.num_of_output_channels = None
        self.weight_decay = None
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.selected_optimizer = None
        self.normalizer_class = None
        self.normalize_head_entity_embeddings = IdentityClass()
        self.normalize_relation_embeddings = IdentityClass()
        self.normalize_tail_entity_embeddings = IdentityClass()
        self.hidden_normalizer = IdentityClass()
        self.param_init = IdentityClass
        self.init_params_with_sanity_checking()

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(self.input_dropout_rate)
        # average minibatch loss per epoch
        self.loss_history = []
        self.byte_pair_encoding = self.args.get("byte_pair_encoding", False)
        self.max_length_subword_tokens = self.args.get("max_length_subword_tokens", None)
        self.block_size=self.args.get("block_size", None)
        self.defer_large_embeddings = bool(self.args.get("fsdp_sharded_entity", False))
        if self.byte_pair_encoding and self.args['model'] != "BytE":
            self.token_embeddings = torch.nn.Embedding(self.num_tokens, self.embedding_dim)
            self.param_init(self.token_embeddings.weight.data)

            # Reduces subword units embedding matrix from T x D into D.
            self.lf = nn.Sequential(nn.Linear(self.embedding_dim * self.max_length_subword_tokens,
                                              self.embedding_dim, bias=False))
            if self.args["scoring_technique"] in ["AllvsAll", "KvsAll"]:
                self.str_to_bpe_entity_to_idx = {str_ent: idx for idx, (str_ent, bpe_ent, shaped_bpe_ent) in
                                                 enumerate(self.args["ordered_bpe_entities"])}
                self.bpe_entity_to_idx = {shaped_bpe_ent: idx for idx, (str_ent, bpe_ent, shaped_bpe_ent) in
                                          enumerate(self.args["ordered_bpe_entities"])}
                self.ordered_bpe_entities = torch.tensor(list(self.bpe_entity_to_idx.keys()), dtype=torch.long)
        elif self.byte_pair_encoding and self.args['model'] == "BytE":
            """ Transformer implements token embeddings"""
        else:
            if self.defer_large_embeddings:
                self.entity_embeddings = None
                self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
                self.param_init(self.relation_embeddings.weight.data)
            else:
                self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
                self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
                self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def forward_byte_pair_encoded_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """KvsAll scoring for BPE-encoded head entities and relations.

        Retrieves subword-unit embeddings for the head entity and relation,
        reduces them to fixed-size vectors via a linear projection, then
        scores against all BPE entity embeddings.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 2, T)`` BPE token indices where dim 1
            indexes ``[head, relation]`` and *T* is
            ``max_length_subword_tokens``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_bpe_entities)`` score matrix.
        """
        # (1) Get unit normalized subword units embedding matrices: (B, T, D)
        bpe_head_ent_emb, bpe_rel_ent_emb = self.get_bpe_head_and_relation_representation(x)
        # Future work: Use attention to model similarity between subword units comprising head and relation
        # attentive_head_rel_emb = self.attention_block(torch.cat((bpe_head_ent_emb, bpe_rel_ent_emb), 1))
        # bpe_head_ent_emb = attentive_head_rel_emb[:, :self.max_length_subword_tokens, :]
        # bpe_rel_ent_emb = attentive_head_rel_emb[:, self.max_length_subword_tokens:, :]

        # (2) Reshaping (1) into row vectors.
        B, T, D = bpe_head_ent_emb.shape

        # Multi-node GPU setting.
        device_r = bpe_head_ent_emb.get_device()
        if device_r >= 0:
            self.ordered_bpe_entities = self.ordered_bpe_entities.to(device_r)
        else:
            self.ordered_bpe_entities = self.ordered_bpe_entities.to("cpu")

        # (3) Get unit normalized subword units embedding matrices of all entities : (E, T, D)
        E = self.token_embeddings(self.ordered_bpe_entities)
        # (4) Reshaping (3) into row vectors (E, T*D) .
        E = E.reshape(len(E), T * D)

        # (5) Reshape and Reduce from (_, T*D) into row vectors.
        bpe_head_ent_emb = self.input_dp_ent_real(bpe_head_ent_emb.reshape(B, T * D))
        bpe_rel_ent_emb = self.input_dp_rel_real(bpe_rel_ent_emb.reshape(B, T * D))
        bpe_head_ent_emb = self.lf(bpe_head_ent_emb)
        bpe_rel_ent_emb = self.lf(bpe_rel_ent_emb)
        E = self.lf(E)

        return self.k_vs_all_score(bpe_head_ent_emb, bpe_rel_ent_emb, E)


    def forward_byte_pair_encoded_triple(self, x: Tuple[torch.LongTensor, torch.LongTensor]) -> torch.FloatTensor:
        """NegSample scoring for BPE-encoded ``(head, relation, tail)`` triples.

        Retrieves subword-unit embeddings for all three elements and reduces
        them to fixed-size vectors via a linear projection before computing
        the triple score.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 3, T)`` BPE token indices.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size,)`` triple scores.
        """
        bpe_head_ent_emb, bpe_rel_ent_emb, bpe_tail_ent_emb = self.get_sentence_representation(x)
        B, T, C = bpe_head_ent_emb.shape
        bpe_head_ent_emb = bpe_head_ent_emb.reshape(B, T * C)
        bpe_rel_ent_emb = bpe_rel_ent_emb.reshape(B, T * C)
        bpe_tail_ent_emb = bpe_tail_ent_emb.reshape(B, T * C)
        bpe_triple_score = self.score(self.lf(bpe_head_ent_emb), self.lf(bpe_rel_ent_emb), self.lf(bpe_tail_ent_emb))
        return bpe_triple_score

    def init_params_with_sanity_checking(self) -> None:
        """Populate model hyper-parameters from ``self.args`` with safe defaults.

        Reads embedding dimension, learning rate, dropout rates, normalisation
        strategy, optimizer name, and parameter initialisation scheme from the
        ``args`` dict.  Falls back to sensible defaults for any missing key so
        that minimal ``args`` dicts (e.g. for unit tests) are still valid.
        """
        if self.args.get('weight_decay'):
            self.weight_decay = self.args['weight_decay']
        else:
            self.weight_decay = 0.0
        if self.args.get('embedding_dim'):
            self.embedding_dim = self.args['embedding_dim']
        else:
            self.embedding_dim = 1

        self.num_entities = self.args.get('num_entities', None)
        self.num_relations = self.args.get('num_relations', None)
        self.num_tokens = self.args.get('num_tokens', None)

        if self.args.get('learning_rate'):
            self.learning_rate = self.args['learning_rate']
        else:
            self.learning_rate = .1

        if self.args.get("input_dropout_rate"):
            self.input_dropout_rate = self.args['input_dropout_rate']
        else:
            self.input_dropout_rate = 0.0
        if self.args.get("hidden_dropout_rate"):
            self.hidden_dropout_rate = self.args['hidden_dropout_rate']
        else:
            self.hidden_dropout_rate = 0.0
        if self.args.get("model") in ['ConvQ', 'ConvO', 'ConEx', 'AConEx', 'AConvQ', 'AConvO']:
            if self.args.get("kernel_size"):
                self.kernel_size = self.args['kernel_size']
            else:
                self.kernel_size = 3
            if self.args.get("num_of_output_channels"):
                self.num_of_output_channels = self.args['num_of_output_channels']
            else:
                self.num_of_output_channels = 3
            if self.args.get("feature_map_dropout_rate"):
                self.feature_map_dropout_rate = self.args['feature_map_dropout_rate']
            else:
                self.feature_map_dropout_rate = 0.0

        if self.args.get("normalization") == 'LayerNorm':
            self.normalizer_class = torch.nn.LayerNorm
            self.normalize_head_entity_embeddings = self.normalizer_class(self.embedding_dim)
            self.normalize_relation_embeddings = self.normalizer_class(self.embedding_dim)
            if self.args['scoring_technique'] in ['NegSample', 'FixedNegSample', 'KvsSample', 'FSDP1vsSample']:
                self.normalize_tail_entity_embeddings = self.normalizer_class(self.embedding_dim)
        elif self.args.get("normalization") == 'BatchNorm1d':
            self.normalizer_class = torch.nn.BatchNorm1d
            self.normalize_head_entity_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
            self.normalize_relation_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
            if self.args['scoring_technique'] in ['NegSample', 'FixedNegSample', 'KvsSample', 'FSDP1vsSample']:
                self.normalize_tail_entity_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
        elif self.args.get("normalization") is None:
            self.normalizer_class = IdentityClass
        else:
            raise NotImplementedError()

        self.optimizer_name = self.args.get('optim',None)

        if self.args.get("init_param") is None:
            self.param_init = IdentityClass
        elif self.args['init_param'] == 'xavier_normal':
            self.param_init = torch.nn.init.xavier_normal_
        else:
            print(f'--init_param (***{self.args.get("init_param")}***) not found')
            self.optimizer_name = IdentityClass

    def forward(self, x: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]],
                y_idx: torch.LongTensor = None) -> torch.FloatTensor:
        """Route the forward pass to the appropriate scoring method.

        Inspects the shape and type of *x* to decide which low-level scorer
        to call:

        * Tuple ``(x, y_idx)``   → :meth:`forward_k_vs_sample`
        * ``(batch, 3)`` tensor  → :meth:`forward_triples`
        * ``(batch, 2)`` tensor  → :meth:`forward_k_vs_all`
        * BPE triple tensor      → :meth:`forward_byte_pair_encoded_triple`
        * BPE pair tensor        → :meth:`forward_byte_pair_encoded_k_vs_all`

        Parameters
        ----------
        x : torch.LongTensor or Tuple[torch.LongTensor, torch.LongTensor]
            Either a plain index tensor or a ``(triple_idx, target_idx)``
            tuple for sample-based labelling.
        y_idx : torch.LongTensor, optional
            Target entity indices used by :meth:`forward_k_vs_sample`.
            Ignored when *x* is a plain tensor.

        Returns
        -------
        torch.FloatTensor
            Score tensor whose shape depends on the selected scorer.
        """
        if isinstance(x, tuple):
            x, y_idx = x
            return self.forward_k_vs_sample(x=x, target_entity_idx=y_idx)
        else:
            shape_info = x.shape
            if len(shape_info) == 2:
                batch_size, dim = x.shape
                if dim == 3:
                    return self.forward_triples(x)
                elif dim == 2:
                    # h, y = x[0], x[1]
                    # Note that y can be relation or tail entity.
                    return self.forward_k_vs_all(x=x)
            else:
                size_of_input_data = shape_info[1]
                if size_of_input_data == 3:
                    # NegSample with BPE
                    return self.forward_byte_pair_encoded_triple(x=x)
                elif size_of_input_data == 2:
                    # KvsAll with BPE
                    return self.forward_byte_pair_encoded_k_vs_all(x)

    def forward_triples(self, x: torch.LongTensor) -> torch.Tensor:
        """Score a batch of ``(head, relation, tail)`` index triples.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 3)`` integer tensor where each row is
            ``[head_idx, relation_idx, tail_idx]``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size,)`` triple scores.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        h_emb, r_emb, t_emb = self.get_triple_representation(x)
        return self.score(h_emb, r_emb, t_emb)

    def forward_k_vs_all(self, *args, **kwargs):
        """Score a ``(head, relation)`` batch against every entity.

        Sub-classes must override this method.  The default implementation
        raises ``ValueError`` to make missing overrides obvious at runtime.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_entities)`` score matrix.
        """
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_all function')

    def forward_k_vs_sample(self, *args, **kwargs):
        """Score a ``(head, relation)`` batch against a sampled subset of entities.

        Used by ``KvsSample`` and ``1vsSample`` datasets.  Sub-classes that
        support sample-based labelling must override this method.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, k)`` score matrix where *k* is the number
            of sampled target entities.
        """
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_sample function')

    def get_triple_representation(self, idx_hrt) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Retrieve and normalise embedding vectors for a triple index batch.

        Parameters
        ----------
        idx_hrt : torch.LongTensor
            Shape ``(batch_size, 3)`` integer tensor with columns
            ``[head_idx, relation_idx, tail_idx]``.

        Returns
        -------
        head_ent_emb, rel_ent_emb, tail_ent_emb : torch.FloatTensor
            Each has shape ``(batch_size, embedding_dim)`` after applying the
            configured dropout and normalisation.
        """
        # (1) Split input into indexes.
        idx_head_entity, idx_relation, idx_tail_entity = idx_hrt[:, 0], idx_hrt[:, 1], idx_hrt[:, 2]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        tail_ent_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(idx_tail_entity))
        return head_ent_emb, rel_ent_emb, tail_ent_emb

    def get_head_relation_representation(self, indexed_triple) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Retrieve and normalise embedding vectors for head entities and relations.

        Parameters
        ----------
        indexed_triple : torch.LongTensor
            Shape ``(batch_size, 2)`` integer tensor with columns
            ``[head_idx, relation_idx]``.

        Returns
        -------
        head_ent_emb, rel_ent_emb : torch.FloatTensor
            Each has shape ``(batch_size, embedding_dim)`` after applying the
            configured dropout and normalisation.
        """
        # (1) Split input into indexes.
        idx_head_entity, idx_relation = indexed_triple[:, 0], indexed_triple[:, 1]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return head_ent_emb, rel_ent_emb

    def get_sentence_representation(self, x: torch.LongTensor) -> Tuple[
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Retrieve BPE subword-unit embeddings for a batch of triples.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 3, T)`` where *T* is
            ``max_length_subword_tokens``.

        Returns
        -------
        head_ent_emb, rel_emb, tail_emb : torch.FloatTensor
            Each has shape ``(batch_size, T, embedding_dim)``.
        """
        h, r, t = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        head_ent_emb = self.token_embeddings(h)
        rel_emb = self.token_embeddings(r)
        tail_emb = self.token_embeddings(t)
        return head_ent_emb, rel_emb, tail_emb

    def get_bpe_head_and_relation_representation(self, x: torch.LongTensor) -> Tuple[
            torch.FloatTensor, torch.FloatTensor]:
        """Retrieve unit-normalised BPE embeddings for head entities and relations.

        Each entity/relation is represented as a sequence of *T* subword
        tokens.  Their token embeddings are L2-normalised across the
        sequence dimension so that the resulting matrix has unit Frobenius
        norm.

        Parameters
        ----------
        x : torch.LongTensor
            Shape ``(batch_size, 2, T)`` where dim 1 indexes
            ``[head, relation]`` and *T* is ``max_length_subword_tokens``.

        Returns
        -------
        head_ent_emb, rel_emb : torch.FloatTensor
            Each has shape ``(batch_size, T, embedding_dim)``, L2-normalised
            over the ``(T, D)`` dimensions.
        """
        # h: batchsize, T where T represents the maximum shaped token size
        # h: B x T, r: B x T
        h, r = x[:, 0, :], x[:, 1, :]
        # B, T, D
        head_ent_emb = self.token_embeddings(h)
        # B, T, D
        rel_emb = self.token_embeddings(r)

        # A sequence of sub-list embeddings representing an embedding of a head entity should be normalized to 0.
        # Therefore, the norm of a row vector obtained from T by D matrix must be 1.
        # B, T, D
        head_ent_emb = F.normalize(head_ent_emb, p=2, dim=(1, 2))
        # B, T, D
        rel_emb = F.normalize(rel_emb, p=2, dim=(1, 2))
        return head_ent_emb, rel_emb

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the entity and relation embedding matrices as numpy arrays.

        Returns
        -------
        entity_embeddings : numpy.ndarray
            Shape ``(num_entities, embedding_dim)``.
        relation_embeddings : numpy.ndarray
            Shape ``(num_relations, embedding_dim)``.
        """
        return self.entity_embeddings.weight.data.data.detach(), self.relation_embeddings.weight.data.detach()


class IdentityClass(torch.nn.Module):
    """No-op normalisation / dropout placeholder.

    Used whenever no normalisation layer is requested (``--normalization None``).
    All inputs are returned unchanged so that the rest of the model code does
    not need conditional checks around normalisation calls.
    """

    def __init__(self, args=None):
        super().__init__()
        self.args = args

    def __call__(self, x):
        return x

    @staticmethod
    def forward(x):
        return x
