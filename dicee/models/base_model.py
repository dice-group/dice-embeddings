from typing import List, Any, Optional, Tuple, Union, Dict
import lightning as pl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Represents one head of self-attention.

    Parameters
    ----------
    head_size : int
        The size of the head.
    n_embd : int
        The size of the embedding.
    block_size : int
        The size of the block.

    Attributes
    ----------
    key : nn.Linear
        Linear transformation for the keys.
    query : nn.Linear
        Linear transformation for the queries.
    value : nn.Linear
        Linear transformation for the values.
    dropout : nn.Dropout
        A dropout layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Perform operations on a 3-dimensional input tensor `x` with shape `(batch, time-step, channels)`.
    """
    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform operations on a 3-dimensional input tensor `x` with shape `(batch, time-step, channels)`.
        The function computes key, query, and value representations of `x`, calculates attention weights, applies dropout, and finally performs a weighted aggregation of the values.

        Parameters
        ----------
        x : torch.Tensor
            A 3-dimensional input tensor with shape `(batch, time-step, channels)`.

        Returns
        -------
        torch.Tensor
            A 3-dimensional output tensor with shape `(batch, time-step, head size)` resulting from the weighted aggregation of the values.

        Notes
        -----
        The `forward` function is typically used in the context of attention mechanisms in transformer models.
        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """
    A module that implements multiple heads of self-attention in parallel.

    Parameters
    ----------
    num_heads : int
        The number of attention heads.
    head_size : int
        The size of each attention head.
    n_embd : int
        The size of the input embedding.
    block_size : int
        The size of the data block.

    Attributes
    ----------
    heads : nn.ModuleList
        A list of `num_heads` Head modules, each representing one head of self-attention.
    proj : nn.Linear
        Linear transformation for projecting the concatenated outputs of the attention heads.
    dropout : nn.Dropout
        A dropout layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Perform the forward pass of the multi-head attention module.

    Notes
    -----
    This module is typically used as part of the transformer architecture for sequence modeling tasks.
    """

    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, block_size=block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the MultiHeadAttention module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after the multi-head attention operation.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """
    A module that implements a simple linear layer followed by a non-linearity.

    Parameters
    ----------
    n_embd : int
        The size of the input embedding.

    Attributes
    ----------
    net : nn.Sequential
        A sequential module consisting of linear layers and non-linear activations.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Perform the forward pass of the feedforward module.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the feedforward module.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after the forward pass.
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: communication followed by computation

    Parameters
    ----------
    n_embd : int
        The embedding dimension.
    n_head : int
        The number of attention heads.
    block_size : int
        The size of the data block.

    Attributes
    ----------
    sa : MultiHeadAttention
        The multi-head attention module.
    ffwd : FeedFoward
        The feedforward module.
    ln1 : nn.LayerNorm
        The first layer normalization module.
    ln2 : nn.LayerNorm
        The second layer normalization module.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Perform the forward pass of the transformer block.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, block_size=block_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the transformer block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after the transformer block.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BaseKGE(pl.LightningModule):
    """
    Base class for Knowledge Graph Embedding (KGE) models.

    Parameters
    ----------
    args : dict
        A dictionary containing the model configuration and hyperparameters.

    Attributes
    ----------
    args : dict
        A dictionary containing the model configuration and hyperparameters.
    embedding_dim : Optional[int]
        The dimension of the embeddings.
    num_entities : Optional[int]
        The number of entities in the knowledge graph.
    num_relations : Optional[int]
        The number of relations in the knowledge graph.
    num_tokens : Optional[int]
        The number of tokens.
    learning_rate : Optional[float]
        The learning rate for the optimizer.
    apply_unit_norm : Optional[bool]
        A flag indicating whether to apply unit normalization.
    input_dropout_rate : Optional[float]
        The dropout rate for the input layer.
    hidden_dropout_rate : Optional[float]
        The dropout rate for the hidden layers.
    optimizer_name : Optional[str]
        The name of the optimizer.
    feature_map_dropout_rate : Optional[float]
        The dropout rate for the feature maps.
    kernel_size : Optional[int]
        The size of the convolutional kernel.
    num_of_output_channels : Optional[int]
        The number of output channels for the convolutional layer.
    weight_decay : Optional[float]
        The weight decay for the optimizer.
    loss : torch.nn.BCEWithLogitsLoss
        The loss function.
    selected_optimizer : Optional[Any]
        The selected optimizer for training.
    normalizer_class : Optional[Any]
        The class for normalization.
    normalize_head_entity_embeddings : Any
        The normalization function for head entity embeddings.
    normalize_relation_embeddings : Any
        The normalization function for relation embeddings.
    normalize_tail_entity_embeddings : Any
        The normalization function for tail entity embeddings.
    hidden_normalizer : Any
        The normalization function for hidden layers.
    param_init : Any
        The function for initializing parameters.
    input_dp_ent_real : torch.nn.Dropout
        The dropout layer for real-valued input entities.
    input_dp_rel_real : torch.nn.Dropout
        The dropout layer for real-valued input relations.
    hidden_dropout : torch.nn.Dropout
        The dropout layer for hidden layers.
    loss_history : List[float]
        A list to store the average minibatch loss per epoch.
    byte_pair_encoding : bool
        A flag indicating whether byte pair encoding is used.
    max_length_subword_tokens : Optional[int]
        The maximum length of subword tokens.

    Methods
    -------
    forward(x: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]], y_idx: torch.LongTensor = None) -> Any
        Perform the forward pass of the model.

    training_step(batch, batch_idx=None) -> torch.Tensor
        Process a training batch and return the loss.

    configure_optimizers(parameters=None) -> Any
        Configure the optimizer for training.

    loss_function(yhat_batch: torch.FloatTensor, y_batch: torch.FloatTensor) -> torch.Tensor
        Compute the loss function.

    mem_of_model() -> Dict
        Return the size of the model in MB and the number of parameters.

    init_params_with_sanity_checking() -> None
        Initialize model parameters with sanity checking.

    forward_byte_pair_encoded_triple(x: Tuple[torch.LongTensor, torch.LongTensor]) -> torch.Tensor
        Perform the forward pass for byte pair encoded triples.

    forward_byte_pair_encoded_k_vs_all(x: torch.LongTensor) -> torch.Tensor
        Perform the forward pass for byte pair encoded K vs. All.

    forward_triples(x: torch.LongTensor) -> torch.Tensor
        Perform the forward pass for triples.

    forward_k_vs_all(*args, kwargs) -> None
        Forward pass for K vs. All.

    forward_k_vs_sample(*args, kwargs) -> None
        Forward pass for K vs. Sample.

    on_train_epoch_end(*args, kwargs) -> None
        Perform actions at the end of a training epoch.

    test_epoch_end(outputs: List[Any]) -> None
        Perform actions at the end of a testing epoch.

    test_dataloader() -> None
        Return the dataloader for testing.

    val_dataloader() -> None
        Return the dataloader for validation.

    predict_dataloader() -> None
        Return the dataloader for prediction.

    train_dataloader() -> None
        Return the dataloader for training.

    get_triple_representation(idx_hrt: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
        Get the representation for a triple.

    get_head_relation_representation(indexed_triple: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]
        Get the representation for the head and relation.

    get_sentence_representation(x: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
        Get the representation for a sentence.

    get_bpe_head_and_relation_representation(x: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]
        Get the representation for BPE head and relation.

    get_embeddings() -> Tuple[np.ndarray, np.ndarray]
        Get the entity and relation embeddings.
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

        if self.byte_pair_encoding:
            self.token_embeddings = torch.nn.Embedding(self.num_tokens, self.embedding_dim)
            # Workaround: Dummy subReducing the impact of dummies
            self.lf = nn.Sequential(
                nn.Linear(self.embedding_dim * self.max_length_subword_tokens, self.embedding_dim, bias=False))

            self.param_init(self.token_embeddings.weight.data)
            if self.args["scoring_technique"] in ["AllvsAll", "KvsAll"]:
                self.str_to_bpe_entity_to_idx = {str_ent: idx for idx, (str_ent, bpe_ent, shaped_bpe_ent) in
                                                 enumerate(self.args["ordered_bpe_entities"])}

                self.bpe_entity_to_idx = {shaped_bpe_ent: idx for idx, (str_ent, bpe_ent, shaped_bpe_ent) in
                                          enumerate(self.args["ordered_bpe_entities"])}
                self.ordered_bpe_entities = torch.tensor(list(self.bpe_entity_to_idx.keys()), dtype=torch.long)
        else:
            self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
            self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
            self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def forward_byte_pair_encoded_triple(self, x: Tuple[torch.LongTensor, torch.LongTensor]) -> torch.Tensor:
        """
        Perform the forward pass for byte pair encoded triples.

        Parameters
        ----------
        x : Tuple[torch.LongTensor, torch.LongTensor]
            The input tuple containing byte pair encoded entities and relations.

        Returns
        -------
        torch.Tensor
            The output tensor containing the scores for the byte pair encoded triples.
        """
        bpe_head_ent_emb, bpe_rel_ent_emb, bpe_tail_ent_emb = self.get_sentence_representation(x)
        B, T, C = bpe_head_ent_emb.shape
        bpe_head_ent_emb = bpe_head_ent_emb.reshape(B, T * C)
        bpe_rel_ent_emb = bpe_rel_ent_emb.reshape(B, T * C)
        bpe_tail_ent_emb = bpe_tail_ent_emb.reshape(B, T * C)
        bpe_triple_score = self.score(self.lf(bpe_head_ent_emb), self.lf(bpe_rel_ent_emb), self.lf(bpe_tail_ent_emb))
        return bpe_triple_score

    def forward_byte_pair_encoded_k_vs_all(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Perform the forward pass for byte pair encoded K vs. All.

        Parameters
        ----------
        x : torch.LongTensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor containing the scores for the byte pair encoded K vs. All.
        """
        bpe_head_ent_emb, bpe_rel_ent_emb = self.get_bpe_head_and_relation_representation(x)

        B, T, C = bpe_head_ent_emb.shape
        bpe_head_ent_emb = bpe_head_ent_emb.reshape(B, T * C)
        bpe_rel_ent_emb = bpe_rel_ent_emb.reshape(B, T * C)

        bpe_head_ent_emb = self.lf(bpe_head_ent_emb)
        bpe_rel_ent_emb = self.lf(bpe_rel_ent_emb)

        device_r = bpe_head_ent_emb.get_device()
        if device_r >= 0:
            self.ordered_bpe_entities = self.ordered_bpe_entities.to(device_r)
        else:
            self.ordered_bpe_entities = self.ordered_bpe_entities.to("cpu")

        all_entities = self.token_embeddings(self.ordered_bpe_entities)
        num_e, token_size, dim = all_entities.shape
        all_entities = all_entities.reshape(num_e, token_size * dim)
        # Normalize each token vector into unit norms
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
        E = F.normalize(self.lf(all_entities), p=2, dim=0)
        return self.k_vs_all_score(bpe_head_ent_emb, bpe_rel_ent_emb, E)

    def mem_of_model(self) -> Dict:
        """
        Calculate the size of the model in MB and the number of parameters.

        Returns
        -------
        Dict
            A dictionary containing the estimated size of the model in MB and the total number of parameters.
        """
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        # (2) Store NumParam and EstimatedSizeMB
        num_params = sum(p.numel() for p in self.parameters())
        # Not quite sure about EstimatedSizeMB ?
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return {'EstimatedSizeMB': (num_params + buffer_size) / 1024 ** 2, 'NumParam': num_params}

    def init_params_with_sanity_checking(self) -> None:
        """Initialize model parameters with sanity checking based on the provided configuration."""
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
            if self.args['scoring_technique'] in ['NegSample', 'KvsSample']:
                self.normalize_tail_entity_embeddings = self.normalizer_class(self.embedding_dim)
        elif self.args.get("normalization") == 'BatchNorm1d':
            self.normalizer_class = torch.nn.BatchNorm1d
            self.normalize_head_entity_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
            self.normalize_relation_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
            if self.args['scoring_technique'] in ['NegSample', 'KvsSample']:
                self.normalize_tail_entity_embeddings = self.normalizer_class(self.embedding_dim, affine=False)
        elif self.args.get("normalization") is None:
            self.normalizer_class = IdentityClass
        else:
            raise NotImplementedError()
        if self.args.get("optim") in ['NAdam', 'Adam', 'SGD']:
            self.optimizer_name = self.args['optim']
        else:
            print(f'--optim (***{self.args.get("optim")}***) not found')
            self.optimizer_name = 'Adam'

        if self.args.get("init_param") is None:
            self.param_init = IdentityClass
        elif self.args['init_param'] == 'xavier_normal':
            self.param_init = torch.nn.init.xavier_normal_
        else:
            print(f'--init_param (***{self.args.get("init_param")}***) not found')
            self.optimizer_name = IdentityClass

    def configure_optimizers(self, parameters=None) -> Any:
        """
        Configure the optimizer for training.

        Parameters
        ----------
        parameters : Any, optional
            The parameters to be optimized (default is None).

        Returns
        -------
        Any
            The selected optimizer for training.
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
        else:
            raise KeyError()
        return self.selected_optimizer

    def loss_function(self, yhat_batch: torch.FloatTensor, y_batch: torch.FloatTensor) -> torch.Tensor:
        """
        Compute the loss function.

        Parameters
        ----------
        yhat_batch : torch.FloatTensor
            The predicted values.
        y_batch : torch.FloatTensor
            The ground truth values.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        return self.loss(yhat_batch, y_batch)

    def forward(self, x: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]], y_idx: torch.LongTensor = None) -> Any:
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]]
            The input tensor or a tuple containing the input tensor and target entity indexes.
        y_idx : torch.LongTensor, optional
            The target entity indexes (default is None).

        Returns
        -------
        Any
            The output of the forward pass.
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
        """
        Perform the forward pass for triples.

        Parameters
        ----------
        x : torch.LongTensor
            The input tensor containing the indexes of head, relation, and tail entities.

        Returns
        -------
        torch.Tensor
            The output tensor containing the scores for the input triples.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        h_emb, r_emb, t_emb = self.get_triple_representation(x)
        return self.score(h_emb, r_emb, t_emb)

    def forward_k_vs_all(self, *args, **kwargs):
        """
        Forward pass for K vs. All.

        Raises
        ------
        ValueError
            This function is not implemented in the current model.
        """
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_all function')

    def forward_k_vs_sample(self, *args, **kwargs):
        """
        Forward pass for K vs. Sample.

        Raises
        ------
        ValueError
            This function is not implemented in the current model.
        """
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_sample function')

    def training_step(self, batch: tuple, batch_idx: Optional[int] = None) -> torch.Tensor:
        """
        Process a training batch and return the loss.

        Parameters
        ----------
        batch : tuple
            A tuple containing the input and target tensors.
        batch_idx : int, optional
            The index of the current batch (default is None).

        Returns
        -------
        torch.Tensor
            The computed loss for the training batch.
        """
        x_batch, y_batch = batch
        yhat_batch = self.forward(x_batch)
        loss_batch = self.loss_function(yhat_batch, y_batch)
        return loss_batch

    def on_train_epoch_end(self, *args, **kwargs):
        """
        Perform actions at the end of a training epoch.

        Raises
        ------
        RuntimeError
            If the arguments or keyword arguments are not empty.
        """
        if len(args)>=1:
            raise RuntimeError(f"Arguments must not be empty:{args}")

        if len(kwargs)>=1:
            raise RuntimeError(f"Keyword Arguments must not be empty:{kwargs}")

        # @TODO: No saving
        """

        batch_losses = [i['loss'].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)
        self.loss_history.append(avg)
        """
    def test_epoch_end(self, outputs: List[Any]):
        """
        @ TODO
        avg_test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('avg_test_accuracy', avg_test_accuracy, on_epoch=True, prog_bar=True)
        """

    def test_dataloader(self) -> None:
        pass

    def val_dataloader(self) -> None:
        pass

    def predict_dataloader(self) -> None:
        pass

    def train_dataloader(self) -> None:
        pass

    def get_triple_representation(self, idx_hrt: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Get the representation for a triple.

        Parameters
        ----------
        idx_hrt : torch.LongTensor
            The indexes of head, relation, and tail entities.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            The representation for the input triple.
        """
        # (1) Split input into indexes.
        idx_head_entity, idx_relation, idx_tail_entity = idx_hrt[:, 0], idx_hrt[:, 1], idx_hrt[:, 2]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        tail_ent_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(idx_tail_entity))
        return head_ent_emb, rel_ent_emb, tail_ent_emb

    def get_head_relation_representation(self, indexed_triple: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get the representation for the head and relation entities.

        Parameters
        ----------
        indexed_triple : torch.LongTensor
            The indexes of the head and relation entities.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            The representation for the head and relation entities.
        """
        # (1) Split input into indexes.
        idx_head_entity, idx_relation = indexed_triple[:, 0], indexed_triple[:, 1]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return head_ent_emb, rel_ent_emb

    def get_sentence_representation(self, x: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Get the representation for a sentence.

        Parameters
        ----------
        x : torch.LongTensor
            The input tensor containing the indexes of head, relation, and tail entities.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            The representation for the input sentence.
        """
        h, r, t = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        head_ent_emb = self.token_embeddings(h)
        rel_emb = self.token_embeddings(r)
        tail_emb = self.token_embeddings(t)
        return head_ent_emb, rel_emb, tail_emb

    def get_bpe_head_and_relation_representation(self, x: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get the representation for BPE head and relation entities.

        Parameters
        ----------
        x : torch.LongTensor
            The input tensor containing the indexes of head and relation entities.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            The representation for BPE head and relation entities.
        """
        h, r = x[:, 0, :], x[:, 1, :]
        # N, T, D
        head_ent_emb = self.token_embeddings(h)
        # N, T, D
        rel_emb = self.token_embeddings(r)
        # A sequence of sub-list embeddings representing an embedding of a head entity should be normalized to 0.
        # Therefore, the norm of a row vector obtained from T by D matrix must be 1.
        head_ent_emb = F.normalize(head_ent_emb, p=2, dim=(1, 2))
        rel_emb = F.normalize(rel_emb, p=2, dim=(1, 2))
        return head_ent_emb, rel_emb

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the entity and relation embeddings.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The entity and relation embeddings.
        """
        return self.entity_embeddings.weight.data.data.detach(), self.relation_embeddings.weight.data.detach()


class IdentityClass(torch.nn.Module):
    """
    A class that represents an identity function.

    Parameters
    ----------
    args : dict, optional
        A dictionary containing arguments (default is None).
    """
    def __init__(self, args: Optional[Dict] = None):
        super().__init__()
        self.args = args

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the identity function.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor, which is the same as the input.
        """
        return x
