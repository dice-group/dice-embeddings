from typing import List, Any, Tuple, Union, Dict
import pytorch_lightning
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
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
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, block_size=block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, block_size=block_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BaseKGE(pytorch_lightning.LightningModule):
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

    def forward_byte_pair_encoded_triple(self, x: Tuple[torch.LongTensor, torch.LongTensor]):
        """
        byte pair encoded neural link predictors

        Parameters
        ----------

        -------

        """

        bpe_head_ent_emb, bpe_rel_ent_emb, bpe_tail_ent_emb = self.get_sentence_representation(x)
        B, T, C = bpe_head_ent_emb.shape
        bpe_head_ent_emb = bpe_head_ent_emb.reshape(B, T * C)
        bpe_rel_ent_emb = bpe_rel_ent_emb.reshape(B, T * C)
        bpe_tail_ent_emb = bpe_tail_ent_emb.reshape(B, T * C)
        bpe_triple_score = self.score(self.lf(bpe_head_ent_emb), self.lf(bpe_rel_ent_emb), self.lf(bpe_tail_ent_emb))
        return bpe_triple_score

    def forward_byte_pair_encoded_k_vs_all(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        bpe_head_ent_emb, bpe_rel_ent_emb = self.get_bpe_head_and_relation_representation(x)

        B, T, C = bpe_head_ent_emb.shape
        bpe_head_ent_emb = bpe_head_ent_emb.reshape(B, T * C)
        bpe_rel_ent_emb = bpe_rel_ent_emb.reshape(B, T * C)

        bpe_head_ent_emb = self.lf(bpe_head_ent_emb)
        bpe_rel_ent_emb = self.lf(bpe_rel_ent_emb)

        device_r=bpe_head_ent_emb.get_device()
        if device_r>=0:
            self.ordered_bpe_entities=self.ordered_bpe_entities.to(device_r)
        else:
            self.ordered_bpe_entities=self.ordered_bpe_entities.to("cpu")


        all_entities = self.token_embeddings(self.ordered_bpe_entities)
        num_e, token_size, dim = all_entities.shape
        all_entities = all_entities.reshape(num_e, token_size * dim)
        
        
        E = self.lf(all_entities)
        return self.k_vs_all_score(bpe_head_ent_emb, bpe_rel_ent_emb, E)

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

    def init_params_with_sanity_checking(self):
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

    def configure_optimizers(self, parameters=None):
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

    def loss_function(self, yhat_batch: torch.FloatTensor, y_batch: torch.FloatTensor):
        """

        Parameters
        ----------
        yhat_batch
        y_batch

        Returns
        -------

        """
        return self.loss(yhat_batch, y_batch)

    def forward(self, x: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]],
                y_idx: torch.LongTensor = None):
        """

        Parameters
        ----------
        x
        y_idx
        ordered_bpe_entities

        Returns
        -------

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

        Parameters
        ----------
        x

        Returns
        -------

        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        h_emb, r_emb, t_emb = self.get_triple_representation(x)
        return self.score(h_emb, r_emb, t_emb)

    def forward_k_vs_all(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_all function')

    def forward_k_vs_sample(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_sample function')

    def training_step(self, batch, batch_idx=None):
        x_batch, y_batch = batch
        yhat_batch = self.forward(x_batch)
        loss_batch = self.loss_function(yhat_batch, y_batch)
        return loss_batch

    def training_epoch_end(self, training_step_outputs):
        batch_losses = [i['loss'].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)
        self.loss_history.append(avg)

    def validation_step(self, batch, batch_idx):
        """
        @ TODO
        # from torchmetrics import Accuracy as accuracy
        if len(batch) == 4:
            h, r, t, y_batch = batch
            predictions = self.forward_triples(h, r, t)
        else:
            h, x, y_batch = batch[:, 0], batch[:, 1], batch[:, 2]
            predictions = self.forward_k_vs_all(h, x)

        val_loss = self.loss_function(predictions, y_batch)
        val_accuracy = accuracy(predictions, y_batch)
        return {'val_acc': val_accuracy, 'val_loss': val_loss}
        """

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        @ TODO

        x = [[x['val_acc'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_loss_per_epoch', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc_per_epoch', avg_val_acc, on_epoch=True, prog_bar=True)
        """

    def test_step(self, batch, batch_idx):
        """
        @ TODO

        if len(batch) == 4:
            h, r, t, y_batch = batch
            predictions = self.forward_triples(h, r, t)
        else:
            h, x, y_batch = batch[:, 0], batch[:, 1], batch[:, 2]
            predictions = self.forward_k_vs_all(h, x)
        test_accuracy = accuracy(predictions, y_batch)
        return {'test_accuracy': test_accuracy}
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

    def get_triple_representation(self, idx_hrt):
        # (1) Split input into indexes.
        idx_head_entity, idx_relation, idx_tail_entity = idx_hrt[:, 0], idx_hrt[:, 1], idx_hrt[:, 2]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        tail_ent_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(idx_tail_entity))
        return head_ent_emb, rel_ent_emb, tail_ent_emb

    def get_head_relation_representation(self, indexed_triple):
        # (1) Split input into indexes.
        idx_head_entity, idx_relation = indexed_triple[:, 0], indexed_triple[:, 1]
        # (2) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(self.entity_embeddings(idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return head_ent_emb, rel_ent_emb

    def get_sentence_representation(self, x: torch.LongTensor):
        """

        Parameters
        ----------
        x shape (b,3,t)

        Returns
        -------

        """
        h, r, t = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        head_ent_emb = self.token_embeddings(h)
        rel_emb = self.token_embeddings(r)
        tail_emb = self.token_embeddings(t)
        return head_ent_emb, rel_emb, tail_emb

    def get_bpe_head_and_relation_representation(self, x: torch.LongTensor):
        h, r = x[:, 0, :], x[:, 1, :]
        head_ent_emb = self.token_embeddings(h)
        rel_emb = self.token_embeddings(r)
        return head_ent_emb, rel_emb

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """

        Returns
        -------

        """
        return self.entity_embeddings.weight.data.data.detach(), self.relation_embeddings.weight.data.detach()


class IdentityClass(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

    @staticmethod
    def forward(x):
        return x
