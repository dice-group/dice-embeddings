from ..types import List, Any, Tuple, Union, Dict, np, torch
import pytorch_lightning


class BaseKGE(pytorch_lightning.LightningModule):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.embedding_dim = None
        self.num_entities = None
        self.num_relations = None
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
        self.param_init = None
        self.init_params_with_sanity_checking()

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(self.input_dropout_rate)
        # average minibatch loss per epoch
        self.loss_history = []

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
        assert self.args['model'] in ['FMult', 'FMult2', 'CMult', 'Keci', 'DistMult', 'ComplEx', 'QMult', 'OMult', 'ConvQ',
                                      'ConvO',
                                      'AConEx', 'ConEx', 'Shallom', 'TransE', 'Pyke', 'KeciBase']
        if self.args.get('weight_decay'):
            self.weight_decay = self.args['weight_decay']
        else:
            self.weight_decay = 0.0
        if self.args.get('embedding_dim'):
            self.embedding_dim = self.args['embedding_dim']
        else:
            self.embedding_dim = 1

        if self.args.get('num_entities'):
            self.num_entities = self.args['num_entities']
        else:
            self.num_entities = 1

        if self.args.get('num_relations'):
            self.num_relations = self.args['num_relations']
        else:
            self.num_relations = 1

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

        if self.args['model'] in ['ConvQ', 'ConvO', 'ConEx', 'AConEx']:
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
        if self.args.get("optim") in ['Adan', 'NAdam', 'Adam', 'SGD', 'ASGD', 'Sls', 'AdamSLS']:
            self.optimizer_name = self.args['optim']
        else:
            print(self.args)
            raise KeyError(f'--optim (***{self.args.get("optim")}***) not found')

        if self.args['init_param'] is None:
            self.param_init = IdentityClass
        elif self.args['init_param'] == 'xavier_normal':
            self.param_init = torch.nn.init.xavier_normal_
        else:
            raise KeyError(f'--init_param (***{self.args.get("init_param")}***) not found')

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        # @TODO why twice data.data.?
        return self.entity_embeddings.weight.data.data.detach(), self.relation_embeddings.weight.data.detach()

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
        elif self.optimizer_name == 'Adan':
            self.selected_optimizer = Adan(parameters, lr=self.learning_rate, weight_decay=self.weight_decay,
                                           betas=(0.98, 0.92, 0.99),
                                           eps=1e-08,
                                           max_grad_norm=0.0,
                                           no_prox=False)
        elif self.optimizer_name == 'Sls':
            self.selected_optimizer = Sls(params=parameters,
                                          n_batches_per_epoch=500,
                                          init_step_size=self.learning_rate,  # 1 originally
                                          c=0.1,
                                          beta_b=0.9,
                                          gamma=2.0,
                                          beta_f=2.0,
                                          reset_option=1,
                                          eta_max=10,
                                          bound_step_size=True,
                                          line_search_fn="armijo")
        elif self.optimizer_name == 'AdamSLS':
            self.selected_optimizer = AdamSLS(params=parameters,
                                              n_batches_per_epoch=500,
                                              init_step_size=self.learning_rate,  # 0.1,0.00001,
                                              c=0.1,
                                              gamma=2.0,
                                              beta=0.999,
                                              momentum=0.9,
                                              gv_option='per_param',
                                              base_opt='adam',
                                              pp_norm_method='pp_armijo',
                                              mom_type='standard',
                                              clip_grad=False,
                                              beta_b=0.9,
                                              beta_f=2.0,
                                              reset_option=1,
                                              line_search_fn="armijo")
        else:
            raise KeyError()
        return self.selected_optimizer

    def get_optimizer_class(self):
        # default params in pytorch.
        if self.optimizer_name == 'SGD':
            return torch.optim.SGD
        elif self.optimizer_name == 'Adam':
            return torch.optim.Adam
        else:
            raise KeyError()

    def loss_function(self, yhat_batch, y_batch):
        return self.loss(input=yhat_batch, target=y_batch)

    def forward_triples(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_triples function')

    def forward_k_vs_all(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_all function')

    def forward_k_vs_sample(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_k_vs_sample function')

    def forward(self, x: Union[torch.LongTensor, Tuple[torch.LongTensor, torch.LongTensor]],
                y_idx: torch.LongTensor = None):
        """

        :param x: a batch of inputs
        :param y_idx: index of selected output labels.
        :return:
        """
        if isinstance(x, tuple):
            x, y_idx = x
            return self.forward_k_vs_sample(x=x, target_entity_idx=y_idx)
        else:
            batch_size, dim = x.shape
            if dim == 3:
                return self.forward_triples(x)
            elif dim == 2:
                # h, y = x[0], x[1]
                # Note that y can be relation or tail entity.
                return self.forward_k_vs_all(x=x)
            else:
                return self.forward_sequence(x=x)

    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            x_batch, y_batch = batch
            yhat_batch = self.forward(x_batch)
        elif len(batch) == 3:
            x_batch, y_idx_batch, y_batch, = batch
            yhat_batch = self.forward(x_batch, y_idx_batch)
        else:
            print(len(batch))
            raise ValueError('Unexpected batch shape..')
        train_loss = self.loss_function(yhat_batch=yhat_batch, y_batch=y_batch)
        return train_loss

    def training_epoch_end(self, training_step_outputs):
        batch_losses = [i['loss'].item() for i in training_step_outputs]
        avg = sum(batch_losses) / len(batch_losses)
        self.loss_history.append(avg)

    def validation_step(self, batch, batch_idx):
        """
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
        x = [[x['val_acc'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_loss_per_epoch', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc_per_epoch', avg_val_acc, on_epoch=True, prog_bar=True)
        """

    def test_step(self, batch, batch_idx):
        """
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

    def get_triple_representation(self, indexed_triple):
        # (1) Split input into indexes.
        idx_head_entity, idx_relation, idx_tail_entity = indexed_triple[:, 0], indexed_triple[:, 1], indexed_triple[:,
                                                                                                     2]
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


class IdentityClass(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

    @staticmethod
    def forward(x):
        return x
