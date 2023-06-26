import argparse


class ParseDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest,
                dict())  # set each name of the attribute to hold the created object(s) as dictionary
        for value in values:
            key, value = value.split('=')
            if value.isdigit():
                getattr(namespace, self.dest)[key] = int(value)
                continue
            getattr(namespace, self.dest)[key] = value


class Args:
    def __init__(self):
        self.path_dataset_folder: str = 'KGs/UMLS'
        self.save_embeddings_as_csv: bool = False
        self.storage_path: str = 'Experiments'
        self.model: str = "Keci"
        self.p: int = 0
        self.q: int = 1
        self.optim: str = 'Adam'
        self.embedding_dim: int = 32
        self.gpus = None
        self.num_epochs: int = 100
        self.batch_size: int = 1024
        self.auto_batch_finder: bool = False
        self.lr: float = 0.1
        self.callbacks: list = []
        self.backend: str = 'pandas'
        self.trainer: str = 'torchCPUTrainer'
        self.scoring_technique: str = 'KvsAll'
        self.neg_ratio: int = 0
        self.weight_decay: float = 0.0
        self.input_dropout_rate: float = 0.0
        self.hidden_dropout_rate: float = 0.0
        self.feature_map_dropout_rate: float = 0.0
        self.normalization: str = "None"
        self.init_param: str = None
        self.gradient_accumulation_steps: int = 0
        self.num_folds_for_cv: int = 0
        self.eval_model: str = "train_val_test"
        self.save_model_at_every_epoch: int = None
        self.label_smoothing_rate: float = 0.0
        self.kernel_size: int = 3
        self.num_of_output_channels: int = 32
        self.num_core: int = 0
        self.seed_for_computation: int = 0
        self.sample_triples_ratio = None
        self.read_only_few = None
        self.pykeen_model_kwargs: ParseDict = None
        self.use_SLCWALitModule: None = False
