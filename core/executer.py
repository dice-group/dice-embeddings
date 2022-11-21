import json
import logging
import time
import warnings
from types import SimpleNamespace
import os
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from core.knowledge_graph import KG
from core.models.base_model import BaseKGE
from core.evaluator import Evaluator
from core.typings import *
from core.static_funcs import *
from core.sanity_checkers import *
from core.trainers import DICE_Trainer

logging.getLogger('pytorch_lightning').setLevel(0)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)


class Execute:
    """ A class for Training, Retraining and Evaluation a model.

    (1) Loading & Preprocessing & Serializing input data.
    (2) Training & Validation & Testing
    (3) Storing all necessary info
    """

    def __init__(self, args, continuous_training=False):
        # (1) Process arguments and sanity checking.
        self.args = preprocesses_input_args(args)
        # (2) Ensure reproducibility.
        seed_everything(args.seed_for_computation, workers=True)
        # (3) Set the continual training flag
        self.is_continual_training = continuous_training
        # (4) Create an experiment folder or use the previous one
        if self.is_continual_training:
            # (4.1) If it is continual, then store new models on previous path.
            self.storage_path = self.args.full_storage_path
        else:
            # (4.2) Create a folder for the experiments.
            self.args.full_storage_path = create_experiment_folder(folder_name=self.args.storage_path)
            self.storage_path = self.args.full_storage_path
            with open(self.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
                temp = vars(self.args)
                json.dump(temp, file_descriptor, indent=3)
        # (5) A variable is initialized for pytorch lightning trainer.
        self.trainer = None
        # (6) A variable is initialized for storing input data.
        self.dataset = None
        # (7) Store few data in memory for numerical results, e.g. runtime, H@1 etc.
        self.report = dict()
        # (8) Create an object to carry out link prediction evaluations
        self.evaluator = None  # e.g. Evaluator(self)
        # (9) Create an object to carry out training
        self.trainer = None  # e.g. DICE_Trainer(self)

    def read_preprocess_index_serialize_data(self) -> None:
        """ Read & Preprocess & Index & Serialize Input Data """
        # (1) Read & Preprocess & Index & Serialize Input Data.
        self.dataset = read_preprocess_index_serialize_kg(self.args, cls=KG)
        # (2) Share some info about data for easy access.
        self.args.num_entities, self.args.num_relations = self.dataset.num_entities, self.dataset.num_relations
        # (3) Sanity checking.
        self.args, self.dataset = config_kge_sanity_checking(self.args, self.dataset)

    def load_indexed_data(self) -> None:
        """ Load Indexed Data"""
        self.dataset = reload_input_data(self.args, cls=KG)

    def save_trained_model(self, trained_model: BaseKGE, start_time: float) -> None:
        """ Save a knowledge graph embedding model (an instance of BaseKGE class) """
        # @TODO: maybe we can read it from a checkout rather than storing only weights.
        # Save it as dictionary
        #  mdict=torch.load('trainer_checkpoint.pt')
        # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks','optimizer_states', 'lr_schedulers'])
        # try:
        #    self.trainer.save_checkpoint(self.storage_path + '/trainer_checkpoint.pt')
        # except AttributeError as e:
        #    print(e)
        #    print('skipped..')
        # (1) Send model to the eval mode
        trained_model.eval()
        trained_model.to('cpu')
        # (2) Store NumParam and EstimatedSizeMB
        self.report.update(extract_model_summary(trained_model.summarize()))
        # (3) Store/Serialize Model for further use.
        if self.is_continual_training is False:
            store(trained_model, model_name='model', full_storage_path=self.storage_path,
                  dataset=self.dataset, save_as_csv=self.args.save_embeddings_as_csv)
        else:
            store(trained_model, model_name='model_' + str(datetime.datetime.now()),
                  dataset=self.dataset,
                  full_storage_path=self.storage_path, save_as_csv=self.args.save_embeddings_as_csv)

        # (4) Store total runtime.
        total_runtime = time.time() - start_time
        if 60 * 60 > total_runtime:
            message = f'{total_runtime / 60:.3f} minutes'
        else:
            message = f'{total_runtime / (60 ** 2):.3f} hours'
        self.report['Runtime'] = message
        self.report['path_experiment_folder'] = self.storage_path
        print(f'Total computation time: {message}')
        print(f'Number of parameters in {trained_model.name}:', self.report["NumParam"])
        # (4) Store the report of training.
        with open(self.args.full_storage_path + '/report.json', 'w') as file_descriptor:
            json.dump(self.report, file_descriptor, indent=4)

    def start(self) -> dict:
        """
        (1) Data Preparation:
            (1.1) Read, Preprocess Index, Serialize.
            (1.2) Load a data that has been in (1.1).
        (2) Train & Eval
        (3) Save the model
        (4) Return a report of the training
        """
        start_time = time.time()
        # (1) Loading the Data
        #  Load the indexed data from disk or read a raw data from disk.
        self.load_indexed_data() if self.is_continual_training else self.read_preprocess_index_serialize_data()
        # (2) Create an evaluator object.
        self.evaluator = Evaluator(self)
        # (3) Create a trainer object.
        self.trainer = DICE_Trainer(self, self.evaluator)
        # (4) Start the training
        trained_model, form_of_labelling = self.trainer.start()
        # (5) Store trained model.
        self.save_trained_model(trained_model, start_time)
        # (6) Eval model.
        self.evaluator.eval(trained_model, form_of_labelling)
        # (7) Return the report of the training process.
        return {**self.report, **self.evaluator.report}


class ContinuousExecute(Execute):
    """ Continue training a pretrained KGE model """

    def __init__(self, args):
        assert os.path.exists(args.path_experiment_folder)
        assert os.path.isfile(args.path_experiment_folder + '/idx_train_df.gzip')
        assert os.path.isfile(args.path_experiment_folder + '/configuration.json')
        # (1) Load Previous input configuration
        previous_args = load_json(args.path_experiment_folder + '/configuration.json')
        dargs = vars(args)
        del args
        for k in list(dargs.keys()):
            if dargs[k] is None:
                del dargs[k]
        # (2) Update (1) with new input
        previous_args.update(dargs)
        report = load_json(dargs['path_experiment_folder'] + '/report.json')
        previous_args['num_entities'] = report['num_entities']
        previous_args['num_relations'] = report['num_relations']
        previous_args = SimpleNamespace(**previous_args)
        previous_args.full_storage_path = previous_args.path_experiment_folder
        print('ContinuousExecute starting...')
        print(previous_args)
        super().__init__(previous_args, continuous_training=True)
