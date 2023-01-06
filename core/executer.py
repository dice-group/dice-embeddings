import json
import logging
import time
import warnings
from types import SimpleNamespace
import os
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

from core.knowledge_graph import KG
from core.models.base_model import BaseKGE
from core.evaluator import Evaluator
from core.static_funcs import *
from core.static_preprocess_funcs import preprocesses_input_args
from core.sanity_checkers import *
from core.trainer import DICE_Trainer

logging.getLogger('pytorch_lightning').setLevel(0)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


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
        continual_training_setup_executor(self)
        # (5) A variable is initialized for pytorch lightning trainer or DICE_Trainer()
        self.trainer = None
        self.trained_model = None
        # (6) A variable is initialized for storing input data.
        self.dataset = None
        # (7) Store few data in memory for numerical results, e.g. runtime, H@1 etc.
        self.report = dict()
        # (8) Create an object to carry out link prediction evaluations
        self.evaluator = None  # e.g. Evaluator(self)

    def read_preprocess_index_serialize_data(self) -> None:
        """ Read & Preprocess & Index & Serialize Input Data """
        # (1) Read & Preprocess & Index & Serialize Input Data.
        self.dataset = read_or_load_kg(self.args, cls=KG)
        # (3) Sanity checking.
        self.args, self.dataset = config_kge_sanity_checking(self.args, self.dataset)
        self.report['num_train_triples'] = len(self.dataset.train_set)
        self.report['num_entities'] = self.args.num_entities
        self.report['num_relations'] = self.args.num_relations

    def load_indexed_data(self) -> None:
        """ Load Indexed Data"""
        self.dataset = read_or_load_kg(self.args, cls=KG)

    @timeit
    def save_trained_model(self) -> None:
        """ Save a knowledge graph embedding model (an instance of BaseKGE class) """
        print('*** Save Trained Model ***')
        self.trained_model.eval()
        self.trained_model.to('cpu')
        # Save the epoch loss
        # (2) Store NumParam and EstimatedSizeMB
        self.report.update(self.trained_model.mem_of_model())
        # (3) Store/Serialize Model for further use.
        if self.is_continual_training is False:
            store(trainer=self.trainer,
                  trained_model=self.trained_model,
                  model_name='model',
                  full_storage_path=self.storage_path,
                  save_as_csv=self.args.save_embeddings_as_csv)
        else:
            store(trainer=self.trainer,
                  trained_model=self.trained_model,
                  model_name='model_' + str(datetime.datetime.now()),
                  full_storage_path=self.storage_path, save_as_csv=self.args.save_embeddings_as_csv)
        self.report['path_experiment_folder'] = self.storage_path
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
        self.evaluator = Evaluator(args=self.args)
        # (3) Create a trainer object.
        self.trainer = DICE_Trainer(args=self.args,
                                    is_continual_training=self.is_continual_training,
                                    storage_path=self.storage_path,
                                    evaluator=self.evaluator)
        # (4) Start the training
        self.trained_model, form_of_labelling = self.trainer.start(dataset=self.dataset)
        # (5) Store trained model.
        self.save_trained_model()
        self.report['Runtime'] = time.time() - start_time
        print(f"Total computation time: {self.report['Runtime']:.3f} seconds")
        # (6) Eval model.
        if self.args.eval_model is None:
            return self.report
        else:
            self.evaluator.eval(dataset=self.dataset, trained_model=self.trained_model, form_of_labelling=form_of_labelling)
            return {**self.report, **self.evaluator.report}


class ContinuousExecute(Execute):
    """ Continue training a pretrained KGE model """

    def __init__(self, args):
        assert os.path.exists(args.path_experiment_folder)
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
        try:
            report = load_json(dargs['path_experiment_folder'] + '/report.json')
            previous_args['num_entities'] = report['num_entities']
            previous_args['num_relations'] = report['num_relations']
        except AssertionError:
            print("Couldn't find report.json.")
        previous_args = SimpleNamespace(**previous_args)
        previous_args.full_storage_path = previous_args.path_experiment_folder
        print('ContinuousExecute starting...')
        print(previous_args)
        super().__init__(previous_args, continuous_training=True)
