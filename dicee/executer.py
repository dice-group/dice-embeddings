import json
import logging
import time
import warnings
from types import SimpleNamespace
import os
import datetime
from pytorch_lightning import seed_everything
from .knowledge_graph import KG
from .evaluator import Evaluator
# Avoid
from .static_preprocess_funcs import preprocesses_input_args
from .trainer import DICE_Trainer
from .static_funcs import timeit, continual_training_setup_executor, read_or_load_kg, load_json, store,create_experiment_folder
import numpy as np

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
        seed_everything(args.random_seed, workers=True)
        # (3) Set the continual training flag
        self.is_continual_training = continuous_training
        # (4) Create an experiment folder or use the previous one
        self.continual_training_setup_executor()
        # (5) A variable is initialized for pytorch lightning trainer or DICE_Trainer()
        self.trainer = None
        self.trained_model = None
        # (6) A variable is initialized for storing input data.
        self.knowledge_graph = None
        # (7) Store few data in memory for numerical results, e.g. runtime, H@1 etc.
        self.report = dict()
        # (8) Create an object to carry out link prediction evaluations
        self.evaluator = None  # e.g. Evaluator(self)
        # (9) Execution start time
        self.start_time = None
        self.multi_gpu_ddp_training_mode=False

    def continual_training_setup_executor(self) -> None:
        if self.is_continual_training:
            # () We continue the training, then we store new models on previous path.
            self.storage_path = self.args.full_storage_path
        else:
            # Create a single directory containing KGE and all related data
            if self.args.path_to_store_single_run:
                os.makedirs(self.args.path_to_store_single_run, exist_ok=True)
                self.args.full_storage_path = self.args.path_to_store_single_run
            else:
                # Create a parent and subdirectory.
                self.args.full_storage_path = create_experiment_folder(folder_name=self.args.storage_path)
            self.storage_path = self.args.full_storage_path
            with open(self.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
                temp = vars(self.args)
                json.dump(temp, file_descriptor, indent=3)

    def dept_read_or_load_kg(self):
        print('*** Read or Load Knowledge Graph  ***')
        start_time = time.time()
        kg = KG(dataset_dir=self.args.dataset_dir,
                byte_pair_encoding=self.args.byte_pair_encoding,
                padding=True if self.args.byte_pair_encoding and self.args.model != "BytE" else False,
                add_noise_rate=self.args.add_noise_rate,
                sparql_endpoint=self.args.sparql_endpoint,
                path_single_kg=self.args.path_single_kg,
                add_reciprocal=self.args.apply_reciprical_or_noise,
                eval_model=self.args.eval_model,
                read_only_few=self.args.read_only_few,
                sample_triples_ratio=self.args.sample_triples_ratio,
                path_for_serialization=self.args.full_storage_path,
                path_for_deserialization=self.args.path_experiment_folder if hasattr(self.args,
                                                                                     'path_experiment_folder') else None,
                backend=self.args.backend,
                training_technique=self.args.scoring_technique)
        print(f'Preprocessing took: {time.time() - start_time:.3f} seconds')
        # (2) Share some info about data for easy access.
        print(kg.description_of_input)
        return kg

    def read_preprocess_index_serialize_data(self) -> None:
        """ Read & Preprocess & Index & Serialize Input Data

        (1) Read or load the data from disk into memory.
        (2) Store the statistics of the data.

        Parameter
        ----------

        Return
        ----------
        None

        """
        # (2) Store the stats and share parameters
        self.args.num_entities = self.knowledge_graph.num_entities
        self.args.num_relations = self.knowledge_graph.num_relations
        self.args.num_tokens = self.knowledge_graph.num_tokens
        self.args.max_length_subword_tokens = self.knowledge_graph.max_length_subword_tokens
        self.args.ordered_bpe_entities=self.knowledge_graph.ordered_bpe_entities
        self.report['num_train_triples'] = len(self.knowledge_graph.train_set)
        self.report['num_entities'] = self.knowledge_graph.num_entities
        self.report['num_relations'] = self.knowledge_graph.num_relations
        self.report['num_relations'] = self.knowledge_graph.num_relations
        self.report[
            'max_length_subword_tokens'] = self.knowledge_graph.max_length_subword_tokens if self.knowledge_graph.max_length_subword_tokens else None

        self.report['runtime_kg_loading'] = time.time() - self.start_time

    @timeit
    def save_trained_model(self) -> None:
        """ Save a knowledge graph embedding model

        (1) Send model to eval mode and cpu.
        (2) Store the memory footprint of the model.
        (3) Save the model into disk.
        (4) Update the stats of KG again ?

        Parameter
        ----------

        Return
        ----------
        None

        """
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
                  save_embeddings_as_csv=self.args.save_embeddings_as_csv)
        else:
            store(trainer=self.trainer,
                  trained_model=self.trained_model,
                  model_name='model_' + str(datetime.datetime.now()),
                  full_storage_path=self.storage_path, save_embeddings_as_csv=self.args.save_embeddings_as_csv)

        self.report['path_experiment_folder'] = self.storage_path
        self.report['num_entities'] = self.args.num_entities
        self.report['num_relations'] = self.args.num_relations
        self.report['path_experiment_folder'] = self.storage_path

    def end(self, form_of_labelling: str) -> dict:
        """
        End training

        (1) Store trained model.
        (2) Report runtimes.
        (3) Eval model if required.

        Parameter
        ---------

        Returns
        -------
        A dict containing information about the training and/or evaluation

        """
        # (1) Save the model
        self.save_trained_model()
        # (2) Report
        self.write_report()
        # (3) Eval model and return eval results.
        if self.args.eval_model is None:
            self.write_report()
            return {**self.report}
        else:
            self.evaluator.eval(dataset=self.knowledge_graph, trained_model=self.trained_model,
                                form_of_labelling=form_of_labelling)
            self.write_report()
            return {**self.report, **self.evaluator.report}

    def write_report(self) -> None:
        """ Report training related information in a report.json file """
        # Report total runtime.
        self.report['Runtime'] = time.time() - self.start_time
        print(f"Total Runtime: {self.report['Runtime']:.3f} seconds")
        with open(self.args.full_storage_path + '/report.json', 'w') as file_descriptor:
            json.dump(self.report, file_descriptor, indent=4)

    def start(self) -> dict:
        """
        Start training

        # (1) Loading the Data
        # (2) Create an evaluator object.
        # (3) Create a trainer object.
        # (4) Start the training

        Parameter
        ---------

        Returns
        -------
        A dict containing information about the training and/or evaluation

        """
        self.start_time = time.time()
        print(f"Start time:{datetime.datetime.now()}")
        # (1) Loading the Data
        #  Load the indexed data from disk or read a raw data from disk into knowledge_graph attribute
        if os.path.exists(self.args.path_to_store_single_run+"/memory_map_train_set.npy"):
            # Read the JSON file
            with open(self.args.path_to_store_single_run+'/memory_map_details.json', 'r') as file_descriptor:
                memory_map_details = json.load(file_descriptor)    
            self.knowledge_graph = np.memmap(self.args.path_to_store_single_run + '/memory_map_train_set.npy', dtype=memory_map_details["dtype"], mode='r', shape=tuple(memory_map_details["shape"]))
            print(self.knowledge_graph[:10])

            self.args.num_entities = memory_map_details["num_entities"]
            self.args.num_relations = memory_map_details["num_relations"]
            self.args.num_tokens = None
            self.args.max_length_subword_tokens = None
            self.args.ordered_bpe_entities = None 

        else:
            self.knowledge_graph = read_or_load_kg(self.args, cls=KG)
            if self.is_continual_training is False:
                self.args.num_entities = self.knowledge_graph.num_entities
                self.args.num_relations = self.knowledge_graph.num_relations
                self.args.num_tokens = self.knowledge_graph.num_tokens
                self.args.max_length_subword_tokens = self.knowledge_graph.max_length_subword_tokens
                self.args.ordered_bpe_entities = self.knowledge_graph.ordered_bpe_entities
                self.report['num_train_triples'] = len(self.knowledge_graph.train_set)
                self.report['num_entities'] = self.knowledge_graph.num_entities
                self.report['num_relations'] = self.knowledge_graph.num_relations
                self.report['max_length_subword_tokens'] = self.knowledge_graph.max_length_subword_tokens if self.knowledge_graph.max_length_subword_tokens else None
                self.report['runtime_kg_loading'] = time.time() - self.start_time

                data={"shape":tuple(self.knowledge_graph.train_set.shape),"dtype":self.knowledge_graph.train_set.dtype.str,
                      "num_entities":self.knowledge_graph.num_entities,"num_relations":self.knowledge_graph.num_relations}
                with open(self.args.full_storage_path + '/memory_map_details.json', 'w') as file_descriptor:
                    json.dump(data, file_descriptor, indent=4)
                print("DEEEEMIRRR\n")
                
                print(self.knowledge_graph.train_set[:10])
        # (2) Create an evaluator object.
        self.evaluator = Evaluator(args=self.args)
        # (3) Create a trainer object.
        self.trainer = DICE_Trainer(args=self.args,
                                    is_continual_training=self.is_continual_training,
                                    storage_path=self.storage_path,
                                    evaluator=self.evaluator)
        # (4) Start the training
        self.trained_model, form_of_labelling = self.trainer.start(knowledge_graph=self.knowledge_graph)
        return self.end(form_of_labelling)


class ContinuousExecute(Execute):
    """ A subclass of Execute Class for retraining

    (1) Loading & Preprocessing & Serializing input data.
    (2) Training & Validation & Testing
    (3) Storing all necessary info

    During the continual learning we can only modify *** num_epochs *** parameter.
    Trained model stored in the same folder as the seed model for the training.
    Trained model is noted with the current time.
    """

    def __init__(self, args):
        # (1) Current input configuration.
        assert os.path.exists(args.continual_learning)
        assert os.path.isfile(args.continual_learning + '/configuration.json')
        # (2) Load previous input configuration.
        previous_args = load_json(args.continual_learning + '/configuration.json')
        args=vars(args)
        #
        previous_args["num_epochs"]=args["num_epochs"]
        previous_args["continual_learning"]=args["continual_learning"]
        print("Updated configuration:",previous_args)
        try:
            report = load_json(args['continual_learning'] + '/report.json')
            previous_args['num_entities'] = report['num_entities']
            previous_args['num_relations'] = report['num_relations']
        except AssertionError:
            print("Couldn't find report.json.")
        previous_args = SimpleNamespace(**previous_args)
        print('ContinuousExecute starting...')
        print(previous_args)
        super().__init__(previous_args, continuous_training=True)

    def continual_start(self) -> dict:
        """
        Start Continual Training

        (1) Initialize training.
        (2) Start continual training.
        (3) Save trained model.

        Parameter
        ---------

        Returns
        -------
        A dict containing information about the training and/or evaluation

        """
        # (1)
        self.trainer = DICE_Trainer(args=self.args, is_continual_training=True,
                                    storage_path=self.args.continual_learning)
        # (2)
        self.trained_model, form_of_labelling = self.trainer.continual_start()

        # (5) Store trained model.
        self.save_trained_model()
        # (6) Eval model.
        if self.args.eval_model is None:
            return self.report
        else:
            self.evaluator = Evaluator(args=self.args, is_continual_training=True)
            self.evaluator.dummy_eval(self.trained_model, form_of_labelling)
            return {**self.report, **self.evaluator.report}
