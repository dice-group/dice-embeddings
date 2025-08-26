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
from .static_preprocess_funcs import preprocesses_input_args
from .trainer import DICE_Trainer
from .static_funcs import timeit, read_or_load_kg, load_json, store, create_experiment_folder
import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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
        # check if we need distributed training
        if getattr(args, "trainer", None) == "torchDDP":
            self.distributed = True
        # initialize distributed training if required
        if self.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            print(f"[Rank {self.rank}] mapped to GPU {self.local_rank}", flush=True)
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, 0

        # (1) Process arguments and sanity checking.
        self.args = preprocesses_input_args(args)
        # (2) Ensure reproducibility.
        seed_everything(args.random_seed, workers=True)
        # (3) Set the continual training flag
        self.is_continual_training = continuous_training
        # (4) Create an experiment folder or use the previous one
        if self.rank == 0:
            self.setup_executor()
        # (5) A variable is initialized for pytorch lightning trainer or DICE_Trainer()
        self.trainer = None
        self.trained_model = None
        # (6) A variable is initialized for storing input data.
        self.knowledge_graph = None
        # (7) Store few data in memory for numerical results, e.g. runtime, H@1 etc.
        self.report = dict()
        # (8) Create an object to carry out link prediction evaluations,  e.g. Evaluator(self)
        self.evaluator = None
        # (9) Execution start time
        self.start_time = None

    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def cleanup(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    @rank_zero_only
    def setup_executor(self) -> None:
        if self.is_continual_training is False:
            # Create a single directory containing KGE and all related data
            if self.args.path_to_store_single_run is not None:
                if os.path.exists(self.args.path_to_store_single_run):
                    print(f"Deleting the existing directory of {self.args.path_to_store_single_run}")
                    os.system(f'rm -rf {self.args.path_to_store_single_run}')
                os.makedirs(self.args.path_to_store_single_run, exist_ok=False)
                self.args.full_storage_path = self.args.path_to_store_single_run
            else:
                self.args.full_storage_path = create_experiment_folder(folder_name=self.args.storage_path)
                self.args.path_to_store_single_run = self.args.full_storage_path

            with open(self.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
                temp = vars(self.args)
                json.dump(temp, file_descriptor, indent=3)

    def create_and_store_kg(self):
        if not self.is_rank_zero():
            return
        memmap_path = os.path.join(self.args.path_to_store_single_run, "memory_map_train_set.npy")
        details_path = os.path.join(self.args.path_to_store_single_run, "memory_map_details.json")

        if os.path.exists(memmap_path) and os.path.exists(details_path):
            print("KG memmap already exists, skipping.")
            return

        print("Creating knowledge graph...")
        self.knowledge_graph = read_or_load_kg(self.args, cls=KG)
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
        data={"shape":tuple(self.knowledge_graph.train_set.shape),
                "dtype":self.knowledge_graph.train_set.dtype.str,
                "num_entities":self.knowledge_graph.num_entities,
                "num_relations":self.knowledge_graph.num_relations}
        with open(self.args.full_storage_path + '/memory_map_details.json', 'w') as file_descriptor:
            json.dump(data, file_descriptor, indent=4)
        memmap_kg = np.memmap(memmap_path,
                              dtype=self.knowledge_graph.train_set.dtype,
                              mode='w+',
                              shape=self.knowledge_graph.train_set.shape)
        memmap_kg[:] = self.knowledge_graph.train_set[:]
        memmap_kg.flush()
        del memmap_kg
    
    def load_from_memmap(self):
        with open(self.args.path_to_store_single_run+'/memory_map_details.json', 'r') as file_descriptor:
                memory_map_details = json.load(file_descriptor)
        self.knowledge_graph = np.memmap(self.args.path_to_store_single_run + '/memory_map_train_set.npy',
                                            mode='r',
                                            dtype=memory_map_details["dtype"],
                                            shape=tuple(memory_map_details["shape"]))
        self.args.num_entities = memory_map_details["num_entities"]
        self.args.num_relations = memory_map_details["num_relations"]
        self.args.num_tokens = None
        self.args.max_length_subword_tokens = None
        self.args.ordered_bpe_entities = None

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
            store(trained_model=self.trained_model,
                  model_name='model',
                  full_storage_path=self.args.full_storage_path,
                  save_embeddings_as_csv=self.args.save_embeddings_as_csv)
        else:
            store(trained_model=self.trained_model,
                  model_name='model', # + str(datetime.datetime.now()),
                  full_storage_path=self.args.full_storage_path,
                  save_embeddings_as_csv=self.args.save_embeddings_as_csv)

        self.report['path_experiment_folder'] = self.args.full_storage_path
        self.report['num_entities'] = self.args.num_entities
        self.report['num_relations'] = self.args.num_relations

    @rank_zero_only
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
            self.evaluator.eval(dataset=self.knowledge_graph,
                                trained_model=self.trained_model,
                                form_of_labelling=form_of_labelling)
            self.write_report()
            return {**self.report, **self.evaluator.report}

    def write_report(self) -> None:
        """ Report training related information in a report.json file """
        # @TODO: Move to static funcs
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
        # (1) Create knowledge graph
        self.create_and_store_kg()
        # (2) Synchronize processes if distributed training is used
        if self.distributed and dist.is_initialized():
            dist.barrier()

        # (3) Reload the memory-map of index knowledge graph stored as a numpy ndarray
        if self.knowledge_graph is None:
            self.load_from_memmap()

        # (4) Create an evaluator object.
        self.evaluator = Evaluator(args=self.args)
        # (5) Create a trainer object.
        if not getattr(self.args, "full_storage_path", None):
            self.args.full_storage_path = self.args.path_to_store_single_run
        self.trainer = DICE_Trainer(args=self.args,
                                    is_continual_training=self.is_continual_training,
                                    storage_path=self.args.full_storage_path,
                                    evaluator=self.evaluator)
        # (6) Start the training
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
        assert os.path.exists(args.continual_learning), f"Path doesn't exist {args.continual_learning}"
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
        self.trainer = DICE_Trainer(args=self.args,
                                    is_continual_training=True,
                                    storage_path=self.args.continual_learning)
        # (2)

        assert os.path.exists(f"{self.args.continual_learning}/memory_map_train_set.npy")
        # (1) Reload the memory-map of index knowledge graph stored as a numpy ndarray.
        with open(f"{self.args.continual_learning}/memory_map_details.json", 'r') as file_descriptor:
            memory_map_details = json.load(file_descriptor)
        knowledge_graph = np.memmap(f"{self.args.continual_learning}/memory_map_train_set.npy",
                                         mode='r',
                                         dtype=memory_map_details["dtype"],
                                         shape=tuple(memory_map_details["shape"]))
        self.args.num_entities = memory_map_details["num_entities"]
        self.args.num_relations = memory_map_details["num_relations"]
        self.args.num_tokens = None
        self.args.max_length_subword_tokens = None
        self.args.ordered_bpe_entities = None

        self.trained_model, form_of_labelling = self.trainer.continual_start(knowledge_graph)
        # (5) Store trained model.
        self.save_trained_model()
        # (6) Eval model.
        if self.args.eval_model is None:
            return self.report
        else:
            self.evaluator = Evaluator(args=self.args, is_continual_training=True)
            self.evaluator.dummy_eval(self.trained_model, form_of_labelling)
            return {**self.report, **self.evaluator.report}
