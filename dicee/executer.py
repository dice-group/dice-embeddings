"""Executor module for training, retraining and evaluating KGE models.

This module provides the Execute and ContinuousExecute classes for managing
the full lifecycle of knowledge graph embedding model training.
"""
import datetime
import json
import logging
import os
import shutil
import time
import warnings
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from .evaluator import Evaluator
from .knowledge_graph import KG
from .static_funcs import (
    create_experiment_folder,
    load_json,
    read_or_load_kg,
    store,
    timeit,
)
from .static_preprocess_funcs import preprocesses_input_args
from .trainer import DICE_Trainer

# Configure logging
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

class Execute:
    """Executor class for training, retraining and evaluating KGE models.

    Handles the complete workflow:
    1. Loading & Preprocessing & Serializing input data
    2. Training & Validation & Testing
    3. Storing all necessary information

    Attributes:
        args: Processed input arguments.
        distributed: Whether distributed training is enabled.
        rank: Process rank in distributed training.
        world_size: Total number of processes.
        local_rank: Local GPU rank.
        trainer: Training handler instance.
        trained_model: The trained model after training completes.
        knowledge_graph: The loaded knowledge graph.
        report: Dictionary storing training metrics and results.
        evaluator: Model evaluation handler.
    """

    def __init__(self, args, continuous_training: bool = False):
        """Initialize the executor.

        Args:
            args: Configuration arguments (Namespace or similar).
            continuous_training: Whether this is continual training.
        """
        # Check if we need distributed training
        self.distributed = getattr(args, "trainer", None) == "torchDDP"
        # Initialize distributed training if required
        self._setup_distributed_training()

        # (1) Process arguments and sanity checking
        self.args = preprocesses_input_args(args)
        # (2) Ensure reproducibility
        seed_everything(args.random_seed, workers=True)
        # (3) Set the continual training flag
        self.is_continual_training = continuous_training
        # (4) Create an experiment folder or use the previous one
        if self.rank == 0:
            self.setup_executor()
        # (5) Initialize trainer and model placeholders
        self.trainer: Optional[DICE_Trainer] = None
        self.trained_model = None
        # (6) Initialize knowledge graph placeholder
        self.knowledge_graph: Optional[KG] = None
        # (7) Store metrics and results
        self.report: Dict = {}
        # (8) Evaluator placeholder
        self.evaluator: Optional[Evaluator] = None
        # (9) Execution start time
        self.start_time: Optional[float] = None

    def _setup_distributed_training(self) -> None:
        """Set up distributed training environment if enabled."""
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

    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def cleanup(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    @rank_zero_only
    def setup_executor(self) -> None:
        """Set up storage directories for the experiment.

        Creates or reuses experiment directories based on configuration.
        Saves the configuration to a JSON file.
        """
        if self.is_continual_training:
            return

        # Determine storage path
        if self.args.path_to_store_single_run is not None:
            self._setup_single_run_directory()
        else:
            # Create a new timestamped experiment folder
            self.args.full_storage_path = create_experiment_folder(
                folder_name=self.args.storage_path
            )
            self.args.path_to_store_single_run = self.args.full_storage_path

        # Save configuration
        config_path = os.path.join(self.args.full_storage_path, 'configuration.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=3)

    def _setup_single_run_directory(self) -> None:
        """Set up a specific directory for a single run."""
        reuse_existing = getattr(self.args, "reuse_existing_run_dir", False)
        path = self.args.path_to_store_single_run

        if os.path.exists(path):
            if not reuse_existing:
                print(f"Deleting existing directory: {path}")
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=False)
            else:
                print(f"Reusing existing directory: {path}")
        else:
            os.makedirs(path, exist_ok=False)

        self.args.full_storage_path = path

    def create_and_store_kg(self) -> None:
        """Create knowledge graph and store as memory-mapped file.

        Only executed on rank 0 in distributed training.
        Skips if memmap already exists.
        """
        if not self.is_rank_zero():
            return

        memmap_path = os.path.join(
            self.args.path_to_store_single_run, "memory_map_train_set.npy"
        )
        details_path = os.path.join(
            self.args.path_to_store_single_run, "memory_map_details.json"
        )

        if os.path.exists(memmap_path) and os.path.exists(details_path):
            print("KG memmap already exists, skipping.")
            return

        print("Creating knowledge graph...")
        self.knowledge_graph = read_or_load_kg(self.args, cls=KG)
        self._update_args_from_kg()
        self._save_kg_memmap(memmap_path, details_path)

    def _update_args_from_kg(self) -> None:
        """Update args with knowledge graph statistics."""
        kg = self.knowledge_graph
        self.args.num_entities = kg.num_entities
        self.args.num_relations = kg.num_relations
        self.args.num_tokens = kg.num_tokens
        self.args.max_length_subword_tokens = kg.max_length_subword_tokens
        self.args.ordered_bpe_entities = kg.ordered_bpe_entities

        self.report['num_train_triples'] = len(kg.train_set)
        self.report['num_entities'] = kg.num_entities
        self.report['num_relations'] = kg.num_relations
        self.report['max_length_subword_tokens'] = kg.max_length_subword_tokens
        self.report['runtime_kg_loading'] = time.time() - self.start_time

    def _save_kg_memmap(self, memmap_path: str, details_path: str) -> None:
        """Save knowledge graph to memory-mapped file."""
        kg = self.knowledge_graph
        data = {
            "shape": tuple(kg.train_set.shape),
            "dtype": kg.train_set.dtype.str,
            "num_entities": kg.num_entities,
            "num_relations": kg.num_relations,
        }

        with open(details_path, 'w') as f:
            json.dump(data, f, indent=4)

        memmap_kg = np.memmap(
            memmap_path,
            dtype=kg.train_set.dtype,
            mode='w+',
            shape=kg.train_set.shape
        )
        memmap_kg[:] = kg.train_set[:]
        memmap_kg.flush()
        del memmap_kg
    
    def load_from_memmap(self) -> None:
        """Load knowledge graph from memory-mapped file."""
        base_path = self.args.path_to_store_single_run
        details_path = os.path.join(base_path, 'memory_map_details.json')
        memmap_path = os.path.join(base_path, 'memory_map_train_set.npy')

        with open(details_path, 'r') as f:
            memory_map_details = json.load(f)

        self.knowledge_graph = np.memmap(
            memmap_path,
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
