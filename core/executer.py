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
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelSummary

from core.callbacks import PrintCallback, KGESaveCallback, PseudoLabellingCallback
from core.dataset_classes import StandardDataModule
from core.helper_classes import LabelRelaxationLoss, BatchRelaxedvsAllLoss
from core.knowledge_graph import KG
from core.models.base_model import BaseKGE
from core.evaluator import Evaluator
from core.typings import *
from core.static_funcs import store, extract_model_summary, model_fitting, select_model, initialize_pl_trainer, \
    config_kge_sanity_checking, \
    preprocesses_input_args, create_experiment_folder, read_preprocess_index_serialize_kg, load_json, reload_input_data

from core.static_funcs import semi_supervised_split, non_conformity_score_diff, construct_p_values, norm_p_value, \
    gen_lr

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
        self.evaluator = Evaluator(self)

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
        self.dataset = reload_input_data(self.storage_path, cls=KG)

    def save_trained_model(self, trained_model: BaseKGE, start_time: float) -> None:
        """ Save a knowledge graph embedding model (an instance of BaseKGE class) """
        # (1) Send model to the eval mode
        trained_model.eval()
        # (2) Store NumParam and EstimatedSizeMB
        self.report.update(extract_model_summary(trained_model.summarize()))
        # (3) Store/Serialize Model for further use.
        if self.is_continual_training is False:
            store(trained_model, model_name='model', full_storage_path=self.storage_path,
                  dataset=self.dataset, save_as_csv=self.args.save_embeddings_as_csv)
        else:
            store(trained_model, model_name='model_' + str(datetime.datetime.now()),
                  dataset=self.dataset,
                  full_storage_path=self.storage_path)

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
        # (1) Data Preparation.
        if self.is_continual_training is False:
            # (1.1) Read, Preprocess, Index, and Serialize input data.
            self.read_preprocess_index_serialize_data()
        else:
            # (1.2) Load indexed input data.
            self.load_indexed_data()
        # (2) Train and Evaluate.
        trained_model = self.train_and_eval()
        # (3) Store trained model.
        self.save_trained_model(trained_model, start_time)
        # (4) Return the report of the training process.
        return self.report

    def train_and_eval(self) -> BaseKGE:
        """
        Training and evaluation procedure

        (1) Collect Callbacks to be used during training
        (2) Initialize Pytorch-lightning Trainer
        (3) Train a KGE modal via (2)
        (4) Eval trained model
        (5) Return trained model
        """
        self.report['num_entities'] = self.dataset.num_entities
        self.report['num_relations'] = self.dataset.num_relations
        print('------------------- Train & Eval -------------------')
        # (1) Collect Callbacks to be used during training
        callbacks = [PrintCallback(),
                     KGESaveCallback(every_x_epoch=self.args.save_model_at_every_epoch,
                                     max_epochs=self.args.max_epochs,
                                     path=self.args.full_storage_path), ModelSummary(max_depth=-1)]

        # PL has some problems with DDPPlugin. It will likely to be solved in their next release.
        # Explicitly setting num_process > 1 gives you
        """
        [W reducer.cpp: 1303] Warning: find_unused_parameters = True
        was specified in DDP constructor, but did not find any unused parameters in the
        forward pass.
        This flag results in an extra traversal of the autograd graph every iteration, which can adversely affect
        performance. If your model indeed never has any unused parameters in the forward pass, 
        consider turning this flag off. Note that this warning may be a false positive 
        if your model has flow        control        causing        later        iterations        to        have        unused
        parameters.(function        operator())
        """
        # Adding plugins=[DDPPlugin(find_unused_parameters=False)] and explicitly using num_process > 1
        """ pytorch_lightning.utilities.exceptions.DeadlockDetectedException: DeadLock detected from rank: 1  """
        # Force using SWA.
        # self.args.stochastic_weight_avg = True

        # (2) Initialize Pytorch-lightning Trainer
        self.trainer = initialize_pl_trainer(self.args, callbacks, plugins=[])
        # (3) Use (2) to train a KGE model
        trained_model, form_of_labelling = self.train()
        # (4) Eval model.
        self.evaluator.eval(trained_model, form_of_labelling)
        # (5) Return trained model
        return trained_model

    def train(self) -> Tuple[BaseKGE, str]:
        """ Train selected model via the selected training strategy """
        if self.args.num_folds_for_cv >= 2:
            return self.k_fold_cross_validation()
        else:
            if self.args.scoring_technique == 'NegSample':
                return self.training_negative_sampling()
            elif self.args.scoring_technique == 'KvsAll':
                return self.training_kvsall()
            elif self.args.scoring_technique == 'PvsAll':
                return self.training_PvsAll()
            elif self.args.scoring_technique == 'CCvsAll':
                return self.training_CCvsAll()
            elif self.args.scoring_technique == '1vsAll':
                return self.training_1vsall()
            elif self.args.scoring_technique == "BatchRelaxedKvsAll" or self.args.scoring_technique == "BatchRelaxed1vsAll":
                return self.train_relaxed_k_vs_all()
            else:
                raise ValueError(f'Invalid argument: {self.args.scoring_technique}')

    def training_CCvsAll(self) -> BaseKGE:
        """ Conformal Credal Self-Supervised Learning for KGE
        D:= {(x,y)}, where
        x is an input is a head-entity & relation pair
        y is a one-hot vector
        """
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        print(f'Conformal Credal Self Training starts: {model.name}')
        # Split the training triples into train, calibration and unlabelled.
        train_set, calibration_set, unlabelled_set = semi_supervised_split(self.dataset.train_set)
        model.calibration_set = torch.LongTensor(calibration_set)
        model.unlabelled_set = torch.LongTensor(unlabelled_set)

        variant = 0
        non_conf_score_fn = non_conformity_score_diff  # or  non_conformity_score_prop
        print('Variant:', variant)
        print('non_conf_score_fn:', non_conf_score_fn)

        dataset = StandardDataModule(train_set_idx=train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form='CCvsAll',
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_processes
                                     )

        def on_epoch_start(self, *args, **kwargs):
            """ Update non-conformity scores"""
            with torch.no_grad():
                # (1.1) Compute non-conformity scores on calibration dataset per epoch.
                self.non_conf_scores = non_conformity_score_diff(
                    torch.nn.functional.softmax(self.forward(self.calibration_set[:, [0, 1]])),
                    self.calibration_set[:, 2])

        setattr(BaseKGE, 'on_epoch_start', on_epoch_start)

        # Define a new raining set
        def training_step(self, batch, batch_idx):
            # (1) SUPERVISED PART
            # (1.1) Extract inputs and labels from a given batch (\mathcal{B}_l)
            x_batch, y_batch = batch
            # (1.2) Compute the supervised Loss
            train_loss = self.loss_function(yhat_batch=self.forward(x_batch), y_batch=y_batch)
            """
            # (1.3.2) Via KL divergence
            yhat = torch.clip(torch.softmax(logits_x, dim=-1), 1e-5, 1.)
            one_hot_targets = torch.clip(y_batch, 1e-5, 1.)
            train_loss = F.kl_div(yhat.log(), one_hot_targets, log_target=False, reduction='batchmean')
            """
            # (2) UNSUPERVISED PART
            # (2.1) Obtain unlabelled batch (\mathcal{B}_u), (x:=(s,p))
            unlabelled_input_batch = self.unlabelled_set[
                                         torch.randint(low=0, high=len(unlabelled_set), size=(len(x_batch),))][:,
                                     [0, 1]]
            # (2.2) Predict unlabelled batch \mathcal{B}_u
            with torch.no_grad():
                # TODO:Important moving this code outside of the no_grad improved the results a lot.
                # (2.2) Predict unlabelled batch \mathcal{B}_u
                pseudo_label = torch.nn.functional.softmax(self.forward(unlabelled_input_batch).detach())
                # (2.3) Construct p values given non-conformity scores and pseudo labels
                p_values = construct_p_values(self.non_conf_scores, pseudo_label.detach(),
                                              non_conf_score_fn=non_conformity_score_diff)
                # (2.4) Normalize (2.3)
                norm_p_values = norm_p_value(p_values, variant=0)

            unlabelled_loss = gen_lr(pseudo_label, norm_p_values)

            return train_loss + unlabelled_loss

        # Dynamically update
        setattr(BaseKGE, 'training_step', training_step)

        if self.args.eval is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=dataset.train_dataloader())
        return model, form_of_labelling

    def training_PvsAll(self) -> BaseKGE:
        """ Pseudo Labelling for KGE """

        # (1) Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        print(f'PvsAll training starts: {model.name}')
        train_set, calibration_set, unlabelled_set = semi_supervised_split(self.dataset.train_set)

        model.calibration_set = torch.LongTensor(calibration_set)
        model.unlabelled_set = torch.LongTensor(unlabelled_set)
        # (2) Create training data.
        dataset = StandardDataModule(train_set_idx=train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form='PvsAll',
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_processes)

        # Define a new raining set
        def training_step(self, batch, batch_idx):
            # (1) SUPERVISED PART
            # (1.1) Extract inputs and labels from a given batch
            x_batch, y_batch = batch
            # (1.2) Predictions
            logits_x = self.forward(x_batch)
            # (1.3) Compute the supervised Loss
            # (1.3.1) Via Cross Entropy
            supervised_loss = self.loss_function(yhat_batch=logits_x, y_batch=y_batch)
            # (2) UNSUPERVISED PART
            # (2.1) Obtain unlabelled batch (\mathcal{B}_u)
            random_idx = torch.randint(low=0, high=len(self.unlabelled_set), size=(len(x_batch),))
            # (2.2) Batch of head entity and relation
            unlabelled_x = self.unlabelled_set[random_idx][:, [0, 1]]
            # (2.3) Create Pseudo-Labels
            with torch.no_grad():
                # (2.2) Compute loss
                _, max_pseudo_tail = torch.max(self.forward(unlabelled_x), dim=1)
                pseudo_labels = F.one_hot(max_pseudo_tail, num_classes=y_batch.shape[1]).float()

            unlabelled_loss = self.loss_function(yhat_batch=self.forward(unlabelled_x), y_batch=pseudo_labels)

            return supervised_loss + unlabelled_loss

        # Dynamically update
        setattr(BaseKGE, 'training_step', training_step)

        if self.args.eval is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=dataset.train_dataloader())
        return model, form_of_labelling

    def training_kvsall(self) -> BaseKGE:
        """
        Train models with KvsAll
        D= {(x,y)_i }_i ^n where
        1. x denotes a tuple of indexes of a head entity and a relation
        2. y denotes a vector of probabilities, y_j corresponds to probability of j.th indexed entity
        :return: trained BASEKGE
        """
        # (1) Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        print(f'KvsAll training starts: {model.name}')  # -labeling:{form_of_labelling}')
        # (2) Create training data.
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_processes,
                                     label_smoothing_rate=self.args.label_smoothing_rate)
        # (3) Train model.
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        if self.args.eval is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
        """
        # @TODO Model Calibration
        from laplace import Laplace
        from laplace.utils.subnetmask import ModuleNameSubnetMask
        from laplace.utils import ModuleNameSubnetMask
        from laplace import Laplace
        # No change in link prediciton results
        subnetwork_mask = ModuleNameSubnetMask(model, module_names=['emb_ent_real'])
        subnetwork_mask.select()
        subnetwork_indices = subnetwork_mask.indices
        la = Laplace(model, 'classification',
                     subset_of_weights='subnetwork',
                     hessian_structure='full',
                     subnetwork_indices=subnetwork_indices)
        # la.fit(dataset.train_dataloader())
        # la.optimize_prior_precision(method='CV', val_loader=dataset.val_dataloader())
        """

        return model, form_of_labelling

    def training_1vsall(self) -> BaseKGE:
        # (1) Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        print(f'1vsAll training starts: {model.name}')
        # (2) Create training data.
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_processes
                                     )
        if self.args.label_relaxation_rate:
            model.loss = LabelRelaxationLoss(alpha=self.args.label_relaxation_rate)
            # model.loss=LabelSmoothingLossCanonical()
        elif self.args.label_smoothing_rate:
            model.loss = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_rate)
        else:
            model.loss = torch.nn.CrossEntropyLoss()
        # (3) Train model
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        if self.args.eval is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def training_negative_sampling(self) -> pl.LightningModule:
        """
        Train models with Negative Sampling
        """
        assert self.args.neg_ratio > 0
        model, _ = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        form_of_labelling = 'NegativeSampling'
        print(f'Training starts: {model.name}-labeling:{form_of_labelling}')
        print('Creating training data...', end='\t')
        start_time = time.time()
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=os.cpu_count() - 1)
        print(f'Done ! {time.time() - start_time:.3f} seconds\n')
        # 3. Train model
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        if self.args.eval is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def train_relaxed_k_vs_all(self) -> pl.LightningModule:
        model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
        print(f'{self.args.scoring_technique}training starts: {model.name}')  # -labeling:{form_of_labelling}')
        # 2. Create training data.)
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=self.args.scoring_technique,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_processes,
                                     label_smoothing_rate=self.args.label_smoothing_rate)
        # 3. Train model.
        train_dataloaders = dataset.train_dataloader()
        # Release some memory
        del dataset
        if self.args.eval is False:
            self.dataset.train_set = None
            self.dataset.valid_set = None
            self.dataset.test_set = None

        model.loss = BatchRelaxedvsAllLoss()
        model_fitting(trainer=self.trainer, model=model, train_dataloaders=train_dataloaders)
        return model, form_of_labelling

    def k_fold_cross_validation(self) -> Tuple[BaseKGE, str]:
        """
        Perform K-fold Cross-Validation

        1. Obtain K train and test splits.
        2. For each split,
            2.1 initialize trainer and model
            2.2. Train model with configuration provided in args.
            2.3. Compute the mean reciprocal rank (MRR) score of the model on the test respective split.
        3. Report the mean and average MRR .

        :param self:
        :return: model
        """
        print(f'{self.args.num_folds_for_cv}-fold cross-validation')
        kf = KFold(n_splits=self.args.num_folds_for_cv, shuffle=True, random_state=1)
        model = None
        eval_folds = []

        for (ith, (train_index, test_index)) in enumerate(kf.split(self.dataset.train_set)):
            trainer = pl.Trainer.from_argparse_args(self.args)
            model, form_of_labelling = select_model(vars(self.args), self.is_continual_training, self.storage_path)
            print(f'{form_of_labelling} training starts: {model.name}')  # -labeling:{form_of_labelling}')

            train_set_for_i_th_fold, test_set_for_i_th_fold = self.dataset.train_set[train_index], \
                                                              self.dataset.train_set[
                                                                  test_index]

            dataset = StandardDataModule(train_set_idx=train_set_for_i_th_fold,
                                         entity_to_idx=self.dataset.entity_to_idx,
                                         relation_to_idx=self.dataset.relation_to_idx,
                                         form=form_of_labelling,
                                         neg_sample_ratio=self.args.neg_ratio,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_processes)
            # 3. Train model
            train_dataloaders = dataset.train_dataloader()
            del dataset
            model_fitting(trainer=trainer, model=model, train_dataloaders=train_dataloaders)

            # 6. Test model on validation and test sets if possible.
            res = self.evaluator.evaluate_lp_k_vs_all(model, test_set_for_i_th_fold,
                                                      form_of_labelling=form_of_labelling)
            print(res)
            eval_folds.append([res['MRR'], res['H@1'], res['H@3'], res['H@10']])
        eval_folds = pd.DataFrame(eval_folds, columns=['MRR', 'H@1', 'H@3', 'H@10'])

        results = {'H@1': eval_folds['H@1'].mean(), 'H@3': eval_folds['H@3'].mean(), 'H@10': eval_folds['H@10'].mean(),
                   'MRR': eval_folds['MRR'].mean()}
        print(f'Evaluate {model.name} on test set: {results}')
        return model, form_of_labelling


class ContinuousExecute(Execute):
    """ Continue training a pretrained KGE model """

    def __init__(self, args):
        assert os.path.exists(args.path_experiment_folder)
        assert os.path.isfile(args.path_experiment_folder + '/idx_train_df.gzip')
        assert os.path.isfile(args.path_experiment_folder + '/configuration.json')
        # (1) Load Previous input configuration
        previous_args = load_json(args.path_experiment_folder + '/configuration.json')
        # (2) Update (1) with new input
        previous_args.update(vars(args))
        report = load_json(args.path_experiment_folder + '/report.json')
        previous_args['num_entities'] = report['num_entities']
        previous_args['num_relations'] = report['num_relations']
        previous_args = SimpleNamespace(**previous_args)
        previous_args.full_storage_path = previous_args.path_experiment_folder
        print('ContinuousExecute starting...')
        super().__init__(previous_args, continuous_training=True)
