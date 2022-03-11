import json
import logging
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from sklearn.model_selection import KFold

from .callbacks import PrintCallback, KGESaveCallback
from .dataset_classes import StandardDataModule
from .helper_classes import LabelRelaxationLoss
from .knowledge_graph import KG
from .static_funcs import *

logging.getLogger('pytorch_lightning').setLevel(0)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


# TODO: Execute can inherit from Trainer and Evaluator Classes
# By doing so we can increase the modularity of our code.
class Execute:
    def __init__(self, args, continuous_training=False):
        # (1) Process arguments and sanity checking.
        self.args = preprocesses_input_args(args)
        # (2) Ensure reproducibility.
        seed_everything(args.seed_for_computation, workers=True)
        # (3) Set the continual training flag
        self.is_continual_training = continuous_training
        # (4) Create an experiment folder or use the previous one
        if self.is_continual_training:
            self.storage_path = self.args.full_storage_path
        else:
            self.args.full_storage_path = create_experiment_folder(folder_name=self.args.storage_path)
            self.storage_path = self.args.full_storage_path
            print('Saving configuration..')
            with open(self.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
                temp = vars(self.args)
                json.dump(temp, file_descriptor, indent=3)
        # (5) A variable is initialized for pytorch lightning trainer.
        self.trainer = None
        # (6) A variable is initialized for storing input data.
        self.dataset = None
        # (7) Store few data in memory for numerical results, e.g. runtime, H@1 etc.
        self.report = dict()

    def start(self) -> dict:
        """
        1. READ/LOAD input knowledge graph
        2. Train and Eval a knowledge graph embedding mode
        3. Store relevant intermediate data including the trained model, embeddings and configuration
        4. Return brief summary of the computation in dictionary format.
        """
        start_time = time.time()
        # (1) READ/LOAD input knowledge graph
        if self.is_continual_training is False:
            self.dataset = read_input_data(self.args, cls=KG)
            self.args.num_entities, self.args.num_relations = self.dataset.num_entities, self.dataset.num_relations
            self.args, self.dataset = config_kge_sanity_checking(self.args, self.dataset)
        else:
            self.dataset = reload_input_data(self.storage_path, cls=KG)
        self.report['num_entities'] = self.dataset.num_entities
        self.report['num_relations'] = self.dataset.num_relations

        # (2) Train and Evaluate
        trained_model = self.train_and_eval()
        # (3) Store trained model
        if self.is_continual_training is False:
            self.store(trained_model, model_name='model')
        else:
            self.store(trained_model, model_name='model_' + str(datetime.datetime.now()))

        total_runtime = time.time() - start_time
        if 60 * 60 > total_runtime:
            message = f'{total_runtime / 60:.3f} minutes'
        else:
            message = f'{total_runtime / (60 ** 2):.3f} hours'
        self.report['Runtime'] = message
        self.report['path_experiment_folder'] = self.storage_path

        print(f'Total computation time: {message}')
        print(f'Number of parameters in {trained_model.name}:', self.report["NumParam"])
        # print(f'Estimated of {trained_model.name}:', self.report["EstimatedSizeMB"])
        with open(self.args.full_storage_path + '/report.json', 'w') as file_descriptor:
            json.dump(self.report, file_descriptor, indent=4)
        return self.report

    def train_and_eval(self) -> BaseKGE:
        """
        Training and evaluation procedure

        1. Create Pytorch-lightning Trainer object from input configuration
        2a. Train and Test a model if  test dataset is available
        2b. Train a model in k-fold cross validation mode if it is requested
        2c. Train a model
        """
        print('------------------- Train & Eval -------------------')
        callbacks = [PrintCallback(),
                     KGESaveCallback(every_x_epoch=self.args.num_epochs, path=self.args.full_storage_path)]

        # PL has some problems with DDPPlugin. It will likely to be solved in their next release.
        # (1) Explicitly setting num_process > 1 gives you
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
        # (2) Adding plugins=[DDPPlugin(find_unused_parameters=False)] and explicitly using num_process > 1
        """ pytorch_lightning.utilities.exceptions.DeadlockDetectedException: DeadLock detected from rank: 1  """

        plugins = []
        self.args.stochastic_weight_avg = True  # => https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # (3) Surprisingly, if you do not ask explicitly num_process > 1, computation runs smoothly while using many CPUs
        if self.args.gpus:
            plugins.append(DDPPlugin(find_unused_parameters=False))
            plugins.append(DeepSpeedPlugin(stage=3))  # experiment with it when we use GPUs
            self.trainer = pl.Trainer.from_argparse_args(self.args, plugins=plugins,
                                                         callbacks=callbacks)
        else:
            self.trainer = pl.Trainer.from_argparse_args(self.args, plugins=plugins,
                                                         callbacks=callbacks)

        if self.args.num_folds_for_cv >= 2:
            trained_model = self.k_fold_cross_validation()
        else:
            if self.args.scoring_technique == 'NegSample':
                trained_model = self.training_negative_sampling()
            elif self.args.scoring_technique == 'KvsAll':
                trained_model = self.training_kvsall()
            elif self.args.scoring_technique == '1vsAll':
                trained_model = self.training_1vsall()
            else:
                raise ValueError(f'Invalid argument: {self.args.scoring_technique}')
        """
        # 2. Check whether validation and test datasets are available.
        if self.dataset.is_valid_test_available():
            if self.args.scoring_technique == 'NegSample':
                trained_model = self.training_negative_sampling()
            elif self.args.scoring_technique == 'KvsAll':
                trained_model = self.training_kvsall()
            elif self.args.scoring_technique == '1vsAll':
                trained_model = self.training_1vsall()
            else:
                raise ValueError(f'Invalid argument: {self.args.scoring_technique}')
        else:
            # 3. If (2) is FALSE, then check whether cross validation will be applied.
            print(f'There is no validation and test sets available.')
            if self.args.num_folds_for_cv < 2:
                print(
                    f'No test set is found and k-fold cross-validation is set to less than 2 (***num_folds_for_cv*** => {self.args.num_folds_for_cv}). Hence we do not evaluate the model')
                # 3.1. NO CROSS VALIDATION => TRAIN WITH 'NegSample' or KvsALL
                if self.args.scoring_technique == 'NegSample':
                    trained_model = self.training_negative_sampling()
                elif self.args.scoring_technique == 'KvsAll':
                    # KvsAll or negative sampling
                    trained_model = self.training_kvsall()
                elif self.args.scoring_technique == '1vsAll':
                    trained_model = self.training_1vsall()
                else:
                    raise ValueError(f'Invalid argument: {self.args.scoring_technique}')
            else:
                trained_model = self.k_fold_cross_validation()
        """

        return trained_model

    def store(self, trained_model, model_name='model') -> None:
        """
        Store trained_model model and save embeddings into csv file.
        :param model_name:
        :param trained_model:
        :return:
        """
        print('------------------- Store -------------------')
        # Save Torch model.
        store_kge(trained_model, path=self.args.full_storage_path + f'/{model_name}.pt')

        # See available memory and decide whether embeddings are stored separately or not.
        available_memory = [i.split() for i in os.popen('free -h').read().splitlines()][1][-1]  # ,e.g., 10Gi
        available_memory_mb = float(available_memory[:-2]) * 1000
        self.report.update(extract_model_summary(trained_model.summarize()))

        if available_memory_mb * .01 > self.report['EstimatedSizeMB']:
            """ We have enough space for data conversion"""
            print('Saving embeddings..')
            entity_emb, relation_ebm = trained_model.get_embeddings()

            if len(entity_emb) > 1:
                torch.save(entity_emb, self.args.full_storage_path + '/' + trained_model.name + '_entity_embeddings.pt')
            else:
                save_embeddings(entity_emb.numpy(), indexes=self.dataset.entities_str,
                                path=self.args.full_storage_path + '/' + trained_model.name + '_entity_embeddings.csv')

            del entity_emb

            if relation_ebm is not None:
                if len(relation_ebm) > 1:
                    torch.save(relation_ebm,
                               self.args.full_storage_path + '/' + trained_model.name + '_relation_embeddings.pt')
                else:
                    save_embeddings(relation_ebm.numpy(), indexes=self.dataset.relations_str,
                                    path=self.args.full_storage_path + '/' + trained_model.name + '_relation_embeddings.csv')
            del relation_ebm

        else:
            print('There is not enough memory to store embeddings separately.')

    def training_kvsall(self) -> BaseKGE:
        """
        Train models with KvsAll
        D= {(x,y)_i }_i ^n where
        1. x denotes a tuple of indexes of a head entity and a relation
        2. y denotes a vector of probabilities, y_j corresponds to probability of j.th indexed entity
        :return: trained BASEKGE
        """
        # 1. Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = self.select_model(vars(self.args))
        print(f'KvsAll training starts: {model.name}')  # -labeling:{form_of_labelling}')
        # 2. Create training data.)
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
        # 3. Train model.
        self.model_fitting(trainer=self.trainer, model=model, dataset=dataset)
        """
        # @TODO
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
        # 4. Test model on the training dataset if it is needed.
        if self.args.eval_on_train:
            res = self.evaluate_lp_k_vs_all(model, self.dataset.train_set,
                                            info=f'Evaluate {model.name} on Train set',
                                            form_of_labelling=form_of_labelling)
            self.report['Train'] = res

        # 5. Test model on the validation and test dataset if it is needed.
        if self.args.eval:
            if len(self.dataset.valid_set) > 0:
                res = self.evaluate_lp_k_vs_all(model, self.dataset.valid_set,
                                                f'Evaluate {model.name} on Validation set',
                                                form_of_labelling=form_of_labelling)
                self.report['Val'] = res
            if len(self.dataset.test_set) > 0:
                res = self.evaluate_lp_k_vs_all(model, self.dataset.test_set, f'Evaluate {model.name} on Test set',
                                                form_of_labelling=form_of_labelling)
                self.report['Test'] = res

        return model

    def training_1vsall(self):
        # 1. Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = self.select_model(vars(self.args))
        print(f'1vsAll training starts: {model.name}')
        # 2. Create training data.
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
            model.loss = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_rate)
        else:
            model.loss = nn.CrossEntropyLoss()
        # 3. Train model
        self.model_fitting(trainer=self.trainer, model=model, dataset=dataset)
        # 4. Test model on the training dataset if it is needed.
        if self.args.eval_on_train:
            res = self.evaluate_lp_k_vs_all(model, self.dataset.train_set,
                                            f'Evaluate {model.name} on train set', form_of_labelling)
            self.report['Train'] = res

        # 5. Test model on the validation and test dataset if it is needed.
        if self.args.eval:
            if len(self.dataset.valid_set) > 0:
                res = self.evaluate_lp_k_vs_all(model, self.dataset.valid_set,
                                                f'Evaluate {model.name} on validation set', form_of_labelling)
                self.report['Val'] = res
            if len(self.dataset.test_set) > 0:
                res = self.evaluate_lp_k_vs_all(model, self.dataset.test_set, f'Evaluate {model.name} on test set',
                                                form_of_labelling)
                self.report['Test'] = res

        return model

    def select_model(self, args: dict):
        if self.is_continual_training:
            print('Loading pre-trained model...')
            model, _ = select_model(args)
            try:
                weights = torch.load(self.storage_path + '/model.pt', torch.device('cpu'))
                model.load_state_dict(weights)
            except FileNotFoundError:
                print(f"{self.storage_path}/model.pt is not found. The model will be trained with random weights")
            model.train()
            return model, _
        else:
            return select_model(args)

    def model_fitting(self, trainer, model, dataset) -> None:
        train_dataloaders = dataset.train_dataloader()
        del dataset
        if self.args.eval is False and self.args.eval_on_train is False:
            # release some memory
            del self.dataset
        print(f'Number of mini-batches to compute for a single epoch: {len(train_dataloaders)}')
        trainer.fit(model, train_dataloaders=train_dataloaders)

    def training_negative_sampling(self) -> pl.LightningModule:
        """
        Train models with Negative Sampling
        """
        assert self.args.neg_ratio > 0
        model, _ = self.select_model(vars(self.args))
        form_of_labelling = 'NegativeSampling'
        print(f'Training starts: {model.name}-labeling:{form_of_labelling}')
        print('Creating training data...')
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.args.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_processes)
        # 3. Train model
        self.model_fitting(trainer=self.trainer, model=model, dataset=dataset)
        # 4. Test model on the training dataset if it is needed.
        if self.args.eval_on_train:
            res = self.evaluate_lp(model, self.dataset.train_set, f'Evaluate {model.name} on Train set')
            self.report['Train'] = res
        # 5. Test model on the validation and test dataset if it is needed.
        if self.args.eval:
            if len(self.dataset.valid_set) > 0:
                self.report['Val'] = self.evaluate_lp(model, self.dataset.valid_set, 'Evaluation of Validation set')

            if len(self.dataset.test_set) > 0:
                self.report['Test'] = self.evaluate_lp(model, self.dataset.test_set, 'Evaluation of Test set')

        return model

    def evaluate_lp_k_vs_all(self, model, triple_idx, info=None, form_of_labelling=None):
        """
        Filtered link prediction evaluation.
        :param model:
        :param triple_idx: test triples
        :param info:
        :param form_of_labelling:
        :return:
        """
        # (1) set model to eval model
        model.eval()
        hits = []
        ranks = []
        if info:
            print(info + ':', end=' ')
        for i in range(10):
            hits.append([])

        # (2) Evaluation mode
        if form_of_labelling == 'RelationPrediction':
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, len(triple_idx), self.args.batch_size):
                data_batch = triple_idx[i:i + self.args.batch_size]
                e1_idx_e2_idx, r_idx = torch.tensor(data_batch[:, [0, 2]]), torch.tensor(data_batch[:, 1])
                # Generate predictions
                predictions = model.forward_k_vs_all(x=e1_idx_e2_idx)
                # Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    filt = self.dataset.ee_vocab[(data_batch[j][0], data_batch[j][2])]
                    target_value = predictions[j, r_idx[j]].item()
                    predictions[j, filt] = -np.Inf
                    predictions[j, r_idx[j]] = target_value
                # Sort predictions.
                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                # This can be also done in parallel
                for j in range(data_batch.shape[0]):
                    rank = torch.where(sort_idxs[j] == r_idx[j])[0].item()
                    ranks.append(rank + 1)

                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)

        else:
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, len(triple_idx), self.args.batch_size):
                data_batch = triple_idx[i:i + self.args.batch_size]
                e1_idx_r_idx, e2_idx = torch.tensor(data_batch[:, [0, 1]]), torch.tensor(data_batch[:, 2])
                predictions = model.forward_k_vs_all(e1_idx_r_idx)
                # Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    filt = self.dataset.er_vocab[(data_batch[j][0], data_batch[j][1])]
                    target_value = predictions[j, e2_idx[j]].item()
                    predictions[j, filt] = -np.Inf
                    predictions[j, e2_idx[j]] = target_value
                # Sort predictions.
                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                # This can be also done in paralel
                for j in range(data_batch.shape[0]):
                    rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item()
                    ranks.append(rank + 1)

                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
        hit_1 = sum(hits[0]) / (float(len(triple_idx)))
        hit_3 = sum(hits[2]) / (float(len(triple_idx)))
        hit_10 = sum(hits[9]) / (float(len(triple_idx)))
        mean_reciprocal_rank = np.mean(1. / np.array(ranks))

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
        if info:
            print(results)
        return results

    def evaluate_lp(self, model, triple_idx, info):
        """
        Evaluate model in a standard link prediction task

        for each triple
        the rank is computed by taking the mean of the filtered missing head entity rank and
        the filtered missing tail entity rank
        :param model:
        :param triple_idx:
        :param info:
        :return:
        """
        model.eval()
        print(info)
        print(f'Num of triples {len(triple_idx)}')
        print('** sequential computation ')
        hits = dict()
        reciprocal_ranks = []
        # Iterate over test triples
        all_entities = torch.arange(0, self.dataset.num_entities).long()
        all_entities = all_entities.reshape(len(all_entities), )
        for i in range(0, len(triple_idx)):
            # 1. Get a triple
            data_point = triple_idx[i]
            s, p, o = data_point[0], data_point[1], data_point[2]

            # 2. Predict missing heads and tails
            x = torch.stack((torch.tensor(s).repeat(self.dataset.num_entities, ),
                             torch.tensor(p).repeat(self.dataset.num_entities, ),
                             all_entities
                             ), dim=1)
            predictions_tails = model.forward_triples(x)
            x = torch.stack((all_entities,
                             torch.tensor(p).repeat(self.dataset.num_entities, ),
                             torch.tensor(o).repeat(self.dataset.num_entities)
                             ), dim=1)

            predictions_heads = model.forward_triples(x)
            del x

            # 3. Computed filtered ranks for missing tail entities.
            # 3.1. Compute filtered tail entity rankings
            filt_tails = self.dataset.er_vocab[(s, p)]
            # filt_tails = data[(data['subject'] == s) & (data['relation'] == p)]['object'].values
            # 3.2 Get the predicted target's score
            target_value = predictions_tails[o].item()
            # 3.3 Filter scores of all triples containing filtered tail entities
            predictions_tails[filt_tails] = -np.Inf
            # 3.4 Reset the target's score
            predictions_tails[o] = target_value
            # 3.5. Sotrt the score
            _, sort_idxs = torch.sort(predictions_tails, descending=True)
            # sort_idxs = sort_idxs.cpu().numpy()
            sort_idxs = sort_idxs.detach()  # cpu().numpy()
            filt_tail_entity_rank = np.where(sort_idxs == o)[0][0]

            # 4. Computed filtered ranks for missing head entities.
            # 4.1. Retrieve head entities to be filterred
            filt_heads = self.dataset.re_vocab[(p, o)]
            # filt_heads = data[(data['relation'] == p) & (data['object'] == o)]['subject'].values
            # 4.2 Get the predicted target's score
            target_value = predictions_heads[s].item()
            # 4.3 Filter scores of all triples containing filtered head entities.
            predictions_heads[filt_heads] = -np.Inf
            predictions_heads[s] = target_value
            _, sort_idxs = torch.sort(predictions_heads, descending=True)
            # sort_idxs = sort_idxs.cpu().numpy()
            sort_idxs = sort_idxs.detach()  # cpu().numpy()
            filt_head_entity_rank = np.where(sort_idxs == s)[0][0]

            # 4. Add 1 to ranks as numpy array first item has the index of 0.
            filt_head_entity_rank += 1
            filt_tail_entity_rank += 1

            rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
            # 5. Store reciprocal ranks.
            reciprocal_ranks.append(rr)
            # print(f'{i}.th triple: mean reciprical rank:{rr}')

            # 4. Compute Hit@N
            for hits_level in range(1, 11):
                I = 1 if filt_head_entity_rank <= hits_level else 0
                I += 1 if filt_tail_entity_rank <= hits_level else 0
                if I > 0:
                    hits.setdefault(hits_level, []).append(I)

        mean_reciprocal_rank = sum(reciprocal_ranks) / (float(len(triple_idx) * 2))

        if 1 in hits:
            hit_1 = sum(hits[1]) / (float(len(triple_idx) * 2))
        else:
            hit_1 = 0

        if 3 in hits:
            hit_3 = sum(hits[3]) / (float(len(triple_idx) * 2))
        else:
            hit_3 = 0

        if 10 in hits:
            hit_10 = sum(hits[10]) / (float(len(triple_idx) * 2))
        else:
            hit_10 = 0

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
                   'MRR': mean_reciprocal_rank}
        print(results)
        return results

    def k_fold_cross_validation(self) -> pl.LightningModule:
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
            model, form_of_labelling = self.select_model(vars(self.args))
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
                                         num_workers=self.args.num_processes
                                         )

            # 3. Train model
            self.model_fitting(trainer=self.trainer, model=model, dataset=dataset)

            # 6. Test model on validation and test sets if possible.
            res = self.evaluate_lp_k_vs_all(model, test_set_for_i_th_fold, form_of_labelling=form_of_labelling)
            print(res)
            eval_folds.append([res['MRR'], res['H@1'], res['H@3'], res['H@10']])
        eval_folds = pd.DataFrame(eval_folds, columns=['MRR', 'H@1', 'H@3', 'H@10'])

        results = {'H@1': eval_folds['H@1'].mean(), 'H@3': eval_folds['H@3'].mean(), 'H@10': eval_folds['H@10'].mean(),
                   'MRR': eval_folds['MRR'].mean()}
        print(f'Evaluate {model.name} on test set: {results}')

        # Return last model.
        return model


class ContinuousExecute(Execute):
    def __init__(self, args):
        assert os.path.exists(args.path_experiment_folder)
        assert os.path.isfile(args.path_experiment_folder + '/idx_train_df.gzip')
        assert os.path.isfile(args.path_experiment_folder + '/configuration.json')
        previous_args = load_json(args.path_experiment_folder + '/configuration.json')
        previous_args.update(vars(args))
        report = load_json(args.path_experiment_folder + '/report.json')
        previous_args['num_entities'] = report['num_entities']
        previous_args['num_relations'] = report['num_relations']
        previous_args = SimpleNamespace(**previous_args)
        super().__init__(previous_args, continuous_training=True)
