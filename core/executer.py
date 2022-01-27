import warnings
import os
from .models import *

from .dataset_classes import StandardDataModule, KvsAll, CVDataModule
from .knowledge_graph import KG
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from .static_funcs import *
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import json
import inspect
import dask.dataframe as dd
import time
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, seed_everything
import logging
from collections import defaultdict

logging.getLogger('pytorch_lightning').setLevel(0)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
seed_everything(1, workers=True)


class Execute:
    def __init__(self, args):
        # (1) Process arguments and sanity checking
        self.args = preprocesses_input_args(args)
        # 2 Create a folder to serialize data
        self.args.full_storage_path = create_experiment_folder(folder_name=self.args.storage_path)
        # 3. Read input data and store its parts for further use
        self.dataset = self.read_input_data(self.args)
        # 4. Store few data in memory for numerical results, e.g. runtime, H@1 etc.
        self.report = dict()
        self.config_kge_sanity_checking()

    @staticmethod
    def read_input_data(args):
        kg = KG(data_dir=args.path_dataset_folder,
                deserialize_flag=args.deserialize_flag,
                large_kg_parse=args.large_kg_parse,
                add_reciprical=args.add_reciprical,
                eval_model=args.eval,
                read_only_few=args.read_only_few,
                path_for_serialization=args.full_storage_path)
        print(kg.description_of_input)
        # 4. Save Create folder to serialize data. This two numerical value will be used in embedding initialization.
        args.num_entities, args.num_relations = kg.num_entities, kg.num_relations
        return kg

    def config_kge_sanity_checking(self):
        """
        Sanity checking for input hyperparams.
        :return:
        """
        if self.args.batch_size > len(self.dataset.train_set):
            self.args.batch_size = len(self.dataset.train_set)
        if self.args.model == 'Shallom' and self.args.scoring_technique == 'NegSample':
            print(
                'Shallom can not be trained with Negative Sampling. Scoring technique is changed to KvsALL')
            self.args.scoring_technique = 'KvsAll'

        if self.args.scoring_technique == 'KvsAll':
            self.args.neg_ratio = None

    def store(self, trained_model) -> None:
        """
        Store trained_model model and save embeddings into csv file.
        :param trained_model:
        :return:
        """
        print('------------------- Store -------------------')
        # Save Torch model.
        print('Saving torch model..')
        torch.save(trained_model.state_dict(), self.args.full_storage_path + '/model.pt')
        print('Saving configuration..')
        with open(self.args.full_storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            json.dump(temp, file_descriptor)
        print('Saving embeddings..')
        if trained_model.name == 'Shallom':
            entity_emb = trained_model.get_embeddings()
        else:
            entity_emb, relation_ebm = trained_model.get_embeddings()
            try:
                df = pd.DataFrame(relation_ebm, index=self.dataset.relations_str)
                df.columns = df.columns.astype(str)
                num_mb = df.memory_usage(index=True, deep=True).sum() / (10 ** 6)
                if num_mb > 10 ** 6:
                    df = dd.from_pandas(df, npartitions=len(df) / 100)
                    # PARQUET wants columns to be stn
                    df.columns = df.columns.astype(str)
                    df.to_parquet(self.args.full_storage_path + '/' + trained_model.name + '_relation_embeddings')
                    # TO READ PARQUET FILE INTO PANDAS
                    # m=dd.read_parquet(self.storage_path + '/' + trained_model.name + '_relation_embeddings').compute()
                else:
                    df.to_csv(self.args.full_storage_path + '/' + trained_model.name + '_relation_embeddings.csv')
            except KeyError or AttributeError as e:
                print('Exception occurred at saving relation embeddings. Computation will continue')
                print(e)

            # Free mem del
            del df
            del relation_ebm
        try:
            df = pd.DataFrame(entity_emb, index=self.dataset.entities_str)
            num_mb = df.memory_usage(index=True, deep=True).sum() / (10 ** 6)
            if num_mb > 10 ** 6:
                df = dd.from_pandas(df, npartitions=len(df) / 100)
                # PARQUET wants columns to be stn
                df.columns = df.columns.astype(str)
                df.to_parquet(self.args.full_storage_path + '/' + trained_model.name + '_relation_embeddings')
            else:
                df.to_csv(self.args.full_storage_path + '/' + trained_model.name + '_entity_embeddings.csv', )
        except KeyError or AttributeError as e:
            print('Exception occurred at saving entity embeddings.Computation will continue')
            print(e)

    def start(self) -> dict:
        """
        Train and/or Evaluate Model
        Store Mode
        """
        start_time = time.time()
        # 1. Train and Evaluate
        trained_model = self.train_and_eval()
        # 2. Store trained model
        self.store(trained_model)
        total_runtime = time.time() - start_time
        if 60 * 60 > total_runtime:
            message = f'{total_runtime / 60:.3f} minutes'
        else:
            message = f'{total_runtime / (60 ** 2):.3f} hours'
        self.report['Runtime'] = message
        self.report.update(extract_model_summary(trained_model.summarize()))
        print(f'Runtime of {trained_model.name}:', total_runtime)
        print(f'NumParam of {trained_model.name}:', self.report["NumParam"])
        print(f'Estimated of {trained_model.name}:', self.report["EstimatedSizeMB"])
        with open(self.args.full_storage_path + '/report.json', 'w') as file_descriptor:
            json.dump(self.report, file_descriptor)
        return self.report

    def train_and_eval(self) -> BaseKGE:
        """
        Training and evaluation procedure
        """
        print('------------------- Train & Eval -------------------')
        # 1. Create Pytorch-lightning Trainer object from input configuration
        if self.args.gpus:
            self.trainer = pl.Trainer.from_argparse_args(self.args, plugins=[DDPPlugin(find_unused_parameters=False)])
        else:
            self.trainer = pl.Trainer.from_argparse_args(self.args)
        # 2. Check whether validation and test datasets are available.
        if self.dataset.is_valid_test_available():
            if self.args.scoring_technique == 'NegSample':
                trained_model = self.training_negative_sampling()
            elif self.args.scoring_technique == 'KvsAll':
                # KvsAll or negative sampling
                trained_model = self.training_kvsall()
            else:
                raise ValueError(f'Invalid argument: {self.scoring_technique}')
        else:
            # 3. If (2) is FALSE, then check whether cross validation will be applied.
            print(f'There is no validation and test sets available.')
            if self.args.num_folds_for_cv < 2:
                print(
                    f'No test set is found and k-fold cross-validation is set to less than 2 (***num_folds_for_cv*** => {self.args.num_folds_for_cv}). Hence we do not evaluate the model')
                # 3.1. NO CROSS VALIDATION => TRAIN WITH 'NegSample' or KvsALL
                if self.scoring_technique == 'NegSample':
                    trained_model = self.training_negative_sampling()
                elif self.scoring_technique == 'KvsAll':
                    # KvsAll or negative sampling
                    trained_model = self.training_kvsall()
                else:
                    raise ValueError(f'Invalid argument: {self.scoring_technique}')
            else:
                trained_model = self.k_fold_cross_validation()
        return trained_model

    def get_batch_1_to_N(self, input_vocab, triples, idx, output_dim):
        batch = triples[idx:idx + self.args.batch_size]
        targets = np.zeros((len(batch), output_dim))
        for idx, pair in enumerate(batch):
            if isinstance(pair,
                          np.ndarray):  # A workaround as test triples in kvold is a numpy array and a numpy array is not hashanle.
                pair = tuple(pair)

            targets[idx, input_vocab[pair]] = 1
        return np.array(batch), torch.FloatTensor(targets)

    def training_kvsall(self):
        """
        Train models with KvsAll or NegativeSampling
        :return:
        """
        # 1. Select model and labelling : Entity Prediction or Relation Prediction.
        model, form_of_labelling = select_model(self.args)
        print(f'KvsAll training starts: {model.name}')  # -labeling:{form_of_labelling}')
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
        # 3. Display the selected model's architecture.
        print(model)
        # 5. Train model
        self.trainer.fit(model, train_dataloaders=dataset.train_dataloader())
        # 6. Test model on validation and test sets if possible.
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

    def training_negative_sampling(self) -> pl.LightningModule:
        """
        Train models with Negative Sampling
        """
        assert self.neg_ratio > 0
        model, _ = select_model(self.args)
        form_of_labelling = 'NegativeSampling'
        print(f' Training starts: {model.name}-labeling:{form_of_labelling}')
        dataset = StandardDataModule(train_set_idx=self.dataset.train_set,
                                     valid_set_idx=self.dataset.valid_set,
                                     test_set_idx=self.dataset.test_set,
                                     entity_to_idx=self.dataset.entity_to_idx,
                                     relation_to_idx=self.dataset.relation_to_idx,
                                     form=form_of_labelling,
                                     neg_sample_ratio=self.neg_ratio,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_processes
                                     )
        print(model)
        self.trainer.fit(model, train_dataloaders=dataset.train_dataloader())
        if self.args.eval:
            if len(self.dataset.valid_set) > 0:
                self.evaluate_lp(model, self.dataset.valid_set, 'Evaluation of Validation set')
            if len(self.dataset.test_set) > 0:
                self.evaluate_lp(model, self.dataset.test_set, 'Evaluation of Test set')
        return model

    def deserialize_index_data(self):
        m = []
        raise NotImplementedError()
        if os.path.isfile(self.args.full_storage_path + '/idx_train_df.gzip'):
            m.append(pd.read_parquet(self.args.full_storage_path + '/idx_train_df.gzip'))
        if os.path.isfile(self.args.full_storage_path + '/idx_valid_df.gzip'):
            m.append(pd.read_parquet(self.storage_path + '/idx_valid_df.gzip'))
        if os.path.isfile(self.storage_path + '/idx_test_df.gzip'):
            m.append(pd.read_parquet(self.storage_path + '/idx_test_df.gzip'))

        return pd.concat(m, ignore_index=True)

    def evaluate_lp_k_vs_all(self, model, triple_idx, info, form_of_labelling):
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
        print(info + ':', end=' ')
        for i in range(10):
            hits.append([])

        # (2) Evaluation mode
        if form_of_labelling == 'RelationPrediction':
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, len(triple_idx), self.args.batch_size):
                # Obtain i.th batch
                data_batch, _ = self.get_batch_1_to_N(self.dataset.ee_vocab, triple_idx, i, self.args.num_relations)
                # From numpy array to torch tensor
                e1_idx, r_idx, e2_idx = torch.tensor(data_batch[:, 0]), torch.tensor(data_batch[:, 1]), torch.tensor(
                    data_batch[:, 2])
                # Generate predictions
                predictions = model.forward_k_vs_all(e1_idx=e1_idx, e2_idx=r_idx)
                # Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    filt = self.dataset.ee_vocab[(data_batch[j][0], data_batch[j][2])]
                    target_value = predictions[j, r_idx[j]].item()
                    predictions[j, filt] = 0.0
                    predictions[j, r_idx[j]] = target_value
                # Sort predictions.
                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                # This can be also done in paralel
                for j in range(data_batch.shape[0]):
                    rank = torch.where(sort_idxs[j] == r_idx[j])[0].item()
                    ranks.append(rank + 1)

                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)

        else:
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, len(triple_idx), self.args.batch_size):
                # Obtain i.th batch
                data_batch, _ = self.get_batch_1_to_N(self.dataset.er_vocab, triple_idx, i, self.args.num_entities)
                # From numpy array to torch tensor
                e1_idx, r_idx, e2_idx = torch.tensor(data_batch[:, 0]), torch.tensor(data_batch[:, 1]), torch.tensor(
                    data_batch[:, 2])
                # Generate predictions
                predictions = model.forward_k_vs_all(e1_idx=e1_idx, rel_idx=r_idx)
                # Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    filt = self.dataset.er_vocab[(data_batch[j][0], data_batch[j][1])]
                    target_value = predictions[j, e2_idx[j]].item()
                    predictions[j, filt] = 0.0
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
        hits = dict()
        reciprocal_ranks = []
        print('LOAD indexed train + valid + test data')
        data = self.deserialize_index_data()

        for i in range(0, len(triple_idx)):
            # 1. Get a triple
            data_point = triple_idx[i]
            s, p, o = data_point[0], data_point[1], data_point[2]

            all_entities = torch.arange(0, self.dataset.num_entities).long()
            all_entities = all_entities.reshape(len(all_entities), )

            # 2. Predict missing heads and tails
            predictions_tails = model.forward_triples(e1_idx=torch.tensor(s).repeat(self.dataset.num_entities, ),
                                                      rel_idx=torch.tensor(p).repeat(self.dataset.num_entities, ),
                                                      e2_idx=all_entities)

            predictions_heads = model.forward_triples(e1_idx=all_entities,
                                                      rel_idx=torch.tensor(p).repeat(self.dataset.num_entities, ),
                                                      e2_idx=torch.tensor(o).repeat(self.dataset.num_entities))

            # 3. Computed filtered ranks.

            # 3.1. Compute filtered tail entity rankings
            # filt_tails = self.dataset.er_vocab[(s, p)]
            filt_tails = data[(data['subject'] == s) & (data['relation'] == p)]['object'].values

            target_value = predictions_tails[o].item()
            predictions_tails[filt_tails] = -np.Inf
            predictions_tails[o] = target_value
            _, sort_idxs = torch.sort(predictions_tails, descending=True)
            # sort_idxs = sort_idxs.cpu().numpy()
            sort_idxs = sort_idxs.detach()  # cpu().numpy()
            filt_tail_entity_rank = np.where(sort_idxs == o)[0][0]

            # 3.1. Compute filtered head entity rankings
            # filt_heads = self.dataset.re_vocab[(p, o)]
            filt_heads = data[(data['relation'] == p) & (data['object'] == o)]['subject'].values

            target_value = predictions_heads[s].item()
            predictions_heads[filt_heads] = -np.Inf
            predictions_heads[s] = target_value
            _, sort_idxs = torch.sort(predictions_heads, descending=True)
            # sort_idxs = sort_idxs.cpu().numpy()
            sort_idxs = sort_idxs.detach()  # cpu().numpy()
            filt_head_entity_rank = np.where(sort_idxs == s)[0][0]

            # 4. Add 1 to ranks as numpy array first item has the index of 0.
            filt_head_entity_rank += 1
            filt_tail_entity_rank += 1
            # 5. Store reciprocal ranks.
            reciprocal_ranks.append(1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank))

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
        print(f'{self.args.num_folds_for_cv}-fold cross-validation starts.')
        kf = KFold(n_splits=self.args.num_folds_for_cv, shuffle=True)
        model = None
        eval_folds = []

        for (ith, (train_index, test_index)) in enumerate(kf.split(self.dataset.train_set)):
            trainer = pl.Trainer.from_argparse_args(self.args)
            model, form_of_labelling = select_model(self.args)
            print(
                f'{ith}-fold cross-validation starts: {model.name}-labeling:{form_of_labelling}, scoring technique: {self.scoring_technique}')

            train_set_for_i_th_fold, test_set_for_i_th_fold = self.dataset.train_set[train_index], \
                                                              self.dataset.train_set[
                                                                  test_index]

            dataset = StandardDataModule(train_set_idx=train_set_for_i_th_fold,
                                         valid_set_idx=self.dataset.valid_set,
                                         test_set_idx=self.dataset.test_set,
                                         entity_to_idx=self.dataset.entity_to_idx,
                                         relation_to_idx=self.dataset.relation_to_idx,
                                         form=form_of_labelling,
                                         neg_sample_ratio=self.neg_ratio,
                                         batch_size=self.args.batch_size,
                                         num_workers=self.args.num_processes
                                         )
            # 5. Train model
            print(model)
            trainer.fit(model, train_dataloaders=dataset.train_dataloader())

            if self.scoring_technique == 'KvsAll' or model.name == 'Shallom':
                res = self.evaluate_lp_k_vs_all(model, test_set_for_i_th_fold,
                                                f'Evaluation at {ith}.th fold via KvsALL', form_of_labelling)
            elif self.scoring_technique == 'NegSample':
                res = self.evaluate_lp(model, test_set_for_i_th_fold, f'Evaluation at {ith}.th fold')
            eval_folds.append([res['MRR'], res['H@1'], res['H@3'], res['H@10']])

        eval_folds = pd.DataFrame(eval_folds, columns=['MRR', 'Hits@1', 'Hits@3', 'Hits@10'])
        # We may want to store it.
        print(eval_folds.describe())

        print('Model trained on last fold will be saved.')
        # Return last model.
        return model
