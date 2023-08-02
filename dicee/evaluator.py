import torch
import numpy as np
import json
from .static_funcs import pickle
from .static_funcs_training import evaluate_lp


class Evaluator:
    """
        Evaluator class to evaluate KGE models in various downstream tasks

        Arguments
       ----------
       executor: Executor class instance
   """

    def __init__(self, args, is_continual_training=None):
        self.re_vocab = None
        self.er_vocab = None
        self.ee_vocab = None
        self.is_continual_training = is_continual_training
        self.num_entities = None
        self.num_relations = None
        self.domain_constraints_per_rel, self.range_constraints_per_rel = None, None
        self.args = args
        self.report = dict()
        self.during_training = False

    def vocab_preparation(self, dataset) -> None:
        """
        A function to wait future objects for the attributes of executor

        Arguments
        ----------

        Return
        ----------
        None
        """
        # print("** VOCAB Prep **")
        if isinstance(dataset.er_vocab, dict):
            self.er_vocab = dataset.er_vocab
        else:
            self.er_vocab = dataset.er_vocab.result()

        if isinstance(dataset.re_vocab, dict):
            self.re_vocab = dataset.re_vocab
        else:
            self.re_vocab = dataset.re_vocab.result()

        if isinstance(dataset.ee_vocab, dict):
            self.ee_vocab = dataset.ee_vocab.result()
        else:
            self.ee_vocab = dataset.ee_vocab.result()

        if isinstance(dataset.constraints, tuple):
            self.domain_constraints_per_rel, self.range_constraints_per_rel = dataset.constraints
        else:
            try:
                self.domain_constraints_per_rel, self.range_constraints_per_rel = dataset.constraints.result()
            except RuntimeError:
                print('Domain constraint exception occurred')

        self.num_entities = dataset.num_entities
        self.num_relations = dataset.num_relations

        pickle.dump(self.er_vocab, open(self.args.full_storage_path + "/er_vocab.p", "wb"))
        pickle.dump(self.re_vocab, open(self.args.full_storage_path + "/re_vocab.p", "wb"))
        pickle.dump(self.ee_vocab, open(self.args.full_storage_path + "/ee_vocab.p", "wb"))

    # @timeit
    def eval(self, dataset, trained_model, form_of_labelling, during_training=False) -> None:
        self.during_training = during_training
        # (1) Exit, if the flag is not set
        if self.args.eval_model is None:
            return
        self.vocab_preparation(dataset)
        if self.args.num_folds_for_cv > 1:
            return
        if isinstance(self.args.eval_model, bool):
            print('Wrong input:RESET')
            self.args.eval_model = 'train_val_test'

        if self.args.scoring_technique == 'NegSample':
            self.eval_rank_of_head_and_tail_entity(train_set=dataset.train_set,
                                                   valid_set=dataset.valid_set,
                                                   test_set=dataset.test_set,
                                                   trained_model=trained_model)
        elif self.args.scoring_technique in ['KvsAll', 'KvsSample', '1vsAll', 'PvsAll', 'CCvsAll']:
            self.eval_with_vs_all(train_set=dataset.train_set,
                                  valid_set=dataset.valid_set,
                                  test_set=dataset.test_set,
                                  trained_model=trained_model,
                                  form_of_labelling=form_of_labelling)
        else:
            raise ValueError(f'Invalid argument: {self.args.scoring_technique}')
        if self.during_training is False:
            with open(self.args.full_storage_path + '/eval_report.json', 'w') as file_descriptor:
                json.dump(self.report, file_descriptor, indent=4)
        return {k: v for k, v in self.report.items()}

    def dummy_eval(self, trained_model, form_of_labelling):

        if self.is_continual_training:
            self.er_vocab = pickle.load(open(self.args.full_storage_path + "/er_vocab.p", "rb"))
            self.re_vocab = pickle.load(open(self.args.full_storage_path + "/re_vocab.p", "rb"))
            self.ee_vocab = pickle.load(open(self.args.full_storage_path + "/ee_vocab.p", "rb"))

        if 'train' in self.args.eval_model:
            train_set = np.load(self.args.full_storage_path + "/train_set.npy")
        else:
            train_set = None
        if 'val' in self.args.eval_model:
            valid_set = np.load(self.args.full_storage_path + "/valid_set.npy")
        else:
            valid_set = None

        if 'test' in self.args.eval_model:
            test_set = np.load(self.args.full_storage_path + "/test_set.npy")
        else:
            test_set = None

        if self.args.scoring_technique == 'NegSample':
            self.eval_rank_of_head_and_tail_entity(train_set=train_set,
                                                   valid_set=valid_set,
                                                   test_set=test_set,
                                                   trained_model=trained_model)
        elif self.args.scoring_technique in ['KvsAll', 'KvsSample', '1vsAll', 'PvsAll', 'CCvsAll']:
            self.eval_with_vs_all(train_set=train_set,
                                  valid_set=valid_set,
                                  test_set=test_set,
                                  trained_model=trained_model, form_of_labelling=form_of_labelling)
        else:
            raise ValueError(f'Invalid argument: {self.args.scoring_technique}')
        with open(self.args.full_storage_path + '/eval_report.json', 'w') as file_descriptor:
            json.dump(self.report, file_descriptor, indent=4)

    def eval_rank_of_head_and_tail_entity(self, *, train_set, valid_set=None, test_set=None, trained_model):
        # 4. Test model on the training dataset if it is needed.
        if 'train' in self.args.eval_model:
            res = self.evaluate_lp(trained_model, train_set,
                                   f'Evaluate {trained_model.name} on Train set')
            self.report['Train'] = res
        # 5. Test model on the validation and test dataset if it is needed.
        if 'val' in self.args.eval_model:
            if valid_set is not None:
                self.report['Val'] = self.evaluate_lp(trained_model, valid_set,
                                                      f'Evaluate {trained_model.name} of Validation set')

        if test_set is not None and 'test' in self.args.eval_model:
            self.report['Test'] = self.evaluate_lp(trained_model, test_set,
                                                   f'Evaluate {trained_model.name} of Test set')

    def eval_with_vs_all(self, *, train_set, valid_set=None, test_set=None, trained_model, form_of_labelling) -> None:
        """ Evaluate model after reciprocal triples are added """
        # 4. Test model on the training dataset if it is needed.
        if 'train' in self.args.eval_model:
            res = self.evaluate_lp_k_vs_all(trained_model, train_set,
                                            info=f'Evaluate {trained_model.name} on Train set',
                                            form_of_labelling=form_of_labelling)
            self.report['Train'] = res

        # 5. Test model on the validation and test dataset if it is needed.
        if 'val' in self.args.eval_model:
            if valid_set is not None:
                res = self.evaluate_lp_k_vs_all(trained_model, valid_set,
                                                f'Evaluate {trained_model.name} on Validation set',
                                                form_of_labelling=form_of_labelling)
                self.report['Val'] = res
        if test_set is not None and 'test' in self.args.eval_model:
            res = self.evaluate_lp_k_vs_all(trained_model, test_set,
                                            f'Evaluate {trained_model.name} on Test set',
                                            form_of_labelling=form_of_labelling)
            self.report['Test'] = res

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
        num_triples = len(triple_idx)
        ranks = []
        # Hit range
        hits_range = [i for i in range(1, 11)]
        hits = {i: [] for i in hits_range}
        if info and self.during_training is False:
            print(info + ':', end=' ')
        if form_of_labelling == 'RelationPrediction':
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, num_triples, self.args.batch_size):
                data_batch = triple_idx[i:i + self.args.batch_size]
                e1_idx_e2_idx, r_idx = torch.LongTensor(data_batch[:, [0, 2]]), torch.LongTensor(data_batch[:, 1])
                # Generate predictions
                predictions = model.forward_k_vs_all(x=e1_idx_e2_idx)
                # Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    filt = self.ee_vocab[(data_batch[j][0], data_batch[j][2])]
                    target_value = predictions[j, r_idx[j]].item()
                    predictions[j, filt] = -np.Inf
                    predictions[j, r_idx[j]] = target_value
                # Sort predictions.
                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                # This can be also done in parallel
                for j in range(data_batch.shape[0]):
                    rank = torch.where(sort_idxs[j] == r_idx[j])[0].item() + 1
                    ranks.append(rank)
                    for hits_level in hits_range:
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
        else:
            # TODO: Why do not we use Pytorch Dataset ? for multiprocessing
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, num_triples, self.args.batch_size):
                # (1) Get a batch of data.
                data_batch = triple_idx[i:i + self.args.batch_size]
                # (2) Extract entities and relations.
                e1_idx_r_idx, e2_idx = torch.LongTensor(data_batch[:, [0, 1]]), torch.tensor(data_batch[:, 2])
                # (3) Predict missing entities, i.e., assign probs to all entities.
                with torch.no_grad():
                    predictions = model(e1_idx_r_idx)
                # (4) Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    # (4.1) Get the ids of the head entity, the relation and the target tail entity in the j.th triple.
                    id_e, id_r, id_e_target = data_batch[j]
                    # (4.2) Get all ids of all entities occurring with the head entity and relation extracted in 4.1.
                    filt = self.er_vocab[(id_e, id_r)]
                    # (4.3) Store the assigned score of the target tail entity extracted in 4.1.
                    target_value = predictions[j, id_e_target].item()
                    # (4.4.1) Filter all assigned scores for entities.
                    predictions[j, filt] = -np.Inf
                    # (4.4.2) Filter entities based on the range of a relation as well.
                    if 'constraint' in self.args.eval_model:
                        predictions[j, self.range_constraints_per_rel[data_batch[j, 1]]] = -np.Inf
                    # (4.5) Insert 4.3. after filtering.
                    predictions[j, id_e_target] = target_value
                # (5) Sort predictions.
                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
                # (6) Compute the filtered ranks.
                for j in range(data_batch.shape[0]):
                    # index between 0 and \inf
                    rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
                    ranks.append(rank)
                    for hits_level in hits_range:
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
        # (7) Sanity checking: a rank for a triple
        assert len(triple_idx) == len(ranks) == num_triples
        hit_1 = sum(hits[1]) / num_triples
        hit_3 = sum(hits[3]) / num_triples
        hit_10 = sum(hits[10]) / num_triples
        mean_reciprocal_rank = np.mean(1. / np.array(ranks))

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
        if info and self.during_training is False:
            print(info)
            print(results)
        return results

    def evaluate_lp(self, model, triple_idx, info):
        """

        """
        # @TODO: Document this method
        return evaluate_lp(model, triple_idx, num_entities=self.num_entities,
                           er_vocab=self.er_vocab,re_vocab=self.re_vocab,info=info)

    def dept_evaluate_lp(self, model, triple_idx, info):
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
        print('** Evaluation without batching')
        hits = dict()
        reciprocal_ranks = []
        # Iterate over test triples
        all_entities = torch.arange(0, self.num_entities).long()
        all_entities = all_entities.reshape(len(all_entities), )
        # Iterating one by one is not good when you are using batch norm
        for i in range(0, len(triple_idx)):
            # (1) Get a triple (head entity, relation, tail entity
            data_point = triple_idx[i]
            h, r, t = data_point[0], data_point[1], data_point[2]

            # (2) Predict missing heads and tails
            x = torch.stack((torch.tensor(h).repeat(self.num_entities, ),
                             torch.tensor(r).repeat(self.num_entities, ),
                             all_entities), dim=1)

            predictions_tails = model.forward_triples(x)
            x = torch.stack((all_entities,
                             torch.tensor(r).repeat(self.num_entities, ),
                             torch.tensor(t).repeat(self.num_entities)
                             ), dim=1)

            predictions_heads = model.forward_triples(x)
            del x

            # 3. Computed filtered ranks for missing tail entities.
            # 3.1. Compute filtered tail entity rankings
            filt_tails = self.er_vocab[(h, r)]
            # 3.2 Get the predicted target's score
            target_value = predictions_tails[t].item()
            # 3.3 Filter scores of all triples containing filtered tail entities
            predictions_tails[filt_tails] = -np.Inf
            # 3.3.1 Filter entities outside of the range
            if 'constraint' in self.args.eval_model:
                predictions_tails[self.range_constraints_per_rel[r]] = -np.Inf
            # 3.4 Reset the target's score
            predictions_tails[t] = target_value
            # 3.5. Sort the score
            _, sort_idxs = torch.sort(predictions_tails, descending=True)
            sort_idxs = sort_idxs.detach()
            filt_tail_entity_rank = np.where(sort_idxs == t)[0][0]

            # 4. Computed filtered ranks for missing head entities.
            # 4.1. Retrieve head entities to be filtered
            filt_heads = self.re_vocab[(r, t)]
            # 4.2 Get the predicted target's score
            target_value = predictions_heads[h].item()
            # 4.3 Filter scores of all triples containing filtered head entities.
            predictions_heads[filt_heads] = -np.Inf
            if isinstance(self.args.eval_model, bool) is False:
                if 'constraint' in self.args.eval_model:
                    # 4.3.1 Filter entities that are outside the domain
                    predictions_heads[self.domain_constraints_per_rel[r]] = -np.Inf
            predictions_heads[h] = target_value
            _, sort_idxs = torch.sort(predictions_heads, descending=True)
            sort_idxs = sort_idxs.detach()
            filt_head_entity_rank = np.where(sort_idxs == h)[0][0]

            # 4. Add 1 to ranks as numpy array first item has the index of 0.
            filt_head_entity_rank += 1
            filt_tail_entity_rank += 1

            rr = 1.0 / filt_head_entity_rank + (1.0 / filt_tail_entity_rank)
            # 5. Store reciprocal ranks.
            reciprocal_ranks.append(rr)
            # print(f'{i}.th triple: mean reciprical rank:{rr}')

            # 4. Compute Hit@N
            for hits_level in range(1, 11):
                res = 1 if filt_head_entity_rank <= hits_level else 0
                res += 1 if filt_tail_entity_rank <= hits_level else 0
                if res > 0:
                    hits.setdefault(hits_level, []).append(res)

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

    def eval_with_data(self, dataset, trained_model, triple_idx: np.ndarray, form_of_labelling: str):
        self.vocab_preparation(dataset)

        """ Evaluate a trained model on a given a dataset"""
        if self.args.scoring_technique == 'NegSample':
            return self.evaluate_lp(trained_model, triple_idx,
                                    info=f'Evaluate {trained_model.name} on a given dataset', )

        elif self.args.scoring_technique in ['KvsAll', 'KvsSample', '1vsAll', 'PvsAll', 'CCvsAll']:
            return self.evaluate_lp_k_vs_all(trained_model, triple_idx,
                                             info=f'Evaluate {trained_model.name} on a given dataset',
                                             form_of_labelling=form_of_labelling)

        elif self.args.scoring_technique in ['BatchRelaxedKvsAll', 'BatchRelaxed1vsAll']:
            return self.evaluate_lp_k_vs_all(trained_model, triple_idx,
                                             info=f'Evaluate {trained_model.name} on a given dataset',
                                             form_of_labelling=form_of_labelling)
        else:
            raise ValueError(f'Invalid argument: {self.args.scoring_technique}')
