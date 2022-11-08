import torch
import numpy as np
import json


class Evaluator:
    def __init__(self, executor):
        self.executor = executor
        self.report = dict()

    def eval(self, trained_model, form_of_labelling) -> None:
        """
        Evaluate model with Standard
        :param form_of_labelling:
        :param trained_model:
        :return:
        """
        print('Evaluation Starts.')
        if self.executor.args.eval is None:
            return
        if self.executor.args.num_folds_for_cv > 1:
            # the evaluation must have done in the training part
            return
        if isinstance(self.executor.args.eval, bool):
            print('Wrong input:RESET')
            self.executor.args.eval = 'train_val_test'

        if self.executor.args.scoring_technique == 'NegSample':
            self.eval_rank_of_head_and_tail_entity(trained_model)
        elif self.executor.args.scoring_technique in ['KvsAll', 'KvsSample', '1vsAll', 'PvsAll', 'CCvsAll']:
            self.eval_with_vs_all(trained_model, form_of_labelling)
        elif self.executor.args.scoring_technique in ['BatchRelaxedKvsAll', 'BatchRelaxed1vsAll']:
            self.eval_with_vs_all(trained_model, form_of_labelling)
        else:
            raise ValueError(f'Invalid argument: {self.executor.args.scoring_technique}')
        with open(self.executor.args.full_storage_path + '/eval_report.json', 'w') as file_descriptor:
            json.dump(self.report, file_descriptor, indent=4)
        print('Evaluation Ends.')

    def eval_rank_of_head_and_tail_entity(self, trained_model):
        # 4. Test model on the training dataset if it is needed.
        if 'train' in self.executor.args.eval:
            res = self.evaluate_lp(trained_model, self.executor.dataset.train_set,
                                   f'Evaluate {trained_model.name} on Train set')
            self.report['Train'] = res
        # 5. Test model on the validation and test dataset if it is needed.
        if 'val' in self.executor.args.eval:
            if self.executor.dataset.valid_set is not None:
                self.report['Val'] = self.evaluate_lp(trained_model, self.executor.dataset.valid_set,
                                                      f'Evaluate {trained_model.name} of Validation set')

        if self.executor.dataset.test_set is not None and 'test' in self.executor.args.eval:
            self.report['Test'] = self.evaluate_lp(trained_model, self.executor.dataset.test_set,
                                                   f'Evaluate {trained_model.name} of Test set')

    def eval_with_vs_all(self, trained_model, form_of_labelling) -> None:
        """ Evaluate model after reciprocal triples are added """
        # 4. Test model on the training dataset if it is needed.
        if 'train' in self.executor.args.eval:
            res = self.evaluate_lp_k_vs_all(trained_model, self.executor.dataset.train_set,
                                            info=f'Evaluate {trained_model.name} on Train set',
                                            form_of_labelling=form_of_labelling)
            self.report['Train'] = res

        # 5. Test model on the validation and test dataset if it is needed.
        if 'val' in self.executor.args.eval:
            if self.executor.dataset.valid_set is not None:
                res = self.evaluate_lp_k_vs_all(trained_model, self.executor.dataset.valid_set,
                                                f'Evaluate {trained_model.name} on Validation set',
                                                form_of_labelling=form_of_labelling)
                self.report['Val'] = res
        if self.executor.dataset.test_set is not None and 'test' in self.executor.args.eval:
            res = self.evaluate_lp_k_vs_all(trained_model, self.executor.dataset.test_set,
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
        hits = []
        ranks = []
        if info:
            print(info + ':', end=' ')
        for i in range(10):
            hits.append([])

        # (2) Evaluation mode
        if form_of_labelling == 'RelationPrediction':
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, len(triple_idx), self.executor.args.batch_size):
                data_batch = triple_idx[i:i + self.executor.args.batch_size]
                e1_idx_e2_idx, r_idx = torch.LongTensor(data_batch[:, [0, 2]]), torch.LongTensor(data_batch[:, 1])
                # Generate predictions
                predictions = model.forward_k_vs_all(x=e1_idx_e2_idx)
                # Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    filt = self.executor.dataset.ee_vocab[(data_batch[j][0], data_batch[j][2])]
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
            # TODO: Why do not we use Pytorch Dataset ? for multiprocessing
            # Iterate over integer indexed triples in mini batch fashion
            for i in range(0, len(triple_idx), self.executor.args.batch_size):
                data_batch = triple_idx[i:i + self.executor.args.batch_size]
                e1_idx_r_idx, e2_idx = torch.LongTensor(data_batch[:, [0, 1]]), torch.tensor(data_batch[:, 2])
                with torch.no_grad():
                    predictions = model(e1_idx_r_idx)
                # Filter entities except the target entity
                for j in range(data_batch.shape[0]):
                    filt = self.executor.dataset.er_vocab[(data_batch[j][0], data_batch[j][1])]
                    target_value = predictions[j, e2_idx[j]].item()
                    predictions[j, filt] = -np.Inf
                    # 3.3.1 Filter entities outside of the range
                    # TODO: Fix it CV resets from str to boolean
                    if 'constraint' in self.executor.args.eval:
                        predictions[j, self.executor.dataset.range_constraints_per_rel[data_batch[j, 1]]] = -np.Inf
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
        all_entities = torch.arange(0, self.executor.dataset.num_entities).long()
        all_entities = all_entities.reshape(len(all_entities), )
        # Iterating one by one is not good when you are using batch norm
        for i in range(0, len(triple_idx)):
            # 1. Get a triple
            data_point = triple_idx[i]
            s, p, o = data_point[0], data_point[1], data_point[2]

            # 2. Predict missing heads and tails
            x = torch.stack((torch.tensor(s).repeat(self.executor.dataset.num_entities, ),
                             torch.tensor(p).repeat(self.executor.dataset.num_entities, ),
                             all_entities
                             ), dim=1)
            predictions_tails = model.forward_triples(x)
            x = torch.stack((all_entities,
                             torch.tensor(p).repeat(self.executor.dataset.num_entities, ),
                             torch.tensor(o).repeat(self.executor.dataset.num_entities)
                             ), dim=1)

            predictions_heads = model.forward_triples(x)
            del x

            # 3. Computed filtered ranks for missing tail entities.
            # 3.1. Compute filtered tail entity rankings
            filt_tails = self.executor.dataset.er_vocab[(s, p)]
            # 3.2 Get the predicted target's score
            target_value = predictions_tails[o].item()
            # 3.3 Filter scores of all triples containing filtered tail entities
            predictions_tails[filt_tails] = -np.Inf
            # 3.3.1 Filter entities outside of the range
            if 'constraint' in self.executor.args.eval:
                predictions_tails[self.executor.dataset.range_constraints_per_rel[p]] = -np.Inf
            # 3.4 Reset the target's score
            predictions_tails[o] = target_value
            # 3.5. Sort the score
            _, sort_idxs = torch.sort(predictions_tails, descending=True)
            # sort_idxs = sort_idxs.cpu().numpy()
            sort_idxs = sort_idxs.detach()  # cpu().numpy()
            filt_tail_entity_rank = np.where(sort_idxs == o)[0][0]

            # 4. Computed filtered ranks for missing head entities.
            # 4.1. Retrieve head entities to be filtered
            filt_heads = self.executor.dataset.re_vocab[(p, o)]
            # filt_heads = data[(data['relation'] == p) & (data['object'] == o)]['subject'].values
            # 4.2 Get the predicted target's score
            target_value = predictions_heads[s].item()
            # 4.3 Filter scores of all triples containing filtered head entities.
            predictions_heads[filt_heads] = -np.Inf
            if isinstance(self.executor.args.eval, bool) is False:
                if 'constraint' in self.executor.args.eval:
                    # 4.3.1 Filter entities that are outside the domain
                    predictions_heads[self.executor.dataset.domain_constraints_per_rel[p]] = -np.Inf
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

    def eval_with_data(self, trained_model, triple_idx: np.ndarray, form_of_labelling: str):
        """ Evaluate a trained model on a given a dataset"""
        if self.executor.args.scoring_technique == 'NegSample':
            return self.evaluate_lp(trained_model, triple_idx,
                                    info=f'Evaluate {trained_model.name} on a given dataset', )

        elif self.executor.args.scoring_technique in ['KvsAll', 'KvsSample', '1vsAll', 'PvsAll', 'CCvsAll']:
            return self.evaluate_lp_k_vs_all(trained_model, triple_idx,
                                             info=f'Evaluate {trained_model.name} on a given dataset',
                                             form_of_labelling=form_of_labelling)

        elif self.executor.args.scoring_technique in ['BatchRelaxedKvsAll', 'BatchRelaxed1vsAll']:
            return self.evaluate_lp_k_vs_all(trained_model, triple_idx,
                                             info=f'Evaluate {trained_model.name} on a given dataset',
                                             form_of_labelling=form_of_labelling)
        else:
            raise ValueError(f'Invalid argument: {self.executor.args.scoring_technique}')
