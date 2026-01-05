"""Main Evaluator class for KGE model evaluation.

This module provides the Evaluator class which orchestrates evaluation
of knowledge graph embedding models across different datasets and
scoring techniques.
"""

import json
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .link_prediction import evaluate_lp, evaluate_bpe_lp
from .utils import (
    compute_metrics_from_ranks_simple,
    update_hits,
    create_hits_dict,
    ALL_HITS_RANGE,
)

# Valid scoring techniques
VALID_SCORING_TECHNIQUES = frozenset([
    "AllvsAll", "KvsAll", "1vsSample", "KvsSample", "1vsAll", "NegSample",
    "BatchRelaxedKvsAll", "BatchRelaxed1vsAll", "PvsAll", "CCvsAll"
])


class Evaluator:
    """Evaluator class for KGE models in various downstream tasks.

    Orchestrates link prediction evaluation with different scoring techniques
    including standard evaluation and byte-pair encoding based evaluation.

    Attributes:
        er_vocab: Entity-relation to tail vocabulary for filtered ranking.
        re_vocab: Relation-entity (tail) to head vocabulary.
        ee_vocab: Entity-entity to relation vocabulary.
        num_entities: Total number of entities in the knowledge graph.
        num_relations: Total number of relations in the knowledge graph.
        args: Configuration arguments.
        report: Dictionary storing evaluation results.
        during_training: Whether evaluation is happening during training.

    Example:
        >>> from dicee.evaluation import Evaluator
        >>> evaluator = Evaluator(args)
        >>> results = evaluator.eval(dataset, model, 'EntityPrediction')
        >>> print(f"Test MRR: {results['Test']['MRR']:.4f}")
    """

    def __init__(self, args, is_continual_training: bool = False):
        """Initialize the evaluator.

        Args:
            args: Configuration arguments containing evaluation settings.
            is_continual_training: Whether this is continual training mode.
        """
        self.re_vocab: Optional[Dict] = None
        self.er_vocab: Optional[Dict] = None
        self.ee_vocab: Optional[Dict] = None
        self.func_triple_to_bpe_representation = None
        self.is_continual_training = is_continual_training
        self.num_entities: Optional[int] = None
        self.num_relations: Optional[int] = None
        self.domain_constraints_per_rel = None
        self.range_constraints_per_rel = None
        self.args = args
        self.report: Dict = {}
        self.during_training = False

    def vocab_preparation(self, dataset) -> None:
        """Prepare vocabularies from the dataset for evaluation.

        Resolves any future objects and saves vocabularies to disk.

        Args:
            dataset: Knowledge graph dataset with vocabulary attributes.
        """
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

        self.num_entities = dataset.num_entities
        self.num_relations = dataset.num_relations
        self.func_triple_to_bpe_representation = dataset.func_triple_to_bpe_representation

        # Save vocabularies to disk
        er_vocab_path = os.path.join(self.args.full_storage_path, "er_vocab.p")
        re_vocab_path = os.path.join(self.args.full_storage_path, "re_vocab.p")
        ee_vocab_path = os.path.join(self.args.full_storage_path, "ee_vocab.p")

        with open(er_vocab_path, "wb") as f:
            pickle.dump(self.er_vocab, f)
        with open(re_vocab_path, "wb") as f:
            pickle.dump(self.re_vocab, f)
        with open(ee_vocab_path, 'wb') as f:
            pickle.dump(self.ee_vocab, f)

    def eval(
        self,
        dataset,
        trained_model,
        form_of_labelling: str,
        during_training: bool = False
    ) -> Optional[Dict]:
        """Evaluate the trained model on the dataset.

        Args:
            dataset: Knowledge graph dataset (KG instance).
            trained_model: The trained KGE model.
            form_of_labelling: Type of labelling ('EntityPrediction' or 'RelationPrediction').
            during_training: Whether evaluation is during training.

        Returns:
            Dictionary of evaluation metrics, or None if evaluation is skipped.
        """
        self.during_training = during_training

        # Exit if evaluation is not requested
        if self.args.eval_model is None:
            return None

        self.vocab_preparation(dataset)

        if self.args.num_folds_for_cv > 1:
            return None

        if isinstance(self.args.eval_model, bool):
            print('Wrong input:RESET')
            self.args.eval_model = 'train_val_test'

        # Route to appropriate evaluation method based on scoring technique
        if self.args.scoring_technique == 'NegSample' and self.args.byte_pair_encoding:
            self.eval_rank_of_head_and_tail_byte_pair_encoded_entity(
                train_set=dataset.train_set,
                valid_set=dataset.valid_set,
                test_set=dataset.test_set,
                ordered_bpe_entities=dataset.ordered_bpe_entities,
                trained_model=trained_model
            )
        elif self.args.scoring_technique in ["AllvsAll", "KvsAll", '1vsSample', "1vsAll"] \
                and self.args.byte_pair_encoding:
            if self.args.model != "BytE":
                self.eval_with_bpe_vs_all(
                    raw_train_set=dataset.raw_train_set,
                    raw_valid_set=dataset.raw_valid_set,
                    raw_test_set=dataset.raw_test_set,
                    trained_model=trained_model,
                    form_of_labelling=form_of_labelling
                )
            else:
                self.eval_with_byte(
                    raw_train_set=dataset.raw_train_set,
                    raw_valid_set=dataset.raw_valid_set,
                    raw_test_set=dataset.raw_test_set,
                    trained_model=trained_model,
                    form_of_labelling=form_of_labelling
                )
        elif self.args.scoring_technique == 'NegSample':
            self.eval_rank_of_head_and_tail_entity(
                train_set=dataset.train_set,
                valid_set=dataset.valid_set,
                test_set=dataset.test_set,
                trained_model=trained_model
            )
        elif self.args.scoring_technique in ["AllvsAll", "KvsAll", '1vsSample', "KvsSample", "1vsAll"]:
            self.eval_with_vs_all(
                train_set=dataset.train_set,
                valid_set=dataset.valid_set,
                test_set=dataset.test_set,
                trained_model=trained_model,
                form_of_labelling=form_of_labelling
            )
        else:
            raise ValueError(f'Invalid scoring technique: {self.args.scoring_technique}')

        # Save evaluation report
        if not self.during_training:
            report_path = os.path.join(self.args.full_storage_path, 'eval_report.json')
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=4)

        return dict(self.report)

    def _load_indexed_datasets(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load indexed datasets based on evaluation settings.

        Returns:
            Tuple of (train_set, valid_set, test_set) arrays, with None for
            datasets not requested in eval_model.
        """
        train_set = None
        valid_set = None
        test_set = None

        if 'train' in self.args.eval_model:
            train_path = os.path.join(self.args.full_storage_path, "train_set.npy")
            train_set = np.load(train_path)

        if 'val' in self.args.eval_model:
            valid_path = os.path.join(self.args.full_storage_path, "valid_set.npy")
            valid_set = np.load(valid_path)

        if 'test' in self.args.eval_model:
            test_path = os.path.join(self.args.full_storage_path, "test_set.npy")
            test_set = np.load(test_path)

        return train_set, valid_set, test_set

    def _load_and_set_mappings(self) -> None:
        """Load vocabularies and mappings from disk."""
        # Import here to avoid circular imports
        from ..static_funcs import load_json

        er_vocab_path = os.path.join(self.args.full_storage_path, "er_vocab.p")
        re_vocab_path = os.path.join(self.args.full_storage_path, "re_vocab.p")
        ee_vocab_path = os.path.join(self.args.full_storage_path, "ee_vocab.p")
        report_path = os.path.join(self.args.full_storage_path, "report.json")

        with open(er_vocab_path, "rb") as f:
            self.er_vocab = pickle.load(f)
        with open(re_vocab_path, "rb") as f:
            self.re_vocab = pickle.load(f)
        with open(ee_vocab_path, "rb") as f:
            self.ee_vocab = pickle.load(f)

        report = load_json(report_path)
        self.num_entities = report["num_entities"]
        self.num_relations = report["num_relations"]

    def eval_rank_of_head_and_tail_entity(
        self,
        *,
        train_set,
        valid_set=None,
        test_set=None,
        trained_model
    ) -> None:
        """Evaluate with negative sampling scoring."""
        if 'train' in self.args.eval_model:
            res = self.evaluate_lp(
                trained_model, train_set,
                f'Evaluate {trained_model.name} on Train set'
            )
            self.report['Train'] = res

        if 'val' in self.args.eval_model and valid_set is not None:
            self.report['Val'] = self.evaluate_lp(
                trained_model, valid_set,
                f'Evaluate {trained_model.name} of Validation set'
            )

        if test_set is not None and 'test' in self.args.eval_model:
            self.report['Test'] = self.evaluate_lp(
                trained_model, test_set,
                f'Evaluate {trained_model.name} of Test set'
            )

    def eval_rank_of_head_and_tail_byte_pair_encoded_entity(
        self,
        *,
        train_set=None,
        valid_set=None,
        test_set=None,
        ordered_bpe_entities,
        trained_model
    ) -> None:
        """Evaluate with BPE-encoded entities and negative sampling."""
        if 'train' in self.args.eval_model:
            self.report['Train'] = evaluate_bpe_lp(
                trained_model, train_set, ordered_bpe_entities,
                er_vocab=self.er_vocab, re_vocab=self.re_vocab,
                info=f'Evaluate {trained_model.name} on NegSample BPE Train set'
            )

        if 'val' in self.args.eval_model and valid_set is not None:
            self.report['Val'] = evaluate_bpe_lp(
                trained_model, valid_set, ordered_bpe_entities,
                er_vocab=self.er_vocab, re_vocab=self.re_vocab,
                info=f'Evaluate {trained_model.name} on NegSample BPE Valid set'
            )

        if test_set is not None and 'test' in self.args.eval_model:
            self.report['Test'] = evaluate_bpe_lp(
                trained_model, test_set, ordered_bpe_entities,
                er_vocab=self.er_vocab, re_vocab=self.re_vocab,
                info=f'Evaluate {trained_model.name} on NegSample BPE Test set'
            )

    def eval_with_byte(
        self,
        *,
        raw_train_set,
        raw_valid_set=None,
        raw_test_set=None,
        trained_model,
        form_of_labelling
    ) -> None:
        """Evaluate BytE model with generation."""
        if 'train' in self.args.eval_model:
            self.report['Train'] = -1

        if 'val' in self.args.eval_model and raw_valid_set is not None:
            assert isinstance(raw_valid_set, pd.DataFrame)
            self.report['Val'] = self.evaluate_lp_with_byte(
                trained_model, raw_valid_set.values.tolist(),
                f'Evaluate {trained_model.name} on BytE Validation set'
            )

        if raw_test_set is not None and 'test' in self.args.eval_model:
            self.report['Test'] = self.evaluate_lp_with_byte(
                trained_model, raw_test_set.values.tolist(),
                f'Evaluate {trained_model.name} on BytE Test set'
            )

    def eval_with_bpe_vs_all(
        self,
        *,
        raw_train_set,
        raw_valid_set=None,
        raw_test_set=None,
        trained_model,
        form_of_labelling
    ) -> None:
        """Evaluate with BPE and KvsAll scoring."""
        if 'train' in self.args.eval_model:
            self.report['Train'] = self.evaluate_lp_bpe_k_vs_all(
                trained_model, raw_train_set.values.tolist(),
                f'Evaluate {trained_model.name} on BPE Train set',
                form_of_labelling=form_of_labelling
            )

        if 'val' in self.args.eval_model and raw_valid_set is not None:
            assert isinstance(raw_valid_set, pd.DataFrame)
            self.report['Val'] = self.evaluate_lp_bpe_k_vs_all(
                trained_model, raw_valid_set.values.tolist(),
                f'Evaluate {trained_model.name} on BPE Validation set',
                form_of_labelling=form_of_labelling
            )

        if raw_test_set is not None and 'test' in self.args.eval_model:
            self.report['Test'] = self.evaluate_lp_bpe_k_vs_all(
                trained_model, raw_test_set.values.tolist(),
                f'Evaluate {trained_model.name} on BPE Test set',
                form_of_labelling=form_of_labelling
            )

    def eval_with_vs_all(
        self,
        *,
        train_set,
        valid_set=None,
        test_set=None,
        trained_model,
        form_of_labelling
    ) -> None:
        """Evaluate with KvsAll or 1vsAll scoring."""
        if 'train' in self.args.eval_model:
            self.report['Train'] = self.evaluate_lp_k_vs_all(
                trained_model, train_set,
                info=f'Evaluate {trained_model.name} on Train set',
                form_of_labelling=form_of_labelling
            )

        if 'val' in self.args.eval_model and valid_set is not None:
            self.report['Val'] = self.evaluate_lp_k_vs_all(
                trained_model, valid_set,
                f'Evaluate {trained_model.name} on Validation set',
                form_of_labelling=form_of_labelling
            )

        if test_set is not None and 'test' in self.args.eval_model:
            self.report['Test'] = self.evaluate_lp_k_vs_all(
                trained_model, test_set,
                f'Evaluate {trained_model.name} on Test set',
                form_of_labelling=form_of_labelling
            )

    def evaluate_lp_k_vs_all(
        self,
        model,
        triple_idx,
        info: Optional[str] = None,
        form_of_labelling: Optional[str] = None
    ) -> Dict[str, float]:
        """Filtered link prediction evaluation with KvsAll scoring.

        Args:
            model: The trained model to evaluate.
            triple_idx: Integer-indexed test triples.
            info: Description to print.
            form_of_labelling: 'EntityPrediction' or 'RelationPrediction'.

        Returns:
            Dictionary with H@1, H@3, H@10, and MRR metrics.
        """
        model.eval()
        num_triples = len(triple_idx)
        ranks: List[int] = []
        hits_range = ALL_HITS_RANGE
        hits = create_hits_dict(hits_range)

        if info and not self.during_training:
            print(info + ':', end=' ')

        if form_of_labelling == 'RelationPrediction':
            ranks, hits = self._evaluate_relation_prediction(
                model, triple_idx, num_triples, hits_range
            )
        else:
            ranks, hits = self._evaluate_entity_prediction(
                model, triple_idx, num_triples, hits_range
            )

        assert len(triple_idx) == len(ranks) == num_triples

        results = compute_metrics_from_ranks_simple(ranks, num_triples, hits)

        if info and not self.during_training:
            print(info)
            print(results)

        return results

    def _evaluate_relation_prediction(
        self,
        model,
        triple_idx,
        num_triples: int,
        hits_range: List[int]
    ) -> Tuple[List[int], Dict[int, List[float]]]:
        """Evaluate relation prediction task."""
        ranks: List[int] = []
        hits = create_hits_dict(hits_range)

        for i in range(0, num_triples, self.args.batch_size):
            data_batch = triple_idx[i:i + self.args.batch_size]
            e1_idx_e2_idx = torch.LongTensor(data_batch[:, [0, 2]])
            r_idx = torch.LongTensor(data_batch[:, 1])

            predictions = model.forward_k_vs_all(x=e1_idx_e2_idx)

            for j in range(data_batch.shape[0]):
                filt = self.ee_vocab[(data_batch[j][0], data_batch[j][2])]
                target_value = predictions[j, r_idx[j]].item()
                predictions[j, filt] = -np.Inf
                predictions[j, r_idx[j]] = target_value

            _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == r_idx[j])[0].item() + 1
                ranks.append(rank)
                update_hits(hits, rank, hits_range)

        return ranks, hits

    def _evaluate_entity_prediction(
        self,
        model,
        triple_idx,
        num_triples: int,
        hits_range: List[int]
    ) -> Tuple[List[int], Dict[int, List[float]]]:
        """Evaluate entity prediction task."""
        ranks: List[int] = []
        hits = create_hits_dict(hits_range)

        for i in range(0, num_triples, self.args.batch_size):
            data_batch = triple_idx[i:i + self.args.batch_size]
            e1_idx_r_idx = torch.LongTensor(data_batch[:, [0, 1]])
            e2_idx = torch.tensor(data_batch[:, 2])

            with torch.no_grad():
                predictions = model(e1_idx_r_idx)

            for j in range(data_batch.shape[0]):
                id_e, id_r, id_e_target = data_batch[j]
                filt = self.er_vocab[(id_e, id_r)]
                target_value = predictions[j, id_e_target].item()
                predictions[j, filt] = -np.Inf

                if 'constraint' in self.args.eval_model:
                    predictions[j, self.range_constraints_per_rel[data_batch[j, 1]]] = -np.Inf

                predictions[j, id_e_target] = target_value

            _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            for j in range(data_batch.shape[0]):
                rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
                ranks.append(rank)
                update_hits(hits, rank, hits_range)

        return ranks, hits

    @torch.no_grad()
    def evaluate_lp_with_byte(
        self,
        model,
        triples: List[List[str]],
        info: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate BytE model with text generation.

        Args:
            model: BytE model.
            triples: String triples.
            info: Description to print.

        Returns:
            Dictionary with placeholder metrics (-1 values).
        """
        import tiktoken

        model.eval()
        num_triples = len(triples)
        enc = tiktoken.get_encoding("gpt2")

        if info and not self.during_training:
            print(info + ':', end=' ')

        for i in range(0, num_triples, self.args.batch_size):
            str_data_batch = triples[i:i + self.args.batch_size]
            for triple in str_data_batch:
                s, p, o = triple
                x = torch.LongTensor([enc.encode(s + " " + p)])
                print("Triple:", triple, end="\t")
                y = model.generate(
                    x, max_new_tokens=100,
                    temperature=model.temperature,
                    top_k=model.topk
                ).tolist()
                print("Generated:", enc.decode(y[0]))

        return {'H@1': -1, 'H@3': -1, 'H@10': -1, 'MRR': -1}

    @torch.no_grad()
    def evaluate_lp_bpe_k_vs_all(
        self,
        model,
        triples: List[List[str]],
        info: Optional[str] = None,
        form_of_labelling: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate BPE model with KvsAll scoring.

        Args:
            model: BPE-enabled model.
            triples: String triples.
            info: Description to print.
            form_of_labelling: Type of labelling.

        Returns:
            Dictionary with H@1, H@3, H@10, and MRR metrics.
        """
        model.eval()
        num_triples = len(triples)
        ranks: List[int] = []
        hits_range = ALL_HITS_RANGE
        hits = create_hits_dict(hits_range)

        if info and not self.during_training:
            print(info + ':', end=' ')

        for i in range(0, num_triples, self.args.batch_size):
            str_data_batch = triples[i:i + self.args.batch_size]
            torch_batch_bpe_triple = torch.LongTensor([
                self.func_triple_to_bpe_representation(t)
                for t in str_data_batch
            ])

            bpe_hr = torch_batch_bpe_triple[:, [0, 1], :]
            predictions = model(bpe_hr)

            for j in range(len(predictions)):
                h, r, t = str_data_batch[j]
                id_e_target = model.str_to_bpe_entity_to_idx[t]
                filt_idx_entities = [
                    model.str_to_bpe_entity_to_idx[_]
                    for _ in self.er_vocab[(h, r)]
                ]
                target_value = predictions[j, id_e_target].item()
                predictions[j, filt_idx_entities] = -np.Inf
                predictions[j, id_e_target] = target_value

            _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            for j in range(len(predictions)):
                t = str_data_batch[j][2]
                rank = torch.where(
                    sort_idxs[j] == model.str_to_bpe_entity_to_idx[t]
                )[0].item() + 1
                ranks.append(rank)
                update_hits(hits, rank, hits_range)

        assert len(triples) == len(ranks) == num_triples

        results = compute_metrics_from_ranks_simple(ranks, num_triples, hits)

        if info and not self.during_training:
            print(info)
            print(results)

        return results

    def evaluate_lp(self, model, triple_idx, info: str) -> Dict[str, float]:
        """Evaluate link prediction with negative sampling.

        Args:
            model: The model to evaluate.
            triple_idx: Integer-indexed triples.
            info: Description to print.

        Returns:
            Dictionary with H@1, H@3, H@10, and MRR metrics.
        """
        assert self.num_entities is not None, "self.num_entities cannot be None"
        assert self.er_vocab is not None, "self.er_vocab cannot be None"
        assert self.re_vocab is not None, "self.re_vocab cannot be None"

        return evaluate_lp(
            model, triple_idx,
            num_entities=self.num_entities,
            er_vocab=self.er_vocab,
            re_vocab=self.re_vocab,
            info=info
        )

    def dummy_eval(self, trained_model, form_of_labelling: str) -> None:
        """Run evaluation from saved data (for continual training).

        Args:
            trained_model: The trained model.
            form_of_labelling: Type of labelling.
        """
        assert trained_model is not None

        if self.is_continual_training:
            self._load_and_set_mappings()

        train_set, valid_set, test_set = self._load_indexed_datasets()

        if self.args.scoring_technique == 'NegSample':
            self.eval_rank_of_head_and_tail_entity(
                train_set=train_set,
                valid_set=valid_set,
                test_set=test_set,
                trained_model=trained_model
            )
        elif self.args.scoring_technique in ["AllvsAll", 'KvsAll', '1vsSample', "KvsSample", '1vsAll']:
            self.eval_with_vs_all(
                train_set=train_set,
                valid_set=valid_set,
                test_set=test_set,
                trained_model=trained_model,
                form_of_labelling=form_of_labelling
            )
        else:
            raise ValueError(f'Invalid scoring technique: {self.args.scoring_technique}')

        report_path = os.path.join(self.args.full_storage_path, 'eval_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=4)

    def eval_with_data(
        self,
        dataset,
        trained_model,
        triple_idx: np.ndarray,
        form_of_labelling: str
    ) -> Dict[str, float]:
        """Evaluate a trained model on a given dataset.

        Args:
            dataset: Knowledge graph dataset.
            trained_model: The trained model.
            triple_idx: Integer-indexed triples to evaluate.
            form_of_labelling: Type of labelling.

        Returns:
            Dictionary with evaluation metrics.

        Raises:
            ValueError: If scoring technique is invalid.
        """
        self.vocab_preparation(dataset)

        if self.args.scoring_technique == 'NegSample':
            return self.evaluate_lp(
                trained_model, triple_idx,
                info=f'Evaluate {trained_model.name} on a given dataset'
            )
        elif self.args.scoring_technique in ['KvsAll', '1vsSample', '1vsAll', 'PvsAll', 'CCvsAll']:
            return self.evaluate_lp_k_vs_all(
                trained_model, triple_idx,
                info=f'Evaluate {trained_model.name} on a given dataset',
                form_of_labelling=form_of_labelling
            )
        elif self.args.scoring_technique in ['BatchRelaxedKvsAll', 'BatchRelaxed1vsAll']:
            return self.evaluate_lp_k_vs_all(
                trained_model, triple_idx,
                info=f'Evaluate {trained_model.name} on a given dataset',
                form_of_labelling=form_of_labelling
            )
        else:
            raise ValueError(f'Invalid scoring technique: {self.args.scoring_technique}')
