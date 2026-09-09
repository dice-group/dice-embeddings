"""Factory functions for constructing training datasets.

The two public functions — ``reload_dataset`` and ``construct_dataset`` —
select the appropriate ``torch.utils.data.Dataset`` sub-class based on the
requested scoring technique and labelling strategy.
"""

import logging
from typing import Union

import numpy as np
import torch

from ..static_funcs import load_term_mapping, timeit
from ._bpe import (
    BPE_NegativeSamplingDataset,
    MultiClassClassificationDataset,
    MultiLabelDataset,
)
from ._label_based import AllvsAll, FSDP1vsSampleDataset, KvsAll, KvsSampleDataset, OnevsAllDataset
from ._negative_sampling import FixedNegSampleDataset, GroupedNegativeSamplingDataset, OnevsSample, TriplePredictionDataset

logger = logging.getLogger(__name__)


@timeit
def reload_dataset(
    path: str,
    form_of_labelling,
    scoring_technique,
    neg_ratio,
    label_smoothing_rate,
):
    """Reload training data from disk and construct a Pytorch dataset."""
    return construct_dataset(
        train_set=np.load(path + "/train_set.npy"),
        valid_set=None,
        test_set=None,
        entity_to_idx=load_term_mapping(file_path=path + "/entity_to_idx"),
        relation_to_idx=load_term_mapping(file_path=path + "/relation_to_idx"),
        form_of_labelling=form_of_labelling,
        scoring_technique=scoring_technique,
        neg_ratio=neg_ratio,
        label_smoothing_rate=label_smoothing_rate,
    )


@timeit
def construct_dataset(
    *,
    train_set: Union[np.ndarray, list],
    valid_set=None,
    test_set=None,
    ordered_bpe_entities=None,
    train_target_indices=None,
    target_dim: int = None,
    entity_to_idx: dict,
    relation_to_idx: dict,
    form_of_labelling: str,
    scoring_technique: str,
    neg_ratio: int,
    label_smoothing_rate: float,
    byte_pair_encoding=None,
    block_size: int = None,
    seed: int = None,
    sort_train_set: bool = True,
    grouped_negative_sampling: bool = False,
    strict_negative_sampling: bool = False,
    adversarial_temperature: float | None = None,
) -> torch.utils.data.Dataset:
    """Build the appropriate dataset for the given training configuration.

    Parameters
    ----------
    train_set : numpy.ndarray or list
        Raw integer-indexed triples.
    entity_to_idx, relation_to_idx : dict
        Name → index mappings.
    form_of_labelling : str
        ``'EntityPrediction'`` or ``'RelationPrediction'``.
    scoring_technique : str
        One of ``'NegSample'``, ``'FixedNegSample'``, ``'1vsAll'``, ``'1vsSample'``, ``'KvsAll'``,
        ``'AllvsAll'``, ``'KvsSample'``.
    neg_ratio : int
        Negative sample ratio.
    label_smoothing_rate : float
        Label smoothing coefficient.

    Returns
    -------
    torch.utils.data.Dataset
    """
    grouped = grouped_negative_sampling or strict_negative_sampling or adversarial_temperature is not None
    if grouped:
        if scoring_technique != "NegSample" or byte_pair_encoding or form_of_labelling != "EntityPrediction":
            raise ValueError("Grouped/strict/adversarial sampling requires indexed NegSample entity prediction")
        return GroupedNegativeSamplingDataset(
            train_set=train_set, num_entities=len(entity_to_idx), num_relations=len(relation_to_idx),
            neg_sample_ratio=neg_ratio, label_smoothing_rate=label_smoothing_rate,
            seed=seed, sort_train_set=sort_train_set, strict_negative_sampling=strict_negative_sampling)
    if (
        ordered_bpe_entities
        and byte_pair_encoding
        and scoring_technique == "NegSample"
    ):
        train_set = BPE_NegativeSamplingDataset(
            train_set=torch.tensor(train_set, dtype=torch.long),
            ordered_shaped_bpe_entities=torch.tensor(
                [
                    shaped_bpe_ent
                    for (str_ent, bpe_ent, shaped_bpe_ent) in ordered_bpe_entities
                ]
            ),
            neg_ratio=neg_ratio,
        )
    elif (
        ordered_bpe_entities
        and byte_pair_encoding
        and scoring_technique in ["KvsAll", "AllvsAll"]
    ):
        train_set = MultiLabelDataset(
            train_set=torch.tensor(train_set, dtype=torch.long),
            train_indices_target=train_target_indices,
            target_dim=target_dim,
            torch_ordered_shaped_bpe_entities=torch.tensor(
                [
                    shaped_bpe_ent
                    for (str_ent, bpe_ent, shaped_bpe_ent) in ordered_bpe_entities
                ]
            ),
        )
    elif byte_pair_encoding:
        train_set = MultiClassClassificationDataset(
            train_set, block_size=block_size
        )
    elif scoring_technique == "NegSample":
        train_set = TriplePredictionDataset(
            train_set=train_set,
            num_entities=len(entity_to_idx),
            num_relations=len(relation_to_idx),
            neg_sample_ratio=neg_ratio,
            label_smoothing_rate=label_smoothing_rate,
            seed=seed,
            sort_train_set=sort_train_set,
        )
    elif scoring_technique == "FSDP1vsSample":
        train_set = FSDP1vsSampleDataset(
            train_set_idx=train_set,
            entity_idxs=entity_to_idx,
            relation_idxs=relation_to_idx,
            form=form_of_labelling,
            neg_ratio=neg_ratio,
            label_smoothing_rate=label_smoothing_rate,
        )
    elif scoring_technique == "FixedNegSample":
        train_set = FixedNegSampleDataset(
            train_set=train_set,
            num_entities=len(entity_to_idx),
            num_relations=len(relation_to_idx),
            neg_sample_ratio=neg_ratio,
            label_smoothing_rate=label_smoothing_rate,
            seed=seed,
        )
    elif form_of_labelling == "EntityPrediction":
        if scoring_technique == "1vsAll":
            train_set = OnevsAllDataset(train_set, entity_idxs=entity_to_idx)
        elif scoring_technique == "1vsSample":
            train_set = OnevsSample(
                train_set=train_set,
                num_entities=len(entity_to_idx),
                num_relations=len(relation_to_idx),
                neg_sample_ratio=neg_ratio,
                label_smoothing_rate=label_smoothing_rate,
            )
        elif scoring_technique == "KvsAll":
            train_set = KvsAll(
                train_set,
                entity_idxs=entity_to_idx,
                relation_idxs=relation_to_idx,
                form=form_of_labelling,
                label_smoothing_rate=label_smoothing_rate,
            )
        elif scoring_technique == "AllvsAll":
            train_set = AllvsAll(
                train_set,
                entity_idxs=entity_to_idx,
                relation_idxs=relation_to_idx,
                label_smoothing_rate=label_smoothing_rate,
            )
        elif scoring_technique == "KvsSample":
            train_set = KvsSampleDataset(
                train_set,
                entity_idxs=entity_to_idx,
                relation_idxs=relation_to_idx,
                form=form_of_labelling,
                neg_ratio=neg_ratio,
                label_smoothing_rate=label_smoothing_rate,
            )
        else:
            raise ValueError(f"Invalid scoring technique : {scoring_technique}")
    elif form_of_labelling == "RelationPrediction":
        train_set = KvsAll(
            train_set,
            entity_idxs=entity_to_idx,
            relation_idxs=relation_to_idx,
            form=form_of_labelling,
            label_smoothing_rate=label_smoothing_rate,
        )
    else:
        raise KeyError("Illegal input.")

    logger.info(f"Number of datapoints: {len(train_set)}")
    return train_set
