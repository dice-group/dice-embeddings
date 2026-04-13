import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..static_funcs import timeit
from .tabular_dataset import EntityCentricDataset, TripleCentricDataset


def get_tabpfn_config(args: object) -> Dict:
    """Normalize optional TabPFN kwargs into a complete config dict."""
    config = dict(args.tabpfn_kwargs)
    config.setdefault("device", "cpu")
    config.setdefault("max_train_samples", None)
    config.setdefault("n_estimators", 8)
    config.setdefault("entity_centric", False)
    return config


@timeit
def load_kg_for_tabpfn(
    dataset_dir: str,
    negative_ratio: float = 1.0,
    entity_centric: bool = False,
    separator: str = "\t",
) -> Dict:
    """Load train/valid/test splits and convert them into tabular inputs."""
    mode = "entity-centric" if entity_centric else "triple-centric"
    print(f"Loading KG dataset from {dataset_dir} ({mode})...")
    train_file = os.path.join(dataset_dir, "train.txt")
    valid_file = os.path.join(dataset_dir, "valid.txt")
    test_file = os.path.join(dataset_dir, "test.txt")

    files_to_use = {
        "train": train_file if os.path.exists(train_file) else None,
        "valid": valid_file if os.path.exists(valid_file) else None,
        "test": test_file if os.path.exists(test_file) else None,
    }
    if files_to_use["train"] is None:
        raise FileNotFoundError(f"Training file not found: {train_file}")

    if entity_centric:
        dataset = EntityCentricDataset(separator=separator)
        return dataset.generate_entity_centric_dataset(
            train_file=files_to_use["train"],
            valid_file=files_to_use["valid"],
            test_file=files_to_use["test"],
            negative_ratio=negative_ratio,
        )

    dataset = TripleCentricDataset(separator=separator)
    return dataset.load_and_convert(
        train_file=files_to_use["train"],
        valid_file=files_to_use["valid"],
        test_file=files_to_use["test"],
        negative_ratio=negative_ratio,
    )


def update_report_with_data_statistics(report: Dict, tabular_data: Dict, start_time: float, runtime_key: str) -> None:
    """Store split-level dataset statistics and runtime in the report dict."""
    for split, split_data in tabular_data.items():
        if isinstance(split_data, dict):
            report[f"{split}_shape"] = tuple(split_data["X"].shape)
            report[f"{split}_positive_samples"] = int(np.sum(split_data["y"] == 1))
            report[f"{split}_negative_samples"] = int(np.sum(split_data["y"] == 0))
        else:
            report[f"{split}_shape"] = tuple(split_data.shape)
            report[f"{split}_positive_samples"] = int(np.sum(split_data["Label"] == 1))
            report[f"{split}_negative_samples"] = int(np.sum(split_data["Label"] == 0))
    report[runtime_key] = __import__("time").time() - start_time


def prepare_tabpfn_splits(data: Dict, entity_centric: bool = False) -> Tuple[np.ndarray, ...]:
    """Convert loaded tabular data into train/valid/test numpy arrays."""
    if "valid" not in data or "test" not in data:
        raise ValueError("TabPFN pipeline requires valid and test splits.")

    if entity_centric:
        # Entity-centric data uses categorical columns that must be encoded first.
        train_df = data["train"].copy()
        valid_df = data["valid"].copy()
        test_df = data["test"].copy()

        # Different splits can expose different relation columns, so align them first.
        feature_columns = list(train_df.columns)
        for dataframe in (valid_df, test_df):
            for column in dataframe.columns:
                if column not in feature_columns:
                    feature_columns.append(column)
        feature_columns = [column for column in feature_columns if column != "Label"]

        train_df = train_df.reindex(columns=feature_columns + ["Label"], fill_value="NotApplicable")
        valid_df = valid_df.reindex(columns=feature_columns + ["Label"], fill_value="NotApplicable")
        test_df = test_df.reindex(columns=feature_columns + ["Label"], fill_value="NotApplicable")

        for column in feature_columns:
            if train_df[column].dtype != "object":
                continue
            label_encoder = LabelEncoder()
            all_values = pd.concat([train_df[column], valid_df[column], test_df[column]]).unique()
            label_encoder.fit(all_values)
            train_df[column] = label_encoder.transform(train_df[column])
            valid_df[column] = label_encoder.transform(valid_df[column])
            test_df[column] = label_encoder.transform(test_df[column])

        x_train = train_df.drop("Label", axis=1).values.astype(np.float32)
        y_train = train_df["Label"].values.astype(np.int32)
        x_valid = valid_df.drop("Label", axis=1).values.astype(np.float32)
        y_valid = valid_df["Label"].values.astype(np.int32)
        x_test = test_df.drop("Label", axis=1).values.astype(np.float32)
        y_test = test_df["Label"].values.astype(np.int32)
    else:
        x_train, y_train = data["train"]["X"], data["train"]["y"]
        x_valid, y_valid = data["valid"]["X"], data["valid"]["y"]
        x_test, y_test = data["test"]["X"], data["test"]["y"]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def print_tabpfn_dataset_overview(
    data: Dict,
    max_train_samples: int | None,
    entity_centric: bool = False,
) -> None:
    """Print split sizes after the tabular dataset is fully prepared."""
    x_train, _, x_valid, _, x_test, _ = prepare_tabpfn_splits(
        data, entity_centric=entity_centric
    )
    num_features = int(x_train.shape[1])

    print("\nDataset Summary")
    train_line = f"Train set: samples={int(x_train.shape[0])}"
    if max_train_samples is not None:
        fitted_train_samples = min(int(x_train.shape[0]), int(max_train_samples))
        train_line += f" fitted_samples={fitted_train_samples}"
    train_line += f" features={num_features}"
    print(train_line)
    print(
        "Valid set: "
        f"samples={int(x_valid.shape[0])} "
        f"features={num_features}"
    )
    print(
        "Test set: "
        f"samples={int(x_test.shape[0])} "
        f"features={num_features}"
    )


def print_tabpfn_results(metrics: Dict) -> None:
    """Print the final validation/test metrics for a TabPFN run."""
    print("\nEvaluation Results")
    print(
        "Valid set: "
        f"Acc={metrics['tabpfn_valid_accuracy']:.4f} "
        f"samples={metrics['tabpfn_valid_samples']}"
    )
    print(
        "Test set: "
        f"Acc={metrics['tabpfn_test_accuracy']:.4f} "
        f"Prec={metrics['tabpfn_test_precision']:.4f} "
        f"Rec={metrics['tabpfn_test_recall']:.4f} "
        f"F1={metrics['tabpfn_test_f1']:.4f} "
        f"AUC={metrics['tabpfn_test_auc']:.4f} "
        f"samples={metrics['tabpfn_test_samples']}"
    )


def convert_only(
    dataset_dir: str = "KGs/UMLS",
    negative_ratio: float = 1.0,
    entity_centric: bool = False,
    separator: str = "\t",
) -> Dict:
    """Load and convert a KG into the tabular representation only."""
    return load_kg_for_tabpfn(
        dataset_dir,
        negative_ratio=negative_ratio,
        entity_centric=entity_centric,
        separator=separator,
    )


def demo_entity_centric() -> pd.DataFrame:
    """Return a small entity-centric demo table built from toy triples."""
    triples = [
        ("CaglarDemir", "LivesIn", "Germany"),
        ("CaglarDemir", "isA", "ComputerScientist"),
        ("CaglarDemir", "isA", "Person"),
        ("Germany", "isA", "Country"),
    ]
    dataset = EntityCentricDataset()
    labels = [1] * len(triples)
    return dataset.triples_to_entity_centric_tabular(triples, labels)


def summarize_converted_data(data: Dict, entity_centric: bool = False) -> Dict:
    """Summarize converted tabular data by split."""
    summary = {}
    if entity_centric:
        for split, dataframe in data.items():
            summary[split] = {
                "shape": tuple(dataframe.shape),
                "positive_samples": int(np.sum(dataframe["Label"] == 1)),
                "negative_samples": int(np.sum(dataframe["Label"] == 0)),
            }
        return summary

    for split, split_data in data.items():
        x_data, y_data = split_data["X"], split_data["y"]
        summary[split] = {
            "shape": tuple(x_data.shape),
            "positive_samples": int(np.sum(y_data == 1)),
            "negative_samples": int(np.sum(y_data == 0)),
        }
    return summary


def summarize_training_inputs(data: Dict, entity_centric: bool = False) -> Dict:
    """Summarize the train/valid/test arrays used by the trainer."""
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_tabpfn_splits(
        data, entity_centric=entity_centric
    )
    return {
        "train_shape": tuple(x_train.shape),
        "valid_shape": tuple(x_valid.shape),
        "test_shape": tuple(x_test.shape),
        "train_positive_samples": int(np.sum(y_train == 1)),
        "valid_positive_samples": int(np.sum(y_valid == 1)),
        "test_positive_samples": int(np.sum(y_test == 1)),
    }
