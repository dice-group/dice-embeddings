from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TabPFNClassifier = None
    TABPFN_AVAILABLE = False

from ..static_funcs import timeit
from .static_funcs import prepare_tabpfn_splits


class TabularTrainer:
    """Train and evaluate a TabPFN classifier on KG-derived features."""

    def __init__(
        self,
        device: str = "cpu",
        max_train_samples: Optional[int] = None,
        n_estimators: int = 8,
        random_seed: int = 0,
        n_preprocessing_jobs: int = 1,
    ) -> None:
        """Configure the TabPFN classifier and training-time limits."""
        if not TABPFN_AVAILABLE:
            raise ImportError(
                "TabPFN is required for tabular training but is not installed.\n"
                "Please install it with: pip install tabpfn\n"
                "Or reinstall dicee with all dependencies: pip install -e '.[dev]'"
            )
        self.device = device
        self.max_train_samples = max_train_samples
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.n_preprocessing_jobs = n_preprocessing_jobs

    def _create_classifier(self) -> TabPFNClassifier:
        """Instantiate a TabPFN classifier with the configured settings."""
        return TabPFNClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            ignore_pretraining_limits=True,
            random_state=self.random_seed,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
        )

    @timeit
    def fit_and_evaluate(self, data: Dict, entity_centric: bool = False) -> Tuple[object, Dict]:
        """Fit TabPFN on the training split and evaluate on valid/test splits."""
        x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_tabpfn_splits(
            data, entity_centric=entity_centric
        )
        classifier = self._create_classifier()

        if self.max_train_samples is not None and x_train.shape[0] > self.max_train_samples:
            # TabPFN is most reliable on smaller training sets.
            print(f"Subsampling training data to {self.max_train_samples} samples...")
            rng = np.random.default_rng(self.random_seed)
            indices = rng.choice(x_train.shape[0], self.max_train_samples, replace=False)
            x_train_fit = x_train[indices]
            y_train_fit = y_train[indices]
        else:
            x_train_fit = x_train
            y_train_fit = y_train

        print("Fitting classifier...")
        classifier.fit(x_train_fit, y_train_fit)
        print("Evaluating validation and test splits...")
        y_pred = classifier.predict(x_test)
        y_pred_proba = classifier.predict_proba(x_test)[:, 1]
        y_valid_pred = classifier.predict(x_valid)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        metrics = {
            "tabpfn_test_accuracy": accuracy_score(y_test, y_pred),
            "tabpfn_test_precision": precision,
            "tabpfn_test_recall": recall,
            "tabpfn_test_f1": f1,
            "tabpfn_test_auc": roc_auc_score(y_test, y_pred_proba),
            "tabpfn_valid_accuracy": accuracy_score(y_valid, y_valid_pred),
            "tabpfn_train_samples": int(x_train.shape[0]),
            "tabpfn_valid_samples": int(x_valid.shape[0]),
            "tabpfn_test_samples": int(x_test.shape[0]),
            "tabpfn_num_features": int(x_train.shape[1]),
            "tabpfn_fitted_train_samples": int(x_train_fit.shape[0]),
        }
        return classifier, metrics
