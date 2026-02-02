"""Literal prediction evaluation functions.

This module provides functions for evaluating literal/attribute prediction
performance of knowledge graph embedding models.
"""

import os
from typing import Optional

import pandas as pd



def evaluate_literal_prediction(
    kge_model,
    eval_file_path: str = None,
    store_lit_preds: bool = True,
    eval_literals: bool = True,
    loader_backend: str = "pandas",
    return_attr_error_metrics: bool = False,
) -> Optional[pd.DataFrame]:
    """Evaluate trained literal prediction model on a test file.

    Evaluates the literal prediction capabilities of a KGE model by
    computing MAE and RMSE metrics for each attribute.

    Args:
        kge_model: Trained KGE model with literal prediction capability.
        eval_file_path: Path to the evaluation file containing test literals.
        store_lit_preds: If True, stores predictions to CSV file.
        eval_literals: If True, evaluates and prints error metrics.
        loader_backend: Backend for loading dataset ('pandas' or 'rdflib').
        return_attr_error_metrics: If True, returns the metrics DataFrame.

    Returns:
        DataFrame with per-attribute MAE and RMSE if return_attr_error_metrics
        is True, otherwise None.

    Raises:
        RuntimeError: If the KGE model doesn't have a trained literal model.
        AssertionError: If model is invalid or test set has no valid data.

    Example:
        >>> from dicee import KGE
        >>> from dicee.evaluation import evaluate_literal_prediction
        >>> model = KGE(path="pretrained_model")
        >>> metrics = evaluate_literal_prediction(
        ...     model,
        ...     eval_file_path="test_literals.csv",
        ...     return_attr_error_metrics=True
        ... )
        >>> print(metrics)
    """
    # Import here to avoid circular imports
    from ..knowledge_graph_embeddings import KGE

    # Model validation
    assert isinstance(kge_model, KGE), "kge_model must be an instance of KGE."
    if not hasattr(kge_model, "literal_model") or kge_model.literal_model is None:
        raise RuntimeError("Literal model is not trained or loaded.")

    # Load and validate test data
    test_df_unfiltered = kge_model.literal_dataset.load_and_validate_literal_data(
        file_path=eval_file_path,
        loader_backend=loader_backend
    )

    # Filter to known entities and attributes
    test_df = test_df_unfiltered[
        test_df_unfiltered["head"].isin(kge_model.entity_to_idx.keys()) &
        test_df_unfiltered["attribute"].isin(kge_model.data_property_to_idx.keys())
    ]

    entities = test_df["head"].to_list()
    attributes = test_df["attribute"].to_list()

    assert len(entities) > 0, "No valid entities in test set — check entity_to_idx."
    assert len(attributes) > 0, "No valid attributes in test set — check data_property_to_idx."

    # Generate predictions
    test_df["predictions"] = kge_model.predict_literals(
        entity=entities,
        attribute=attributes
    )

    # Store predictions if requested
    if store_lit_preds:
        prediction_df = test_df[["head", "attribute", "predictions"]]
        prediction_path = os.path.join(kge_model.path, "lit_predictions.csv")
        prediction_df.to_csv(prediction_path, index=False)
        print(f"Literal predictions saved to {prediction_path}")
    try:
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
    except ImportError:
        raise ImportError(
            "scikit-learn is required for evaluating literal prediction metrics. "
            "Please install it using 'pip install scikit-learn'."
        )
    # Calculate and store error metrics
    if eval_literals:
        attr_error_metrics = test_df.groupby("attribute").agg(
            MAE=("value", lambda x: mean_absolute_error(
                x, test_df.loc[x.index, "predictions"]
            )),
            RMSE=("value", lambda x: root_mean_squared_error(
                x, test_df.loc[x.index, "predictions"]
            ))
        ).reset_index()

        pd.options.display.float_format = "{:.6f}".format
        print("Literal-Prediction evaluation results on Test Set")
        print(attr_error_metrics)

        results_path = os.path.join(kge_model.path, "lit_eval_results.csv")
        attr_error_metrics.to_csv(results_path, index=False)
        print(f"Literal-Prediction evaluation results saved to {results_path}")

        if return_attr_error_metrics:
            return attr_error_metrics

    return None
