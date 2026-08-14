"""Threshold analysis utilities for binary classification."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from machine_failure.data import load_data
from machine_failure.features import split_features_target
from machine_failure.model import ModelType
from machine_failure.train import (
    DATA_PATH,
    create_train_validation_test_split,
    get_model_path,
    get_validation_model_path,
)

TUNED_RANDOM_FOREST_PATH = Path("models/random_forest_tuned_production.joblib")

TUNED_XGBOOST_PATH = Path("models/xgboost_tuned_production.joblib")


def evaluate_thresholds(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate classification metrics across probability thresholds."""

    if thresholds is None:
        thresholds = np.arange(
            0.05,
            0.96,
            0.01,
        )

    probabilities = model.predict_proba(features)[:, 1]

    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(
                    target,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    target,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    target,
                    predictions,
                    zero_division=0,
                ),
                "f2": fbeta_score(
                    target,
                    predictions,
                    beta=2,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def select_best_threshold(
    results: pd.DataFrame,
    metric: str = "f2",
) -> pd.Series:
    """Return the row with the highest selected metric."""

    if metric not in results.columns:
        raise ValueError(f"Metric '{metric}' was not found in threshold results.")

    best_index = results[metric].idxmax()

    return results.loc[best_index]


def plot_threshold_metrics(
    results: pd.DataFrame,
    model_name: str,
) -> None:
    """Plot precision, recall, F1 and F2 across thresholds."""

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["threshold"],
        results["precision"],
        label="Precision",
    )

    plt.plot(
        results["threshold"],
        results["recall"],
        label="Recall",
    )

    plt.plot(
        results["threshold"],
        results["f1"],
        label="F1",
    )

    plt.plot(
        results["threshold"],
        results["f2"],
        label="F2",
    )

    plt.xlabel("Decision Threshold")

    plt.ylabel("Metric Value")

    plt.title(f"Threshold Tuning — {model_name}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def load_threshold_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Load production features and target for threshold analysis."""

    dataframe = load_data(DATA_PATH)

    return split_features_target(
        dataframe,
        production_mode=True,
    )


def load_threshold_model(
    model_path: Path,
    description: str,
) -> Pipeline:
    """Load a saved model after validating that it exists."""

    if not model_path.exists():
        raise FileNotFoundError(f"{description} was not found: {model_path.resolve()}")

    return joblib.load(model_path)


def get_validation_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    int,
]:
    """Return validation features, target, and reserved test-set size."""

    features, target = load_threshold_data()

    (
        _X_train,
        X_validation,
        X_test,
        _y_train,
        y_validation,
        _y_test,
    ) = create_train_validation_test_split(
        features,
        target,
    )

    return (
        X_validation,
        y_validation,
        len(X_test),
    )


def analyze_model_thresholds_on_validation(
    model: Pipeline,
    model_name: str,
) -> pd.DataFrame:
    """Analyze decision thresholds using the validation set only."""

    (
        X_validation,
        y_validation,
        test_size,
    ) = get_validation_data()

    results = evaluate_thresholds(
        model,
        X_validation,
        y_validation,
    )

    best_f1 = select_best_threshold(
        results,
        metric="f1",
    )

    best_f2 = select_best_threshold(
        results,
        metric="f2",
    )

    print("=" * 60)
    print(f"{model_name} — VALIDATION BEST THRESHOLD BY F1")
    print("=" * 60)

    print(best_f1.to_string())

    print("\n" + "=" * 60)

    print(f"{model_name} — VALIDATION BEST THRESHOLD BY F2")

    print("=" * 60)

    print(best_f2.to_string())

    print("\nThreshold selection dataset:")

    print(f"Validation samples: {len(X_validation)}")

    print(f"Reserved test samples: {test_size}")

    plot_threshold_metrics(
        results,
        model_name=(f"{model_name} — Validation"),
    )

    return results


def run_threshold_analysis_for_path(
    model_path: Path,
    model_name: str,
    description: str,
) -> pd.DataFrame:
    """Load a saved model and run validation-based threshold analysis."""

    model = load_threshold_model(
        model_path,
        description=description,
    )

    return analyze_model_thresholds_on_validation(
        model=model,
        model_name=model_name,
    )


def run_validation_threshold_analysis(
    model_type: ModelType,
) -> pd.DataFrame:
    """Tune a validation-stage model threshold on validation data."""

    model_path = get_validation_model_path(model_type)

    display_name = model_type.replace("_", " ").title()

    return run_threshold_analysis_for_path(
        model_path=model_path,
        model_name=display_name,
        description="Validation-stage model",
    )


# ------------------------------------------------------------------
# Older experimental model artifacts
# ------------------------------------------------------------------
#
# These functions are retained for reproducibility of earlier
# experiments. Final model selection should use the validation-stage
# workflow above.
# ------------------------------------------------------------------


def run_threshold_analysis(
    model_type: ModelType,
) -> pd.DataFrame:
    """Analyze an older production model on validation data."""

    model_path = get_model_path(model_type)

    display_name = model_type.replace("_", " ").title()

    return run_threshold_analysis_for_path(
        model_path=model_path,
        model_name=display_name,
        description="Trained model",
    )


def run_tuned_random_forest_threshold_analysis() -> pd.DataFrame:
    """Analyze the previously tuned Random Forest."""

    return run_threshold_analysis_for_path(
        model_path=TUNED_RANDOM_FOREST_PATH,
        model_name="Tuned Random Forest",
        description="Tuned Random Forest model",
    )


def run_tuned_xgboost_threshold_analysis() -> pd.DataFrame:
    """Analyze the previously tuned XGBoost model."""

    return run_threshold_analysis_for_path(
        model_path=TUNED_XGBOOST_PATH,
        model_name="Tuned XGBoost",
        description="Tuned XGBoost model",
    )


if __name__ == "__main__":
    run_validation_threshold_analysis(
        model_type="xgboost",
    )
