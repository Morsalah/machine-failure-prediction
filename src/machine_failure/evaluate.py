"""Model evaluation utilities for machine-failure classification."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from machine_failure.data import load_data
from machine_failure.features import split_features_target
from machine_failure.model import ModelType
from machine_failure.reporting import (
    append_to_model_comparison,
    format_threshold_label,
    save_figure,
    save_metrics,
)
from machine_failure.train import (
    DATA_PATH,
    create_development_test_split,
    create_train_test_split,
    get_model_path,
)

DEFAULT_THRESHOLD = 0.50

TUNED_RANDOM_FOREST_PATH = Path("models/random_forest_tuned_production.joblib")

TUNED_XGBOOST_PATH = Path("models/xgboost_tuned_production.joblib")

FINAL_XGBOOST_PATH = Path("models/xgboost_final.joblib")

FINAL_XGBOOST_THRESHOLD = 0.09


def calculate_metrics(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Calculate classification metrics at a selected threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    probabilities = model.predict_proba(features)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            target,
            predictions,
        ),
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
        "roc_auc": roc_auc_score(
            target,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            target,
            probabilities,
        ),
    }


def print_metrics(
    metrics: dict[str, float],
    title: str,
) -> None:
    """Print evaluation metrics in a readable format."""

    print("=" * 50)
    print(title)
    print("=" * 50)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:>10}: {metric_value:.4f}")


def plot_confusion_matrix(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float,
    model_name: str,
    model_type: str,
) -> None:
    """Create and save a confusion matrix."""

    probabilities = model.predict_proba(features)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        target,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No Failure",
            "Failure",
        ],
    )

    display.plot(
        values_format="d",
    )

    figure = plt.gcf()

    plt.title(f"{model_name} Confusion Matrix\nThreshold = {threshold:.2f}")

    plt.tight_layout()

    threshold_label = format_threshold_label(threshold)

    output_path = save_figure(
        figure=figure,
        filename=(f"{model_type}_confusion_matrix_threshold_{threshold_label}.png"),
    )

    print(f"Confusion matrix saved to: {output_path}")

    plt.show()


def plot_roc_curve(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
    model_type: str,
) -> None:
    """Create and save the ROC curve."""

    probabilities = model.predict_proba(features)[:, 1]

    (
        false_positive_rate,
        true_positive_rate,
        _,
    ) = roc_curve(
        target,
        probabilities,
    )

    roc_auc = roc_auc_score(
        target,
        probabilities,
    )

    figure = plt.figure(figsize=(8, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=(f"{model_name} (AUC = {roc_auc:.3f})"),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"ROC Curve — {model_name}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = save_figure(
        figure=figure,
        filename=(f"{model_type}_roc_curve.png"),
    )

    print(f"ROC curve saved to: {output_path}")

    plt.show()


def plot_precision_recall_curve(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
    model_type: str,
) -> None:
    """Create and save the precision-recall curve."""

    probabilities = model.predict_proba(features)[:, 1]

    (
        precision,
        recall,
        _,
    ) = precision_recall_curve(
        target,
        probabilities,
    )

    average_precision = average_precision_score(
        target,
        probabilities,
    )

    baseline = target.mean()

    figure = plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=(f"{model_name} (AP = {average_precision:.3f})"),
    )

    plt.axhline(
        y=baseline,
        linestyle="--",
        label=(f"Positive-class baseline ({baseline:.3f})"),
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(f"Precision–Recall Curve — {model_name}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = save_figure(
        figure=figure,
        filename=(f"{model_type}_precision_recall_curve.png"),
    )

    print(f"Precision-recall curve saved to: {output_path}")

    plt.show()


def save_evaluation_results(
    metrics: dict[str, float],
    model_name: str,
    threshold: float,
) -> None:
    """Save individual metrics and update the comparison table."""

    metrics_path = save_metrics(
        metrics=metrics,
        model_name=model_name,
        threshold=threshold,
    )

    print(f"Metrics saved to: {metrics_path}")

    comparison_path = append_to_model_comparison(
        metrics=metrics,
        model_name=model_name,
        threshold=threshold,
    )

    print(f"Comparison table updated: {comparison_path}")


def load_evaluation_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Load production features and target for evaluation."""

    dataframe = load_data(DATA_PATH)

    return split_features_target(
        dataframe,
        production_mode=True,
    )


def load_saved_model(
    model_path: Path,
    description: str,
) -> Pipeline:
    """Load a persisted model after validating its path."""

    if not model_path.exists():
        raise FileNotFoundError(f"{description} was not found: {model_path.resolve()}")

    return joblib.load(model_path)


def evaluate_model_on_dataset(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    model_name: str,
    model_type: str,
    title: str,
) -> dict[str, float]:
    """Evaluate, report and plot a model on a supplied dataset."""

    metrics = calculate_metrics(
        model,
        X_test,
        y_test,
        threshold=threshold,
    )

    print_metrics(
        metrics,
        title=title,
    )

    save_evaluation_results(
        metrics=metrics,
        model_name=model_type,
        threshold=threshold,
    )

    plot_confusion_matrix(
        model,
        X_test,
        y_test,
        threshold=threshold,
        model_name=model_name,
        model_type=model_type,
    )

    plot_roc_curve(
        model,
        X_test,
        y_test,
        model_name=model_name,
        model_type=model_type,
    )

    plot_precision_recall_curve(
        model,
        X_test,
        y_test,
        model_name=model_name,
        model_type=model_type,
    )

    return metrics


def get_legacy_test_set() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Return the test set used by the earlier experiments."""

    features, target = load_evaluation_data()

    (
        _X_train,
        X_test,
        _y_train,
        y_test,
    ) = create_train_test_split(
        features,
        target,
    )

    return X_test, y_test


def get_reserved_test_set() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Return the untouched final test set."""

    features, target = load_evaluation_data()

    (
        _X_development,
        X_test,
        _y_development,
        y_test,
    ) = create_development_test_split(
        features,
        target,
    )

    return X_test, y_test


def evaluate_saved_model(
    model_type: ModelType,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evaluate an older production model."""

    model_path = get_model_path(model_type)

    model = load_saved_model(
        model_path,
        description="Trained model",
    )

    X_test, y_test = get_legacy_test_set()

    display_name = model_type.replace("_", " ").title()

    return evaluate_model_on_dataset(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        model_name=display_name,
        model_type=model_type,
        title=f"{display_name} Evaluation",
    )


def evaluate_tuned_random_forest(
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evaluate the previously tuned Random Forest."""

    model = load_saved_model(
        TUNED_RANDOM_FOREST_PATH,
        description="Tuned Random Forest model",
    )

    X_test, y_test = get_legacy_test_set()

    return evaluate_model_on_dataset(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        model_name="Tuned Random Forest",
        model_type="random_forest_tuned",
        title="Tuned Random Forest Evaluation",
    )


def evaluate_tuned_xgboost(
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evaluate the previously tuned XGBoost model."""

    model = load_saved_model(
        TUNED_XGBOOST_PATH,
        description="Tuned XGBoost model",
    )

    X_test, y_test = get_legacy_test_set()

    return evaluate_model_on_dataset(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        model_name="Tuned XGBoost",
        model_type="xgboost_tuned",
        title="Tuned XGBoost Evaluation",
    )


def evaluate_final_xgboost() -> dict[str, float]:
    """Evaluate the final XGBoost model on the reserved test set.

    The model was refitted on the full development set
    (train + validation).

    The decision threshold was selected using validation
    data before the final model was evaluated.
    """

    model = load_saved_model(
        FINAL_XGBOOST_PATH,
        description="Final XGBoost model",
    )

    X_test, y_test = get_reserved_test_set()

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    print("Model: XGBoost")

    print(f"Locked threshold: {FINAL_XGBOOST_THRESHOLD:.2f}")

    print(f"Reserved test samples: {len(X_test)}")

    print()

    metrics = evaluate_model_on_dataset(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=FINAL_XGBOOST_THRESHOLD,
        model_name="Final XGBoost",
        model_type="xgboost_final",
        title="Final XGBoost — Reserved Test Set",
    )

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 60)

    print("The reserved test set was used only for final evaluation.")

    print("Do not adjust the model or threshold based on these results.")

    return metrics


if __name__ == "__main__":
    evaluate_final_xgboost()
