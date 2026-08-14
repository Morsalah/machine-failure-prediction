"""Model explainability utilities."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from machine_failure.reporting import save_figure
from machine_failure.train import (
    create_train_test_split,
    load_production_features_target,
)

TUNED_RANDOM_FOREST_PATH = Path("models/random_forest_tuned_production.joblib")

RANDOM_STATE = 42
PERMUTATION_REPEATS = 10


def load_tuned_random_forest() -> Pipeline:
    """Load the tuned Random Forest model."""

    if not TUNED_RANDOM_FOREST_PATH.exists():
        raise FileNotFoundError(
            "Tuned Random Forest model was not found: "
            f"{TUNED_RANDOM_FOREST_PATH.resolve()}"
        )

    return joblib.load(TUNED_RANDOM_FOREST_PATH)


def get_explainability_test_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Return the held-out test data used for explainability analysis."""

    features, target = load_production_features_target()

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


def get_feature_importance(
    model: Pipeline,
) -> pd.DataFrame:
    """Extract feature importances from a fitted Random Forest pipeline."""

    preprocessor = model.named_steps["preprocessor"]

    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    return importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
) -> None:
    """Plot and save impurity-based feature importance values."""

    figure = plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Feature Importance")

    plt.ylabel("Feature")

    plt.title("Tuned Random Forest — Feature Importance")

    plt.tight_layout()

    output_path = save_figure(
        figure=figure,
        filename=("random_forest_tuned_feature_importance.png"),
    )

    print(f"Feature importance plot saved to: {output_path}")

    plt.show()


def calculate_permutation_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Calculate permutation importance on held-out test data."""

    result = permutation_importance(
        model,
        features,
        target,
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": (result.importances_mean),
            "importance_std": (result.importances_std),
        }
    )

    return importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def plot_permutation_importance(
    importance_df: pd.DataFrame,
) -> None:
    """Plot and save permutation-importance values."""

    figure = plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance_mean"],
        xerr=importance_df["importance_std"],
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Decrease in Average Precision")

    plt.ylabel("Feature")

    plt.title("Tuned Random Forest — Permutation Importance")

    plt.tight_layout()

    output_path = save_figure(
        figure=figure,
        filename=("random_forest_tuned_permutation_importance.png"),
    )

    print(f"Permutation importance plot saved to: {output_path}")

    plt.show()


def print_importance_table(
    title: str,
    importance_df: pd.DataFrame,
) -> None:
    """Print an explainability result table."""

    print("=" * 60)
    print(title)
    print("=" * 60)

    print(importance_df.to_string(index=False))


def run_feature_importance_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run feature-importance and permutation-importance analysis."""

    model = load_tuned_random_forest()

    X_test, y_test = get_explainability_test_data()

    feature_importance_df = get_feature_importance(model)

    print_importance_table(
        title=("TUNED RANDOM FOREST — FEATURE IMPORTANCE"),
        importance_df=feature_importance_df,
    )

    plot_feature_importance(feature_importance_df)

    permutation_df = calculate_permutation_importance(
        model,
        X_test,
        y_test,
    )

    print()

    print_importance_table(
        title=("TUNED RANDOM FOREST — PERMUTATION IMPORTANCE"),
        importance_df=permutation_df,
    )

    plot_permutation_importance(permutation_df)

    return (
        feature_importance_df,
        permutation_df,
    )


if __name__ == "__main__":
    run_feature_importance_analysis()
