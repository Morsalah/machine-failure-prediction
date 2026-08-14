"""SHAP analysis for the tuned Random Forest model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from machine_failure.reporting import save_figure
from machine_failure.train import (
    create_train_test_split,
    load_production_features_target,
)

TUNED_RANDOM_FOREST_PATH = Path("models/random_forest_tuned_production.joblib")

SHAP_SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def load_tuned_random_forest() -> Pipeline:
    """Load the tuned Random Forest model."""

    if not TUNED_RANDOM_FOREST_PATH.exists():
        raise FileNotFoundError(
            "Tuned Random Forest model was not found: "
            f"{TUNED_RANDOM_FOREST_PATH.resolve()}"
        )

    return joblib.load(TUNED_RANDOM_FOREST_PATH)


def get_shap_test_sample() -> pd.DataFrame:
    """Return a reproducible sample from the held-out test set."""

    features, target = load_production_features_target()

    (
        _X_train,
        X_test,
        _y_train,
        _y_test,
    ) = create_train_test_split(
        features,
        target,
    )

    return X_test.sample(
        n=min(
            SHAP_SAMPLE_SIZE,
            len(X_test),
        ),
        random_state=RANDOM_STATE,
    )


def transform_shap_sample(
    model: Pipeline,
    X_sample: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """Transform the SHAP sample exactly as the model sees it."""

    preprocessor = model.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X_sample)

    feature_names = preprocessor.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X_sample.index,
    )

    return (
        X_transformed_df,
        feature_names,
    )


def calculate_failure_shap_values(
    model: Pipeline,
    X_transformed_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    shap.Explanation,
]:
    """Calculate SHAP values for the machine-failure class."""

    classifier = model.named_steps["classifier"]

    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer(X_transformed_df)

    if shap_values.values.ndim == 3:
        failure_values = shap_values.values[:, :, 1]
    else:
        failure_values = shap_values.values

    return (
        failure_values,
        shap_values,
    )


def plot_shap_summary(
    failure_values: np.ndarray,
    X_transformed_df: pd.DataFrame,
) -> None:
    """Create and save the SHAP summary plot."""

    plt.figure()

    shap.summary_plot(
        failure_values,
        X_transformed_df,
        show=False,
    )

    plt.title("Tuned Random Forest — SHAP Summary")

    plt.tight_layout()

    output_path = save_figure(
        figure=plt.gcf(),
        filename=("random_forest_tuned_shap_summary.png"),
    )

    print(f"SHAP summary plot saved to: {output_path}")

    plt.show()


def plot_shap_dependence(
    X_sample: pd.DataFrame,
    feature_names: np.ndarray,
    failure_values: np.ndarray,
    original_feature: str,
    transformed_feature: str,
    display_name: str,
    filename: str,
) -> None:
    """Create and save a SHAP dependence plot for one feature."""

    feature_index = list(feature_names).index(transformed_feature)

    figure = plt.figure(figsize=(8, 6))

    plt.scatter(
        X_sample[original_feature],
        failure_values[:, feature_index],
        alpha=0.5,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel(display_name)

    plt.ylabel("SHAP Value (Impact on Failure Prediction)")

    plt.title(f"Tuned Random Forest — SHAP Dependence: {display_name}")

    plt.tight_layout()

    output_path = save_figure(
        figure=figure,
        filename=filename,
    )

    print(f"{display_name} SHAP dependence plot saved to: {output_path}")

    plt.show()


def plot_torque_speed_interaction(
    X_sample: pd.DataFrame,
    feature_names: np.ndarray,
    failure_values: np.ndarray,
) -> None:
    """Create a Torque vs Rotational Speed SHAP interaction view."""

    torque_feature = "numeric__Torque [Nm]"

    torque_index = list(feature_names).index(torque_feature)

    figure = plt.figure(figsize=(9, 6))

    scatter = plt.scatter(
        X_sample["Torque [Nm]"],
        failure_values[
            :,
            torque_index,
        ],
        c=X_sample["Rotational speed [rpm]"],
        alpha=0.6,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel("Torque [Nm]")

    plt.ylabel("Torque SHAP Value (Impact on Failure Prediction)")

    plt.title("SHAP Interaction View — Torque vs Rotational Speed")

    colorbar = plt.colorbar(scatter)

    colorbar.set_label("Rotational Speed [rpm]")

    plt.tight_layout()

    output_path = save_figure(
        figure=figure,
        filename=("random_forest_tuned_shap_interaction_torque_rotational_speed.png"),
    )

    print(f"Torque / rotational-speed interaction plot saved to: {output_path}")

    plt.show()


def run_shap_analysis() -> None:
    """Analyze the tuned Random Forest using SHAP."""

    model = load_tuned_random_forest()

    X_sample = get_shap_test_sample()

    (
        X_transformed_df,
        feature_names,
    ) = transform_shap_sample(
        model,
        X_sample,
    )

    (
        failure_values,
        shap_values,
    ) = calculate_failure_shap_values(
        model,
        X_transformed_df,
    )

    print("=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)

    print(f"Samples analyzed: {len(X_sample)}")

    print(f"Features analyzed: {len(feature_names)}")

    print(f"SHAP values shape: {shap_values.values.shape}")

    plot_shap_summary(
        failure_values,
        X_transformed_df,
    )

    plot_shap_dependence(
        X_sample=X_sample,
        feature_names=feature_names,
        failure_values=failure_values,
        original_feature="Torque [Nm]",
        transformed_feature=("numeric__Torque [Nm]"),
        display_name="Torque [Nm]",
        filename=("random_forest_tuned_shap_dependence_torque.png"),
    )

    plot_shap_dependence(
        X_sample=X_sample,
        feature_names=feature_names,
        failure_values=failure_values,
        original_feature=("Rotational speed [rpm]"),
        transformed_feature=("numeric__Rotational speed [rpm]"),
        display_name=("Rotational Speed [rpm]"),
        filename=("random_forest_tuned_shap_dependence_rotational_speed.png"),
    )

    plot_torque_speed_interaction(
        X_sample,
        feature_names,
        failure_values,
    )


if __name__ == "__main__":
    run_shap_analysis()
