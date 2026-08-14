"""Training entry point for machine-failure classification models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from machine_failure.data import load_data
from machine_failure.features import split_features_target
from machine_failure.model import (
    RANDOM_STATE,
    ModelType,
    build_model_pipeline,
)

DATA_PATH = Path("data/raw/train.csv")
MODELS_DIR = Path("models")

TEST_SIZE = 0.20

# 20% of the remaining 80% development data.
# Final split:
# 64% train / 16% validation / 20% test.
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.20

# Final decision threshold selected using validation data only.
FINAL_XGBOOST_THRESHOLD = 0.09


def load_production_features_target() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and return production features and target."""

    dataframe = load_data(DATA_PATH)

    return split_features_target(
        dataframe,
        production_mode=True,
    )


def create_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create the original reproducible stratified train-test split.

    Kept for compatibility with previous experiments.
    """

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def create_train_validation_test_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create reproducible train-validation-test splits.

    Approximate final proportions:

    - Train:      64%
    - Validation: 16%
    - Test:       20%
    """

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE_WITHIN_DEVELOPMENT,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def create_development_test_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create development and final-test sets.

    The development set contains 80% of the data and corresponds
    to the combined train + validation data.

    The remaining 20% is reserved for final evaluation.
    """

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def get_model_path(
    model_type: ModelType,
) -> Path:
    """Return the output path for an older production model."""

    return MODELS_DIR / f"{model_type}_production.joblib"


def get_validation_model_path(
    model_type: ModelType,
) -> Path:
    """Return the output path for a validation-stage model."""

    return MODELS_DIR / f"{model_type}_validation.joblib"


def get_final_model_path(
    model_type: ModelType,
) -> Path:
    """Return the output path for the final selected model."""

    return MODELS_DIR / f"{model_type}_final.joblib"


def save_model(
    pipeline: Pipeline,
    model_path: Path,
) -> None:
    """Persist a fitted model pipeline."""

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )


def fit_model(
    model_type: ModelType,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Build and fit a model pipeline."""

    pipeline = build_model_pipeline(
        model_type=model_type,
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


def train_model(
    model_type: ModelType = "logistic_regression",
) -> Pipeline:
    """Train a model using the original train-test workflow.

    Kept only for compatibility with previous experiments.
    """

    features, target = load_production_features_target()

    (
        X_train,
        X_test,
        y_train,
        _y_test,
    ) = create_train_test_split(
        features,
        target,
    )

    print(f"Training model: {model_type}")

    pipeline = fit_model(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
    )

    model_path = get_model_path(model_type)

    save_model(
        pipeline,
        model_path,
    )

    print("\nTraining completed successfully.")
    print(f"Model type: {model_type}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Model saved to: {model_path}")

    return pipeline


def train_validation_model(
    model_type: ModelType = "xgboost",
) -> Pipeline:
    """Train only on train data for validation-based model selection."""

    features, target = load_production_features_target()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        _y_validation,
        _y_test,
    ) = create_train_validation_test_split(
        features,
        target,
    )

    print("=" * 60)
    print("VALIDATION-STAGE MODEL TRAINING")
    print("=" * 60)

    print(f"Training validation model: {model_type}")

    pipeline = fit_model(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
    )

    model_path = get_validation_model_path(model_type)

    save_model(
        pipeline,
        model_path,
    )

    print("\nValidation model training completed.")
    print(f"Model type: {model_type}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Reserved test samples: {len(X_test)}")
    print(f"Model saved to: {model_path}")

    return pipeline


def train_final_model(
    model_type: ModelType = "xgboost",
) -> Pipeline:
    """Refit the final selected model on all development data.

    The model is trained on the combined train + validation set
    (80% of the full dataset).

    The final 20% test set remains completely untouched.
    """

    features, target = load_production_features_target()

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = create_development_test_split(
        features,
        target,
    )

    print("=" * 60)
    print("FINAL MODEL REFIT")
    print("=" * 60)

    print(f"Final model: {model_type}")
    print(f"Selected threshold: {FINAL_XGBOOST_THRESHOLD:.2f}")
    print(f"Development samples: {len(X_development)}")
    print(f"Reserved test samples: {len(X_test)}")

    print("\nTraining final model on Train + Validation...")

    pipeline = fit_model(
        model_type=model_type,
        X_train=X_development,
        y_train=y_development,
    )

    model_path = get_final_model_path(model_type)

    save_model(
        pipeline,
        model_path,
    )

    print("\nFinal model training completed successfully.")

    print(f"Final model saved to: {model_path}")

    print("\nDevelopment target distribution:")

    print(y_development.value_counts(normalize=True).sort_index())

    print("\nReserved test target distribution:")

    print(y_test.value_counts(normalize=True).sort_index())

    print(
        "\nIMPORTANT:"
        "\nThe reserved test set has NOT been used "
        "for fitting or threshold selection."
    )

    return pipeline


if __name__ == "__main__":
    train_final_model(
        model_type="xgboost",
    )
