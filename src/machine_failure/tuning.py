"""Hyperparameter tuning utilities for machine-failure models."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from machine_failure.model import (
    RANDOM_STATE,
    ModelType,
    build_model_pipeline,
)
from machine_failure.train import (
    create_train_test_split,
    load_production_features_target,
)

RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [5, 10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}


XGBOOST_PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [200, 300, 500, 700],
    "classifier__max_depth": [3, 4, 5, 6, 8],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 3, 5],
}


RANDOM_FOREST_TUNED_MODEL_PATH = Path("models/random_forest_tuned_production.joblib")

XGBOOST_TUNED_MODEL_PATH = Path("models/xgboost_tuned_production.joblib")


def get_tuning_training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """Return the training split used for hyperparameter tuning."""

    features, target = load_production_features_target()

    (
        X_train,
        _X_test,
        y_train,
        _y_test,
    ) = create_train_test_split(
        features,
        target,
    )

    return X_train, y_train


def run_randomized_search(
    model_type: ModelType,
    param_distributions: dict[str, list[Any]],
    n_iter: int,
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV for a selected model."""

    X_train, y_train = get_tuning_training_data()

    pipeline = build_model_pipeline(
        model_type=model_type,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )

    display_name = model_type.replace("_", " ").title()

    print(f"Starting {display_name} hyperparameter tuning...")

    search.fit(
        X_train,
        y_train,
    )

    return search


def save_tuned_model(
    search: RandomizedSearchCV,
    output_path: Path,
    model_name: str,
) -> None:
    """Save the best model found during hyperparameter tuning."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        search.best_estimator_,
        output_path,
    )

    print(f"\nTuned {model_name} model saved to: {output_path}")


def print_tuning_results(
    search: RandomizedSearchCV,
) -> None:
    """Print the best hyperparameters and cross-validation score."""

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest cross-validation score:")
    print(search.best_score_)


def tune_model(
    model_type: ModelType,
    param_distributions: dict[str, list[Any]],
    n_iter: int,
    output_path: Path,
    display_name: str,
) -> Pipeline:
    """Tune, report and persist a selected model."""

    search = run_randomized_search(
        model_type=model_type,
        param_distributions=param_distributions,
        n_iter=n_iter,
    )

    print_tuning_results(search)

    save_tuned_model(
        search=search,
        output_path=output_path,
        model_name=display_name,
    )

    return search.best_estimator_


def tune_random_forest() -> Pipeline:
    """Tune Random Forest hyperparameters using cross-validation."""

    return tune_model(
        model_type="random_forest",
        param_distributions=RANDOM_FOREST_PARAM_DISTRIBUTIONS,
        n_iter=20,
        output_path=RANDOM_FOREST_TUNED_MODEL_PATH,
        display_name="Random Forest",
    )


def tune_xgboost() -> Pipeline:
    """Tune XGBoost hyperparameters using cross-validation."""

    return tune_model(
        model_type="xgboost",
        param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
        n_iter=25,
        output_path=XGBOOST_TUNED_MODEL_PATH,
        display_name="XGBoost",
    )


if __name__ == "__main__":
    tune_xgboost()
