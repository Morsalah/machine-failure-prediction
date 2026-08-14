"""Model pipeline construction utilities."""

from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from machine_failure.preprocessing import build_preprocessor

RANDOM_STATE = 42

ModelType = Literal[
    "logistic_regression",
    "random_forest",
    "xgboost",
]


def build_model_pipeline(
    model_type: ModelType = "logistic_regression",
) -> Pipeline:
    """Build a complete preprocessing and classification pipeline."""

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )

    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_type == "xgboost":
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


if __name__ == "__main__":
    from machine_failure.data import load_data
    from machine_failure.features import split_features_target

    dataframe = load_data("data/raw/train.csv")

    features, target = split_features_target(
        dataframe,
        production_mode=True,
    )

    pipeline = build_model_pipeline(
        model_type="xgboost",
    )

    pipeline.fit(
        features.head(5_000),
        target.head(5_000),
    )

    predictions = pipeline.predict(
        features.head(5),
    )

    probabilities = pipeline.predict_proba(
        features.head(5),
    )[:, 1]

    print("Pipeline:")
    print(pipeline)

    print("\nPredictions:")
    print(predictions)

    print("\nFailure probabilities:")
    print(probabilities)
