import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from machine_failure.model import build_model_pipeline


def create_sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small binary-classification dataset for testing."""

    features = pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L", "M", "H"],
            "Air temperature [K]": [
                298.1,
                299.2,
                300.3,
                301.0,
                302.1,
                303.0,
            ],
            "Process temperature [K]": [
                308.5,
                309.4,
                310.2,
                311.0,
                312.0,
                313.0,
            ],
            "Rotational speed [rpm]": [
                1500,
                1600,
                1700,
                1550,
                1650,
                1750,
            ],
            "Torque [Nm]": [
                40.0,
                35.0,
                30.0,
                42.0,
                34.0,
                28.0,
            ],
            "Tool wear [min]": [
                10,
                20,
                30,
                180,
                200,
                220,
            ],
        }
    )

    target = pd.Series(
        [0, 0, 0, 1, 1, 1],
        name="Machine failure",
    )

    return features, target


def test_build_model_pipeline_returns_pipeline() -> None:
    pipeline = build_model_pipeline()

    assert isinstance(
        pipeline,
        Pipeline,
    )


def test_pipeline_contains_expected_steps() -> None:
    pipeline = build_model_pipeline()

    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_default_classifier_is_logistic_regression() -> None:
    pipeline = build_model_pipeline()

    classifier = pipeline.named_steps["classifier"]

    assert isinstance(
        classifier,
        LogisticRegression,
    )

    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 42


def test_random_forest_classifier_is_created() -> None:
    pipeline = build_model_pipeline(
        model_type="random_forest",
    )

    classifier = pipeline.named_steps["classifier"]

    assert isinstance(
        classifier,
        RandomForestClassifier,
    )

    assert classifier.n_estimators == 300
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 42


def test_xgboost_classifier_is_created() -> None:
    pipeline = build_model_pipeline(
        model_type="xgboost",
    )

    classifier = pipeline.named_steps["classifier"]

    assert isinstance(
        classifier,
        XGBClassifier,
    )

    assert classifier.n_estimators == 300
    assert classifier.max_depth == 6
    assert classifier.learning_rate == pytest.approx(0.05)
    assert classifier.random_state == 42


def test_model_pipeline_can_fit_and_predict() -> None:
    features, target = create_sample_training_data()

    pipeline = build_model_pipeline()

    pipeline.fit(
        features,
        target,
    )

    predictions = pipeline.predict(
        features
    )

    assert predictions.shape == (6,)
    assert set(
        np.unique(predictions)
    ).issubset(
        {0, 1}
    )


def test_model_pipeline_returns_probabilities() -> None:
    features, target = create_sample_training_data()

    pipeline = build_model_pipeline()

    pipeline.fit(
        features,
        target,
    )

    probabilities = pipeline.predict_proba(
        features
    )

    assert probabilities.shape == (6, 2)

    assert np.all(
        probabilities >= 0
    )

    assert np.all(
        probabilities <= 1
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_unsupported_model_type_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported model type",
    ):
        build_model_pipeline(
            model_type="unsupported_model",  # type: ignore[arg-type]
        )