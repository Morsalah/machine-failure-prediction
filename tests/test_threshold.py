import numpy as np
import pandas as pd
import pytest

from machine_failure.threshold import (
    evaluate_thresholds,
    select_best_threshold,
)


class DummyProbabilityModel:
    """Small test model that returns predefined probabilities."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.array(probabilities)

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Return probabilities in scikit-learn predict_proba format."""

        positive_probabilities = self.probabilities

        negative_probabilities = (
            1.0 - positive_probabilities
        )

        return np.column_stack(
            (
                negative_probabilities,
                positive_probabilities,
            )
        )


def create_sample_features() -> pd.DataFrame:
    """Create simple dummy features for threshold tests."""

    return pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )


def create_sample_target() -> pd.Series:
    """Create a simple binary target."""

    return pd.Series(
        [0, 0, 1, 1],
        name="Machine failure",
    )


def test_evaluate_thresholds_returns_expected_columns() -> None:
    features = create_sample_features()
    target = create_sample_target()

    model = DummyProbabilityModel(
        probabilities=[
            0.10,
            0.40,
            0.60,
            0.90,
        ]
    )

    results = evaluate_thresholds(
        model,
        features,
        target,
        thresholds=np.array(
            [0.30, 0.50, 0.70]
        ),
    )

    expected_columns = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
    }

    assert set(results.columns) == expected_columns


def test_evaluate_thresholds_returns_one_row_per_threshold() -> None:
    features = create_sample_features()
    target = create_sample_target()

    model = DummyProbabilityModel(
        probabilities=[
            0.10,
            0.40,
            0.60,
            0.90,
        ]
    )

    thresholds = np.array(
        [0.30, 0.50, 0.70]
    )

    results = evaluate_thresholds(
        model,
        features,
        target,
        thresholds=thresholds,
    )

    assert len(results) == len(thresholds)


def test_evaluate_thresholds_calculates_expected_metrics() -> None:
    features = create_sample_features()
    target = create_sample_target()

    model = DummyProbabilityModel(
        probabilities=[
            0.10,
            0.40,
            0.60,
            0.90,
        ]
    )

    results = evaluate_thresholds(
        model,
        features,
        target,
        thresholds=np.array(
            [0.50]
        ),
    )

    row = results.iloc[0]

    assert row["threshold"] == pytest.approx(0.50)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    assert row["f2"] == pytest.approx(1.0)


def test_select_best_threshold_by_f1() -> None:
    results = pd.DataFrame(
        {
            "threshold": [0.10, 0.20, 0.30],
            "precision": [0.20, 0.50, 0.60],
            "recall": [0.90, 0.70, 0.40],
            "f1": [0.32, 0.58, 0.48],
            "f2": [0.55, 0.65, 0.43],
        }
    )

    best = select_best_threshold(
        results,
        metric="f1",
    )

    assert best["threshold"] == pytest.approx(0.20)
    assert best["f1"] == pytest.approx(0.58)


def test_select_best_threshold_by_f2() -> None:
    results = pd.DataFrame(
        {
            "threshold": [0.10, 0.20, 0.30],
            "precision": [0.20, 0.50, 0.60],
            "recall": [0.90, 0.70, 0.40],
            "f1": [0.32, 0.58, 0.48],
            "f2": [0.55, 0.65, 0.43],
        }
    )

    best = select_best_threshold(
        results,
        metric="f2",
    )

    assert best["threshold"] == pytest.approx(0.20)
    assert best["f2"] == pytest.approx(0.65)


def test_select_best_threshold_raises_error_for_unknown_metric() -> None:
    results = pd.DataFrame(
        {
            "threshold": [0.10, 0.20],
            "f1": [0.40, 0.50],
            "f2": [0.45, 0.55],
        }
    )

    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        select_best_threshold(
            results,
            metric="unknown_metric",
        )