import numpy as np
import pandas as pd
import pytest

from machine_failure.evaluate import calculate_metrics


class DummyProbabilityModel:
    """Small model that returns predefined probabilities."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.array(probabilities)

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        positive = self.probabilities
        negative = 1.0 - positive

        return np.column_stack(
            (
                negative,
                positive,
            )
        )


def create_sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )


def create_sample_target() -> pd.Series:
    return pd.Series(
        [0, 0, 1, 1],
        name="Machine failure",
    )


def test_calculate_metrics_returns_expected_keys() -> None:
    model = DummyProbabilityModel(
        [0.1, 0.4, 0.6, 0.9]
    )

    metrics = calculate_metrics(
        model,
        create_sample_features(),
        create_sample_target(),
        threshold=0.5,
    )

    assert set(metrics.keys()) == {
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
    }


def test_calculate_metrics_perfect_predictions() -> None:
    model = DummyProbabilityModel(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = calculate_metrics(
        model,
        create_sample_features(),
        create_sample_target(),
        threshold=0.5,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["f2"] == pytest.approx(1.0)


def test_calculate_metrics_respects_threshold() -> None:
    model = DummyProbabilityModel(
        [0.1, 0.4, 0.6, 0.9]
    )

    metrics_low = calculate_metrics(
        model,
        create_sample_features(),
        create_sample_target(),
        threshold=0.3,
    )

    metrics_high = calculate_metrics(
        model,
        create_sample_features(),
        create_sample_target(),
        threshold=0.8,
    )

    assert metrics_low["recall"] >= metrics_high["recall"]


def test_calculate_metrics_rejects_threshold_below_zero() -> None:
    model = DummyProbabilityModel(
        [0.1, 0.2, 0.8, 0.9]
    )

    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1",
    ):
        calculate_metrics(
            model,
            create_sample_features(),
            create_sample_target(),
            threshold=-0.1,
        )


def test_calculate_metrics_rejects_threshold_above_one() -> None:
    model = DummyProbabilityModel(
        [0.1, 0.2, 0.8, 0.9]
    )

    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1",
    ):
        calculate_metrics(
            model,
            create_sample_features(),
            create_sample_target(),
            threshold=1.1,
        )