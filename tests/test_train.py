import pandas as pd
import pytest

from machine_failure.train import (
    TEST_SIZE,
    VALIDATION_SIZE_WITHIN_DEVELOPMENT,
    create_development_test_split,
    create_train_test_split,
    create_train_validation_test_split,
    get_final_model_path,
    get_model_path,
    get_validation_model_path,
)


def create_sample_dataset(
    size: int = 1000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a reproducible imbalanced binary-classification dataset."""

    features = pd.DataFrame(
        {
            "feature_1": range(size),
            "feature_2": range(size, size * 2),
        }
    )

    # 10% positive class.
    target = pd.Series(
        [0] * int(size * 0.90)
        + [1] * int(size * 0.10),
        name="Machine failure",
    )

    return features, target


def test_create_train_test_split_sizes() -> None:
    features, target = create_sample_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_train_test_split(
        features,
        target,
    )

    expected_test_size = int(
        len(features) * TEST_SIZE
    )

    assert len(X_test) == expected_test_size
    assert len(y_test) == expected_test_size

    assert len(X_train) + len(X_test) == len(features)
    assert len(y_train) + len(y_test) == len(target)


def test_create_train_test_split_is_stratified() -> None:
    features, target = create_sample_dataset()

    (
        _,
        _,
        y_train,
        y_test,
    ) = create_train_test_split(
        features,
        target,
    )

    original_positive_rate = target.mean()
    train_positive_rate = y_train.mean()
    test_positive_rate = y_test.mean()

    assert train_positive_rate == pytest.approx(
        original_positive_rate,
        abs=0.01,
    )

    assert test_positive_rate == pytest.approx(
        original_positive_rate,
        abs=0.01,
    )


def test_train_validation_test_split_sizes() -> None:
    features, target = create_sample_dataset()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_train_validation_test_split(
        features,
        target,
    )

    total_size = len(features)

    expected_test_size = int(
        total_size * TEST_SIZE
    )

    development_size = (
        total_size - expected_test_size
    )

    expected_validation_size = int(
        development_size
        * VALIDATION_SIZE_WITHIN_DEVELOPMENT
    )

    expected_train_size = (
        total_size
        - expected_test_size
        - expected_validation_size
    )

    assert len(X_test) == expected_test_size
    assert len(X_validation) == expected_validation_size
    assert len(X_train) == expected_train_size

    assert len(y_test) == len(X_test)
    assert len(y_validation) == len(X_validation)
    assert len(y_train) == len(X_train)


def test_train_validation_test_split_is_stratified() -> None:
    features, target = create_sample_dataset()

    (
        _,
        _,
        _,
        y_train,
        y_validation,
        y_test,
    ) = create_train_validation_test_split(
        features,
        target,
    )

    original_positive_rate = target.mean()

    assert y_train.mean() == pytest.approx(
        original_positive_rate,
        abs=0.01,
    )

    assert y_validation.mean() == pytest.approx(
        original_positive_rate,
        abs=0.01,
    )

    assert y_test.mean() == pytest.approx(
        original_positive_rate,
        abs=0.01,
    )


def test_train_validation_test_sets_do_not_overlap() -> None:
    features, target = create_sample_dataset()

    (
        X_train,
        X_validation,
        X_test,
        _,
        _,
        _,
    ) = create_train_validation_test_split(
        features,
        target,
    )

    train_indices = set(X_train.index)
    validation_indices = set(X_validation.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)


def test_development_test_split_sizes() -> None:
    features, target = create_sample_dataset()

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = create_development_test_split(
        features,
        target,
    )

    expected_test_size = int(
        len(features) * TEST_SIZE
    )

    assert len(X_test) == expected_test_size

    assert (
        len(X_development)
        + len(X_test)
        == len(features)
    )

    assert len(y_development) == len(X_development)
    assert len(y_test) == len(X_test)


def test_development_test_split_is_stratified() -> None:
    features, target = create_sample_dataset()

    (
        _,
        _,
        y_development,
        y_test,
    ) = create_development_test_split(
        features,
        target,
    )

    original_positive_rate = target.mean()

    assert y_development.mean() == pytest.approx(
        original_positive_rate,
        abs=0.01,
    )

    assert y_test.mean() == pytest.approx(
        original_positive_rate,
        abs=0.01,
    )


def test_get_model_path() -> None:
    path = get_model_path(
        "xgboost"
    )

    assert path.name == (
        "xgboost_production.joblib"
    )

    assert path.parent.name == "models"


def test_get_validation_model_path() -> None:
    path = get_validation_model_path(
        "xgboost"
    )

    assert path.name == (
        "xgboost_validation.joblib"
    )

    assert path.parent.name == "models"


def test_get_final_model_path() -> None:
    path = get_final_model_path(
        "xgboost"
    )

    assert path.name == (
        "xgboost_final.joblib"
    )

    assert path.parent.name == "models"