import pandas as pd
import pytest

from machine_failure.features import split_features_target


def create_sample_dataframe() -> pd.DataFrame:
    """Create a small machine-failure dataset for feature tests."""

    return pd.DataFrame(
        {
            "id": [1, 2],
            "Product ID": ["A1", "A2"],
            "Type": ["L", "M"],
            "Air temperature [K]": [298.1, 299.2],
            "TWF": [0, 1],
            "HDF": [0, 0],
            "PWF": [0, 1],
            "OSF": [0, 0],
            "RNF": [0, 0],
            "Machine failure": [0, 1],
        }
    )


def test_split_features_target() -> None:
    dataframe = create_sample_dataframe()

    features, target = split_features_target(
        dataframe
    )

    assert "Machine failure" not in features.columns
    assert "id" not in features.columns
    assert "Product ID" not in features.columns

    assert "Type" in features.columns
    assert "Air temperature [K]" in features.columns

    assert len(features) == 2
    assert len(target) == 2
    assert target.tolist() == [0, 1]


def test_production_mode_removes_leakage_columns() -> None:
    dataframe = create_sample_dataframe()

    features, _ = split_features_target(
        dataframe,
        production_mode=True,
    )

    leakage_columns = [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    for column in leakage_columns:
        assert column not in features.columns


def test_non_production_mode_keeps_leakage_columns() -> None:
    dataframe = create_sample_dataframe()

    features, _ = split_features_target(
        dataframe,
        production_mode=False,
    )

    leakage_columns = [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    for column in leakage_columns:
        assert column in features.columns


def test_split_features_target_raises_error_when_target_missing() -> None:
    dataframe = create_sample_dataframe()

    dataframe = dataframe.drop(
        columns=["Machine failure"]
    )

    with pytest.raises(
        ValueError,
        match="Target column 'Machine failure' is missing",
    ):
        split_features_target(
            dataframe
        )


def test_target_is_returned_as_independent_copy() -> None:
    dataframe = create_sample_dataframe()

    _, target = split_features_target(
        dataframe
    )

    target.iloc[0] = 1

    assert dataframe.loc[
        0,
        "Machine failure",
    ] == 0