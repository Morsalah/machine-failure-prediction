import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from machine_failure.preprocessing import build_preprocessor


def create_sample_features() -> pd.DataFrame:
    """Create a small representative feature dataframe for testing."""
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [
                298.1,
                299.2,
                300.3,
                301.0,
            ],
            "Process temperature [K]": [
                308.5,
                309.4,
                310.2,
                311.0,
            ],
            "Rotational speed [rpm]": [
                1500,
                1600,
                1700,
                1550,
            ],
            "Torque [Nm]": [
                40.0,
                35.0,
                30.0,
                37.0,
            ],
            "Tool wear [min]": [
                10,
                20,
                30,
                40,
            ],
        }
    )


def test_build_preprocessor_returns_column_transformer() -> None:
    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_returns_expected_shape() -> None:
    features = create_sample_features()
    preprocessor = build_preprocessor()

    transformed_features = preprocessor.fit_transform(features)

    # Five numeric columns and three one-hot encoded Type columns.
    assert transformed_features.shape == (4, 8)


def test_preprocessor_output_contains_no_missing_values() -> None:
    features = create_sample_features()
    preprocessor = build_preprocessor()

    transformed_features = preprocessor.fit_transform(features)

    assert not np.isnan(transformed_features).any()


def test_preprocessor_handles_missing_values() -> None:
    features = create_sample_features()

    features.loc[0, "Torque [Nm]"] = np.nan

    # Replace one of the duplicated "L" values, so L, M and H
    # are still present during fitting.
    features.loc[3, "Type"] = np.nan

    preprocessor = build_preprocessor()
    transformed_features = preprocessor.fit_transform(features)

    assert transformed_features.shape == (4, 8)
    assert not np.isnan(transformed_features).any()


def test_preprocessor_handles_unknown_category() -> None:
    training_features = create_sample_features()
    preprocessor = build_preprocessor()

    preprocessor.fit(training_features)

    new_features = pd.DataFrame(
        {
            "Type": ["UNKNOWN"],
            "Air temperature [K]": [301.0],
            "Process temperature [K]": [311.0],
            "Rotational speed [rpm]": [1550],
            "Torque [Nm]": [37.0],
            "Tool wear [min]": [45],
        }
    )

    transformed_features = preprocessor.transform(new_features)

    assert transformed_features.shape == (1, 8)
    assert not np.isnan(transformed_features).any()
