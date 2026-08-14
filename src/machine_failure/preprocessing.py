"""Preprocessing utilities for the machine-failure project."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = [
    "Type",
]


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline for model input features.

    Returns
    -------
    ColumnTransformer
        A transformer that scales numeric features and one-hot encodes
        categorical features.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    from machine_failure.data import load_data
    from machine_failure.features import split_features_target

    dataframe = load_data("data/raw/train.csv")

    features, _ = split_features_target(
        dataframe,
        production_mode=True,
    )

    preprocessor = build_preprocessor()

    transformed_features = preprocessor.fit_transform(features)

    print("Original shape:", features.shape)
    print("Transformed shape:", transformed_features.shape)

    print("\nTransformed feature names:")
    print(preprocessor.get_feature_names_out())
