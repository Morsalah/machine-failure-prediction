import pandas as pd

TARGET_COLUMN = "Machine failure"

BASE_COLUMNS_TO_DROP = [
    "id",
    "Product ID",
]

LEAKAGE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def split_features_target(
    dataframe: pd.DataFrame,
    production_mode: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into model features and target values.

    Parameters
    ----------
    dataframe:
        The complete machine-failure dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X contains the model features.
        y contains the target column.
    """
    columns_to_drop = BASE_COLUMNS_TO_DROP.copy()

    if production_mode:
        columns_to_drop.extend(LEAKAGE_COLUMNS)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    features = dataframe.drop(
        columns=[TARGET_COLUMN, *columns_to_drop],
        errors="ignore",
    )

    target = dataframe[TARGET_COLUMN].copy()

    return features, target


if __name__ == "__main__":
    from machine_failure.data import load_data

    df = load_data()

    X, y = split_features_target(df, production_mode=False)

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())
