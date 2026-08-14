"""Data loading utilities for the machine-failure dataset."""

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

DEFAULT_DATA_PATH = Path("data/raw/train.csv")


REQUIRED_COLUMNS = {
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
}


def load_data(
    path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """Load and validate the machine-failure dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    try:
        dataframe = pd.read_csv(path)
    except EmptyDataError as exc:
        raise ValueError("Dataset is empty") from exc

    if dataframe.empty:
        raise ValueError("Dataset is empty")

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    return dataframe


if __name__ == "__main__":
    dataframe = load_data()

    print(dataframe.head())

    print(f"\nShape: {dataframe.shape}")
