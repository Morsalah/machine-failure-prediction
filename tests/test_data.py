from pathlib import Path

import pandas as pd
import pytest

from machine_failure.data import REQUIRED_COLUMNS, load_data


def test_load_data_returns_non_empty_dataframe() -> None:
    dataframe = load_data()

    assert isinstance(dataframe, pd.DataFrame)
    assert not dataframe.empty


def test_load_data_contains_required_columns() -> None:
    dataframe = load_data()

    assert REQUIRED_COLUMNS.issubset(dataframe.columns)


def test_load_data_raises_error_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_file)


def test_load_data_raises_error_when_dataset_is_empty(
    tmp_path: Path,
) -> None:
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset is empty"):
        load_data(empty_file)


def test_load_data_raises_error_when_columns_are_missing(
    tmp_path: Path,
) -> None:
    invalid_file = tmp_path / "invalid.csv"

    pd.DataFrame(
        {
            "id": [1],
            "Machine failure": [0],
        }
    ).to_csv(invalid_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_data(invalid_file)
