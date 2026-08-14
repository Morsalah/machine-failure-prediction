"""Exploratory data analysis utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from machine_failure.data import load_data

TARGET_COLUMN = "Machine failure"

NUMERIC_COLUMNS_TO_SKIP = {
    "id",
    TARGET_COLUMN,
}


def print_section_title(
    title: str,
) -> None:
    """Print a formatted EDA section title."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def dataset_summary(
    dataframe: pd.DataFrame,
) -> None:
    """Print a high-level summary of the dataset."""

    print_section_title("DATASET SHAPE")

    print(dataframe.shape)

    print_section_title("COLUMN TYPES")

    print(dataframe.dtypes)

    print_section_title("MISSING VALUES")

    print(dataframe.isnull().sum())

    print_section_title("DUPLICATED ROWS")

    print(dataframe.duplicated().sum())

    print_section_title("DESCRIPTIVE STATISTICS")

    print(dataframe.describe(include="all"))


def validate_target_column(
    dataframe: pd.DataFrame,
) -> None:
    """Ensure that the target column exists."""

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataframe."
        )


def plot_target_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """Plot the distribution of the machine-failure target."""

    validate_target_column(dataframe)

    target_counts = dataframe[TARGET_COLUMN].value_counts().sort_index()

    print_section_title("TARGET DISTRIBUTION")

    print(target_counts)

    failure_rate = dataframe[TARGET_COLUMN].mean() * 100

    print(f"\nFailure rate: {failure_rate:.2f}%")

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=dataframe,
        x=TARGET_COLUMN,
    )

    plt.title("Machine Failure Distribution")

    plt.xlabel("Machine Failure")

    plt.ylabel("Number of Samples")

    plt.xticks(
        ticks=[0, 1],
        labels=[
            "No Failure",
            "Failure",
        ],
    )

    plt.tight_layout()
    plt.show()


def get_numeric_analysis_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return numeric columns suitable for distribution analysis."""

    numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()

    return [
        column for column in numeric_columns if column not in NUMERIC_COLUMNS_TO_SKIP
    ]


def plot_numeric_distributions(
    dataframe: pd.DataFrame,
) -> None:
    """Plot distributions for numeric model features."""

    numeric_columns = get_numeric_analysis_columns(dataframe)

    for column in numeric_columns:
        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=dataframe,
            x=column,
            bins=30,
            kde=True,
        )

        plt.title(f"Distribution of {column}")

        plt.xlabel(column)

        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(
    dataframe: pd.DataFrame,
) -> None:
    """Plot the correlation matrix for numeric columns."""

    correlation_matrix = dataframe.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()
    plt.show()


def run_eda() -> None:
    """Run the complete exploratory data analysis workflow."""

    dataframe = load_data()

    dataset_summary(dataframe)

    plot_target_distribution(dataframe)

    plot_numeric_distributions(dataframe)

    plot_correlation_matrix(dataframe)


if __name__ == "__main__":
    run_eda()
