"""Model comparison reporting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from machine_failure.reporting import save_figure

COMPARISON_PATH = Path("reports/tables/model_comparison.csv")

PLOT_METRICS = [
    "precision",
    "recall",
    "f1",
    "f2",
    "pr_auc",
]

BEST_MODEL_METRICS = [
    "precision",
    "recall",
    "f1",
    "f2",
    "roc_auc",
    "pr_auc",
]


def load_comparison_table() -> pd.DataFrame:
    """Load the model comparison table."""

    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"Model comparison table was not found: {COMPARISON_PATH.resolve()}"
        )

    comparison = pd.read_csv(COMPARISON_PATH)

    validate_comparison_table(comparison)

    return comparison


def validate_comparison_table(
    comparison: pd.DataFrame,
) -> None:
    """Validate required columns in the comparison table."""

    required_columns = {
        "model",
        "threshold",
        *BEST_MODEL_METRICS,
    }

    missing_columns = required_columns - set(comparison.columns)

    if missing_columns:
        raise ValueError(
            "Model comparison table is missing "
            f"required columns: {sorted(missing_columns)}"
        )


def create_model_label(
    row: pd.Series,
) -> str:
    """Create a readable model + threshold label."""

    model_name = str(row["model"]).replace("_", " ").title()

    return f"{model_name}\nt={row['threshold']:.2f}"


def prepare_plot_data(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare labeled metric data for plotting."""

    plot_data = comparison.copy()

    plot_data["label"] = plot_data.apply(
        create_model_label,
        axis=1,
    )

    return plot_data.set_index("label")[PLOT_METRICS]


def plot_model_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Compare selected model metrics."""

    plot_data = prepare_plot_data(comparison)

    figure = plt.figure(figsize=(14, 7))

    ax = figure.add_subplot(111)

    plot_data.plot(
        kind="bar",
        ax=ax,
    )

    plt.ylabel("Score")

    plt.xlabel("Model Configuration")

    plt.title("Machine Failure Prediction — Model Comparison")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.ylim(
        0,
        1,
    )

    plt.legend(
        title="Metric",
    )

    plt.tight_layout()

    output_path = save_figure(
        figure=figure,
        filename="model_comparison.png",
    )

    print(f"Model comparison plot saved to: {output_path}")

    plt.show()


def get_best_model_for_metric(
    comparison: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """Return the best model configuration for one metric."""

    if metric not in comparison.columns:
        raise ValueError(f"Metric '{metric}' was not found in the comparison table.")

    best_index = comparison[metric].idxmax()

    return comparison.loc[best_index]


def print_best_models(
    comparison: pd.DataFrame,
) -> None:
    """Print the best configuration for key metrics."""

    print("=" * 70)
    print("BEST MODEL CONFIGURATIONS")
    print("=" * 70)

    for metric in BEST_MODEL_METRICS:
        best = get_best_model_for_metric(
            comparison,
            metric,
        )

        print(
            f"{metric:>10}: "
            f"{best['model']} "
            f"(threshold={best['threshold']:.2f}) "
            f"-> {best[metric]:.4f}"
        )


def run_model_comparison() -> None:
    """Run model comparison analysis."""

    comparison = load_comparison_table()

    print_best_models(comparison)

    plot_model_comparison(comparison)


if __name__ == "__main__":
    run_model_comparison()
