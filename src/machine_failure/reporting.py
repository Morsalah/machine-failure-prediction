"""Utilities for saving model metrics and generated figures."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MODEL_COMPARISON_FILENAME = "model_comparison.csv"


def create_report_directories() -> None:
    """Create all report output directories when they do not exist."""

    for directory in (
        METRICS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def format_threshold_label(
    threshold: float,
) -> str:
    """Convert a probability threshold into a filename-safe label.

    Examples
    --------
    0.50 becomes ``050``.
    0.46 becomes ``046``.
    0.30 becomes ``030``.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    return f"{threshold:.2f}".replace(
        ".",
        "",
    )


def sanitize_filename(
    value: str,
) -> str:
    """Convert a descriptive name into a safe filename component."""

    return value.strip().lower().replace(" ", "_").replace("-", "_")


def ensure_csv_filename(
    filename: str,
) -> str:
    """Return a safe CSV filename."""

    safe_filename = Path(filename).name

    if not safe_filename.lower().endswith(".csv"):
        safe_filename = f"{safe_filename}.csv"

    return safe_filename


def build_metrics_row(
    metrics: dict[str, float],
    model_name: str,
    threshold: float,
) -> dict[str, Any]:
    """Build a normalized metrics row for report tables."""

    safe_model_name = sanitize_filename(model_name)

    return {
        "model": safe_model_name,
        **metrics,
        "threshold": threshold,
    }


def save_metrics(
    metrics: dict[str, float],
    model_name: str,
    threshold: float,
) -> Path:
    """Save one model evaluation result as a one-row CSV file."""

    create_report_directories()

    threshold_label = format_threshold_label(threshold)

    safe_model_name = sanitize_filename(model_name)

    output_path = METRICS_DIR / (f"{safe_model_name}_threshold_{threshold_label}.csv")

    row = build_metrics_row(
        metrics=metrics,
        model_name=model_name,
        threshold=threshold,
    )

    pd.DataFrame([row]).to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_metrics_table(
    rows: list[dict[str, Any]],
    filename: str = MODEL_COMPARISON_FILENAME,
) -> Path:
    """Save multiple model evaluation rows as a comparison CSV."""

    if not rows:
        raise ValueError("At least one metrics row is required.")

    create_report_directories()

    safe_filename = ensure_csv_filename(filename)

    output_path = TABLES_DIR / safe_filename

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    return output_path


def append_to_model_comparison(
    metrics: dict[str, float],
    model_name: str,
    threshold: float,
) -> Path:
    """Append or update a model result in the comparison table.

    A combination of model name and threshold is treated as unique.
    If that combination already exists, its row is replaced.
    """

    create_report_directories()

    output_path = TABLES_DIR / MODEL_COMPARISON_FILENAME

    row = build_metrics_row(
        metrics=metrics,
        model_name=model_name,
        threshold=threshold,
    )

    new_row = pd.DataFrame([row])

    if output_path.exists():
        comparison = pd.read_csv(output_path)

        duplicate_mask = (comparison["model"] == row["model"]) & (
            comparison["threshold"].round(4)
            == round(
                threshold,
                4,
            )
        )

        comparison = comparison.loc[~duplicate_mask]

        comparison = pd.concat(
            [
                comparison,
                new_row,
            ],
            ignore_index=True,
        )

    else:
        comparison = new_row

    comparison = comparison.sort_values(
        by=[
            "model",
            "threshold",
        ],
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_figure(
    figure: Figure,
    filename: str,
    dpi: int = 150,
    close: bool = False,
) -> Path:
    """Save a specific Matplotlib figure."""

    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    create_report_directories()

    safe_filename = Path(filename).name

    if not Path(safe_filename).suffix:
        safe_filename = f"{safe_filename}.png"

    output_path = FIGURES_DIR / safe_filename

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if close:
        plt.close(figure)

    return output_path


def save_current_figure(
    filename: str,
    dpi: int = 150,
    close: bool = False,
) -> Path:
    """Save the currently active Matplotlib figure."""

    return save_figure(
        figure=plt.gcf(),
        filename=filename,
        dpi=dpi,
        close=close,
    )
