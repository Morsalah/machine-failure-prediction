from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from machine_failure import reporting
from machine_failure.reporting import (
    append_to_model_comparison,
    create_report_directories,
    format_threshold_label,
    sanitize_filename,
    save_figure,
    save_metrics,
    save_metrics_table,
)


def create_sample_metrics() -> dict[str, float]:
    """Create sample classification metrics for reporting tests."""

    return {
        "threshold": 0.50,
        "accuracy": 0.95,
        "precision": 0.60,
        "recall": 0.70,
        "f1": 0.64,
        "f2": 0.68,
        "roc_auc": 0.90,
        "pr_auc": 0.50,
    }


def configure_temporary_report_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redirect report output to a temporary test directory."""

    reports_dir = tmp_path / "reports"
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"

    monkeypatch.setattr(
        reporting,
        "REPORTS_DIR",
        reports_dir,
    )

    monkeypatch.setattr(
        reporting,
        "METRICS_DIR",
        metrics_dir,
    )

    monkeypatch.setattr(
        reporting,
        "FIGURES_DIR",
        figures_dir,
    )

    monkeypatch.setattr(
        reporting,
        "TABLES_DIR",
        tables_dir,
    )


def test_create_report_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_temporary_report_directories(
        tmp_path,
        monkeypatch,
    )

    create_report_directories()

    assert reporting.METRICS_DIR.exists()
    assert reporting.FIGURES_DIR.exists()
    assert reporting.TABLES_DIR.exists()


def test_format_threshold_label() -> None:
    assert format_threshold_label(0.50) == "050"
    assert format_threshold_label(0.46) == "046"
    assert format_threshold_label(0.30) == "030"
    assert format_threshold_label(0.09) == "009"


def test_format_threshold_label_rejects_invalid_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="Threshold must be between 0 and 1",
    ):
        format_threshold_label(1.5)


def test_sanitize_filename() -> None:
    assert sanitize_filename(
        "Tuned XGBoost"
    ) == "tuned_xgboost"

    assert sanitize_filename(
        "Random-Forest"
    ) == "random_forest"


def test_save_metrics_creates_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_temporary_report_directories(
        tmp_path,
        monkeypatch,
    )

    metrics = create_sample_metrics()

    output_path = save_metrics(
        metrics=metrics,
        model_name="XGBoost",
        threshold=0.50,
    )

    assert output_path.exists()

    dataframe = pd.read_csv(
        output_path
    )

    assert len(dataframe) == 1
    assert dataframe.loc[0, "model"] == "xgboost"
    assert dataframe.loc[0, "threshold"] == pytest.approx(0.50)
    assert dataframe.loc[0, "recall"] == pytest.approx(0.70)


def test_save_metrics_table_creates_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_temporary_report_directories(
        tmp_path,
        monkeypatch,
    )

    rows = [
        {
            "model": "model_a",
            "threshold": 0.5,
            "f1": 0.7,
        },
        {
            "model": "model_b",
            "threshold": 0.3,
            "f1": 0.8,
        },
    ]

    output_path = save_metrics_table(
        rows,
        filename="comparison",
    )

    assert output_path.exists()
    assert output_path.name == "comparison.csv"

    dataframe = pd.read_csv(
        output_path
    )

    assert len(dataframe) == 2


def test_save_metrics_table_rejects_empty_rows() -> None:
    with pytest.raises(
        ValueError,
        match="At least one metrics row is required",
    ):
        save_metrics_table([])


def test_append_to_model_comparison_adds_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_temporary_report_directories(
        tmp_path,
        monkeypatch,
    )

    metrics = create_sample_metrics()

    output_path = append_to_model_comparison(
        metrics=metrics,
        model_name="xgboost",
        threshold=0.50,
    )

    comparison = pd.read_csv(
        output_path
    )

    assert len(comparison) == 1
    assert comparison.loc[0, "model"] == "xgboost"
    assert comparison.loc[0, "threshold"] == pytest.approx(0.50)


def test_append_to_model_comparison_replaces_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_temporary_report_directories(
        tmp_path,
        monkeypatch,
    )

    original_metrics = create_sample_metrics()

    append_to_model_comparison(
        metrics=original_metrics,
        model_name="xgboost",
        threshold=0.50,
    )

    updated_metrics = {
        **original_metrics,
        "recall": 0.85,
    }

    output_path = append_to_model_comparison(
        metrics=updated_metrics,
        model_name="xgboost",
        threshold=0.50,
    )

    comparison = pd.read_csv(
        output_path
    )

    assert len(comparison) == 1
    assert comparison.loc[0, "recall"] == pytest.approx(0.85)


def test_save_figure_creates_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_temporary_report_directories(
        tmp_path,
        monkeypatch,
    )

    figure = plt.figure()

    output_path = save_figure(
        figure=figure,
        filename="test_figure",
        close=True,
    )

    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_save_figure_rejects_invalid_dpi() -> None:
    figure = plt.figure()

    try:
        with pytest.raises(
            ValueError,
            match="DPI must be greater than zero",
        ):
            save_figure(
                figure=figure,
                filename="test.png",
                dpi=0,
            )
    finally:
        plt.close(figure)