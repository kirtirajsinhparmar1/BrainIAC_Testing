"""Summarize classifier prediction confidence and probability outputs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.visual_debugging import (  # noqa: E402
    CLASS_NAMES,
    PROBABILITY_COLUMNS,
    ensure_dir,
    import_pyplot,
    label_name,
    prediction_fieldnames,
    read_prediction_rows,
    save_empty_figure,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _save_confidence_histogram(rows: list[dict[str, Any]], output_path: Path) -> None:
    confidence = _finite([float(row.get("confidence", math.nan)) for row in rows])
    if confidence.size == 0:
        save_empty_figure(output_path, "No finite confidence values.")
        return

    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(confidence, bins=np.linspace(0.0, 1.0, 21), color="#4C78A8", edgecolor="white")
    axis.set_title("Prediction Confidence Distribution")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Number of cases")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_confidence_correct_vs_incorrect(rows: list[dict[str, Any]], output_path: Path) -> None:
    correct = _finite([float(row.get("confidence", math.nan)) for row in rows if row.get("correct", False)])
    incorrect = _finite([float(row.get("confidence", math.nan)) for row in rows if not row.get("correct", False)])
    if correct.size == 0 and incorrect.size == 0:
        save_empty_figure(output_path, "No finite confidence values.")
        return

    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 21)
    if correct.size:
        axis.hist(correct, bins=bins, alpha=0.72, label="Correct", color="#59A14F", edgecolor="white")
    if incorrect.size:
        axis.hist(incorrect, bins=bins, alpha=0.72, label="Incorrect", color="#E15759", edgecolor="white")
    axis.set_title("Confidence: Correct vs Incorrect Predictions")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Number of cases")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_class_confidence_boxplot(rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped = [
        _finite([float(row.get("confidence", math.nan)) for row in rows if row.get("true_label") == class_index])
        for class_index in range(len(CLASS_NAMES))
    ]
    if not any(group.size for group in grouped):
        save_empty_figure(output_path, "No finite confidence values by true class.")
        return

    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        [group if group.size else np.asarray([np.nan]) for group in grouped],
        labels=CLASS_NAMES,
        showmeans=True,
    )
    axis.set_title("Confidence by True Class")
    axis.set_xlabel("True class")
    axis.set_ylabel("Confidence")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_probability_distribution(rows: list[dict[str, Any]], group_key: str, output_path: Path) -> None:
    plt = import_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    any_data = False
    for class_index, axis in enumerate(axes.ravel()):
        class_rows = [row for row in rows if row.get(group_key) == class_index]
        grouped_probabilities = [
            _finite([float(row.get(column, math.nan)) for row in class_rows])
            for column in PROBABILITY_COLUMNS
        ]
        if any(values.size for values in grouped_probabilities):
            any_data = True
            axis.boxplot(
                [values if values.size else np.asarray([np.nan]) for values in grouped_probabilities],
                labels=CLASS_NAMES,
                showmeans=True,
            )
        axis.set_title(f"{label_name(class_index)} cases")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_ylabel("Predicted probability")

    if not any_data:
        plt.close(fig)
        save_empty_figure(output_path, f"No finite probability values grouped by {group_key}.")
        return

    group_label = "true class" if group_key == "true_label" else "predicted class"
    fig.suptitle(f"Probability Distribution by {group_label.title()}", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_top_csvs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    wrong_rows = [row for row in rows if not row.get("correct", False)]
    top_wrong = sorted(wrong_rows, key=lambda row: float(row.get("confidence", -math.inf)), reverse=True)[:20]
    top_uncertain = sorted(rows, key=lambda row: float(row.get("confidence", math.inf)))[:20]

    error_examples: list[dict[str, Any]] = []
    for class_index in range(len(CLASS_NAMES)):
        class_wrong = [row for row in wrong_rows if row.get("true_label") == class_index]
        class_wrong = sorted(class_wrong, key=lambda row: float(row.get("confidence", -math.inf)), reverse=True)
        for rank, row in enumerate(class_wrong[:5], start=1):
            error_row = dict(row)
            error_row["true_class_name"] = label_name(class_index)
            error_row["error_rank_within_true_class"] = rank
            error_examples.append(error_row)

    fieldnames = prediction_fieldnames()
    write_csv_rows(output_dir / "top_20_wrong_predictions.csv", top_wrong, fieldnames)
    write_csv_rows(output_dir / "top_20_uncertain_predictions.csv", top_uncertain, fieldnames)
    write_csv_rows(
        output_dir / "per_class_error_examples.csv",
        error_examples,
        [*fieldnames, "true_class_name", "error_rank_within_true_class"],
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.expanduser().resolve())
    rows = read_prediction_rows(args.predictions_csv.expanduser().resolve())
    if not rows:
        raise ValueError(f"No prediction rows found in {args.predictions_csv}")

    outputs = [
        (output_dir / "confidence_histogram.png", lambda path: _save_confidence_histogram(rows, path)),
        (
            output_dir / "confidence_correct_vs_incorrect.png",
            lambda path: _save_confidence_correct_vs_incorrect(rows, path),
        ),
        (output_dir / "class_wise_confidence_boxplot.png", lambda path: _save_class_confidence_boxplot(rows, path)),
        (
            output_dir / "probability_distribution_by_true_class.png",
            lambda path: _save_probability_distribution(rows, "true_label", path),
        ),
        (
            output_dir / "probability_distribution_by_pred_class.png",
            lambda path: _save_probability_distribution(rows, "pred_label", path),
        ),
    ]
    for output_path, writer in outputs:
        writer(output_path)
        print(f"wrote: {output_path}")

    _write_top_csvs(rows, output_dir)
    print(f"wrote: {output_dir / 'top_20_wrong_predictions.csv'}")
    print(f"wrote: {output_dir / 'top_20_uncertain_predictions.csv'}")
    print(f"wrote: {output_dir / 'per_class_error_examples.csv'}")


if __name__ == "__main__":
    main()
