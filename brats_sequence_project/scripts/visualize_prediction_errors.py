"""Visualize original and preprocessed images for prediction examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import BraTSSequenceDataset, SUPPORTED_PREPROCESSING_VARIANTS  # noqa: E402
from utils.visual_debugging import (  # noqa: E402
    PROBABILITY_COLUMNS,
    build_dataset_index,
    confidence,
    ensure_dir,
    import_pyplot,
    label_name,
    load_original_volume,
    load_preprocessed_volume_for_prediction,
    middle_slices,
    prediction_title,
    read_prediction_rows,
    safe_filename,
    save_empty_figure,
    show_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--variant", choices=SUPPORTED_PREPROCESSING_VARIANTS, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_k_wrong", type=int, default=20)
    parser.add_argument("--top_k_correct", type=int, default=20)
    parser.add_argument("--top_k_uncertain", type=int, default=20)
    return parser.parse_args()


def _select_rows(rows: list[dict[str, Any]], top_k_wrong: int, top_k_correct: int, top_k_uncertain: int) -> dict[str, list[dict[str, Any]]]:
    wrong_rows = [row for row in rows if not row.get("correct", False)]
    correct_rows = [row for row in rows if row.get("correct", False)]
    return {
        "wrong_predictions": sorted(wrong_rows, key=confidence, reverse=True)[:top_k_wrong],
        "correct_high_confidence": sorted(correct_rows, key=confidence, reverse=True)[:top_k_correct],
        "uncertain_predictions": sorted(rows, key=confidence)[:top_k_uncertain],
    }


def _case_filename(row: dict[str, Any], rank: int) -> str:
    true_name = label_name(row.get("true_label"))
    pred_name = label_name(row.get("pred_label"))
    parts = [
        f"{rank:02d}",
        safe_filename(row.get("patient_id", "case")),
        safe_filename(row.get("modality", "modality")),
        f"true-{safe_filename(true_name)}",
        f"pred-{safe_filename(pred_name)}",
    ]
    return "_".join(parts) + ".png"


def _save_case_figure(
    row: dict[str, Any],
    original_volume: np.ndarray,
    preprocessed_volume: np.ndarray,
    output_path: Path,
    variant: str,
) -> None:
    plt = import_pyplot()
    view_names = ["axial", "sagittal", "coronal"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), squeeze=False)
    original_slices = middle_slices(original_volume)
    preprocessed_slices = middle_slices(preprocessed_volume)

    for column_index, view_name in enumerate(view_names):
        show_slice(axes[0, column_index], original_slices[view_name], f"Original {view_name}")
        show_slice(axes[1, column_index], preprocessed_slices[view_name], f"{variant} {view_name}")

    axes[0, 0].set_ylabel("Original", rotation=0, ha="right", va="center", fontsize=11, labelpad=42)
    axes[1, 0].set_ylabel("Preprocessed", rotation=0, ha="right", va="center", fontsize=11, labelpad=42)
    fig.suptitle(prediction_title(row), fontsize=13)
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.90))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_contact_sheet(
    selected_rows: list[dict[str, Any]],
    dataset: BraTSSequenceDataset,
    dataset_index: dict[tuple[str, str], int],
    output_path: Path,
    title: str,
) -> None:
    if not selected_rows:
        save_empty_figure(output_path, f"No cases selected for {title}.")
        return

    plt = import_pyplot()
    columns = 5
    rows_count = int(np.ceil(len(selected_rows) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, max(3.1 * rows_count, 3.2)), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")

    for index, row in enumerate(selected_rows):
        axis = axes[index // columns, index % columns]
        try:
            preprocessed_volume, _ = load_preprocessed_volume_for_prediction(dataset, dataset_index, row)
            show_slice(axis, middle_slices(preprocessed_volume)["axial"], prediction_title(row, compact=True))
        except Exception as error:  # pragma: no cover - defensive per-case reporting
            axis.text(0.5, 0.5, f"load failed\n{error}", ha="center", va="center", fontsize=8)
            axis.set_title(prediction_title(row, compact=True), fontsize=8)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_group(
    group_name: str,
    rows: list[dict[str, Any]],
    dataset: BraTSSequenceDataset,
    dataset_index: dict[tuple[str, str], int],
    output_dir: Path,
    variant: str,
) -> None:
    group_dir = ensure_dir(output_dir / group_name)
    for rank, row in enumerate(rows, start=1):
        output_path = group_dir / _case_filename(row, rank)
        try:
            original_volume = load_original_volume(str(row.get("image_path", "")), dataset)
            preprocessed_volume, _ = load_preprocessed_volume_for_prediction(dataset, dataset_index, row)
            _save_case_figure(row, original_volume, preprocessed_volume, output_path, variant)
            print(f"wrote: {output_path}")
        except Exception as error:
            failed_path = group_dir / f"{output_path.stem}_LOAD_FAILED.txt"
            failed_path.write_text(f"{prediction_title(row)}\n\n{error}\n")
            print(f"skip: {output_path.name}: {error}")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.expanduser().resolve())
    rows = read_prediction_rows(args.predictions_csv.expanduser().resolve())
    if not rows:
        raise ValueError(f"No prediction rows found in {args.predictions_csv}")

    if args.features_path is not None:
        print(f"note: --features_path is accepted for provenance but image visualization uses --csv_path and --predictions_csv: {args.features_path}")

    dataset = BraTSSequenceDataset(args.csv_path, preprocessing_variant=args.variant)
    dataset_index = build_dataset_index(dataset)
    selected = _select_rows(rows, args.top_k_wrong, args.top_k_correct, args.top_k_uncertain)

    for group_name, group_rows in selected.items():
        _write_group(group_name, group_rows, dataset, dataset_index, output_dir, args.variant)

    contact_specs = [
        ("wrong_predictions", "wrong_predictions_grid.png", "Highest-confidence wrong predictions"),
        ("uncertain_predictions", "uncertain_predictions_grid.png", "Most uncertain predictions"),
        ("correct_high_confidence", "correct_predictions_grid.png", "Highest-confidence correct predictions"),
    ]
    for group_name, filename, title in contact_specs:
        output_path = output_dir / filename
        _save_contact_sheet(selected[group_name], dataset, dataset_index, output_path, title)
        print(f"wrote: {output_path}")

    print("probability columns:", ", ".join(PROBABILITY_COLUMNS))


if __name__ == "__main__":
    main()
