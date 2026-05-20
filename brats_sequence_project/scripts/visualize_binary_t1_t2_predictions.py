"""Visualize binary T1-vs-T2 prediction batches and error examples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import BraTSSequenceDataset, SUPPORTED_PREPROCESSING_VARIANTS  # noqa: E402
from utils.visual_debugging import (  # noqa: E402
    build_dataset_index,
    ensure_dir,
    find_dataset_index,
    import_pyplot,
    load_preprocessed_volume_for_prediction,
    middle_slices,
    safe_filename,
    save_empty_figure,
    show_slice,
)


CLASS_NAMES = ["T1", "T2"]
PROBABILITY_COLUMNS = ["probability_T1", "probability_T2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--variant", choices=SUPPORTED_PREPROCESSING_VARIANTS, default="crop_pad_zscore")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--only_wrong", action="store_true")
    return parser.parse_args()


def label_name(value: Any) -> str:
    try:
        return CLASS_NAMES[int(value)]
    except (ValueError, IndexError, TypeError):
        return f"label_{value}"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            for key in ["true_label", "pred_label"]:
                parsed[key] = int(parsed[key])
            for key in ["confidence", *PROBABILITY_COLUMNS]:
                parsed[key] = float(parsed.get(key, 0.0))
            parsed["correct"] = parse_bool(parsed.get("correct", False))
            rows.append(parsed)
    return rows


def confidence(row: dict[str, Any]) -> float:
    return float(row.get("confidence", 0.0))


def prediction_title(row: dict[str, Any], compact: bool = False) -> str:
    true_name = label_name(row.get("true_label"))
    pred_name = label_name(row.get("pred_label"))
    status = "correct" if row.get("correct") else "wrong"
    first_line = f"{row.get('patient_id', 'case')} | true {true_name} -> pred {pred_name} | {status}"
    second_line = (
        f"conf={confidence(row):.3f} "
        f"P(T1)={float(row.get('probability_T1', 0.0)):.3f} "
        f"P(T2)={float(row.get('probability_T2', 0.0)):.3f}"
    )
    if compact:
        return first_line.replace(" | ", "\n", 1) + "\n" + second_line
    return first_line + "\n" + second_line


def select_rows(
    rows: list[dict[str, Any]],
    dataset_index: dict[tuple[str, str], int],
    batch_size: int,
    only_wrong: bool,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if (not only_wrong or not row.get("correct", False))]
    selected: list[dict[str, Any]] = []
    for row in candidates:
        if find_dataset_index(dataset_index, row) is not None or row.get("image_path"):
            selected.append(row)
        if len(selected) >= batch_size:
            break
    return selected


def select_groups(rows: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    wrong_rows = [row for row in rows if not row.get("correct", False)]
    correct_rows = [row for row in rows if row.get("correct", False)]
    return {
        "wrong_predictions": sorted(wrong_rows, key=confidence, reverse=True)[:top_k],
        "uncertain_predictions": sorted(rows, key=confidence)[:top_k],
        "correct_high_confidence": sorted(correct_rows, key=confidence, reverse=True)[:top_k],
    }


def save_batch_figure(
    rows: list[dict[str, Any]],
    dataset: BraTSSequenceDataset,
    dataset_index: dict[tuple[str, str], int],
    output_path: Path,
    variant: str,
    only_wrong: bool,
) -> None:
    if not rows:
        save_empty_figure(output_path, "No prediction rows matched the requested batch filter.")
        return

    plt = import_pyplot()
    view_names = ["axial", "sagittal", "coronal"]
    fig, axes = plt.subplots(len(rows), len(view_names), figsize=(12.5, max(2.15 * len(rows), 4.0)), squeeze=False)

    for column_index, view_name in enumerate(view_names):
        axes[0, column_index].set_title(view_name.title(), fontsize=12, pad=10)

    for row_index, row in enumerate(rows):
        try:
            volume, _ = load_preprocessed_volume_for_prediction(dataset, dataset_index, row)
            slices = middle_slices(volume)
            for column_index, view_name in enumerate(view_names):
                show_slice(axes[row_index, column_index], slices[view_name])
        except Exception as error:  # pragma: no cover - defensive case reporting
            for column_index in range(len(view_names)):
                axes[row_index, column_index].text(0.5, 0.5, f"load failed\n{error}", ha="center", va="center")
                axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(
            prediction_title(row, compact=True),
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=92,
        )

    filter_label = "wrong-only" if only_wrong else "first matched"
    fig.suptitle(f"{filter_label} T1-vs-T2 prediction batch after {variant} preprocessing", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.25, 0.0, 1.0, 0.985))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_case_figure(
    row: dict[str, Any],
    dataset: BraTSSequenceDataset,
    dataset_index: dict[tuple[str, str], int],
    output_path: Path,
    variant: str,
) -> None:
    plt = import_pyplot()
    view_names = ["axial", "sagittal", "coronal"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), squeeze=False)
    volume, _ = load_preprocessed_volume_for_prediction(dataset, dataset_index, row)
    slices = middle_slices(volume)
    for column_index, view_name in enumerate(view_names):
        show_slice(axes[0, column_index], slices[view_name], f"{variant} {view_name}")
    fig.suptitle(prediction_title(row), fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_contact_sheet(
    rows: list[dict[str, Any]],
    dataset: BraTSSequenceDataset,
    dataset_index: dict[tuple[str, str], int],
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        save_empty_figure(output_path, f"No cases selected for {title}.")
        return

    plt = import_pyplot()
    columns = 5
    rows_count = int(np.ceil(len(rows) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, max(3.1 * rows_count, 3.2)), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")

    for index, row in enumerate(rows):
        axis = axes[index // columns, index % columns]
        try:
            volume, _ = load_preprocessed_volume_for_prediction(dataset, dataset_index, row)
            show_slice(axis, middle_slices(volume)["axial"], prediction_title(row, compact=True))
        except Exception as error:  # pragma: no cover - defensive case reporting
            axis.text(0.5, 0.5, f"load failed\n{error}", ha="center", va="center", fontsize=8)
            axis.set_title(prediction_title(row, compact=True), fontsize=8)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def case_filename(row: dict[str, Any], rank: int) -> str:
    parts = [
        f"{rank:02d}",
        safe_filename(row.get("patient_id", "case")),
        f"true-{safe_filename(label_name(row.get('true_label')))}",
        f"pred-{safe_filename(label_name(row.get('pred_label')))}",
    ]
    return "_".join(parts) + ".png"


def write_group(
    group_name: str,
    rows: list[dict[str, Any]],
    dataset: BraTSSequenceDataset,
    dataset_index: dict[tuple[str, str], int],
    output_dir: Path,
    variant: str,
) -> None:
    group_dir = ensure_dir(output_dir / group_name)
    for rank, row in enumerate(rows, start=1):
        output_path = group_dir / case_filename(row, rank)
        try:
            save_case_figure(row, dataset, dataset_index, output_path, variant)
            print(f"wrote: {output_path}")
        except Exception as error:
            failed_path = group_dir / f"{output_path.stem}_LOAD_FAILED.txt"
            failed_path.write_text(f"{prediction_title(row)}\n\n{error}\n")
            print(f"skip: {output_path.name}: {error}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")

    output_dir = ensure_dir(args.output_dir.expanduser().resolve())
    rows = read_prediction_rows(args.predictions_csv.expanduser().resolve())
    if not rows:
        raise ValueError(f"No prediction rows found in {args.predictions_csv}")

    dataset = BraTSSequenceDataset(args.csv_path, preprocessing_variant=args.variant)
    dataset_index = build_dataset_index(dataset)
    selected_rows = select_rows(rows, dataset_index, args.batch_size, args.only_wrong)
    batch_path = output_dir / "validation_batch_predictions_t1_t2.png"
    save_batch_figure(selected_rows, dataset, dataset_index, batch_path, args.variant, args.only_wrong)
    print(f"wrote: {batch_path}")

    selected = select_groups(rows, args.batch_size)
    for group_name, group_rows in selected.items():
        write_group(group_name, group_rows, dataset, dataset_index, output_dir, args.variant)

    contact_specs = [
        ("wrong_predictions", "wrong_predictions_grid_t1_t2.png", "Highest-confidence wrong T1-vs-T2 predictions"),
        ("uncertain_predictions", "uncertain_predictions_grid_t1_t2.png", "Most uncertain T1-vs-T2 predictions"),
        ("correct_high_confidence", "correct_predictions_grid_t1_t2.png", "Highest-confidence correct T1-vs-T2 predictions"),
    ]
    for group_name, filename, title in contact_specs:
        output_path = output_dir / filename
        save_contact_sheet(selected[group_name], dataset, dataset_index, output_path, title)
        print(f"wrote: {output_path}")

    print("probability columns:", ", ".join(PROBABILITY_COLUMNS))


if __name__ == "__main__":
    main()
