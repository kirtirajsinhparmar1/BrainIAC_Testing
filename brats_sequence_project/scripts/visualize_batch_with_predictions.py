"""Show one validation/test batch with classifier predictions overlaid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


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
    prediction_title,
    read_prediction_rows,
    save_empty_figure,
    show_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--variant", choices=SUPPORTED_PREPROCESSING_VARIANTS, required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--only_wrong", action="store_true")
    return parser.parse_args()


def _select_rows(rows: list[dict[str, Any]], dataset_index: dict[tuple[str, str], int], batch_size: int, only_wrong: bool) -> list[dict[str, Any]]:
    candidates = [row for row in rows if (not only_wrong or not row.get("correct", False))]
    selected: list[dict[str, Any]] = []
    for row in candidates:
        if find_dataset_index(dataset_index, row) is not None or row.get("image_path"):
            selected.append(row)
        if len(selected) >= batch_size:
            break
    return selected


def _save_batch_figure(
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
        except Exception as error:  # pragma: no cover - defensive per-case reporting
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
    fig.suptitle(f"{filter_label} prediction batch after {variant} preprocessing", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.25, 0.0, 1.0, 0.985))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


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
    selected_rows = _select_rows(rows, dataset_index, args.batch_size, args.only_wrong)
    output_path = output_dir / f"validation_batch_predictions_{args.variant}.png"
    _save_batch_figure(selected_rows, dataset, dataset_index, output_path, args.variant, args.only_wrong)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
