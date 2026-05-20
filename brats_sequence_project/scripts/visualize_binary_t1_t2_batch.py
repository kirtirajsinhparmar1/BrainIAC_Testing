"""Visualize one binary T1-vs-T2 BraTSSequenceDataset batch."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import BraTSSequenceDataset, SUPPORTED_PREPROCESSING_VARIANTS  # noqa: E402
from utils.visual_debugging import (  # noqa: E402
    ensure_dir,
    format_stats,
    import_pyplot,
    middle_slices,
    show_slice,
    tensor_to_volume,
    volume_stats,
    write_csv_rows,
)


CLASS_NAMES = ["T1", "T2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--variant", choices=SUPPORTED_PREPROCESSING_VARIANTS, default="crop_pad_zscore")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_name", choices=["train", "val", "test"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def label_name(label: Any) -> str:
    try:
        return CLASS_NAMES[int(label)]
    except (ValueError, IndexError):
        return f"label_{label}"


def batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value[index]
    return value[index]


def label_to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def case_metadata(batch: dict[str, Any], index: int, volume: np.ndarray) -> dict[str, Any]:
    stats = volume_stats(volume)
    label = label_to_int(batch_value(batch, "label", index))
    return {
        "batch_index": index,
        "patient_id": str(batch_value(batch, "patient_id", index)),
        "modality": str(batch_value(batch, "modality", index)),
        "label": label,
        "label_name": label_name(label),
        "image_path": str(batch_value(batch, "image_path", index)),
        "shape": "x".join(str(dimension) for dimension in volume.shape),
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "std": stats["std"],
        "nonzero_fraction": stats["nonzero_fraction"],
    }


def save_detailed_batch_figure(
    volumes: list[np.ndarray],
    rows: list[dict[str, Any]],
    output_path: Path,
    split_name: str,
    variant: str,
) -> None:
    plt = import_pyplot()
    view_names = ["axial", "sagittal", "coronal"]
    case_count = len(volumes)
    fig, axes = plt.subplots(case_count, len(view_names), figsize=(12.5, max(2.1 * case_count, 4.0)), squeeze=False)

    for column_index, view_name in enumerate(view_names):
        axes[0, column_index].set_title(view_name.title(), fontsize=12, pad=10)

    for row_index, volume in enumerate(volumes):
        slices = middle_slices(volume)
        for column_index, view_name in enumerate(view_names):
            show_slice(axes[row_index, column_index], slices[view_name])

        row = rows[row_index]
        stats = {key: float(row[key]) for key in ["min", "max", "mean", "std", "nonzero_fraction"]}
        row_title = (
            f"{row['patient_id']} | {row['modality']} | label {row['label']} ({row['label_name']})\n"
            f"{format_stats(stats)}"
        )
        axes[row_index, 0].set_ylabel(row_title, rotation=0, ha="right", va="center", fontsize=7.5, labelpad=74)

    fig.suptitle(f"{split_name.title()} T1-vs-T2 batch after {variant} preprocessing ({case_count} cases)", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0.20, 0.0, 1.0, 0.985))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_axial_grid(
    volumes: list[np.ndarray],
    rows: list[dict[str, Any]],
    output_path: Path,
    split_name: str,
    variant: str,
) -> None:
    plt = import_pyplot()
    case_count = len(volumes)
    columns = 5
    rows_count = int(np.ceil(case_count / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, max(3 * rows_count, 3)), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")

    for index, volume in enumerate(volumes):
        axis = axes[index // columns, index % columns]
        row = rows[index]
        show_slice(axis, middle_slices(volume)["axial"], f"{row['patient_id']}\n{row['modality']} label {row['label_name']}")

    fig.suptitle(f"{split_name.title()} T1-vs-T2 axial middle slices after {variant} preprocessing", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = ensure_dir(args.output_dir.expanduser().resolve())
    dataset = BraTSSequenceDataset(args.csv_path, preprocessing_variant=args.variant)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if args.shuffle else None,
    )
    batch = next(iter(loader))
    images = batch["image"]
    if images.ndim != 5:
        raise ValueError(f"Expected image batch [B,C,H,W,D], got {tuple(images.shape)}")

    volumes = [tensor_to_volume(images[index]) for index in range(images.shape[0])]
    metadata_rows = [case_metadata(batch, index, volume) for index, volume in enumerate(volumes)]
    case_count = len(volumes)

    detailed_path = output_dir / f"batch_{args.split_name}_t1_t2_{args.variant}_{case_count}cases.png"
    axial_grid_path = output_dir / f"batch_{args.split_name}_t1_t2_{args.variant}_axial_grid.png"
    metadata_path = output_dir / f"batch_{args.split_name}_t1_t2_{args.variant}_metadata.csv"
    stats_path = output_dir / f"batch_{args.split_name}_t1_t2_{args.variant}_stats.json"

    save_detailed_batch_figure(volumes, metadata_rows, detailed_path, args.split_name, args.variant)
    save_axial_grid(volumes, metadata_rows, axial_grid_path, args.split_name, args.variant)
    write_csv_rows(metadata_path, metadata_rows)

    stacked = np.stack(volumes, axis=0)
    stats_payload = {
        "csv_path": str(args.csv_path.expanduser().resolve()),
        "split_name": args.split_name,
        "variant": args.variant,
        "task": "binary_t1_t2",
        "label_mapping": {"0": "T1", "1": "T2"},
        "batch_size_requested": int(args.batch_size),
        "batch_size_actual": int(case_count),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "image_batch_shape": list(images.shape),
        "batch_stats": volume_stats(stacked),
        "cases": metadata_rows,
        "outputs": {
            "detailed_figure": str(detailed_path),
            "axial_grid": str(axial_grid_path),
            "metadata_csv": str(metadata_path),
        },
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2) + "\n")

    print(f"wrote: {detailed_path}")
    print(f"wrote: {axial_grid_path}")
    print(f"wrote: {metadata_path}")
    print(f"wrote: {stats_path}")


if __name__ == "__main__":
    main()
