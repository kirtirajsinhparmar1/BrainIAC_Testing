"""Compare the same BraTS patient under multiple preprocessing variants."""

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
    build_dataset_index,
    ensure_dir,
    find_dataset_index,
    format_stats,
    import_pyplot,
    middle_slices,
    read_csv_rows,
    rows_for_complete_patient,
    safe_filename,
    show_slice,
    tensor_to_volume,
    volume_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--patient_id", type=str, default=None)
    parser.add_argument("--modalities", nargs="+", default=["T1", "T2", "FLAIR", "T1CE"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["resize_zscore", "crop_pad_zscore", "resize_none"],
        choices=SUPPORTED_PREPROCESSING_VARIANTS,
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _load_patient_volumes(
    csv_path: Path,
    patient_rows: dict[str, dict[str, str]],
    variants: list[str],
    modalities: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    volumes: dict[str, dict[str, np.ndarray]] = {modality: {} for modality in modalities}
    for variant in variants:
        dataset = BraTSSequenceDataset(csv_path, preprocessing_variant=variant)
        dataset_index = build_dataset_index(dataset)
        for modality in modalities:
            row = patient_rows[modality]
            index = find_dataset_index(dataset_index, row)
            if index is None:
                raise KeyError(f"Could not find {row['patient_id']} {modality} in dataset for {variant}.")
            sample = dataset[index]
            volumes[modality][variant] = tensor_to_volume(sample["image"])
    return volumes


def _save_plane_grid(
    patient_id: str,
    volumes: dict[str, dict[str, np.ndarray]],
    modalities: list[str],
    variants: list[str],
    plane: str,
    output_path: Path,
) -> None:
    plt = import_pyplot()
    fig, axes = plt.subplots(
        len(modalities),
        len(variants),
        figsize=(3.4 * len(variants), 3.0 * len(modalities)),
        squeeze=False,
    )
    for row_index, modality in enumerate(modalities):
        for column_index, variant in enumerate(variants):
            axis = axes[row_index, column_index]
            volume = volumes[modality][variant]
            show_slice(axis, middle_slices(volume)[plane])
            if row_index == 0:
                axis.set_title(variant, fontsize=11)
            if column_index == 0:
                axis.set_ylabel(modality, rotation=0, ha="right", va="center", fontsize=11, labelpad=38)

    fig.suptitle(f"{patient_id}: {plane} middle slice by modality and preprocessing variant", fontsize=15)
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_modality_detail(
    patient_id: str,
    modality: str,
    modality_volumes: dict[str, np.ndarray],
    variants: list[str],
    output_path: Path,
) -> None:
    plt = import_pyplot()
    view_names = ["axial", "sagittal", "coronal"]
    fig, axes = plt.subplots(
        len(variants),
        len(view_names),
        figsize=(12.5, max(3.0 * len(variants), 4.0)),
        squeeze=False,
    )
    for column_index, view_name in enumerate(view_names):
        axes[0, column_index].set_title(view_name.title(), fontsize=12, pad=8)

    for row_index, variant in enumerate(variants):
        volume = modality_volumes[variant]
        slices = middle_slices(volume)
        stats = volume_stats(volume)
        for column_index, view_name in enumerate(view_names):
            show_slice(axes[row_index, column_index], slices[view_name])
        axes[row_index, 0].set_ylabel(
            f"{variant}\n{format_stats(stats)}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=82,
        )

    fig.suptitle(f"{patient_id} {modality}: all views by preprocessing variant", fontsize=15)
    fig.tight_layout(rect=(0.22, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir.expanduser().resolve())
    csv_rows = read_csv_rows(args.csv_path)
    patient_id, patient_rows = rows_for_complete_patient(csv_rows, args.patient_id, args.modalities)
    volumes = _load_patient_volumes(args.csv_path, patient_rows, args.variants, args.modalities)

    for plane in ["axial", "sagittal", "coronal"]:
        output_path = output_dir / f"compare_variants_{plane}.png"
        _save_plane_grid(patient_id, volumes, args.modalities, args.variants, plane, output_path)
        print(f"wrote: {output_path}")

    safe_patient_id = safe_filename(patient_id)
    for modality in args.modalities:
        output_path = output_dir / f"compare_{safe_patient_id}_{safe_filename(modality)}_all_views.png"
        _save_modality_detail(patient_id, modality, volumes[modality], args.variants, output_path)
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
