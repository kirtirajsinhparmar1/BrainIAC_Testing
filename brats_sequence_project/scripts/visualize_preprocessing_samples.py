"""Save professor-friendly before/after resize slice views for one BraTS patient."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from monai.transforms import EnsureChannelFirstd, LoadImaged, Resized
except ImportError as error:  # pragma: no cover - import guard
    raise SystemExit(
        "MONAI is required for visualize_preprocessing_samples.py. "
        "Install the BrainIAC project environment first."
    ) from error

from utils.preprocessing_audit_utils import (  # noqa: E402
    CLASS_NAMES,
    detect_archive_zip,
    ensure_dir,
    find_complete_patient,
    load_nifti_from_source,
    materialize_nifti_path,
    middle_slices,
    prepare_display_slice,
    read_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--patient_id", type=str, default=None)
    parser.add_argument("--archive_zip", type=Path, default=None)
    return parser.parse_args()


def save_modality_figure(
    patient_id: str,
    modality: str,
    original_volume: np.ndarray,
    resized_volume: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = middle_slices(original_volume)
    resized = middle_slices(resized_volume)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))

    for col_idx, plane in enumerate(["axial", "coronal", "sagittal"]):
        axes[0, col_idx].imshow(prepare_display_slice(original[plane]).T, cmap="gray", origin="lower")
        axes[0, col_idx].set_title(f"Original {plane}")
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(prepare_display_slice(resized[plane]).T, cmap="gray", origin="lower")
        axes[1, col_idx].set_title(f"Resized {plane}")
        axes[1, col_idx].axis("off")

    figure.suptitle(f"{patient_id} {modality}: original vs resized to 96x96x96")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    archive_zip = detect_archive_zip(args.csv_path, args.archive_zip)
    rows = read_csv_rows(args.csv_path)
    patient_id, patient_rows = find_complete_patient(rows, args.patient_id)

    resize_pipeline = [
        ("load", LoadImaged(keys=["image"])),
        ("channel", EnsureChannelFirstd(keys=["image"])),
        ("resize", Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear")),
    ]

    for modality in CLASS_NAMES:
        row = patient_rows[modality]
        image_path = Path(row["image_path"])
        image, source = load_nifti_from_source(image_path, archive_zip)
        original_volume = np.asarray(image.dataobj)

        with materialize_nifti_path(image_path, archive_zip) as materialized_path:
            state: dict[str, object] = {"image": str(materialized_path)}
            for _, transform in resize_pipeline:
                state = transform(state)

        resized_volume = state["image"].detach().cpu().numpy()
        output_path = output_dir / f"{patient_id}_{modality.lower()}_before_after.png"
        save_modality_figure(patient_id, modality, original_volume, resized_volume, output_path)
        print(f"{modality}: {output_path} ({source})")


if __name__ == "__main__":
    main()
