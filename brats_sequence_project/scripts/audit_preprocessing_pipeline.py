"""Audit the exact BraTS -> MONAI -> BrainIAC preprocessing path on a few samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from monai.transforms import EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, Resized, ToTensord
except ImportError as error:  # pragma: no cover - import guard
    raise SystemExit(
        "MONAI is required for audit_preprocessing_pipeline.py. "
        "Install the BrainIAC project environment first."
    ) from error

from utils.preprocessing_audit_utils import (  # noqa: E402
    array_stats,
    detect_archive_zip,
    ensure_dir,
    infer_spacing_after_resize,
    load_nifti_from_source,
    materialize_nifti_path,
    meta_value_to_list,
    middle_slices,
    orientation_codes,
    prepare_display_slice,
    read_csv_rows,
    rows_to_csv_dicts,
    select_representative_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--archive_zip", type=Path, default=None)
    return parser.parse_args()


def collect_step_summary(step_name: str, state: dict[str, Any]) -> dict[str, Any]:
    image = state["image"]
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    meta = getattr(image, "meta", {}) if hasattr(image, "meta") else {}
    affine = meta.get("affine")
    affine_list = np.asarray(affine).tolist() if affine is not None else None

    return {
        "step": step_name,
        "type": type(image).__name__,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        **array_stats(array),
        "pixdim": meta_value_to_list(meta.get("pixdim")),
        "spatial_shape": meta_value_to_list(meta.get("spatial_shape")),
        "space": str(meta.get("space")) if meta.get("space") is not None else None,
        "affine": affine_list,
    }


def save_slice_grid(records: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        return

    figure, axes = plt.subplots(len(records), 6, figsize=(18, 4 * len(records)))
    if len(records) == 1:
        axes = np.asarray([axes])

    for row_idx, record in enumerate(records):
        original = record["original_array"]
        resized = record["resized_array"]
        original_slices = middle_slices(original)
        resized_slices = middle_slices(resized)
        title_prefix = f"{record['patient_id']} {record['modality']}"

        for col_idx, plane in enumerate(["axial", "coronal", "sagittal"]):
            axes[row_idx, col_idx].imshow(prepare_display_slice(original_slices[plane]).T, cmap="gray", origin="lower")
            axes[row_idx, col_idx].set_title(f"{title_prefix}\nOriginal {plane}")
            axes[row_idx, col_idx].axis("off")

            axes[row_idx, col_idx + 3].imshow(
                prepare_display_slice(resized_slices[plane]).T,
                cmap="gray",
                origin="lower",
            )
            axes[row_idx, col_idx + 3].set_title(f"{title_prefix}\nResized {plane}")
            axes[row_idx, col_idx + 3].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    archive_zip = detect_archive_zip(args.csv_path, args.archive_zip)
    rows = read_csv_rows(args.csv_path)
    selected_rows = select_representative_rows(rows, args.num_samples)

    transforms = [
        ("load", LoadImaged(keys=["image"])),
        ("channel", EnsureChannelFirstd(keys=["image"])),
        ("resize", Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear")),
        ("normalize", NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)),
        ("tensor", ToTensord(keys=["image"])),
    ]

    records: list[dict[str, Any]] = []

    for row in selected_rows:
        image_path = Path(row["image_path"])
        image, source = load_nifti_from_source(image_path, archive_zip)
        original_array = np.asarray(image.dataobj)
        original_shape = tuple(int(value) for value in image.shape[:3])
        original_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        original_affine = np.asarray(image.affine)
        original_orientation = orientation_codes(original_affine)
        original_dtype = str(image.header.get_data_dtype())

        with materialize_nifti_path(image_path, archive_zip) as materialized_path:
            state: dict[str, Any] = {"image": str(materialized_path)}
            step_summaries: dict[str, dict[str, Any]] = {}
            resized_array: np.ndarray | None = None
            for step_name, transform in transforms:
                state = transform(state)
                step_summaries[step_name] = collect_step_summary(step_name, state)
                if step_name == "resize":
                    resized_image = state["image"]
                    resized_array = (
                        resized_image.detach().cpu().numpy()
                        if hasattr(resized_image, "detach")
                        else np.asarray(resized_image)
                    )

        inferred_spacing = infer_spacing_after_resize(original_shape, original_spacing, (96, 96, 96))
        resize_affine = step_summaries["resize"].get("affine")
        resize_meta_note = (
            "Resize updates MetaTensor affine scaling but leaves pixdim/spatial_shape metadata at original values."
            if resize_affine is not None
            else "Resize result did not expose affine metadata."
        )

        records.append(
            {
                "patient_id": row["patient_id"],
                "modality": row["modality"],
                "image_path": row["image_path"],
                "label": int(row["label"]),
                "split_source": row["split_source"],
                "source": source,
                "original_shape": list(original_shape),
                "original_spacing": list(original_spacing),
                "original_affine": original_affine.tolist(),
                "original_affine_diag": [float(original_affine[i, i]) for i in range(3)],
                "original_orientation": original_orientation,
                "original_dtype": original_dtype,
                **{f"original_{key}": value for key, value in array_stats(original_array).items()},
                "shape_after_load": step_summaries["load"]["shape"],
                "shape_after_channel": step_summaries["channel"]["shape"],
                "shape_after_resize": step_summaries["resize"]["shape"],
                "shape_after_normalize": step_summaries["normalize"]["shape"],
                "final_tensor_shape": step_summaries["tensor"]["shape"],
                "final_tensor_dtype": step_summaries["tensor"]["dtype"],
                "final_tensor_min": step_summaries["tensor"]["min"],
                "final_tensor_max": step_summaries["tensor"]["max"],
                "final_tensor_mean": step_summaries["tensor"]["mean"],
                "final_tensor_std": step_summaries["tensor"]["std"],
                "load_meta": step_summaries["load"],
                "channel_meta": step_summaries["channel"],
                "resize_meta": step_summaries["resize"],
                "normalize_meta": step_summaries["normalize"],
                "tensor_meta": step_summaries["tensor"],
                "inferred_spacing_after_resize": inferred_spacing,
                "notes": resize_meta_note,
                "original_array": original_array,
                "resized_array": resized_array if resized_array is not None else np.empty((1, 96, 96, 96), dtype=np.float32),
            }
        )

    json_path = output_dir / "preprocessing_audit.json"
    csv_path = output_dir / "preprocessing_audit.csv"
    figure_path = output_dir / "sample_slices_before_after.png"

    serializable_records = []
    for record in records:
        serializable = dict(record)
        serializable.pop("original_array", None)
        serializable.pop("resized_array", None)
        serializable_records.append(serializable)

    json_path.write_text(json.dumps(serializable_records, indent=2))

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_to_csv_dicts(serializable_records)[0].keys()))
        writer.writeheader()
        writer.writerows(rows_to_csv_dicts(serializable_records))

    save_slice_grid(records, figure_path)

    print(f"audit_samples: {len(records)}")
    print(f"archive_zip: {archive_zip if archive_zip is not None else 'not used'}")
    print(f"modalities: {[record['modality'] for record in records]}")
    print(f"json_report: {json_path}")
    print(f"csv_report: {csv_path}")
    print(f"slice_figure: {figure_path}")
    for record in records:
        print(
            f"{record['patient_id']} {record['modality']}: "
            f"original={tuple(record['original_shape'])} spacing={tuple(record['original_spacing'])} "
            f"-> resized={tuple(record['shape_after_resize'])} "
            f"inferred_spacing={tuple(round(value, 6) for value in record['inferred_spacing_after_resize'])}"
        )


if __name__ == "__main__":
    main()
