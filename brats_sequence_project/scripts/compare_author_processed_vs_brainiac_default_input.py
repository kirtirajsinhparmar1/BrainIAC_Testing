#!/usr/bin/env python3
"""Compare author processed NIfTI files with our BraTS images after BrainIAC input transforms.

This diagnostic intentionally uses the BrainIAC author validation input transform:

LoadImaged -> EnsureChannelFirstd -> Resized(96,96,96) ->
NormalizeIntensityd(nonzero=True, channel_wise=True) -> ToTensord

It does not use this project's crop_pad_zscore preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_AUTHOR_PROCESSED_DIR = "/content/BrainIAC_Testing/src/data/sample/processed"
DEFAULT_BRATS_ROOT = "/content/data"
DEFAULT_OUTPUT_DIR = (
    "/content/BrainIAC_Testing/brats_sequence_project/"
    "outputs/author_vs_ours_brainiac_default_input"
)

MODALITY_TO_SUFFIX = {
    "T1": "_t1.nii",
    "T2": "_t2.nii",
    "FLAIR": "_flair.nii",
    "T1CE": "_t1ce.nii",
}

BRATS_SOURCE_DIRS = {
    "BraTS2020_TrainingData": "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    "BraTS2020_ValidationData": "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",
}

RAW_STATS_FIELDS = [
    "source_group",
    "image_id",
    "modality",
    "path",
    "exists",
    "shape",
    "spacing",
    "orientation",
    "dtype",
    "raw_min",
    "raw_max",
    "raw_mean",
    "raw_std",
    "raw_p0",
    "raw_p0_5",
    "raw_p1",
    "raw_p5",
    "raw_p50",
    "raw_p95",
    "raw_p99",
    "raw_p99_5",
    "raw_p100",
    "nonzero_count",
    "total_voxel_count",
    "nonzero_fraction",
    "nonzero_min",
    "nonzero_max",
    "nonzero_mean",
    "nonzero_std",
    "nonzero_p1",
    "nonzero_p5",
    "nonzero_p50",
    "nonzero_p95",
    "nonzero_p99",
    "negative_count",
    "nan_count",
    "inf_count",
]

TENSOR_STATS_FIELDS = [
    "source_group",
    "image_id",
    "modality",
    "path",
    "tensor_shape",
    "tensor_min",
    "tensor_max",
    "tensor_mean",
    "tensor_std",
    "tensor_p0",
    "tensor_p0_5",
    "tensor_p1",
    "tensor_p5",
    "tensor_p50",
    "tensor_p95",
    "tensor_p99",
    "tensor_p99_5",
    "tensor_p100",
    "tensor_nonzero_count",
    "tensor_nonzero_fraction",
    "tensor_nonzero_mean",
    "tensor_nonzero_std",
    "tensor_negative_count",
    "tensor_nan_count",
    "tensor_inf_count",
]

TRANSFORM_DESCRIPTION = (
    "LoadImaged(keys=['image']) -> EnsureChannelFirstd(keys=['image']) -> "
    "Resized(keys=['image'], spatial_size={image_size}, mode='trilinear') -> "
    "NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True) -> "
    "ToTensord(keys=['image'])"
)

MAX_PLOT_IMAGES = 24


@dataclass(frozen=True)
class ImageRecord:
    source_group: str
    image_id: str
    modality: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BrainIAC author sample processed NIfTI images with our images "
            "after the original BrainIAC default validation input transform."
        )
    )
    parser.add_argument(
        "--author_processed_dir",
        default=DEFAULT_AUTHOR_PROCESSED_DIR,
        help="Folder containing BrainIAC author sample processed .nii/.nii.gz files.",
    )
    parser.add_argument(
        "--brats_root",
        default=DEFAULT_BRATS_ROOT,
        help="Root folder for the BraTS2020 Kaggle dataset.",
    )
    parser.add_argument(
        "--n4_csv",
        default=None,
        help="Optional CSV from N4-only preprocessing containing image_path rows.",
    )
    parser.add_argument(
        "--brainiac_style_csv",
        default=None,
        help="Optional CSV from BrainIAC-style preprocessing containing image_path rows.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/JSON/PNG comparison outputs will be saved.",
    )
    parser.add_argument("--max_author_images", type=int, default=8)
    parser.add_argument("--max_brats_patients", type=int, default=4)
    parser.add_argument("--max_rows_per_csv", type=int, default=16)
    parser.add_argument(
        "--image_size",
        default="96,96,96",
        help="BrainIAC input image size as comma-separated integers, e.g. 96,96,96.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(include_original_brats=True)
    parser.add_argument(
        "--include_original_brats",
        dest="include_original_brats",
        action="store_true",
        help="Include original BraTS images. This is the default.",
    )
    parser.add_argument(
        "--no_include_original_brats",
        dest="include_original_brats",
        action="store_false",
        help="Do not include original BraTS images.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Save sampled transformed tensors as a compressed NPZ.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selected files and exit without computing statistics or figures.",
    )
    return parser.parse_args()


def import_runtime_dependencies() -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import nibabel as nib
        import numpy as np
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            LoadImaged,
            NormalizeIntensityd,
            Resized,
            ToTensord,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing required dependency for this diagnostic script. "
            "Install the BrainIAC runtime dependencies, including numpy, nibabel, "
            "matplotlib, torch, and MONAI. Original error: "
            f"{exc}"
        ) from exc

    return {
        "np": np,
        "nib": nib,
        "plt": plt,
        "Compose": Compose,
        "LoadImaged": LoadImaged,
        "EnsureChannelFirstd": EnsureChannelFirstd,
        "Resized": Resized,
        "NormalizeIntensityd": NormalizeIntensityd,
        "ToTensord": ToTensord,
    }


def parse_image_size(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"--image_size must have three comma-separated integers: {value}")
    try:
        size = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"--image_size must have integer values: {value}") from exc
    if any(dim <= 0 for dim in size):
        raise ValueError(f"--image_size values must be positive: {value}")
    return size  # type: ignore[return-value]


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def infer_modality(filename: str) -> str:
    lower_name = filename.lower()
    compact = re.sub(r"[^a-z0-9]+", "_", lower_name)
    tokens = [token for token in compact.split("_") if token]

    if "t1ce" in tokens or "t1c" in tokens or "t1ce" in lower_name or "t1c" in lower_name:
        return "T1CE"
    if "t2f" in tokens or "flair" in tokens or "t2f" in lower_name or "flair" in lower_name:
        return "FLAIR"
    if "t2w" in tokens or "t2" in tokens or "t2w" in lower_name:
        return "T2"
    if "t1n" in tokens or "t1w" in tokens or "t1" in tokens or "t1n" in lower_name or "t1w" in lower_name:
        return "T1"
    return "UNKNOWN"


def modality_from_csv_row(row: dict[str, str], path: Path) -> str:
    modality = row.get("modality", "").strip()
    if modality:
        normalized = modality.upper()
        if normalized in {"T1", "T2", "FLAIR", "T1CE"}:
            return normalized
    return infer_modality(path.name)


def discover_author_records(author_processed_dir: Path, max_author_images: int) -> list[ImageRecord]:
    if not author_processed_dir.exists():
        raise SystemExit(
            f"Author processed directory does not exist: {author_processed_dir}\n"
            "Point --author_processed_dir to the BrainIAC author sample processed folder, "
            "typically src/data/sample/processed inside the BrainIAC repo."
        )
    if not author_processed_dir.is_dir():
        raise SystemExit(f"--author_processed_dir is not a directory: {author_processed_dir}")

    files = sorted(set(author_processed_dir.glob("*.nii")) | set(author_processed_dir.glob("*.nii.gz")))
    if not files:
        raise SystemExit(
            f"No .nii or .nii.gz files found in --author_processed_dir: {author_processed_dir}"
        )
    selected = files[: max(0, max_author_images)]
    return [
        ImageRecord(
            source_group="author_processed",
            image_id=strip_nii_suffix(path),
            modality=infer_modality(path.name),
            path=path,
        )
        for path in selected
    ]


def discover_brats_records(
    brats_root: Path,
    max_brats_patients: int,
    seed: int,
) -> list[ImageRecord]:
    if not brats_root.exists():
        raise SystemExit(
            f"BraTS root does not exist: {brats_root}\n"
            "Use --no_include_original_brats to compare author processed images only, "
            "or point --brats_root to the BraTS2020 Kaggle dataset root."
        )

    patient_entries: list[tuple[str, str, Path]] = []
    for split_source, relative_dir in BRATS_SOURCE_DIRS.items():
        split_dir = brats_root / relative_dir
        if not split_dir.exists():
            print(f"WARNING: BraTS split directory not found, skipping: {split_dir}", file=sys.stderr)
            continue
        for patient_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            patient_entries.append((split_source, patient_dir.name, patient_dir))

    if not patient_entries:
        raise SystemExit(
            f"No BraTS patient folders found under {brats_root}. "
            "Expected BraTS2020_TrainingData and/or BraTS2020_ValidationData folders."
        )

    rng = random.Random(seed)
    patient_entries = sorted(patient_entries, key=lambda item: (item[0], item[1]))
    rng.shuffle(patient_entries)
    selected_patients = patient_entries[: max(0, max_brats_patients)]

    records: list[ImageRecord] = []
    for split_source, patient_id, patient_dir in selected_patients:
        for modality, suffix in MODALITY_TO_SUFFIX.items():
            image_path = patient_dir / f"{patient_id}{suffix}"
            if not image_path.exists():
                print(
                    f"WARNING: expected {modality} file missing for {patient_id}: {image_path}",
                    file=sys.stderr,
                )
                continue
            records.append(
                ImageRecord(
                    source_group="our_original_brats",
                    image_id=f"{split_source}_{patient_id}_{modality}",
                    modality=modality,
                    path=image_path,
                )
            )
    return records


def read_csv_records(
    csv_path: Path | None,
    source_group: str,
    max_rows: int,
) -> list[ImageRecord]:
    if csv_path is None:
        return []
    if not csv_path.exists():
        print(f"WARNING: {source_group} CSV does not exist, skipping: {csv_path}", file=sys.stderr)
        return []

    records: list[ImageRecord] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_path" not in (reader.fieldnames or []):
            raise SystemExit(f"{source_group} CSV is missing required image_path column: {csv_path}")
        for index, row in enumerate(reader):
            if len(records) >= max_rows:
                break
            raw_path = row.get("image_path", "").strip()
            if not raw_path:
                continue
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = (csv_path.parent / image_path).resolve()
            patient_id = row.get("patient_id", "").strip()
            image_id = patient_id or strip_nii_suffix(image_path)
            modality = modality_from_csv_row(row, image_path)
            records.append(
                ImageRecord(
                    source_group=source_group,
                    image_id=f"{image_id}_{modality}_{index:04d}",
                    modality=modality,
                    path=image_path,
                )
            )
    return records


def collect_records(args: argparse.Namespace) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    records.extend(discover_author_records(Path(args.author_processed_dir), args.max_author_images))
    if args.include_original_brats:
        records.extend(
            discover_brats_records(Path(args.brats_root), args.max_brats_patients, args.seed)
        )
    records.extend(
        read_csv_records(Path(args.n4_csv) if args.n4_csv else None, "our_n4_only", args.max_rows_per_csv)
    )
    records.extend(
        read_csv_records(
            Path(args.brainiac_style_csv) if args.brainiac_style_csv else None,
            "our_brainiac_style",
            args.max_rows_per_csv,
        )
    )
    if not records:
        raise SystemExit("No images selected for comparison.")
    return records


def print_dry_run(records: list[ImageRecord]) -> None:
    print("Dry run: selected images")
    for index, record in enumerate(records, start=1):
        print(
            f"{index:03d} | {record.source_group} | {record.modality} | "
            f"{record.image_id} | {record.path}"
        )
    counts = Counter(record.source_group for record in records)
    print("\nCounts by source group:")
    for source_group, count in sorted(counts.items()):
        print(f"  {source_group}: {count}")


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return str(value)


def stringify_for_csv(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(to_jsonable(value))
    return value


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify_for_csv(row.get(field, "")) for field in fieldnames})


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(to_jsonable(data), handle, indent=2)


def finite_values(np: Any, array: Any) -> Any:
    flat = np.asarray(array).reshape(-1)
    return flat[np.isfinite(flat)]


def percentile_dict(np: Any, values: Any, prefix: str) -> dict[str, float | None]:
    if values.size == 0:
        return {
            f"{prefix}_p0": None,
            f"{prefix}_p0_5": None,
            f"{prefix}_p1": None,
            f"{prefix}_p5": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
            f"{prefix}_p99_5": None,
            f"{prefix}_p100": None,
        }
    percentiles = np.percentile(values, [0, 0.5, 1, 5, 50, 95, 99, 99.5, 100])
    return {
        f"{prefix}_p0": float(percentiles[0]),
        f"{prefix}_p0_5": float(percentiles[1]),
        f"{prefix}_p1": float(percentiles[2]),
        f"{prefix}_p5": float(percentiles[3]),
        f"{prefix}_p50": float(percentiles[4]),
        f"{prefix}_p95": float(percentiles[5]),
        f"{prefix}_p99": float(percentiles[6]),
        f"{prefix}_p99_5": float(percentiles[7]),
        f"{prefix}_p100": float(percentiles[8]),
    }


def load_nifti_3d(np: Any, nib: Any, path: Path) -> tuple[Any, Any]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    data = np.asarray(data)
    data = np.squeeze(data)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image after squeeze, got shape {data.shape} for {path}")
    return image, data


def compute_raw_stats(np: Any, nib: Any, record: ImageRecord) -> dict[str, Any]:
    base: dict[str, Any] = {
        "source_group": record.source_group,
        "image_id": record.image_id,
        "modality": record.modality,
        "path": str(record.path),
        "exists": record.path.exists(),
    }
    if not record.path.exists():
        return base

    image, data = load_nifti_3d(np, nib, record.path)
    finite = finite_values(np, data)
    nonzero = finite[finite != 0]
    total_voxels = int(data.size)
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())
    negative_count = int((finite < 0).sum())

    stats = {
        **base,
        "shape": list(data.shape),
        "spacing": list(image.header.get_zooms()[:3]),
        "orientation": list(nib.aff2axcodes(image.affine)),
        "dtype": str(image.header.get_data_dtype()),
        "total_voxel_count": total_voxels,
        "nonzero_count": int(nonzero.size),
        "nonzero_fraction": float(nonzero.size / total_voxels) if total_voxels else None,
        "negative_count": negative_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }

    if finite.size:
        stats.update(
            {
                "raw_min": float(np.min(finite)),
                "raw_max": float(np.max(finite)),
                "raw_mean": float(np.mean(finite)),
                "raw_std": float(np.std(finite)),
            }
        )
        stats.update(percentile_dict(np, finite, "raw"))
    else:
        stats.update(
            {
                "raw_min": None,
                "raw_max": None,
                "raw_mean": None,
                "raw_std": None,
            }
        )
        stats.update(percentile_dict(np, finite, "raw"))

    if nonzero.size:
        stats.update(
            {
                "nonzero_min": float(np.min(nonzero)),
                "nonzero_max": float(np.max(nonzero)),
                "nonzero_mean": float(np.mean(nonzero)),
                "nonzero_std": float(np.std(nonzero)),
            }
        )
        nonzero_percentiles = np.percentile(nonzero, [1, 5, 50, 95, 99])
        stats.update(
            {
                "nonzero_p1": float(nonzero_percentiles[0]),
                "nonzero_p5": float(nonzero_percentiles[1]),
                "nonzero_p50": float(nonzero_percentiles[2]),
                "nonzero_p95": float(nonzero_percentiles[3]),
                "nonzero_p99": float(nonzero_percentiles[4]),
            }
        )
    else:
        stats.update(
            {
                "nonzero_min": None,
                "nonzero_max": None,
                "nonzero_mean": None,
                "nonzero_std": None,
                "nonzero_p1": None,
                "nonzero_p5": None,
                "nonzero_p50": None,
                "nonzero_p95": None,
                "nonzero_p99": None,
            }
        )

    return stats


def collect_global_raw_window_values(np: Any, nib: Any, records: list[ImageRecord]) -> dict[str, Any]:
    raw_min_values: list[float] = []
    raw_max_values: list[float] = []
    nonzero_chunks: list[Any] = []

    for record in records:
        if not record.path.exists():
            continue
        _, data = load_nifti_3d(np, nib, record.path)
        finite = finite_values(np, data)
        if finite.size:
            raw_min_values.append(float(np.min(finite)))
            raw_max_values.append(float(np.max(finite)))
            nonzero = finite[finite != 0]
            if nonzero.size:
                nonzero_chunks.append(nonzero.astype(np.float32, copy=True))

    if not raw_min_values or not raw_max_values:
        return {
            "global_raw_min": 0.0,
            "global_raw_max": 1.0,
            "global_nonzero_p1": 0.0,
            "global_nonzero_p99": 1.0,
        }

    if nonzero_chunks:
        all_nonzero = np.concatenate(nonzero_chunks)
        global_nonzero_p1, global_nonzero_p99 = np.percentile(all_nonzero, [1, 99])
    else:
        global_nonzero_p1 = min(raw_min_values)
        global_nonzero_p99 = max(raw_max_values)

    return {
        "global_raw_min": float(min(raw_min_values)),
        "global_raw_max": float(max(raw_max_values)),
        "global_nonzero_p1": float(global_nonzero_p1),
        "global_nonzero_p99": float(global_nonzero_p99),
    }


def build_brainiac_default_transform(deps: dict[str, Any], image_size: tuple[int, int, int]) -> Any:
    return deps["Compose"](
        [
            deps["LoadImaged"](keys=["image"]),
            deps["EnsureChannelFirstd"](keys=["image"]),
            deps["Resized"](keys=["image"], spatial_size=image_size, mode="trilinear"),
            deps["NormalizeIntensityd"](keys=["image"], nonzero=True, channel_wise=True),
            deps["ToTensord"](keys=["image"]),
        ]
    )


def tensor_to_numpy(np: Any, tensor: Any) -> Any:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return np.asarray(tensor.numpy(), dtype=np.float32)
    return np.asarray(tensor, dtype=np.float32)


def apply_brainiac_transform(
    np: Any,
    transform: Any,
    record: ImageRecord,
) -> tuple[Any | None, dict[str, Any]]:
    base: dict[str, Any] = {
        "source_group": record.source_group,
        "image_id": record.image_id,
        "modality": record.modality,
        "path": str(record.path),
    }
    if not record.path.exists():
        return None, {**base, "error": "path_missing"}

    transformed = transform({"image": str(record.path)})["image"]
    tensor = tensor_to_numpy(np, transformed)
    if tensor.ndim == 3:
        tensor = tensor[np.newaxis, ...]
    finite = finite_values(np, tensor)
    nonzero = finite[finite != 0]
    total = int(tensor.size)

    stats: dict[str, Any] = {
        **base,
        "tensor_shape": list(tensor.shape),
        "tensor_nonzero_count": int(nonzero.size),
        "tensor_nonzero_fraction": float(nonzero.size / total) if total else None,
        "tensor_negative_count": int((finite < 0).sum()),
        "tensor_nan_count": int(np.isnan(tensor).sum()),
        "tensor_inf_count": int(np.isinf(tensor).sum()),
    }

    if finite.size:
        stats.update(
            {
                "tensor_min": float(np.min(finite)),
                "tensor_max": float(np.max(finite)),
                "tensor_mean": float(np.mean(finite)),
                "tensor_std": float(np.std(finite)),
            }
        )
        stats.update(percentile_dict(np, finite, "tensor"))
    else:
        stats.update(
            {
                "tensor_min": None,
                "tensor_max": None,
                "tensor_mean": None,
                "tensor_std": None,
            }
        )
        stats.update(percentile_dict(np, finite, "tensor"))

    if nonzero.size:
        stats.update(
            {
                "tensor_nonzero_mean": float(np.mean(nonzero)),
                "tensor_nonzero_std": float(np.std(nonzero)),
            }
        )
    else:
        stats.update(
            {
                "tensor_nonzero_mean": None,
                "tensor_nonzero_std": None,
            }
        )

    return tensor, stats


def orientation_for_record(nib: Any, record: ImageRecord) -> list[str]:
    if not record.path.exists():
        return ["NA"]
    image = nib.load(str(record.path))
    return list(nib.aff2axcodes(image.affine))


def canonical_center_slices(np: Any, nib: Any, record: ImageRecord) -> tuple[list[Any], dict[str, Any]]:
    image = nib.load(str(record.path))
    canonical = nib.as_closest_canonical(image)
    data = canonical.get_fdata(dtype=np.float32)
    data = np.asarray(data)
    data = np.squeeze(data)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D canonical data, got {data.shape} for {record.path}")
    sagittal = data[data.shape[0] // 2, :, :]
    coronal = data[:, data.shape[1] // 2, :]
    axial = data[:, :, data.shape[2] // 2]
    metadata = {
        "shape": list(data.shape),
        "orientation": list(nib.aff2axcodes(image.affine)),
        "canonical_orientation": list(nib.aff2axcodes(canonical.affine)),
        "actual_min": float(np.nanmin(data)),
        "actual_max": float(np.nanmax(data)),
    }
    return [sagittal, coronal, axial], metadata


def tensor_center_slices(np: Any, tensor: Any) -> list[Any]:
    array = np.asarray(tensor)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected tensor shape [1,D,H,W] or [D,H,W], got {tensor.shape}")
    sagittal = array[array.shape[0] // 2, :, :]
    coronal = array[:, array.shape[1] // 2, :]
    axial = array[:, :, array.shape[2] // 2]
    return [sagittal, coronal, axial]


def plot_slice_grid(
    np: Any,
    plt: Any,
    output_path: Path,
    rows: list[dict[str, Any]],
    suptitle: str,
    fixed_vmin: float | None = None,
    fixed_vmax: float | None = None,
    dpi: int = 180,
) -> None:
    if not rows:
        return

    view_names = ["sagittal", "coronal", "axial"]
    n_rows = len(rows)
    fig_width = 13.5
    fig_height = max(2.4, min(2.0 * n_rows, 42.0))
    fig, axes = plt.subplots(n_rows, 3, figsize=(fig_width, fig_height), squeeze=False)

    for row_index, row in enumerate(rows):
        vmin = float(row["vmin"] if fixed_vmin is None else fixed_vmin)
        vmax = float(row["vmax"] if fixed_vmax is None else fixed_vmax)
        for col_index, view_name in enumerate(view_names):
            axis = axes[row_index][col_index]
            image_slice = np.rot90(row["slices"][col_index])
            axis.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axis.axis("off")
            axis.set_title(
                (
                    f"{view_name} | {row['basename']}\n"
                    f"{row['source_group']} | {row['modality']} | "
                    f"shape={row['shape']} | orient={row['orientation']}\n"
                    f"actual=[{row['actual_min']:.3g},{row['actual_max']:.3g}] | "
                    f"display=[{vmin:.3g},{vmax:.3g}]\n"
                    f"{row['window_label']}"
                ),
                fontsize=6,
            )

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def make_raw_plot_rows(
    np: Any,
    nib: Any,
    records: list[ImageRecord],
    raw_stats_by_path: dict[str, dict[str, Any]],
    window_mode: str,
    shared_vmin: float | None = None,
    shared_vmax: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records[:MAX_PLOT_IMAGES]:
        if not record.path.exists():
            continue
        slices, metadata = canonical_center_slices(np, nib, record)
        raw_stats = raw_stats_by_path.get(str(record.path), {})
        if window_mode == "per_image_nonzero_p1_p99":
            vmin = raw_stats.get("nonzero_p1")
            vmax = raw_stats.get("nonzero_p99")
            if vmin is None or vmax is None or vmin == vmax:
                vmin = raw_stats.get("raw_min", 0.0)
                vmax = raw_stats.get("raw_max", 1.0)
        else:
            vmin = shared_vmin
            vmax = shared_vmax
        if vmin is None or vmax is None or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        rows.append(
            {
                "source_group": record.source_group,
                "modality": record.modality,
                "basename": record.path.name,
                "shape": metadata["shape"],
                "orientation": metadata["orientation"],
                "actual_min": metadata["actual_min"],
                "actual_max": metadata["actual_max"],
                "vmin": float(vmin),
                "vmax": float(vmax),
                "window_label": window_mode,
                "slices": slices,
            }
        )
    return rows


def make_tensor_plot_rows(
    np: Any,
    records: list[ImageRecord],
    tensors_by_path: dict[str, Any],
    tensor_stats_by_path: dict[str, dict[str, Any]],
    window_label: str,
    shared_vmin: float | None = None,
    shared_vmax: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records[:MAX_PLOT_IMAGES]:
        tensor = tensors_by_path.get(str(record.path))
        if tensor is None:
            continue
        stats = tensor_stats_by_path.get(str(record.path), {})
        if shared_vmin is None or shared_vmax is None:
            vmin = stats.get("tensor_p1", -3.0)
            vmax = stats.get("tensor_p99", 3.0)
        else:
            vmin = shared_vmin
            vmax = shared_vmax
        if vmin == vmax:
            vmin, vmax = -3.0, 3.0
        rows.append(
            {
                "source_group": record.source_group,
                "modality": record.modality,
                "basename": record.path.name,
                "shape": stats.get("tensor_shape", list(tensor.shape)),
                "orientation": "BrainIAC default tensor path, no Orientationd",
                "actual_min": float(stats.get("tensor_min", 0.0)),
                "actual_max": float(stats.get("tensor_max", 0.0)),
                "vmin": float(vmin),
                "vmax": float(vmax),
                "window_label": window_label,
                "slices": tensor_center_slices(np, tensor),
            }
        )
    return rows


def summarize_batch(
    np: Any,
    records: list[ImageRecord],
    tensors_by_path: dict[str, Any],
    image_size: tuple[int, int, int],
) -> dict[str, Any]:
    tensors = [tensors_by_path[str(record.path)] for record in records if str(record.path) in tensors_by_path]
    if tensors:
        batch = np.stack(tensors, axis=0)
        batch_summary = {
            "number_of_images": int(batch.shape[0]),
            "batch_shape": list(batch.shape),
            "batch_min": float(np.nanmin(batch)),
            "batch_max": float(np.nanmax(batch)),
            "batch_mean": float(np.nanmean(batch)),
            "batch_std": float(np.nanstd(batch)),
        }
    else:
        batch_summary = {
            "number_of_images": 0,
            "batch_shape": [],
            "batch_min": None,
            "batch_max": None,
            "batch_mean": None,
            "batch_std": None,
        }

    source_counts = Counter(record.source_group for record in records if str(record.path) in tensors_by_path)
    modality_counts = Counter(record.modality for record in records if str(record.path) in tensors_by_path)
    batch_summary.update(
        {
            "per_source_counts": dict(sorted(source_counts.items())),
            "per_modality_counts": dict(sorted(modality_counts.items())),
            "transform_description": TRANSFORM_DESCRIPTION.format(image_size=image_size),
        }
    )
    return batch_summary


def summarize_by_source(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for source_group, group_rows in grouped.items():
        source_summary: dict[str, Any] = {"count": len(group_rows)}
        for key in keys:
            values = [row.get(key) for row in group_rows if isinstance(row.get(key), (int, float))]
            if values:
                source_summary[f"{key}_min"] = min(values)
                source_summary[f"{key}_max"] = max(values)
                source_summary[f"{key}_mean"] = sum(values) / len(values)
        summary[source_group] = source_summary
    return summary


def print_console_summary(
    raw_rows: list[dict[str, Any]],
    tensor_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    source_counts = Counter(row["source_group"] for row in raw_rows)
    raw_summary = summarize_by_source(raw_rows, ["raw_min", "raw_max"])
    tensor_summary = summarize_by_source(
        tensor_rows,
        ["tensor_min", "tensor_max", "tensor_nonzero_mean", "tensor_nonzero_std"],
    )

    print("\nComparison summary")
    print("Images by source group:")
    for source_group, count in sorted(source_counts.items()):
        print(f"  {source_group}: {count}")

    print("\nRaw intensity ranges by source group:")
    for source_group, summary in sorted(raw_summary.items()):
        raw_min = summary.get("raw_min_min")
        raw_max = summary.get("raw_max_max")
        print(f"  {source_group}: raw_min={raw_min}, raw_max={raw_max}")

    print("\nBrainIAC default tensor ranges and nonzero normalization by source group:")
    for source_group, summary in sorted(tensor_summary.items()):
        tensor_min = summary.get("tensor_min_min")
        tensor_max = summary.get("tensor_max_max")
        nz_mean = summary.get("tensor_nonzero_mean_mean")
        nz_std = summary.get("tensor_nonzero_std_mean")
        normalized = "unknown"
        if isinstance(nz_mean, (int, float)) and isinstance(nz_std, (int, float)):
            normalized = "yes" if abs(nz_mean) < 0.1 and 0.8 <= nz_std <= 1.2 else "check"
        print(
            f"  {source_group}: tensor_min={tensor_min}, tensor_max={tensor_max}, "
            f"nonzero_mean={nz_mean}, nonzero_std={nz_std}, approx_normalized={normalized}"
        )

    print(f"\nOutputs saved under: {output_dir}")


def main() -> int:
    args = parse_args()
    image_size = parse_image_size(args.image_size)
    records = collect_records(args)

    if args.dry_run:
        print_dry_run(records)
        return 0

    deps = import_runtime_dependencies()
    np = deps["np"]
    nib = deps["nib"]
    plt = deps["plt"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(records)} images for comparison.")
    print("Computing raw NIfTI statistics...")
    raw_rows = [compute_raw_stats(np, nib, record) for record in records]
    save_csv(output_dir / "raw_nifti_stats.csv", raw_rows, RAW_STATS_FIELDS)
    save_json(output_dir / "raw_nifti_stats.json", raw_rows)

    raw_stats_by_path = {row["path"]: row for row in raw_rows}
    raw_windows = collect_global_raw_window_values(np, nib, records)
    save_json(output_dir / "display_window_summary.json", raw_windows)

    print("Applying BrainIAC default validation transform...")
    transform = build_brainiac_default_transform(deps, image_size)
    tensor_rows: list[dict[str, Any]] = []
    tensors_by_path: dict[str, Any] = {}
    for record in records:
        try:
            tensor, stats = apply_brainiac_transform(np, transform, record)
        except Exception as exc:  # noqa: BLE001 - record transform failures and continue.
            print(
                f"WARNING: BrainIAC default transform failed for {record.path}: {exc}",
                file=sys.stderr,
            )
            stats = {
                "source_group": record.source_group,
                "image_id": record.image_id,
                "modality": record.modality,
                "path": str(record.path),
                "error": str(exc),
            }
            tensor = None
        tensor_rows.append(stats)
        if tensor is not None:
            tensors_by_path[str(record.path)] = tensor

    save_csv(output_dir / "brainiac_default_tensor_stats.csv", tensor_rows, TENSOR_STATS_FIELDS)
    save_json(output_dir / "brainiac_default_tensor_stats.json", tensor_rows)

    tensor_stats_by_path = {row["path"]: row for row in tensor_rows}
    tensor_chunks = [
        np.asarray(tensors_by_path[str(record.path)]).reshape(-1)
        for record in records
        if str(record.path) in tensors_by_path
    ]
    if tensor_chunks:
        all_tensor_values = np.concatenate(tensor_chunks)
        tensor_shared_p1, tensor_shared_p99 = np.percentile(all_tensor_values, [1, 99])
    else:
        tensor_shared_p1, tensor_shared_p99 = -3.0, 3.0

    batch_summary = summarize_batch(np, records, tensors_by_path, image_size)
    save_json(output_dir / "batch_debug_summary.json", batch_summary)

    if args.save_npz and tensors_by_path:
        tensors = [tensors_by_path[str(record.path)] for record in records if str(record.path) in tensors_by_path]
        np.savez_compressed(
            output_dir / "sampled_tensors_brainiac_default_input.npz",
            tensors=np.stack(tensors, axis=0),
            source_group=np.array([record.source_group for record in records if str(record.path) in tensors_by_path]),
            image_id=np.array([record.image_id for record in records if str(record.path) in tensors_by_path]),
            modality=np.array([record.modality for record in records if str(record.path) in tensors_by_path]),
            path=np.array([str(record.path) for record in records if str(record.path) in tensors_by_path]),
        )

    print("Creating visualization PNGs...")
    raw_global_rows = make_raw_plot_rows(
        np,
        nib,
        records,
        raw_stats_by_path,
        "global raw window",
        raw_windows["global_raw_min"],
        raw_windows["global_raw_max"],
    )
    plot_slice_grid(
        np,
        plt,
        output_dir / "raw_fixed_global_window.png",
        raw_global_rows,
        "Raw NIfTI comparison - one global min/max window across all compared images",
    )

    raw_shared_rows = make_raw_plot_rows(
        np,
        nib,
        records,
        raw_stats_by_path,
        "shared nonzero p1/p99 raw window",
        raw_windows["global_nonzero_p1"],
        raw_windows["global_nonzero_p99"],
    )
    plot_slice_grid(
        np,
        plt,
        output_dir / "raw_shared_nonzero_p1_p99_window.png",
        raw_shared_rows,
        "Raw NIfTI comparison - shared nonzero p1/p99 display window",
    )

    raw_per_image_rows = make_raw_plot_rows(
        np,
        nib,
        records,
        raw_stats_by_path,
        "per-image nonzero p1/p99 - ANATOMY ONLY",
    )
    plot_slice_grid(
        np,
        plt,
        output_dir / "raw_per_image_p1_p99_ANATOMY_ONLY.png",
        raw_per_image_rows,
        "ANATOMY ONLY - NOT VALID FOR INTENSITY SCALE COMPARISON",
    )

    tensor_fixed_rows = make_tensor_plot_rows(
        np,
        records,
        tensors_by_path,
        tensor_stats_by_path,
        "normalized tensor fixed window [-3,3]",
        -3.0,
        3.0,
    )
    plot_slice_grid(
        np,
        plt,
        output_dir / "brainiac_default_tensor_fixed_minus3_to3.png",
        tensor_fixed_rows,
        "BrainIAC default model-input tensors - fixed display window [-3,3]",
        fixed_vmin=-3.0,
        fixed_vmax=3.0,
    )

    tensor_shared_rows = make_tensor_plot_rows(
        np,
        records,
        tensors_by_path,
        tensor_stats_by_path,
        "normalized tensor shared p1/p99",
        float(tensor_shared_p1),
        float(tensor_shared_p99),
    )
    plot_slice_grid(
        np,
        plt,
        output_dir / "brainiac_default_tensor_shared_p1_p99.png",
        tensor_shared_rows,
        "BrainIAC default model-input tensors - shared p1/p99 display window",
    )

    batch_debug_rows = make_tensor_plot_rows(
        np,
        records,
        tensors_by_path,
        tensor_stats_by_path,
        "exact BrainIAC default input batch, fixed [-3,3]",
        -3.0,
        3.0,
    )
    plot_slice_grid(
        np,
        plt,
        output_dir / "batch_debug_exact_brainiac_default_input.png",
        batch_debug_rows,
        "Batch debug - exact BrainIAC default validation input tensors",
        fixed_vmin=-3.0,
        fixed_vmax=3.0,
    )

    print_console_summary(raw_rows, tensor_rows, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
