"""Shared helpers for focused BraTS preprocessing audits."""

from __future__ import annotations

import csv
import io
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np


CLASS_NAMES = ["T1", "T2", "FLAIR", "T1CE"]
MODALITY_TO_LABEL = {"T1": 0, "T2": 1, "FLAIR": 2, "T1CE": 3}
MODALITY_TO_SUFFIX = {
    "T1": "_t1.nii",
    "T2": "_t2.nii",
    "FLAIR": "_flair.nii",
    "T1CE": "_t1ce.nii",
}
REQUIRED_COLUMNS = {"patient_id", "image_path", "label", "modality", "split_source"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def detect_archive_zip(csv_path: Path, requested: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            csv_path.parents[3] / "archive.zip",
            csv_path.parents[2] / "archive.zip",
            Path.cwd() / "archive.zip",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _build_archive_member(image_path: Path) -> str:
    parts = image_path.parts
    dataset_markers = ("BraTS2020_TrainingData", "BraTS2020_ValidationData")
    for marker in dataset_markers:
        if marker in parts:
            start = parts.index(marker)
            return Path(*parts[start:]).as_posix()
    if "Dataset" in parts:
        start = parts.index("Dataset") + 1
        return Path(*parts[start:]).as_posix()
    raise FileNotFoundError(f"Cannot infer archive member from path: {image_path}")


def resolve_image_source(image_path: Path, archive_zip: Path | None = None) -> tuple[str, Path | None]:
    if image_path.is_file():
        return "file", image_path

    if archive_zip is None:
        raise FileNotFoundError(
            f"Image path does not exist: {image_path}. "
            "Provide --archive_zip or restore the extracted Dataset directory."
        )

    member = _build_archive_member(image_path)
    with zipfile.ZipFile(archive_zip) as zip_file:
        if member not in zip_file.namelist():
            raise FileNotFoundError(f"{member} not found inside archive: {archive_zip}")
    return "zip", Path(member)


def load_nifti_from_source(image_path: Path, archive_zip: Path | None = None) -> tuple[nib.Nifti1Image, str]:
    source_kind, source_value = resolve_image_source(image_path, archive_zip)
    if source_kind == "file":
        return nib.load(str(source_value)), str(source_value)

    assert source_value is not None
    assert archive_zip is not None
    with zipfile.ZipFile(archive_zip) as zip_file:
        payload = zip_file.read(source_value.as_posix())
    return nib.Nifti1Image.from_bytes(payload), f"{archive_zip}!{source_value.as_posix()}"


@contextmanager
def materialize_nifti_path(image_path: Path, archive_zip: Path | None = None) -> Iterator[Path]:
    source_kind, source_value = resolve_image_source(image_path, archive_zip)
    if source_kind == "file":
        assert source_value is not None
        yield source_value
        return

    assert source_value is not None
    assert archive_zip is not None
    with zipfile.ZipFile(archive_zip) as zip_file:
        payload = zip_file.read(source_value.as_posix())

    tmp_dir = Path(tempfile.mkdtemp(prefix="brats_audit_"))
    tmp_path = tmp_dir / image_path.name
    tmp_path.write_bytes(payload)
    try:
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if tmp_dir.exists():
            tmp_dir.rmdir()


def orientation_codes(affine: np.ndarray) -> str:
    return "".join(nib.orientations.aff2axcodes(affine))


def array_stats(array: np.ndarray) -> dict[str, float]:
    stats = {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
    nonzero = array[array != 0]
    if nonzero.size:
        stats["nonzero_mean"] = float(nonzero.mean())
        stats["nonzero_std"] = float(nonzero.std())
    else:
        stats["nonzero_mean"] = 0.0
        stats["nonzero_std"] = 0.0
    return stats


def infer_spacing_after_resize(
    original_shape: tuple[int, int, int],
    original_spacing: tuple[float, float, float],
    resized_shape: tuple[int, int, int],
) -> list[float]:
    return [
        float(original_spacing[idx] * original_shape[idx] / resized_shape[idx])
        for idx in range(3)
    ]


def select_representative_rows(rows: list[dict[str, str]], num_samples: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    rows_sorted = sorted(rows, key=lambda row: (row["patient_id"], row["modality"], row["image_path"]))
    for modality in CLASS_NAMES:
        for row in rows_sorted:
            key = (row["patient_id"], row["modality"])
            if row["modality"] == modality and key not in seen_keys:
                selected.append(row)
                seen_keys.add(key)
                break

    if len(selected) >= num_samples:
        return selected[:num_samples]

    for row in rows_sorted:
        key = (row["patient_id"], row["modality"])
        if key in seen_keys:
            continue
        selected.append(row)
        seen_keys.add(key)
        if len(selected) >= num_samples:
            break

    return selected


def find_complete_patient(rows: list[dict[str, str]], requested_patient_id: str | None = None) -> tuple[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["patient_id"]][row["modality"]] = row

    if requested_patient_id is not None:
        modalities = grouped.get(requested_patient_id)
        if modalities is None or set(modalities) != set(CLASS_NAMES):
            raise ValueError(f"Patient {requested_patient_id} does not have all four modalities in the CSV.")
        return requested_patient_id, modalities

    for patient_id in sorted(grouped):
        if set(grouped[patient_id]) == set(CLASS_NAMES):
            return patient_id, grouped[patient_id]
    raise ValueError("No patient with all four modalities was found in the CSV.")


def prepare_display_slice(slice_2d: np.ndarray) -> np.ndarray:
    slice_float = slice_2d.astype(np.float32)
    positive = slice_float[slice_float > 0]
    if positive.size == 0:
        return np.zeros_like(slice_float)

    low = float(np.percentile(positive, 1))
    high = float(np.percentile(positive, 99))
    if high <= low:
        high = low + 1.0
    clipped = np.clip(slice_float, low, high)
    return (clipped - low) / (high - low)


def middle_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    if volume.ndim == 4:
        volume = volume[0]
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2
    return {
        "sagittal": volume[x_mid, :, :],
        "coronal": volume[:, y_mid, :],
        "axial": volume[:, :, z_mid],
    }


def meta_value_to_list(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def rows_to_csv_dicts(records: list[dict[str, object]]) -> list[dict[str, object]]:
    csv_rows: list[dict[str, object]] = []
    for record in records:
        csv_rows.append(
            {
                "patient_id": record["patient_id"],
                "modality": record["modality"],
                "image_path": record["image_path"],
                "source": record["source"],
                "original_shape": "x".join(map(str, record["original_shape"])),
                "original_spacing": "x".join(f"{value:.6f}" for value in record["original_spacing"]),
                "original_orientation": record["original_orientation"],
                "original_dtype": record["original_dtype"],
                "shape_after_load": "x".join(map(str, record["shape_after_load"])),
                "shape_after_channel": "x".join(map(str, record["shape_after_channel"])),
                "shape_after_resize": "x".join(map(str, record["shape_after_resize"])),
                "shape_after_normalize": "x".join(map(str, record["shape_after_normalize"])),
                "final_tensor_shape": "x".join(map(str, record["final_tensor_shape"])),
                "final_tensor_dtype": record["final_tensor_dtype"],
                "final_tensor_min": f"{record['final_tensor_min']:.6f}",
                "final_tensor_max": f"{record['final_tensor_max']:.6f}",
                "final_tensor_mean": f"{record['final_tensor_mean']:.6f}",
                "final_tensor_std": f"{record['final_tensor_std']:.6f}",
                "inferred_spacing_after_resize": "x".join(
                    f"{value:.6f}" for value in record["inferred_spacing_after_resize"]
                ),
                "notes": record["notes"],
            }
        )
    return csv_rows

