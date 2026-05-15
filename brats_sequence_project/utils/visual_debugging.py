"""Shared helpers for BraTS sequence visual debugging scripts."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


CLASS_NAMES = ["T1", "T2", "FLAIR", "T1CE"]
CLASS_LABELS = [0, 1, 2, 3]
LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))
PROBABILITY_COLUMNS = ["probability_T1", "probability_T2", "probability_FLAIR", "probability_T1CE"]


def import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_name(label: object) -> str:
    try:
        return LABEL_TO_NAME[int(label)]
    except (TypeError, ValueError, KeyError):
        return str(label)


def safe_filename(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._")
    return text or "case"


def parse_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_correct(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def read_prediction_rows(predictions_csv: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(predictions_csv)
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = dict(row)
        parsed["true_label"] = parse_int(parsed.get("true_label"))
        parsed["pred_label"] = parse_int(parsed.get("pred_label"))
        for column in PROBABILITY_COLUMNS:
            parsed[column] = parse_float(parsed.get(column))
        parsed["confidence"] = parse_float(parsed.get("confidence"))
        if not np.isfinite(parsed["confidence"]):
            probabilities = [parsed[column] for column in PROBABILITY_COLUMNS]
            finite = [value for value in probabilities if np.isfinite(value)]
            parsed["confidence"] = float(max(finite)) if finite else math.nan
        correct = parse_correct(parsed.get("correct"))
        if correct is None and parsed["true_label"] is not None and parsed["pred_label"] is not None:
            correct = parsed["true_label"] == parsed["pred_label"]
        parsed["correct"] = bool(correct)
        parsed_rows.append(parsed)
    return parsed_rows


def prediction_fieldnames() -> list[str]:
    return [
        "patient_id",
        "image_path",
        "modality",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        *PROBABILITY_COLUMNS,
    ]


def confidence(row: dict[str, Any]) -> float:
    return parse_float(row.get("confidence"), default=-math.inf)


def tensor_to_volume(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image volume after channel squeeze, got shape {array.shape}")
    return np.asarray(array, dtype=np.float32)


def volume_stats(volume: np.ndarray) -> dict[str, float]:
    finite = np.asarray(volume[np.isfinite(volume)], dtype=np.float32)
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan, "nonzero_fraction": 0.0}
    nonzero_fraction = float(np.count_nonzero(np.abs(finite) > 1e-8) / finite.size)
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "nonzero_fraction": nonzero_fraction,
    }


def normalize_slice_for_display(slice_2d: np.ndarray) -> np.ndarray:
    array = np.asarray(slice_2d, dtype=np.float32)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=np.float32)

    active_mask = finite_mask & (np.abs(array) > 1e-8)
    if np.count_nonzero(active_mask) < 16:
        active_mask = finite_mask
    values = array[active_mask]
    low = float(np.percentile(values, 1))
    high = float(np.percentile(values, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)

    display = np.clip(array, low, high)
    display = (display - low) / (high - low)
    display[~finite_mask] = 0.0
    return display.astype(np.float32)


def middle_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    volume = tensor_to_volume(volume)
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2
    return {
        "axial": volume[:, :, z_mid],
        "sagittal": volume[x_mid, :, :],
        "coronal": volume[:, y_mid, :],
    }


def show_slice(axis: Any, slice_2d: np.ndarray, title: str | None = None) -> None:
    axis.imshow(normalize_slice_for_display(slice_2d).T, cmap="gray", origin="lower", interpolation="nearest")
    if title:
        axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def format_stats(stats: dict[str, float]) -> str:
    return (
        f"min {stats['min']:.2f} max {stats['max']:.2f} "
        f"mean {stats['mean']:.2f} std {stats['std']:.2f}"
    )


def format_probability_line(row: dict[str, Any]) -> str:
    values = [parse_float(row.get(column)) for column in PROBABILITY_COLUMNS]
    return " ".join(f"{name}={value:.3f}" for name, value in zip(CLASS_NAMES, values))


def prediction_title(row: dict[str, Any], compact: bool = False) -> str:
    true_label = row.get("true_label")
    pred_label = row.get("pred_label")
    correctness = "correct" if row.get("correct") else "wrong"
    head = (
        f"{row.get('patient_id', '')} {row.get('modality', '')} | "
        f"{label_name(true_label)} -> {label_name(pred_label)} | "
        f"{correctness} | conf {parse_float(row.get('confidence')):.3f}"
    )
    if compact:
        return head
    return f"{head}\n{format_probability_line(row)}"


def build_dataset_index(dataset: Any) -> dict[tuple[str, str], int]:
    index: dict[tuple[str, str], int] = {}
    for row_index, row in enumerate(getattr(dataset, "rows", [])):
        patient_id = row.get("patient_id", "")
        modality = row.get("modality", "")
        image_path = row.get("image_path", "")
        index[("patient_modality", f"{patient_id}\n{modality}")] = row_index
        index[("image_path", str(Path(image_path)))] = row_index
        index[("image_path_raw", image_path)] = row_index
    return index


def find_dataset_index(dataset_index: dict[tuple[str, str], int], row: dict[str, Any]) -> int | None:
    patient_id = str(row.get("patient_id", ""))
    modality = str(row.get("modality", ""))
    image_path = str(row.get("image_path", ""))
    for key in [
        ("patient_modality", f"{patient_id}\n{modality}"),
        ("image_path", str(Path(image_path))),
        ("image_path_raw", image_path),
    ]:
        if key in dataset_index:
            return dataset_index[key]
    return None


def load_original_volume(image_path: str | Path, dataset: Any | None = None) -> np.ndarray:
    import nibabel as nib

    source = str(image_path)
    if dataset is not None and hasattr(dataset, "_resolve_image_source"):
        source = dataset._resolve_image_source(source)
    image = nib.load(source)
    return np.asarray(image.dataobj, dtype=np.float32)


def load_preprocessed_volume_for_prediction(
    dataset: Any,
    dataset_index: dict[tuple[str, str], int],
    row: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    row_index = find_dataset_index(dataset_index, row)
    if row_index is not None:
        sample = dataset[row_index]
        return tensor_to_volume(sample["image"]), sample

    image_path = row.get("image_path")
    if not image_path:
        raise KeyError("Prediction row has no image_path and was not found in the dataset CSV.")
    image_source = dataset._resolve_image_source(str(image_path))
    sample = dataset.transform({"image": image_source})
    return tensor_to_volume(sample["image"]), {
        "patient_id": row.get("patient_id", ""),
        "modality": row.get("modality", ""),
        "image_path": image_path,
        "label": torch.tensor(row.get("true_label", -1)),
    }


def rows_for_complete_patient(
    csv_rows: list[dict[str, str]],
    patient_id: str | None,
    modalities: list[str],
) -> tuple[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in csv_rows:
        grouped.setdefault(row["patient_id"], {})[row["modality"]] = row

    required = set(modalities)
    if patient_id is not None:
        patient_rows = grouped.get(patient_id)
        if patient_rows is None or not required.issubset(patient_rows):
            raise ValueError(f"Patient {patient_id} does not contain modalities: {sorted(required)}")
        return patient_id, {modality: patient_rows[modality] for modality in modalities}

    for candidate_id in sorted(grouped):
        patient_rows = grouped[candidate_id]
        if required.issubset(patient_rows):
            return candidate_id, {modality: patient_rows[modality] for modality in modalities}
    raise ValueError(f"No patient with modalities {sorted(required)} was found.")


def save_empty_figure(output_path: Path, title: str) -> None:
    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.text(0.5, 0.5, title, ha="center", va="center", fontsize=14)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
