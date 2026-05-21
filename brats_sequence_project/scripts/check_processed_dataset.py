"""Audit processed BraTS CSV/image outputs from preprocessing ablations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


LABEL_MAPPING = {
    "T1": 0,
    "T2": 1,
    "FLAIR": 2,
    "T1CE": 3,
}

REQUIRED_COLUMNS = ["patient_id", "image_path", "label", "modality", "split_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_spacing", type=str, default="1,1,1")
    parser.add_argument("--expected_modalities", type=str, default="T1,T2,FLAIR,T1CE")
    parser.add_argument("--sample_visuals", type=int, default=8)
    return parser.parse_args()


def require_simpleitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("SimpleITK is required for dataset auditing. Install it with: pip install SimpleITK") from exc
    return sitk


def require_visual_deps() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("Visualization requires numpy and matplotlib.") from exc
    return np, plt


def parse_triplet(value: str, name: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError(f"{name} must contain three positive comma-separated numbers, got {value!r}")
    return tuple(parts)  # type: ignore[return-value]


def parse_modalities(value: str) -> list[str]:
    modalities = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not modalities:
        raise ValueError("--expected_modalities did not contain any modalities.")
    unknown = sorted(set(modalities) - set(LABEL_MAPPING))
    if unknown:
        raise ValueError(f"Unsupported expected modalities: {unknown}")
    return modalities


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        return [dict(row) for row in reader]


def to_float_list(values: tuple[float, ...]) -> list[float]:
    return [float(item) for item in values]


def spacing_close(actual: list[float], expected: tuple[float, float, float], atol: float = 1e-3) -> bool:
    return len(actual) == 3 and all(abs(actual[index] - expected[index]) <= atol for index in range(3))


def image_audit_row(
    sitk: Any,
    csv_path: Path,
    row: dict[str, str],
    expected_spacing: tuple[float, float, float],
    expected_modalities: set[str],
) -> dict[str, Any]:
    image_path = Path(row["image_path"]).expanduser()
    if not image_path.is_absolute():
        image_path = (csv_path.parent / image_path).resolve()

    modality = str(row.get("modality", "")).upper()
    try:
        label = int(row.get("label", ""))
    except ValueError:
        label = -1

    expected_label = LABEL_MAPPING.get(modality)
    audit: dict[str, Any] = {
        "patient_id": row.get("patient_id", ""),
        "image_path": str(image_path),
        "label": label,
        "modality": modality,
        "split_source": row.get("split_source", ""),
        "path_exists": image_path.is_file(),
        "modality_expected": modality in expected_modalities,
        "label_matches_modality": expected_label is not None and label == expected_label,
        "shape": "",
        "spacing": "",
        "nonzero_count": "",
        "min": "",
        "max": "",
        "mean_nonzero": "",
        "spacing_matches_expected": False,
        "status": "pending",
        "error_message": "",
    }

    if not audit["path_exists"]:
        audit["status"] = "missing_path"
        audit["error_message"] = "image_path does not exist"
        return audit
    if "seg" in image_path.name.lower() or "segm" in image_path.name.lower():
        audit["status"] = "invalid_mask_path"
        audit["error_message"] = "image_path appears to be a segmentation mask"
        return audit

    try:
        image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        array = sitk.GetArrayFromImage(image)
        nonzero = array[array != 0]
        spacing = to_float_list(image.GetSpacing())
        audit.update(
            {
                "shape": json.dumps(list(image.GetSize())),
                "spacing": json.dumps(spacing),
                "nonzero_count": int(nonzero.size),
                "min": float(array.min()) if array.size else "",
                "max": float(array.max()) if array.size else "",
                "mean_nonzero": float(nonzero.mean()) if nonzero.size else "",
                "spacing_matches_expected": spacing_close(spacing, expected_spacing),
                "status": "ok",
            }
        )
    except Exception as exc:  # noqa: BLE001 - audit records file-level errors.
        audit["status"] = "read_failed"
        audit["error_message"] = str(exc)
    return audit


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "patient_id",
        "image_path",
        "label",
        "modality",
        "split_source",
        "path_exists",
        "modality_expected",
        "label_matches_modality",
        "shape",
        "spacing",
        "nonzero_count",
        "min",
        "max",
        "mean_nonzero",
        "spacing_matches_expected",
        "status",
        "error_message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(audit_rows: list[dict[str, Any]], expected_spacing: tuple[float, float, float]) -> dict[str, Any]:
    nonzero_counts = [
        int(row["nonzero_count"])
        for row in audit_rows
        if row.get("status") == "ok" and row.get("nonzero_count") not in {"", None}
    ]
    return {
        "total_rows": len(audit_rows),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in audit_rows).items())),
        "path_missing_count": sum(1 for row in audit_rows if not row["path_exists"]),
        "segmentation_like_path_count": sum(1 for row in audit_rows if row["status"] == "invalid_mask_path"),
        "label_mismatch_count": sum(1 for row in audit_rows if not row["label_matches_modality"]),
        "unexpected_modality_count": sum(1 for row in audit_rows if not row["modality_expected"]),
        "spacing_mismatch_count": sum(1 for row in audit_rows if row["status"] == "ok" and not row["spacing_matches_expected"]),
        "unique_shapes": sorted({str(row["shape"]) for row in audit_rows if row.get("shape")}),
        "unique_spacings": sorted({str(row["spacing"]) for row in audit_rows if row.get("spacing")}),
        "expected_spacing": list(expected_spacing),
        "modality_counts": dict(sorted(Counter(str(row["modality"]) for row in audit_rows).items())),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in audit_rows).items())),
        "patient_count": len({str(row["patient_id"]) for row in audit_rows}),
        "nonzero_count_summary": {
            "min": min(nonzero_counts) if nonzero_counts else None,
            "median": median(nonzero_counts) if nonzero_counts else None,
            "max": max(nonzero_counts) if nonzero_counts else None,
        },
    }


def middle_axial_slice(array: Any) -> Any:
    return array[array.shape[0] // 2, :, :]


def robust_window(array: Any, np: Any) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def create_sample_visual_grid(sitk: Any, audit_rows: list[dict[str, Any]], output_dir: Path, sample_visuals: int) -> None:
    if sample_visuals <= 0:
        return
    ok_rows = [row for row in audit_rows if row["status"] == "ok"][:sample_visuals]
    if not ok_rows:
        return

    np, plt = require_visual_deps()
    cols = min(4, len(ok_rows))
    rows = (len(ok_rows) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for index, row in enumerate(ok_rows):
        ax = axes.flat[index]
        image = sitk.ReadImage(row["image_path"], sitk.sitkFloat32)
        array = sitk.GetArrayFromImage(image)
        image_slice = middle_axial_slice(array)
        vmin, vmax = robust_window(array, np)
        ax.imshow(np.rot90(image_slice), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{row['patient_id']} {row['modality']}\nshape {row['shape']}", fontsize=8)
        ax.axis("off")

    fig.savefig(output_dir / "sample_visual_grid.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_spacing = parse_triplet(args.expected_spacing, "--expected_spacing")
    expected_modalities = set(parse_modalities(args.expected_modalities))
    rows = read_rows(csv_path)
    sitk = require_simpleitk()

    audit_rows = [
        image_audit_row(sitk, csv_path, row, expected_spacing, expected_modalities)
        for row in rows
    ]
    summary = summarize(audit_rows, expected_spacing)

    audit_csv = output_dir / "processed_dataset_audit.csv"
    audit_json = output_dir / "processed_dataset_audit.json"
    write_csv(audit_csv, audit_rows)
    audit_json.write_text(json.dumps(summary, indent=2) + "\n")
    create_sample_visual_grid(sitk, audit_rows, output_dir, args.sample_visuals)

    print(f"wrote audit CSV: {audit_csv}")
    print(f"wrote audit JSON: {audit_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
