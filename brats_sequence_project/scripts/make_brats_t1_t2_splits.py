"""Create patient-level BraTS binary T1-vs-T2 CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


TRAINING_SUBDIR = Path("BraTS2020_TrainingData") / "MICCAI_BraTS2020_TrainingData"
VALIDATION_SUBDIR = Path("BraTS2020_ValidationData") / "MICCAI_BraTS2020_ValidationData"

LABEL_MAPPING = {"T1": 0, "T2": 1}
LABEL_NAME_MAPPING = {"0": "T1", "1": "T2"}
MODALITY_SUFFIXES = {"T1": "_t1.nii", "T2": "_t2.nii"}
CSV_COLUMNS = ["patient_id", "image_path", "label", "modality", "split_source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    return parser.parse_args()


def discover_patient_dirs(root: Path, expected_prefix: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Expected dataset directory does not exist: {root}")

    patient_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(expected_prefix))
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories with prefix {expected_prefix!r} found under {root}")
    return patient_dirs


def validate_patient_modalities(patient_dir: Path) -> None:
    patient_id = patient_dir.name
    missing_files = []
    for suffix in MODALITY_SUFFIXES.values():
        image_path = patient_dir / f"{patient_id}{suffix}"
        if not image_path.is_file():
            missing_files.append(str(image_path))

    if missing_files:
        raise FileNotFoundError(
            f"Patient {patient_id} is missing required T1/T2 files:\n" + "\n".join(missing_files)
        )


def expand_patient_rows(patient_dirs: list[Path], split_source: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for patient_dir in patient_dirs:
        validate_patient_modalities(patient_dir)
        patient_id = patient_dir.name
        for modality, suffix in MODALITY_SUFFIXES.items():
            image_path = patient_dir / f"{patient_id}{suffix}"
            lowered_name = image_path.name.lower()
            if any(excluded in lowered_name for excluded in ("flair", "t1ce", "seg", "segm")):
                raise ValueError(f"Excluded modality/mask was incorrectly selected: {image_path}")
            rows.append(
                {
                    "patient_id": patient_id,
                    "image_path": str(image_path.resolve()),
                    "label": LABEL_MAPPING[modality],
                    "modality": modality,
                    "split_source": split_source,
                }
            )
    return rows


def split_training_patients(patient_dirs: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1, got {train_ratio}")

    shuffled = list(patient_dirs)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    if train_count == 0 or train_count == len(shuffled):
        raise ValueError("Train/val split would be empty. Adjust --train_ratio.")
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def patient_ids(rows: list[dict[str, str | int]]) -> set[str]:
    return {str(row["patient_id"]) for row in rows}


def class_counts(rows: list[dict[str, str | int]]) -> dict[str, int]:
    counts = Counter(str(row["modality"]) for row in rows)
    return {modality: int(counts.get(modality, 0)) for modality in LABEL_MAPPING}


def label_counts(rows: list[dict[str, str | int]]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {LABEL_NAME_MAPPING[str(label)]: int(counts.get(label, 0)) for label in sorted(int(k) for k in LABEL_NAME_MAPPING)}


def summarize_split(rows: list[dict[str, str | int]]) -> dict[str, Any]:
    return {
        "patients": len(patient_ids(rows)),
        "image_rows": len(rows),
        "class_counts": class_counts(rows),
        "label_counts": label_counts(rows),
        "balanced_t1_t2": class_counts(rows).get("T1") == class_counts(rows).get("T2"),
    }


def assert_no_patient_overlap(split_rows: dict[str, list[dict[str, str | int]]]) -> None:
    split_patient_ids = {name: patient_ids(rows) for name, rows in split_rows.items()}
    overlaps: dict[str, list[str]] = {}
    split_names = list(split_patient_ids)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = split_patient_ids[left_name] & split_patient_ids[right_name]
            if overlap:
                overlaps[f"{left_name}/{right_name}"] = sorted(overlap)

    if overlaps:
        raise AssertionError(f"Patients appear in multiple splits: {overlaps}")
    print("\npatient_overlap_check: passed; no patient appears in more than one split.")


def assert_balanced(split_rows: dict[str, list[dict[str, str | int]]]) -> None:
    imbalanced = {name: class_counts(rows) for name, rows in split_rows.items() if class_counts(rows)["T1"] != class_counts(rows)["T2"]}
    if imbalanced:
        raise AssertionError(f"T1/T2 class counts are not balanced: {imbalanced}")
    print("class_balance_check: passed; each split has equal T1 and T2 rows.")


def print_split_preview(name: str, rows: list[dict[str, str | int]]) -> None:
    summary = summarize_split(rows)
    print(f"\n{name}")
    print(f"patient_count: {summary['patients']}")
    print(f"row_count: {summary['image_rows']}")
    print(f"class_counts: {summary['class_counts']}")
    print("first_rows:")
    for row in rows[:5]:
        print(row)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_root = dataset_root / TRAINING_SUBDIR
    validation_root = dataset_root / VALIDATION_SUBDIR

    training_patients = discover_patient_dirs(training_root, "BraTS20_Training_")
    validation_patients = discover_patient_dirs(validation_root, "BraTS20_Validation_")
    train_patients, val_patients = split_training_patients(training_patients, args.train_ratio, args.seed)

    split_rows = {
        "train": expand_patient_rows(train_patients, "training"),
        "val": expand_patient_rows(val_patients, "training"),
        "test": expand_patient_rows(validation_patients, "validation"),
    }
    assert_no_patient_overlap(split_rows)
    assert_balanced(split_rows)

    output_paths = {
        "train": output_dir / "train_t1_t2.csv",
        "val": output_dir / "val_t1_t2.csv",
        "test": output_dir / "test_t1_t2.csv",
    }
    for split_name, rows in split_rows.items():
        write_csv(output_paths[split_name], rows)

    label_mapping_path = output_dir / "label_mapping_t1_t2.json"
    label_mapping_path.write_text(json.dumps(LABEL_NAME_MAPPING, indent=2) + "\n")

    summary = {
        "dataset_root": str(dataset_root),
        "training_root": str(training_root),
        "validation_root": str(validation_root),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "label_mapping": LABEL_NAME_MAPPING,
        "modality_to_label": LABEL_MAPPING,
        "excluded_modalities": ["FLAIR", "T1CE"],
        "excluded_masks": ["seg", "segm"],
        "splits": {name: summarize_split(rows) for name, rows in split_rows.items()},
        "patient_overlap_check": "passed",
        "class_balance_check": "passed",
    }
    summary_path = output_dir / "split_summary_t1_t2.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    for path in [*output_paths.values(), label_mapping_path, summary_path]:
        print(f"wrote: {path}")

    for split_name, rows in split_rows.items():
        print_split_preview(split_name, rows)


if __name__ == "__main__":
    main()
