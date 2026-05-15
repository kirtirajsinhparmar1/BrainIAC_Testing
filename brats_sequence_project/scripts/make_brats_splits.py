"""Create patient-level BraTS sequence-classification CSV splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


TRAINING_SUBDIR = Path("BraTS2020_TrainingData") / "MICCAI_BraTS2020_TrainingData"
VALIDATION_SUBDIR = Path("BraTS2020_ValidationData") / "MICCAI_BraTS2020_ValidationData"

LABEL_MAPPING = {
    "T1": 0,
    "T2": 1,
    "FLAIR": 2,
    "T1CE": 3,
}

MODALITY_SUFFIXES = {
    "T1": "_t1.nii",
    "T2": "_t2.nii",
    "FLAIR": "_flair.nii",
    "T1CE": "_t1ce.nii",
}

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

    patient_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith(expected_prefix)
    )
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
            f"Patient {patient_id} is missing required modality files:\n" + "\n".join(missing_files)
        )


def expand_patient_rows(patient_dirs: list[Path], split_source: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for patient_dir in patient_dirs:
        validate_patient_modalities(patient_dir)
        patient_id = patient_dir.name
        for modality, suffix in MODALITY_SUFFIXES.items():
            image_path = patient_dir / f"{patient_id}{suffix}"
            if "seg" in image_path.name.lower() or "segm" in image_path.name.lower():
                raise ValueError(f"Segmentation file was incorrectly selected as input: {image_path}")
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


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def split_training_patients(
    patient_dirs: list[Path],
    train_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1, got {train_ratio}")

    shuffled = list(patient_dirs)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    if train_count == 0 or train_count == len(shuffled):
        raise ValueError("Train/val split would be empty. Adjust --train_ratio.")

    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def patient_ids(rows: list[dict[str, str | int]]) -> set[str]:
    return {str(row["patient_id"]) for row in rows}


def summarize_split(rows: list[dict[str, str | int]]) -> dict[str, object]:
    return {
        "patients": len(patient_ids(rows)),
        "image_rows": len(rows),
        "modality_counts": dict(sorted(Counter(str(row["modality"]) for row in rows).items())),
    }


def print_split_preview(name: str, rows: list[dict[str, str | int]]) -> None:
    print(f"\n{name}")
    print(json.dumps(summarize_split(rows), indent=2))
    print("first_rows:")
    for row in rows[:5]:
        print(row)


def assert_no_patient_overlap(split_rows: dict[str, list[dict[str, str | int]]]) -> None:
    split_patient_ids = {name: patient_ids(rows) for name, rows in split_rows.items()}
    overlaps = {}
    split_names = list(split_patient_ids)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = split_patient_ids[left_name] & split_patient_ids[right_name]
            if overlap:
                overlaps[f"{left_name}/{right_name}"] = sorted(overlap)

    if overlaps:
        raise AssertionError(f"Patients appear in multiple splits: {overlaps}")

    print("\npatient_overlap_check: passed; no patient appears in more than one split.")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_root = dataset_root / TRAINING_SUBDIR
    validation_root = dataset_root / VALIDATION_SUBDIR

    training_patients = discover_patient_dirs(training_root, "BraTS20_Training_")
    validation_patients = discover_patient_dirs(validation_root, "BraTS20_Validation_")
    train_patients, val_patients = split_training_patients(
        training_patients,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    split_rows = {
        "train": expand_patient_rows(train_patients, "training"),
        "val": expand_patient_rows(val_patients, "training"),
        "test": expand_patient_rows(validation_patients, "validation"),
    }

    assert_no_patient_overlap(split_rows)

    for split_name, rows in split_rows.items():
        write_csv(output_dir / f"{split_name}.csv", rows)

    label_mapping_path = output_dir / "label_mapping.json"
    label_mapping_path.write_text(json.dumps(LABEL_MAPPING, indent=2) + "\n")

    summary = {
        "dataset_root": str(dataset_root),
        "training_root": str(training_root),
        "validation_root": str(validation_root),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "splits": {name: summarize_split(rows) for name, rows in split_rows.items()},
        "label_mapping": LABEL_MAPPING,
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote: {output_dir / 'train.csv'}")
    print(f"wrote: {output_dir / 'val.csv'}")
    print(f"wrote: {output_dir / 'test.csv'}")
    print(f"wrote: {label_mapping_path}")
    print(f"wrote: {summary_path}")

    for split_name, rows in split_rows.items():
        print_split_preview(split_name, rows)


if __name__ == "__main__":
    main()
