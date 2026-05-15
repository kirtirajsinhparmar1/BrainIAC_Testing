"""Check BraTS sequence CSV label mapping, class balance, and patient-level split integrity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing_audit_utils import (  # noqa: E402
    MODALITY_TO_LABEL,
    MODALITY_TO_SUFFIX,
    read_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def summarize_split(name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    label_counts = Counter(int(row["label"]) for row in rows)
    modality_counts = Counter(row["modality"] for row in rows)
    patients = {row["patient_id"] for row in rows}
    bad_suffix_rows = []
    seg_rows = []

    for row in rows:
        path_lower = row["image_path"].lower()
        modality = row["modality"]
        expected_suffix = MODALITY_TO_SUFFIX[modality]
        expected_label = MODALITY_TO_LABEL[modality]
        if not path_lower.endswith(expected_suffix):
            bad_suffix_rows.append(row["image_path"])
        if int(row["label"]) != expected_label:
            raise ValueError(
                f"{name}: label mismatch for {row['image_path']}. "
                f"Expected {expected_label} for modality {modality}, got {row['label']}."
            )
        if "seg" in path_lower or "segm" in path_lower:
            seg_rows.append(row["image_path"])

    patient_modalities = Counter(row["patient_id"] for row in rows)
    incomplete_patients = [patient_id for patient_id, count in patient_modalities.items() if count != 4]

    return {
        "split": name,
        "rows": len(rows),
        "patients": len(patients),
        "label_counts": dict(sorted(label_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "bad_suffix_rows": bad_suffix_rows,
        "seg_rows": seg_rows,
        "incomplete_patients": incomplete_patients,
        "patient_ids": sorted(patients),
    }


def main() -> None:
    args = parse_args()
    summaries = {
        "train": summarize_split("train", read_csv_rows(args.train_csv)),
        "val": summarize_split("val", read_csv_rows(args.val_csv)),
        "test": summarize_split("test", read_csv_rows(args.test_csv)),
    }

    train_patients = set(summaries["train"]["patient_ids"])
    val_patients = set(summaries["val"]["patient_ids"])
    test_patients = set(summaries["test"]["patient_ids"])
    overlap = {
        "train_val": sorted(train_patients & val_patients),
        "train_test": sorted(train_patients & test_patients),
        "val_test": sorted(val_patients & test_patients),
    }

    result = {
        "summaries": {
            key: {inner_key: inner_value for inner_key, inner_value in value.items() if inner_key != "patient_ids"}
            for key, value in summaries.items()
        },
        "patient_overlap": {key: len(value) for key, value in overlap.items()},
        "patient_overlap_examples": {key: value[:5] for key, value in overlap.items()},
    }

    for split_name, summary in result["summaries"].items():
        print(
            f"{split_name}: rows={summary['rows']} patients={summary['patients']} "
            f"labels={summary['label_counts']} modalities={summary['modality_counts']}"
        )
    print(f"patient_overlap: {result['patient_overlap']}")

    if any(result["patient_overlap"].values()):
        raise SystemExit("Patient overlap detected across splits.")

    for split_name, summary in result["summaries"].items():
        if summary["bad_suffix_rows"]:
            raise SystemExit(f"{split_name}: found filename/modality mismatches.")
        if summary["seg_rows"]:
            raise SystemExit(f"{split_name}: segmentation rows were included unexpectedly.")
        if summary["incomplete_patients"]:
            raise SystemExit(f"{split_name}: found patients without exactly four modality rows.")

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"output_json: {args.output_json}")

    print("label_split_check: passed")


if __name__ == "__main__":
    main()
