"""Read-only audit for possible BrainIAC pretraining subject overlap.

The published BrainIAC tables identify datasets, but the released repository
does not provide a subject-level foundation-pretraining manifest.  This script
therefore compares the 494 patient IDs in the current BraTS2020 splits against
any text manifests or mapping files supplied by the user.  It never changes a
CSV, NIfTI file, checkpoint, or split assignment.

An empty result is deliberately reported as UNKNOWN: absence from the
manifests inspected by this script is not proof that no other manifest or
alias mapping exists.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from finetune_common import (
    CLASS_TO_INDEX,
    resolve_repo_path,
    validate_all_splits,
)


DEFAULT_TRAIN_CSV = "brats_sequence_project/outputs/train.csv"
DEFAULT_VAL_CSV = "brats_sequence_project/outputs/val.csv"
DEFAULT_TEST_CSV = "brats_sequence_project/outputs/test.csv"
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".list",
    ".lst",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BraTS2020 patient IDs with supplied BrainIAC pretraining "
            "manifests or historical mapping files without modifying data."
        )
    )
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Text manifest/mapping file or directory to inspect. Repeat this "
            "option for multiple sources."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path. The input splits and manifests are read-only.",
    )
    return parser.parse_args()


def _manifest_files(manifest_values: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for value in manifest_values:
        path = resolve_repo_path(value)
        if not path.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {path}")
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            raise ValueError(f"Manifest path is neither a file nor directory: {path}")
        directory_files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES
        )
        if not directory_files:
            raise ValueError(
                f"No supported text manifest files found under directory: {path}"
            )
        files.extend(directory_files)

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique_files.append(resolved)
            seen.add(resolved)
    return unique_files


def _patient_pattern(patient_ids: set[str]) -> tuple[re.Pattern[str], dict[str, str]]:
    if not patient_ids:
        raise ValueError("No patient IDs were found in the input split CSVs")
    ordered = sorted(patient_ids, key=lambda value: (-len(value), value))
    lookup = {patient_id.casefold(): patient_id for patient_id in ordered}
    alternatives = "|".join(re.escape(patient_id) for patient_id in ordered)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return pattern, lookup


def _scan_manifest(
    path: Path,
    pattern: re.Pattern[str],
    lookup: dict[str, str],
) -> dict[str, Any]:
    matched_lines: dict[str, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            for match in pattern.finditer(line):
                canonical_id = lookup[match.group(0).casefold()]
                matched_lines[canonical_id].add(line_number)

    return {
        "path": str(path),
        "matched_patient_ids": sorted(matched_lines),
        "match_line_numbers": {
            patient_id: sorted(line_numbers)
            for patient_id, line_numbers in sorted(matched_lines.items())
        },
    }


def _build_report(
    split_rows: dict[str, list[dict[str, Any]]],
    split_summary: dict[str, Any],
    manifest_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    all_patient_ids = {
        str(row["patient_id"])
        for rows in split_rows.values()
        for row in rows
    }
    matched_patient_ids = sorted(
        {
            patient_id
            for manifest_report in manifest_reports
            for patient_id in manifest_report["matched_patient_ids"]
        }
    )
    if not manifest_reports:
        status = "UNKNOWN_NO_MANIFESTS"
        interpretation = (
            "No subject-level manifests were supplied. Published dataset-level "
            "tables cannot establish subject-level non-overlap."
        )
    elif matched_patient_ids:
        status = "OVERLAP_FOUND_IN_INSPECTED_MANIFESTS"
        interpretation = (
            "These patient IDs occur in at least one inspected text source. "
            "Confirm that the source is an actual BrainIAC foundation-pretraining "
            "manifest before treating the match as contamination."
        )
    else:
        status = "UNKNOWN_NO_MATCH_IN_INSPECTED_MANIFESTS"
        interpretation = (
            "No exact patient-ID matches were found in the inspected sources, "
            "but this does not prove absence from unreleased manifests or alias "
            "mappings."
        )

    return {
        "audit": "BrainIAC foundation-pretraining subject overlap",
        "read_only": True,
        "status": status,
        "interpretation": interpretation,
        "class_mapping": dict(CLASS_TO_INDEX),
        "patient_count": len(all_patient_ids),
        "expected_patient_count": 494,
        "patient_count_matches_expected": len(all_patient_ids) == 494,
        "splits": {
            split_name: {
                "patients": split_summary["splits"][split_name]["patients"],
                "images": split_summary["splits"][split_name]["images"],
                "patient_ids": sorted(
                    {str(row["patient_id"]) for row in rows}
                ),
            }
            for split_name, rows in split_rows.items()
        },
        "split_integrity": {
            "zero_patient_overlap": split_summary["zero_patient_overlap"],
            "all_patients_have_four_modalities": split_summary[
                "all_patients_have_four_modalities"
            ],
            "no_segmentation_inputs": split_summary["no_segmentation_inputs"],
        },
        "manifest_sources_inspected": manifest_reports,
        "matched_patient_ids": matched_patient_ids,
        "limitations": [
            "Matching is exact text matching against the supplied sources.",
            "The report cannot infer that a source is a BrainIAC pretraining input without provenance.",
            "No match is not proof of no overlap when subject-level manifests or alias mappings are unavailable.",
        ],
    }


def main() -> None:
    args = _parse_args()
    split_rows, split_summary = validate_all_splits(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        validate_paths=False,
    )
    patient_ids = {
        str(row["patient_id"])
        for rows in split_rows.values()
        for row in rows
    }
    pattern, lookup = _patient_pattern(patient_ids)
    manifest_reports = [
        _scan_manifest(path, pattern, lookup)
        for path in _manifest_files(args.manifest)
    ]
    report = _build_report(split_rows, split_summary, manifest_reports)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        output_path = resolve_repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote_report={output_path}")


if __name__ == "__main__":
    main()
