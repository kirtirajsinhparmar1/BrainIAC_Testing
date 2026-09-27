#!/usr/bin/env python3
"""Create and validate one deterministic patient-level training subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from finetune_common import (
    REPO_ROOT,
    load_yaml_config,
    read_csv_rows,
    resolve_repo_path,
    select_patient_ids,
    validate_all_splits,
    validate_fixed_evaluation_splits,
    validate_finetune_config,
    validate_patient_subset,
    write_csv_rows,
    write_json,
)


def seed_plan_from_config(config: Mapping[str, Any]) -> dict[int, list[int]]:
    """Normalize the editable YAML seed plan and reject malformed entries."""

    data_efficiency = config.get("data_efficiency")
    if not isinstance(data_efficiency, Mapping):
        raise ValueError("Configuration is missing data_efficiency")
    raw_counts = data_efficiency.get("patient_counts")
    raw_plan = data_efficiency.get("seed_plan")
    if not isinstance(raw_counts, list) or not isinstance(raw_plan, Mapping):
        raise ValueError("data_efficiency must define patient_counts and seed_plan")

    patient_counts = [int(value) for value in raw_counts]
    normalized: dict[int, list[int]] = {}
    for raw_count, raw_seeds in raw_plan.items():
        count = int(raw_count)
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ValueError(f"seed_plan[{count}] must be a non-empty list")
        seeds = [int(seed) for seed in raw_seeds]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"seed_plan[{count}] contains duplicate seeds")
        normalized[count] = seeds
    if set(normalized) != set(patient_counts):
        raise ValueError("patient_counts and seed_plan keys must describe the same grid")
    if len(set(patient_counts)) != len(patient_counts):
        raise ValueError("patient_counts contains duplicates")
    return {count: normalized[count] for count in patient_counts}


def validate_experiment_request(
    config: Mapping[str, Any], patient_count: int, seed: int
) -> dict[int, list[int]]:
    plan = seed_plan_from_config(config)
    if patient_count not in plan:
        raise ValueError(
            f"patient_count={patient_count} is not configured; choose one of {list(plan)}"
        )
    if seed not in plan[patient_count]:
        raise ValueError(
            f"seed={seed} is not configured for {patient_count} patients; "
            f"choose one of {plan[patient_count]}"
        )
    return plan


def default_manifest_path(config: Mapping[str, Any], patient_count: int, seed: int) -> Path:
    root = resolve_repo_path(config["output"]["root_dir"])
    return root / f"patients_{patient_count:03d}" / f"seed_{seed}" / "subset_manifest.csv"


def prepare_subset_manifest(
    *,
    train_csv: str | Path,
    validation_csv: str | Path,
    test_csv: str | Path,
    patient_count: int,
    seed: int,
    output_path: str | Path,
    check_paths: bool = False,
) -> dict[str, Any]:
    """Prepare one manifest, reusing it only when it matches the deterministic selection."""

    split_rows, split_summary = validate_all_splits(
        train_csv,
        validation_csv,
        test_csv,
        validate_paths=check_paths,
    )
    train_rows = split_rows["train"]
    validation_rows = split_rows["validation"]
    test_rows = split_rows["test"]
    validate_fixed_evaluation_splits(validation_rows, test_rows)

    selected_order, complete_order = select_patient_ids(train_rows, patient_count, seed)
    selected_set = set(selected_order)
    subset_rows = [row for row in train_rows if str(row["patient_id"]) in selected_set]
    subset_summary = validate_patient_subset(
        subset_rows,
        patient_count,
        train_rows,
        validation_rows,
        test_rows,
    )

    manifest = resolve_repo_path(output_path)
    manifest_created = False
    if manifest.exists():
        existing_rows = read_csv_rows(manifest, validate_paths=False)
        validate_patient_subset(
            existing_rows,
            patient_count,
            train_rows,
            validation_rows,
            test_rows,
        )
        if [dict(row) for row in existing_rows] != [dict(row) for row in subset_rows]:
            raise ValueError(
                f"Existing subset manifest does not match seed {seed} and "
                f"patient_count {patient_count}: {manifest}"
            )
    else:
        write_csv_rows(manifest, subset_rows)
        manifest_created = True

    return {
        "manifest_path": str(manifest),
        "manifest_created": manifest_created,
        "source_train_csv": str(resolve_repo_path(train_csv)),
        "validation_csv": str(resolve_repo_path(validation_csv)),
        "test_csv": str(resolve_repo_path(test_csv)),
        "seed": seed,
        "patient_count": patient_count,
        "scan_count": patient_count * 4,
        "scans_per_class": patient_count,
        "selected_patient_ids": sorted(selected_set),
        "selected_patient_order": selected_order,
        "deterministic_patient_order": complete_order,
        "split_summary": split_summary,
        "subset_summary": subset_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/config/data_efficiency.yml"),
    )
    parser.add_argument("--patients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", help="Subset manifest path; defaults to the systematic run directory")
    parser.add_argument("--validation-json", help="Optional path for the subset validation report")
    parser.add_argument("--check-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_finetune_config(config)
    validate_experiment_request(config, args.patients, args.seed)
    data_config = config["data"]
    output_path = args.output or default_manifest_path(config, args.patients, args.seed)
    report = prepare_subset_manifest(
        train_csv=data_config["train_csv"],
        validation_csv=data_config["val_csv"],
        test_csv=data_config["test_csv"],
        patient_count=args.patients,
        seed=args.seed,
        output_path=output_path,
        check_paths=args.check_paths,
    )
    validation_json = args.validation_json or str(
        Path(report["manifest_path"]).with_name("subset_validation.json")
    )
    write_json(validation_json, report)
    print(json.dumps(report["subset_summary"], indent=2, sort_keys=True))
    print(f"selected_patient_order={report['selected_patient_order']}")
    print(f"manifest={report['manifest_path']}")
    print(f"validation_report={resolve_repo_path(validation_json)}")


if __name__ == "__main__":
    main()
