#!/usr/bin/env python3
"""Validate patient-level BraTS splits and emit a reproducible summary."""

from __future__ import annotations

import argparse
import json

from finetune_common import (
    REPO_ROOT,
    load_yaml_config,
    resolve_repo_path,
    summarize_splits,
    validate_all_splits,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "brats_sequence_project/finetune/config/full_finetune.yml"))
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--check-paths", action="store_true", help="Also require every image file to exist")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/results/split_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    data_config = config["data"]
    train_csv = args.train_csv or data_config["train_csv"]
    val_csv = args.val_csv or data_config["val_csv"]
    test_csv = args.test_csv or data_config["test_csv"]
    _, summary = validate_all_splits(
        train_csv,
        val_csv,
        test_csv,
        validate_paths=args.check_paths,
    )
    summary["csv_paths"] = {
        "train": str(resolve_repo_path(train_csv)),
        "validation": str(resolve_repo_path(val_csv)),
        "test": str(resolve_repo_path(test_csv)),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote split summary: {resolve_repo_path(args.output_json)}")


if __name__ == "__main__":
    main()
