"""Run a tiny Phase 3 smoke test without full training."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_frozen_brainiac_classifier.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_frozen_brainiac_classifier.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainiac_src", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_rows", type=int, default=8)
    parser.add_argument("--max_val_rows", type=int, default=8)
    return parser.parse_args()


def write_subset_csv(source_csv: Path, output_csv: Path, max_rows: int) -> None:
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")

    with source_csv.open("r", newline="") as source_file:
        reader = csv.DictReader(source_file)
        rows = [row for _, row in zip(range(max_rows), reader)]
        if not rows:
            raise ValueError(f"No rows found in {source_csv}")

    with output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str]) -> None:
    print("\nrunning:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_train_csv = output_dir / "smoke_train.csv"
    smoke_val_csv = output_dir / "smoke_val.csv"
    train_output_dir = output_dir / "train"
    eval_output_dir = output_dir / "eval"

    write_subset_csv(args.train_csv.expanduser().resolve(), smoke_train_csv, args.max_train_rows)
    write_subset_csv(args.val_csv.expanduser().resolve(), smoke_val_csv, args.max_val_rows)

    run_command(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--brainiac_src",
            str(args.brainiac_src.expanduser().resolve()),
            "--checkpoint_path",
            str(args.checkpoint_path.expanduser().resolve()),
            "--train_csv",
            str(smoke_train_csv),
            "--val_csv",
            str(smoke_val_csv),
            "--output_dir",
            str(train_output_dir),
            "--epochs",
            "1",
            "--batch_size",
            str(args.batch_size),
            "--lr",
            "0.001",
            "--weight_decay",
            "0.0001",
            "--num_workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--patience",
            "1",
        ]
    )

    best_classifier_path = train_output_dir / "best_classifier_head.pt"
    if not best_classifier_path.is_file():
        raise FileNotFoundError(f"Smoke test did not create {best_classifier_path}")

    run_command(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            "--brainiac_src",
            str(args.brainiac_src.expanduser().resolve()),
            "--checkpoint_path",
            str(args.checkpoint_path.expanduser().resolve()),
            "--classifier_path",
            str(best_classifier_path),
            "--csv_path",
            str(smoke_val_csv),
            "--output_dir",
            str(eval_output_dir),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ]
    )

    required_outputs = [
        best_classifier_path,
        train_output_dir / "train_history.csv",
        train_output_dir / "best_metrics.json",
        eval_output_dir / "test_metrics.json",
        eval_output_dir / "predictions.csv",
    ]
    missing_outputs = [path for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise FileNotFoundError(f"Smoke test missing expected outputs: {missing_outputs}")

    print("\nphase3_smoke_test: passed")
    print(f"best_classifier_head: {best_classifier_path}")
    print(f"eval_metrics: {eval_output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
