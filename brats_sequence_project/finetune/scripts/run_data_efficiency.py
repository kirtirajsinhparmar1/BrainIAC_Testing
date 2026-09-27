#!/usr/bin/env python3
"""Run or prepare one controlled BrainIAC low-data fine-tuning experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from aggregate_data_efficiency import aggregate_results, safe_plot_run_artifacts

from finetune_common import (
    EXPECTED_TEST_SCANS,
    REPO_ROOT,
    is_valid_metrics_payload,
    load_yaml_config,
    resolve_repo_path,
    validate_general_checkpoint_path,
    validate_finetune_config,
    write_json,
)
from make_subset_manifest import (
    prepare_subset_manifest,
    seed_plan_from_config,
    validate_experiment_request,
)


TRAINER = Path(__file__).resolve().with_name("finetune_model.py")
EVALUATOR = Path(__file__).resolve().with_name("evaluate_checkpoint.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/config/data_efficiency.yml"),
    )
    parser.add_argument("--patients", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", help="Must resolve to the configured general BrainIAC.ckpt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--dry-run-matrix", action="store_true")
    mode.add_argument("--run-matrix", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly retrain/re-evaluate an existing run instead of resuming it",
    )
    return parser.parse_args()


def run_directory(config: Mapping[str, Any], patient_count: int, seed: int) -> Path:
    root = resolve_repo_path(config["output"]["root_dir"])
    return root / f"patients_{patient_count:03d}" / f"seed_{seed}"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def valid_metrics_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return is_valid_metrics_payload(_read_json(path), expected_test_scans=EXPECTED_TEST_SCANS)
    except (OSError, json.JSONDecodeError):
        return False


def _git_commit_hash() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    return commit or None


def _load_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "training_history.json"
    if not history_path.is_file():
        return []
    value = _read_json(history_path)
    if not isinstance(value, list):
        raise ValueError(f"Training history is not a list: {history_path}")
    return [record for record in value if isinstance(record, dict)]


def _best_history_record(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for record in history:
        metrics = record.get("validation_metrics")
        if isinstance(metrics, Mapping) and "balanced_accuracy" in metrics:
            candidates.append((float(metrics["balanced_accuracy"]), record))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def write_history_csv(run_dir: Path) -> None:
    history = _load_history(run_dir)
    if not history:
        return
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "validation_loss",
        "train_balanced_accuracy",
        "train_accuracy",
        "validation_balanced_accuracy",
        "validation_accuracy",
        "seconds",
    ]
    output_path = run_dir / "training_history.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            train_metrics = record.get("train_metrics", {})
            validation_metrics = record.get("validation_metrics", {})
            writer.writerow(
                {
                    "epoch": record.get("epoch"),
                    "learning_rate": record.get("learning_rate"),
                    "train_loss": record.get("train_loss"),
                    "validation_loss": record.get("validation_loss"),
                    "train_balanced_accuracy": train_metrics.get("balanced_accuracy"),
                    "train_accuracy": train_metrics.get("accuracy"),
                    "validation_balanced_accuracy": validation_metrics.get("balanced_accuracy"),
                    "validation_accuracy": validation_metrics.get("accuracy"),
                    "seconds": record.get("seconds"),
                }
            )


def _planned_metadata(
    config: Mapping[str, Any],
    config_path: str | Path,
    checkpoint_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    return {
        "status": "planned",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "patient_count": report["patient_count"],
        "scan_count": report["scan_count"],
        "scans_per_class": report["scans_per_class"],
        "seed": report["seed"],
        "selected_patient_ids": report["selected_patient_ids"],
        "selected_patient_order": report["selected_patient_order"],
        "source_train_csv": report["source_train_csv"],
        "validation_csv": report["validation_csv"],
        "test_csv": report["test_csv"],
        "validation_scan_count": report["subset_summary"]["validation_scan_count"],
        "test_scan_count": report["subset_summary"]["test_scan_count"],
        "general_brainiac_checkpoint": str(checkpoint_path),
        "model_initialized_from": "checkpoints/BrainIAC.ckpt",
        "training_config": str(resolve_repo_path(config_path)),
        "config_path": str(resolve_repo_path(config_path)),
        "epochs": int(training["epochs"]),
        "batch_size": int(config["data"]["batch_size"]),
        "validation_batch_size": int(config["data"]["validation_batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "validation_num_workers": int(config["data"]["validation_num_workers"]),
        "pin_memory": bool(config["data"]["pin_memory"]),
        "preprocessing": config["data"]["preprocessing"],
        "optimizer": training["optimizer"],
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "scheduler": training["scheduler"],
        "scheduler_mode": training["scheduler_mode"],
        "scheduler_factor": float(training["scheduler_factor"]),
        "scheduler_patience": int(training["scheduler_patience"]),
        "precision": training["precision"],
        "checkpoint_metric": training["checkpoint_metric"],
        "checkpoint_mode": training["checkpoint_mode"],
        "full_fine_tuning": True,
        "backbone_frozen": False,
        "test_used_for_checkpoint_selection": False,
        "checkpoint_selected_without_test": True,
        "best_validation_balanced_accuracy": None,
        "best_epoch": None,
    }


def _training_metadata(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    history = _load_history(run_dir)
    best_record = _best_history_record(history)
    if best_record is not None:
        validation_metrics = best_record.get("validation_metrics", {})
        metadata["best_epoch"] = int(best_record["epoch"])
        metadata["best_validation_balanced_accuracy"] = float(
            validation_metrics["balanced_accuracy"]
        )
    metadata["training_history_json"] = str(run_dir / "training_history.json")
    metadata["training_history_csv"] = str(run_dir / "training_history.csv")
    return metadata


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def matrix_candidates(plan: Mapping[int, list[int]]) -> list[tuple[int, int]]:
    """Return matrix candidates in configured patient-count/seed order."""

    return [
        (patient_count, seed)
        for patient_count, seeds in plan.items()
        for seed in seeds
    ]


def candidate_status(
    config: Mapping[str, Any], patient_count: int, seed: int
) -> tuple[str, float | None]:
    """Classify a candidate using only resumable artifacts on disk."""

    run_dir = run_directory(config, patient_count, seed)
    metrics_path = run_dir / "test_evaluation" / "metrics.json"
    if valid_metrics_file(metrics_path):
        try:
            metrics = _read_json(metrics_path)
            return "COMPLETE", float(metrics["balanced_accuracy"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            pass
    if (run_dir / "best_model.ckpt").is_file():
        return "CHECKPOINT", None
    if (run_dir / "subset_manifest.csv").is_file():
        return "SUBSET", None
    return "PENDING", None


def _build_commands(
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    validation_csv: Path,
    test_csv: Path,
    seed: int,
    run_dir: Path,
) -> tuple[list[str], list[str]]:
    training_command = [
        sys.executable,
        str(TRAINER),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--train-csv",
        str(manifest_path),
        "--val-csv",
        str(validation_csv),
        "--fraction",
        "1.0",
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
    ]
    evaluation_command = [
        sys.executable,
        str(EVALUATOR),
        "--config",
        str(config_path),
        "--checkpoint",
        str(run_dir / "best_model.ckpt"),
        "--test-csv",
        str(test_csv),
        "--output-dir",
        str(run_dir / "test_evaluation"),
        "--device",
        "cuda",
    ]
    return training_command, evaluation_command


def run_one(
    config: Mapping[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    patient_count: int,
    seed: int,
    *,
    prepare_only: bool = False,
    dry_run: bool = False,
    train_only: bool = False,
    evaluate_only: bool = False,
    force: bool = False,
    show_details: bool = True,
) -> str:
    data_config = config["data"]
    run_dir = run_directory(config, patient_count, seed)
    metrics_path = run_dir / "test_evaluation" / "metrics.json"

    # Check completed scientific outputs before touching the subset manifest.
    # This keeps a valid result immutable when the matrix is resumed.
    if valid_metrics_file(metrics_path) and not force and not dry_run and not prepare_only:
        print(
            f"patients={patient_count} scans={patient_count * 4} seed={seed} "
            f"run_dir={run_dir}"
        )
        if show_details:
            print("validation=74 patients / 296 scans (fixed)")
            print("test=125 patients / 500 scans (fixed, outcome-only)")
        print(f"resume=complete; skipping existing valid metrics: {metrics_path}")
        safe_plot_run_artifacts(run_dir)
        return "skipped"

    manifest_path = run_dir / "subset_manifest.csv"
    report = prepare_subset_manifest(
        train_csv=data_config["train_csv"],
        validation_csv=data_config["val_csv"],
        test_csv=data_config["test_csv"],
        patient_count=patient_count,
        seed=seed,
        output_path=manifest_path,
        check_paths=False,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "subset_validation.json", report)

    validation_csv = resolve_repo_path(data_config["val_csv"])
    test_csv = resolve_repo_path(data_config["test_csv"])
    training_command, evaluation_command = _build_commands(
        config_path,
        checkpoint_path,
        resolve_repo_path(manifest_path),
        validation_csv,
        test_csv,
        seed,
        run_dir,
    )

    print(
        f"patients={patient_count} scans={patient_count * 4} seed={seed} "
        f"run_dir={run_dir}"
    )
    if show_details:
        print(f"selected_patient_order={report['selected_patient_order']}")
        print("validation=74 patients / 296 scans (fixed)")
        print("test=125 patients / 500 scans (fixed, outcome-only)")

    if dry_run:
        print(f"training_command={_command_text(training_command)}")
        print(f"evaluation_command={_command_text(evaluation_command)}")
        print("dry_run=True; no model was instantiated and no training/evaluation was started")
        return "dry-run"
    if prepare_only:
        print("prepare_only=True; no training/evaluation was started")
        return "prepared"

    if evaluate_only and not (run_dir / "best_model.ckpt").is_file():
        raise FileNotFoundError(
            f"--evaluate-only requires an existing checkpoint: {run_dir / 'best_model.ckpt'}"
        )
    validate_general_checkpoint_path(checkpoint_path, require_exists=True)

    metadata_path = run_dir / "run_metadata.json"
    metadata = _planned_metadata(config, config_path, checkpoint_path, report)
    if metadata_path.is_file():
        try:
            old_metadata = _read_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            old_metadata = {}
        if isinstance(old_metadata, Mapping):
            metadata.update(old_metadata)
    write_json(metadata_path, metadata)

    best_checkpoint = run_dir / "best_model.ckpt"
    should_train = not evaluate_only and (force or not best_checkpoint.is_file())
    if should_train:
        print(f"starting_training_from={checkpoint_path}")
        subprocess.run(training_command, cwd=REPO_ROOT, check=True)
        if not best_checkpoint.is_file():
            raise RuntimeError(f"Training completed without writing {best_checkpoint}")
    else:
        print(f"resume=checkpoint_exists; skipping training: {best_checkpoint}")

    write_history_csv(run_dir)
    metadata = _training_metadata(run_dir, metadata)
    if train_only:
        metadata["status"] = "trained_not_evaluated"
        write_json(metadata_path, metadata)
        print(f"train_only=True; checkpoint={best_checkpoint}")
        return "trained"

    print(f"starting_test_evaluation_on={test_csv}")
    subprocess.run(evaluation_command, cwd=REPO_ROOT, check=True)
    if not valid_metrics_file(metrics_path):
        raise RuntimeError(
            f"Evaluator did not produce valid full-test metrics at {metrics_path}"
        )
    metrics = _read_json(metrics_path)
    metadata["status"] = "completed"
    metadata["best_model_checkpoint"] = str(best_checkpoint)
    metadata["test_metrics_path"] = str(metrics_path)
    metadata["test_balanced_accuracy"] = float(metrics["balanced_accuracy"])
    metadata["test_accuracy"] = float(metrics["accuracy"])
    metadata["test_evaluated_on_full_test_scans"] = EXPECTED_TEST_SCANS
    write_json(metadata_path, metadata)
    print(
        f"completed=True test_balanced_accuracy={float(metrics['balanced_accuracy']):.6f} "
        f"test_accuracy={float(metrics['accuracy']):.6f}"
    )
    safe_plot_run_artifacts(run_dir)
    return "completed"


def _matrix_command(config_path: Path, patient_count: int, seed: int) -> str:
    return _command_text(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--patients",
            str(patient_count),
            "--seed",
            str(seed),
        ]
    )


def _print_matrix_progress(
    config: Mapping[str, Any], candidates: list[tuple[int, int]]
) -> int:
    print("BrainIAC Data Efficiency Progress")
    print("Patients  Scans  Seed   Test BA   Status")
    completed = 0
    for patient_count, seed in candidates:
        status, balanced_accuracy = candidate_status(config, patient_count, seed)
        if status == "COMPLETE":
            completed += 1
        metric_text = f"{balanced_accuracy:.4f}" if balanced_accuracy is not None else "-"
        print(
            f"{patient_count:<9}{patient_count * 4:<7}{seed:<7}"
            f"{metric_text:<10}{status}"
        )
    print(f"Completed: {completed} / {len(candidates)}")
    return completed


def run_matrix(
    config: Mapping[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    plan: Mapping[int, list[int]],
    *,
    dry_run: bool = False,
    run_function: Callable[..., str] | None = None,
    aggregate_function: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    """Execute the configured matrix sequentially and aggregate after success."""

    candidates = matrix_candidates(plan)
    _print_matrix_progress(config, candidates)
    run_function = run_function or run_one
    for index, (patient_count, seed) in enumerate(candidates, start=1):
        print(
            f"matrix_start={index}/{len(candidates)} patients={patient_count} "
            f"scans={patient_count * 4} seed={seed}"
        )
        try:
            result = run_function(
                config,
                config_path,
                checkpoint_path,
                patient_count,
                seed,
                dry_run=dry_run,
                show_details=False,
            )
        except Exception as exc:
            print(
                f"MATRIX FAILURE patients={patient_count} scans={patient_count * 4} seed={seed}",
                file=sys.stderr,
            )
            print(f"command={_matrix_command(config_path, patient_count, seed)}", file=sys.stderr)
            print(f"failure={exc}", file=sys.stderr)
            raise
        status, balanced_accuracy = candidate_status(config, patient_count, seed)
        if dry_run:
            status = "DRY-RUN"
        elif result == "skipped":
            status = "SKIPPED"
        elif status == "COMPLETE":
            status = "COMPLETE"
        metric_text = f" test_ba={balanced_accuracy:.4f}" if balanced_accuracy is not None else ""
        print(
            f"matrix_finished={index}/{len(candidates)} patients={patient_count} "
            f"seed={seed} status={status}{metric_text}"
        )

    if dry_run:
        print(f"dry_run_matrix_runs={len(candidates)}")
        return None

    aggregate_function = aggregate_function or aggregate_results
    result = aggregate_function(
        config_path=config_path,
        results_root=config["output"]["root_dir"],
    )
    print(f"matrix_aggregation_complete={result['summary_path']}")
    completed = sum(
        candidate_status(config, patient_count, seed)[0] == "COMPLETE"
        for patient_count, seed in candidates
    )
    print(f"Completed: {completed} / {len(candidates)}")
    return result


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    config = load_yaml_config(config_path)
    validate_finetune_config(config)
    plan = seed_plan_from_config(config)

    if args.dry_run_matrix or args.run_matrix:
        if args.patients is not None or args.seed is not None:
            raise SystemExit("--patients and --seed cannot be combined with a matrix mode")
        if args.force:
            raise SystemExit("--force cannot be combined with a matrix mode")
        checkpoint_path = resolve_repo_path(args.checkpoint or config["model"]["checkpoint_path"])
        if args.run_matrix:
            configured_checkpoint = resolve_repo_path(config["model"]["checkpoint_path"])
            if checkpoint_path != configured_checkpoint:
                raise ValueError(
                    "--checkpoint must resolve to the same general checkpoint configured in the YAML; "
                    "the evaluator uses that configured path"
                )
        run_matrix(
            config,
            config_path,
            checkpoint_path,
            plan,
            dry_run=args.dry_run_matrix,
        )
        return

    if args.patients is None or args.seed is None:
        raise SystemExit(
            "--patients and --seed are required unless --dry-run-matrix or --run-matrix is used"
        )
    validate_experiment_request(config, args.patients, args.seed)

    configured_checkpoint = resolve_repo_path(config["model"]["checkpoint_path"])
    checkpoint_path = resolve_repo_path(args.checkpoint or configured_checkpoint)
    if checkpoint_path != configured_checkpoint:
        raise ValueError(
            "--checkpoint must resolve to the same general checkpoint configured in the YAML; "
            "the evaluator uses that configured path"
        )

    run_one(
        config,
        config_path,
        checkpoint_path,
        args.patients,
        args.seed,
        prepare_only=args.prepare_only,
        dry_run=args.dry_run,
        train_only=args.train_only,
        evaluate_only=args.evaluate_only,
        force=args.force,
    )


if __name__ == "__main__":
    main()
