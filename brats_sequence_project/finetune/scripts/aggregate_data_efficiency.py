#!/usr/bin/env python3
"""Aggregate completed data-efficiency test evaluations without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from finetune_common import (
    CLASS_TO_INDEX,
    EXPECTED_TEST_SCANS,
    REPO_ROOT,
    is_valid_metrics_payload,
    load_yaml_config,
    resolve_repo_path,
    validate_finetune_config,
)


PATIENT_DIRECTORY = re.compile(r"^patients_(\d+)$")
SEED_DIRECTORY = re.compile(r"^seed_(-?\d+)$")
DEFAULT_COMPATIBILITY_CONFIG = (
    REPO_ROOT / "brats_sequence_project/finetune/config/data_efficiency.yml"
)
PATH_CONFIG_FIELDS = (
    ("model", "checkpoint_path"),
    ("data", "train_csv"),
    ("data", "val_csv"),
    ("data", "test_csv"),
)
SCALAR_CONFIG_FIELDS = (
    ("model", "image_size"),
    ("model", "hidden_size"),
    ("model", "num_classes"),
    ("model", "dropout"),
    ("model", "freeze_backbone"),
    ("data", "preprocessing"),
    ("data", "batch_size"),
    ("data", "validation_batch_size"),
    ("data", "num_workers"),
    ("data", "validation_num_workers"),
    ("data", "pin_memory"),
    ("training", "epochs"),
    ("training", "optimizer"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "scheduler"),
    ("training", "scheduler_mode"),
    ("training", "scheduler_factor"),
    ("training", "scheduler_patience"),
    ("training", "precision"),
    ("training", "checkpoint_metric"),
    ("training", "checkpoint_mode"),
)
FLOAT_CONFIG_FIELDS = {
    ("model", "dropout"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "scheduler_factor"),
    ("training", "scheduler_patience"),
}
STRING_CONFIG_FIELDS = {
    ("training", "optimizer"),
    ("training", "scheduler"),
    ("training", "scheduler_mode"),
    ("training", "precision"),
    ("training", "checkpoint_metric"),
    ("training", "checkpoint_mode"),
}
KNOWN_REPO_REFERENCES = (
    "checkpoints/BrainIAC.ckpt",
    "brats_sequence_project/outputs/train.csv",
    "brats_sequence_project/outputs/val.csv",
    "brats_sequence_project/outputs/test.csv",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _infer_patient_count(run_dir: Path, metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("patient_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    for key in ("selected_patient_ids", "selected_train_patients"):
        selected = metadata.get(key)
        if isinstance(selected, list) and selected:
            return len(selected)
    match = PATIENT_DIRECTORY.match(run_dir.parent.name)
    if match:
        return int(match.group(1))
    return None


def _infer_seed(run_dir: Path, metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("seed")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = SEED_DIRECTORY.match(run_dir.name)
    return int(match.group(1)) if match else None


def _best_validation_metric(run_dir: Path, metadata: Mapping[str, Any]) -> float | None:
    direct = _numeric(metadata.get("best_validation_balanced_accuracy"))
    if direct is not None:
        return direct
    history_path = run_dir / "training_history.json"
    if not history_path.is_file():
        return None
    try:
        history = _read_json(history_path)
    except (OSError, json.JSONDecodeError):
        return None
    values = []
    if isinstance(history, list):
        for record in history:
            if not isinstance(record, Mapping):
                continue
            metrics = record.get("validation_metrics")
            if isinstance(metrics, Mapping):
                value = _numeric(metrics.get("balanced_accuracy"))
                if value is not None:
                    values.append(value)
    return max(values) if values else None


def discover_completed_runs(results_root: str | Path) -> list[dict[str, Any]]:
    """Find valid test results under a root, ignoring partial/corrupt runs."""

    root = resolve_repo_path(results_root)
    if not root.is_dir():
        return []
    completed: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("test_evaluation/metrics.json")):
        try:
            metrics = _read_json(metrics_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not is_valid_metrics_payload(metrics, expected_test_scans=EXPECTED_TEST_SCANS):
            continue
        run_dir = metrics_path.parent.parent
        metadata_path = run_dir / "run_metadata.json"
        try:
            metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        patient_count = _infer_patient_count(run_dir, metadata)
        seed = _infer_seed(run_dir, metadata)
        if patient_count is None or seed is None:
            continue
        completed.append(
            {
                "patient_count": patient_count,
                "training_scans": patient_count * 4,
                "scans_per_class": patient_count,
                "seed": seed,
                "best_validation_balanced_accuracy": _best_validation_metric(run_dir, metadata),
                "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "test_accuracy": float(metrics["accuracy"]),
                "best_epoch": metadata.get("best_epoch"),
                "run_directory": str(run_dir),
                "metrics_path": str(metrics_path),
                "source": "data_efficiency",
                "metadata": dict(metadata),
                "metrics": metrics,
            }
        )
    return completed


def _looks_like_fixed_test_csv(metrics: Mapping[str, Any]) -> bool:
    value = metrics.get("test_csv")
    if not value:
        return False
    path = Path(str(value))
    return path.name == "test.csv" and "outputs" in path.parts


def _repo_path_reference(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    try:
        relative = resolve_repo_path(path).resolve().relative_to(REPO_ROOT.resolve())
        return relative.as_posix()
    except (OSError, ValueError):
        raw = path.as_posix()
        for reference in KNOWN_REPO_REFERENCES:
            if raw == reference or raw.endswith(f"/{reference}"):
                return reference
        return raw


def _same_repo_path(left: Any, right: Any) -> bool:
    left_reference = _repo_path_reference(left)
    right_reference = _repo_path_reference(right)
    return left_reference is not None and left_reference == right_reference


def _config_value(config: Mapping[str, Any], section: str, key: str) -> Any:
    values = config.get(section, {})
    return values.get(key) if isinstance(values, Mapping) else None


def _normalized_config_value(section: str, key: str, value: Any) -> Any:
    if (section, key) in FLOAT_CONFIG_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if (section, key) in STRING_CONFIG_FIELDS:
        return str(value).lower()
    return value


def _same_reproduction_config(
    candidate_config: Mapping[str, Any], expected_config: Mapping[str, Any]
) -> bool:
    for section, key in PATH_CONFIG_FIELDS:
        if not _same_repo_path(
            _config_value(candidate_config, section, key),
            _config_value(expected_config, section, key),
        ):
            return False
    for section, key in SCALAR_CONFIG_FIELDS:
        candidate_value = _normalized_config_value(
            section, key, _config_value(candidate_config, section, key)
        )
        expected_value = _normalized_config_value(
            section, key, _config_value(expected_config, section, key)
        )
        if candidate_value is None or candidate_value != expected_value:
            return False
    return True


def _load_candidate_config(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    config_reference = metadata.get("config_path") or metadata.get("training_config")
    if not config_reference:
        return None
    config_path = resolve_repo_path(str(config_reference))
    if not config_path.is_file():
        return None
    try:
        config = load_yaml_config(config_path)
        validate_finetune_config(config)
    except Exception:
        return None
    return config


def _class_mapping_matches(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        normalized = {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError):
        return False
    return normalized == CLASS_TO_INDEX


def _full_finetuning_metadata_matches(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("full_fine_tuning") is False or metadata.get("backbone_frozen") is True:
        return False
    report = metadata.get("parameter_report")
    if isinstance(report, Mapping):
        try:
            total = int(report["total_parameters"])
            trainable = int(report["trainable_parameters"])
            frozen = int(report["frozen_parameters"])
        except (KeyError, TypeError, ValueError):
            return False
        if total <= 0 or trainable != total or frozen != 0:
            return False
        return True
    return metadata.get("full_fine_tuning") is True


def _checkpoint_selection_metadata_matches(metadata: Mapping[str, Any]) -> bool:
    if any(
        metadata.get(key) is True
        for key in ("test_used_for_selection", "test_used_for_checkpoint_selection")
    ):
        return False
    explicitly_excludes_test = (
        metadata.get("checkpoint_selected_without_test") is True
        or metadata.get("test_used_for_selection") is False
        or metadata.get("test_used_for_checkpoint_selection") is False
    )
    if not explicitly_excludes_test:
        return False
    metric = metadata.get("checkpoint_selection_metric") or metadata.get("checkpoint_metric")
    normalized_metric = str(metric).lower().replace("_", " ")
    return normalized_metric in {"validation balanced accuracy", "val balanced accuracy"}


def _compatible_full_data_baseline(
    run: Mapping[str, Any], expected_config: Mapping[str, Any]
) -> bool:
    if run.get("patient_count") != 295 or run.get("training_scans") != 1180:
        return False
    metadata = run.get("metadata", {})
    metrics = run.get("metrics", {})
    if not isinstance(metadata, Mapping) or not isinstance(metrics, Mapping):
        return False
    selected_patients = metadata.get("selected_train_patients") or metadata.get(
        "selected_patient_ids"
    )
    if (
        not isinstance(selected_patients, list)
        or len(selected_patients) != 295
        or len(set(map(str, selected_patients))) != 295
    ):
        return False
    if metadata.get("scan_count") is not None and metadata.get("scan_count") != 1180:
        return False
    if metadata.get("scans_per_class") is not None and metadata.get("scans_per_class") != 295:
        return False
    if metadata.get("smoke_test") is True:
        return False
    fraction = _numeric(metadata.get("fraction"))
    if fraction is not None and fraction != 1.0:
        return False
    if not _full_finetuning_metadata_matches(metadata):
        return False
    if not _checkpoint_selection_metadata_matches(metadata):
        return False

    candidate_config = _load_candidate_config(metadata)
    if candidate_config is None or not _same_reproduction_config(candidate_config, expected_config):
        return False

    expected_model = expected_config["model"]
    checkpoint_references = [
        metadata.get("general_brainiac_checkpoint"),
        metadata.get("checkpoint_path"),
        metadata.get("model_initialized_from"),
    ]
    checkpoint_references = [reference for reference in checkpoint_references if reference]
    if not checkpoint_references or not all(
        _same_repo_path(reference, expected_model["checkpoint_path"])
        for reference in checkpoint_references
    ):
        return False

    if not _class_mapping_matches(metadata.get("class_mapping")):
        return False
    if not _class_mapping_matches(metrics.get("class_mapping")):
        return False

    expected_data = expected_config["data"]
    if not _same_repo_path(metrics.get("test_csv"), expected_data["test_csv"]):
        return False
    for metadata_key, expected_key in (
        ("source_train_csv", "train_csv"),
        ("validation_csv", "val_csv"),
        ("source_validation_csv", "val_csv"),
        ("test_csv", "test_csv"),
    ):
        if metadata.get(metadata_key) is not None and not _same_repo_path(
            metadata[metadata_key], expected_data[expected_key]
        ):
            return False
    return True


def find_full_data_baseline(
    data_efficiency_root: str | Path,
    baseline_root: str | Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Find a demonstrably compatible 295-patient result without inventing its metric."""

    expected_config = expected_config or load_yaml_config(DEFAULT_COMPATIBILITY_CONFIG)

    candidates: list[dict[str, Any]] = []
    seen_metrics: set[str] = set()
    for root, source in (
        (data_efficiency_root, "data_efficiency"),
        (baseline_root, "existing_baseline"),
    ):
        for run in discover_completed_runs(root):
            metrics_path = str(run["metrics_path"])
            if metrics_path in seen_metrics:
                continue
            seen_metrics.add(metrics_path)
            run["source"] = source
            candidates.append(run)
    compatible = [
        run for run in candidates if _compatible_full_data_baseline(run, expected_config)
    ]
    if not compatible:
        return None
    return sorted(
        compatible,
        key=lambda run: (
            Path(run["run_directory"]).name != "fraction_100",
            run["source"] != "existing_baseline",
            run["run_directory"],
        ),
    )[0]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(
    runs: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for run in runs:
        groups.setdefault(int(run["patient_count"]), []).append(run)
    baseline_ba = _numeric(baseline.get("test_balanced_accuracy")) if baseline else None
    summary: list[dict[str, Any]] = []
    for patient_count in sorted(groups):
        group = groups[patient_count]
        test_ba = [float(run["test_balanced_accuracy"]) for run in group]
        test_accuracy = [float(run["test_accuracy"]) for run in group]
        val_ba = [
            float(run["best_validation_balanced_accuracy"])
            for run in group
            if run.get("best_validation_balanced_accuracy") is not None
        ]
        mean_test_ba = statistics.fmean(test_ba)
        retained = (
            mean_test_ba / baseline_ba * 100
            if baseline_ba is not None and baseline_ba > 0
            else None
        )
        summary.append(
            {
                "patient_count": patient_count,
                "training_scans": patient_count * 4,
                "scans_per_class": patient_count,
                "number_of_completed_seeds": len({int(run["seed"]) for run in group}),
                "mean_test_balanced_accuracy": mean_test_ba,
                "std_test_balanced_accuracy": _sample_std(test_ba),
                "min_test_balanced_accuracy": min(test_ba),
                "max_test_balanced_accuracy": max(test_ba),
                "mean_test_accuracy": statistics.fmean(test_accuracy),
                "mean_best_validation_balanced_accuracy": _mean(val_ba),
                "performance_retained_percent": retained,
                "full_data_test_balanced_accuracy": baseline_ba,
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = [
    "patient_count",
    "training_scans",
    "scans_per_class",
    "number_of_completed_seeds",
    "mean_test_balanced_accuracy",
    "std_test_balanced_accuracy",
    "min_test_balanced_accuracy",
    "max_test_balanced_accuracy",
    "mean_test_accuracy",
    "mean_best_validation_balanced_accuracy",
    "performance_retained_percent",
    "full_data_test_balanced_accuracy",
]

RUN_FIELDS = [
    "patient_count",
    "patients",
    "training_scans",
    "scans",
    "scans_per_class",
    "seed",
    "best_validation_balanced_accuracy",
    "val_ba",
    "test_balanced_accuracy",
    "test_ba",
    "test_accuracy",
    "best_epoch",
    "run_directory",
    "source",
]


def run_rows_for_csv(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "patient_count": run["patient_count"],
            "patients": run["patient_count"],
            "training_scans": run["training_scans"],
            "scans": run["training_scans"],
            "scans_per_class": run["scans_per_class"],
            "seed": run["seed"],
            "best_validation_balanced_accuracy": run["best_validation_balanced_accuracy"],
            "val_ba": run["best_validation_balanced_accuracy"],
            "test_balanced_accuracy": run["test_balanced_accuracy"],
            "test_ba": run["test_balanced_accuracy"],
            "test_accuracy": run["test_accuracy"],
            "best_epoch": run.get("best_epoch"),
            "run_directory": run["run_directory"],
            "source": run["source"],
        }
        for run in sorted(runs, key=lambda item: (int(item["patient_count"]), int(item["seed"])))
    ]


def plot_curves(summary: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires an existing matplotlib installation") from exc
    if not summary:
        raise RuntimeError("Cannot plot a data-efficiency curve with no completed runs")

    scans = [int(row["training_scans"]) for row in summary]
    patients = [int(row["patient_count"]) for row in summary]
    means = [float(row["mean_test_balanced_accuracy"]) for row in summary]
    errors = [float(row["std_test_balanced_accuracy"]) for row in summary]
    for x_values, path, xlabel in (
        (scans, output_root / "data_efficiency_curve.png", "Training scans"),
        (patients, output_root / "data_efficiency_curve_by_patients.png", "Training patients"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.errorbar(x_values, means, yerr=errors, marker="o", capsize=3)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Test balanced accuracy")
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_COMPATIBILITY_CONFIG),
        help="Data-efficiency config whose fixed split and training contract must match baselines",
    )
    parser.add_argument(
        "--results-root",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/results/data_efficiency"),
    )
    parser.add_argument(
        "--baseline-root",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/results"),
        help="Also inspect this root for an existing compatible 295-patient baseline",
    )
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compatibility_config = load_yaml_config(args.config)
    validate_finetune_config(compatibility_config)
    results_root = resolve_repo_path(args.results_root)
    runs = discover_completed_runs(results_root)
    baseline = find_full_data_baseline(
        results_root,
        args.baseline_root,
        expected_config=compatibility_config,
    )
    if baseline is not None and not any(
        run["metrics_path"] == baseline["metrics_path"] for run in runs
    ):
        baseline = dict(baseline)
        baseline["source"] = "existing_baseline"
        runs.append(baseline)
    summary = aggregate_runs(runs, baseline)

    summary_path = results_root / "data_efficiency_summary.csv"
    all_runs_path = results_root / "data_efficiency_all_runs.csv"
    write_csv(summary_path, summary, SUMMARY_FIELDS)
    write_csv(all_runs_path, run_rows_for_csv(runs), RUN_FIELDS)
    if args.plot:
        plot_curves(summary, results_root)

    baseline_text = (
        f"{baseline['test_balanced_accuracy']:.6f} from {baseline['run_directory']}"
        if baseline is not None
        else "absent; retention columns left blank"
    )
    print(f"completed_runs={len(runs)}")
    print(f"full_data_baseline={baseline_text}")
    print(f"summary={summary_path}")
    print(f"all_runs={all_runs_path}")
    if args.plot:
        print(f"plots={results_root / 'data_efficiency_curve.png'}")


if __name__ == "__main__":
    main()
