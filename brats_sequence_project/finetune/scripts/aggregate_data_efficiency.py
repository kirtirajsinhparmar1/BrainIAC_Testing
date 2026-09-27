#!/usr/bin/env python3
"""Aggregate completed data-efficiency test evaluations without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from finetune_common import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    EXPECTED_TEST_SCANS,
    REPO_ROOT,
    is_valid_metrics_payload,
    load_yaml_config,
    resolve_repo_path,
    validate_finetune_config,
)
from make_subset_manifest import seed_plan_from_config


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


def filter_runs_to_plan(
    runs: Sequence[Mapping[str, Any]], seed_plan: Mapping[int, Sequence[int]]
) -> list[dict[str, Any]]:
    """Keep only valid results belonging to the configured low-data matrix."""

    configured = {
        (int(patient_count), int(seed))
        for patient_count, seeds in seed_plan.items()
        for seed in seeds
    }
    return [
        dict(run)
        for run in runs
        if (int(run["patient_count"]), int(run["seed"])) in configured
    ]


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
                "completed_seed_count": len({int(run["seed"]) for run in group}),
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
    "completed_seed_count",
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


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires an existing matplotlib installation") from exc
    return plt


def _plot_history_points(
    history: Sequence[Mapping[str, Any]],
    field: str,
    *,
    nested_field: str | None = None,
) -> tuple[list[float], list[float]]:
    epochs: list[float] = []
    values: list[float] = []
    for index, record in enumerate(history, start=1):
        raw_value: Any
        if nested_field is None:
            raw_value = record.get(field)
        else:
            metrics = record.get(nested_field)
            raw_value = metrics.get(field) if isinstance(metrics, Mapping) else None
        value = _numeric(raw_value)
        if value is None:
            continue
        epoch = _numeric(record.get("epoch"))
        epochs.append(epoch if epoch is not None else float(index))
        values.append(value)
    return epochs, values


def _load_plot_history(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "training_history.json"
    if not history_path.is_file():
        raise RuntimeError(f"training history is missing: {history_path}")
    history = _read_json(history_path)
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"training history is empty or invalid: {history_path}")
    records = [record for record in history if isinstance(record, Mapping)]
    if not records:
        raise RuntimeError(f"training history contains no usable records: {history_path}")
    return [dict(record) for record in records]


def plot_run_artifacts(run_directory: str | Path) -> list[Path]:
    """Create plots for one completed run from its saved history and metrics."""

    plt = _matplotlib()
    run_dir = resolve_repo_path(run_directory)
    history = _load_plot_history(run_dir)
    metrics_path = run_dir / "test_evaluation" / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"test metrics are missing: {metrics_path}")
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"test metrics are invalid: {metrics_path}")

    train_epochs, train_losses = _plot_history_points(history, "train_loss")
    val_epochs, val_losses = _plot_history_points(history, "validation_loss")
    if not val_losses:
        val_epochs, val_losses = _plot_history_points(history, "val_loss")
    if not train_losses or not val_losses:
        raise RuntimeError(f"training history has no train/validation loss series: {metrics_path}")

    output_paths: list[Path] = []
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(train_epochs, train_losses, label="Training loss")
    axis.plot(val_epochs, val_losses, label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("BrainIAC fine-tuning loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    loss_path = run_dir / "loss_curve.png"
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)
    output_paths.append(loss_path)

    val_epochs, val_balanced_accuracy = _plot_history_points(
        history,
        "balanced_accuracy",
        nested_field="validation_metrics",
    )
    if not val_balanced_accuracy:
        val_epochs, val_balanced_accuracy = _plot_history_points(
            history, "validation_balanced_accuracy"
        )
    if not val_balanced_accuracy:
        raise RuntimeError(f"training history has no validation balanced-accuracy series: {metrics_path}")
    best_index = max(range(len(val_balanced_accuracy)), key=val_balanced_accuracy.__getitem__)
    best_epoch = val_epochs[best_index]
    best_value = val_balanced_accuracy[best_index]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(val_epochs, val_balanced_accuracy, marker="o", label="Validation balanced accuracy")
    axis.scatter([best_epoch], [best_value], color="tab:red", zorder=3, label=f"Best epoch {int(best_epoch)}")
    axis.annotate(
        f"{best_value:.4f}",
        (best_epoch, best_value),
        xytext=(6, 8),
        textcoords="offset points",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation balanced accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Validation balanced accuracy")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    validation_path = run_dir / "validation_balanced_accuracy_curve.png"
    figure.savefig(validation_path, dpi=160)
    plt.close(figure)
    output_paths.append(validation_path)

    matrix = metrics.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(CLASS_NAMES)
        or any(not isinstance(row, list) or len(row) != len(CLASS_NAMES) for row in matrix)
    ):
        raise RuntimeError(f"test metrics have no 4x4 confusion matrix: {metrics_path}")
    figure, axis = plt.subplots(figsize=(6, 5.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Final 500-scan test confusion matrix",
    )
    threshold = max(max(row) for row in matrix) / 2 if matrix else 0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    confusion_path = run_dir / "confusion_matrix.png"
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)
    output_paths.append(confusion_path)
    return output_paths


def safe_plot_run_artifacts(run_directory: str | Path) -> list[Path]:
    """Create per-run plots while keeping plotting failures non-fatal."""

    try:
        paths = plot_run_artifacts(run_directory)
    except Exception as exc:  # Plotting must never invalidate scientific outputs.
        print(f"WARNING: could not generate plots for {run_directory}: {exc}", file=sys.stderr)
        return []
    print(f"run_plots={','.join(str(path) for path in paths)}")
    return paths


def _summary_with_baseline(
    summary: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    rows = [dict(row) for row in summary]
    if baseline is None or any(int(row["patient_count"]) == 295 for row in rows):
        return rows
    balanced_accuracy = _numeric(baseline.get("test_balanced_accuracy"))
    accuracy = _numeric(baseline.get("test_accuracy"))
    if balanced_accuracy is None or accuracy is None:
        return rows
    rows.append(
        {
            "patient_count": 295,
            "training_scans": 1180,
            "scans_per_class": 295,
            "completed_seed_count": 1,
            "number_of_completed_seeds": 1,
            "mean_test_balanced_accuracy": balanced_accuracy,
            "std_test_balanced_accuracy": 0.0,
            "min_test_balanced_accuracy": balanced_accuracy,
            "max_test_balanced_accuracy": balanced_accuracy,
            "mean_test_accuracy": accuracy,
            "mean_best_validation_balanced_accuracy": baseline.get(
                "best_validation_balanced_accuracy"
            ),
            "performance_retained_percent": 100.0,
            "full_data_test_balanced_accuracy": balanced_accuracy,
        }
    )
    return rows


def _save_figure(figure: Any, path: Path, plt: Any) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_curves(
    summary: Sequence[Mapping[str, Any]],
    output_root: Path,
    runs: Sequence[Mapping[str, Any]] = (),
    baseline: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write aggregate balanced-accuracy, seed, retention, and accuracy plots."""

    plt = _matplotlib()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        _summary_with_baseline(summary, baseline),
        key=lambda row: int(row["training_scans"]),
    )
    if not rows:
        raise RuntimeError("Cannot plot a data-efficiency curve with no completed runs")

    scans = [int(row["training_scans"]) for row in rows]
    patients = [int(row["patient_count"]) for row in rows]
    means = [float(row["mean_test_balanced_accuracy"]) for row in rows]
    errors = [float(row.get("std_test_balanced_accuracy") or 0.0) for row in rows]
    output_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(scans, means, yerr=errors, marker="o", capsize=3)
    axis.set_xlabel("Training scans")
    axis.set_ylabel("Test balanced accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("BrainIAC data efficiency")
    axis.grid(True, alpha=0.3)
    output_paths.append(_save_figure(figure, output_root / "data_efficiency_curve.png", plt))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(patients, means, yerr=errors, marker="o", capsize=3)
    axis.set_xlabel("Training patients")
    axis.set_ylabel("Test balanced accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("BrainIAC data efficiency by patient count")
    axis.grid(True, alpha=0.3)
    output_paths.append(
        _save_figure(figure, output_root / "data_efficiency_curve_by_patients.png", plt)
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    seeds = sorted(
        {
            int(run["seed"])
            for run in runs
            if int(run["patient_count"]) != 295
        }
    )
    for seed in seeds:
        seed_runs = sorted(
            (
                run
                for run in runs
                if int(run["seed"]) == seed and int(run["patient_count"]) != 295
            ),
            key=lambda run: int(run["training_scans"]),
        )
        axis.plot(
            [int(run["training_scans"]) for run in seed_runs],
            [float(run["test_balanced_accuracy"]) for run in seed_runs],
            marker="o",
            label=f"seed {seed}",
        )
    if baseline is not None:
        baseline_ba = _numeric(baseline.get("test_balanced_accuracy"))
        if baseline_ba is not None:
            axis.scatter(
                [1180],
                [baseline_ba],
                marker="*",
                s=140,
                label="full-data baseline (one run)",
                zorder=3,
            )
    axis.set_xlabel("Training scans")
    axis.set_ylabel("Test balanced accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Data efficiency by patient-subset seed")
    axis.legend()
    axis.grid(True, alpha=0.3)
    output_paths.append(
        _save_figure(figure, output_root / "data_efficiency_individual_seeds.png", plt)
    )

    if baseline is not None:
        retention_rows = [
            row
            for row in rows
            if _numeric(row.get("performance_retained_percent")) is not None
        ]
        if retention_rows:
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(
                [int(row["training_scans"]) for row in retention_rows],
                [float(row["performance_retained_percent"]) for row in retention_rows],
                marker="o",
            )
            axis.set_xlabel("Training scans")
            axis.set_ylabel("Percentage of full-data balanced accuracy retained")
            axis.set_title("Performance retained versus full-data baseline")
            axis.grid(True, alpha=0.3)
            output_paths.append(
                _save_figure(figure, output_root / "performance_retained.png", plt)
            )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        scans,
        [float(row["mean_test_accuracy"]) for row in rows],
        marker="o",
    )
    axis.set_xlabel("Training scans")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("BrainIAC data efficiency by ordinary accuracy")
    axis.grid(True, alpha=0.3)
    output_paths.append(
        _save_figure(figure, output_root / "data_efficiency_accuracy_curve.png", plt)
    )
    return output_paths


def aggregate_results(
    *,
    config_path: str | Path = DEFAULT_COMPATIBILITY_CONFIG,
    results_root: str | Path = REPO_ROOT / "brats_sequence_project/finetune/results/data_efficiency",
    baseline_root: str | Path = REPO_ROOT / "brats_sequence_project/finetune/results",
) -> dict[str, Any]:
    """Aggregate the configured matrix and generate all available plots."""

    compatibility_config = load_yaml_config(config_path)
    validate_finetune_config(compatibility_config)
    seed_plan = seed_plan_from_config(compatibility_config)
    resolved_results_root = resolve_repo_path(results_root)
    discovered_runs = discover_completed_runs(resolved_results_root)
    runs = filter_runs_to_plan(discovered_runs, seed_plan)
    baseline = find_full_data_baseline(
        resolved_results_root,
        baseline_root,
        expected_config=compatibility_config,
    )
    if baseline is not None and not any(
        run["metrics_path"] == baseline["metrics_path"] for run in runs
    ):
        baseline = dict(baseline)
        baseline["source"] = "existing_baseline"
        runs.append(baseline)

    summary = aggregate_runs(runs, baseline)
    summary_path = resolved_results_root / "data_efficiency_summary.csv"
    all_runs_path = resolved_results_root / "data_efficiency_all_runs.csv"
    write_csv(summary_path, summary, SUMMARY_FIELDS)
    write_csv(all_runs_path, run_rows_for_csv(runs), RUN_FIELDS)

    for run in runs:
        safe_plot_run_artifacts(run["run_directory"])
    plot_paths: list[Path] = []
    if summary:
        try:
            plot_paths = plot_curves(summary, resolved_results_root, runs, baseline)
        except Exception as exc:  # Plotting must never invalidate CSV/numeric outputs.
            print(f"WARNING: could not generate aggregate plots: {exc}", file=sys.stderr)

    return {
        "runs": runs,
        "summary": summary,
        "baseline": baseline,
        "summary_path": summary_path,
        "all_runs_path": all_runs_path,
        "plot_paths": plot_paths,
    }


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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Retained for compatibility; aggregate plots are generated automatically",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_results(
        config_path=args.config,
        results_root=args.results_root,
        baseline_root=args.baseline_root,
    )
    runs = result["runs"]
    baseline = result["baseline"]
    summary_path = result["summary_path"]
    all_runs_path = result["all_runs_path"]

    baseline_text = (
        f"{baseline['test_balanced_accuracy']:.6f} from {baseline['run_directory']}"
        if baseline is not None
        else "absent; retention columns left blank"
    )
    print(f"completed_runs={len(runs)}")
    print(f"full_data_baseline={baseline_text}")
    print(f"summary={summary_path}")
    print(f"all_runs={all_runs_path}")
    print(f"plots={','.join(str(path) for path in result['plot_paths']) or 'unavailable'}")


if __name__ == "__main__":
    main()
