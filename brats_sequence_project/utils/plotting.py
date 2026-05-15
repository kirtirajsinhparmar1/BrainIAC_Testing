"""Matplotlib plotting and metric helpers for Phase 3 outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_curve,
)


CLASS_NAMES = ["T1", "T2", "FLAIR", "T1CE"]
CLASS_LABELS = [0, 1, 2, 3]
PROBABILITY_COLUMNS = ["probability_T1", "probability_T2", "probability_FLAIR", "probability_T1CE"]


def _import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_history_rows(history_csv: Path) -> list[dict[str, float]]:
    if not history_csv.is_file():
        return []

    rows: list[dict[str, float]] = []
    with history_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def read_prediction_rows(predictions_csv: Path) -> list[dict[str, Any]]:
    if not predictions_csv.is_file():
        return []

    rows: list[dict[str, Any]] = []
    with predictions_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed: dict[str, Any] = dict(row)
            for key in ["true_label", "pred_label"]:
                try:
                    parsed[key] = int(parsed[key])
                except (TypeError, ValueError):
                    parsed[key] = None
            for key in [*PROBABILITY_COLUMNS, "confidence"]:
                if key in parsed:
                    try:
                        parsed[key] = float(parsed[key])
                    except (TypeError, ValueError):
                        parsed[key] = np.nan
            if "confidence" not in parsed or not np.isfinite(parsed["confidence"]):
                probabilities = [parsed.get(column, np.nan) for column in PROBABILITY_COLUMNS]
                parsed["confidence"] = float(np.nanmax(probabilities)) if probabilities else np.nan
            if "correct" in parsed:
                parsed["correct"] = str(parsed["correct"]).lower() in {"true", "1", "yes"}
            elif parsed.get("true_label") is not None and parsed.get("pred_label") is not None:
                parsed["correct"] = parsed["true_label"] == parsed["pred_label"]
            rows.append(parsed)
    return rows


def _series(rows: list[dict[str, float]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
        else:
            values.append(np.nan)
    return values


def _has_data(values: list[float]) -> bool:
    return any(np.isfinite(value) for value in values)


def _plot_lines(
    rows: list[dict[str, float]],
    series: list[tuple[str, str]],
    title: str,
    y_label: str,
    output_path: Path,
) -> Path | None:
    epochs = _series(rows, "epoch")
    if not rows or not _has_data(epochs):
        return None

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for key, label in series:
        values = _series(rows, key)
        if _has_data(values):
            axis.plot(epochs, values, marker="o", linewidth=1.8, label=label)
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_training_history(history_csv: Path, output_dir: Path) -> list[Path]:
    rows = read_history_rows(history_csv)
    output_dir = ensure_dir(output_dir)
    created: list[Path] = []

    plot_specs = [
        (
            "loss_curve.png",
            [("train_loss", "train_loss"), ("val_loss", "val_loss")],
            "Training and Validation Loss",
            "Loss",
        ),
        (
            "accuracy_curve.png",
            [("train_accuracy", "train_accuracy"), ("val_accuracy", "val_accuracy")],
            "Training and Validation Accuracy",
            "Accuracy",
        ),
        (
            "f1_curve.png",
            [
                ("train_macro_f1", "train_macro_f1"),
                ("val_macro_f1", "val_macro_f1"),
                ("train_weighted_f1", "train_weighted_f1"),
                ("val_weighted_f1", "val_weighted_f1"),
            ],
            "Macro and Weighted F1 Over Epochs",
            "F1 score",
        ),
        (
            "precision_recall_curve_over_epochs.png",
            [
                ("train_precision_macro", "train_precision_macro"),
                ("val_precision_macro", "val_precision_macro"),
                ("train_recall_macro", "train_recall_macro"),
                ("val_recall_macro", "val_recall_macro"),
            ],
            "Macro Precision and Recall Over Epochs",
            "Score",
        ),
        (
            "learning_rate_curve.png",
            [("learning_rate", "learning_rate")],
            "Learning Rate Over Epochs",
            "Learning rate",
        ),
    ]
    for filename, series, title, y_label in plot_specs:
        path = _plot_lines(rows, series, title, y_label, output_dir / filename)
        if path is not None:
            created.append(path)

    overfitting_path = _plot_overfitting_gap(rows, output_dir / "overfitting_gap_curve.png")
    if overfitting_path is not None:
        created.append(overfitting_path)

    summary_path = _plot_combined_training_summary(rows, output_dir / "combined_training_summary.png")
    if summary_path is not None:
        created.append(summary_path)

    return created


def _plot_overfitting_gap(rows: list[dict[str, float]], output_path: Path) -> Path | None:
    epochs = _series(rows, "epoch")
    train_accuracy = _series(rows, "train_accuracy")
    val_accuracy = _series(rows, "val_accuracy")
    train_loss = _series(rows, "train_loss")
    val_loss = _series(rows, "val_loss")
    if not rows or not _has_data(epochs):
        return None

    accuracy_gap = np.asarray(train_accuracy) - np.asarray(val_accuracy)
    loss_gap = np.asarray(train_loss) - np.asarray(val_loss)
    if not (_has_data(accuracy_gap.tolist()) or _has_data(loss_gap.tolist())):
        return None

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    if _has_data(accuracy_gap.tolist()):
        axis.plot(epochs, accuracy_gap, marker="o", linewidth=1.8, label="train_accuracy - val_accuracy")
    if _has_data(loss_gap.tolist()):
        axis.plot(epochs, loss_gap, marker="o", linewidth=1.8, label="train_loss - val_loss")
    axis.axhline(0, color="black", linewidth=1, alpha=0.5)
    axis.set_title("Overfitting Gap Over Epochs")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Gap")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_combined_training_summary(rows: list[dict[str, float]], output_path: Path) -> Path | None:
    epochs = _series(rows, "epoch")
    if not rows or not _has_data(epochs):
        return None

    plt = _import_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        (axes[0, 0], [("train_loss", "train"), ("val_loss", "val")], "Loss", "Loss"),
        (axes[0, 1], [("train_accuracy", "train"), ("val_accuracy", "val")], "Accuracy", "Accuracy"),
        (axes[1, 0], [("train_macro_f1", "train"), ("val_macro_f1", "val")], "Macro F1", "F1"),
        (axes[1, 1], [("train_weighted_f1", "train"), ("val_weighted_f1", "val")], "Weighted F1", "F1"),
    ]

    plotted_any = False
    for axis, series, title, y_label in specs:
        plotted = False
        for key, label in series:
            values = _series(rows, key)
            if _has_data(values):
                axis.plot(epochs, values, marker="o", linewidth=1.8, label=label)
                plotted = True
                plotted_any = True
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend()

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle("Phase 3 Frozen BrainIAC Classifier Training Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def prediction_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray([row["true_label"] for row in rows if row.get("true_label") is not None], dtype=int)
    y_pred = np.asarray([row["pred_label"] for row in rows if row.get("pred_label") is not None], dtype=int)
    probabilities = np.asarray(
        [[float(row.get(column, np.nan)) for column in PROBABILITY_COLUMNS] for row in rows],
        dtype=float,
    )
    confidence = np.asarray([float(row.get("confidence", np.nan)) for row in rows], dtype=float)
    return y_true, y_pred, probabilities, confidence


def compute_prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true, y_pred, probabilities, _ = prediction_arrays(rows)
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    raw_matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    row_sums = raw_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(raw_matrix, row_sums, out=np.zeros_like(raw_matrix, dtype=float), where=row_sums != 0)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_metrics": {
            class_name: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix_raw": raw_matrix.astype(int).tolist(),
        "confusion_matrix_normalized": normalized_matrix.tolist(),
        "total_samples": int(len(y_true)),
        "correct_samples": int(np.sum(y_true == y_pred)),
        "incorrect_samples": int(np.sum(y_true != y_pred)),
    }

    roc_auc_per_class: dict[str, float] = {}
    average_precision_per_class: dict[str, float] = {}
    if probabilities.shape == (len(y_true), len(CLASS_NAMES)) and np.isfinite(probabilities).all():
        for class_index, class_name in enumerate(CLASS_NAMES):
            binary_true = (y_true == class_index).astype(int)
            if len(np.unique(binary_true)) < 2:
                continue
            roc_fpr, roc_tpr, _ = roc_curve(binary_true, probabilities[:, class_index])
            roc_auc_per_class[class_name] = float(auc(roc_fpr, roc_tpr))
            average_precision_per_class[class_name] = float(
                average_precision_score(binary_true, probabilities[:, class_index])
            )

    if roc_auc_per_class:
        metrics["roc_auc_per_class"] = roc_auc_per_class
        metrics["macro_roc_auc"] = float(np.mean(list(roc_auc_per_class.values())))
    if average_precision_per_class:
        metrics["average_precision_per_class"] = average_precision_per_class
        metrics["macro_average_precision"] = float(np.mean(list(average_precision_per_class.values())))

    return metrics


def save_predictions_csv(path: Path, prediction_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "patient_id",
        "image_path",
        "modality",
        "true_label",
        "pred_label",
        "correct",
        *PROBABILITY_COLUMNS,
        "confidence",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)


def plot_prediction_outputs(
    predictions_csv: Path,
    output_dir: Path,
    metrics_json: Path | None = None,
    split_name: str = "test",
) -> list[Path]:
    rows = read_prediction_rows(predictions_csv)
    output_dir = ensure_dir(output_dir)
    if not rows:
        return []

    metrics = compute_prediction_metrics(rows)
    if metrics_json is not None and metrics_json.is_file():
        try:
            metrics = {**metrics, **json.loads(metrics_json.read_text())}
        except json.JSONDecodeError:
            pass

    created: list[Path] = []
    y_true, y_pred, probabilities, confidence = prediction_arrays(rows)
    raw_matrix = np.asarray(metrics.get("confusion_matrix_raw", confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)))
    normalized_matrix = np.asarray(
        metrics.get("confusion_matrix_normalized", _normalize_confusion_matrix(raw_matrix)),
        dtype=float,
    )

    for matrix, filename, title, value_format in [
        (raw_matrix, "confusion_matrix_raw.png", f"{split_name.title()} Confusion Matrix (Counts)", "d"),
        (
            normalized_matrix,
            "confusion_matrix_normalized.png",
            f"{split_name.title()} Confusion Matrix (Row-Normalized)",
            ".2f",
        ),
    ]:
        path = _plot_confusion_matrix(matrix, output_dir / filename, title, value_format)
        if path is not None:
            created.append(path)

    for plotter in [
        lambda: _plot_per_class_prf(metrics, output_dir / "per_class_precision_recall_f1.png"),
        lambda: _plot_per_class_accuracy(y_true, y_pred, output_dir / "per_class_accuracy.png"),
        lambda: _plot_class_distribution(y_true, output_dir / "class_distribution_true.png", "True Label Distribution"),
        lambda: _plot_class_distribution(
            y_pred,
            output_dir / "class_distribution_predicted.png",
            "Predicted Label Distribution",
        ),
        lambda: _plot_confidence_histogram(confidence, output_dir / "confidence_histogram.png"),
        lambda: _plot_confidence_correct_vs_incorrect(
            confidence,
            y_true == y_pred,
            output_dir / "confidence_correct_vs_incorrect.png",
        ),
        lambda: _plot_prediction_error_by_class(y_true, y_pred, output_dir / "prediction_error_by_class.png"),
        lambda: _plot_roc_curves(y_true, probabilities, output_dir / "roc_curves_multiclass.png"),
        lambda: _plot_precision_recall_curves(
            y_true,
            probabilities,
            output_dir / "precision_recall_curves_multiclass.png",
        ),
    ]:
        path = plotter()
        if path is not None:
            created.append(path)

    return created


def _normalize_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


def _plot_confusion_matrix(matrix: np.ndarray, output_path: Path, title: str, value_format: str) -> Path | None:
    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text = format(int(value), value_format) if value_format == "d" else format(float(value), value_format)
            axis.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_per_class_prf(metrics: dict[str, Any], output_path: Path) -> Path | None:
    per_class = metrics.get("per_class_metrics")
    if not per_class:
        return None

    precision = [float(per_class.get(name, {}).get("precision", 0.0)) for name in CLASS_NAMES]
    recall = [float(per_class.get(name, {}).get("recall", 0.0)) for name in CLASS_NAMES]
    f1 = [float(per_class.get(name, {}).get("f1", 0.0)) for name in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - width, precision, width, label="Precision")
    axis.bar(x, recall, width, label="Recall")
    axis.bar(x + width, f1, width, label="F1")
    axis.set_title("Per-Class Precision, Recall, and F1")
    axis.set_xlabel("Class")
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels(CLASS_NAMES)
    axis.set_ylim(0, 1.05)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path | None:
    accuracies = []
    for label in CLASS_LABELS:
        mask = y_true == label
        accuracies.append(float(np.mean(y_pred[mask] == label)) if np.any(mask) else 0.0)

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(CLASS_NAMES, accuracies)
    axis.set_title("Per-Class Accuracy")
    axis.set_xlabel("Class")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1.05)
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_class_distribution(labels: np.ndarray, output_path: Path, title: str) -> Path | None:
    counts = [int(np.sum(labels == label)) for label in CLASS_LABELS]
    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(CLASS_NAMES, counts)
    axis.set_title(title)
    axis.set_xlabel("Class")
    axis.set_ylabel("Samples")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_confidence_histogram(confidence: np.ndarray, output_path: Path) -> Path | None:
    confidence = confidence[np.isfinite(confidence)]
    if confidence.size == 0:
        return None

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(confidence, bins=20, color="#4C78A8", edgecolor="black", alpha=0.85)
    axis.set_title("Prediction Confidence Histogram")
    axis.set_xlabel("Max predicted probability")
    axis.set_ylabel("Samples")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_confidence_correct_vs_incorrect(
    confidence: np.ndarray,
    correct_mask: np.ndarray,
    output_path: Path,
) -> Path | None:
    valid = np.isfinite(confidence)
    if not np.any(valid):
        return None

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    axis.hist(
        confidence[valid & correct_mask],
        bins=bins,
        alpha=0.7,
        label="Correct",
        color="#59A14F",
        edgecolor="black",
    )
    axis.hist(
        confidence[valid & ~correct_mask],
        bins=bins,
        alpha=0.7,
        label="Incorrect",
        color="#E15759",
        edgecolor="black",
    )
    axis.set_title("Confidence: Correct vs Incorrect Predictions")
    axis.set_xlabel("Max predicted probability")
    axis.set_ylabel("Samples")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_prediction_error_by_class(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> Path | None:
    mistakes = [int(np.sum((y_true == label) & (y_pred != label))) for label in CLASS_LABELS]
    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(CLASS_NAMES, mistakes, color="#E15759")
    axis.set_title("Prediction Errors by True Class")
    axis.set_xlabel("True class")
    axis.set_ylabel("Mistakes")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_roc_curves(y_true: np.ndarray, probabilities: np.ndarray, output_path: Path) -> Path | None:
    if probabilities.shape != (len(y_true), len(CLASS_NAMES)) or not np.isfinite(probabilities).all():
        print("warning: skipped ROC plot because probability columns are missing or invalid.")
        return None

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(7, 6))
    plotted = False
    auc_values: list[float] = []
    for label, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == label).astype(int)
        if len(np.unique(binary_true)) < 2:
            print(f"warning: skipped ROC for {class_name}; class is missing from true labels.")
            continue
        fpr, tpr, _ = roc_curve(binary_true, probabilities[:, label])
        roc_auc = auc(fpr, tpr)
        auc_values.append(float(roc_auc))
        axis.plot(fpr, tpr, linewidth=1.8, label=f"{class_name} AUC={roc_auc:.3f}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    if auc_values:
        axis.set_title(f"One-vs-Rest ROC Curves (Macro AUC={np.mean(auc_values):.3f})")
    else:
        axis.set_title("One-vs-Rest ROC Curves")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_precision_recall_curves(y_true: np.ndarray, probabilities: np.ndarray, output_path: Path) -> Path | None:
    if probabilities.shape != (len(y_true), len(CLASS_NAMES)) or not np.isfinite(probabilities).all():
        print("warning: skipped precision-recall plot because probability columns are missing or invalid.")
        return None

    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(7, 6))
    plotted = False
    ap_values: list[float] = []
    for label, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == label).astype(int)
        if len(np.unique(binary_true)) < 2:
            print(f"warning: skipped precision-recall curve for {class_name}; class is missing from true labels.")
            continue
        precision, recall, _ = precision_recall_curve(binary_true, probabilities[:, label])
        average_precision = average_precision_score(binary_true, probabilities[:, label])
        ap_values.append(float(average_precision))
        axis.plot(recall, precision, linewidth=1.8, label=f"{class_name} AP={average_precision:.3f}")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    if ap_values:
        axis.set_title(f"One-vs-Rest Precision-Recall Curves (Macro AP={np.mean(ap_values):.3f})")
    else:
        axis.set_title("One-vs-Rest Precision-Recall Curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def save_top_prediction_csvs(predictions_csv: Path, output_dir: Path, top_k: int = 20) -> list[Path]:
    rows = read_prediction_rows(predictions_csv)
    if not rows:
        return []

    output_dir = ensure_dir(output_dir)
    wrong_rows = [row for row in rows if not row.get("correct", False)]
    uncertain_rows = list(rows)
    wrong_rows.sort(key=lambda row: float(row.get("confidence", -np.inf)), reverse=True)
    uncertain_rows.sort(key=lambda row: float(row.get("confidence", np.inf)))

    fieldnames = [
        "patient_id",
        "image_path",
        "modality",
        "true_label",
        "pred_label",
        "confidence",
        *PROBABILITY_COLUMNS,
    ]
    created: list[Path] = []
    for filename, selected_rows in [
        ("top_confident_wrong_predictions.csv", wrong_rows[:top_k]),
        ("top_uncertain_predictions.csv", uncertain_rows[:top_k]),
    ]:
        path = output_dir / filename
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected_rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        created.append(path)
    return created
