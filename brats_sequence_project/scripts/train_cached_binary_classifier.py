"""Train a binary T1-vs-T2 classifier on cached frozen BrainIAC features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve as sklearn_precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["T1", "T2"]
PROBABILITY_COLUMNS = ["probability_T1", "probability_T2"]
PREDICTION_COLUMNS = [
    "patient_id",
    "image_path",
    "modality",
    "true_label",
    "pred_label",
    "correct",
    "confidence",
    "probability_T1",
    "probability_T2",
    "split_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_features", type=Path, required=True)
    parser.add_argument("--val_features", type=Path, required=True)
    parser.add_argument("--test_features", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=200)
    return parser.parse_args()


def load_torch_payload(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cached features not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CachedBinaryFeatureDataset(Dataset):
    """Serve cached BrainIAC features plus CSV metadata for binary classification."""

    def __init__(self, cache_path: Path) -> None:
        payload = load_torch_payload(cache_path)
        self.cache_path = cache_path.expanduser().resolve()
        self.features = payload["features"].float()
        self.labels = payload["labels"].long()
        self.patient_ids = list(payload.get("patient_ids", ["unknown"] * len(self.labels)))
        self.modalities = list(payload.get("modalities", ["unknown"] * len(self.labels)))
        self.image_paths = list(payload.get("image_paths", [""] * len(self.labels)))
        self.split_names = list(payload.get("split_names", [payload.get("split_name", "unknown")] * len(self.labels)))
        self.variant = payload.get("variant", "unknown")
        self.label_mapping = payload.get("label_mapping", {"0": "T1", "1": "T2"})

        if self.features.ndim != 2 or self.features.shape[1] != 768:
            raise ValueError(f"Expected cached features [N,768], got {tuple(self.features.shape)}")
        if len(self.labels) != self.features.shape[0]:
            raise ValueError("Cached feature/label lengths do not match.")
        metadata_lengths = {
            "patient_ids": len(self.patient_ids),
            "modalities": len(self.modalities),
            "image_paths": len(self.image_paths),
            "split_names": len(self.split_names),
        }
        bad_lengths = {name: value for name, value in metadata_lengths.items() if value != len(self.labels)}
        if bad_lengths:
            raise ValueError(f"Cached metadata lengths do not match labels: {bad_lengths}")

        observed_labels = sorted({int(value) for value in self.labels.tolist()})
        if not set(observed_labels).issubset({0, 1}):
            raise ValueError(f"Binary classifier expects labels 0/1 only, found {observed_labels}")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": self.features[index],
            "label": self.labels[index],
            "patient_id": self.patient_ids[index],
            "modality": self.modalities[index],
            "image_path": self.image_paths[index],
            "split_name": self.split_names[index],
        }


class BinaryFeatureClassifier(nn.Module):
    """Small MLP head over frozen BrainIAC features."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optional_tqdm(iterable: Any, desc: str) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable


def safe_metric(callable_metric: Any, default: float | None = None) -> float | None:
    try:
        return float(callable_metric())
    except ValueError:
        return default


def compute_metrics(y_true: list[int], y_pred: list[int], probability_t2: list[float]) -> dict[str, float | None]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "binary_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "binary_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": safe_metric(lambda: roc_auc_score(y_true, probability_t2)),
        "average_precision": safe_metric(lambda: average_precision_score(y_true, probability_t2)),
    }


def save_history(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "train_macro_f1",
        "val_macro_f1",
        "train_weighted_f1",
        "val_weighted_f1",
        "train_binary_f1",
        "val_binary_f1",
        "train_precision_macro",
        "val_precision_macro",
        "train_recall_macro",
        "val_recall_macro",
        "train_roc_auc",
        "val_roc_auc",
        "train_average_precision",
        "val_average_precision",
        "learning_rate",
        "epoch_time_seconds",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_predictions_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float | None]:
    classifier.train()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    probability_t2: list[float] = []

    for batch in optional_tqdm(loader, desc=f"train epoch {epoch}"):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = classifier(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        running_loss += float(loss.item()) * features.shape[0]
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        probability_t2.extend(probabilities[:, 1].detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, probability_t2)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def evaluate_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    classifier.eval()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    probability_t2: list[float] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in optional_tqdm(loader, desc=desc):
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = classifier(features)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            running_loss += float(loss.item()) * features.shape[0]

            batch_true = labels.detach().cpu().tolist()
            batch_pred = predictions.detach().cpu().tolist()
            batch_probs = probabilities.detach().cpu().tolist()
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)
            probability_t2.extend([float(probabilities_for_case[1]) for probabilities_for_case in batch_probs])

            for index, true_label in enumerate(batch_true):
                probs = [float(value) for value in batch_probs[index]]
                row = {
                    "patient_id": batch["patient_id"][index],
                    "image_path": batch["image_path"][index],
                    "modality": batch["modality"][index],
                    "true_label": int(true_label),
                    "pred_label": int(batch_pred[index]),
                    "correct": bool(true_label == batch_pred[index]),
                    "confidence": float(max(probs)),
                    "probability_T1": probs[0],
                    "probability_T2": probs[1],
                    "split_name": batch["split_name"][index],
                }
                prediction_rows.append(row)

    metrics = compute_metrics(y_true, y_pred, probability_t2)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics, prediction_rows


def import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_history_rows(history_path: Path) -> list[dict[str, float]]:
    with history_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, float]] = []
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value in ("", None, "None"):
                    parsed[key] = float("nan")
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def plot_line_series(
    history_rows: list[dict[str, float]],
    series: list[tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = [int(row["epoch"]) for row in history_rows]
    for key, label in series:
        axis.plot(epochs, [row.get(key, float("nan")) for row in history_rows], marker="o", linewidth=1.6, label=label)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_overfitting_gap(history_rows: list[dict[str, float]], output_path: Path) -> Path:
    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = [int(row["epoch"]) for row in history_rows]
    loss_gap = [row["val_loss"] - row["train_loss"] for row in history_rows]
    f1_gap = [row["train_macro_f1"] - row["val_macro_f1"] for row in history_rows]
    axis.plot(epochs, loss_gap, marker="o", linewidth=1.6, label="val_loss - train_loss")
    axis.plot(epochs, f1_gap, marker="s", linewidth=1.6, label="train_macro_f1 - val_macro_f1")
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_title("Overfitting Gap")
    axis.set_xlabel("Epoch")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_training_history(history_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_history_rows(history_path)
    created = [
        plot_line_series(rows, [("train_loss", "train"), ("val_loss", "val")], "Loss Curve", "Cross-entropy loss", output_dir / "loss_curve.png"),
        plot_line_series(rows, [("train_accuracy", "train"), ("val_accuracy", "val")], "Accuracy Curve", "Accuracy", output_dir / "accuracy_curve.png"),
        plot_line_series(
            rows,
            [("train_macro_f1", "train macro F1"), ("val_macro_f1", "val macro F1"), ("train_binary_f1", "train T2 F1"), ("val_binary_f1", "val T2 F1")],
            "F1 Curve",
            "F1",
            output_dir / "f1_curve.png",
        ),
        plot_line_series(
            rows,
            [
                ("train_precision_macro", "train precision"),
                ("val_precision_macro", "val precision"),
                ("train_recall_macro", "train recall"),
                ("val_recall_macro", "val recall"),
            ],
            "Precision and Recall Over Epochs",
            "Score",
            output_dir / "precision_recall_curve_over_epochs.png",
        ),
        plot_line_series(rows, [("learning_rate", "learning rate")], "Learning Rate Curve", "Learning rate", output_dir / "learning_rate_curve.png"),
        plot_overfitting_gap(rows, output_dir / "overfitting_gap_curve.png"),
    ]
    return created


def read_prediction_rows(predictions_path: Path) -> list[dict[str, Any]]:
    with predictions_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            parsed = dict(row)
            for key in ["true_label", "pred_label"]:
                parsed[key] = int(parsed[key])
            for key in ["confidence", *PROBABILITY_COLUMNS]:
                parsed[key] = float(parsed[key])
            parsed["correct"] = str(parsed.get("correct", "")).lower() in {"true", "1", "yes"}
            rows.append(parsed)
    return rows


def plot_confusion_matrices(y_true: list[int], y_pred: list[int], output_dir: Path) -> list[Path]:
    plt = import_pyplot()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    created: list[Path] = []
    for normalized, filename, title in [
        (False, "confusion_matrix_raw.png", "Confusion Matrix"),
        (True, "confusion_matrix_normalized.png", "Normalized Confusion Matrix"),
    ]:
        display_matrix = matrix.astype(float)
        if normalized:
            row_sums = display_matrix.sum(axis=1, keepdims=True)
            display_matrix = np.divide(display_matrix, np.maximum(row_sums, 1.0))
        fig, axis = plt.subplots(figsize=(5.5, 5.0))
        image = axis.imshow(display_matrix, cmap="Blues", vmin=0.0)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xticks([0, 1], CLASS_NAMES)
        axis.set_yticks([0, 1], CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(title)
        for row_index in range(2):
            for column_index in range(2):
                text = f"{display_matrix[row_index, column_index]:.2f}" if normalized else str(int(display_matrix[row_index, column_index]))
                axis.text(column_index, row_index, text, ha="center", va="center", color="black")
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=250)
        plt.close(fig)
        created.append(output_path)
    return created


def plot_per_class_scores(y_true: list[int], y_pred: list[int], output_path: Path) -> Path:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.bar(x - width, precision, width=width, label="precision")
    axis.bar(x, recall, width=width, label="recall")
    axis.bar(x + width, f1, width=width, label="F1")
    axis.set_xticks(x, CLASS_NAMES)
    axis.set_ylim(0, 1.05)
    axis.set_title("Per-class Precision, Recall, F1")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_confidence(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt = import_pyplot()
    confidences = [float(row["confidence"]) for row in rows]
    created: list[Path] = []

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(confidences, bins=20, color="#4C72B0", edgecolor="white")
    axis.set_title("Prediction Confidence Histogram")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Cases")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "confidence_histogram.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    created.append(output_path)

    correct = [float(row["confidence"]) for row in rows if row.get("correct")]
    incorrect = [float(row["confidence"]) for row in rows if not row.get("correct")]
    fig, axis = plt.subplots(figsize=(6.5, 5))
    boxplot_values = [correct or [np.nan], incorrect or [np.nan]]
    try:
        axis.boxplot(boxplot_values, tick_labels=["correct", "incorrect"], showmeans=True)
    except TypeError:
        axis.boxplot(boxplot_values, labels=["correct", "incorrect"], showmeans=True)
    axis.set_ylim(0, 1.05)
    axis.set_title("Confidence: Correct vs Incorrect")
    axis.set_ylabel("Confidence")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "confidence_correct_vs_incorrect.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    created.append(output_path)
    return created


def plot_binary_curves(y_true: list[int], probability_t2: list[float], output_dir: Path) -> list[Path]:
    plt = import_pyplot()
    created: list[Path] = []
    if len(set(y_true)) < 2:
        return created

    fpr, tpr, _ = roc_curve(y_true, probability_t2)
    roc_auc = roc_auc_score(y_true, probability_t2)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, linewidth=2.0, label=f"AUC = {roc_auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    axis.set_title("Binary ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    fig.tight_layout()
    output_path = output_dir / "roc_curve_binary.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    created.append(output_path)

    precision, recall, _ = sklearn_precision_recall_curve(y_true, probability_t2)
    average_precision = average_precision_score(y_true, probability_t2)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, linewidth=2.0, label=f"AP = {average_precision:.3f}")
    axis.set_title("Binary Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower left")
    fig.tight_layout()
    output_path = output_dir / "precision_recall_curve_binary.png"
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    created.append(output_path)
    return created


def plot_prediction_outputs(predictions_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_prediction_rows(predictions_path)
    y_true = [int(row["true_label"]) for row in rows]
    y_pred = [int(row["pred_label"]) for row in rows]
    probability_t2 = [float(row["probability_T2"]) for row in rows]
    created: list[Path] = []
    created.extend(plot_confusion_matrices(y_true, y_pred, output_dir))
    created.append(plot_per_class_scores(y_true, y_pred, output_dir / "per_class_precision_recall_f1.png"))
    created.extend(plot_confidence(rows, output_dir))
    created.extend(plot_binary_curves(y_true, probability_t2, output_dir))
    return created


def metrics_with_class_names(metrics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metrics)
    enriched["class_names"] = CLASS_NAMES
    enriched["label_mapping"] = {"0": "T1", "1": "T2"}
    enriched["positive_class"] = "T2"
    return enriched


def metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("nan")
    return float(value)


def make_history_row(
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    learning_rate: float,
    epoch_time_seconds: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss": metric_value(train_metrics, "loss"),
        "val_loss": metric_value(val_metrics, "loss"),
        "train_accuracy": metric_value(train_metrics, "accuracy"),
        "val_accuracy": metric_value(val_metrics, "accuracy"),
        "train_macro_f1": metric_value(train_metrics, "macro_f1"),
        "val_macro_f1": metric_value(val_metrics, "macro_f1"),
        "train_weighted_f1": metric_value(train_metrics, "weighted_f1"),
        "val_weighted_f1": metric_value(val_metrics, "weighted_f1"),
        "train_binary_f1": metric_value(train_metrics, "binary_f1"),
        "val_binary_f1": metric_value(val_metrics, "binary_f1"),
        "train_precision_macro": metric_value(train_metrics, "precision_macro"),
        "val_precision_macro": metric_value(val_metrics, "precision_macro"),
        "train_recall_macro": metric_value(train_metrics, "recall_macro"),
        "val_recall_macro": metric_value(val_metrics, "recall_macro"),
        "train_roc_auc": train_metrics.get("roc_auc"),
        "val_roc_auc": val_metrics.get("roc_auc"),
        "train_average_precision": train_metrics.get("average_precision"),
        "val_average_precision": val_metrics.get("average_precision"),
        "learning_rate": learning_rate,
        "epoch_time_seconds": epoch_time_seconds,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    train_dataset = CachedBinaryFeatureDataset(args.train_features)
    val_dataset = CachedBinaryFeatureDataset(args.val_features)
    test_dataset = CachedBinaryFeatureDataset(args.test_features) if args.test_features else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"train_samples: {len(train_dataset)} val_samples: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"test_samples: {len(test_dataset)}")

    classifier = BinaryFeatureClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if test_dataset is not None
        else None
    )

    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    best_val_prediction_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    best_path = output_dir / "best_binary_classifier.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(classifier, train_loader, criterion, optimizer, device, epoch)
        val_metrics, val_prediction_rows = evaluate_classifier(
            classifier,
            val_loader,
            criterion,
            device,
            desc=f"val epoch {epoch}",
        )
        epoch_time_seconds = time.perf_counter() - epoch_start
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history_row = make_history_row(epoch, train_metrics, val_metrics, learning_rate, epoch_time_seconds)
        history_rows.append(history_row)
        save_history(output_dir / "train_history.csv", history_rows)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={history_row['train_loss']:.6f} train_acc={history_row['train_accuracy']:.4f} "
            f"val_loss={history_row['val_loss']:.6f} val_acc={history_row['val_accuracy']:.4f} "
            f"val_macro_f1={history_row['val_macro_f1']:.4f}"
        )

        score = metric_value(val_metrics, "macro_f1")
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            best_metrics = {
                "epoch": epoch,
                "selection_metric": "val_macro_f1",
                "train_metrics": metrics_with_class_names(train_metrics),
                "val_metrics": metrics_with_class_names(val_metrics),
                "variant": train_dataset.variant,
                "class_names": CLASS_NAMES,
                "label_mapping": {"0": "T1", "1": "T2"},
            }
            best_val_prediction_rows = val_prediction_rows
            torch.save(
                {
                    "classifier_state_dict": classifier.state_dict(),
                    "classifier_config": {"input_dim": 768, "hidden_dim": 256, "num_classes": 2},
                    "epoch": epoch,
                    "label_mapping": {"0": "T1", "1": "T2"},
                    "class_names": CLASS_NAMES,
                    "val_metrics": val_metrics,
                    "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "variant": train_dataset.variant,
                },
                best_path,
            )
            (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2) + "\n")
            save_predictions_csv(output_dir / "val_predictions.csv", best_val_prediction_rows)
            print(f"saved best classifier: {best_path}")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stop: no val_macro_f1 improvement for {args.patience} epochs.")
            break

    if best_metrics is None:
        raise RuntimeError("Training finished without a best binary classifier checkpoint.")

    for plot_path in plot_training_history(output_dir / "train_history.csv", plots_dir):
        print(f"plot: {plot_path}")

    checkpoint = load_torch_payload(best_path)
    classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    classifier.to(device)
    classifier.eval()

    if test_loader is not None:
        test_metrics, prediction_rows = evaluate_classifier(classifier, test_loader, criterion, device, desc="test")
        metrics_payload = metrics_with_class_names(test_metrics)
        metrics_payload["cached_variant"] = train_dataset.variant
        metrics_payload["classifier_epoch"] = checkpoint.get("epoch")
        metrics_path = output_dir / "test_metrics.json"
        predictions_path = output_dir / "predictions.csv"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")
        save_predictions_csv(predictions_path, prediction_rows)
    else:
        metrics_path = output_dir / "best_metrics.json"
        predictions_path = output_dir / "predictions.csv"
        save_predictions_csv(predictions_path, best_val_prediction_rows)

    for path in plot_prediction_outputs(predictions_path, plots_dir):
        print(f"wrote: {path}")

    final_metrics = json.loads(metrics_path.read_text())
    print(
        f"prediction_complete: accuracy={float(final_metrics.get('accuracy', final_metrics.get('val_metrics', {}).get('accuracy', 0.0))):.4f} "
        f"macro_f1={float(final_metrics.get('macro_f1', final_metrics.get('val_metrics', {}).get('macro_f1', 0.0))):.4f}"
    )
    print(f"training_complete: best_val_macro_f1={best_score:.4f}")
    print(f"best_binary_classifier: {best_path}")
    print(f"val_predictions: {output_dir / 'val_predictions.csv'}")
    print(f"predictions: {predictions_path}")


if __name__ == "__main__":
    main()
