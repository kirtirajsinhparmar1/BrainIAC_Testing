"""Train a classifier on cached BrainIAC features for preprocessing ablations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting import (
    CLASS_NAMES,
    PROBABILITY_COLUMNS,
    compute_prediction_metrics,
    plot_prediction_outputs,
    plot_training_history,
    save_predictions_csv,
    save_top_prediction_csvs,
)


LABEL_MAPPING = {"T1": 0, "T2": 1, "FLAIR": 2, "T1CE": 3}
ID_TO_LABEL = {value: key for key, value in LABEL_MAPPING.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_features", type=Path, required=True)
    parser.add_argument("--val_features", type=Path, required=True)
    parser.add_argument("--test_features", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


class CachedFeatureDataset(Dataset):
    """Serve cached BrainIAC features plus metadata for classifier training or evaluation."""

    def __init__(self, cache_path: Path) -> None:
        cache_path = cache_path.expanduser().resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Cached features not found: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)

        self.cache_path = cache_path
        self.features = payload["features"].float()
        self.labels = payload["labels"].long()
        self.patient_ids = list(payload["patient_ids"])
        self.modalities = list(payload["modalities"])
        self.image_paths = list(payload["image_paths"])
        self.split_names = list(payload.get("split_names", [payload.get("split_name", "unknown")] * len(self.labels)))
        self.variant = payload.get("variant", "unknown")
        self.label_mapping = payload.get("label_mapping", LABEL_MAPPING)

        if self.features.ndim != 2 or self.features.shape[1] != 768:
            raise ValueError(f"Expected cached features [N,768], got {tuple(self.features.shape)}")
        if len(self.labels) != self.features.shape[0]:
            raise ValueError("Cached feature/label lengths do not match.")

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


class CachedFeatureClassifier(nn.Module):
    """MLP head matching the frozen-backbone Phase 3 classifier."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 4) -> None:
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


def compute_basic_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_history(output_path: Path, rows: list[dict[str, float | int]]) -> None:
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
        "train_precision_macro",
        "val_precision_macro",
        "train_recall_macro",
        "val_recall_macro",
        "learning_rate",
        "epoch_time_seconds",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    classifier.train()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in optional_tqdm(loader, desc=f"train epoch {epoch}"):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = classifier(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * features.shape[0]
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def evaluate_classifier(
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    classifier.eval()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
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

            for index, true_label in enumerate(batch_true):
                row = {
                    "patient_id": batch["patient_id"][index],
                    "image_path": batch["image_path"][index],
                    "modality": batch["modality"][index],
                    "true_label": true_label,
                    "pred_label": batch_pred[index],
                    "correct": bool(true_label == batch_pred[index]),
                    "confidence": float(max(batch_probs[index])),
                    "split_name": batch["split_name"][index],
                }
                row.update(
                    {
                        probability_column: float(batch_probs[index][probability_index])
                        for probability_index, probability_column in enumerate(PROBABILITY_COLUMNS)
                    }
                )
                prediction_rows.append(row)

    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics, prediction_rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CachedFeatureDataset(args.train_features)
    val_dataset = CachedFeatureDataset(args.val_features)
    test_dataset = CachedFeatureDataset(args.test_features) if args.test_features else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    classifier = CachedFeatureClassifier().to(device)
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
    history_rows: list[dict[str, float | int]] = []
    best_path = output_dir / "best_cached_classifier.pt"
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

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "train_precision_macro": train_metrics["precision_macro"],
            "val_precision_macro": val_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "learning_rate": learning_rate,
            "epoch_time_seconds": epoch_time_seconds,
        }
        history_rows.append(row)
        save_history(output_dir / "train_history.csv", history_rows)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.6f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.6f} val_acc={row['val_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f}"
        )

        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            best_metrics = {
                "epoch": epoch,
                "selection_metric": "val_macro_f1",
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "variant": train_dataset.variant,
            }
            best_val_prediction_rows = val_prediction_rows
            torch.save(
                {
                    "classifier_state_dict": classifier.state_dict(),
                    "classifier_config": {"input_dim": 768, "hidden_dim": 256, "num_classes": 4},
                    "epoch": epoch,
                    "label_mapping": train_dataset.label_mapping,
                    "id_to_label": ID_TO_LABEL,
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
        raise RuntimeError("Training finished without a best cached classifier checkpoint.")

    created_plots = plot_training_history(output_dir / "train_history.csv", output_dir / "plots")
    for plot_path in created_plots:
        print(f"plot: {plot_path}")

    if test_loader is not None:
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
        classifier.to(device)
        classifier.eval()
        test_metrics, prediction_rows = evaluate_classifier(classifier, test_loader, criterion, device, desc="test")
        metrics = compute_prediction_metrics(prediction_rows)
        metrics["cached_variant"] = train_dataset.variant
        metrics["classifier_epoch"] = checkpoint.get("epoch")
        metrics["label_mapping"] = LABEL_MAPPING
        metrics["class_names"] = CLASS_NAMES

        metrics_path = output_dir / "test_metrics.json"
        predictions_path = output_dir / "predictions.csv"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        save_predictions_csv(predictions_path, prediction_rows)
        eval_plots = plot_prediction_outputs(
            predictions_path,
            output_dir / "plots",
            metrics_json=metrics_path,
            split_name="test",
        )
        extra_csvs = save_top_prediction_csvs(predictions_path, output_dir / "plots")
        for path in [*eval_plots, *extra_csvs]:
            print(f"wrote: {path}")
        print(
            f"test_complete: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
        )

    print(f"training_complete: best_val_macro_f1={best_score:.4f}")
    print(f"best_cached_classifier: {best_path}")
    print(f"val_predictions: {output_dir / 'val_predictions.csv'}")


if __name__ == "__main__":
    main()
