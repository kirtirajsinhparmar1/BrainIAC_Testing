"""Train a small classifier head on frozen BrainIAC backbone features."""

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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting import PROBABILITY_COLUMNS, plot_training_history, save_predictions_csv


LABEL_MAPPING = {"T1": 0, "T2": 1, "FLAIR": 2, "T1CE": 3}
ID_TO_LABEL = {value: key for key, value in LABEL_MAPPING.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainiac_src", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


class SequenceClassifierHead(nn.Module):
    """Small MLP classifier for BrainIAC CLS-token features."""

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


def import_brats_dataset() -> type:
    try:
        from datasets import BraTSSequenceDataset
    except ImportError as exc:
        raise ImportError(
            "Could not import BraTSSequenceDataset. If the error mentions MONAI, install "
            "BrainIAC runtime dependencies first, including monai==1.3.2 from "
            "BrainIAC/requirements.txt."
        ) from exc

    return BraTSSequenceDataset


def import_vit_backbone(brainiac_src: Path) -> type[nn.Module]:
    brainiac_src = brainiac_src.expanduser().resolve()
    if not (brainiac_src / "model.py").is_file():
        raise FileNotFoundError(f"Could not find BrainIAC model.py under --brainiac_src: {brainiac_src}")

    sys.path.insert(0, str(brainiac_src))
    try:
        from model import ViTBackboneNet
    except ImportError as exc:
        raise ImportError(
            f"Could not import ViTBackboneNet from {brainiac_src / 'model.py'}. "
            "Install BrainIAC runtime dependencies before training."
        ) from exc

    return ViTBackboneNet


def load_frozen_backbone(brainiac_src: Path, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BrainIAC checkpoint not found: {checkpoint_path}")

    ViTBackboneNet = import_vit_backbone(brainiac_src)
    try:
        backbone = ViTBackboneNet(str(checkpoint_path))
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load BrainIAC checkpoint into ViTBackboneNet. Expected checkpoint "
            "keys must match BrainIAC/src/model.py and include backbone-prefixed ViT weights."
        ) from exc

    backbone.to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


def make_loader(csv_path: Path, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    BraTSSequenceDataset = import_brats_dataset()
    dataset = BraTSSequenceDataset(csv_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_basic_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_one_epoch(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> dict[str, float]:
    classifier.train()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in optional_tqdm(loader, desc=f"train epoch {epoch}"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Backbone is frozen: no gradient graph is built through BrainIAC.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                features = backbone(images)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = classifier(features)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def evaluate(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    desc: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    classifier.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in optional_tqdm(loader, desc=desc):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                features = backbone(images)
                logits = classifier(features)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            batch_true = labels.detach().cpu().tolist()
            batch_pred = predictions.detach().cpu().tolist()
            batch_probs = probabilities.detach().cpu().tolist()
            total_loss += loss.item() * labels.size(0)
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)

            for index, true_label in enumerate(batch_true):
                confidence = float(max(batch_probs[index]))
                row = {
                    "patient_id": batch["patient_id"][index],
                    "image_path": batch["image_path"][index],
                    "modality": batch["modality"][index],
                    "true_label": true_label,
                    "pred_label": batch_pred[index],
                    "correct": bool(true_label == batch_pred[index]),
                    "confidence": confidence,
                }
                row.update(
                    {
                        probability_column: float(batch_probs[index][probability_index])
                        for probability_index, probability_column in enumerate(PROBABILITY_COLUMNS)
                    }
                )
                prediction_rows.append(row)

    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, prediction_rows


def save_history(path: Path, rows: list[dict[str, float | int]]) -> None:
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
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def args_to_jsonable(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")
    if args.use_amp and not use_amp:
        print("amp_status: --use_amp requested but CUDA is unavailable; running without AMP.")
    print(f"device: {device}")

    backbone = load_frozen_backbone(args.brainiac_src, args.checkpoint_path, device)
    classifier = SequenceClassifierHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loader = make_loader(args.train_csv, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(args.val_csv, args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_score = -1.0
    best_metrics: dict[str, Any] | None = None
    best_prediction_rows: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []
    best_path = output_dir / "best_classifier_head.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            backbone,
            classifier,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
        )
        val_metrics, val_prediction_rows = evaluate(
            backbone,
            classifier,
            val_loader,
            criterion,
            device,
            use_amp,
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
            f"val_macro_f1={row['val_macro_f1']:.4f} val_weighted_f1={row['val_weighted_f1']:.4f}"
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
            }
            best_prediction_rows = val_prediction_rows
            torch.save(
                {
                    "classifier_state_dict": classifier.state_dict(),
                    "classifier_config": {"input_dim": 768, "hidden_dim": 256, "num_classes": 4},
                    "epoch": epoch,
                    "label_mapping": LABEL_MAPPING,
                    "id_to_label": ID_TO_LABEL,
                    "val_metrics": val_metrics,
                    "train_args": args_to_jsonable(args),
                },
                best_path,
            )
            (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2) + "\n")
            save_predictions_csv(output_dir / "val_predictions.csv", best_prediction_rows)
            print(f"saved best classifier head: {best_path}")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early_stop: no val_macro_f1 improvement for {args.patience} epochs.")
            break

    if best_metrics is None:
        raise RuntimeError("Training finished without saving a best classifier checkpoint.")

    created_plots = plot_training_history(output_dir / "train_history.csv", output_dir / "plots")
    print(f"training_complete: best_val_macro_f1={best_score:.4f} at epoch {best_metrics['epoch']}")
    print(f"best_classifier_head: {best_path}")
    print(f"val_predictions: {output_dir / 'val_predictions.csv'}")
    for plot_path in created_plots:
        print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
