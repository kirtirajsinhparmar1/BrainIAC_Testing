"""Evaluate a frozen BrainIAC backbone plus saved classifier head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting import (
    CLASS_NAMES,
    PROBABILITY_COLUMNS,
    compute_prediction_metrics,
    plot_prediction_outputs,
    save_predictions_csv,
    save_top_prediction_csvs,
)


LABEL_MAPPING = {"T1": 0, "T2": 1, "FLAIR": 2, "T1CE": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainiac_src", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--classifier_path", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


class SequenceClassifierHead(nn.Module):
    """Classifier architecture matching the Phase 3 training script."""

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
            "Install BrainIAC runtime dependencies before evaluation."
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


def load_classifier(classifier_path: Path, device: torch.device) -> tuple[SequenceClassifierHead, dict[str, Any]]:
    classifier_path = classifier_path.expanduser().resolve()
    if not classifier_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

    checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("classifier_config", {"input_dim": 768, "hidden_dim": 256, "num_classes": 4})
    classifier = SequenceClassifierHead(**config)
    classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    classifier.to(device)
    classifier.eval()
    return classifier, checkpoint


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    backbone = load_frozen_backbone(args.brainiac_src, args.checkpoint_path, device)
    classifier, classifier_checkpoint = load_classifier(args.classifier_path, device)

    BraTSSequenceDataset = import_brats_dataset()
    dataset = BraTSSequenceDataset(args.csv_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in optional_tqdm(loader, desc="evaluate"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            features = backbone(images)
            logits = classifier(features)

            assert tuple(logits.shape) == (images.shape[0], 4), f"Expected logits [B,4], got {tuple(logits.shape)}"

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            batch_true = labels.detach().cpu().tolist()
            batch_pred = predictions.detach().cpu().tolist()
            batch_probs = probabilities.detach().cpu().tolist()

            for index, true_label in enumerate(batch_true):
                row = {
                    "patient_id": batch["patient_id"][index],
                    "image_path": batch["image_path"][index],
                    "modality": batch["modality"][index],
                    "true_label": true_label,
                    "pred_label": batch_pred[index],
                    "correct": bool(true_label == batch_pred[index]),
                    "confidence": float(max(batch_probs[index])),
                }
                row.update(
                    {
                        probability_column: float(batch_probs[index][probability_index])
                        for probability_index, probability_column in enumerate(PROBABILITY_COLUMNS)
                    }
                )
                prediction_rows.append(row)

    metrics = compute_prediction_metrics(prediction_rows)
    metrics["classifier_epoch"] = classifier_checkpoint.get("epoch")
    metrics["label_mapping"] = LABEL_MAPPING
    metrics["class_names"] = CLASS_NAMES

    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "predictions.csv"
    plots_dir = output_dir / "plots"

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    save_predictions_csv(predictions_path, prediction_rows)
    created_plots = plot_prediction_outputs(
        predictions_path,
        plots_dir,
        metrics_json=metrics_path,
        split_name="test",
    )
    created_top_csvs = save_top_prediction_csvs(predictions_path, plots_dir)

    print(
        f"evaluation_complete: accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    print(f"wrote: {metrics_path}")
    print(f"wrote: {predictions_path}")
    for path in [*created_plots, *created_top_csvs]:
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
