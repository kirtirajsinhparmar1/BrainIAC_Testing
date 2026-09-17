"""Shared utilities for the isolated BrainIAC full-fine-tuning experiment.

The existing project scripts intentionally remain unchanged.  This module adapts
the official BrainIAC transforms/model to the project's patient-level BraTS CSV
format and keeps all split, metric, and checkpoint invariants in one place.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from torch.utils.data import Dataset
except ImportError:  # Keep CSV/split validation usable in a lightweight environment.
    class Dataset:  # type: ignore[no-redef]
        pass

CLASS_NAMES = ("T1", "T2", "FLAIR", "T1CE")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
CLASS_INDICES = tuple(range(len(CLASS_NAMES)))
REQUIRED_CSV_COLUMNS = ("patient_id", "image_path", "label", "modality")
MODALITY_SUFFIXES = {
    "T1": ("_t1.nii", "_t1.nii.gz"),
    "T2": ("_t2.nii", "_t2.nii.gz"),
    "FLAIR": ("_flair.nii", "_flair.nii.gz"),
    "T1CE": ("_t1ce.nii", "_t1ce.nii.gz"),
}

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a config/CLI path relative to the repository root."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _ensure_src_on_path() -> None:
    src_path = str(REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    path = resolve_repo_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def _validate_row(row: Mapping[str, str], row_number: int) -> dict[str, Any]:
    patient_id = str(row.get("patient_id", "")).strip()
    image_path = str(row.get("image_path", "")).strip()
    modality = str(row.get("modality", "")).strip().upper()
    raw_label = str(row.get("label", "")).strip()

    if not patient_id or not image_path or not modality or not raw_label:
        raise ValueError(f"CSV row {row_number} has an empty required field")
    if modality not in CLASS_TO_INDEX:
        raise ValueError(f"CSV row {row_number} has unknown modality {modality!r}")
    try:
        label = int(raw_label)
    except ValueError as exc:
        raise ValueError(f"CSV row {row_number} label is not an integer: {raw_label!r}") from exc
    if label not in CLASS_INDICES:
        raise ValueError(f"CSV row {row_number} label must be one of {CLASS_INDICES}, got {label}")
    expected_label = CLASS_TO_INDEX[modality]
    if label != expected_label:
        raise ValueError(
            f"CSV row {row_number} label/modality mismatch: {label} != "
            f"{expected_label} for {modality}"
        )

    lower_path = image_path.lower()
    if "seg" in Path(image_path).name.lower():
        raise ValueError(f"Segmentation mask is not a valid input: {image_path}")
    if not any(lower_path.endswith(suffix) for suffix in MODALITY_SUFFIXES[modality]):
        raise ValueError(
            f"CSV row {row_number} path does not match {modality} NIfTI suffix: {image_path}"
        )

    normalized = dict(row)
    normalized.update(
        {
            "patient_id": patient_id,
            "image_path": image_path,
            "label": label,
            "modality": modality,
        }
    )
    return normalized


def read_csv_rows(csv_path: str | Path, *, validate_paths: bool = False) -> list[dict[str, Any]]:
    """Read and validate one split CSV without loading any image data."""

    path = resolve_repo_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        rows = [_validate_row(row, index) for index, row in enumerate(reader, start=2)]

    seen_images: set[str] = set()
    seen_patient_modalities: set[tuple[str, str]] = set()
    for row in rows:
        image_key = row["image_path"]
        patient_modality = (row["patient_id"], row["modality"])
        if image_key in seen_images:
            raise ValueError(f"Duplicate image path in {path}: {image_key}")
        if patient_modality in seen_patient_modalities:
            raise ValueError(f"Duplicate patient/modality row in {path}: {patient_modality}")
        seen_images.add(image_key)
        seen_patient_modalities.add(patient_modality)
        if validate_paths and not Path(image_key).expanduser().is_file():
            raise FileNotFoundError(f"Image path from {path} does not exist: {image_key}")

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    return rows


def _patients(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row["patient_id"]) for row in rows}


def summarize_splits(split_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Build the required patient/image/class/overlap split summary."""

    split_summary: dict[str, Any] = {}
    for split_name, rows in split_rows.items():
        by_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_patient[str(row["patient_id"])].append(row)

        class_counts = Counter(int(row["label"]) for row in rows)
        modality_counts = Counter(str(row["modality"]) for row in rows)
        incomplete = {
            patient_id: len(patient_rows)
            for patient_id, patient_rows in by_patient.items()
            if len(patient_rows) != len(CLASS_NAMES)
        }
        missing_modalities = {
            patient_id: sorted(set(CLASS_NAMES) - {str(row["modality"]) for row in patient_rows})
            for patient_id, patient_rows in by_patient.items()
            if set(str(row["modality"]) for row in patient_rows) != set(CLASS_NAMES)
        }
        split_summary[split_name] = {
            "patients": len(by_patient),
            "images": len(rows),
            "images_per_class": {
                INDEX_TO_CLASS[index]: class_counts.get(index, 0) for index in CLASS_INDICES
            },
            "images_per_modality": {
                modality: modality_counts.get(modality, 0) for modality in CLASS_NAMES
            },
            "incomplete_patients": incomplete,
            "patients_missing_modalities": missing_modalities,
            "segmentation_input_rows": sum(
                "seg" in Path(str(row["image_path"])).name.lower() for row in rows
            ),
        }

    overlap_checks: dict[str, Any] = {}
    split_names = tuple(split_rows)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = sorted(_patients(split_rows[left_name]) & _patients(split_rows[right_name]))
            overlap_checks[f"{left_name}_vs_{right_name}"] = {
                "overlap_count": len(overlap),
                "overlapping_patients": overlap,
            }

    summary = {
        "class_mapping": dict(CLASS_TO_INDEX),
        "splits": split_summary,
        "overlap_checks": overlap_checks,
        "zero_patient_overlap": all(
            check["overlap_count"] == 0 for check in overlap_checks.values()
        ),
        "all_patients_have_four_modalities": all(
            not split_data["incomplete_patients"]
            and not split_data["patients_missing_modalities"]
            for split_data in split_summary.values()
        ),
        "no_segmentation_inputs": all(
            split_data["segmentation_input_rows"] == 0 for split_data in split_summary.values()
        ),
    }
    return summary


def validate_all_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    *,
    validate_paths: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    split_rows = {
        "train": read_csv_rows(train_csv, validate_paths=validate_paths),
        "validation": read_csv_rows(val_csv, validate_paths=validate_paths),
        "test": read_csv_rows(test_csv, validate_paths=validate_paths),
    }
    summary = summarize_splits(split_rows)
    if not summary["zero_patient_overlap"]:
        raise ValueError(f"Patient overlap detected: {summary['overlap_checks']}")
    if not summary["all_patients_have_four_modalities"]:
        raise ValueError("Every patient must have exactly T1, T2, FLAIR, and T1CE")
    if not summary["no_segmentation_inputs"]:
        raise ValueError("Segmentation masks were found in an input split")
    return split_rows, summary


def select_patient_subset(
    rows: Sequence[Mapping[str, Any]],
    fraction: float,
    seed: int,
    *,
    max_patients: int | None = None,
) -> tuple[list[int], list[str]]:
    """Select complete patient groups deterministically, preserving CSV order."""

    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    patient_ids = sorted({str(row["patient_id"]) for row in rows})
    if not patient_ids:
        raise ValueError("Cannot select a subset from an empty split")
    random.Random(seed).shuffle(patient_ids)
    patient_count = len(patient_ids) if fraction == 1 else max(1, int(len(patient_ids) * fraction))
    if max_patients is not None:
        if max_patients < 1:
            raise ValueError(f"max_patients must be positive, got {max_patients}")
        patient_count = min(patient_count, max_patients)
    selected_patients = set(patient_ids[:patient_count])
    selected_indices = [
        index for index, row in enumerate(rows) if str(row["patient_id"]) in selected_patients
    ]
    if len(selected_indices) != patient_count * len(CLASS_NAMES):
        raise ValueError("Patient subset is not composed of complete four-sequence groups")
    return selected_indices, sorted(selected_patients)


def get_brainiac_train_transform():
    """Use the released downstream training transform verbatim."""

    _ensure_src_on_path()
    from dataset import get_default_transform

    return get_default_transform()


def get_brainiac_eval_transform():
    """Use the released downstream validation/test transform verbatim."""

    _ensure_src_on_path()
    from dataset import get_validation_transform

    return get_validation_transform()


def build_brainiac_model(checkpoint_path: str | Path, *, num_classes: int = 4):
    """Construct the released ViT-B model with a fresh task head."""

    import torch

    checkpoint = resolve_repo_path(checkpoint_path)
    validate_general_checkpoint_path(checkpoint, require_exists=True)
    _ensure_src_on_path()
    from model import Classifier, SingleScanModel, ViTBackboneNet

    backbone = ViTBackboneNet(simclr_ckpt_path=str(checkpoint))
    classifier = Classifier(d_model=768, num_classes=num_classes)
    model = SingleScanModel(backbone, classifier)
    # Keep this import local and make the intent explicit without ever freezing
    # or silently unfreezing a parameter.
    if not isinstance(model, torch.nn.Module):
        raise TypeError("BrainIAC model construction did not return a torch.nn.Module")
    return model


def validate_general_checkpoint_path(checkpoint_path: str | Path, *, require_exists: bool) -> Path:
    checkpoint = resolve_repo_path(checkpoint_path)
    if checkpoint.name != "BrainIAC.ckpt":
        raise ValueError(
            "This experiment must initialize from the general checkpoint named "
            f"BrainIAC.ckpt, not {checkpoint.name!r}"
        )
    lowered_path = str(checkpoint).lower()
    if "sequence_classification" in lowered_path or "downstream" in lowered_path:
        raise ValueError(f"Downstream checkpoint is forbidden: {checkpoint}")
    if require_exists and not checkpoint.is_file():
        raise FileNotFoundError(
            f"General BrainIAC checkpoint not found: {checkpoint}. "
            "Place it at checkpoints/BrainIAC.ckpt on the lab server."
        )
    return checkpoint


def parameter_report(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable}


def assert_full_finetuning(model: Any) -> dict[str, int]:
    report = parameter_report(model)
    backbone_frozen = [
        name for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and not parameter.requires_grad
    ]
    all_frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if backbone_frozen:
        raise RuntimeError(
            "ABORT: BrainIAC backbone parameters are frozen: "
            + ", ".join(backbone_frozen[:5])
        )
    if all_frozen:
        raise RuntimeError(
            "ABORT: full fine-tuning requires every encoder/head parameter to be trainable; "
            f"frozen parameters include {all_frozen[:5]}"
        )
    classifier = getattr(model, "classifier", None)
    head = getattr(classifier, "fc", None)
    if getattr(head, "out_features", None) != len(CLASS_NAMES):
        raise RuntimeError("ABORT: classifier output dimension is not 4")
    return report


def validate_finetune_config(config: Mapping[str, Any]) -> None:
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    if model_config.get("num_classes") != len(CLASS_NAMES):
        raise ValueError("Configuration must define a four-class classifier")
    if model_config.get("hidden_size") != 768:
        raise ValueError("Configuration must use the released BrainIAC SimCLR ViT-B hidden size 768")
    if model_config.get("image_size") != [96, 96, 96]:
        raise ValueError("Configuration must use the released 96 x 96 x 96 input size")
    if float(model_config.get("dropout", -1)) != 0.2:
        raise ValueError("Configuration must retain the released classifier dropout 0.2")
    if model_config.get("freeze_backbone") is not False:
        raise ValueError("Configuration must set model.freeze_backbone: false")
    checkpoint_path = model_config.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("Configuration is missing model.checkpoint_path")
    validate_general_checkpoint_path(checkpoint_path, require_exists=False)
    if str(training_config.get("optimizer", "")).lower() != "adam":
        raise ValueError("The reproduction optimizer must be Adam")
    if training_config.get("epochs") != 100:
        raise ValueError("Production configuration must use the published 100 epochs")
    if data_config.get("batch_size") != 16:
        raise ValueError("Production configuration must use the published batch size 16")
    if data_config.get("validation_batch_size") != 1:
        raise ValueError("Validation must retain the released downstream loader batch size 1")
    if data_config.get("num_workers") != 4:
        raise ValueError("Training must retain the released downstream loader worker count 4")
    if data_config.get("validation_num_workers") != 1:
        raise ValueError("Validation must retain the released downstream loader worker count 1")
    if data_config.get("pin_memory") is not False:
        raise ValueError("DataLoader pin_memory must remain disabled as in the released script")
    if float(training_config.get("learning_rate", -1)) != 0.0001:
        raise ValueError("Production configuration must use BrainIAC learning rate 0.0001")
    if float(training_config.get("weight_decay", -1)) != 0.0001:
        raise ValueError("Production configuration must retain released weight decay 0.0001")
    if str(training_config.get("scheduler", "")).lower() != "reducelronplateau":
        raise ValueError("The reproduction scheduler must be ReduceLROnPlateau")
    if training_config.get("scheduler_mode") != "max":
        raise ValueError("ReduceLROnPlateau must maximize validation balanced accuracy")
    if float(training_config.get("scheduler_factor", -1)) != 0.1:
        raise ValueError("The configured scheduler factor must remain the PyTorch default 0.1")
    if int(training_config.get("scheduler_patience", -1)) != 10:
        raise ValueError("The configured scheduler patience must remain the PyTorch default 10")
    if training_config.get("checkpoint_metric") != "val_balanced_accuracy":
        raise ValueError("Checkpoint selection must use validation balanced accuracy")
    if training_config.get("checkpoint_mode") != "max":
        raise ValueError("Checkpoint selection must maximize validation balanced accuracy")
    if training_config.get("precision") != "16-mixed":
        raise ValueError("The configured precision must remain the released-code default 16-mixed")
    if data_config.get("preprocessing") != "official_runtime_resize_zscore":
        raise ValueError("Configuration must use the official runtime resize/normalization pipeline")
    for key in ("train_csv", "val_csv", "test_csv"):
        if not data_config.get(key):
            raise ValueError(f"Configuration is missing data.{key}")


class BraTSFineTuneDataset(Dataset):
    """CSV-backed dataset using official BrainIAC transforms and no mask input."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        transform: Any,
        *,
        validate_paths: bool = True,
    ) -> None:
        if not isinstance(rows, Sequence) or not rows:
            raise ValueError("BraTSFineTuneDataset requires non-empty validated rows")
        self.rows = [dict(row) for row in rows]
        self.transform = transform
        if validate_paths:
            missing = [
                str(row["image_path"])
                for row in self.rows
                if not Path(str(row["image_path"])).expanduser().is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} image paths do not exist; first missing path: {missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        transformed = self.transform({"image": row["image_path"]})
        image = transformed["image"]
        return (
            image,
            torch.tensor(int(row["label"]), dtype=torch.long),
            row["patient_id"],
            row["modality"],
            row["image_path"],
        )


def load_finetuned_state_dict(model: Any, checkpoint_path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    path = resolve_repo_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuned checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported checkpoint payload in {path}")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Checkpoint has no usable state_dict: {path}")

    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        while normalized_key.startswith("module.") or normalized_key.startswith("model."):
            normalized_key = normalized_key.split(".", 1)[1]
        normalized[normalized_key] = value
    model.load_state_dict(normalized, strict=True)
    return dict(payload)


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(CLASS_INDICES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        INDEX_TO_CLASS[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    }
    return {
        "primary_metric": "balanced_accuracy",
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": per_class,
        "class_mapping": dict(CLASS_TO_INDEX),
    }


def write_json(path: str | Path, value: Any) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
