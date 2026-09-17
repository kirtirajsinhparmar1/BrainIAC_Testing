#!/usr/bin/env python3
"""Full end-to-end BrainIAC fine-tuning on the BraTS2020 sequence task.

The test CSV is deliberately not loaded by this script.  Checkpoint selection
uses validation balanced accuracy only; use evaluate_checkpoint.py once the
selected checkpoint is fixed.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from finetune_common import (
    CLASS_NAMES,
    REPO_ROOT,
    BraTSFineTuneDataset,
    assert_full_finetuning,
    build_brainiac_model,
    calculate_metrics,
    get_brainiac_eval_transform,
    get_brainiac_train_transform,
    load_yaml_config,
    read_csv_rows,
    resolve_repo_path,
    select_patient_subset,
    validate_general_checkpoint_path,
    validate_finetune_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/config/full_finetune.yml"),
    )
    parser.add_argument("--checkpoint", help="Override config model.checkpoint_path")
    parser.add_argument("--fraction", type=float, help="Patient fraction of the train split")
    parser.add_argument("--seed", type=int, help="Override experiment seed")
    parser.add_argument("--output-dir", help="Override the generated run directory")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Engineering-only run: two train patients, two validation patients, one epoch",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config and, when available, construct/check the model without training",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(seed: int):
    def init_worker(worker_id: int) -> None:
        import random

        worker_seed = seed + worker_id
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed)
        except ImportError:
            pass

    return init_worker


def _loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    *,
    amp_enabled: bool,
) -> tuple[float, dict[str, object]]:
    model.train()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels, *_ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if not losses:
        raise RuntimeError("Training loader produced no batches")
    return sum(losses) / len(losses), calculate_metrics(y_true, y_pred)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> tuple[float, dict[str, object]]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels, *_ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if not losses:
        raise RuntimeError("Validation loader produced no batches")
    return sum(losses) / len(losses), calculate_metrics(y_true, y_pred)


def _run_directory(config: dict, args: argparse.Namespace, fraction: float) -> Path:
    if args.output_dir:
        return resolve_repo_path(args.output_dir)
    root = resolve_repo_path(config["output"]["root_dir"])
    if args.smoke_test:
        return root / "smoke_test"
    return root / f"fraction_{int(round(fraction * 100)):03d}"


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_finetune_config(config)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    experiment_config = config["experiment"]

    checkpoint_path = args.checkpoint or model_config["checkpoint_path"]
    validate_general_checkpoint_path(checkpoint_path, require_exists=False)
    fraction = float(args.fraction if args.fraction is not None else experiment_config["fraction"])
    seed = int(args.seed if args.seed is not None else experiment_config["seed"])
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    if args.check_only:
        print("Configuration is valid.")
        print(f"class_mapping={dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))}")
        print("freeze_backbone=False")
        checkpoint = resolve_repo_path(checkpoint_path)
        if checkpoint.is_file():
            model = build_brainiac_model(checkpoint, num_classes=model_config["num_classes"])
            report = assert_full_finetuning(model)
            print(f"general_checkpoint={checkpoint}")
            print(f"classifier_output_dimension={model.classifier.fc.out_features}")
            print(f"total_parameters={report['total_parameters']}")
            print(f"trainable_parameters={report['trainable_parameters']}")
            print(f"frozen_parameters={report['frozen_parameters']}")
        else:
            print(f"model_check_skipped_missing_checkpoint={checkpoint}")
        return

    global torch, nn, DataLoader, Subset
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training is intentionally CUDA-only. No training was started; run this command on the lab server."
        )

    torch.set_float32_matmul_precision("medium")
    seed_everything(seed)
    train_rows = read_csv_rows(data_config["train_csv"], validate_paths=True)
    val_rows = read_csv_rows(data_config["val_csv"], validate_paths=True)
    train_patients = {str(row["patient_id"]) for row in train_rows}
    val_patients = {str(row["patient_id"]) for row in val_rows}
    overlap = sorted(train_patients & val_patients)
    if overlap:
        raise ValueError(f"Train/validation patient overlap detected: {overlap}")
    print("test_csv_not_loaded_for_training=True")

    smoke_config = experiment_config.get("smoke_test", {})
    max_train_patients = int(smoke_config["train_patients"]) if args.smoke_test else None
    max_val_patients = int(smoke_config["validation_patients"]) if args.smoke_test else None
    train_indices, selected_patients = select_patient_subset(
        train_rows,
        fraction,
        seed,
        max_patients=max_train_patients,
    )
    val_indices, selected_val_patients = select_patient_subset(
        val_rows,
        1.0,
        seed + 1,
        max_patients=max_val_patients,
    )
    print(f"selected_train_patients={len(selected_patients)}")
    print(f"selected_train_images={len(train_indices)}")
    print(f"selected_validation_patients={len(selected_val_patients)}")
    print(f"selected_validation_images={len(val_indices)}")
    if args.smoke_test:
        print("WARNING: smoke-test output is engineering validation only, not a scientific result.")

    train_dataset = BraTSFineTuneDataset(
        train_rows,
        get_brainiac_train_transform(),
        validate_paths=False,
    )
    val_dataset = BraTSFineTuneDataset(
        val_rows,
        get_brainiac_eval_transform(),
        validate_paths=False,
    )
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    num_workers = int(data_config["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", False)),
        worker_init_fn=_worker_init_fn(seed),
        generator=_loader_generator(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_config["validation_batch_size"]),
        shuffle=False,
        num_workers=int(data_config.get("validation_num_workers", 1)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        worker_init_fn=_worker_init_fn(seed + 1),
        generator=_loader_generator(seed + 1),
    )

    model = build_brainiac_model(checkpoint_path, num_classes=model_config["num_classes"])
    report = assert_full_finetuning(model)
    print(f"general_checkpoint={resolve_repo_path(checkpoint_path)}")
    print(f"classifier_output_dimension={model.classifier.fc.out_features}")
    print(f"total_parameters={report['total_parameters']}")
    print(f"trainable_parameters={report['trainable_parameters']}")
    print(f"frozen_parameters={report['frozen_parameters']}")

    device = torch.device("cuda")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(training_config["scheduler_mode"]),
        factor=float(training_config["scheduler_factor"]),
        patience=int(training_config["scheduler_patience"]),
    )
    amp_enabled = training_config["precision"] == "16-mixed"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    epochs = int(smoke_config["epochs"]) if args.smoke_test else int(training_config["epochs"])
    output_dir = _run_directory(config, args, fraction)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_metric = -math.inf
    best_checkpoint = output_dir / "best_model.ckpt"

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, val_metrics = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        val_balanced_accuracy = float(val_metrics["balanced_accuracy"])
        scheduler.step(val_balanced_accuracy)
        epoch_record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch}/{epochs} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_balanced_accuracy={val_balanced_accuracy:.6f} "
            f"val_accuracy={float(val_metrics['accuracy']):.6f}"
        )
        if val_balanced_accuracy > best_metric:
            best_metric = val_balanced_accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "best_validation_balanced_accuracy": best_metric,
                    "class_mapping": dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)))),
                    "general_checkpoint": str(resolve_repo_path(checkpoint_path)),
                    "fraction": fraction,
                    "seed": seed,
                    "full_finetuning": True,
                    "test_used_for_selection": False,
                },
                best_checkpoint,
            )

    write_json(output_dir / "training_history.json", history)
    write_json(
        output_dir / "run_metadata.json",
        {
            "config_path": str(resolve_repo_path(args.config)),
            "checkpoint_path": str(resolve_repo_path(checkpoint_path)),
            "fraction": fraction,
            "seed": seed,
            "smoke_test": args.smoke_test,
            "selected_train_patients": selected_patients,
            "selected_validation_patients": selected_val_patients,
            "parameter_report": report,
            "class_mapping": dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)))),
            "checkpoint_selection_metric": "validation balanced accuracy",
            "test_used_for_selection": False,
        },
    )
    print(f"best_checkpoint={best_checkpoint}")
    print(f"best_validation_balanced_accuracy={best_metric:.6f}")
    print("Test data was not loaded or used for checkpoint selection.")


if __name__ == "__main__":
    main()
