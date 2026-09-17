#!/usr/bin/env python3
"""Evaluate one selected BrainIAC fine-tuned checkpoint on the test CSV once."""

from __future__ import annotations

import argparse
import csv
import json

from finetune_common import (
    CLASS_NAMES,
    REPO_ROOT,
    BraTSFineTuneDataset,
    build_brainiac_model,
    calculate_metrics,
    get_brainiac_eval_transform,
    load_finetuned_state_dict,
    load_yaml_config,
    read_csv_rows,
    resolve_repo_path,
    validate_finetune_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/config/full_finetune.yml"),
    )
    parser.add_argument("--checkpoint", required=True, help="Selected best_model.ckpt from fine-tuning")
    parser.add_argument("--test-csv", help="Override config data.test_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--check-only", action="store_true", help="Validate config/metric contract without evaluation")
    return parser.parse_args()


def evaluate_once(model, loader: DataLoader, device: torch.device) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for images, labels, patient_ids, modalities, image_paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            predictions = logits.argmax(dim=1).cpu().tolist()
            true_labels = labels.cpu().tolist()
            for index, (true_label, predicted_label) in enumerate(zip(true_labels, predictions)):
                y_true.append(int(true_label))
                y_pred.append(int(predicted_label))
                prediction_rows.append(
                    {
                        "patient_id": patient_ids[index],
                        "image_path": image_paths[index],
                        "true_label": int(true_label),
                        "predicted_label": int(predicted_label),
                        "true_label_name": CLASS_NAMES[int(true_label)],
                        "predicted_label_name": CLASS_NAMES[int(predicted_label)],
                        **{
                            f"probability_{CLASS_NAMES[class_index]}": float(
                                probabilities[index][class_index]
                            )
                            for class_index in range(len(CLASS_NAMES))
                        },
                    }
                )
    if not prediction_rows:
        raise RuntimeError("Test loader produced no batches")
    return calculate_metrics(y_true, y_pred), prediction_rows


def write_predictions(path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "patient_id",
        "image_path",
        "true_label",
        "predicted_label",
        "true_label_name",
        "predicted_label_name",
        *(f"probability_{class_name}" for class_name in CLASS_NAMES),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row])


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_finetune_config(config)
    if args.check_only:
        print("Configuration is valid and balanced_accuracy is the required primary metric.")
        return
    global torch, DataLoader
    import torch
    from torch.utils.data import DataLoader

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")

    device = torch.device(args.device)
    test_csv = args.test_csv or config["data"]["test_csv"]
    test_rows = read_csv_rows(test_csv, validate_paths=True)
    dataset = BraTSFineTuneDataset(
        test_rows,
        get_brainiac_eval_transform(),
        validate_paths=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["data"]["validation_batch_size"]),
        shuffle=False,
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(config["data"]["validation_num_workers"])
        ),
        pin_memory=args.device == "cuda" and bool(config["data"].get("pin_memory", False)),
    )

    model = build_brainiac_model(config["model"]["checkpoint_path"], num_classes=4)
    load_finetuned_state_dict(model, args.checkpoint, map_location="cpu")
    model.to(device)
    metrics, predictions = evaluate_once(model, loader, device)

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "metrics.json",
        {
            **metrics,
            "test_csv": str(resolve_repo_path(test_csv)),
            "fine_tuned_checkpoint": str(resolve_repo_path(args.checkpoint)),
            "evaluation_passes": 1,
            "checkpoint_selected_without_test": True,
        },
    )
    write_predictions(output_dir / "predictions.csv", predictions)
    write_confusion_matrix(output_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    print(f"balanced_accuracy (primary)={metrics['balanced_accuracy']:.6f}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(json.dumps(metrics["per_class"], indent=2, sort_keys=True))
    print(f"evaluation_passes=1")
    print(f"predictions={output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
