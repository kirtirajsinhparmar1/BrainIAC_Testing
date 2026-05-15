"""Run one dataloader batch through a selected preprocessing variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import BraTSSequenceDataset, SUPPORTED_PREPROCESSING_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--variant", choices=SUPPORTED_PREPROCESSING_VARIANTS, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BraTSSequenceDataset(args.csv_path, preprocessing_variant=args.variant)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]
    summary = {
        "variant": args.variant,
        "csv_path": str(args.csv_path.expanduser().resolve()),
        "batch_size": int(images.shape[0]),
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "image_min": float(images.min().item()),
        "image_max": float(images.max().item()),
        "image_mean": float(images.mean().item()),
        "image_std": float(images.std().item()),
        "labels": [int(value) for value in labels.tolist()],
        "patient_ids": list(batch["patient_id"]),
        "modalities": list(batch["modality"]),
        "image_paths": list(batch["image_path"]),
    }

    if images.ndim != 5 or tuple(images.shape[1:]) != (1, 96, 96, 96):
        raise AssertionError(f"Expected image batch [B,1,96,96,96], got {tuple(images.shape)}")

    output_path = output_dir / "preprocessing_variant_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"variant: {args.variant}")
    print(f"image_shape: {tuple(images.shape)}")
    print(f"label_shape: {tuple(labels.shape)}")
    print(f"image_dtype: {images.dtype}")
    print(
        "image_stats: "
        f"min={summary['image_min']:.6f} max={summary['image_max']:.6f} "
        f"mean={summary['image_mean']:.6f} std={summary['image_std']:.6f}"
    )
    print(f"patient_ids: {summary['patient_ids']}")
    print(f"modalities: {summary['modalities']}")
    print(f"wrote: {output_path}")
    print("preprocessing_variant_check: passed")


if __name__ == "__main__":
    main()
