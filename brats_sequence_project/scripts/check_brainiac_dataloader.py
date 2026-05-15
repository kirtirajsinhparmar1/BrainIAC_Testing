"""Run one BraTSSequenceDataset/DataLoader sanity check batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import BraTSSequenceDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = BraTSSequenceDataset(args.csv_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    batch = next(iter(dataloader))
    images = batch["image"]
    labels = batch["label"]

    print(f"batch['image'].shape: {tuple(images.shape)}")
    print(f"batch['label'].shape: {tuple(labels.shape)}")
    print(f"labels: {labels.tolist()}")
    print(f"patient_ids: {list(batch['patient_id'])}")
    print(f"modalities: {list(batch['modality'])}")
    print(f"image dtype: {images.dtype}")
    print(f"image min/max/mean: {images.min().item():.6f} / {images.max().item():.6f} / {images.mean().item():.6f}")

    expected_shape = (images.shape[0], 1, 96, 96, 96)
    assert tuple(images.shape) == expected_shape, f"Expected image shape {expected_shape}, got {tuple(images.shape)}"
    assert labels.dtype == torch.long, f"Expected labels dtype torch.long, got {labels.dtype}"
    assert set(labels.tolist()).issubset({0, 1, 2, 3}), f"Labels out of range: {labels.tolist()}"

    print("dataloader_check: passed")


if __name__ == "__main__":
    main()
