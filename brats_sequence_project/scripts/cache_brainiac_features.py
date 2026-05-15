"""Cache frozen BrainIAC features for one preprocessing variant and CSV split."""

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

from datasets import BraTSSequenceDataset, SUPPORTED_PREPROCESSING_VARIANTS


LABEL_MAPPING = {"T1": 0, "T2": 1, "FLAIR": 2, "T1CE": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainiac_src", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--split_name", choices=["train", "val", "test"], required=True)
    parser.add_argument("--variant", choices=SUPPORTED_PREPROCESSING_VARIANTS, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def optional_tqdm(iterable: Any, desc: str) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable


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
            "Install BrainIAC runtime dependencies before feature caching."
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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")
    if args.use_amp and not use_amp:
        print("amp_status: --use_amp requested but CUDA is unavailable; running without AMP.")
    print(f"device: {device}")

    dataset = BraTSSequenceDataset(args.csv_path, preprocessing_variant=args.variant)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    backbone = load_frozen_backbone(args.brainiac_src, args.checkpoint_path, device)

    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    patient_ids: list[str] = []
    modalities: list[str] = []
    image_paths: list[str] = []
    split_names: list[str] = []

    with torch.no_grad():
        for batch in optional_tqdm(loader, desc=f"cache {args.split_name} {args.variant}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu()
            with torch.cuda.amp.autocast(enabled=use_amp):
                features = backbone(images)

            if tuple(features.shape[1:]) != (768,):
                raise AssertionError(f"Expected feature shape [B,768], got {tuple(features.shape)}")

            features_list.append(features.detach().cpu().float())
            labels_list.append(labels)
            patient_ids.extend(batch["patient_id"])
            modalities.extend(batch["modality"])
            image_paths.extend(batch["image_path"])
            split_names.extend([args.split_name] * images.shape[0])

    features_tensor = torch.cat(features_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0).long()
    output_path = output_dir / f"features_{args.split_name}_{args.variant}.pt"
    torch.save(
        {
            "features": features_tensor,
            "labels": labels_tensor,
            "patient_ids": patient_ids,
            "modalities": modalities,
            "image_paths": image_paths,
            "split_names": split_names,
            "variant": args.variant,
            "split_name": args.split_name,
            "label_mapping": LABEL_MAPPING,
            "source_csv": str(args.csv_path.expanduser().resolve()),
        },
        output_path,
    )

    summary = {
        "output_path": str(output_path),
        "variant": args.variant,
        "split_name": args.split_name,
        "num_samples": int(features_tensor.shape[0]),
        "feature_shape": list(features_tensor.shape),
        "label_shape": list(labels_tensor.shape),
        "feature_dtype": str(features_tensor.dtype),
        "label_mapping": LABEL_MAPPING,
    }
    summary_path = output_dir / f"features_{args.split_name}_{args.variant}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"variant: {args.variant}")
    print(f"split_name: {args.split_name}")
    print(f"feature_shape: {tuple(features_tensor.shape)}")
    print(f"wrote: {output_path}")
    print(f"wrote: {summary_path}")
    print("feature_cache: passed")


if __name__ == "__main__":
    main()
