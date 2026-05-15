"""Run one frozen BrainIAC backbone forward pass on BraTS sequence data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRAINIAC_ROOT = Path(__file__).resolve().parents[2]
BRAINIAC_SRC = BRAINIAC_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BRAINIAC_SRC))

try:
    from datasets import BraTSSequenceDataset  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment-dependent.
    raise ImportError(
        "Could not import BraTSSequenceDataset. If the error mentions MONAI, install "
        "BrainIAC runtime dependencies first, including monai==1.3.2 from "
        "BrainIAC/requirements.txt."
    ) from exc

try:
    from model import ViTBackboneNet  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment-dependent.
    raise ImportError(
        f"Could not import ViTBackboneNet from {BRAINIAC_SRC / 'model.py'}. "
        "Confirm this script is still inside BrainIAC/brats_sequence_project and that "
        "BrainIAC runtime dependencies are installed. If the error mentions MONAI, "
        "install monai==1.3.2 from BrainIAC/requirements.txt."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def load_brainiac_backbone(checkpoint_path: Path) -> ViTBackboneNet:
    """Load BrainIAC ViT backbone and add context for checkpoint format failures."""

    try:
        return ViTBackboneNet(str(checkpoint_path))
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load the checkpoint into ViTBackboneNet. The BrainIAC backbone "
            "loader expects a checkpoint whose state_dict contains keys prefixed with "
            "'backbone.' and whose tensor shapes match the MONAI ViT configuration in "
            "BrainIAC/src/model.py: in_channels=1, img_size=(96,96,96), "
            "patch_size=(16,16,16), hidden_size=768, num_layers=12, num_heads=12. "
            f"Checkpoint path: {checkpoint_path}"
        ) from exc


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"BrainIAC checkpoint not found: {checkpoint_path}\n"
            "Download/place the pretrained BrainIAC checkpoint locally and rerun with "
            "--checkpoint_path pointing to that exact file. Example local path: "
            "/Users/kp/Documents/BRAINIAC/BrainIAC/checkpoints/BrainIAC.ckpt. "
            "Example Colab path: /content/checkpoints/BrainIAC.ckpt."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"cuda_status: available; using {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(
            "cuda_status: unavailable; using CPU for this one-pass sanity check. "
            "Use a CUDA Colab runtime for real training."
        )

    model = load_brainiac_backbone(checkpoint_path)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    dataset = BraTSSequenceDataset(args.csv_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    with torch.no_grad():
        features = model(images)

    print(f"device: {device}")
    print(f"input batch shape: {tuple(images.shape)}")
    print(f"feature shape: {tuple(features.shape)}")
    print(f"feature dtype: {features.dtype}")
    print(
        "feature min/max/mean: "
        f"{features.min().item():.6f} / {features.max().item():.6f} / {features.mean().item():.6f}"
    )
    print(f"trainable backbone params after freezing: {trainable_params}")

    expected_input_shape = (images.shape[0], 1, 96, 96, 96)
    expected_feature_shape = (images.shape[0], 768)
    assert tuple(images.shape) == expected_input_shape, (
        f"Expected input shape {expected_input_shape}, got {tuple(images.shape)}"
    )
    assert tuple(features.shape) == expected_feature_shape, (
        f"Expected feature shape {expected_feature_shape}, got {tuple(features.shape)}"
    )

    print("backbone_check: passed; BrainIAC features are [B, 768]")


if __name__ == "__main__":
    main()
