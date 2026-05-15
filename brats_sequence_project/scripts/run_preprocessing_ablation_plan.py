"""Print reproducible commands for BrainIAC preprocessing ablation runs."""

from __future__ import annotations

import argparse
from pathlib import Path


VARIANTS = [
    "resize_zscore",
    "resize_none",
    "resize_percentile",
    "crop_pad_zscore",
    "physical_crop_zscore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainiac_src", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    outputs_root = project_root / "outputs" / "preprocessing_ablation"

    print("Preprocessing ablation command plan")
    print("=" * 40)
    for variant in VARIANTS:
        variant_dir = outputs_root / variant
        print(f"\n# Variant: {variant}")
        print(
            "python scripts/check_preprocessing_variant.py "
            f"--csv_path outputs/train.csv --variant {variant} --batch_size 2 --num_workers 0 "
            f"--output_dir {variant_dir / 'check'}"
        )
        for split_name in ["train", "val", "test"]:
            print(
                "python scripts/cache_brainiac_features.py "
                f"--brainiac_src {args.brainiac_src} --checkpoint_path {args.checkpoint_path} "
                f"--csv_path outputs/{split_name}.csv --split_name {split_name} --variant {variant} "
                f"--output_dir {variant_dir / 'features'} --batch_size {args.batch_size} "
                f"--num_workers {args.num_workers}"
            )
        print(
            "python scripts/train_cached_feature_classifier.py "
            f"--train_features {variant_dir / 'features' / f'features_train_{variant}.pt'} "
            f"--val_features {variant_dir / 'features' / f'features_val_{variant}.pt'} "
            f"--test_features {variant_dir / 'features' / f'features_test_{variant}.pt'} "
            f"--output_dir {variant_dir / 'classifier'} --epochs {args.epochs} "
            "--lr 1e-3 --weight_decay 1e-4 --batch_size 32 --seed 42 --patience 5"
        )
        print(
            "python scripts/analyze_feature_separability.py "
            f"--features_path {variant_dir / 'features' / f'features_train_{variant}.pt'} "
            f"--output_dir {variant_dir / 'analysis'} --method pca"
        )


if __name__ == "__main__":
    main()
