"""Print Colab command plans for offline BrainIAC/BraTS preprocessing ablations."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=Path("/content/BrainIAC_Testing"))
    parser.add_argument("--project_root", type=Path, default=Path("/content/BrainIAC_Testing/brats_sequence_project"))
    parser.add_argument("--dataset_root", type=Path, default=Path("/content/data"))
    parser.add_argument("--checkpoint_path", type=Path, default=Path("/content/checkpoints/BrainIAC.ckpt"))
    parser.add_argument(
        "--template_path",
        type=Path,
        default=Path("/content/BrainIAC_Testing/src/preprocessing/<TEMPLATE_FILE_HERE>"),
    )
    parser.add_argument("--subset_patients", type=int, default=10)
    parser.add_argument("--visualize_examples", type=int, default=4)
    parser.add_argument("--variant", type=str, default="crop_pad_zscore")
    parser.add_argument("--cache_batch_size", type=int, default=2)
    parser.add_argument("--cache_num_workers", type=int, default=2)
    return parser.parse_args()


def block(title: str, body: str) -> str:
    return f"\n# {title}\n{textwrap.dedent(body).strip()}\n"


def cache_commands(
    repo_root: Path,
    checkpoint_path: Path,
    csv_dir: Path,
    feature_dir: Path,
    csv_suffix: str,
    variant: str,
    batch_size: int,
    num_workers: int,
) -> str:
    lines = []
    for split_name in ["train", "val", "test"]:
        lines.append(
            f"""python scripts/cache_brainiac_features.py \\
  --brainiac_src {repo_root / 'src'} \\
  --checkpoint_path {checkpoint_path} \\
  --csv_path {csv_dir / f'{split_name}_{csv_suffix}.csv'} \\
  --split_name {split_name} \\
  --variant {variant} \\
  --output_dir {feature_dir} \\
  --batch_size {batch_size} \\
  --num_workers {num_workers} \\
  --use_amp"""
        )
    return "\n\n".join(lines)


def print_n4_commands(args: argparse.Namespace, inputs_root: Path) -> None:
    n4_root = inputs_root / "n4_only"
    print(
        block(
            "A. N4-only subset run",
            f"""
            cd {args.project_root}

            python scripts/preprocess_brats_n4_only.py \\
              --dataset_root {args.dataset_root} \\
              --output_root {n4_root / 'images'} \\
              --splits_output_dir {n4_root / 'csvs'} \\
              --max_patients {args.subset_patients} \\
              --visualize_examples {args.visualize_examples}
            """,
        )
    )
    print(
        block(
            "B. N4-only full run",
            f"""
            cd {args.project_root}

            python scripts/preprocess_brats_n4_only.py \\
              --dataset_root {args.dataset_root} \\
              --output_root {n4_root / 'images'} \\
              --splits_output_dir {n4_root / 'csvs'} \\
              --visualize_examples 8
            """,
        )
    )
    print(
        block(
            "Audit N4 output CSVs",
            f"""
            python scripts/check_processed_dataset.py \\
              --csv_path {n4_root / 'csvs' / 'train_n4_only.csv'} \\
              --output_dir {n4_root / 'audit_train'}

            python scripts/check_processed_dataset.py \\
              --csv_path {n4_root / 'csvs' / 'test_n4_only.csv'} \\
              --output_dir {n4_root / 'audit_test'}
            """,
        )
    )


def print_brainiac_style_commands(args: argparse.Namespace, inputs_root: Path) -> None:
    style_root = inputs_root / "brainiac_style"
    print(
        block(
            "C. BrainIAC-style subset run without HD-BET",
            f"""
            cd {args.project_root}

            python scripts/preprocess_brats_brainiac_style.py \\
              --dataset_root {args.dataset_root} \\
              --output_root {style_root / 'images'} \\
              --splits_output_dir {style_root / 'csvs'} \\
              --template_path {args.template_path} \\
              --max_patients {args.subset_patients} \\
              --run_n4 \\
              --run_registration \\
              --visualize_examples {args.visualize_examples}
            """,
        )
    )
    print(
        block(
            "D. BrainIAC-style full run without HD-BET",
            f"""
            cd {args.project_root}

            python scripts/preprocess_brats_brainiac_style.py \\
              --dataset_root {args.dataset_root} \\
              --output_root {style_root / 'images'} \\
              --splits_output_dir {style_root / 'csvs'} \\
              --template_path {args.template_path} \\
              --run_n4 \\
              --run_registration \\
              --visualize_examples 8
            """,
        )
    )
    print(
        block(
            "Experimental BrainIAC-style run with HD-BET",
            f"""
            # WARNING: BraTS is already skull-stripped. HD-BET may remove valid brain/tumor regions.
            python scripts/preprocess_brats_brainiac_style.py \\
              --dataset_root {args.dataset_root} \\
              --output_root {style_root / 'images_hdbet_experimental'} \\
              --splits_output_dir {style_root / 'csvs_hdbet_experimental'} \\
              --template_path {args.template_path} \\
              --max_patients {args.subset_patients} \\
              --run_n4 \\
              --run_registration \\
              --run_hdbet \\
              --run_hdbet_on_already_skullstripped \\
              --visualize_examples {args.visualize_examples}
            """,
        )
    )
    print(
        block(
            "Audit BrainIAC-style output CSVs",
            f"""
            python scripts/check_processed_dataset.py \\
              --csv_path {style_root / 'csvs' / 'train_brainiac_style.csv'} \\
              --output_dir {style_root / 'audit_train'}

            python scripts/check_processed_dataset.py \\
              --csv_path {style_root / 'csvs' / 'test_brainiac_style.csv'} \\
              --output_dir {style_root / 'audit_test'}
            """,
        )
    )


def print_followup_commands(args: argparse.Namespace, inputs_root: Path) -> None:
    n4_root = inputs_root / "n4_only"
    style_root = inputs_root / "brainiac_style"
    n4_cache = cache_commands(
        args.repo_root,
        args.checkpoint_path,
        n4_root / "csvs",
        n4_root / "features",
        "n4_only",
        args.variant,
        args.cache_batch_size,
        args.cache_num_workers,
    )
    style_cache = cache_commands(
        args.repo_root,
        args.checkpoint_path,
        style_root / "csvs",
        style_root / "features",
        "brainiac_style",
        args.variant,
        args.cache_batch_size,
        args.cache_num_workers,
    )
    print(
        block(
            "E. Cache BrainIAC frozen features from processed CSVs",
            f"{n4_cache}\n\n{style_cache}",
        )
    )
    print(
        block(
            "F. Train 4-class classifiers from processed features",
            f"""
            python scripts/train_cached_feature_classifier.py \\
              --train_features {n4_root / 'features' / f'features_train_{args.variant}.pt'} \\
              --val_features {n4_root / 'features' / f'features_val_{args.variant}.pt'} \\
              --test_features {n4_root / 'features' / f'features_test_{args.variant}.pt'} \\
              --output_dir {n4_root / 'classifier'} \\
              --epochs 30 \\
              --lr 1e-3 \\
              --weight_decay 1e-4 \\
              --batch_size 32 \\
              --seed 42 \\
              --patience 5

            python scripts/train_cached_feature_classifier.py \\
              --train_features {style_root / 'features' / f'features_train_{args.variant}.pt'} \\
              --val_features {style_root / 'features' / f'features_val_{args.variant}.pt'} \\
              --test_features {style_root / 'features' / f'features_test_{args.variant}.pt'} \\
              --output_dir {style_root / 'classifier'} \\
              --epochs 30 \\
              --lr 1e-3 \\
              --weight_decay 1e-4 \\
              --batch_size 32 \\
              --seed 42 \\
              --patience 5
            """,
        )
    )
    print(
        block(
            "Optional binary T1/T2 classifier command shape",
            f"""
            # The generated preprocessing CSVs are 4-class CSVs. For binary T1/T2,
            # first create T1/T2-only processed CSVs by filtering labels 0 and 1, then cache those features.
            # Once binary cached features exist, use:

            python scripts/train_cached_binary_classifier.py \\
              --train_features <binary_features_train_{args.variant}.pt> \\
              --val_features <binary_features_val_{args.variant}.pt> \\
              --test_features <binary_features_test_{args.variant}.pt> \\
              --output_dir <binary_classifier_output_dir> \\
              --epochs 200 \\
              --lr 1e-3 \\
              --weight_decay 1e-4 \\
              --batch_size 64 \\
              --seed 42 \\
              --patience 200
            """,
        )
    )


def main() -> None:
    args = parse_args()
    inputs_root = args.project_root / "outputs" / "preprocessing_ablation_inputs"

    print("BrainIAC/BraTS offline preprocessing ablation command plan")
    print("=" * 64)
    print("This script only prints commands. It does not run preprocessing, registration, HD-BET, caching, or training.")
    print("Replace <TEMPLATE_FILE_HERE> with a real template image before running BrainIAC-style registration.")
    print_n4_commands(args, inputs_root)
    print_brainiac_style_commands(args, inputs_root)
    print_followup_commands(args, inputs_root)


if __name__ == "__main__":
    main()
