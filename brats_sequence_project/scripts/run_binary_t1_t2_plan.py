"""Print the Colab command plan for the binary T1-vs-T2 BrainIAC pipeline."""

from __future__ import annotations

import textwrap


def main() -> None:
    commands = r"""
    cd /content/BrainIAC/brats_sequence_project

    python scripts/make_brats_t1_t2_splits.py \
      --dataset_root /content/data \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2 \
      --seed 42 \
      --train_ratio 0.8

    python scripts/check_preprocessing_variant.py \
      --csv_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/train_t1_t2.csv \
      --variant crop_pad_zscore \
      --batch_size 2 \
      --num_workers 0 \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/preprocess_check

    python scripts/cache_brainiac_features.py \
      --brainiac_src /content/BrainIAC/src \
      --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
      --csv_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/train_t1_t2.csv \
      --split_name train \
      --variant crop_pad_zscore \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features \
      --batch_size 2 \
      --num_workers 2 \
      --use_amp

    python scripts/cache_brainiac_features.py \
      --brainiac_src /content/BrainIAC/src \
      --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
      --csv_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/val_t1_t2.csv \
      --split_name val \
      --variant crop_pad_zscore \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features \
      --batch_size 2 \
      --num_workers 2 \
      --use_amp

    python scripts/cache_brainiac_features.py \
      --brainiac_src /content/BrainIAC/src \
      --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
      --csv_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/test_t1_t2.csv \
      --split_name test \
      --variant crop_pad_zscore \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features \
      --batch_size 2 \
      --num_workers 2 \
      --use_amp

    python scripts/train_cached_binary_classifier.py \
      --train_features /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features/features_train_crop_pad_zscore.pt \
      --val_features /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features/features_val_crop_pad_zscore.pt \
      --test_features /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features/features_test_crop_pad_zscore.pt \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/classifier \
      --epochs 200 \
      --lr 1e-3 \
      --weight_decay 1e-4 \
      --batch_size 64 \
      --seed 42 \
      --patience 200

    python scripts/visualize_binary_t1_t2_batch.py \
      --csv_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/train_t1_t2.csv \
      --variant crop_pad_zscore \
      --batch_size 20 \
      --split_name train \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/visual_debug/train_batch

    python scripts/visualize_binary_t1_t2_predictions.py \
      --csv_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/test_t1_t2.csv \
      --predictions_csv /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/classifier/predictions.csv \
      --variant crop_pad_zscore \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/visual_debug/predictions

    python scripts/analyze_binary_t1_t2_features.py \
      --features_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features/features_train_crop_pad_zscore.pt \
      --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/feature_analysis \
      --method pca

    cd /content/BrainIAC/brats_sequence_project
    zip -r outputs/binary_t1_t2_results.zip outputs/binary_t1_t2 \
      -x "outputs/binary_t1_t2/features/*.pt" \
      -x "outputs/binary_t1_t2/classifier/*.pt"
    """
    print(textwrap.dedent(commands).strip())


if __name__ == "__main__":
    main()
