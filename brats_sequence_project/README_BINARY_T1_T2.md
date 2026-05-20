# Binary T1 vs T2 BrainIAC Experiment

This workflow runs a separate binary MRI sequence-classification experiment using only BraTS T1 and T2 volumes.
It does not replace or modify the existing four-class T1/T2/FLAIR/T1CE pipeline.

## Goal

Train a binary classifier:

- `0 = T1`
- `1 = T2`

The BrainIAC backbone is frozen. We cache one `[768]` feature vector per image and train only a small MLP classifier on top.

## Dataset Filtering

The split script keeps only:

- `_t1.nii` -> label `0`, modality `T1`
- `_t2.nii` -> label `1`, modality `T2`

It excludes:

- `_flair.nii`
- `_t1ce.nii`
- segmentation masks such as `_seg.nii` or files containing `seg` / `segm`

Splits are patient-level. BraTS2020 training patients are split into train/val, and BraTS2020 validation patients are used as held-out test.

Expected full-data sizes with `--train_ratio 0.8`:

- train: about `295 patients x 2 = 590 rows`
- val: about `74 patients x 2 = 148 rows`
- test: `125 patients x 2 = 250 rows`

## Preprocessing

Default variant: `crop_pad_zscore`.

For BraTS2020, the original volumes are `240 x 240 x 155` with `1 x 1 x 1 mm` spacing. `crop_pad_zscore` uses center crop/pad to produce:

- BrainIAC tensor input: `[B, 1, 96, 96, 96]`
- Frozen BrainIAC feature output: `[B, 768]`

This avoids writing preprocessed NIfTI copies and reads directly from the CSV `image_path` column.

## Colab Commands

```bash
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
```

To print the full command plan, including visualization, PCA, and zipping:

```bash
python scripts/run_binary_t1_t2_plan.py
```

## Outputs

All binary outputs live under:

```text
outputs/binary_t1_t2/
```

Main files:

- `train_t1_t2.csv`, `val_t1_t2.csv`, `test_t1_t2.csv`
- `label_mapping_t1_t2.json`
- `split_summary_t1_t2.json`
- `features/features_train_crop_pad_zscore.pt`
- `features/features_val_crop_pad_zscore.pt`
- `features/features_test_crop_pad_zscore.pt`
- `classifier/best_binary_classifier.pt`
- `classifier/train_history.csv`
- `classifier/best_metrics.json`
- `classifier/test_metrics.json`
- `classifier/val_predictions.csv`
- `classifier/predictions.csv`

## Plots To Inspect

Classifier plots in `classifier/plots/`:

- `loss_curve.png`: train/val loss over epochs.
- `accuracy_curve.png`: train/val accuracy.
- `f1_curve.png`: macro F1 and positive-class T2 F1.
- `precision_recall_curve_over_epochs.png`: macro precision/recall over epochs.
- `overfitting_gap_curve.png`: increasing train-val gaps suggest overfitting.
- `confusion_matrix_raw.png` and `confusion_matrix_normalized.png`: T1/T2 mistakes.
- `confidence_histogram.png`: confidence calibration signal.
- `confidence_correct_vs_incorrect.png`: checks whether wrong predictions are overconfident.
- `roc_curve_binary.png` and `precision_recall_curve_binary.png`: threshold behavior for T2 as the positive class.

Visual debugging:

```bash
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
```

Feature PCA:

```bash
python scripts/analyze_binary_t1_t2_features.py \
  --features_path /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/features/features_train_crop_pad_zscore.pt \
  --output_dir /content/BrainIAC/brats_sequence_project/outputs/binary_t1_t2/feature_analysis \
  --method pca
```

## How To Interpret Wrong Predictions

Start with `visual_debug/predictions/wrong_predictions_grid_t1_t2.png`.
If wrong predictions are visually blank, heavily cropped, or anatomically inconsistent, inspect preprocessing first.
If the images look correct but the confidence is high, inspect the confusion matrix and PCA plot to see whether frozen BrainIAC features separate T1/T2 cleanly.
If the training score is much higher than validation/test, use the overfitting gap curve and consider stronger regularization or fewer epochs.

## PPT Suggestions

For a concise presentation, show:

- split summary: patient counts and balanced T1/T2 rows
- preprocessing batch grid: proves the model sees `[1,96,96,96]` brain volumes
- confusion matrix normalized: main performance summary
- ROC and precision-recall curves: threshold-independent performance
- wrong-prediction grid: qualitative failure analysis
- PCA plot: whether frozen BrainIAC features cluster by sequence type
