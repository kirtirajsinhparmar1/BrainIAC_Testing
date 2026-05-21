# BraTS Preprocessing Ablations for BrainIAC

This folder adds controlled offline preprocessing ablations for the BraTS2020 MRI sequence classification project.

The goal is to compare:

1. Original BraTS NIfTI files plus runtime `crop_pad_zscore`
2. N4-corrected BraTS NIfTI files plus runtime `crop_pad_zscore`
3. BrainIAC-style processed BraTS NIfTI files plus runtime `crop_pad_zscore`

This is a BrainIAC-style ablation based on the public repo preprocessing script. It is not guaranteed to be identical to the authors' internal preprocessing.

## Why This Matters

BrainIAC documentation and paper-level descriptions mention offline MRI preprocessing steps such as N4 bias correction, physical resampling or registration, and skull stripping. BraTS2020 is already skull-stripped, co-registered, and distributed at approximately 1 mm spacing, but it is not guaranteed to match BrainIAC pretraining preprocessing exactly.

These scripts test whether an offline preprocessing mismatch affects frozen BrainIAC features and downstream sequence classification.

## Offline vs Runtime Preprocessing

Offline anatomical preprocessing writes new NIfTI files. Examples are N4 correction, registration to a template, and optional skull stripping.

Runtime model preprocessing happens inside `BraTSSequenceDataset` and does not write NIfTI files. The baseline variant is:

```text
LoadImaged
EnsureChannelFirstd
ResizeWithPadOrCropd(96,96,96)
NormalizeIntensityd(nonzero=True, channel_wise=True)
ToTensord
```

The BrainIAC frozen backbone still receives `[B,1,96,96,96]` and outputs `[B,768]`.

## N4-only Script

`scripts/preprocess_brats_n4_only.py` applies SimpleITK N4 bias field correction to T1, T2, FLAIR, and T1CE only. It excludes segmentation masks, preserves the original image geometry, and does not register, resample, or skull-strip.

Outputs include corrected NIfTI files, metadata CSVs, header summaries, train/val/test CSVs, label mapping, split summary, and before/after visualizations.

Run this first because it isolates the simplest possible BrainIAC-paper preprocessing difference.

## BrainIAC-style Script

`scripts/preprocess_brats_brainiac_style.py` implements a configurable offline ablation:

- Optional N4 correction, enabled by default
- Optional registration/resampling to a template, enabled by default
- Optional HD-BET, disabled by default

Registration requires an explicit `--template_path`. This is intentional because no safe default template should be guessed.

HD-BET is off by default because BraTS is already skull-stripped. Running it on BraTS may remove valid brain or tumor regions, so only use it for an explicit experimental subset after inspecting visual outputs.

## Run a Small Subset First

```bash
cd /content/BrainIAC_Testing/brats_sequence_project

python scripts/preprocess_brats_n4_only.py \
  --dataset_root /content/data \
  --output_root /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/images \
  --splits_output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/csvs \
  --max_patients 10 \
  --visualize_examples 4
```

BrainIAC-style subset:

```bash
cd /content/BrainIAC_Testing/brats_sequence_project

python scripts/preprocess_brats_brainiac_style.py \
  --dataset_root /content/data \
  --output_root /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/brainiac_style/images \
  --splits_output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/brainiac_style/csvs \
  --template_path /content/BrainIAC_Testing/src/preprocessing/<TEMPLATE_FILE_HERE> \
  --max_patients 10 \
  --run_n4 \
  --run_registration \
  --visualize_examples 4
```

Replace `<TEMPLATE_FILE_HERE>` with a real template image. Do not run registration without confirming the template.

## Audit Outputs

```bash
python scripts/check_processed_dataset.py \
  --csv_path /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/csvs/train_n4_only.csv \
  --output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/audit_train

python scripts/check_processed_dataset.py \
  --csv_path /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/brainiac_style/csvs/train_brainiac_style.csv \
  --output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/brainiac_style/audit_train
```

The audit checks path existence, label and modality consistency, shapes, spacings, nonzero voxel counts, and creates a sample axial grid.

## Feature Caching

Use the processed CSVs with the existing frozen BrainIAC feature cacher:

```bash
python scripts/cache_brainiac_features.py \
  --brainiac_src /content/BrainIAC_Testing/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --csv_path /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/csvs/train_n4_only.csv \
  --split_name train \
  --variant crop_pad_zscore \
  --output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/features \
  --batch_size 2 \
  --num_workers 2 \
  --use_amp
```

Repeat for `val_n4_only.csv` and `test_n4_only.csv`, then repeat the same pattern for `train_brainiac_style.csv`, `val_brainiac_style.csv`, and `test_brainiac_style.csv`.

## Train Classifiers

Four-class classifier:

```bash
python scripts/train_cached_feature_classifier.py \
  --train_features /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/features/features_train_crop_pad_zscore.pt \
  --val_features /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/features/features_val_crop_pad_zscore.pt \
  --test_features /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/features/features_test_crop_pad_zscore.pt \
  --output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/classifier \
  --epochs 30 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --batch_size 32 \
  --seed 42 \
  --patience 5
```

For binary T1/T2, first filter the processed CSVs to labels `0` and `1`, cache features from those filtered CSVs, then use `scripts/train_cached_binary_classifier.py`.

## Full Runs

N4-only full run:

```bash
python scripts/preprocess_brats_n4_only.py \
  --dataset_root /content/data \
  --output_root /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/images \
  --splits_output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/csvs \
  --visualize_examples 8
```

BrainIAC-style full run without HD-BET:

```bash
python scripts/preprocess_brats_brainiac_style.py \
  --dataset_root /content/data \
  --output_root /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/brainiac_style/images \
  --splits_output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/brainiac_style/csvs \
  --template_path /content/BrainIAC_Testing/src/preprocessing/<TEMPLATE_FILE_HERE> \
  --run_n4 \
  --run_registration \
  --visualize_examples 8
```

## What to Inspect

Inspect these before training:

- `n4_only/visual_debug/n4_before_after_grid.png`
- `brainiac_style/visual_debug/brainiac_style_before_after_grid.png`
- `n4_only_metadata.csv`
- `brainiac_style_metadata.csv`
- `n4_only_header_summary.json`
- `brainiac_style_header_summary.json`
- `processed_dataset_audit.json`
- `sample_visual_grid.png`

After training, compare:

- validation and test accuracy
- macro F1
- confusion matrices
- confidence plots
- wrong-prediction grids
- overfitting gap curves

## Failure Modes

Registration can fail or distort images if the template is not appropriate for BraTS. HD-BET can remove useful tumor or brain tissue because BraTS is already skull-stripped. N4 can change intensity distributions in ways that help or hurt frozen features. Output shape or spacing mismatches should be caught by `check_processed_dataset.py`.

Always run a subset first, inspect the before/after PNGs and audit JSON, then run the full preprocessing only if the subset looks anatomically reasonable.
