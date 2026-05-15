# Phase 4 — BrainIAC Preprocessing Ablations

## Why this phase exists

The frozen BrainIAC backbone plus small classifier head plateaued near 74% validation/test accuracy. The preprocessing audit showed a likely mismatch between:

- BraTS source geometry: `240 x 240 x 155 @ 1 mm`
- Runtime BrainIAC input: `96 x 96 x 96`
- Active runtime behavior: interpolation resize, not true physical 1 mm resampling

This phase makes the preprocessing choices measurable instead of speculative.

## Variants

All variants keep:

- original BraTS `.nii` files
- existing patient-level `train.csv`, `val.csv`, `test.csv`
- frozen BrainIAC ViT backbone
- classifier input shape `[B, 1, 96, 96, 96]`
- BrainIAC feature shape `[B, 768]`

Implemented variants:

- `resize_zscore`
  - `LoadImaged -> EnsureChannelFirstd -> Resized -> NormalizeIntensityd(nonzero=True, channel_wise=True) -> ToTensord`
  - current baseline
- `resize_none`
  - `LoadImaged -> EnsureChannelFirstd -> Resized -> ToTensord`
  - tests whether z-score normalization removes useful modality cues
- `resize_percentile`
  - `LoadImaged -> EnsureChannelFirstd -> Resized -> custom nonzero percentile clip [1,99] -> scale [0,1] -> ToTensord`
  - preserves relative contrast differently from z-score normalization
- `crop_pad_zscore`
  - `LoadImaged -> EnsureChannelFirstd -> ResizeWithPadOrCropd -> NormalizeIntensityd -> ToTensord`
  - removes interpolation resizing and replaces it with centered crop/pad behavior
- `physical_crop_zscore`
  - `LoadImaged -> EnsureChannelFirstd -> Orientationd(LPS) -> Spacingd(1,1,1) -> CenterSpatialCropd -> ResizeWithPadOrCropd -> NormalizeIntensityd -> ToTensord`
  - makes physical-space intent explicit even though BraTS is already 1 mm

## Files

- `datasets/brats_sequence_dataset.py`
  - adds `preprocessing_variant`
- `scripts/check_preprocessing_variant.py`
  - sanity-check one batch for a chosen variant
- `scripts/cache_brainiac_features.py`
  - caches frozen BrainIAC `[768]` features
- `scripts/train_cached_feature_classifier.py`
  - trains classifier on cached features only
- `scripts/analyze_feature_separability.py`
  - PCA-based feature analysis
- `scripts/run_preprocessing_ablation_plan.py`
  - prints a reproducible command plan

## Output layout

Recommended layout:

`outputs/preprocessing_ablation/<variant>/`

Typical contents:

- `check/preprocessing_variant_summary.json`
- `features/features_train_<variant>.pt`
- `features/features_val_<variant>.pt`
- `features/features_test_<variant>.pt`
- `classifier/best_cached_classifier.pt`
- `classifier/train_history.csv`
- `classifier/best_metrics.json`
- `classifier/val_predictions.csv`
- `classifier/test_metrics.json`
- `classifier/predictions.csv`
- `classifier/plots/*.png`
- `analysis/pca_2d_by_modality.png`
- `analysis/explained_variance.png`
- `analysis/class_centroid_distances.json`

## Local smoke checks

From `BrainIAC/brats_sequence_project`:

```bash
python scripts/check_preprocessing_variant.py \
  --csv_path outputs/train.csv \
  --variant resize_zscore \
  --batch_size 2 \
  --num_workers 0 \
  --output_dir outputs/preprocessing_ablation/smoke_resize_zscore

python scripts/check_preprocessing_variant.py \
  --csv_path outputs/train.csv \
  --variant resize_none \
  --batch_size 2 \
  --num_workers 0 \
  --output_dir outputs/preprocessing_ablation/smoke_resize_none
```

## First two Colab experiments

Assume:

- project root: `/content/BrainIAC/brats_sequence_project`
- BrainIAC source: `/content/BrainIAC/src`
- checkpoint: `/content/checkpoints/BrainIAC.ckpt`
- CSVs already generated under `outputs/`

### Variant `resize_zscore`

```bash
cd /content/BrainIAC/brats_sequence_project

python scripts/cache_brainiac_features.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --csv_path outputs/train.csv \
  --split_name train \
  --variant resize_zscore \
  --output_dir outputs/preprocessing_ablation/resize_zscore/features \
  --batch_size 4 \
  --num_workers 2 \
  --use_amp

python scripts/cache_brainiac_features.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --csv_path outputs/val.csv \
  --split_name val \
  --variant resize_zscore \
  --output_dir outputs/preprocessing_ablation/resize_zscore/features \
  --batch_size 4 \
  --num_workers 2 \
  --use_amp

python scripts/cache_brainiac_features.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --csv_path outputs/test.csv \
  --split_name test \
  --variant resize_zscore \
  --output_dir outputs/preprocessing_ablation/resize_zscore/features \
  --batch_size 4 \
  --num_workers 2 \
  --use_amp

python scripts/train_cached_feature_classifier.py \
  --train_features outputs/preprocessing_ablation/resize_zscore/features/features_train_resize_zscore.pt \
  --val_features outputs/preprocessing_ablation/resize_zscore/features/features_val_resize_zscore.pt \
  --test_features outputs/preprocessing_ablation/resize_zscore/features/features_test_resize_zscore.pt \
  --output_dir outputs/preprocessing_ablation/resize_zscore/classifier \
  --epochs 30 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --batch_size 32 \
  --seed 42 \
  --patience 5

python scripts/analyze_feature_separability.py \
  --features_path outputs/preprocessing_ablation/resize_zscore/features/features_train_resize_zscore.pt \
  --output_dir outputs/preprocessing_ablation/resize_zscore/analysis \
  --method pca
```

### Variant `resize_none`

```bash
cd /content/BrainIAC/brats_sequence_project

python scripts/cache_brainiac_features.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --csv_path outputs/train.csv \
  --split_name train \
  --variant resize_none \
  --output_dir outputs/preprocessing_ablation/resize_none/features \
  --batch_size 4 \
  --num_workers 2 \
  --use_amp

python scripts/cache_brainiac_features.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --csv_path outputs/val.csv \
  --split_name val \
  --variant resize_none \
  --output_dir outputs/preprocessing_ablation/resize_none/features \
  --batch_size 4 \
  --num_workers 2 \
  --use_amp

python scripts/cache_brainiac_features.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --csv_path outputs/test.csv \
  --split_name test \
  --variant resize_none \
  --output_dir outputs/preprocessing_ablation/resize_none/features \
  --batch_size 4 \
  --num_workers 2 \
  --use_amp

python scripts/train_cached_feature_classifier.py \
  --train_features outputs/preprocessing_ablation/resize_none/features/features_train_resize_none.pt \
  --val_features outputs/preprocessing_ablation/resize_none/features/features_val_resize_none.pt \
  --test_features outputs/preprocessing_ablation/resize_none/features/features_test_resize_none.pt \
  --output_dir outputs/preprocessing_ablation/resize_none/classifier \
  --epochs 30 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --batch_size 32 \
  --seed 42 \
  --patience 5

python scripts/analyze_feature_separability.py \
  --features_path outputs/preprocessing_ablation/resize_none/features/features_train_resize_none.pt \
  --output_dir outputs/preprocessing_ablation/resize_none/analysis \
  --method pca
```

## Comparison table

Track results in this format:

| variant | val_accuracy | val_macro_f1 | test_accuracy | test_macro_f1 | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| resize_zscore |  |  |  |  | baseline |
| resize_none |  |  |  |  | tests intensity cue loss |
| resize_percentile |  |  |  |  | tests contrast-preserving normalization |
| crop_pad_zscore |  |  |  |  | tests crop/pad vs interpolation |
| physical_crop_zscore |  |  |  |  | tests explicit physical-space logic |

## Professor-facing plots

Most useful plots:

- classifier `plots/loss_curve.png`
- classifier `plots/accuracy_curve.png`
- classifier `plots/f1_curve.png`
- classifier `plots/confusion_matrix_normalized.png`
- analysis `pca_2d_by_modality.png`
- analysis `explained_variance.png`

## Interpretation guidance

- If `resize_none` beats `resize_zscore`, per-volume z-score normalization is likely erasing sequence cues.
- If `crop_pad_zscore` beats resize variants, interpolation distortion is likely a major factor.
- If PCA shows poor class separation even before classifier training, the frozen BrainIAC features are likely the bottleneck.
- If PCA separates classes well but classifier accuracy stays low, the issue is more likely classifier capacity, optimization, or label noise.
