# Author Vs Ours BrainIAC Input Audit

This diagnostic compares BrainIAC author-released processed NIfTI images against our selected BraTS images after the original BrainIAC repository's default validation input pipeline.

It does not use `crop_pad_zscore`. The point is to answer a narrower question: when the author's processed samples and our images are sent through the same BrainIAC default model-input transform, do the resulting `[1,96,96,96]` tensors look and behave similarly?

## Why Plain Imshow Is Misleading

`plt.imshow(slice)` silently rescales each slice independently unless `vmin` and `vmax` are provided. That can make two images with very different intensity ranges look similar. This audit therefore writes multiple controlled-window figures:

- Global raw min/max window: shows true raw intensity-scale differences.
- Shared raw nonzero p1/p99 window: main fair anatomical comparison before model normalization.
- Per-image raw p1/p99 window: anatomy-only view, not valid for intensity-scale comparison.
- Fixed tensor window `[-3,3]`: shows what the BrainIAC backbone receives after default normalization.
- Shared tensor p1/p99 window: shows normalized tensor contrast using one shared display window.

Every `imshow` call in the script explicitly sets `cmap="gray"`, `vmin=...`, and `vmax=...`.

## BrainIAC Default Validation Transform

The script uses the author validation transform from `src/dataset.py`:

```python
Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(96, 96, 96), mode="trilinear"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ToTensord(keys=["image"]),
])
```

`ScaleIntensityd(0,1)` is not used because it is commented out in the author repository's validation transform. No `Orientationd` is added to the actual model-input path. Raw images are canonicalized to RAS only for visualization, so raw slice displays are easier to compare.

## Outputs

The default output directory is:

```bash
/content/BrainIAC_Testing/brats_sequence_project/outputs/author_vs_ours_brainiac_default_input
```

The script saves:

- `raw_nifti_stats.csv`
- `raw_nifti_stats.json`
- `brainiac_default_tensor_stats.csv`
- `brainiac_default_tensor_stats.json`
- `batch_debug_summary.json`
- `display_window_summary.json`
- `raw_fixed_global_window.png`
- `raw_shared_nonzero_p1_p99_window.png`
- `raw_per_image_p1_p99_ANATOMY_ONLY.png`
- `brainiac_default_tensor_fixed_minus3_to3.png`
- `brainiac_default_tensor_shared_p1_p99.png`
- `batch_debug_exact_brainiac_default_input.png`

If `--save_npz` is passed, it also saves `sampled_tensors_brainiac_default_input.npz`.

## Colab Commands

Run from the project directory:

```bash
cd /content/BrainIAC_Testing/brats_sequence_project
```

### 1. Author Samples Vs Original BraTS Only

```bash
python scripts/compare_author_processed_vs_brainiac_default_input.py \
  --author_processed_dir /content/BrainIAC_Testing/src/data/sample/processed \
  --brats_root /content/data \
  --output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/author_vs_ours_brainiac_default_input \
  --max_author_images 8 \
  --max_brats_patients 4
```

### 2. Author Samples Vs Original BraTS Plus N4

```bash
python scripts/compare_author_processed_vs_brainiac_default_input.py \
  --author_processed_dir /content/BrainIAC_Testing/src/data/sample/processed \
  --brats_root /content/data \
  --n4_csv /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/csvs/train_n4_only.csv \
  --output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/author_vs_ours_brainiac_default_input_with_n4 \
  --max_author_images 8 \
  --max_brats_patients 4 \
  --max_rows_per_csv 16
```

### 3. Author Samples Vs Original BraTS Plus N4 And BrainIAC-Style

```bash
python scripts/compare_author_processed_vs_brainiac_default_input.py \
  --author_processed_dir /content/BrainIAC_Testing/src/data/sample/processed \
  --brats_root /content/data \
  --n4_csv /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/n4_only/csvs/train_n4_only.csv \
  --brainiac_style_csv /content/BrainIAC_Testing/brats_sequence_project/outputs/preprocessing_ablation_inputs/brainiac_style/csvs/train_brainiac_style.csv \
  --output_dir /content/BrainIAC_Testing/brats_sequence_project/outputs/author_vs_ours_brainiac_default_input_with_n4_and_brainiac_style \
  --max_author_images 8 \
  --max_brats_patients 4 \
  --max_rows_per_csv 16
```

## How To Interpret Results

Start with `raw_nifti_stats.csv` to compare shape, spacing, orientation, and raw intensity distributions before any BrainIAC transform. Large raw intensity differences may be expected because the author samples are already processed and BraTS has its own challenge preprocessing.

Then inspect `brainiac_default_tensor_stats.csv`. After `NormalizeIntensityd(nonzero=True, channel_wise=True)`, nonzero voxels should usually have mean near 0 and standard deviation near 1. If one source group has a very different normalized tensor range, many NaNs/Infs, or unusual nonzero fraction, that is evidence of a preprocessing or input-domain mismatch.

For figures, use `raw_shared_nonzero_p1_p99_window.png` as the main anatomical raw comparison and `brainiac_default_tensor_fixed_minus3_to3.png` as the main model-input comparison. Use `raw_per_image_p1_p99_ANATOMY_ONLY.png` only to inspect anatomy, cropping, and gross image content; it should not be used to compare intensity scale.

`batch_debug_summary.json` confirms the stacked tensor shape. For BrainIAC feature extraction, the expected model input is `[N,1,96,96,96]`.

## Notes And Limits

This is a diagnostic comparison script. It does not preprocess a dataset, cache BrainIAC features, train a classifier, run registration, run HD-BET, or modify existing CSVs.

The BrainIAC-style preprocessing ablation is still an ablation based on repo-visible behavior and command flags. It should not be described as a guaranteed reproduction of the authors' internal preprocessing unless that is independently verified.
