# BraTS2020 + BrainIAC Preprocessing Audit

## A. Executive Summary

- The active BraTS classifier pipeline does **not** perform physical 1 mm resampling at runtime. The current path is `LoadImaged -> EnsureChannelFirstd -> Resized(96,96,96) -> NormalizeIntensityd(nonzero=True, channel_wise=True) -> ToTensord` in `brats_sequence_project/datasets/brats_sequence_dataset.py:52`.
- BrainIAC’s original sequence-classification runtime path uses the same core resize-based transform chain in `src/dataset.py:37`; it also does **not** use `Spacingd`, `Orientationd`, SimpleITK resampling, or nibabel affine resampling during runtime loading.
- The separate offline script `src/preprocessing/mri_preprocess_3d_simple.py:31` does registration, N4 bias correction, skull stripping, and SimpleITK resampling, but it is not called anywhere in the BraTS Phase 3 training path.
- BraTS source MRI files in the available archive are spatially consistent: 1,976 non-segmentation MRI files across 494 patients all have shape `240x240x155`, spacing `1.0x1.0x1.0`, and `LPS` orientation. No modality-level shape/spacing/orientation mismatches were found.
- Runtime resize from `240x240x155` to `96x96x96` implies an effective voxel size of about `2.5 x 2.5 x 1.615 mm` if interpreted physically. That is not equivalent to true 1 mm isotropic resampling.
- `NormalizeIntensityd(nonzero=True, channel_wise=True)` standardizes only nonzero voxels per volume. With one channel, `channel_wise=True` does not change semantics; it remains per-volume, per-channel normalization.
- Label/split logic is internally consistent in the BraTS project: `0=T1, 1=T2, 2=FLAIR, 3=T1CE`, class counts are balanced, no segmentation rows are included, and there is no patient overlap across `train.csv`, `val.csv`, and `test.csv`.
- The most plausible causes of ~74% accuracy are: frozen feature mismatch for this downstream task, spatial distortion from plain resizing, modality cues being reduced by per-volume normalization, and BrainIAC pretraining/runtime expectation mismatch around physically standardized inputs.

## B. File-by-File Preprocessing Map

| File | Role | Loads image? | Changes spacing? | Resizes shape? | Normalizes intensity? | Final output affected? |
| --- | --- | --- | --- | --- | --- | --- |
| `BrainIAC/brats_sequence_project/datasets/brats_sequence_dataset.py:17` | Active BraTS dataset for Phase 2/3 | Yes | No physical spacing change | Yes, `Resized(96,96,96)` | Yes | Yes |
| `BrainIAC/src/dataset.py:9` | Original BrainIAC sequence dataset + transforms | Yes | No physical spacing change at runtime | Yes, `Resized(96,96,96)` | Yes | Yes |
| `BrainIAC/src/model.py:7` | BrainIAC ViT backbone definition | No | No | Assumes `img_size=(96,96,96)` | No | Yes, input contract |
| `BrainIAC/src/load_brainiac.py:7` | Checkpoint/model loader helper | No | No | No | No | Only model loading |
| `BrainIAC/src/train_lightning_multiclass.py:18` | Original BrainIAC sequence classifier training entrypoint | Indirectly via dataset | No | Indirectly via dataset/config | Indirectly via dataset | Yes |
| `BrainIAC/src/preprocessing/mri_preprocess_3d_simple.py:31` | Offline MRI preprocessing script | Yes | Yes, via SimpleITK resampling | Indirectly via registration target space | Bias correction + skull stripping pipeline | Separate offline outputs |
| `BrainIAC/docs/downstream_tasks/MR_sequence_classification.md:11` | BrainIAC sequence classification doc | No | Claims preprocessing expectations, not runtime enforcement | Documents `96x96x96` | Documents normalized pipeline | Reference only |
| `BrainIAC/brats_sequence_project/scripts/make_brats_splits.py:31` | BraTS CSV generation/split logic | No | No | No | No | Labels/splits only |
| `BrainIAC/brats_sequence_project/scripts/check_brainiac_dataloader.py:19` | One-batch dataloader sanity check | Indirectly | No | Checks resized output | Indirectly | Validation only |
| `BrainIAC/brats_sequence_project/scripts/check_brainiac_backbone.py:80` | One-forward-pass backbone sanity check | Indirectly | No | Checks resized input shape | Indirectly | Validation only |
| `BrainIAC/brats_sequence_project/scripts/train_frozen_brainiac_classifier.py:108` | Frozen-backbone classifier training | Indirectly | No | Uses dataset output as-is | Indirectly | Yes |
| `BrainIAC/brats_sequence_project/scripts/evaluate_frozen_brainiac_classifier.py:121` | Frozen-backbone evaluation | Indirectly | No | Uses dataset output as-is | Indirectly | Yes |

## C. Original Data Metadata

### Source Used

- The generated CSVs currently point to `/Users/kp/Documents/BRAINIAC/BrainIAC/Dataset/...`, but that extracted dataset directory is not present in this workspace.
- The archive `archive.zip` was inspected directly with nibabel header reads. This preserved the original BraTS file metadata without extracting or modifying the dataset.

### Full Lightweight Header Summary

- Non-segmentation MRI files scanned: `1,976`
- Segmentation files present in archive: `369`
- Patients covered: `494` (`369` training + `125` validation)
- Unique MRI shape: `240x240x155` for all 1,976 non-segmentation MRI files
- Unique MRI spacing: `1.0x1.0x1.0 mm` for all 1,976 non-segmentation MRI files
- Unique orientation from affine: `LPS` for all 1,976 non-segmentation MRI files
- Modality-level shape mismatches within a patient: `0`
- Modality-level spacing mismatches within a patient: `0`
- Modality-level orientation mismatches within a patient: `0`

### Dtype / Intensity Notes

- Header dtypes across non-segmentation MRI files:
  - `int16`: `1,797`
  - `float32`: `179`
- Forty-five patients in the tail of the archive use `float32` MRI files. One patient, `BraTS20_Validation_028`, has mixed MRI dtypes across modalities (`T2=int16`, the other three modalities `float32`).
- Sample intensity statistics for `BraTS20_Training_001` show strong modality-specific raw intensity scales before normalization:
  - `T1`: min `0`, max `678`, nonzero mean `354.27`
  - `T2`: min `0`, max `376`, nonzero mean `114.69`
  - `FLAIR`: min `0`, max `625`, nonzero mean `173.00`
  - `T1CE`: min `0`, max `1845`, nonzero mean `417.33`

### Modality Consistency

- For each patient, the four imaging modalities (`T1`, `T2`, `FLAIR`, `T1CE`) are shape-aligned and spacing-aligned in the source data.
- Segmentation masks exist only in the training source and are not part of the generated BraTS sequence CSVs.
- Validation MRI files follow the same shape/spacing/orientation conventions as training MRI files.

## D. Runtime Transform Pipeline

### Active BraTS Runtime Path

The active Phase 3 pipeline is implemented in `BrainIAC/brats_sequence_project/datasets/brats_sequence_dataset.py:52`:

1. `LoadImaged(keys=["image"])`
2. `EnsureChannelFirstd(keys=["image"])`
3. `Resized(keys=["image"], spatial_size=(96,96,96), mode="trilinear")`
4. `NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)`
5. `ToTensord(keys=["image"])`

### Step-by-Step Behavior on a Real BraTS Sample

Real sample used: `BraTS20_Training_001_t1.nii` from the archive.

| Step | Input shape | Output shape | Physical spacing changed? | Affine/meta changed? | Intensity changed? |
| --- | --- | --- | --- | --- | --- |
| `LoadImaged` | file path | `240x240x155` | No | Returns a `MetaTensor`; sample meta exposed original affine and `pixdim=1.0` | No |
| `EnsureChannelFirstd` | `240x240x155` | `1x240x240x155` | No | Adds channel dimension only; no observed reorientation | No |
| `Resized` | `1x240x240x155` | `1x96x96x96` | No true physical resampling | MetaTensor affine scale changed to roughly `[-2.5, -2.5, 1.6146]` on the diagonal, but `pixdim` remained at `1.0` in sample meta | Interpolates values |
| `NormalizeIntensityd(nonzero=True, channel_wise=True)` | `1x96x96x96` | `1x96x96x96` | No | Affine/meta preserved from prior step | Yes; only nonzero voxels are standardized per volume |
| `ToTensord` | `1x96x96x96` | `1x96x96x96` | No | Final output remained a `MetaTensor` in the local probe, but downstream training uses it as a tensor input | No new intensity change |

### Additional Transform Semantics

- `LoadImaged` preserves metadata through MONAI `MetaTensor`.
- `EnsureChannelFirstd` does not reorient images; it only ensures channel-first layout.
- `Resized` uses interpolation in voxel/grid space. It is not the same as MONAI `Spacingd`.
- `NormalizeIntensityd(nonzero=True, channel_wise=True)` computes mean/std on the current volume only and excludes zeros from the normalization mask.
- Because the background zeros remain zero while only nonzero voxels are standardized, the **whole-volume** post-normalization standard deviation is not expected to be exactly `1.0`.
- The final tensor that reaches BrainIAC is `torch.float32` with shape `[B, 1, 96, 96, 96]`.

## E. Physical Interpretation

- Original BraTS MRI shape: `240x240x155`
- Original BraTS MRI spacing: `1.0 x 1.0 x 1.0 mm`
- Runtime BrainIAC input shape: `1x96x96x96`

If the original image is simply resized to `96x96x96`, then the effective physical voxel size implied by preserving the original field of view is approximately:

- X: `240 / 96 = 2.5 mm`
- Y: `240 / 96 = 2.5 mm`
- Z: `155 / 96 = 1.6146 mm`

This is different from true `1 mm` resampling:

- **True physical resampling to 1 mm spacing** changes the array so that each voxel corresponds to 1 mm in physical space, using the original affine/spacing as the resampling target.
- **Simple resizing to a fixed array shape** changes the number of voxels by interpolation but does not preserve the original physical voxel spacing claim unless spacing metadata is recomputed and interpreted correctly.

The current runtime path is doing **simple resizing**, not true 1 mm physical resampling.

Why this matters:

- A pretrained model described as expecting physically standardized 1 mm inputs may receive a different anatomical scale distribution if inputs are only shape-resized.
- The anisotropic downsampling from `240x240x155` to `96x96x96` compresses in-plane dimensions more aggressively than through-plane depth, which can distort anatomy relative to the pretraining regime.

## F. Comparison with BrainIAC Preprocessing Claims

### What the Documentation Says

- `docs/downstream_tasks/MR_sequence_classification.md:16` states: `Image Size: 96×96×96 voxels (automatically resized)`.
- The same doc labels the task as sequence classification with semantic order `0: T1, 1: T2, 2: FLAIR, 3: T1CE` at `docs/downstream_tasks/MR_sequence_classification.md:11`.
- Broader BrainIAC material describes standardized/skull-stripped/registered inputs, but that requirement is not enforced by the runtime sequence dataset itself.

### What the Runtime Code Does

- `src/dataset.py:39` loads MRI files with `LoadImaged`.
- `src/dataset.py:41` resizes them to `96x96x96` with `Resized`.
- No runtime `Spacingd`, `Orientationd`, or SimpleITK resampling is present.

### What the Offline Script Does

- `src/preprocessing/mri_preprocess_3d_simple.py:31` runs SimpleITK registration to a fixed image space.
- `src/preprocessing/mri_preprocess_3d_simple.py:79` sets `new_spacing = (1,1,1)`.
- `src/preprocessing/mri_preprocess_3d_simple.py:133` resamples moving images into the fixed image space with `sitk.Resample`.
- `src/preprocessing/mri_preprocess_3d_simple.py:141` writes output `.nii.gz` files.

### What Our Current BraTS Pipeline Does

- `brats_sequence_project/datasets/brats_sequence_dataset.py:52` mirrors the runtime BrainIAC resize-based path.
- It does **not** invoke the offline preprocessing script automatically.
- It therefore relies on the BraTS data already being skull-stripped/aligned/1 mm, and then applies a fixed resize to `96x96x96`.

## G. Accuracy Impact Hypotheses

Ranked most likely to less likely based on current evidence:

1. **Frozen BrainIAC features may not separate MRI sequences well without fine-tuning.** The checkpoint is used as a fixed feature extractor; no adaptation occurs inside the backbone.
2. **Runtime resizing may distort physical scale/context.** The current pipeline shape-resizes rather than physically resamples, so the effective spacing at model input is not 1 mm.
3. **Per-volume nonzero normalization may suppress modality cues.** Standardizing each volume independently can reduce absolute intensity-scale differences that help distinguish T1/T2/FLAIR/T1CE.
4. **BrainIAC checkpoint may be self-supervised and not directly optimized for sequence classification.** Useful generic features are not guaranteed to produce high separability for this exact label set under a frozen linear head.
5. **Label/order issue is unlikely but should stay explicitly verified.** Current BraTS CSVs, project scripts, and evaluation mapping are internally consistent.
6. **Dataset/split issue is unlikely but not impossible.** Splits are balanced and patient-level, so no obvious leakage or imbalance issue was found.

## H. Recommended Next Experiments

Do not implement these inside the preprocessing audit. Run them as Phase 4 experiments:

1. **Label baseline check**: train a trivial baseline on raw metadata/intensity summaries to prove label mapping is sane and not accidentally permuted.
2. **Feature separability audit**: cache BrainIAC `[B,768]` features and inspect PCA / t-SNE / UMAP by modality.
3. **Preprocessing ablation**: compare current `Resize(96,96,96)` against alternatives such as crop/pad or `Spacingd` + crop/resize.
4. **Normalization ablation**: compare current nonzero per-volume normalization against no normalization or a more conservative scaling strategy.
5. **Partial fine-tuning experiment**: unfreeze only the last ViT block or a small subset of layers instead of keeping the entire backbone frozen.
6. **Confusion-matrix review**: identify which modality pairs dominate the errors, especially `FLAIR` vs `T2` and `T1` vs `T1CE`.

## I. Commands

### Preprocessing audit

```bash
cd /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project
python scripts/audit_preprocessing_pipeline.py \
  --csv_path outputs/train.csv \
  --num_samples 4 \
  --output_dir outputs/preprocessing_audit
```

If the extracted `Dataset/` directory is missing, the script automatically tries `archive.zip` from the workspace root. You can also pass it explicitly:

```bash
python scripts/audit_preprocessing_pipeline.py \
  --csv_path outputs/train.csv \
  --num_samples 4 \
  --output_dir outputs/preprocessing_audit \
  --archive_zip /Users/kp/Documents/BRAINIAC/archive.zip
```

### Visualization audit

```bash
cd /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project
python scripts/visualize_preprocessing_samples.py \
  --csv_path outputs/train.csv \
  --output_dir outputs/preprocessing_audit/visuals
```

### Label / split check

```bash
cd /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project
python scripts/check_brats_label_splits.py \
  --train_csv outputs/train.csv \
  --val_csv outputs/val.csv \
  --test_csv outputs/test.csv \
  --output_json outputs/preprocessing_audit/label_split_check.json
```

### Report review

```bash
cd /Users/kp/Documents/BRAINIAC/BrainIAC
rtk read brats_sequence_project/PREPROCESSING_AUDIT_REPORT.md
```

## Label Order Audit

- BrainIAC doc semantic order: `0=T1, 1=T2, 2=FLAIR, 3=T1CE` in `docs/downstream_tasks/MR_sequence_classification.md:11`.
- BrainIAC example CSVs are 1-based in the docs, and `SequenceDataset` converts them to 0-based via `int(label) - 1` in `src/dataset.py:103`.
- The BraTS project uses already 0-based labels in `make_brats_splits.py:16`.
- Training, validation, and evaluation scripts in `brats_sequence_project/scripts/` all assume the same BraTS 0-based mapping.
- Because the classifier head is trained from scratch on our own labels, external BrainIAC label order would only matter if our train/eval mapping were internally inconsistent. No such inconsistency was found.

## Operational Caveat

- The current `train.csv`, `val.csv`, and `test.csv` use absolute dataset paths under `/Users/kp/Documents/BRAINIAC/BrainIAC/Dataset/...`.
- That extracted dataset directory is not present in this workspace snapshot, so runtime re-checks require either restoring that directory or using the archive fallback added in the new audit scripts.
