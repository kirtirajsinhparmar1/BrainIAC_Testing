# Phase 2: BraTS Sequence Dataset and BrainIAC Sanity Checks

This folder contains the Phase 2 execution-readiness pipeline for classifying one BraTS MRI volume as `T1`, `T2`, `FLAIR`, or `T1CE` with a BrainIAC pretrained backbone.

No script in this phase trains a model, copies BraTS files, renames BraTS files, or saves preprocessed NIfTI volumes.

## Folder Structure

```text
brats_sequence_project/
  datasets/
    brats_sequence_dataset.py
  scripts/
    make_brats_splits.py
    check_brainiac_dataloader.py
    check_brainiac_backbone.py
  outputs/
    train.csv
    val.csv
    test.csv
    label_mapping.json
    split_summary.json
```

## Labels

```text
0 = T1
1 = T2
2 = FLAIR
3 = T1CE
```

## Create Patient-Level Splits

```bash
python scripts/make_brats_splits.py \
  --dataset_root /Users/kp/Documents/BRAINIAC/BrainIAC/Dataset \
  --output_dir /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs \
  --seed 42 \
  --train_ratio 0.8
```

The script uses:

- `BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData` for train/val patient splitting.
- `BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData` as held-out test.
- Direct `.nii` image paths in every CSV row.
- Patient-level splitting before expanding each patient into four modality rows.

Expected CSV columns:

```text
patient_id,image_path,label,modality,split_source
```

## Check Runtime Dataloader

Install BrainIAC runtime dependencies first, including `monai==1.3.2` from `BrainIAC/requirements.txt`.

```bash
python scripts/check_brainiac_dataloader.py \
  --csv_path /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/train.csv \
  --batch_size 2
```

Expected batch shape:

```text
[B, 1, 96, 96, 96]
```

The dataset uses the BrainIAC runtime transform pattern:

```text
LoadImaged
EnsureChannelFirstd
Resized(spatial_size=(96, 96, 96), mode="trilinear")
NormalizeIntensityd(nonzero=True, channel_wise=True)
ToTensord
```

## Check BrainIAC Backbone

Run this only when the BrainIAC pretrained checkpoint is available locally:

Recommended local checkpoint location:

```text
/Users/kp/Documents/BRAINIAC/BrainIAC/checkpoints/BrainIAC.ckpt
```

Recommended Colab checkpoint location:

```text
/content/checkpoints/BrainIAC.ckpt
```

```bash
python scripts/check_brainiac_backbone.py \
  --checkpoint_path /Users/kp/Documents/BRAINIAC/BrainIAC/checkpoints/BrainIAC.ckpt \
  --csv_path /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/train.csv \
  --batch_size 2
```

Expected model input and feature shapes:

```text
input:    [B, 1, 96, 96, 96]
features: [B, 768]
```

The script imports `ViTBackboneNet` from `BrainIAC/src/model.py`, freezes all backbone parameters, runs exactly one `torch.no_grad()` forward pass, and does not save a model.

For later training scripts, use `shuffle=True` for the training `DataLoader` and `shuffle=False` for validation and test `DataLoader`s.

## Why No Offline Resampling

The local BraTS audit found all modality volumes are already `240 x 240 x 155` with `1.0 x 1.0 x 1.0 mm` spacing. Because BrainIAC runtime preprocessing resizes voxel grids to `96 x 96 x 96` but does not perform physical spacing resampling, this phase uses original BraTS files directly and avoids creating duplicate preprocessed NIfTI files.

## Why Not BrainIAC SequenceDataset

BrainIAC's existing `SequenceDataset` expects columns and paths shaped like:

```text
PatientID,SequenceLabel,ScanID,Sequence,Dataset
root_dir/{Dataset}/data/{PatientID}-{ScanID}-{Sequence}.nii.gz
```

BraTS uses patient folders with files like:

```text
{patient_id}_t1.nii
{patient_id}_t2.nii
{patient_id}_flair.nii
{patient_id}_t1ce.nii
```

Using `SequenceDataset` would require copying or renaming the dataset. The custom `BraTSSequenceDataset` instead reads direct `image_path` values.

## Colab Transfer Notes

In Colab, keep BrainIAC, the checkpoint, generated CSVs, and extracted BraTS files under `/content`. Use the same scripts with Colab paths, for example:

```text
/content/BrainIAC
/content/Dataset
/content/brats_sequence_project/outputs
```

The CSVs should contain direct paths valid for the active Colab runtime. Do not store or copy a second preprocessed BraTS dataset.
