# BrainIAC BraTS Visual Debugging

These tools make the preprocessing and classifier outputs inspectable without changing the dataset, training models, or writing full preprocessed NIfTI copies. They are intended for small batches and presentation-friendly figures under `outputs/visual_debug/`.

## Why Visualize

The best current preprocessing variant, `crop_pad_zscore`, improved validation accuracy and macro F1, but it changes the spatial handling compared with `resize_zscore`. Visual checks answer the practical questions:

- Does the model receive recognizable axial, sagittal, and coronal brain views?
- Does `crop_pad_zscore` clip anatomy or tumor context for any sequence?
- Are wrong predictions low-confidence ambiguity cases or high-confidence failures?
- Do probability outputs show consistent class confusion patterns?

## Commands

Run commands from `brats_sequence_project/`.

Visualize one training batch:

```bash
python scripts/visualize_preprocessed_batch.py \
  --csv_path outputs/train.csv \
  --variant crop_pad_zscore \
  --batch_size 20 \
  --split_name train \
  --output_dir outputs/visual_debug/train_batch_crop
```

Compare resize vs crop/pad on the same patient:

```bash
python scripts/compare_preprocessing_visuals.py \
  --csv_path outputs/train.csv \
  --variants resize_zscore crop_pad_zscore \
  --output_dir outputs/visual_debug/compare_resize_vs_crop
```

Inspect a validation batch with predictions:

```bash
python scripts/visualize_batch_with_predictions.py \
  --csv_path outputs/val.csv \
  --predictions_csv outputs/preprocessing_ablation/crop_pad_zscore/classifier/val_predictions.csv \
  --variant crop_pad_zscore \
  --batch_size 20 \
  --output_dir outputs/visual_debug/val_batch_predictions
```

Inspect wrong, correct, and uncertain prediction examples:

```bash
python scripts/visualize_prediction_errors.py \
  --csv_path outputs/test.csv \
  --predictions_csv outputs/preprocessing_ablation/crop_pad_zscore/classifier/predictions.csv \
  --variant crop_pad_zscore \
  --output_dir outputs/visual_debug/error_analysis
```

Summarize prediction confidence and probability distributions:

```bash
python scripts/inspect_validation_outputs.py \
  --predictions_csv outputs/preprocessing_ablation/crop_pad_zscore/classifier/val_predictions.csv \
  --output_dir outputs/visual_debug/validation_output_inspection
```

Generate the BrainIAC alignment report:

```bash
python scripts/audit_brainiac_paper_repo_alignment.py
```

Compile all scripts:

```bash
python -m py_compile \
  scripts/visualize_preprocessed_batch.py \
  scripts/compare_preprocessing_visuals.py \
  scripts/visualize_prediction_errors.py \
  scripts/inspect_validation_outputs.py \
  scripts/visualize_batch_with_predictions.py \
  scripts/audit_brainiac_paper_repo_alignment.py
```

## Outputs And Interpretation

`batch_train_crop_pad_zscore_20cases.png` shows one batch with rows as cases and columns as axial, sagittal, and coronal middle slices. Row labels include patient ID, modality, label, and tensor min/max/mean/std.

`batch_train_crop_pad_zscore_axial_grid.png` is the compact 4 x 5 view for a slide. Use it to show the batch-level input distribution quickly.

`compare_variants_axial.png`, `compare_variants_sagittal.png`, and `compare_variants_coronal.png` compare the same patient across preprocessing variants. If crop/pad is cutting away anatomy, this is where it should be visible.

`compare_<patient_id>_<modality>_all_views.png` is the modality-level detailed figure. Use it when a single sequence looks suspicious.

`validation_batch_predictions_crop_pad_zscore.png` overlays true label, predicted label, confidence, and correctness on a validation/test batch.

Use `classifier/val_predictions.csv` for validation plots and `classifier/predictions.csv` for the held-out test split.

`wrong_predictions/`, `correct_high_confidence/`, and `uncertain_predictions/` contain case-by-case figures with original and preprocessed views. These are the main qualitative error-analysis figures.

`wrong_predictions_grid.png`, `uncertain_predictions_grid.png`, and `correct_predictions_grid.png` are slide-friendly contact sheets.

`inspect_validation_outputs.py` creates confidence histograms, class-wise confidence boxplots, probability distribution plots, and CSVs for the top wrong and uncertain cases.

## Reading Wrong Prediction Examples

For each wrong prediction, first compare the original and preprocessed rows. If the preprocessed row lost obvious anatomy, the error may be preprocessing-driven. If the image looks intact but confidence is high, the classifier may have learned a misleading feature. If confidence is low and probabilities are split between plausible sequence classes, the case is more likely an ambiguous boundary case.

The probability line reports `T1`, `T2`, `FLAIR`, and `T1CE` probabilities in that order. Look for repeated confusion pairs, such as T1 vs T1CE or T2 vs FLAIR.

## What To Show In PPT

- One compact training batch axial grid to prove exactly what enters BrainIAC.
- One resize vs crop/pad comparison patient across all four modalities.
- Two or three high-confidence wrong predictions with original vs preprocessed views.
- The confidence correct-vs-incorrect plot.
- The probability distribution by true class plot.
- One sentence from `brainiac_alignment_report.md` explaining that our current pipeline is frozen-feature linear/MLP classification, while BrainIAC also supports fine-tuning.
