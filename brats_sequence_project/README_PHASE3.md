# Phase 3: Frozen BrainIAC Classifier Training

Phase 3 trains only a small classifier head on top of frozen BrainIAC features for single-volume MRI sequence classification:

```text
0 = T1
1 = T2
2 = FLAIR
3 = T1CE
```

The BrainIAC backbone remains frozen. The scripts do not copy BraTS files, rename BraTS files, save preprocessed NIfTI files, or save full BrainIAC backbone weights.

## Design

Runtime path:

```text
original .nii file
-> BraTSSequenceDataset MONAI transforms
-> tensor [B, 1, 96, 96, 96]
-> frozen ViTBackboneNet
-> features [B, 768]
-> classifier head
-> logits [B, 4]
```

Classifier head:

```text
Linear(768, 256)
ReLU
Dropout(0.2)
Linear(256, 4)
```

Training uses `CrossEntropyLoss` and `AdamW` on classifier-head parameters only. The training `DataLoader` uses `shuffle=True`; validation and test loaders use `shuffle=False`.

## Checkpoint Locations

Local BrainIAC checkpoint:

```text
/Users/kp/Documents/BRAINIAC/BrainIAC/src/checkpoint/BrainIAC.ckpt
```

Colab BrainIAC checkpoint:

```text
/content/checkpoints/BrainIAC.ckpt
```

## Local Smoke Test

This runs one tiny epoch on small train/val CSV subsets and then verifies that evaluation can load the saved classifier head. It is a sanity check, not a real training run.

```bash
python scripts/run_phase3_smoke_test.py \
  --brainiac_src /Users/kp/Documents/BRAINIAC/BrainIAC/src \
  --checkpoint_path /Users/kp/Documents/BRAINIAC/BrainIAC/src/checkpoint/BrainIAC.ckpt \
  --train_csv /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/train.csv \
  --val_csv /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/val.csv \
  --output_dir /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/phase3_smoke \
  --batch_size 2 \
  --num_workers 0
```

## Local Training Command

Use this only on a local GPU runtime. Prefer Colab for real training.

```bash
python scripts/train_frozen_brainiac_classifier.py \
  --brainiac_src /Users/kp/Documents/BRAINIAC/BrainIAC/src \
  --checkpoint_path /Users/kp/Documents/BRAINIAC/BrainIAC/src/checkpoint/BrainIAC.ckpt \
  --train_csv /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/train.csv \
  --val_csv /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/val.csv \
  --output_dir /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/phase3_frozen_classifier \
  --epochs 20 \
  --batch_size 4 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --num_workers 2 \
  --seed 42 \
  --use_amp \
  --patience 5
```

## Full Colab Training Command

Use Colab GPU for real training:

```bash
python scripts/train_frozen_brainiac_classifier.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --train_csv /content/brats_sequence_project/outputs/train.csv \
  --val_csv /content/brats_sequence_project/outputs/val.csv \
  --output_dir /content/brats_sequence_project/outputs/phase3_frozen_classifier \
  --epochs 20 \
  --batch_size 4 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --num_workers 2 \
  --seed 42 \
  --use_amp \
  --patience 5
```

If GPU memory is limited, reduce `--batch_size` to `2`. Keep the backbone frozen for this phase.

## Test Evaluation Command

After training, evaluate the best classifier head on `test.csv`. This writes metrics, predictions, and evaluation plots:

```bash
python scripts/evaluate_frozen_brainiac_classifier.py \
  --brainiac_src /content/BrainIAC/src \
  --checkpoint_path /content/checkpoints/BrainIAC.ckpt \
  --classifier_path /content/brats_sequence_project/outputs/phase3_frozen_classifier/best_classifier_head.pt \
  --csv_path /content/brats_sequence_project/outputs/test.csv \
  --output_dir /content/brats_sequence_project/outputs/phase3_test_eval \
  --batch_size 4 \
  --num_workers 2
```

Local equivalent:

```bash
python scripts/evaluate_frozen_brainiac_classifier.py \
  --brainiac_src /Users/kp/Documents/BRAINIAC/BrainIAC/src \
  --checkpoint_path /Users/kp/Documents/BRAINIAC/BrainIAC/src/checkpoint/BrainIAC.ckpt \
  --classifier_path /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/phase3_frozen_classifier/best_classifier_head.pt \
  --csv_path /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/test.csv \
  --output_dir /Users/kp/Documents/BRAINIAC/BrainIAC/brats_sequence_project/outputs/phase3_test_eval \
  --batch_size 4 \
  --num_workers 2
```

## Regenerate Plots Without Retraining

Use this if training/evaluation already finished and only the figures need to be rebuilt:

```bash
python scripts/plot_phase3_results.py \
  --history_csv /content/brats_sequence_project/outputs/phase3_frozen_classifier/train_history.csv \
  --predictions_csv /content/brats_sequence_project/outputs/phase3_test_eval/predictions.csv \
  --metrics_json /content/brats_sequence_project/outputs/phase3_test_eval/test_metrics.json \
  --output_dir /content/brats_sequence_project/outputs/phase3_plots_regenerated \
  --split_name test
```

## Outputs

Training writes:

```text
best_classifier_head.pt
train_history.csv
best_metrics.json
val_predictions.csv
plots/loss_curve.png
plots/accuracy_curve.png
plots/f1_curve.png
plots/precision_recall_curve_over_epochs.png
plots/learning_rate_curve.png
plots/overfitting_gap_curve.png
plots/combined_training_summary.png
```

`best_classifier_head.pt` contains:

```text
classifier_state_dict
classifier_config
epoch
label_mapping
id_to_label
val_metrics
train_args
```

It does not contain BrainIAC backbone weights.

Only the classifier head is saved because the BrainIAC checkpoint is a fixed external input for this phase. This keeps outputs small and prevents accidentally versioning or duplicating the pretrained backbone.

`train_history.csv` contains per-epoch loss, accuracy, macro/weighted F1, macro precision/recall, learning rate, and epoch runtime.

Evaluation writes:

```text
test_metrics.json
predictions.csv
plots/confusion_matrix_raw.png
plots/confusion_matrix_normalized.png
plots/per_class_precision_recall_f1.png
plots/per_class_accuracy.png
plots/class_distribution_true.png
plots/class_distribution_predicted.png
plots/confidence_histogram.png
plots/confidence_correct_vs_incorrect.png
plots/prediction_error_by_class.png
plots/roc_curves_multiclass.png
plots/precision_recall_curves_multiclass.png
plots/top_confident_wrong_predictions.csv
plots/top_uncertain_predictions.csv
```

`predictions.csv` columns:

```text
patient_id,image_path,modality,true_label,pred_label,correct,probability_T1,probability_T2,probability_FLAIR,probability_T1CE,confidence
```

`test_metrics.json` contains accuracy, macro F1, weighted F1, macro precision, macro recall, per-class metrics, raw and normalized confusion matrices, ROC AUC values when computable, average precision values when computable, and total/correct/incorrect sample counts.

## Plot Guide

Most important professor/presentation plots:

1. `loss_curve.png` — shows whether optimization is stable and whether validation loss tracks training loss.
2. `accuracy_curve.png` — shows train/validation classification accuracy over epochs.
3. `f1_curve.png` — shows macro and weighted F1, which is more informative than accuracy if classes become imbalanced.
4. `confusion_matrix_normalized.png` — shows per-class recall and which sequences are confused.
5. `per_class_precision_recall_f1.png` — summarizes class-specific strengths and weaknesses.
6. `confidence_correct_vs_incorrect.png` — shows whether wrong predictions are low-confidence or confidently wrong.

Additional plots:

- `precision_recall_curve_over_epochs.png` tracks macro precision and recall during training.
- `learning_rate_curve.png` confirms the optimizer learning rate used each epoch.
- `overfitting_gap_curve.png` visualizes train-validation metric gaps.
- `combined_training_summary.png` provides a single slide-friendly 2x2 summary.
- `confusion_matrix_raw.png` shows raw mistake counts.
- `per_class_accuracy.png` shows class-wise accuracy.
- `class_distribution_true.png` and `class_distribution_predicted.png` show label balance and prediction bias.
- `confidence_histogram.png` shows overall model confidence distribution.
- `prediction_error_by_class.png` counts mistakes by true class.
- `roc_curves_multiclass.png` and `precision_recall_curves_multiclass.png` show one-vs-rest threshold behavior when all required classes are present.
- `top_confident_wrong_predictions.csv` lists the highest-confidence mistakes for qualitative review.
- `top_uncertain_predictions.csv` lists the least-confident predictions for error analysis.

## Why Patient-Level Splits

Each BraTS patient contributes four modality samples. Splitting by image would leak patient anatomy and tumor information across train/validation/test. Phase 3 uses the existing patient-level CSVs from Phase 2.

## Why No Offline Preprocessing

The local audit confirmed this BraTS copy is already co-shaped with `1.0 x 1.0 x 1.0 mm` spacing. Runtime preprocessing uses BrainIAC-compatible loading, channel-first conversion, resize to `96 x 96 x 96`, nonzero intensity normalization, and tensor conversion. There is no reason in Phase 3 to save duplicate preprocessed NIfTI files.
