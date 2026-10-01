# BrainIAC low-data/data-efficiency experiment

This experiment measures how much labeled MRI data is needed to fine-tune the
existing BrainIAC model for the four-way sequence task:

| Label | Sequence |
| ---: | --- |
| 0 | T1 |
| 1 | T2 |
| 2 | FLAIR |
| 3 | T1CE |

Each patient contributes exactly four scans. The fixed validation cohort is 74
patients and 296 scans. The fixed test cohort is 125 patients and 500 scans.
Every final model is evaluated on that same full test cohort, including models
trained on only four scans.

The existing 295-patient/1,180-scan full-data run is a separate reference. When
a compatible saved result is available, aggregation reads its measured metrics
and does not retrain it or hardcode a value.

## Scientific design

The configured low-data grid uses two deterministic patient-subset seeds:
42 and 123. Two repeated subsets provide a compute-conscious estimate of
sensitivity to which patients are selected. They are not a rigorous uncertainty
estimate and should not be presented as one.

| Training patients | Training scans | Seeds |
| ---: | ---: | --- |
| 1 | 4 | 42, 123 |
| 2 | 8 | 42, 123 |
| 5 | 20 | 42, 123 |
| 10 | 40 | 42, 123 |
| 25 | 100 | 42, 123 |
| 50 | 200 | 42, 123 |
| 75 | 300 | 42, 123 |
| 100 | 400 | 42, 123 |
| 150 | 600 | 42, 123 |
| 200 | 800 | 42, 123 |
| 250 | 1,000 | 42, 123 |

There are 22 configured low-data candidate runs. The existing
`patients_001/seed_42` result is one of them, so 21 new low-data runs are
needed when that compatible result is present.

For each seed, the runner sorts all training patient IDs, shuffles that list
once with the seed, and takes prefixes of it. Therefore the 1-patient subset
is a prefix of the 2-patient subset, which is a prefix of the 5-patient subset,
and so on. The two seeds use separate deterministic orderings. The runner
rejects patient/seed pairs that are not in the configured plan.

The runner calls the existing `finetune_model.py` and
`evaluate_checkpoint.py`. It does not change BrainIAC, the classifier,
optimizer, learning rate, scheduler, epoch count, preprocessing,
augmentations, checkpoint selection, validation set, or test set. Every
training run starts from the general `checkpoints/BrainIAC.ckpt`, and
checkpoint selection uses validation balanced accuracy only. Test metrics are
outcome measurements and never tune a later run.

## Files and result layout

- `finetune/config/data_efficiency.yml` — fixed training contract and the
  11-count/two-seed plan.
- `finetune/scripts/make_subset_manifest.py` — deterministic nested subset
  creation and split assertions.
- `finetune/scripts/run_data_efficiency.py` — single-candidate and sequential
  matrix orchestration with resume behavior.
- `finetune/scripts/aggregate_data_efficiency.py` — CSV aggregation, baseline
  reuse, and plotting.
- `finetune/tests/test_data_efficiency.py` — CPU-safe orchestration,
  aggregation, and plotting tests.

Each candidate is stored under a directory such as:

```text
finetune/results/data_efficiency/
└── patients_005/
    └── seed_42/
        ├── subset_manifest.csv
        ├── subset_validation.json
        ├── run_metadata.json
        ├── training_history.json
        ├── training_history.csv
        ├── loss_curve.png
        ├── validation_balanced_accuracy_curve.png
        ├── confusion_matrix.png
        ├── best_model.ckpt
        └── test_evaluation/
            ├── metrics.json
            ├── confusion_matrix.csv
            └── predictions.csv
```

The per-run plots are generated from the saved training history and final test
metrics. Plotting is best effort: if matplotlib or a required plot input is
unavailable, the runner prints a warning and preserves the numerical outputs.

## Lab-server commands

Do not run production training on the Mac. On `bnac-lakeeffect-gpu`:

```bash
cd /path/to/BrainIAC_Testing
source ~/pytorch/bin/activate
```

Dry-run one candidate:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --patients 5 \
  --seed 42 \
  --dry-run
```

Run one candidate:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --patients 5 \
  --seed 42
```

Prepare one manifest without training:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --patients 5 \
  --seed 42 \
  --prepare-only
```

Dry-run the complete configured matrix without training or evaluation:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --dry-run-matrix
```

Run the complete matrix sequentially:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --run-matrix
```

The same `--run-matrix` command is the restart command after an interruption.
Candidates run in this order: each patient count in the table above, then seed
42 followed by seed 123. The runner prints a compact progress table and stops
with a nonzero error at the first unexpected failure. Completed results remain
available for the next restart.

Resume behavior is artifact-based:

1. A valid `test_evaluation/metrics.json` is skipped without rewriting the
   completed scientific result.
2. An existing `best_model.ckpt` without valid final metrics skips training and
   runs evaluation only.
3. An existing valid subset manifest without a checkpoint trains using that
   exact manifest.
4. With neither artifact, the runner prepares the nested subset, trains, and
   evaluates.

The fixed validation CSV remains 296 scans and the fixed test CSV remains 500
scans in every mode. `--dry-run` and `--dry-run-matrix` print intended commands
without instantiating the model or starting scientific processes.

## Aggregation and plots

`--run-matrix` automatically aggregates after every configured candidate has
completed. Aggregation can also be run independently:

```bash
python brats_sequence_project/finetune/scripts/aggregate_data_efficiency.py
```

The aggregate CSV outputs are:

- `data_efficiency_all_runs.csv` — one row per completed patient-count/seed
  result, plus the compatible full-data baseline when available. It includes
  `patient_count`, `training_scans`, `seed`,
  `best_validation_balanced_accuracy`, `best_epoch`,
  `test_balanced_accuracy`, and `test_accuracy`.
- `data_efficiency_summary.csv` — one row per completed training size. It
  includes `patient_count`, `training_scans`, `completed_seed_count`, mean
  test balanced accuracy, sample standard deviation, minimum, maximum, mean
  ordinary accuracy, and mean best validation balanced accuracy.

The aggregate plot outputs are:

- `data_efficiency_curve.png` — mean test balanced accuracy versus training
  scans, with standard-deviation error bars where available.
- `data_efficiency_individual_seeds.png` — separate seed-42 and seed-123
  curves, with the one full-data baseline shown separately when available.
- `performance_retained.png` — mean test balanced accuracy as a percentage of
  the compatible full-data baseline; generated only when that baseline is
  verified.
- `data_efficiency_accuracy_curve.png` — secondary mean ordinary test
  accuracy curve.
- `data_efficiency_curve_by_patients.png` — retained compatibility view with
  training patients on the x-axis.

Each completed candidate also receives:

- `loss_curve.png` — training and validation loss by epoch.
- `validation_balanced_accuracy_curve.png` — validation balanced accuracy by
  epoch with the best validation epoch marked.
- `confusion_matrix.png` — final 500-scan test confusion matrix with T1, T2,
  FLAIR, and T1CE labels.

## Fine-tuning input visualization

Use the input visualization utility to make the data path reviewable at three
stages: the source NIfTI, the existing deterministic load/channel-first/resize/
normalization path, and an actual augmented batch yielded by the training
Dataset and DataLoader. The deterministic display is composed from the actual
training transform's existing objects up to its first random transform, with
the same tensor conversion. The final batch uses the shared seeded loader
constructor from `finetune_model.py`; no model or checkpoint is loaded.

On the lab server, visualize all four scans in the 1-patient/seed-42 manifest:

```bash
python brats_sequence_project/finetune/scripts/visualize_model_input.py \
  --config brats_sequence_project/finetune/config/data_efficiency.yml \
  --train-csv brats_sequence_project/finetune/results/data_efficiency/patients_001/seed_42/subset_manifest.csv \
  --seed 42 \
  --output-dir brats_sequence_project/finetune/results/data_efficiency/patients_001/seed_42/input_visualization
```

For a larger manifest, the utility still loads one actual training batch by
default; use `--num-samples 8` to limit the number drawn in figures. The batch
metadata retains every item from that batch. `--batch-index 1` selects the next
shuffled batch. Set the seed to reproduce the selected order and random
transform realization. Random augmentation is part of the final model-input
stage and can differ between seeds or runs.

The output directory contains `raw_triplanar.png`,
`preprocessed_triplanar.png`, `actual_training_batch.png`,
`actual_training_batch_triplanar.png`, `raw_vs_preprocessed.png`,
`raw_vs_final_batch.png`, `preprocessing_statistics.csv`,
`batch_metadata.json`, and `pipeline_summary.txt`. Raw NIfTI plotting changes
axis order/flips for closest-RAS display only; it does not resample or alter
voxel values. Per-slice percentile windows affect display only, while the CSV
reports unwindowed values.

No second full-data value is fabricated when only one compatible baseline run
exists. Retention is omitted when the compatibility checks cannot verify a
295-patient/1,180-scan baseline.

## Verification and scientific guardrails

CPU-safe checks can run on the Mac:

```bash
python -m unittest brats_sequence_project/finetune/tests/test_data_efficiency.py -v
python -m py_compile \
  brats_sequence_project/finetune/scripts/finetune_common.py \
  brats_sequence_project/finetune/scripts/finetune_model.py \
  brats_sequence_project/finetune/scripts/make_subset_manifest.py \
  brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  brats_sequence_project/finetune/scripts/aggregate_data_efficiency.py
```

These tests do not run GPU training or evaluate real checkpoints. Missing CUDA,
MONAI, or the ignored server checkpoint on the Mac is not a failure of the
orchestration tests. The lab environment should continue to use the verified
`~/pytorch` environment, including MONAI 1.3.2.
