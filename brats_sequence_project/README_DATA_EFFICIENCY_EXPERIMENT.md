# BrainIAC low-data/data-efficiency experiment

This experiment measures how much labeled MRI data is needed to fine-tune the
existing BrainIAC model for the four-way sequence task:

| Label | Sequence |
| ---: | --- |
| 0 | T1 |
| 1 | T2 |
| 2 | FLAIR |
| 3 | T1CE |

The existing full-data reference is 295 training patients (1,180 scans), 74
validation patients (296 scans), and 125 test patients (500 scans). Its verified
untouched-test result is 0.938 balanced accuracy and 0.938 ordinary accuracy.
That value is context only: this framework never hardcodes it into aggregation.

## Scientific design

The independent variable is the number of labeled training patients/scans. Each
patient contributes exactly four scans, one per modality, so patient-level
sampling is mandatory. Selecting scans independently could put different
modalities from one patient in different splits and would invalidate the
experiment.

The configured grid is:

| Training patients | Training scans | Seeds |
| ---: | ---: | --- |
| 1 | 4 | 42, 123, 456, 789, 2026 |
| 2 | 8 | 42, 123, 456, 789, 2026 |
| 5 | 20 | 42, 123, 456, 789, 2026 |
| 10 | 40 | 42, 123, 456, 789, 2026 |
| 25 | 100 | 42, 123, 456, 789, 2026 |
| 50 | 200 | 42, 123, 456 |
| 75 | 300 | 42, 123, 456 |
| 100 | 400 | 42, 123, 456 |
| 150 | 600 | 42 |
| 200 | 800 | 42 |
| 250 | 1000 | 42 |
| 295 | 1180 | 42 |

This is 38 planned runs. The editable configuration is
`finetune/config/data_efficiency.yml`.

For each seed, the runner sorts all training patient IDs, shuffles that list
once with the seed, and takes prefixes of it. Thus the 2-patient subset contains
the 1-patient subset, the 5-patient subset contains the 2-patient subset, and
so on. A different seed gets an independent ordering.

The validation CSV is always the existing full
`brats_sequence_project/outputs/val.csv` (296 scans), and the test CSV is always
the existing full `brats_sequence_project/outputs/test.csv` (500 scans). The
runner asserts those fixed sizes and checks that validation and test patients do
not overlap with one another or with a selected training subset. It also aborts
on incomplete patients, class imbalance, duplicate patient/modality rows, or
segmentation-mask inputs.

Every training run constructs BrainIAC from
`checkpoints/BrainIAC.ckpt` and creates a fresh four-class classifier. The
backbone remains trainable. The runner calls the existing
`finetune_model.py`; it does not copy or replace its model, transforms, loss,
optimizer, scheduler, batch size, epoch count, precision, or validation
checkpoint-selection logic. Checkpoint selection remains validation balanced
accuracy only. Test data is loaded only by `evaluate_checkpoint.py` after the
best checkpoint is fixed.

The paper's 10/20/40/60/80/100% comparison is related but not identical. This
study expands the low-data regime using explicit patient counts and nested
subsets; it does not claim that the requested counts exactly reproduce those
paper percentages.

## Files and result layout

New files:

- `finetune/config/data_efficiency.yml` — unchanged reproduction hyperparameters plus the patient/seed plan.
- `finetune/scripts/make_subset_manifest.py` — deterministic subset creation and assertions.
- `finetune/scripts/run_data_efficiency.py` — one-run orchestration and resumability.
- `finetune/scripts/aggregate_data_efficiency.py` — completed-run aggregation and optional plots.
- `finetune/tests/test_data_efficiency.py` — CPU-safe deterministic, integrity, resume, and aggregation checks.

The existing trainer has only a backward-compatible `--train-csv` and
`--val-csv` override so the selected manifest can be passed into the same
training implementation.

Each completed run is stored as:

```text
finetune/results/data_efficiency/
└── patients_005/
    └── seed_42/
        ├── subset_manifest.csv
        ├── subset_validation.json
        ├── run_metadata.json
        ├── training_history.json
        ├── training_history.csv
        ├── best_model.ckpt
        └── test_evaluation/
            ├── metrics.json
            ├── confusion_matrix.csv
            └── predictions.csv
```

`run_metadata.json` records the selected IDs, source split paths, fixed
validation/test paths, seed, checkpoint path, training settings, best epoch and
validation balanced accuracy, full-fine-tuning status, test-selection guard,
timestamp, and available git commit hash.

## Lab-server commands

Do not run production training on the Mac. On `bnac-lakeeffect-gpu`:

```bash
cd /path/to/BrainIAC_Testing
source ~/pytorch/bin/activate
```

One tiny real run (1 patient, seed 42):

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --patients 1 \
  --seed 42
```

One 5-patient run:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --patients 5 \
  --seed 42
```

Prepare and validate one manifest without training:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --patients 5 \
  --seed 42 \
  --prepare-only
```

Print one run's selected IDs and intended training/evaluation commands without
instantiating the model:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --patients 5 \
  --seed 42 \
  --dry-run
```

Prepare every configured manifest and print every intended command without
training:

```bash
python brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  --dry-run-matrix
```

There is intentionally no training-all command. Execute the printed or
per-run commands one at a time (or submit them through the lab's controlled
GPU scheduler), allowing the runner to resume each completed result safely.

The runner is resumable:

1. A valid `test_evaluation/metrics.json` causes the run to be skipped.
2. An existing `best_model.ckpt` without valid final metrics skips retraining and runs only evaluation.
3. An existing subset without a checkpoint trains that exact manifest.
4. With neither artifact, it prepares, trains, and evaluates.

Use `--train-only` to stop after checkpoint creation or `--evaluate-only` to
evaluate an existing checkpoint. `--force` explicitly permits retraining and
rewriting the run's generated checkpoint/history/evaluation artifacts; without
it, a valid completed result is never silently overwritten.

## Aggregation and plots

Aggregate all valid completed data-efficiency runs:

```bash
python brats_sequence_project/finetune/scripts/aggregate_data_efficiency.py
```

This writes:

- `finetune/results/data_efficiency/data_efficiency_summary.csv` — one row per patient count with mean, sample standard deviation, minimum, maximum, completed-seed count, mean ordinary accuracy, and mean best validation balanced accuracy.
- `finetune/results/data_efficiency/data_efficiency_all_runs.csv` — one row per patient-count/seed result.

If a compatible 295-patient result already exists under the data-efficiency
root or the broader `finetune/results/` tree, aggregation detects its measured
test balanced accuracy and adds `performance_retained_percent`. It does not
retrain or substitute the known 0.938 reference. If no valid baseline exists,
retention columns remain blank.

With an existing matplotlib installation, add plots:

```bash
python brats_sequence_project/finetune/scripts/aggregate_data_efficiency.py \
  --plot
```

This additionally writes `data_efficiency_curve.png` (training scans on the
x-axis) and `data_efficiency_curve_by_patients.png` (training patients on the
x-axis), with seed standard-deviation error bars.

## Verification and scientific guardrails

CPU-safe checks can run on the Mac:

```bash
python -m unittest discover \
  -s brats_sequence_project/finetune/tests \
  -p 'test_*.py' \
  -v
python -m py_compile \
  brats_sequence_project/finetune/scripts/finetune_common.py \
  brats_sequence_project/finetune/scripts/finetune_model.py \
  brats_sequence_project/finetune/scripts/make_subset_manifest.py \
  brats_sequence_project/finetune/scripts/run_data_efficiency.py \
  brats_sequence_project/finetune/scripts/aggregate_data_efficiency.py
```

Missing CUDA, PyTorch, MONAI, or the ignored checkpoint on the Mac is not a
failure of the orchestration framework. The runner intentionally refuses to
train without CUDA. No test accuracy may be used to select an epoch, change a
learning rate, alter preprocessing/augmentation, tune architecture, or change
training duration. The fixed test result is an outcome measurement only.

The lab environment should continue to use the verified `~/pytorch` virtual
environment, including MONAI 1.3.2; do not upgrade MONAI to work around a
checkpoint state-dict mismatch.
