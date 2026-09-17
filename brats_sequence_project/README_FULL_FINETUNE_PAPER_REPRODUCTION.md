# BrainIAC full end-to-end fine-tuning reproduction

This experiment evaluates the general BrainIAC foundation model on four-way
MRI sequence classification using BraTS2020. It is isolated under
`brats_sequence_project/finetune/`; the existing frozen-feature experiments
and their outputs are preserved.

The scientific question is whether the general pretrained BrainIAC SimCLR
ViT-B representation can distinguish T1, T2, FLAIR, and T1CE volumes after
full end-to-end adaptation. This is a reproduction-oriented engineering
pipeline, not a hyperparameter search.

Primary references:

- [Final Nature Neuroscience paper](https://www.nature.com/articles/s41593-026-02202-6)
- [Published full text (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13061609/)
- [Official BrainIAC sequence-classification documentation](https://github.com/AIM-KannLab/BrainIAC/blob/main/docs/downstream_tasks/MR_sequence_classification.md)
- [Official released model implementation](https://github.com/AIM-KannLab/BrainIAC/blob/main/src/model.py)
- [Official released downstream data transforms](https://github.com/AIM-KannLab/BrainIAC/blob/main/src/dataset.py)
- [Official released offline MRI preprocessing utility](https://github.com/AIM-KannLab/BrainIAC/blob/main/src/preprocessing/mri_preprocess_3d_simple.py)
- [Official BraTS2020 data/preprocessing description](https://www.med.upenn.edu/cbica/brats2020/data.html)
- [Official BrainIAC repository](https://github.com/AIM-KannLab/BrainIAC)
- [Supplementary methods PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fs41593-026-02202-6/MediaObjects/41593_2026_2202_MOESM1_ESM.pdf)

## Important guardrails

- Initialization is always from `checkpoints/BrainIAC.ckpt`, the general
  foundation checkpoint. `sequence_classification.ckpt` and other downstream
  checkpoints are rejected by the new runner.
- The BrainIAC ViT-B encoder and the four-class head are both trainable. The
  runner aborts if any model parameter is frozen and prints total, trainable,
  and frozen parameter counts.
- The training script never loads the test CSV. Checkpoint selection uses
  validation balanced accuracy only. `evaluate_checkpoint.py` performs one
  forward pass over the untouched test CSV after a checkpoint has been chosen.
- No GPU training has been run in this repository. The training entry point
  intentionally refuses to start without CUDA.
- Segmentation masks are rejected at CSV validation time.

## Task and labels

The class mapping is fixed and must not be changed:

| Label | Sequence |
| ---: | --- |
| 0 | T1 |
| 1 | T2 |
| 2 | FLAIR |
| 3 | T1CE |

Each patient contributes all four sequences. The input is one NIfTI volume at
a time; labels are sequence labels, not segmentation labels.

## Model initialization

The new runner reuses the released final-model implementation in `src/model.py`:

1. `ViTBackboneNet` loads `checkpoints/BrainIAC.ckpt` on CPU using the
   released checkpoint-loading convention.
2. The backbone is the SimCLR ViT-B configured for a one-channel `96 x 96 x
   96` volume, hidden size 768, 12 layers, 12 heads, patch size 16, and MLP
   dimension 3072.
3. A new `Classifier(d_model=768, num_classes=4)` is instantiated after the
   backbone load, so its linear weights are randomly initialized.
4. `SingleScanModel` applies the released dropout (`p=0.2`) before the linear
   classifier.
5. The code verifies `classifier.fc.out_features == 4` and that every
   backbone parameter has `requires_grad=True`. It also verifies that no
   head parameter is frozen.

The supplement describes the sequence-classification head generically as a
2048-dimensional latent representation followed by a matching fully connected
layer and a four-neuron output layer. That description conflicts with the
released final SimCLR ViT-B code, whose selected encoder output is 768 and
whose task head is the direct `Linear(768, 4)` used here. We record the
conflict rather than silently inventing a 2048-dimensional adapter.

## Data split

The existing patient-level split is retained:

- `BraTS2020_TrainingData`: deterministic seed-42 split into train (80%) and
  validation (20%).
- `BraTS2020_ValidationData`: test split.
- All four sequences for one patient remain in one split.
- There must be zero patient overlap among train, validation, and test.

The current audited split contains:

| Split | Patients | Images | T1 | T2 | FLAIR | T1CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 295 | 1,180 | 295 | 295 | 295 | 295 |
| Validation | 74 | 296 | 74 | 74 | 74 | 74 |
| Test | 125 | 500 | 125 | 125 | 125 | 125 |

Run `validate_splits.py` to regenerate a detailed JSON summary with these
counts, class/modality counts, incomplete-patient checks, segmentation-input
checks, and every pairwise overlap check. The summary is generated under the
ignored `brats_sequence_project/finetune/results/` directory.

This split is intentionally different from the published cohort. The paper
uses BraTS2023 with 5,880 scans, a development/training cohort of 5,004 scans,
and an independent holdout of 876 scans. We use BraTS2020 and preserve the
repository's requested source split instead of changing cohorts or inventing a
new split.

## Preprocessing audit

The project audit found that the BraTS2020 volumes are already 1 x 1 x 1 mm
and 240 x 240 x 155. The released downstream runtime does not physically
resample them; it resizes the loaded volume to 96 x 96 x 96 and applies
nonzero, channel-wise intensity normalization.

The new runner imports the released transforms from `src/dataset.py` rather
than using the old frozen-feature pipeline:

- Training: `LoadImaged` -> `EnsureChannelFirstd` -> `Resized((96,96,96),
  trilinear)` -> `NormalizeIntensityd(nonzero=True, channel_wise=True)` ->
  the released training augmentations -> `ToTensord`.
- Validation/test: the same load/channel/resize/normalization sequence without
  training augmentations.

### Paper vs official code vs this implementation

| Setting | PAPER SAYS | OFFICIAL CODE DOES | OUR IMPLEMENTATION | UNAVOIDABLE DIFFERENCE |
| --- | --- | --- | --- | --- |
| Input preparation | The published methods describe conversion to NIfTI, N4 bias correction, isotropic 1 mm processing, rigid MNI registration, and HD-BET skull stripping. | The downstream runtime transform in `src/dataset.py` performs load, channel-first conversion, resize to 96 cubed, nonzero channel-wise normalization, then augmentations for training. It does not call the heavier offline preprocessing in the runtime path. | Uses the released runtime transform exactly; no physical resampling, registration, skull stripping, or mask input is added to the BraTS CSV path. | BraTS2020 is a different prepared NIfTI cohort from the paper's BraTS2023 data; raw DICOM-to-paper preprocessing cannot be reconstructed from these CSV inputs without changing the experiment. |
| N4 provenance | The paper states that N4 bias-field correction was part of the global MRI preprocessing pipeline. It does not publish a BraTS2023 subject-level processing manifest or N4 parameter table. | The released offline utility calls `sitk.N4BiasFieldCorrection(moving_img)` before its registration/resampling and HD-BET stages. The downstream `src/dataset.py` loader does not call N4. | Leaves the supplied BraTS2020 NIfTI files unchanged and uses the released runtime transform. N4 is not omitted because the files are NIfTI; NIfTI is a valid input to N4. It is omitted because the BraTS2020 release does not document N4 provenance and the exact paper-cohort execution trace is unavailable. | Whether the supplied BraTS2020 files already received N4 is not established. Re-running the complete offline utility would unnecessarily re-register, resample, and skull-strip already prepared data. |
| Encoder/head | Supplement A.2.1 describes a 2048-dimensional latent, matching fully connected layer, and four-neuron output. | Final released SimCLR ViT-B code emits 768-dimensional CLS features and uses `Classifier(768, 4)` with dropout. | Uses the released final ViT-B architecture: 768 CLS -> dropout 0.2 -> fresh `Linear(768,4)`. | The generic supplement head description and released final-model code are inconsistent; the code-supported ViT-B path is the closest reproducible interpretation. |
| Epochs | 100 epochs. | Current config/docs say 200 epochs. | 100 epochs. | Current docs/config are stale relative to the final supplement. |
| Batch size | 16. | Current config says 64; docs say 32; the released validation loader default is 1. | Training batch size 16; validation loader batch size 1 to retain the released loader default. | Validation batch size is not specified by the paper; it does not change deterministic evaluation metrics. |
| Optimizer/LR | Adam; BrainIAC initialization uses LR 0.0001. | Script uses Adam; current config says 0.0008 and includes an unused `momentum: 0.9` field. | Adam, LR 0.0001. The unused momentum field is not passed to Adam. | The final supplement supersedes the stale config LR. |
| Weight decay | Not specified in the sequence-classification supplement. | Current config/docs use 0.0001. | Adam weight decay 0.0001, retained from the released config/documentation because the paper omits it. | Exact final-paper weight decay cannot be independently recovered from the published methods. |
| Scheduler | ReduceLROnPlateau. | Script hardcodes CosineAnnealingWarmRestarts (`T_0=50`, `T_mult=2`). | ReduceLROnPlateau, mode `max`, monitoring validation balanced accuracy. Factor 0.1 and patience 10 are PyTorch defaults because the supplement omits them. | Scheduler identity is a direct paper/code discrepancy; factor/patience remain an explicitly documented omission. |
| Precision | Not stated in the supplement. | Lightning defaults to `16-mixed` in the released training script. | `16-mixed` on CUDA, as the released-code default. The script refuses to train without CUDA. | The paper does not provide enough information to prove the original precision setting. |
| Checkpoint selection | Best checkpoint based on validation performance; balanced accuracy is the primary task metric. | Script monitors `val_auc`, not balanced accuracy. | Selects the maximum validation balanced accuracy and never reads test data during training. | Metric selection follows the paper's stated primary metric and the experiment requirement, not the stale AUC monitor. |

The source files requested for comparison (`src/train_lightning_multiclass.py`,
`src/model.py`, `src/dataset.py`, `src/config_finetune.yml`,
`src/test_inference_finetune.py`, and the official task documentation) match
the corresponding files in the official `AIM-KannLab/BrainIAC` `main` branch.
The new code is therefore additive and does not alter those frozen/official
files.

There is also a label-indexing discrepancy: the official documentation's CSV
examples show labels 1 through 4, while the released `SequenceDataset`
subtracts 1 before returning a target. The existing project split CSVs and this
runner use the explicit zero-based mapping in the task requirement (0 through
3), so no hidden subtraction is performed.

### Final N4 bias-field audit

This audit separates what is documented about the released data from what is
documented about the BrainIAC authors' processing pipeline:

| Question | Audit conclusion |
| --- | --- |
| Was BraTS2020 officially N4 bias corrected? | **Not documented; status UNKNOWN.** The official BraTS2020 release description explicitly documents NIfTI delivery, common-template co-registration, 1 mm³ resolution, and skull stripping, but does not identify N4 bias correction. The local header audit confirms the expected shape/spacing for the available BraTS MRI files; NIfTI headers and the empty `descrip` field do not provide a reliable record of whether N4 was run. |
| Did the authors apply N4 to the sequence-classification cohort? | **Yes according to the paper's stated global methodology:** the methods say that all MRI scans passed through N4 as part of preprocessing, and sequence classification used the BraTS2023 cohort. **The execution trace is not independently verifiable:** no BraTS2023 processed-file manifest or per-cohort N4 log is released, and the downstream loader itself does not run N4. |
| Can the exact N4 procedure/parameters be recovered? | **Only partially.** The released offline script exposes the call `sitk.N4BiasFieldCorrection(moving_img)` with no explicit N4 parameters. Thus the SimpleITK defaults are implicit, and the released files do not identify the exact SimpleITK version or prove that this utility generated the sequence-classification inputs. The complete paper preprocessing execution cannot be reproduced exactly from the release. |
| Is N4 required for this BraTS2020 run? | **No additional N4 step is inserted now.** The supplied BraTS2020 files are already NIfTI, 1 mm, co-registered, and skull stripped, but that fact does not by itself answer their N4 provenance. Adding N4 now would be a new, unverified intervention rather than a recoverable reproduction step. |

If N4 is later mandated by a separately approved preprocessing variant, the
controlled operation should be: read each MRI image (never a segmentation
mask) with the exact lab-pinned SimpleITK version, apply the released call
`sitk.N4BiasFieldCorrection(image)` using the official utility's implicit
defaults, preserve the original geometry, write to a separate derived data
tree, and record the software version and command. Do **not** run the complete
`mri_preprocess_3d_simple.py` utility on these inputs because that utility also
resamples, registers, and invokes HD-BET. Since the paper does not publish the
full N4 parameter/version/output provenance, this would remain a documented
methodology deviation, not an exact reconstruction. No such preprocessing has
been run.

The unavoidable preprocessing deviation is therefore explicit: the current
experiment uses the official downstream resize/normalization path on the
already prepared BraTS2020 NIfTIs, while the paper describes a global N4-based
pipeline for its BraTS2023 cohort. The raw DICOM provenance and exact execution
trace are unavailable from these CSV/NIfTI inputs; that is an auditability
limitation, not a limitation imposed by the NIfTI format.

### Final BrainIAC pretraining-contamination audit

The paper and supplement distinguish the overall curated data pool from the
foundation-pretraining subset:

- The paper describes 48,965 scans from 34 datasets overall, including
  BraTS2023. Supplementary Data Table 4 states that foundation pretraining used
  32,015 images from 16 datasets. Those 16 listed datasets are ABCD, ADNI,
  DFCI/BCH LGG, OASIS-3, MCSA, SOOP, ABIDE, CBTN LGG, MIRIAD, PPMI, DLBS,
  RadART LGG, OASIS-2, DFCI/BCH HGG, QIN-GBM, and RIDER. BraTS2020 and BraTS2023
  are not listed in that explicit 16-dataset pretraining table.
- BraTS2023 is explicitly the downstream sequence-classification cohort:
  5,880 balanced scans in the available pool, with 5,004 development/fine-
  tuning scans and 876 reserved holdout scans. This is not the BraTS2020
  cohort used here.
- The official BrainIAC repository contains code, example data, and sample
  CSVs, but no released subject-level foundation-pretraining manifest or
  BrainIAC split CSV that can be joined to our IDs. The official BraTS2020
  documentation describes historical BraTS20-to-TCGA source naming mappings,
  but that is not a mapping to BrainIAC pretraining inputs.

Therefore the current subject-level conclusion is **UNKNOWN**:

1. No supplied BraTS2020 patient can be proven to have been seen by BrainIAC
   pretraining from the released dataset-level tables alone.
2. No supplied BraTS2020 patient can be proven not to have been seen, because
   the pretraining subject manifest and any alias/cross-dataset mapping are not
   released.
3. The absence of BraTS2020 from the explicit 16-dataset pretraining table is
   evidence against intentional inclusion as a named pretraining dataset, but
   it is not a subject-level decontamination proof.

The read-only checker
`brats_sequence_project/finetune/scripts/audit_pretraining_overlap.py` validates
the current 494 patient IDs and exact zero-overlap split invariants, then scans
any text manifests or historical mapping files supplied with `--manifest`.
With no subject manifest available, its expected status is
`UNKNOWN_NO_MANIFESTS`; no-match results remain UNKNOWN rather than being
reported as proof of no overlap. For a future manifest audit:

```bash
python brats_sequence_project/finetune/scripts/audit_pretraining_overlap.py \
  --manifest /path/to/BrainIAC-pretraining-manifest.csv \
  --manifest /path/to/historical-id-mapping.json \
  --output brats_sequence_project/finetune/results/pretraining_overlap_audit.json
```

The checker is read-only with respect to the CSVs, NIfTIs, checkpoints, and
split assignments. Its JSON report records exact matching patient IDs and line
numbers when a supplied source contains them.

### Final scientific-audit disposition

- **N4/preprocessing:** BraTS2020 N4 status is not documented; no extra N4,
  resampling, registration, or skull stripping is applied. This is the least
  assumption-dependent choice for already prepared BraTS2020 files, and the
  N4 provenance limitation is recorded rather than hidden.
- **Pretraining overlap:** UNKNOWN at subject level. The published 16-dataset
  pretraining table does not name BraTS2020 or BraTS2023, but no subject-level
  pretraining manifest is available to prove exclusion of every current ID.
- **Engineering readiness:** YES. The code is ready for the CUDA lab-server
  engineering smoke test once the canonical checkpoint, CSV paths, and data
  root are present there. The smoke test is not a scientific result and does
  not resolve the two audit uncertainties.
- **Scientific interpretation:** Any reported result must retain the
  BraTS2020-vs-BraTS2023 cohort difference and the UNKNOWN subject-level
  contamination status as limitations.

## Training and checkpointing

The production configuration is
`brats_sequence_project/finetune/config/full_finetune.yml`:

- 100 epochs.
- Training batch size 16; validation batch size 1.
- `CrossEntropyLoss`.
- Adam, learning rate `1e-4`, weight decay `1e-4`.
- `ReduceLROnPlateau(mode="max")` monitoring validation balanced accuracy,
  with PyTorch default factor `0.1` and patience `10` because the supplement
  does not publish those values.
- CUDA `16-mixed` precision, matching the released Lightning default.
- Best checkpoint is the epoch with maximum validation balanced accuracy.
- The general BrainIAC checkpoint is loaded anew for every fraction; no run
  resumes from another fraction's checkpoint.

The isolated runner writes local JSON/CSV artifacts and does not instantiate the
released script's optional W&B logger. This changes experiment logging only; it
does not change the model, data, loss, optimizer, scheduler, precision, or
checkpoint-selection decisions.

The saved checkpoint contains the model state and metadata recording the
general checkpoint path, fraction, seed, validation metric, and the fact that
test data was not used for selection.

## Metrics

Balanced accuracy is the primary reported metric during validation and final
test evaluation. Every report includes:

- balanced accuracy;
- ordinary accuracy;
- 4 x 4 confusion matrix with fixed class order T1, T2, FLAIR, T1CE;
- per-class precision, recall, F1, and support.

The evaluator additionally writes `predictions.csv` with:

`patient_id`, `image_path`, `true_label`, `predicted_label`, label names, and
`probability_T1`, `probability_T2`, `probability_FLAIR`, `probability_T1CE`.

## Fractions and smoke test

Supported fractions are 10%, 20%, 40%, 60%, 80%, and 100%. Selection is
patient-level: patient IDs are sorted, shuffled with the requested seed, and
the floor of `fraction * number_of_train_patients` complete patient groups is
selected. All four rows for each selected patient are included. The current
train/validation/test layout is preserved: fractions select from `train.csv`,
the complete `val.csv` remains the checkpoint-selection set, and `test.csv`
remains untouched until final evaluation.

`--smoke-test` overrides only the in-memory run settings to use two train
patients, two validation patients, and one epoch. It is an engineering smoke
test for data loading, model initialization, backward propagation, metric
calculation, and checkpoint writing. Its output is not a scientific result
and must not be included in paper tables.

## Lab-server commands

Run all commands below from the repository root on the CUDA lab server. Do
not run the training command on this Mac.

### 0. Prepare and audit data/checkpoint

Set `BRAINIACTEST_DATASET_ROOT` to the directory containing the two BraTS2020
source directories, then regenerate the existing patient-level CSVs if they
are not already present on the server:

```bash
cd /path/to/BrainIAC_Testing
export BRAINIACTEST_DATASET_ROOT=/path/to/BraTS2020

python brats_sequence_project/scripts/make_brats_splits.py \
  --dataset_root "$BRAINIACTEST_DATASET_ROOT" \
  --output_dir brats_sequence_project/outputs \
  --seed 42 \
  --train_ratio 0.8

python brats_sequence_project/finetune/scripts/validate_splits.py \
  --config brats_sequence_project/finetune/config/full_finetune.yml \
  --check-paths

python brats_sequence_project/finetune/scripts/verify_setup.py \
  --config brats_sequence_project/finetune/config/full_finetune.yml \
  --checkpoint checkpoints/BrainIAC.ckpt
```

`checkpoints/BrainIAC.ckpt` must be the general foundation checkpoint. Do not
rename or substitute a downstream checkpoint to satisfy this path.

### 1. Engineering smoke test

```bash
python brats_sequence_project/finetune/scripts/finetune_model.py \
  --config brats_sequence_project/finetune/config/full_finetune.yml \
  --checkpoint checkpoints/BrainIAC.ckpt \
  --smoke-test \
  --output-dir brats_sequence_project/finetune/results/smoke_test
```

### 2. Full 100% fine-tuning

```bash
python brats_sequence_project/finetune/scripts/finetune_model.py \
  --config brats_sequence_project/finetune/config/full_finetune.yml \
  --checkpoint checkpoints/BrainIAC.ckpt \
  --fraction 1.0 \
  --seed 42 \
  --output-dir brats_sequence_project/finetune/results/fraction_100
```

### 3. One final test-set evaluation

Run this only after selecting the best checkpoint from the completed training
run:

```bash
python brats_sequence_project/finetune/scripts/evaluate_checkpoint.py \
  --config brats_sequence_project/finetune/config/full_finetune.yml \
  --checkpoint brats_sequence_project/finetune/results/fraction_100/best_model.ckpt \
  --test-csv brats_sequence_project/outputs/test.csv \
  --output-dir brats_sequence_project/finetune/results/fraction_100/test_evaluation \
  --device cuda
```

This writes `metrics.json`, `confusion_matrix.csv`, and `predictions.csv` and
reports `balanced_accuracy` first as the primary metric. The evaluator makes
one test-loader pass and does not alter the checkpoint.

### 4. Later 10/20/40/60/80/100% reproduction

Each fraction is an independent run initialized from the general checkpoint:

```bash
for BRAINIACTEST_FRACTION in 0.10 0.20 0.40 0.60 0.80 1.00; do
  BRAINIACTEST_PERCENT=$(awk "BEGIN {printf \"%03d\", 100 * $BRAINIACTEST_FRACTION}")
  python brats_sequence_project/finetune/scripts/finetune_model.py \
    --config brats_sequence_project/finetune/config/full_finetune.yml \
    --checkpoint checkpoints/BrainIAC.ckpt \
    --fraction "$BRAINIACTEST_FRACTION" \
    --seed 42 \
    --output-dir "brats_sequence_project/finetune/results/fraction_${BRAINIACTEST_PERCENT}"
done
```

Evaluate each selected checkpoint exactly once with the fixed test CSV:

```bash
for BRAINIACTEST_FRACTION in 0.10 0.20 0.40 0.60 0.80 1.00; do
  BRAINIACTEST_PERCENT=$(awk "BEGIN {printf \"%03d\", 100 * $BRAINIACTEST_FRACTION}")
  python brats_sequence_project/finetune/scripts/evaluate_checkpoint.py \
    --config brats_sequence_project/finetune/config/full_finetune.yml \
    --checkpoint "brats_sequence_project/finetune/results/fraction_${BRAINIACTEST_PERCENT}/best_model.ckpt" \
    --test-csv brats_sequence_project/outputs/test.csv \
    --output-dir "brats_sequence_project/finetune/results/fraction_${BRAINIACTEST_PERCENT}/test_evaluation" \
    --device cuda
done
```

Do not use test performance to choose among epochs or fractions.

## CPU-safe checks used before handoff

The Mac-side verification is limited to static/CPU-safe checks:

- Python syntax compilation for all new scripts.
- YAML configuration parsing and invariant validation.
- CSV parsing, label/modality mapping, NIfTI suffix checks, mask rejection,
  four-row patient checks, and pairwise overlap checks.
- General checkpoint basename/path contract.
- Model construction from a local general checkpoint when dependencies and the
  ignored local file are available; output dimension and every parameter's
  `requires_grad` state are asserted.
- Presence of balanced-accuracy metric code and test evaluation outputs.

No production training, test-set model selection, commit, or push is performed
by this preparation change.
