#!/usr/bin/env python3
"""CPU-safe checks for the additive data-efficiency orchestration helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_data_efficiency import (  # noqa: E402
    DEFAULT_COMPATIBILITY_CONFIG,
    aggregate_runs,
    discover_completed_runs,
    find_full_data_baseline,
)
from finetune_common import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_INDEX,
    REPO_ROOT,
    read_csv_rows,
    select_patient_ids,
    validate_patient_subset,
    write_csv_rows,
    load_yaml_config,
)
from make_subset_manifest import prepare_subset_manifest  # noqa: E402
from run_data_efficiency import run_directory, valid_metrics_file  # noqa: E402


def synthetic_rows(prefix: str, patient_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, patient_count + 1):
        patient_id = f"{prefix}_{index:03d}"
        for label, modality in enumerate(CLASS_NAMES):
            suffix = modality.lower()
            rows.append(
                {
                    "patient_id": patient_id,
                    "image_path": f"/synthetic/{patient_id}_{suffix}.nii",
                    "label": label,
                    "modality": modality,
                    "split_source": prefix,
                }
            )
    return rows


class DataEfficiencyHelpersTest(unittest.TestCase):
    def test_nested_and_deterministic_patient_selection(self) -> None:
        rows = synthetic_rows("train", 295)
        one, ordering = select_patient_ids(rows, 1, 42)
        two, same_ordering = select_patient_ids(rows, 2, 42)
        five, _ = select_patient_ids(rows, 5, 42)
        other, other_ordering = select_patient_ids(rows, 5, 123)

        self.assertEqual(ordering, same_ordering)
        self.assertEqual(set(one), set(two[:1]))
        self.assertEqual(set(two), set(five[:2]))
        self.assertNotEqual(ordering, other_ordering)
        self.assertNotEqual(set(five), set(other))

    def test_subset_completeness_balance_and_no_leakage(self) -> None:
        train = synthetic_rows("train", 295)
        validation = synthetic_rows("validation", 74)
        test = synthetic_rows("test", 125)
        selected_ids, _ = select_patient_ids(train, 5, 42)
        subset = [row for row in train if row["patient_id"] in set(selected_ids)]

        summary = validate_patient_subset(subset, 5, train, validation, test)
        self.assertEqual(summary["scan_count"], 20)
        self.assertEqual(summary["images_per_class"], {name: 5 for name in CLASS_NAMES})
        self.assertTrue(summary["no_patient_leakage"])

    def test_manifest_reuse_and_run_directory_naming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.csv"
            validation_path = root / "val.csv"
            test_path = root / "test.csv"
            write_csv_rows(train_path, synthetic_rows("train", 295))
            write_csv_rows(validation_path, synthetic_rows("validation", 74))
            write_csv_rows(test_path, synthetic_rows("test", 125))
            manifest_path = root / "patients_005" / "seed_42" / "subset_manifest.csv"

            first = prepare_subset_manifest(
                train_csv=train_path,
                validation_csv=validation_path,
                test_csv=test_path,
                patient_count=5,
                seed=42,
                output_path=manifest_path,
            )
            second = prepare_subset_manifest(
                train_csv=train_path,
                validation_csv=validation_path,
                test_csv=test_path,
                patient_count=5,
                seed=42,
                output_path=manifest_path,
            )
            self.assertTrue(first["manifest_created"])
            self.assertFalse(second["manifest_created"])
            self.assertEqual(len(read_csv_rows(manifest_path)), 20)
            self.assertEqual(
                run_directory({"output": {"root_dir": root}}, 5, 42),
                root / "patients_005" / "seed_42",
            )

    def test_resume_detection_and_mock_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, balanced_accuracy in ((42, 0.5), (123, 0.75)):
                run_dir = root / "patients_001" / f"seed_{seed}"
                metrics_path = run_dir / "test_evaluation" / "metrics.json"
                metrics_path.parent.mkdir(parents=True)
                metrics = {
                    "primary_metric": "balanced_accuracy",
                    "balanced_accuracy": balanced_accuracy,
                    "accuracy": balanced_accuracy,
                    "confusion_matrix": [
                        [125, 0, 0, 0],
                        [0, 125, 0, 0],
                        [0, 0, 125, 0],
                        [0, 0, 0, 125],
                    ],
                    "test_csv": "/server/repo/brats_sequence_project/outputs/test.csv",
                    "checkpoint_selected_without_test": True,
                }
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
                (run_dir / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "patient_count": 1,
                            "seed": seed,
                            "best_validation_balanced_accuracy": 0.4,
                            "full_fine_tuning": True,
                        }
                    ),
                    encoding="utf-8",
                )

            completed = discover_completed_runs(root)
            self.assertEqual(len(completed), 2)
            self.assertTrue(valid_metrics_file(root / "patients_001" / "seed_42" / "test_evaluation" / "metrics.json"))
            summary = aggregate_runs(completed, baseline=None)
            self.assertEqual(summary[0]["number_of_completed_seeds"], 2)
            self.assertAlmostEqual(summary[0]["mean_test_balanced_accuracy"], 0.625)
            self.assertAlmostEqual(summary[0]["std_test_balanced_accuracy"], 0.1767766953)
            self.assertIsNone(summary[0]["performance_retained_percent"])

    def test_baseline_requires_compatibility_and_reads_fraction_100_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_efficiency_root = root / "data_efficiency"
            baseline_root = root / "baselines"
            arbitrary_run = data_efficiency_root / "patients_295" / "seed_42"
            compatible_run = baseline_root / "fraction_100"

            def write_candidate(run_dir: Path, *, validation_csv: str) -> None:
                run_dir.mkdir(parents=True)
                (run_dir / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "patient_count": 295,
                            "scan_count": 1180,
                            "scans_per_class": 295,
                            "seed": 42,
                            "fraction": 1.0,
                            "selected_train_patients": [f"P{index:03d}" for index in range(295)],
                            "config_path": str(
                                REPO_ROOT / "brats_sequence_project/finetune/config/full_finetune.yml"
                            ),
                            "checkpoint_path": str(REPO_ROOT / "checkpoints/BrainIAC.ckpt"),
                            "general_brainiac_checkpoint": str(
                                REPO_ROOT / "checkpoints/BrainIAC.ckpt"
                            ),
                            "source_train_csv": str(REPO_ROOT / "brats_sequence_project/outputs/train.csv"),
                            "validation_csv": validation_csv,
                            "test_csv": str(REPO_ROOT / "brats_sequence_project/outputs/test.csv"),
                            "parameter_report": {
                                "total_parameters": 88343812,
                                "trainable_parameters": 88343812,
                                "frozen_parameters": 0,
                            },
                            "class_mapping": dict(CLASS_TO_INDEX),
                            "checkpoint_selection_metric": "validation balanced accuracy",
                            "test_used_for_selection": False,
                            "best_validation_balanced_accuracy": 0.966216,
                        }
                    ),
                    encoding="utf-8",
                )
                evaluation_dir = run_dir / "test_evaluation"
                evaluation_dir.mkdir()
                (evaluation_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "primary_metric": "balanced_accuracy",
                            "balanced_accuracy": 0.938,
                            "accuracy": 0.938,
                            "confusion_matrix": [
                                [114, 0, 2, 9],
                                [0, 117, 8, 0],
                                [1, 2, 122, 0],
                                [9, 0, 0, 116],
                            ],
                            "class_mapping": dict(CLASS_TO_INDEX),
                            "test_csv": str(REPO_ROOT / "brats_sequence_project/outputs/test.csv"),
                            "checkpoint_selected_without_test": True,
                        }
                    ),
                    encoding="utf-8",
                )

            write_candidate(arbitrary_run, validation_csv="/server/other/val.csv")
            expected_config = load_yaml_config(DEFAULT_COMPATIBILITY_CONFIG)
            self.assertIsNone(
                find_full_data_baseline(
                    data_efficiency_root,
                    baseline_root,
                    expected_config=expected_config,
                )
            )

            write_candidate(
                compatible_run,
                validation_csv=str(REPO_ROOT / "brats_sequence_project/outputs/val.csv"),
            )
            baseline = find_full_data_baseline(
                data_efficiency_root,
                baseline_root,
                expected_config=expected_config,
            )
            self.assertIsNotNone(baseline)
            assert baseline is not None
            self.assertEqual(Path(baseline["run_directory"]).name, "fraction_100")
            self.assertAlmostEqual(baseline["test_balanced_accuracy"], 0.938)


if __name__ == "__main__":
    unittest.main()
