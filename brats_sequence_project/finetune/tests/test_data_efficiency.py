#!/usr/bin/env python3
"""CPU-safe checks for the additive data-efficiency orchestration helpers."""

from __future__ import annotations

import json
import copy
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_data_efficiency import (  # noqa: E402
    DEFAULT_COMPATIBILITY_CONFIG,
    aggregate_runs,
    discover_completed_runs,
    find_full_data_baseline,
    plot_curves,
    plot_run_artifacts,
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
from make_subset_manifest import (  # noqa: E402
    prepare_subset_manifest,
    seed_plan_from_config,
)
from run_data_efficiency import (  # noqa: E402
    EVALUATOR,
    TRAINER,
    candidate_status,
    matrix_candidates,
    run_directory,
    run_matrix,
    run_one,
    valid_metrics_file,
)


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


def temporary_config(root: Path) -> dict[str, object]:
    config = copy.deepcopy(load_yaml_config(DEFAULT_COMPATIBILITY_CONFIG))
    outputs = root / "outputs"
    config["data"]["train_csv"] = str(outputs / "train.csv")
    config["data"]["val_csv"] = str(outputs / "val.csv")
    config["data"]["test_csv"] = str(outputs / "test.csv")
    config["model"]["checkpoint_path"] = str(root / "BrainIAC.ckpt")
    config["output"]["root_dir"] = str(root / "results")
    outputs.mkdir(parents=True, exist_ok=True)
    write_csv_rows(outputs / "train.csv", synthetic_rows("train", 295))
    write_csv_rows(outputs / "val.csv", synthetic_rows("validation", 74))
    write_csv_rows(outputs / "test.csv", synthetic_rows("test", 125))
    return config


def valid_metrics_payload(balanced_accuracy: float) -> dict[str, object]:
    return {
        "primary_metric": "balanced_accuracy",
        "balanced_accuracy": balanced_accuracy,
        "accuracy": balanced_accuracy,
        "confusion_matrix": [
            [125, 0, 0, 0],
            [0, 125, 0, 0],
            [0, 0, 125, 0],
            [0, 0, 0, 125],
        ],
        "test_csv": "outputs/test.csv",
        "checkpoint_selected_without_test": True,
    }


def write_valid_metrics(run_dir: Path, balanced_accuracy: float) -> Path:
    metrics_path = run_dir / "test_evaluation" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(valid_metrics_payload(balanced_accuracy)), encoding="utf-8"
    )
    return metrics_path


class DataEfficiencyHelpersTest(unittest.TestCase):
    def test_configured_matrix_has_exactly_two_seeds_and_22_candidates(self) -> None:
        config = load_yaml_config(DEFAULT_COMPATIBILITY_CONFIG)
        plan = seed_plan_from_config(config)
        self.assertEqual(list(plan), [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 250])
        self.assertEqual({seed for seeds in plan.values() for seed in seeds}, {42, 123})
        self.assertTrue(all(seeds == [42, 123] for seeds in plan.values()))
        candidates = matrix_candidates(plan)
        self.assertEqual(len(candidates), 22)
        self.assertEqual(candidates[:4], [(1, 42), (1, 123), (2, 42), (2, 123)])
        self.assertEqual(candidates[-2:], [(250, 42), (250, 123)])

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

    def test_completed_candidate_is_skipped_without_training_or_manifest_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = temporary_config(root)
            config_path = root / "data_efficiency.yml"
            run_dir = run_directory(config, 1, 42)
            write_valid_metrics(run_dir, 0.446)
            checkpoint_path = Path(config["model"]["checkpoint_path"])

            with (
                mock.patch("run_data_efficiency.safe_plot_run_artifacts", return_value=[]),
                mock.patch("run_data_efficiency.subprocess.run") as subprocess_run,
            ):
                result = run_one(config, config_path, checkpoint_path, 1, 42)

            self.assertEqual(result, "skipped")
            subprocess_run.assert_not_called()
            self.assertFalse((run_dir / "subset_manifest.csv").exists())
            self.assertEqual(candidate_status(config, 1, 42)[0], "COMPLETE")

    def test_checkpoint_resume_runs_evaluation_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = temporary_config(root)
            config_path = root / "data_efficiency.yml"
            checkpoint_path = Path(config["model"]["checkpoint_path"])
            checkpoint_path.write_bytes(b"general checkpoint")
            run_dir = run_directory(config, 1, 42)
            run_dir.mkdir(parents=True)
            (run_dir / "best_model.ckpt").write_bytes(b"best checkpoint")

            commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> object:
                commands.append(command)
                output_dir = Path(command[command.index("--output-dir") + 1])
                write_valid_metrics(output_dir.parent, 0.446)
                return mock.DEFAULT

            with (
                mock.patch("run_data_efficiency._git_commit_hash", return_value="test"),
                mock.patch("run_data_efficiency.safe_plot_run_artifacts", return_value=[]),
                mock.patch("run_data_efficiency.subprocess.run", side_effect=fake_run),
            ):
                result = run_one(config, config_path, checkpoint_path, 1, 42)

            self.assertEqual(result, "completed")
            self.assertEqual(len(commands), 1)
            self.assertEqual(Path(commands[0][1]), EVALUATOR)
            self.assertNotIn(str(TRAINER), " ".join(commands[0]))
            self.assertEqual(candidate_status(config, 1, 42)[0], "COMPLETE")

    def test_matrix_order_is_sequential_and_aggregation_runs_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = temporary_config(root)
            config_path = root / "data_efficiency.yml"
            checkpoint_path = Path(config["model"]["checkpoint_path"])
            plan = {1: [42, 123], 2: [42, 123], 5: [42, 123]}
            seen: list[tuple[int, int]] = []

            def fake_run(
                _config: object,
                _config_path: Path,
                _checkpoint_path: Path,
                patient_count: int,
                seed: int,
                **_: object,
            ) -> str:
                seen.append((patient_count, seed))
                return "completed"

            aggregate_calls: list[dict[str, object]] = []

            def fake_aggregate(**kwargs: object) -> dict[str, object]:
                aggregate_calls.append(kwargs)
                return {"summary_path": root / "summary.csv"}

            result = run_matrix(
                config,
                config_path,
                checkpoint_path,
                plan,
                run_function=fake_run,
                aggregate_function=fake_aggregate,
            )

            self.assertEqual(seen, matrix_candidates(plan))
            self.assertEqual(len(aggregate_calls), 1)
            self.assertEqual(result["summary_path"], root / "summary.csv")

    def test_single_candidate_dry_run_does_not_start_scientific_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = temporary_config(root)
            with mock.patch("run_data_efficiency.subprocess.run") as subprocess_run:
                result = run_one(
                    config,
                    root / "data_efficiency.yml",
                    Path(config["model"]["checkpoint_path"]),
                    5,
                    42,
                    dry_run=True,
                )
            self.assertEqual(result, "dry-run")
            subprocess_run.assert_not_called()

    def test_dry_run_matrix_is_non_training_and_preserves_matrix_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = temporary_config(root)
            plan = {1: [42, 123], 2: [42, 123]}
            with mock.patch("run_data_efficiency.subprocess.run") as subprocess_run:
                result = run_matrix(
                    config,
                    root / "data_efficiency.yml",
                    Path(config["model"]["checkpoint_path"]),
                    plan,
                    dry_run=True,
                )
            self.assertIsNone(result)
            subprocess_run.assert_not_called()
            self.assertEqual(
                [
                    (patient_count, seed)
                    for patient_count, seed in matrix_candidates(plan)
                    if (run_directory(config, patient_count, seed) / "subset_manifest.csv").is_file()
                ],
                matrix_candidates(plan),
            )

    def test_aggregation_across_two_seeds_calculates_sample_sd_and_retention(self) -> None:
        runs = [
            {
                "patient_count": 5,
                "training_scans": 20,
                "scans_per_class": 5,
                "seed": 42,
                "best_validation_balanced_accuracy": 0.4,
                "test_balanced_accuracy": 0.4,
                "test_accuracy": 0.45,
            },
            {
                "patient_count": 5,
                "training_scans": 20,
                "scans_per_class": 5,
                "seed": 123,
                "best_validation_balanced_accuracy": 0.6,
                "test_balanced_accuracy": 0.6,
                "test_accuracy": 0.55,
            },
        ]
        summary = aggregate_runs(
            runs,
            {"test_balanced_accuracy": 0.8, "test_accuracy": 0.82},
        )
        self.assertEqual(summary[0]["completed_seed_count"], 2)
        self.assertAlmostEqual(summary[0]["mean_test_balanced_accuracy"], 0.5)
        self.assertAlmostEqual(summary[0]["std_test_balanced_accuracy"], 0.1414213562)
        self.assertAlmostEqual(summary[0]["performance_retained_percent"], 62.5)
        self.assertAlmostEqual(summary[0]["mean_best_validation_balanced_accuracy"], 0.5)

    def test_plot_generation_uses_saved_history_and_final_confusion_matrix(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "patients_001" / "seed_42"
            run_dir.mkdir(parents=True)
            (run_dir / "training_history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 1,
                            "train_loss": 1.0,
                            "validation_loss": 1.1,
                            "validation_metrics": {"balanced_accuracy": 0.4},
                        },
                        {
                            "epoch": 2,
                            "train_loss": 0.8,
                            "validation_loss": 0.9,
                            "validation_metrics": {"balanced_accuracy": 0.6},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            write_valid_metrics(run_dir, 0.5)
            paths = plot_run_artifacts(run_dir)
            self.assertEqual(
                {path.name for path in paths},
                {"loss_curve.png", "validation_balanced_accuracy_curve.png", "confusion_matrix.png"},
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))

    def test_aggregate_curve_files_include_baseline_when_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        summary = aggregate_runs(
            [
                {
                    "patient_count": 1,
                    "training_scans": 4,
                    "scans_per_class": 1,
                    "seed": 42,
                    "best_validation_balanced_accuracy": 0.3,
                    "test_balanced_accuracy": 0.4,
                    "test_accuracy": 0.4,
                },
                {
                    "patient_count": 1,
                    "training_scans": 4,
                    "scans_per_class": 1,
                    "seed": 123,
                    "best_validation_balanced_accuracy": 0.5,
                    "test_balanced_accuracy": 0.6,
                    "test_accuracy": 0.6,
                },
            ],
            {"test_balanced_accuracy": 0.8, "test_accuracy": 0.8},
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_curves(
                summary,
                Path(temporary),
                runs=[
                    {
                        "patient_count": 1,
                        "training_scans": 4,
                        "seed": 42,
                        "test_balanced_accuracy": 0.4,
                    },
                    {
                        "patient_count": 1,
                        "training_scans": 4,
                        "seed": 123,
                        "test_balanced_accuracy": 0.6,
                    },
                ],
                baseline={"test_balanced_accuracy": 0.8},
            )
            names = {path.name for path in paths}
            self.assertIn("data_efficiency_curve.png", names)
            self.assertIn("data_efficiency_individual_seeds.png", names)
            self.assertIn("performance_retained.png", names)
            self.assertIn("data_efficiency_accuracy_curve.png", names)

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
