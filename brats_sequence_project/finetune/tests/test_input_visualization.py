from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "brats_sequence_project/finetune/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from finetune_common import CLASS_TO_INDEX, CLASS_NAMES  # noqa: E402
from visualize_model_input import (  # noqa: E402
    array_statistics,
    build_batch_metadata,
    ensure_output_dir,
    inspect_transform_stages,
    parse_args,
    _transform_lines,
    save_actual_batch_figure,
    save_pair_figure,
    save_triplanar_figure,
)


class TestInputVisualization(unittest.TestCase):
    def test_class_mapping_and_cli(self) -> None:
        self.assertEqual(CLASS_TO_INDEX, {"T1": 0, "T2": 1, "FLAIR": 2, "T1CE": 3})
        self.assertEqual(CLASS_NAMES, ("T1", "T2", "FLAIR", "T1CE"))
        args = parse_args(
            ["--output-dir", "out", "--seed", "42", "--batch-index", "1", "--num-samples", "2"]
        )
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.batch_index, 1)
        self.assertEqual(args.num_samples, 2)

    def test_output_directory_creation_and_array_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = ensure_output_dir(Path(temporary) / "nested" / "visualization")
            self.assertTrue(output_dir.is_dir())
            stats = array_statistics(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
        self.assertEqual(stats["shape"], [2, 2, 2])
        self.assertEqual(stats["dtype"], "float32")
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 7.0)
        self.assertAlmostEqual(stats["mean"], 3.5)

    def test_training_transform_prefix_is_extracted_with_tensor_conversion(self) -> None:
        class LoadImaged:
            def __repr__(self) -> str:
                return "LoadImaged(keys=['image'])"

        class Resized:
            def __repr__(self) -> str:
                return "Resized(spatial_size=(96, 96, 96))"

        class RandAffined:
            def __repr__(self) -> str:
                return "RandAffined(prob=0.5)"

        class ToTensord:
            def __repr__(self) -> str:
                return "ToTensord(keys=['image'])"

        class Compose:
            def __init__(self, transforms: list[object]) -> None:
                self.transforms = transforms

        prefix = [LoadImaged(), Resized()]
        preprocessing_items, random_items = inspect_transform_stages(
            Compose(prefix + [RandAffined(), ToTensord()])
        )
        self.assertEqual(
            [type(item).__name__ for item in preprocessing_items],
            ["LoadImaged", "Resized", "ToTensord"],
        )
        self.assertEqual([type(item).__name__ for item in random_items], ["RandAffined"])

    def test_transform_summary_includes_nested_parameters(self) -> None:
        class Resize:
            def __init__(self) -> None:
                self.spatial_size = (96, 96, 96)

        class Resized:
            def __init__(self) -> None:
                self.keys = ("image",)
                self.mode = ("trilinear",)
                self.resizer = Resize()

        class Compose:
            def __init__(self, transforms: list[object]) -> None:
                self.transforms = transforms

        description = _transform_lines(Compose([Resized()]))[0]
        self.assertIn("spatial_size=(96, 96, 96)", description)
        self.assertIn("mode=('trilinear',)", description)

    def test_batch_metadata_serializes_with_actual_mapping_and_stats(self) -> None:
        images = np.arange(2 * 1 * 2 * 2 * 2, dtype=np.float32).reshape(2, 1, 2, 2, 2)
        metadata = build_batch_metadata(
            images,
            [0, 3],
            ["P001", "P001"],
            ["T1", "T1CE"],
            ["/data/P001_t1.nii.gz", "/data/P001_t1ce.nii.gz"],
            seed=42,
            train_csv="/data/subset_manifest.csv",
            batch_size_configured=16,
            batch_index=0,
            random_transform_names=["RandAffined"],
        )
        encoded = json.dumps(metadata, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["batch_shape"], [2, 1, 2, 2, 2])
        self.assertTrue(decoded["random_augmentation_applied"])
        self.assertEqual(decoded["samples"][1]["modality"], "T1CE")
        self.assertEqual(decoded["samples"][1]["class_id"], 3)
        self.assertEqual(decoded["samples"][0]["min"], 0.0)

    def test_plot_helpers_write_synthetic_volume_figures(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
        sample = {
            "batch_index": 0,
            "patient_id": "P001",
            "modality": "T1",
            "class_id": 0,
            "raw_volume": volume,
            "raw_spacing": (1.0, 1.0, 1.2),
            "raw_shape_label": [5, 6, 7],
            "preprocessed_volume": volume,
            "preprocessed_shape_label": [1, 96, 96, 96],
            "final_volume": volume,
            "final_shape_label": [1, 96, 96, 96],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            triplanar = root / "raw_triplanar.png"
            batch = root / "actual_training_batch.png"
            pair = root / "raw_vs_final_batch.png"
            save_triplanar_figure(
                [sample],
                "raw_volume",
                triplanar,
                title="Synthetic raw",
                shape_key="raw_shape_label",
                spacing_key="raw_spacing",
            )
            save_actual_batch_figure([sample], batch, batch_shape=[1, 1, 96, 96, 96])
            save_pair_figure(
                [sample],
                "raw_volume",
                "final_volume",
                pair,
                title="Synthetic pair",
                left_title="Raw",
                right_title="Final",
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in (triplanar, batch, pair)))

    def test_visualization_path_does_not_call_model_or_training(self) -> None:
        source = (SCRIPT_DIR / "visualize_model_input.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("build_brainiac_model", called_names)
        self.assertNotIn("run_epoch", called_names)
        self.assertNotIn("validate_epoch", called_names)
        self.assertNotIn("torch.save", source)


if __name__ == "__main__":
    unittest.main()
