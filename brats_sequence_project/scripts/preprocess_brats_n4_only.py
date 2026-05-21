"""Create N4-only BraTS preprocessing-ablation inputs and CSV splits.

This script writes new NIfTI files. It never modifies the original BraTS
dataset and does not run BrainIAC feature extraction or classifier training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRAINING_SUBDIR = Path("BraTS2020_TrainingData") / "MICCAI_BraTS2020_TrainingData"
VALIDATION_SUBDIR = Path("BraTS2020_ValidationData") / "MICCAI_BraTS2020_ValidationData"

LABEL_MAPPING = {
    "T1": 0,
    "T2": 1,
    "FLAIR": 2,
    "T1CE": 3,
}

MODALITY_SUFFIXES = {
    "T1": "_t1.nii",
    "T2": "_t2.nii",
    "FLAIR": "_flair.nii",
    "T1CE": "_t1ce.nii",
}

CSV_COLUMNS = ["patient_id", "image_path", "label", "modality", "split_source"]
METADATA_COLUMNS = [
    "patient_id",
    "split_source",
    "modality",
    "label",
    "original_path",
    "processed_path",
    "original_shape",
    "processed_shape",
    "original_spacing",
    "processed_spacing",
    "original_min",
    "original_max",
    "original_mean_nonzero",
    "processed_min",
    "processed_max",
    "processed_mean_nonzero",
    "status",
    "error_message",
]


@dataclass(frozen=True)
class ImageTask:
    patient_id: str
    split_source: str
    source_group: str
    modality: str
    label: int
    original_path: Path
    processed_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--splits_output_dir", type=Path, required=True)
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient IDs to process.")
    parser.add_argument("--include_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--visualize_examples", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop on the first failed image.")
    parser.add_argument("--n4_iterations", type=str, default="50,50,30,20")
    parser.add_argument("--n4_convergence_threshold", type=float, default=1e-7)
    parser.add_argument("--n4_shrink_factor", type=int, default=4)
    return parser.parse_args()


def require_simpleitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("SimpleITK is required for N4 preprocessing. Install it with: pip install SimpleITK") from exc
    return sitk


def require_visual_deps() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("Visualization requires numpy and matplotlib.") from exc
    return np, plt


def parse_iterations(value: str) -> list[int]:
    iterations = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not iterations or any(item <= 0 for item in iterations):
        raise ValueError(f"--n4_iterations must be positive comma-separated integers, got {value!r}")
    return iterations


def parse_patient_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    patient_ids = {item.strip() for item in value.split(",") if item.strip()}
    if not patient_ids:
        raise ValueError("--patient_ids was provided but no patient IDs were parsed.")
    return patient_ids


def discover_patient_dirs(root: Path, expected_prefix: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Expected dataset directory does not exist: {root}")
    patient_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(expected_prefix))
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories with prefix {expected_prefix!r} found under {root}")
    return patient_dirs


def validate_modality_path(patient_dir: Path, modality: str) -> Path:
    patient_id = patient_dir.name
    path = patient_dir / f"{patient_id}{MODALITY_SUFFIXES[modality]}"
    name = path.name.lower()
    if "seg" in name or "segm" in name:
        raise ValueError(f"Segmentation file was incorrectly selected: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {modality} file for {patient_id}: {path}")
    return path


def patient_dirs_for_args(args: argparse.Namespace) -> list[tuple[Path, str, str]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    requested_ids = parse_patient_ids(args.patient_ids)
    selected: list[tuple[Path, str, str]] = []

    if args.include_training:
        training_root = dataset_root / TRAINING_SUBDIR
        for patient_dir in discover_patient_dirs(training_root, "BraTS20_Training_"):
            selected.append((patient_dir, "training", "BraTS2020_TrainingData"))

    if args.include_validation:
        validation_root = dataset_root / VALIDATION_SUBDIR
        for patient_dir in discover_patient_dirs(validation_root, "BraTS20_Validation_"):
            selected.append((patient_dir, "validation", "BraTS2020_ValidationData"))

    if requested_ids is not None:
        selected = [item for item in selected if item[0].name in requested_ids]
        missing = requested_ids - {item[0].name for item in selected}
        if missing:
            raise FileNotFoundError(f"Requested patient IDs were not found: {sorted(missing)}")

    selected = sorted(selected, key=lambda item: (item[1], item[0].name))
    if args.max_patients is not None:
        if args.max_patients <= 0:
            raise ValueError("--max_patients must be positive when provided.")
        selected = selected[: args.max_patients]

    if not selected:
        raise ValueError("No patients selected for processing.")
    return selected


def build_tasks(patient_entries: list[tuple[Path, str, str]], output_root: Path) -> list[ImageTask]:
    tasks: list[ImageTask] = []
    for patient_dir, split_source, source_group in patient_entries:
        patient_id = patient_dir.name
        for modality, label in LABEL_MAPPING.items():
            original_path = validate_modality_path(patient_dir, modality)
            output_dir = output_root / source_group / patient_id
            processed_path = output_dir / f"{patient_id}{MODALITY_SUFFIXES[modality].removesuffix('.nii')}_n4.nii.gz"
            tasks.append(
                ImageTask(
                    patient_id=patient_id,
                    split_source=split_source,
                    source_group=source_group,
                    modality=modality,
                    label=label,
                    original_path=original_path,
                    processed_path=processed_path,
                )
            )
    return tasks


def image_stats(sitk: Any, image: Any) -> dict[str, Any]:
    import numpy as np

    array = sitk.GetArrayFromImage(image).astype("float64", copy=False)
    nonzero = array[array != 0]
    return {
        "shape": list(image.GetSize()),
        "spacing": [float(item) for item in image.GetSpacing()],
        "min": float(np.min(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "mean_nonzero": float(np.mean(nonzero)) if nonzero.size else None,
        "nonzero_count": int(nonzero.size),
    }


def n4_bias_correct(
    sitk: Any,
    image: Any,
    iterations: list[int],
    convergence_threshold: float,
    shrink_factor: int,
) -> Any:
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.Cast(image > 0, sitk.sitkUInt8)
    mask_count = int(sitk.GetArrayViewFromImage(mask).sum())
    if mask_count == 0:
        raise ValueError("N4 mask is empty because the image has no nonzero voxels.")

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iterations)
    corrector.SetConvergenceThreshold(convergence_threshold)

    if shrink_factor > 1:
        shrink = [int(shrink_factor)] * image.GetDimension()
        working_image = sitk.Shrink(image, shrink)
        working_mask = sitk.Shrink(mask, shrink)
        corrector.Execute(working_image, working_mask)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        corrected = image / sitk.Exp(log_bias_field)
    else:
        corrected = corrector.Execute(image, mask)

    corrected = sitk.Cast(corrected, sitk.sitkFloat32)
    corrected.CopyInformation(image)
    return corrected


def metadata_row_from_stats(
    task: ImageTask,
    original_stats: dict[str, Any] | None,
    processed_stats: dict[str, Any] | None,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "patient_id": task.patient_id,
        "split_source": task.split_source,
        "modality": task.modality,
        "label": task.label,
        "original_path": str(task.original_path),
        "processed_path": str(task.processed_path),
        "original_shape": json.dumps(original_stats["shape"]) if original_stats else "",
        "processed_shape": json.dumps(processed_stats["shape"]) if processed_stats else "",
        "original_spacing": json.dumps(original_stats["spacing"]) if original_stats else "",
        "processed_spacing": json.dumps(processed_stats["spacing"]) if processed_stats else "",
        "original_min": original_stats["min"] if original_stats else "",
        "original_max": original_stats["max"] if original_stats else "",
        "original_mean_nonzero": original_stats["mean_nonzero"] if original_stats else "",
        "processed_min": processed_stats["min"] if processed_stats else "",
        "processed_max": processed_stats["max"] if processed_stats else "",
        "processed_mean_nonzero": processed_stats["mean_nonzero"] if processed_stats else "",
        "status": status,
        "error_message": error_message,
    }


def process_one_task(
    task: ImageTask,
    iterations: list[int],
    convergence_threshold: float,
    shrink_factor: int,
    force: bool,
) -> dict[str, Any]:
    sitk = require_simpleitk()
    try:
        original = sitk.ReadImage(str(task.original_path), sitk.sitkFloat32)
        original_stats = image_stats(sitk, original)
        if task.processed_path.exists() and not force:
            processed = sitk.ReadImage(str(task.processed_path), sitk.sitkFloat32)
            processed_stats = image_stats(sitk, processed)
            return metadata_row_from_stats(task, original_stats, processed_stats, "success_existing")

        task.processed_path.parent.mkdir(parents=True, exist_ok=True)
        corrected = n4_bias_correct(sitk, original, iterations, convergence_threshold, shrink_factor)
        sitk.WriteImage(corrected, str(task.processed_path))
        processed = sitk.ReadImage(str(task.processed_path), sitk.sitkFloat32)
        processed_stats = image_stats(sitk, processed)
        return metadata_row_from_stats(task, original_stats, processed_stats, "success")
    except Exception as exc:  # noqa: BLE001 - failure must be recorded per image.
        return metadata_row_from_stats(task, None, None, "failed", str(exc))


def run_tasks(args: argparse.Namespace, tasks: list[ImageTask]) -> list[dict[str, Any]]:
    iterations = parse_iterations(args.n4_iterations)
    if args.n4_shrink_factor < 1:
        raise ValueError("--n4_shrink_factor must be >= 1")

    try:
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("tqdm is required for progress reporting. Install it with: pip install tqdm") from exc

    if args.num_workers <= 1:
        rows = []
        for task in tqdm(tasks, desc="N4 preprocessing"):
            row = process_one_task(
                task,
                iterations=iterations,
                convergence_threshold=args.n4_convergence_threshold,
                shrink_factor=args.n4_shrink_factor,
                force=args.force,
            )
            if args.strict and row["status"] == "failed":
                raise RuntimeError(f"Failed processing {task.original_path}: {row['error_message']}")
            rows.append(row)
        return rows

    rows = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_one_task,
                task,
                iterations,
                args.n4_convergence_threshold,
                args.n4_shrink_factor,
                args.force,
            )
            for task in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="N4 preprocessing"):
            row = future.result()
            if args.strict and row["status"] == "failed":
                raise RuntimeError(f"Failed processing {row['original_path']}: {row['error_message']}")
            rows.append(row)
    return sorted(rows, key=lambda row: (row["split_source"], row["patient_id"], int(row["label"])))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def successful_complete_patients(metadata_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        if row["status"] in {"success", "success_existing"}:
            grouped[(str(row["split_source"]), str(row["patient_id"]))].append(row)
    return {
        key: sorted(rows, key=lambda item: int(item["label"]))
        for key, rows in grouped.items()
        if {str(row["modality"]) for row in rows} == set(LABEL_MAPPING)
    }


def split_training_patient_ids(patient_ids: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    if not patient_ids:
        return set(), set()
    if len(patient_ids) == 1:
        return set(patient_ids), set()
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1, got {train_ratio}")
    shuffled = list(patient_ids)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    return set(shuffled[:train_count]), set(shuffled[train_count:])


def build_split_rows(
    metadata_rows: list[dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    complete = successful_complete_patients(metadata_rows)
    training_ids = sorted(patient_id for split_source, patient_id in complete if split_source == "training")
    validation_ids = sorted(patient_id for split_source, patient_id in complete if split_source == "validation")
    train_ids, val_ids = split_training_patient_ids(training_ids, train_ratio, seed)

    split_rows = {"train": [], "val": [], "test": []}
    for (split_source, patient_id), rows in complete.items():
        if split_source == "training" and patient_id in train_ids:
            target_split = "train"
        elif split_source == "training" and patient_id in val_ids:
            target_split = "val"
        elif split_source == "validation" and patient_id in validation_ids:
            target_split = "test"
        else:
            continue

        for row in rows:
            split_rows[target_split].append(
                {
                    "patient_id": patient_id,
                    "image_path": row["processed_path"],
                    "label": int(row["label"]),
                    "modality": row["modality"],
                    "split_source": split_source,
                }
            )

    for rows in split_rows.values():
        rows.sort(key=lambda item: (str(item["patient_id"]), int(item["label"])))
    return split_rows


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patient_ids = {str(row["patient_id"]) for row in rows}
    return {
        "patients": len(patient_ids),
        "image_rows": len(rows),
        "modality_counts": dict(sorted(Counter(str(row["modality"]) for row in rows).items())),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
    }


def write_split_outputs(
    splits_output_dir: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    metadata_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    splits_output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_rows.items():
        write_csv(splits_output_dir / f"{split_name}_n4_only.csv", rows, CSV_COLUMNS)

    (splits_output_dir / "label_mapping.json").write_text(json.dumps(LABEL_MAPPING, indent=2) + "\n")
    successful_patients = successful_complete_patients(metadata_rows)
    summary = {
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "output_root": str(args.output_root.expanduser().resolve()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "selected_patients": len({row["patient_id"] for row in metadata_rows}),
        "complete_successful_patients": len(successful_patients),
        "failed_image_rows": sum(1 for row in metadata_rows if row["status"] == "failed"),
        "splits": {name: summarize_split(rows) for name, rows in split_rows.items()},
        "label_mapping": LABEL_MAPPING,
    }
    (splits_output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def header_summary(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(column: str) -> list[str]:
        return sorted({str(row[column]) for row in metadata_rows if row.get(column) not in {"", None}})

    return {
        "unique_original_shapes": values("original_shape"),
        "unique_processed_shapes": values("processed_shape"),
        "unique_original_spacings": values("original_spacing"),
        "unique_processed_spacings": values("processed_spacing"),
        "total_rows": len(metadata_rows),
        "total_processed": sum(1 for row in metadata_rows if row["status"] in {"success", "success_existing"}),
        "failed_count": sum(1 for row in metadata_rows if row["status"] == "failed"),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in metadata_rows).items())),
    }


def middle_slices(array: Any) -> list[Any]:
    return [
        array[array.shape[0] // 2, :, :],
        array[:, array.shape[1] // 2, :],
        array[:, :, array.shape[2] // 2],
    ]


def robust_window(arrays: list[Any], np: Any) -> tuple[float, float]:
    values = []
    for array in arrays:
        finite = array[np.isfinite(array)]
        nonzero = finite[finite != 0]
        if nonzero.size:
            values.append(nonzero.reshape(-1))
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    low, high = np.percentile(merged, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(np.min(merged)), float(np.max(merged) or 1.0)
    return float(low), float(high)


def add_slice(ax: Any, image: Any, title: str, vmin: float, vmax: float, np: Any) -> None:
    ax.imshow(np.rot90(image), cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def create_visualizations(metadata_rows: list[dict[str, Any]], output_root: Path, max_examples: int) -> None:
    if max_examples <= 0:
        return
    sitk = require_simpleitk()
    np, plt = require_visual_deps()

    examples = [row for row in metadata_rows if row["status"] in {"success", "success_existing"}][:max_examples]
    if not examples:
        return

    visual_dir = output_root.parent / "visual_debug"
    visual_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = []

    for row in examples:
        original = sitk.ReadImage(str(row["original_path"]), sitk.sitkFloat32)
        processed = sitk.ReadImage(str(row["processed_path"]), sitk.sitkFloat32)
        original_array = sitk.GetArrayFromImage(original)
        processed_array = sitk.GetArrayFromImage(processed)
        vmin, vmax = robust_window([original_array, processed_array], np)

        fig, axes = plt.subplots(2, 3, figsize=(9, 5.8), constrained_layout=True)
        for col, (view_name, original_slice, processed_slice) in enumerate(
            zip(["axial", "coronal", "sagittal"], middle_slices(original_array), middle_slices(processed_array))
        ):
            add_slice(axes[0, col], original_slice, f"original {view_name}", vmin, vmax, np)
            add_slice(axes[1, col], processed_slice, f"N4 {view_name}", vmin, vmax, np)
        fig.suptitle(
            f"{row['patient_id']} {row['modality']} | "
            f"shape {row['original_shape']} -> {row['processed_shape']} | "
            f"spacing {row['original_spacing']} -> {row['processed_spacing']}",
            fontsize=10,
        )
        fig.savefig(visual_dir / f"n4_before_after_{row['patient_id']}_{row['modality']}.png", dpi=180)
        plt.close(fig)
        grid_rows.append((row, original_array, processed_array, vmin, vmax))

    fig, axes = plt.subplots(len(grid_rows), 2, figsize=(7, max(2.2, 2.4 * len(grid_rows))), constrained_layout=True)
    if len(grid_rows) == 1:
        axes = np.array([axes])
    for row_index, (row, original_array, processed_array, vmin, vmax) in enumerate(grid_rows):
        add_slice(axes[row_index, 0], middle_slices(original_array)[0], f"{row['patient_id']} {row['modality']} original", vmin, vmax, np)
        add_slice(axes[row_index, 1], middle_slices(processed_array)[0], "N4 corrected", vmin, vmax, np)
    fig.savefig(visual_dir / "n4_before_after_grid.png", dpi=180)
    plt.close(fig)


def print_dry_run(tasks: list[ImageTask], args: argparse.Namespace) -> None:
    print("DRY RUN: no N4 preprocessing or files will be written.")
    print(f"dataset_root: {args.dataset_root.expanduser().resolve()}")
    print(f"output_root: {args.output_root.expanduser().resolve()}")
    print(f"splits_output_dir: {args.splits_output_dir.expanduser().resolve()}")
    print(f"selected_patients: {len({task.patient_id for task in tasks})}")
    print(f"image_tasks: {len(tasks)}")
    print("first_tasks:")
    for task in tasks[:12]:
        print(f"  {task.patient_id} {task.modality}: {task.original_path} -> {task.processed_path}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    splits_output_dir = args.splits_output_dir.expanduser().resolve()

    patient_entries = patient_dirs_for_args(args)
    tasks = build_tasks(patient_entries, output_root)
    if args.dry_run:
        print_dry_run(tasks, args)
        return

    metadata_rows = run_tasks(args, tasks)
    metadata_path = output_root.parent / "n4_only_metadata.csv"
    write_csv(metadata_path, metadata_rows, METADATA_COLUMNS)

    summary = header_summary(metadata_rows)
    summary_path = output_root.parent / "n4_only_header_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    split_rows = build_split_rows(metadata_rows, train_ratio=args.train_ratio, seed=args.seed)
    write_split_outputs(splits_output_dir, split_rows, metadata_rows, args)
    create_visualizations(metadata_rows, output_root, args.visualize_examples)

    print(f"wrote metadata: {metadata_path}")
    print(f"wrote header summary: {summary_path}")
    print(f"wrote split CSVs under: {splits_output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
