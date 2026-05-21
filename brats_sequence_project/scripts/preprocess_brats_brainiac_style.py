"""Create BrainIAC-style offline preprocessing-ablation inputs for BraTS.

This is a controlled ablation inspired by the public BrainIAC preprocessing
script. It is not guaranteed to reproduce the authors' internal preprocessing.
It writes new NIfTI files and never modifies the original BraTS dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
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
    "stage_n4",
    "stage_registration",
    "stage_hdbet",
    "template_path",
    "original_shape",
    "processed_shape",
    "original_spacing",
    "processed_spacing",
    "original_origin",
    "processed_origin",
    "original_direction",
    "processed_direction",
    "original_nonzero_count",
    "processed_nonzero_count",
    "original_mean_nonzero",
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
    parser.add_argument("--template_path", type=Path, default=None)
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--patient_ids", type=str, default=None, help="Comma-separated patient IDs to process.")
    parser.add_argument("--include_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1, help="Reserved for compatibility; processing is sequential.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--visualize_examples", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop on the first failed image.")
    parser.add_argument("--run_n4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_registration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_hdbet", action="store_true")
    parser.add_argument("--run_hdbet_on_already_skullstripped", action="store_true")
    parser.add_argument("--target_spacing", type=str, default="1,1,1")
    parser.add_argument("--registration_transform", choices=["rigid", "affine"], default="rigid")
    parser.add_argument("--registration_metric", choices=["mattes", "correlation"], default="mattes")
    parser.add_argument("--hdbet_device", type=str, default=None, help="Default: cuda device 0 if available, else cpu.")
    parser.add_argument("--n4_iterations", type=str, default="50,50,30,20")
    parser.add_argument("--n4_convergence_threshold", type=float, default=1e-7)
    parser.add_argument("--n4_shrink_factor", type=int, default=4)
    return parser.parse_args()


def require_simpleitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("SimpleITK is required for preprocessing. Install it with: pip install SimpleITK") from exc
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


def parse_triplet(value: str, name: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError(f"{name} must contain three positive comma-separated numbers, got {value!r}")
    return tuple(parts)  # type: ignore[return-value]


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


def validate_args(args: argparse.Namespace) -> None:
    if args.run_registration:
        if args.template_path is None:
            raise SystemExit("--template_path is required when --run_registration is enabled.")
        if not args.template_path.expanduser().is_file():
            raise SystemExit(f"--template_path does not exist: {args.template_path}")
    if args.run_hdbet:
        print(
            "WARNING: BraTS is already skull-stripped; HD-BET may remove valid brain/tumor regions. "
            "This branch is experimental."
        )
        if not args.run_hdbet_on_already_skullstripped:
            print(
                "WARNING: --run_hdbet_on_already_skullstripped was not passed. "
                "Proceeding because --run_hdbet was explicitly requested, but inspect outputs carefully."
            )
    if args.num_workers != 1:
        print("WARNING: BrainIAC-style preprocessing is sequential; --num_workers is accepted but ignored.")
    if args.n4_shrink_factor < 1:
        raise ValueError("--n4_shrink_factor must be >= 1")


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
        for patient_dir in discover_patient_dirs(dataset_root / TRAINING_SUBDIR, "BraTS20_Training_"):
            selected.append((patient_dir, "training", "BraTS2020_TrainingData"))
    if args.include_validation:
        for patient_dir in discover_patient_dirs(dataset_root / VALIDATION_SUBDIR, "BraTS20_Validation_"):
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
            suffix = MODALITY_SUFFIXES[modality].removesuffix(".nii")
            processed_path = output_dir / f"{patient_id}{suffix}_brainiacproc.nii.gz"
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
        "origin": [float(item) for item in image.GetOrigin()],
        "direction": [float(item) for item in image.GetDirection()],
        "nonzero_count": int(nonzero.size),
        "mean_nonzero": float(np.mean(nonzero)) if nonzero.size else None,
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


def resample_fixed_to_spacing(sitk: Any, fixed_image: Any, target_spacing: tuple[float, float, float]) -> Any:
    old_size = fixed_image.GetSize()
    old_spacing = fixed_image.GetSpacing()
    new_size = [
        int(round((old_size[index] * old_spacing[index]) / target_spacing[index]))
        for index in range(fixed_image.GetDimension())
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32))


def register_to_template(
    sitk: Any,
    moving_image: Any,
    fixed_image: Any,
    transform_kind: str,
    metric: str,
) -> Any:
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    if transform_kind == "rigid":
        base_transform = sitk.Euler3DTransform()
    else:
        base_transform = sitk.AffineTransform(3)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        base_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()
    if metric == "mattes":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == "correlation":
        registration_method.SetMetricAsCorrelation()
    else:  # pragma: no cover - argparse constrains values.
        raise ValueError(f"Unsupported registration metric: {metric}")

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)

    return sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def choose_hdbet_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def import_hdbet() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    local_preprocessing = repo_root / "src" / "preprocessing"
    if local_preprocessing.is_dir():
        sys.path.insert(0, str(local_preprocessing))
    try:
        from HD_BET.hd_bet import hd_bet
    except ImportError as exc:  # pragma: no cover - depends on optional dependency.
        raise ImportError(
            "HD-BET was requested but is not importable. Install/configure HD-BET first, for example "
            "`pip install HD-BET`, or use the BrainIAC repo's src/preprocessing/HD_BET package on PYTHONPATH."
        ) from exc
    return hd_bet


def run_hdbet_on_image(sitk: Any, image: Any, task: ImageTask, device: str) -> Any:
    hd_bet = import_hdbet()
    with tempfile.TemporaryDirectory(prefix="brainiac_hdbet_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        input_path = input_dir / f"{task.patient_id}_{task.modality}_0000.nii.gz"
        sitk.WriteImage(image, str(input_path))
        hd_bet(str(input_dir), str(output_dir), mode="fast", device=device, tta=0, save_mask=0, overwrite_existing=1)
        candidates = sorted(output_dir.glob("*.nii.gz"))
        if not candidates:
            raise RuntimeError("HD-BET completed but did not produce a .nii.gz output.")
        output_image = sitk.ReadImage(str(candidates[0]), sitk.sitkFloat32)
        return output_image


def metadata_row_from_stats(
    task: ImageTask,
    args: argparse.Namespace,
    original_stats: dict[str, Any] | None,
    processed_stats: dict[str, Any] | None,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    template_path = str(args.template_path.expanduser().resolve()) if args.template_path else ""
    return {
        "patient_id": task.patient_id,
        "split_source": task.split_source,
        "modality": task.modality,
        "label": task.label,
        "original_path": str(task.original_path),
        "processed_path": str(task.processed_path),
        "stage_n4": bool(args.run_n4),
        "stage_registration": bool(args.run_registration),
        "stage_hdbet": bool(args.run_hdbet),
        "template_path": template_path,
        "original_shape": json.dumps(original_stats["shape"]) if original_stats else "",
        "processed_shape": json.dumps(processed_stats["shape"]) if processed_stats else "",
        "original_spacing": json.dumps(original_stats["spacing"]) if original_stats else "",
        "processed_spacing": json.dumps(processed_stats["spacing"]) if processed_stats else "",
        "original_origin": json.dumps(original_stats["origin"]) if original_stats else "",
        "processed_origin": json.dumps(processed_stats["origin"]) if processed_stats else "",
        "original_direction": json.dumps(original_stats["direction"]) if original_stats else "",
        "processed_direction": json.dumps(processed_stats["direction"]) if processed_stats else "",
        "original_nonzero_count": original_stats["nonzero_count"] if original_stats else "",
        "processed_nonzero_count": processed_stats["nonzero_count"] if processed_stats else "",
        "original_mean_nonzero": original_stats["mean_nonzero"] if original_stats else "",
        "processed_mean_nonzero": processed_stats["mean_nonzero"] if processed_stats else "",
        "status": status,
        "error_message": error_message,
    }


def process_one_task(
    sitk: Any,
    task: ImageTask,
    args: argparse.Namespace,
    fixed_template: Any | None,
    iterations: list[int],
    hdbet_device: str,
    capture_stages: bool,
) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    stages: list[tuple[str, Any]] = []
    try:
        original = sitk.ReadImage(str(task.original_path), sitk.sitkFloat32)
        original_stats = image_stats(sitk, original)
        if task.processed_path.exists() and not args.force:
            processed = sitk.ReadImage(str(task.processed_path), sitk.sitkFloat32)
            processed_stats = image_stats(sitk, processed)
            if capture_stages:
                stages = [("original", original), ("existing final", processed)]
            return metadata_row_from_stats(task, args, original_stats, processed_stats, "success_existing"), stages

        image = original
        if capture_stages:
            stages.append(("original", image))

        if args.run_n4:
            image = n4_bias_correct(sitk, image, iterations, args.n4_convergence_threshold, args.n4_shrink_factor)
            if capture_stages:
                stages.append(("after N4", image))

        if args.run_registration:
            if fixed_template is None:
                raise ValueError("Registration requested but no fixed template was loaded.")
            image = register_to_template(
                sitk,
                moving_image=image,
                fixed_image=fixed_template,
                transform_kind=args.registration_transform,
                metric=args.registration_metric,
            )
            if capture_stages:
                stages.append(("after registration", image))

        if args.run_hdbet:
            image = run_hdbet_on_image(sitk, image, task, hdbet_device)
            if capture_stages:
                stages.append(("after HD-BET", image))

        task.processed_path.parent.mkdir(parents=True, exist_ok=True)
        image = sitk.Cast(image, sitk.sitkFloat32)
        sitk.WriteImage(image, str(task.processed_path))
        processed = sitk.ReadImage(str(task.processed_path), sitk.sitkFloat32)
        processed_stats = image_stats(sitk, processed)
        if capture_stages:
            stages.append(("final", processed))
        return metadata_row_from_stats(task, args, original_stats, processed_stats, "success"), stages
    except Exception as exc:  # noqa: BLE001 - failures are part of the audit surface.
        return metadata_row_from_stats(task, args, None, None, "failed", str(exc)), stages


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


def build_split_rows(metadata_rows: list[dict[str, Any]], train_ratio: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    complete = successful_complete_patients(metadata_rows)
    training_ids = sorted(patient_id for split_source, patient_id in complete if split_source == "training")
    validation_ids = sorted(patient_id for split_source, patient_id in complete if split_source == "validation")
    train_ids, val_ids = split_training_patient_ids(training_ids, train_ratio, seed)

    split_rows = {"train": [], "val": [], "test": []}
    for (split_source, patient_id), rows in complete.items():
        if split_source == "training" and patient_id in train_ids:
            split_name = "train"
        elif split_source == "training" and patient_id in val_ids:
            split_name = "val"
        elif split_source == "validation" and patient_id in validation_ids:
            split_name = "test"
        else:
            continue
        for row in rows:
            split_rows[split_name].append(
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
    return {
        "patients": len({str(row["patient_id"]) for row in rows}),
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
        write_csv(splits_output_dir / f"{split_name}_brainiac_style.csv", rows, CSV_COLUMNS)
    (splits_output_dir / "label_mapping.json").write_text(json.dumps(LABEL_MAPPING, indent=2) + "\n")

    summary = {
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "output_root": str(args.output_root.expanduser().resolve()),
        "template_path": str(args.template_path.expanduser().resolve()) if args.template_path else "",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "stages": {
            "run_n4": bool(args.run_n4),
            "run_registration": bool(args.run_registration),
            "run_hdbet": bool(args.run_hdbet),
            "registration_transform": args.registration_transform,
            "registration_metric": args.registration_metric,
            "target_spacing": args.target_spacing,
        },
        "selected_patients": len({row["patient_id"] for row in metadata_rows}),
        "complete_successful_patients": len(successful_complete_patients(metadata_rows)),
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
        "unique_original_origins": values("original_origin"),
        "unique_processed_origins": values("processed_origin"),
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


def create_stage_visualization(
    sitk: Any,
    visual_dir: Path,
    metadata_row: dict[str, Any],
    stages: list[tuple[str, Any]],
) -> tuple[dict[str, Any], Any, Any, float, float] | None:
    if not stages:
        return None
    np, plt = require_visual_deps()
    arrays = [(name, sitk.GetArrayFromImage(image)) for name, image in stages]
    vmin, vmax = robust_window([array for _, array in arrays], np)

    fig, axes = plt.subplots(len(arrays), 3, figsize=(9, max(2.4, 2.0 * len(arrays))), constrained_layout=True)
    if len(arrays) == 1:
        axes = np.array([axes])
    for row_index, (stage_name, array) in enumerate(arrays):
        for col, (view_name, image_slice) in enumerate(zip(["axial", "coronal", "sagittal"], middle_slices(array))):
            add_slice(axes[row_index, col], image_slice, f"{stage_name} {view_name}", vmin, vmax, np)
    fig.suptitle(
        f"{metadata_row['patient_id']} {metadata_row['modality']} | "
        f"shape {metadata_row['original_shape']} -> {metadata_row['processed_shape']}",
        fontsize=10,
    )
    path = visual_dir / f"brainiac_style_before_after_{metadata_row['patient_id']}_{metadata_row['modality']}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    first_array = arrays[0][1]
    last_array = arrays[-1][1]
    return metadata_row, first_array, last_array, vmin, vmax


def create_grid_visualization(grid_items: list[tuple[dict[str, Any], Any, Any, float, float]], visual_dir: Path) -> None:
    if not grid_items:
        return
    np, plt = require_visual_deps()
    fig, axes = plt.subplots(len(grid_items), 2, figsize=(7, max(2.2, 2.4 * len(grid_items))), constrained_layout=True)
    if len(grid_items) == 1:
        axes = np.array([axes])
    for row_index, (row, original_array, final_array, vmin, vmax) in enumerate(grid_items):
        add_slice(axes[row_index, 0], middle_slices(original_array)[0], f"{row['patient_id']} {row['modality']} original", vmin, vmax, np)
        add_slice(axes[row_index, 1], middle_slices(final_array)[0], "final processed", vmin, vmax, np)
    fig.savefig(visual_dir / "brainiac_style_before_after_grid.png", dpi=180)
    plt.close(fig)


def run_tasks(args: argparse.Namespace, tasks: list[ImageTask]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], Any, Any, float, float]]]:
    try:
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError("tqdm is required for progress reporting. Install it with: pip install tqdm") from exc

    sitk = require_simpleitk()
    iterations = parse_iterations(args.n4_iterations)
    target_spacing = parse_triplet(args.target_spacing, "--target_spacing")
    fixed_template = None
    if args.run_registration:
        fixed_template = sitk.ReadImage(str(args.template_path.expanduser().resolve()), sitk.sitkFloat32)
        fixed_template = resample_fixed_to_spacing(sitk, fixed_template, target_spacing)

    hdbet_device = choose_hdbet_device(args.hdbet_device)
    if args.run_hdbet:
        import_hdbet()

    visual_dir = args.output_root.expanduser().resolve().parent / "visual_debug"
    visual_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []
    grid_items: list[tuple[dict[str, Any], Any, Any, float, float]] = []

    for task in tqdm(tasks, desc="BrainIAC-style preprocessing"):
        capture_stages = len(grid_items) < args.visualize_examples
        row, stages = process_one_task(sitk, task, args, fixed_template, iterations, hdbet_device, capture_stages)
        metadata_rows.append(row)
        if args.strict and row["status"] == "failed":
            raise RuntimeError(f"Failed processing {task.original_path}: {row['error_message']}")
        if capture_stages and row["status"] in {"success", "success_existing"}:
            item = create_stage_visualization(sitk, visual_dir, row, stages)
            if item is not None:
                grid_items.append(item)

    return metadata_rows, grid_items


def print_dry_run(tasks: list[ImageTask], args: argparse.Namespace) -> None:
    print("DRY RUN: no preprocessing or files will be written.")
    print("This is a BrainIAC-style ablation, not a guaranteed exact reproduction of author preprocessing.")
    print(f"dataset_root: {args.dataset_root.expanduser().resolve()}")
    print(f"output_root: {args.output_root.expanduser().resolve()}")
    print(f"splits_output_dir: {args.splits_output_dir.expanduser().resolve()}")
    print(f"template_path: {args.template_path.expanduser().resolve() if args.template_path else ''}")
    print(
        "stages: "
        f"N4={args.run_n4}, registration={args.run_registration}, HD-BET={args.run_hdbet}, "
        f"transform={args.registration_transform}, metric={args.registration_metric}, spacing={args.target_spacing}"
    )
    print(f"selected_patients: {len({task.patient_id for task in tasks})}")
    print(f"image_tasks: {len(tasks)}")
    print("first_tasks:")
    for task in tasks[:12]:
        print(f"  {task.patient_id} {task.modality}: {task.original_path} -> {task.processed_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_root = args.output_root.expanduser().resolve()
    splits_output_dir = args.splits_output_dir.expanduser().resolve()

    patient_entries = patient_dirs_for_args(args)
    tasks = build_tasks(patient_entries, output_root)
    if args.dry_run:
        print_dry_run(tasks, args)
        return

    metadata_rows, grid_items = run_tasks(args, tasks)
    metadata_path = output_root.parent / "brainiac_style_metadata.csv"
    write_csv(metadata_path, metadata_rows, METADATA_COLUMNS)

    summary = header_summary(metadata_rows)
    summary_path = output_root.parent / "brainiac_style_header_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    split_rows = build_split_rows(metadata_rows, train_ratio=args.train_ratio, seed=args.seed)
    write_split_outputs(splits_output_dir, split_rows, metadata_rows, args)
    create_grid_visualization(grid_items, output_root.parent / "visual_debug")

    if output_root.joinpath("temp_registered").exists():
        shutil.rmtree(output_root / "temp_registered")

    print(f"wrote metadata: {metadata_path}")
    print(f"wrote header summary: {summary_path}")
    print(f"wrote split CSVs under: {splits_output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
