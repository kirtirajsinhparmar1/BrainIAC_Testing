#!/usr/bin/env python3
"""Trace raw, deterministic, and actual fine-tuning MRI inputs for review.

The final batch is loaded through the same BraTSFineTuneDataset and seeded
training DataLoader used by finetune_model.py. No model or checkpoint is
constructed.
"""

from __future__ import annotations

import argparse
import csv
import enum
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from finetune_common import (
    CLASS_NAMES,
    REPO_ROOT,
    BraTSFineTuneDataset,
    get_brainiac_train_transform,
    load_yaml_config,
    read_csv_rows,
    resolve_repo_path,
    validate_finetune_config,
)
from finetune_model import build_train_loader, seed_everything


STAT_COLUMNS = (
    "patient_id",
    "modality",
    "class_id",
    "source_path",
    "raw_shape",
    "raw_dtype",
    "raw_min",
    "raw_max",
    "raw_mean",
    "raw_std",
    "preprocessed_shape",
    "preprocessed_dtype",
    "preprocessed_min",
    "preprocessed_max",
    "preprocessed_mean",
    "preprocessed_std",
    "final_shape",
    "final_min",
    "final_max",
    "final_mean",
    "final_std",
)
TRANSFORM_CONFIG_PROPERTIES = (
    "image_only",
    "dtype",
    "reader",
    "channel_dim",
    "strict_check",
    "spatial_size",
    "mode",
    "align_corners",
    "anti_aliasing",
    "anti_aliasing_sigma",
    "nonzero",
    "channel_wise",
    "subtrahend",
    "divisor",
    "rotate_range",
    "translate_range",
    "scale_range",
    "shear_range",
    "prob",
    "padding_mode",
    "spatial_axis",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "mean",
    "std",
    "gamma",
    "invert_image",
    "method",
    "minv",
    "maxv",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/config/data_efficiency.yml"),
    )
    parser.add_argument("--train-csv", help="Training subset CSV; defaults to config data.train_csv")
    parser.add_argument("--seed", type=int, help="Defaults to experiment.seed in the config")
    parser.add_argument("--output-dir", required=True, help="Directory for visualization artifacts")
    parser.add_argument("--batch-index", type=int, default=0, help="Shuffled training batch to inspect")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Maximum samples to draw in figures; the entire selected batch is recorded in metadata",
    )
    args = parser.parse_args(argv)
    if args.batch_index < 0:
        parser.error("--batch-index must be zero or greater")
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")
    return args


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    return output_dir.resolve()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def array_statistics(value: Any) -> dict[str, Any]:
    array = _to_numpy(value)
    return {
        "shape": [int(size) for size in array.shape],
        "dtype": str(array.dtype),
        "min": _number(np.min(array)),
        "max": _number(np.max(array)),
        "mean": _number(np.mean(array)),
        "std": _number(np.std(array)),
    }


def _path_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _spatial_volume(array: np.ndarray, source_path: str) -> np.ndarray:
    """Return a NIfTI's 3 spatial dimensions, retaining singleton spatial axes."""

    if array.ndim < 3:
        raise ValueError(f"Expected a 3D MRI volume in {source_path}; found shape {array.shape}")
    if any(size != 1 for size in array.shape[3:]):
        raise ValueError(
            f"Expected one 3D MRI volume in {source_path}; found non-singleton trailing dimensions "
            f"in shape {array.shape}"
        )
    if array.ndim == 3:
        return array
    return array[(slice(None), slice(None), slice(None)) + (0,) * (array.ndim - 3)]


def _display_orientation(volume: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Reorder/flip voxel axes for display only; do not interpolate MRI values."""

    import nibabel as nib

    start = nib.orientations.io_orientation(affine)
    ras = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(start, ras)
    oriented = nib.orientations.apply_orientation(volume, transform)
    display_affine = affine @ nib.orientations.inv_ornt_aff(transform, volume.shape)
    spacing = tuple(
        float(np.linalg.norm(display_affine[:3, axis])) for axis in range(3)
    )
    return oriented, spacing


def load_raw_nifti(source_path: str) -> dict[str, Any]:
    """Read source voxels directly and retain original metadata for statistics."""

    import nibabel as nib

    image = nib.load(str(Path(source_path).expanduser()))
    raw = np.asanyarray(image.dataobj)
    spatial = _spatial_volume(raw, source_path)
    affine = np.asarray(image.affine, dtype=np.float64)
    display_volume, spacing = _display_orientation(spatial, affine)
    return {
        "raw": raw,
        "shape": [int(size) for size in raw.shape],
        "dtype": str(raw.dtype),
        "statistics": array_statistics(raw),
        "affine": affine,
        "display_volume": display_volume,
        "display_spacing": spacing,
    }


def _transform_items(transform: Any) -> list[Any]:
    items = getattr(transform, "transforms", None)
    if items is None:
        raise TypeError(f"Expected an inspectable Compose transform, got {type(transform).__name__}")
    return list(items)


def _is_random_transform(transform: Any) -> bool:
    return type(transform).__name__.lower().startswith(("rand", "random"))


def inspect_transform_stages(train_transform: Any) -> tuple[list[Any], list[Any]]:
    """Extract the deterministic prefix and random stages from the training transform."""
    train_items = _transform_items(train_transform)
    first_random = next(
        (index for index, item in enumerate(train_items) if _is_random_transform(item)),
        len(train_items),
    )
    deterministic_prefix = train_items[:first_random]
    tensor_transform = next(
        (item for item in train_items if type(item).__name__ == "ToTensord"),
        None,
    )
    if tensor_transform is None:
        raise RuntimeError("The existing training transform has no ToTensord conversion")
    preprocessing_items = list(deterministic_prefix)
    if tensor_transform not in preprocessing_items:
        preprocessing_items.append(tensor_transform)
    random_items = [item for item in train_items if _is_random_transform(item)]
    return preprocessing_items, random_items


def _format_config_value(value: Any, depth: int = 0) -> str:
    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, type):
        return value.__name__
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{key!r}: {_format_config_value(item, depth + 1)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ) + "}"
    if isinstance(value, (list, tuple)):
        opener, closer = ("[", "]") if isinstance(value, list) else ("(", ")")
        items = ", ".join(_format_config_value(item, depth + 1) for item in value)
        if isinstance(value, tuple) and len(value) == 1:
            items += ","
        return f"{opener}{items}{closer}"
    if inspect.isroutine(value):
        return getattr(value, "__name__", type(value).__name__)
    attributes = getattr(value, "__dict__", {}) or {}
    if depth < 2:
        stable_attributes = {
            key: item
            for key, item in attributes.items()
            if key not in {"R", "_do_transform"}
        }
        for name in TRANSFORM_CONFIG_PROPERTIES:
            if name not in stable_attributes:
                try:
                    if hasattr(value, name):
                        stable_attributes[name] = getattr(value, name)
                except Exception:
                    pass
        if stable_attributes:
            rendered = ", ".join(
                f"{key.lstrip('_')}={_format_config_value(item, depth + 1)}"
                for key, item in sorted(stable_attributes.items())
            )
            return f"{type(value).__name__}({rendered})"
    return type(value).__name__


def _transform_lines(transform: Any) -> list[str]:
    lines = []
    for item in _transform_items(transform):
        attributes = vars(item)
        stable_attributes = {
            key: value
            for key, value in attributes.items()
            if key not in {"R", "_do_transform"}
        }
        rendered = ", ".join(
            f"{key.lstrip('_')}={_format_config_value(value)}"
            for key, value in sorted(stable_attributes.items())
        )
        lines.append(f"{type(item).__name__}({rendered})")
    return lines


def _metadata_values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _sample_batch_parts(batch: Any) -> tuple[Any, list[int], list[str], list[str], list[str]]:
    if not isinstance(batch, (list, tuple)) or len(batch) != 5:
        raise ValueError(
            "Expected the fine-tuning dataset batch (images, labels, patient IDs, modalities, paths)"
        )
    images, labels, patient_ids, modalities, source_paths = batch
    label_values = [int(label) for label in _metadata_values(labels)]
    patient_values = [str(value) for value in _metadata_values(patient_ids)]
    modality_values = [str(value).upper() for value in _metadata_values(modalities)]
    path_values = [str(value) for value in _metadata_values(source_paths)]
    lengths = {len(label_values), len(patient_values), len(modality_values), len(path_values)}
    if len(lengths) != 1:
        raise ValueError("Training batch metadata fields have inconsistent lengths")
    return images, label_values, patient_values, modality_values, path_values


def build_batch_metadata(
    images: Any,
    labels: Sequence[int],
    patient_ids: Sequence[str],
    modalities: Sequence[str],
    source_paths: Sequence[str],
    *,
    seed: int,
    train_csv: str,
    batch_size_configured: int,
    batch_index: int,
    random_transform_names: Sequence[str],
) -> dict[str, Any]:
    image_array = _to_numpy(images)
    if image_array.ndim < 2 or image_array.shape[0] != len(labels):
        raise ValueError("Image batch and sample metadata have different batch sizes")
    samples = []
    for index, (label, patient_id, modality, source_path) in enumerate(
        zip(labels, patient_ids, modalities, source_paths)
    ):
        if label < 0 or label >= len(CLASS_NAMES) or CLASS_NAMES[label] != modality:
            raise ValueError(f"Batch label/modality mismatch at index {index}: {label}, {modality}")
        stats = array_statistics(image_array[index])
        samples.append(
            {
                "batch_index": index,
                "patient_id": patient_id,
                "modality": modality,
                "class_id": int(label),
                "source_path": source_path,
                "sample_shape": stats["shape"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
            }
        )
    return {
        "seed": int(seed),
        "train_csv": train_csv,
        "batch_index": int(batch_index),
        "batch_size_configured": int(batch_size_configured),
        "actual_batch_size": int(image_array.shape[0]),
        "batch_shape": [int(size) for size in image_array.shape],
        "dtype": str(image_array.dtype),
        "random_augmentation_applied": bool(random_transform_names),
        "random_augmentation_transforms": list(random_transform_names),
        "samples": samples,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _number(value)
    return value


def _center_slices(volume: np.ndarray, spacing: Sequence[float] | None = None) -> list[tuple[str, np.ndarray, float]]:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume for plotting, found shape {volume.shape}")
    x_size, y_size, z_size = volume.shape
    sx, sy, sz = spacing if spacing is not None else (1.0, 1.0, 1.0)
    return [
        ("Axial", volume[:, :, z_size // 2].T, sy / sx),
        ("Coronal", volume[:, y_size // 2, :].T, sz / sx),
        ("Sagittal", volume[x_size // 2, :, :].T, sz / sy),
    ]


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + max(abs(float(low)) * 1e-6, 1e-6)
    return float(low), float(high)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_triplanar_figure(
    samples: Sequence[Mapping[str, Any]],
    volume_key: str,
    output_path: Path,
    *,
    title: str,
    shape_key: str,
    spacing_key: str | None = None,
) -> None:
    plt = _pyplot()
    if not samples:
        raise ValueError("No samples were selected for a triplanar figure")
    columns = min(4, len(samples))
    groups = math.ceil(len(samples) / columns)
    fig, axes = plt.subplots(groups * 3, columns, figsize=(4.0 * columns, 3.25 * groups * 3), squeeze=False)
    plane_names = ("Axial", "Coronal", "Sagittal")
    for index, sample in enumerate(samples):
        group, column = divmod(index, columns)
        volume = np.asarray(sample[volume_key])
        spacing = sample.get(spacing_key) if spacing_key else None
        slices = _center_slices(volume, spacing)
        header = (
            f"{sample['modality']} (class {sample['class_id']})\n"
            f"Patient {sample['patient_id']} | shape {sample[shape_key]}"
        )
        for plane_index, (plane, image, aspect) in enumerate(slices):
            axis = axes[group * 3 + plane_index, column]
            low, high = _display_limits(image)
            axis.imshow(image, cmap="gray", vmin=low, vmax=high, aspect=aspect, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if plane_index == 0:
                axis.set_title(header, fontsize=10)
            if column == 0:
                axis.set_ylabel(plane, fontsize=11, rotation=0, labelpad=38, va="center")
    for index in range(len(samples), groups * columns):
        group, column = divmod(index, columns)
        for plane_index in range(3):
            axes[group * 3 + plane_index, column].axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_actual_batch_figure(
    samples: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    batch_shape: Sequence[int],
) -> None:
    plt = _pyplot()
    if not samples:
        raise ValueError("No samples were selected for the actual-batch figure")
    columns = 2 if len(samples) <= 4 else 4
    rows = math.ceil(len(samples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.1 * rows), squeeze=False)
    for index, sample in enumerate(samples):
        axis = axes[index // columns, index % columns]
        slices = _center_slices(np.asarray(sample["final_volume"]))
        axial = slices[0][1]
        low, high = _display_limits(axial)
        axis.imshow(axial, cmap="gray", vmin=low, vmax=high, aspect="equal", interpolation="nearest")
        axis.set_title(
            f"Batch sample {sample['batch_index']} | {sample['modality']} (class {sample['class_id']})\n"
            f"Patient {sample['patient_id']}",
            fontsize=11,
        )
        axis.axis("off")
    for index in range(len(samples), rows * columns):
        axes[index // columns, index % columns].axis("off")
    fig.suptitle(f"Actual augmented training input | batch shape {list(batch_shape)}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pair_figure(
    samples: Sequence[Mapping[str, Any]],
    left_key: str,
    right_key: str,
    output_path: Path,
    *,
    title: str,
    left_title: str,
    right_title: str,
    left_spacing_key: str | None = None,
    right_spacing_key: str | None = None,
) -> None:
    plt = _pyplot()
    if not samples:
        raise ValueError("No samples were selected for a paired figure")
    fig, axes = plt.subplots(len(samples), 2, figsize=(9.0, 4.0 * len(samples)), squeeze=False)
    for index, sample in enumerate(samples):
        for column, (key, panel_title, spacing_key) in enumerate(
            (
                (left_key, left_title, left_spacing_key),
                (right_key, right_title, right_spacing_key),
            )
        ):
            axis = axes[index, column]
            volume = np.asarray(sample[key])
            spacing = sample.get(spacing_key) if spacing_key else None
            _, axial, aspect = _center_slices(volume, spacing)[0]
            low, high = _display_limits(axial)
            axis.imshow(axial, cmap="gray", vmin=low, vmax=high, aspect=aspect, interpolation="nearest")
            if index == 0:
                axis.set_title(panel_title, fontsize=12)
            if column == 0:
                axis.set_ylabel(
                    f"{sample['modality']} (class {sample['class_id']})\nPatient {sample['patient_id']}",
                    fontsize=10,
                )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _tensor_volume(tensor: Any, context: str) -> np.ndarray:
    array = _to_numpy(tensor)
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(
            f"Expected one-channel [C,D,H,W] data for {context}, got shape {array.shape}"
        )
    return np.asarray(array[0])


def _write_statistics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _pipeline_summary(
    *,
    config_path: Path,
    train_csv: Path,
    rows: Sequence[Mapping[str, Any]],
    data_config: Mapping[str, Any],
    seed: int,
    batch_metadata: Mapping[str, Any],
    train_transform: Any,
    preprocessing_transform: Any,
    random_transform_names: Sequence[str],
) -> str:
    lines = [
        "BrainIAC input visualization pipeline",
        "",
        "No model/checkpoint was instantiated. The actual batch was loaded from the training Dataset/DataLoader path.",
        f"Config: {config_path}",
        f"Training CSV: {train_csv}",
        f"Seed: {seed}",
        f"Patients: {len({str(row['patient_id']) for row in rows})}",
        f"Patient IDs: {', '.join(sorted({str(row['patient_id']) for row in rows}))}",
        f"Training scans: {len(rows)}",
        f"Configured batch size: {data_config['batch_size']}",
        f"Selected batch index: {batch_metadata['batch_index']}",
        f"Actual selected batch size: {batch_metadata['actual_batch_size']}",
        f"Actual selected batch shape: {batch_metadata['batch_shape']}",
        f"Actual batch dtype: {batch_metadata['dtype']}",
        "",
        "Raw data and deterministic preprocessing:",
        "  RAW NIfTI source path -> direct NIfTI read for raw statistics/figures",
        "  -> deterministic prefix extracted from the existing training transform",
        "  -> BraTSFineTuneDataset.__getitem__()",
        "",
        "Actual training input:",
        "  RAW NIfTI -> official get_brainiac_train_transform() ->",
    ]
    for index, description in enumerate(_transform_lines(train_transform), start=1):
        lines.append(f"    {index}. {description}")
    lines.extend(
        [
            "  -> BraTSFineTuneDataset.__getitem__() returns image, label, patient_id, modality, source_path",
            "  -> build_train_loader() shared with finetune_model.py",
            f"     batch_size={data_config['batch_size']}, shuffle=True, num_workers={data_config['num_workers']}, "
            f"pin_memory={bool(data_config.get('pin_memory', False))}, seed={seed}",
            "  -> actual batch image tensor (the tensor consumed by BrainIAC in run_epoch)",
            "",
            "Deterministic transform used for Stage B (existing training-transform objects, before the first random transform):",
        ]
    )
    for index, description in enumerate(_transform_lines(preprocessing_transform), start=1):
        lines.append(f"  {index}. {description}")
    lines.extend(
        [
            "",
            "The deterministic preprocessing transform is composed from the same transform objects used by the training Dataset, stopping before its first random transform and retaining the same ToTensord conversion.",
            f"Random augmentation pipeline present: {bool(random_transform_names)}",
            "Random augmentation transforms: "
            + (", ".join(random_transform_names) if random_transform_names else "none"),
            "The random transforms are part of the actual batch path; their configured probabilities mean each operation may or may not fire for a given sample.",
            "Seeded with the existing fine-tuning seed, worker initialization, and DataLoader generator functions.",
            "",
            "Display notes:",
            "  Raw NIfTI arrays are shown in closest-RAS axis order using permutation/flips only; no resampling is done for display.",
            "  Tensor center planes use the source NIfTI axis orientation and are labeled by anatomical plane.",
            "  Display window uses each plotted slice's finite 1st-99th intensity percentiles; voxel values/statistics are unchanged.",
            "  Axes use nearest-neighbor image rendering with physical aspect for raw voxel spacing and equal pixel aspect for 96-cube tensors.",
            "  raw_vs_final_batch pairs by the source path returned by the same training Dataset; sample order is the actual shuffled batch order.",
            "",
            "Configured batch size is an upper bound; the actual selected batch can be smaller for a short subset or final batch.",
        ]
    )
    return "\n".join(lines) + "\n"


def _get_batch(loader: Any, batch_index: int) -> Any:
    iterator = iter(loader)
    batch = None
    for _ in range(batch_index + 1):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise IndexError(f"Training loader has no batch at index {batch_index}") from exc
    return batch


def run_visualization(args: argparse.Namespace) -> list[Path]:
    config_path = resolve_repo_path(args.config)
    config = load_yaml_config(config_path)
    validate_finetune_config(config)
    data_config = config["data"]
    train_csv = resolve_repo_path(args.train_csv or data_config["train_csv"])
    rows = read_csv_rows(train_csv, validate_paths=True)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    output_dir = ensure_output_dir(resolve_repo_path(args.output_dir))

    # Match the fine-tuning seed/transform/dataset/loader path. No model is built.
    seed_everything(seed)
    train_transform = get_brainiac_train_transform()
    preprocessing_items, random_transforms = inspect_transform_stages(train_transform)
    random_names = [type(item).__name__ for item in random_transforms]
    from monai.transforms import Compose

    preprocessing_transform = Compose(preprocessing_items)

    train_dataset = BraTSFineTuneDataset(rows, train_transform, validate_paths=False)
    train_loader = build_train_loader(train_dataset, data_config, seed)
    batch = _get_batch(train_loader, args.batch_index)
    images, labels, patient_ids, modalities, source_paths = _sample_batch_parts(batch)
    batch_metadata = build_batch_metadata(
        images,
        labels,
        patient_ids,
        modalities,
        source_paths,
        seed=seed,
        train_csv=str(train_csv),
        batch_size_configured=int(data_config["batch_size"]),
        batch_index=args.batch_index,
        random_transform_names=random_names,
    )
    image_batch = _to_numpy(images)
    if image_batch.ndim != 5 or image_batch.shape[1] != 1:
        raise ValueError(
            "Expected the fine-tuning batch to have shape [B,1,D,H,W]; "
            f"found {image_batch.shape}"
        )

    selected_count = len(source_paths)
    if args.num_samples is not None:
        selected_count = min(selected_count, args.num_samples)
    display_paths = {_path_key(path) for path in source_paths[:selected_count]}
    raw_display: dict[str, dict[str, Any]] = {}
    statistics_rows: list[dict[str, Any]] = []
    deterministic_dataset = BraTSFineTuneDataset(
        rows, preprocessing_transform, validate_paths=False
    )

    for dataset_index, row in enumerate(rows):
        source_path = str(row["image_path"])
        key = _path_key(source_path)
        raw = load_raw_nifti(source_path)
        deterministic_image, _, _, _, _ = deterministic_dataset[dataset_index]
        deterministic_array = _to_numpy(deterministic_image)
        deterministic_stats = array_statistics(deterministic_array)
        record = {
            "patient_id": str(row["patient_id"]),
            "modality": str(row["modality"]).upper(),
            "class_id": int(row["label"]),
            "source_path": source_path,
            "raw_shape": json.dumps(raw["shape"]),
            "raw_dtype": raw["dtype"],
            "raw_min": raw["statistics"]["min"],
            "raw_max": raw["statistics"]["max"],
            "raw_mean": raw["statistics"]["mean"],
            "raw_std": raw["statistics"]["std"],
            "preprocessed_shape": json.dumps(deterministic_stats["shape"]),
            "preprocessed_dtype": deterministic_stats["dtype"],
            "preprocessed_min": deterministic_stats["min"],
            "preprocessed_max": deterministic_stats["max"],
            "preprocessed_mean": deterministic_stats["mean"],
            "preprocessed_std": deterministic_stats["std"],
            "final_shape": "",
            "final_min": "",
            "final_max": "",
            "final_mean": "",
            "final_std": "",
        }
        if key in display_paths:
            raw_display[key] = {
                "raw_volume": raw["display_volume"],
                "raw_spacing": raw["display_spacing"],
                "raw_shape_label": raw["shape"],
                "preprocessed_volume": _display_orientation(
                    _tensor_volume(deterministic_image, "deterministic preprocessing"),
                    raw["affine"],
                )[0],
                "preprocessed_shape_label": deterministic_stats["shape"],
                "affine": raw["affine"],
            }
        statistics_rows.append(record)

    batch_samples: list[dict[str, Any]] = []
    record_by_path = {_path_key(str(row["image_path"])): row for row in statistics_rows}
    raw_display_by_path = raw_display
    for index, (label, patient_id, modality, source_path) in enumerate(
        zip(labels, patient_ids, modalities, source_paths)
    ):
        key = _path_key(source_path)
        if key not in record_by_path:
            raise RuntimeError(f"Batch source path cannot be matched to the input CSV: {source_path}")
        sample_tensor = images[index]
        final_stats = array_statistics(sample_tensor)
        stats_row = record_by_path[key]
        stats_row.update(
            {
                "final_shape": json.dumps(final_stats["shape"]),
                "final_min": final_stats["min"],
                "final_max": final_stats["max"],
                "final_mean": final_stats["mean"],
                "final_std": final_stats["std"],
            }
        )
        final_volume = _tensor_volume(sample_tensor, "actual training batch")
        raw_sample = raw_display_by_path.get(key)
        if raw_sample is not None:
            display_final, _ = _display_orientation(final_volume, raw_sample["affine"])
            batch_samples.append(
                {
                    "batch_index": index,
                    "patient_id": patient_id,
                    "modality": modality,
                    "class_id": label,
                    "source_path": source_path,
                    "raw_volume": raw_sample["raw_volume"],
                    "raw_spacing": raw_sample["raw_spacing"],
                    "raw_shape_label": raw_sample["raw_shape_label"],
                    "preprocessed_volume": raw_sample["preprocessed_volume"],
                    "preprocessed_shape_label": raw_sample["preprocessed_shape_label"],
                    "final_volume": display_final,
                    "final_shape_label": final_stats["shape"],
                }
            )

    if len(batch_samples) != selected_count:
        raise RuntimeError(
            f"Could display {len(batch_samples)} of {selected_count} selected batch samples; "
            "refusing to create incomplete comparison figures."
        )

    statistics_path = output_dir / "preprocessing_statistics.csv"
    _write_statistics(statistics_path, statistics_rows)
    metadata_path = output_dir / "batch_metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(batch_metadata), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    output_files = [
        output_dir / "raw_triplanar.png",
        output_dir / "preprocessed_triplanar.png",
        output_dir / "actual_training_batch.png",
        output_dir / "actual_training_batch_triplanar.png",
        output_dir / "raw_vs_preprocessed.png",
        output_dir / "raw_vs_final_batch.png",
        statistics_path,
        metadata_path,
        output_dir / "pipeline_summary.txt",
    ]
    modality_ordered_samples = sorted(
        batch_samples,
        key=lambda sample: (sample["class_id"], sample["patient_id"]),
    )
    save_triplanar_figure(
        modality_ordered_samples,
        "raw_volume",
        output_files[0],
        title="Raw NIfTI input | center triplanar slices",
        shape_key="raw_shape_label",
        spacing_key="raw_spacing",
    )
    save_triplanar_figure(
        modality_ordered_samples,
        "preprocessed_volume",
        output_files[1],
        title="Deterministic preprocessed input | before random training augmentations",
        shape_key="preprocessed_shape_label",
    )
    save_actual_batch_figure(
        batch_samples,
        output_files[2],
        batch_shape=batch_metadata["batch_shape"],
    )
    save_triplanar_figure(
        batch_samples,
        "final_volume",
        output_files[3],
        title=f"Actual augmented model input | batch shape {batch_metadata['batch_shape']}",
        shape_key="final_shape_label",
    )
    save_pair_figure(
        modality_ordered_samples,
        "raw_volume",
        "preprocessed_volume",
        output_files[4],
        title="Raw MRI versus deterministic preprocessed input",
        left_title="Raw NIfTI | center axial",
        right_title="Deterministic preprocessed | center axial",
        left_spacing_key="raw_spacing",
    )
    save_pair_figure(
        batch_samples,
        "raw_volume",
        "final_volume",
        output_files[5],
        title="Raw MRI versus actual augmented training input",
        left_title="Raw NIfTI | center axial",
        right_title="Actual training batch | center axial",
        left_spacing_key="raw_spacing",
    )

    summary = _pipeline_summary(
        config_path=config_path,
        train_csv=train_csv,
        rows=rows,
        data_config=data_config,
        seed=seed,
        batch_metadata=batch_metadata,
        train_transform=train_transform,
        preprocessing_transform=preprocessing_transform,
        random_transform_names=random_names,
    )
    output_files[8].write_text(summary, encoding="utf-8")
    return output_files


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = run_visualization(args)
    metadata = json.loads(paths[7].read_text(encoding="utf-8"))
    print(f"Training scans: {len(_read_statistics_rows(paths[6]))}")
    print(f"Actual batch shape: {metadata['batch_shape']}")
    print(f"Actual batch metadata: {paths[7]}")
    print("Generated visualization files:")
    for path in paths:
        print(f"  {path}")


def _read_statistics_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
