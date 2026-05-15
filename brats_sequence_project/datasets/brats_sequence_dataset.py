"""CSV-driven BraTS sequence dataset for BrainIAC runtime preprocessing."""

from __future__ import annotations

import atexit
import csv
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset

try:
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        MapTransform,
        NormalizeIntensityd,
        Orientationd,
        Resized,
        ResizeWithPadOrCropd,
        Spacingd,
        ToTensord,
    )
except ImportError as exc:  # pragma: no cover - exercised only when MONAI is absent.
    raise ImportError(
        "MONAI is required for BraTSSequenceDataset. Install BrainIAC requirements before "
        "running dataloader or backbone checks."
    ) from exc


REQUIRED_COLUMNS = ("patient_id", "image_path", "label", "modality", "split_source")
SUPPORTED_PREPROCESSING_VARIANTS = (
    "resize_zscore",
    "resize_none",
    "resize_percentile",
    "crop_pad_zscore",
    "physical_crop_zscore",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_CANDIDATES = (
    PROJECT_ROOT.parents[1] / "archive.zip",
    PROJECT_ROOT.parent / "archive.zip",
)


class NonzeroPercentileScaled(MapTransform):
    """Clip nonzero voxels to percentile bounds, then scale each channel to [0, 1]."""

    def __init__(
        self,
        keys: Sequence[str],
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> None:
        super().__init__(keys)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        for key in self.keys:
            output[key] = self._scale_image(output[key])
        return output

    def _scale_image(self, image: torch.Tensor) -> torch.Tensor:
        squeeze_channel = False
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze_channel = True

        scaled_image = image.clone()
        for channel_index in range(scaled_image.shape[0]):
            channel = scaled_image[channel_index]
            mask = channel != 0
            if not torch.any(mask):
                scaled_image[channel_index] = torch.zeros_like(channel)
                continue

            values = channel[mask].float()
            lower = torch.quantile(values, self.lower_percentile / 100.0)
            upper = torch.quantile(values, self.upper_percentile / 100.0)
            if (not torch.isfinite(lower)) or (not torch.isfinite(upper)) or float(upper) <= float(lower):
                channel = channel.float()
                channel[~mask] = 0.0
                scaled_image[channel_index] = channel.to(dtype=scaled_image.dtype)
                continue

            clipped = channel.float().clamp(min=float(lower), max=float(upper))
            clipped = (clipped - float(lower)) / float(upper - lower)
            clipped = torch.clamp(clipped, 0.0, 1.0)
            clipped[~mask] = 0.0
            scaled_image[channel_index] = clipped.to(dtype=scaled_image.dtype)

        if squeeze_channel:
            return scaled_image[0]
        return scaled_image


def build_preprocessing_transform(
    preprocessing_variant: str,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
) -> Compose:
    """Build a MONAI preprocessing pipeline for a named ablation variant."""

    common_prefix: list[Any] = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ]

    if preprocessing_variant == "resize_zscore":
        transforms = [
            *common_prefix,
            Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    elif preprocessing_variant == "resize_none":
        transforms = [
            *common_prefix,
            Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
            ToTensord(keys=["image"]),
        ]
    elif preprocessing_variant == "resize_percentile":
        transforms = [
            *common_prefix,
            Resized(keys=["image"], spatial_size=spatial_size, mode="trilinear"),
            NonzeroPercentileScaled(keys=["image"], lower_percentile=1.0, upper_percentile=99.0),
            ToTensord(keys=["image"]),
        ]
    elif preprocessing_variant == "crop_pad_zscore":
        transforms = [
            *common_prefix,
            ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    elif preprocessing_variant == "physical_crop_zscore":
        transforms = [
            *common_prefix,
            Orientationd(keys=["image"], axcodes="LPS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            CenterSpatialCropd(keys=["image"], roi_size=spatial_size),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=spatial_size),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    else:
        raise ValueError(
            f"Unsupported preprocessing_variant={preprocessing_variant!r}. "
            f"Choose one of: {', '.join(SUPPORTED_PREPROCESSING_VARIANTS)}"
        )

    return Compose(transforms)


class BraTSSequenceDataset(Dataset):
    """Load one BraTS modality volume per CSV row and apply BrainIAC transforms.

    The CSV stores direct paths to original BraTS .nii files. No copied, renamed,
    or preprocessed NIfTI files are required.
    """

    def __init__(
        self,
        csv_path: str | Path,
        spatial_size: tuple[int, int, int] = (96, 96, 96),
        preprocessing_variant: str = "resize_zscore",
        validate_paths: bool = True,
        archive_zip: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.preprocessing_variant = preprocessing_variant
        self.rows = self._read_rows(self.csv_path)
        self.archive_zip = self._resolve_archive_zip(archive_zip)
        self._archive_member_lookup: set[str] | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        if validate_paths:
            self._validate_image_paths()

        self.transform = build_preprocessing_transform(
            preprocessing_variant=preprocessing_variant,
            spatial_size=spatial_size,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_source = self._resolve_image_source(row["image_path"])
        sample = self.transform({"image": image_source})

        return {
            "image": sample["image"],
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "patient_id": row["patient_id"],
            "modality": row["modality"],
            "image_path": row["image_path"],
        }

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open("r", newline="") as file:
            reader = csv.DictReader(file)
            missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
            if missing_columns:
                raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")

            rows = [dict(row) for row in reader]

        if not rows:
            raise ValueError(f"{csv_path} contains no rows.")

        for row_number, row in enumerate(rows, start=2):
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError(f"Invalid label at {csv_path}:{row_number}: {row['label']}") from exc
            if label not in {0, 1, 2, 3}:
                raise ValueError(f"Label out of range at {csv_path}:{row_number}: {label}")

        return rows

    def _validate_image_paths(self) -> None:
        missing_paths = [
            row["image_path"]
            for row in self.rows
            if not Path(row["image_path"]).is_file() and not self._has_archive_member(row["image_path"])
        ]
        if missing_paths:
            preview = "\n".join(missing_paths[:10])
            raise FileNotFoundError(
                f"{len(missing_paths)} image paths from {self.csv_path} do not exist. "
                f"First missing paths:\n{preview}"
            )

    @staticmethod
    def _resolve_archive_zip(archive_zip: str | Path | None) -> Path | None:
        if archive_zip is not None:
            resolved = Path(archive_zip).expanduser().resolve()
            return resolved if resolved.is_file() else None

        for candidate in DEFAULT_ARCHIVE_CANDIDATES:
            resolved = candidate.expanduser().resolve()
            if resolved.is_file():
                return resolved
        return None

    def _archive_members(self) -> set[str]:
        if self.archive_zip is None:
            return set()
        if self._archive_member_lookup is None:
            with ZipFile(self.archive_zip) as archive:
                self._archive_member_lookup = set(archive.namelist())
        return self._archive_member_lookup

    def _candidate_archive_members(self, image_path: str) -> list[str]:
        path = Path(image_path)
        parts = path.parts
        if "Dataset" in parts:
            dataset_index = parts.index("Dataset")
            return ["/".join(parts[dataset_index + 1 :])]
        return [path.name]

    def _has_archive_member(self, image_path: str) -> bool:
        if self.archive_zip is None:
            return False
        archive_members = self._archive_members()
        return any(candidate in archive_members for candidate in self._candidate_archive_members(image_path))

    def _resolve_image_source(self, image_path: str) -> str:
        file_path = Path(image_path)
        if file_path.is_file():
            return str(file_path)
        if self.archive_zip is None:
            raise FileNotFoundError(f"Image path not found and no archive fallback is configured: {image_path}")

        archive_members = self._archive_members()
        for candidate in self._candidate_archive_members(image_path):
            if candidate not in archive_members:
                continue
            extract_root = self._get_extract_root()
            extracted_path = extract_root / candidate
            if not extracted_path.is_file():
                with ZipFile(self.archive_zip) as archive:
                    archive.extract(candidate, path=extract_root)
            return str(extracted_path)

        raise FileNotFoundError(f"Image path not found locally or in archive fallback: {image_path}")

    def _get_extract_root(self) -> Path:
        if self._temporary_directory is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="brats_sequence_")
            atexit.register(self._temporary_directory.cleanup)
        return Path(self._temporary_directory.name)
