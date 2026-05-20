"""Run PCA and centroid analysis for cached binary T1-vs-T2 BrainIAC features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA


CLASS_NAMES = ["T1", "T2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", choices=["pca"], default="pca")
    return parser.parse_args()


def import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_torch_payload(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cached features not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_binary_features(features: np.ndarray, labels: np.ndarray) -> None:
    if features.ndim != 2 or features.shape[1] != 768:
        raise ValueError(f"Expected cached features [N,768], got {tuple(features.shape)}")
    observed_labels = sorted({int(value) for value in labels.tolist()})
    if not set(observed_labels).issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1 only, found {observed_labels}")
    if len(observed_labels) < 2:
        raise ValueError("PCA separability needs both T1 and T2 samples.")
    if features.shape[0] < 2:
        raise ValueError("PCA needs at least two samples.")


def plot_scatter(points: np.ndarray, groups: list[str], output_path: Path) -> Path:
    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 6))
    unique_groups = ["T1", "T2"]
    colors = {"T1": "#4C72B0", "T2": "#DD8452"}
    for group in unique_groups:
        mask = np.array([value == group for value in groups])
        if np.any(mask):
            axis.scatter(points[mask, 0], points[mask, 1], s=28, alpha=0.78, label=group, color=colors[group])
    axis.set_title("PCA of BrainIAC Features: T1 vs T2")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def plot_explained_variance(explained_variance_ratio: np.ndarray, output_path: Path) -> Path:
    plt = import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative = np.cumsum(explained_variance_ratio)
    axis.bar(components, explained_variance_ratio, alpha=0.7, label="individual")
    axis.plot(components, cumulative, marker="o", linewidth=1.8, label="cumulative")
    axis.set_title("PCA Explained Variance: T1 vs T2")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance ratio")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def compute_centroid_summary(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    t1_features = features[labels == 0]
    t2_features = features[labels == 1]
    t1_centroid = t1_features.mean(axis=0)
    t2_centroid = t2_features.mean(axis=0)
    distance = float(np.linalg.norm(t1_centroid - t2_centroid))
    t1_within = np.linalg.norm(t1_features - t1_centroid, axis=1)
    t2_within = np.linalg.norm(t2_features - t2_centroid, axis=1)
    pooled_within = float((t1_within.mean() + t2_within.mean()) / 2.0)
    separation_ratio = float(distance / pooled_within) if pooled_within > 0 else None
    return {
        "class_counts": {"T1": int(t1_features.shape[0]), "T2": int(t2_features.shape[0])},
        "centroid_distance_t1_t2": distance,
        "mean_within_class_distance": {"T1": float(t1_within.mean()), "T2": float(t2_within.mean())},
        "pooled_mean_within_class_distance": pooled_within,
        "centroid_distance_to_pooled_within_ratio": separation_ratio,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_torch_payload(args.features_path)
    features = payload["features"].float().cpu().numpy()
    labels = payload["labels"].long().cpu().numpy()
    modalities = list(payload.get("modalities", [CLASS_NAMES[int(label)] for label in labels]))
    groups = [str(value).upper() for value in modalities]
    groups = [value if value in CLASS_NAMES else CLASS_NAMES[int(labels[index])] for index, value in enumerate(groups)]
    validate_binary_features(features, labels)

    n_components = min(10, features.shape[0], features.shape[1])
    if n_components < 2:
        raise ValueError("Need at least two PCA components for a 2D scatter plot.")
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features)

    created_paths = [
        plot_scatter(transformed[:, :2], groups, output_dir / "pca_2d_t1_t2_by_modality.png"),
        plot_explained_variance(pca.explained_variance_ratio_, output_dir / "explained_variance_t1_t2.png"),
    ]

    centroid_summary = compute_centroid_summary(features, labels)
    centroid_path = output_dir / "centroid_distance_t1_t2.json"
    centroid_path.write_text(json.dumps(centroid_summary, indent=2) + "\n")

    summary = {
        "features_path": str(args.features_path.expanduser().resolve()),
        "variant": payload.get("variant", "unknown"),
        "split_name": payload.get("split_name", "unknown"),
        "method": args.method,
        "num_samples": int(features.shape[0]),
        "feature_shape": list(features.shape),
        "class_names": CLASS_NAMES,
        "label_mapping": {"0": "T1", "1": "T2"},
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "cumulative_explained_variance_ratio": [float(value) for value in np.cumsum(pca.explained_variance_ratio_)],
        "centroid_summary": centroid_summary,
        "created_plots": [str(path) for path in created_paths],
    }
    summary_path = output_dir / "feature_separability_summary_t1_t2.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    for path in created_paths:
        print(f"wrote: {path}")
    print(f"wrote: {centroid_path}")
    print(f"wrote: {summary_path}")
    print("binary_t1_t2_feature_analysis: passed")


if __name__ == "__main__":
    main()
