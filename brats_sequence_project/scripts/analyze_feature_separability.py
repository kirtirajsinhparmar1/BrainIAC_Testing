"""Analyze cached BrainIAC features with PCA and class-centroid summaries."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA


CLASS_NAMES = ["T1", "T2", "FLAIR", "T1CE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", choices=["pca"], default="pca")
    parser.add_argument("--max_samples", type=int)
    return parser.parse_args()


def _import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _sample_indices(num_samples: int, max_samples: int | None) -> np.ndarray:
    if max_samples is None or num_samples <= max_samples:
        return np.arange(num_samples)
    return np.linspace(0, num_samples - 1, num=max_samples, dtype=int)


def _plot_scatter(points: np.ndarray, groups: list[str], title: str, output_path: Path) -> Path:
    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 6))
    unique_groups = sorted(set(groups))
    colors = plt.cm.get_cmap("tab10", len(unique_groups))
    for index, group in enumerate(unique_groups):
        mask = np.array([value == group for value in groups])
        axis.scatter(points[mask, 0], points[mask, 1], s=24, alpha=0.75, label=group, color=colors(index))
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def _plot_explained_variance(explained_variance_ratio: np.ndarray, output_path: Path) -> Path:
    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative = np.cumsum(explained_variance_ratio)
    axis.bar(components, explained_variance_ratio, alpha=0.7, label="individual")
    axis.plot(components, cumulative, marker="o", linewidth=1.8, label="cumulative")
    axis.set_title("PCA Explained Variance")
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance ratio")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    return output_path


def compute_centroid_distances(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    centroids: dict[int, np.ndarray] = {}
    for label in sorted(set(labels.tolist())):
        centroids[int(label)] = features[labels == label].mean(axis=0)

    distances: dict[str, float] = {}
    for first_label, second_label in combinations(sorted(centroids), 2):
        first_name = CLASS_NAMES[first_label]
        second_name = CLASS_NAMES[second_label]
        distance = float(np.linalg.norm(centroids[first_label] - centroids[second_label]))
        distances[f"{first_name}_vs_{second_name}"] = distance

    return {
        "class_centroids_present": {CLASS_NAMES[label]: int(np.sum(labels == label)) for label in centroids},
        "pairwise_euclidean_distances": distances,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.features_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    features = payload["features"].float().cpu().numpy()
    labels = payload["labels"].long().cpu().numpy()
    modalities = list(payload.get("modalities", []))
    split_names = list(payload.get("split_names", [payload.get("split_name", "unknown")] * len(labels)))

    sample_indices = _sample_indices(features.shape[0], args.max_samples)
    sampled_features = features[sample_indices]
    sampled_labels = labels[sample_indices]
    sampled_modalities = [modalities[index] if modalities else CLASS_NAMES[int(labels[index])] for index in sample_indices]
    sampled_splits = [split_names[index] for index in sample_indices]

    pca = PCA(n_components=min(10, sampled_features.shape[0], sampled_features.shape[1]))
    transformed = pca.fit_transform(sampled_features)

    created_paths: list[Path] = []
    created_paths.append(
        _plot_scatter(
            transformed[:, :2],
            sampled_modalities,
            title="PCA of BrainIAC Features by Modality",
            output_path=output_dir / "pca_2d_by_modality.png",
        )
    )
    created_paths.append(_plot_explained_variance(pca.explained_variance_ratio_, output_dir / "explained_variance.png"))

    if len(set(sampled_splits)) > 1:
        created_paths.append(
            _plot_scatter(
                transformed[:, :2],
                sampled_splits,
                title="PCA of BrainIAC Features by Split",
                output_path=output_dir / "pca_2d_by_patient_split.png",
            )
        )

    centroid_summary = compute_centroid_distances(sampled_features, sampled_labels)
    centroid_path = output_dir / "class_centroid_distances.json"
    centroid_path.write_text(json.dumps(centroid_summary, indent=2) + "\n")

    summary = {
        "features_path": str(args.features_path.expanduser().resolve()),
        "variant": payload.get("variant", "unknown"),
        "split_name": payload.get("split_name", "unknown"),
        "num_samples": int(sampled_features.shape[0]),
        "method": args.method,
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "cumulative_explained_variance_ratio": [float(value) for value in np.cumsum(pca.explained_variance_ratio_)],
        "class_centroid_distances": centroid_summary,
        "created_plots": [str(path) for path in created_paths],
    }
    summary_path = output_dir / "feature_separability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    for path in created_paths:
        print(f"wrote: {path}")
    print(f"wrote: {centroid_path}")
    print(f"wrote: {summary_path}")
    print("feature_separability: passed")


if __name__ == "__main__":
    main()
