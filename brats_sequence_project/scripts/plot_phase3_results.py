"""Regenerate Phase 3 plots from saved CSV/JSON files without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting import plot_prediction_outputs, plot_training_history, save_top_prediction_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_csv", type=Path)
    parser.add_argument("--predictions_csv", type=Path)
    parser.add_argument("--metrics_json", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_name", choices=["train", "val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created_paths: list[Path] = []

    if args.history_csv:
        history_csv = args.history_csv.expanduser().resolve()
        if history_csv.is_file():
            history_plot_dir = output_dir / "training"
            created_paths.extend(plot_training_history(history_csv, history_plot_dir))
        else:
            print(f"skip: history_csv not found: {history_csv}")

    if args.predictions_csv:
        predictions_csv = args.predictions_csv.expanduser().resolve()
        metrics_json = args.metrics_json.expanduser().resolve() if args.metrics_json else None
        if predictions_csv.is_file():
            prediction_plot_dir = output_dir / args.split_name
            created_paths.extend(
                plot_prediction_outputs(
                    predictions_csv,
                    prediction_plot_dir,
                    metrics_json=metrics_json,
                    split_name=args.split_name,
                )
            )
            created_paths.extend(save_top_prediction_csvs(predictions_csv, prediction_plot_dir))
        else:
            print(f"skip: predictions_csv not found: {predictions_csv}")

    if args.metrics_json and not args.metrics_json.expanduser().resolve().is_file():
        print(f"skip: metrics_json not found: {args.metrics_json.expanduser().resolve()}")

    if not created_paths:
        print("no plots created; provide an existing --history_csv and/or --predictions_csv.")
        return

    print("created outputs:")
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
