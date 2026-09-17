#!/usr/bin/env python3
"""CPU-safe model contract check; this never trains or loads MRI volumes."""

from __future__ import annotations

import argparse

from finetune_common import (
    REPO_ROOT,
    assert_full_finetuning,
    build_brainiac_model,
    load_yaml_config,
    resolve_repo_path,
    validate_finetune_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "brats_sequence_project/finetune/config/full_finetune.yml"),
    )
    parser.add_argument("--checkpoint", help="Optional local path; basename must be BrainIAC.ckpt")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    validate_finetune_config(config)
    checkpoint = args.checkpoint or config["model"]["checkpoint_path"]
    model = build_brainiac_model(checkpoint, num_classes=config["model"]["num_classes"])
    report = assert_full_finetuning(model)
    print(f"general_checkpoint={resolve_repo_path(checkpoint)}")
    print(f"classifier_output_dimension={model.classifier.fc.out_features}")
    print(f"total_parameters={report['total_parameters']}")
    print(f"trainable_parameters={report['trainable_parameters']}")
    print(f"frozen_parameters={report['frozen_parameters']}")
    print("backbone_requires_grad=True")
    print("full_fine_tuning=True")


if __name__ == "__main__":
    main()
