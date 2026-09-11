from __future__ import annotations

import argparse
import json

from .data import load_bundle, split_statistics
from .episodes import episode_count, validation_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ARC-AGI-2 dataset statistics")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.data)
    report = {
        "train": split_statistics(bundle.train),
        "evaluation": split_statistics(bundle.evaluation),
        "local_test": split_statistics(bundle.test),
        "training_augmented_episode_specs_per_logical_epoch": episode_count(bundle.train),
        "fixed_validation_query_shot_cases": len(validation_cases(bundle.evaluation)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
