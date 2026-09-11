from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import median

from .data import load_bundle, split_statistics
from .episodes import build_episode_specs, episode_count, validation_cases
from .types import ArcTask, Pair


def _challenge_key(task: ArcTask) -> str:
    """Canonical challenge content, deliberately excluding labelled test outputs."""
    payload = {
        "train": [
            {"input": pair.input, "output": pair.output} for pair in task.train_pairs
        ],
        "test": [{"input": grid} for grid in task.test_inputs],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _pair_key(pair: Pair) -> str:
    return json.dumps(
        {"input": pair.input, "output": pair.output},
        sort_keys=True,
        separators=(",", ":"),
    )


def _overlap_report(bundle) -> dict:
    train_challenges = {_challenge_key(task) for task in bundle.train.values()}
    eval_challenges = {_challenge_key(task) for task in bundle.evaluation.values()}
    local_test_challenges = {_challenge_key(task) for task in bundle.test.values()}

    train_pairs = {_pair_key(pair) for task in bundle.train.values() for pair in task.known_pairs}
    eval_pairs = {
        _pair_key(pair) for task in bundle.evaluation.values() for pair in task.known_pairs
    }

    return {
        "train_eval_task_id_overlap": len(set(bundle.train) & set(bundle.evaluation)),
        "train_eval_exact_challenge_overlap": len(train_challenges & eval_challenges),
        "train_eval_exact_labelled_pair_overlap": len(train_pairs & eval_pairs),
        "local_test_task_ids_also_in_train": len(set(bundle.test) & set(bundle.train)),
        "local_test_exact_challenges_also_in_train": len(
            local_test_challenges & train_challenges
        ),
    }


def _augmentation_report(tasks: dict[str, ArcTask]) -> dict:
    specs = build_episode_specs(tasks)
    per_task = Counter(spec.task_id for spec in specs)
    known_pair_counts = Counter(len(task.known_pairs) for task in tasks.values())
    counts = list(per_task.values())
    return {
        "known_pair_count_distribution": dict(sorted(known_pair_counts.items())),
        "episode_specs_total": len(specs),
        "episode_specs_per_task_min": min(counts) if counts else 0,
        "episode_specs_per_task_median": median(counts) if counts else 0,
        "episode_specs_per_task_max": max(counts) if counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ARC-AGI-2 dataset statistics")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.data)
    report = {
        "train": split_statistics(bundle.train),
        "evaluation": split_statistics(bundle.evaluation),
        "local_test": split_statistics(bundle.test),
        "overlap_audit": _overlap_report(bundle),
        "augmentation_audit": _augmentation_report(bundle.train),
        "training_augmented_episode_specs_per_logical_epoch": episode_count(bundle.train),
        "fixed_validation_query_shot_cases": len(validation_cases(bundle.evaluation)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
